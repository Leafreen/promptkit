# core/steps/actions/chat_util.py
'''聊天场景的**声明式 Step**。

这些 Step 只做一件事：把「这个聊天机器人是谁、要干什么、有什么规矩」
写进 Context。之后 `SystemPrompt` 把它们编译成 system prompt 装到 Agent 上，
`ChatLoop` 再把多轮对话跑起来。

于是整个聊天机器人就是一串普通 Step，现有的步骤树 / 事件 / Verbosity 全都适用：

    wf = Workflow(
        Role("市场调研导师"),
        Want("帮用户找到合适的选题方向"),
        Rules("像导师一样专业、耐心", "每次回复不超过 300 字"),
        Deliverables(Deliverable("选题建议", format="markdown", save_to="out.md")),
        Tools(WEB_SEARCH),
        SystemPrompt(),      # 把上面这些编译成 system prompt
        ChatLoop(),          # 多轮对话
        SaveDeliverables(),  # 结束时落盘
    )

**为什么先写进 Context 再编译，而不是直接拼一段字符串？**
因为这样每一步都能单独用：你可以只放 `Role(...)`，或者
`Role(...)` 中间插别的步骤再 `SystemPrompt()`。如果 `Role("x")` 直接返回
prompt 文本，就绑死在「必须一次性拼完」上了。
'''
from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context

from promptkit.core.steps.interface import Step, coerce_steps, execute_step
from promptkit.core.steps.logics.expr_sys.expr import VarExpr

__all__ = [
    "CHAT_ROLE_KEY",
    "CHAT_WANT_KEY",
    "CHAT_RULES_KEY",
    "CHAT_DELIVERABLES_KEY",
    "CHAT_TOOLS_KEY",
    "CHAT_EXIT_KEY",
    "CHAT_SIGNAL_KEY",
    "CHAT_INPUT_KEY",
    "CHAT_REPLY_KEY",
    "DEFAULT_EXIT_WORDS",
    "Role",
    "Want",
    "Rules",
    "Tools",
    "Deliverables",
    "ExitWords",
    "Behavior",
    "SystemPrompt",
    "ChatLoop",
    "ChatTurn",
    "MakeFile",
    "SaveDeliverables",
    "build_system_prompt",
    "save_format",
    "chat_context_vars",
    "flatten_single",
    "extension_for",
    "set_system_prompt_reset",
]

# ---- Context 键名：集中定义，`ChatWorkflow` 和测试都从这里取 ----
CHAT_ROLE_KEY = "chat_role"
CHAT_WANT_KEY = "chat_want"
CHAT_RULES_KEY = "chat_rules"
CHAT_DELIVERABLES_KEY = "chat_deliverables"
CHAT_TOOLS_KEY = "chat_tools"
CHAT_EXIT_KEY = "chat_exit_words"

CHAT_SIGNAL_KEY = "chat_signal"
CHAT_INPUT_KEY = "usr_input"
CHAT_REPLY_KEY = "model_reply"

#: 默认退出词
DEFAULT_EXIT_WORDS = ("exit", "quit", ":q")


def chat_context_vars() -> List[str]:
    '''聊天相关的 Context 键名（用来预声明变量 / 调试）。'''
    return [
        CHAT_ROLE_KEY,
        CHAT_WANT_KEY,
        CHAT_RULES_KEY,
        CHAT_DELIVERABLES_KEY,
        CHAT_TOOLS_KEY,
        CHAT_EXIT_KEY,
        CHAT_SIGNAL_KEY,
        CHAT_INPUT_KEY,
        CHAT_REPLY_KEY,
    ]


def flatten_single(items: Sequence[Any]) -> Tuple[Any, ...]:
    '''`Rules(["a", "b"])` 和 `Rules("a", "b")` 都当两个处理。

    这是最容易写错的一处：`Deliverables([Deliverable(...)])` 传的是**一个列表**，
    不拆开的话 `items` 会变成 `[[Deliverable(...)]]`，后面所有 `getattr(item, ...)`
    全部落空 —— 交付物**永远不会保存**，而且一声不响。
    '''
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        return tuple(items[0])
    return tuple(items)


# ============================ 声明：写进 Context ============================
class Role(Step):
    '''这个机器人是谁。

        Role("市场调研导师")

    注意用的是**赋值**而不是 `Context.declare`：`declare` 在键已存在时
    不会覆盖，那样 `Role("A")` 后再 `Role("B")` 会静默保留 A。
    '''

    name = "role"

    def __init__(self, role: str = "通用 AI 助手"):
        self.role = str(role)

    def execute(self, context: Context) -> None:
        context[CHAT_ROLE_KEY] = self.role
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"Role({self.role!r})"


class Want(Step):
    '''它要帮用户达成什么。

        Want("帮用户找到合适的选题方向")
    '''

    name = "want"

    def __init__(self, want: str = ""):
        self.want = str(want)

    def execute(self, context: Context) -> None:
        context[CHAT_WANT_KEY] = self.want
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"Want({self.want!r})"


class Rules(Step):
    '''行为规矩，一条一条写。

        Rules("像导师一样专业、耐心", "每次回复不超过 300 字")
    '''

    name = "rules"

    def __init__(self, *rules: Any, replace: bool = False):
        self.rules = [str(rule) for rule in flatten_single(rules)]
        self.replace = replace

    def execute(self, context: Context) -> None:
        existing = [] if self.replace else list(context.get(CHAT_RULES_KEY) or [])
        context[CHAT_RULES_KEY] = existing + self.rules
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"Rules({len(self.rules)} 条)"

    def __repr__(self) -> str:
        return f"<Rules {self.rules!r}>"


class Tools(Step):
    '''这个机器人能用的工具。

        Tools(WEB_SEARCH, RUN_COMMAND)

    只是记在 Context 上，`ChatLoop` 会把它传给 `LLMStep`。
    '''

    name = "tools"

    def __init__(self, *tools: Any, replace: bool = False):
        self.tools = list(flatten_single(tools))
        self.replace = replace

    def execute(self, context: Context) -> None:
        existing = [] if self.replace else list(context.get(CHAT_TOOLS_KEY) or [])
        context[CHAT_TOOLS_KEY] = existing + self.tools
        context.trace.append(self.name)

    def describe(self) -> str:
        names = ", ".join(
            getattr(tool, "name", getattr(tool, "__name__", repr(tool)))
            for tool in self.tools
        )
        return f"Tools({names})"


class Deliverables(Step):
    '''要产出的交付物。

        Deliverables(
            Deliverable("选题建议", format="markdown", save_to="out/选题.md"),
            Deliverable("参考资料", format="json", save_to="out/refs.json"),
        )

    只是记在 Context 上：`SystemPrompt` 会写进提示词，`SaveDeliverables` 会落盘。
    '''

    name = "deliverables"

    def __init__(self, *items: Any, replace: bool = False):
        self.items = list(flatten_single(items))
        self.replace = replace

    def execute(self, context: Context) -> None:
        existing = (
            [] if self.replace else list(context.get(CHAT_DELIVERABLES_KEY) or [])
        )
        context[CHAT_DELIVERABLES_KEY] = existing + self.items
        context.trace.append(self.name)

    def describe(self) -> str:
        names = ", ".join(getattr(item, "name", str(item)) for item in self.items)
        return f"Deliverables({names})"


class ExitWords(Step):
    '''用户输入这些词就结束对话（默认 `exit` / `quit` / `:q`）。'''

    name = "exit_words"

    def __init__(self, *words: Any):
        flat = [str(word) for word in flatten_single(words)]
        self.words = flat or list(DEFAULT_EXIT_WORDS)

    def execute(self, context: Context) -> None:
        context[CHAT_EXIT_KEY] = list(self.words)
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"ExitWords({', '.join(self.words)})"


class Behavior(Step):
    '''每轮里在模型调用**之前**额外执行的步骤（逃生舱）。

        Behavior(
            If(Var("usr_input") == "help").Then(Echo("输入 exit 退出")),
        )
    '''

    name = "behavior"

    def __init__(self, *steps: Any):
        self.steps = tuple(coerce_steps(*steps))

    def execute(self, context: Context) -> None:
        for step in self.steps:
            execute_step(step, context)
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"Behavior({len(self.steps)} 步)"

    def children(self) -> List[Step]:
        return list(self.steps)


# ============================ 编译：Context -> system prompt ============================
def build_system_prompt(
    role: str = "",
    want: str = "",
    rules: Optional[Sequence[str]] = None,
    deliverables: Optional[Sequence[Any]] = None,
    variables: Optional[Dict[str, Any]] = None,
) -> str:
    '''把角色 / 目标 / 规矩 / 交付物拼成 system prompt 文本。

    `variables` 给了就把 `{{名字}}` 换成真实值。**这一步不能省** ——
    system prompt 是直接塞进 `agent.start_prompts` 的，不经过 `Prompt.render`；
    不替换的话 `Rules("团队成员信息：{{basic_info}}")` 里的占位符会原样留下，
    模型看到的就是 `{{basic_info}}` 这串字符（然后告诉你「这里是空的」）。

    用 `allow_dollar=False`：system prompt 是散文，`$` 该当字面量
    （公式、价格都可能出现）。
    '''
    from promptkit.prompt import render_text

    lines: List[str] = [f"你是{role}。" if role else "你是一个 AI 助手。"]
    if want:
        lines.append(f"你的目标：{want}")
    if rules:
        lines.append("")
        lines.append("行为规则：")
        lines.extend(f"- {rule}" for rule in rules)
    if deliverables:
        lines.append("")
        lines.append("你需要产出以下交付物：")
        for item in deliverables:
            description = getattr(item, "description", "") or ""
            fmt = getattr(item, "format", "") or ""
            suffix = f"（{fmt} 格式）" if fmt else ""
            lines.append(f"- {getattr(item, 'name', item)}：{description}{suffix}")

    text = "\n".join(lines)
    if variables is None:
        return text
    return render_text(text, dict(variables), allow_dollar=False)


class SystemPrompt(Step):
    '''把 `Role` / `Want` / `Rules` / `Deliverables` 编译成 system prompt。

    装到 **Agent** 上（`context["agent"].start_prompts`），不是塞进 user 消息 ——
    system 消息才是模型的「人格设定」，塞进 user 消息的话每轮都要重复一遍，
    既浪费 token 又容易让模型把它当成「用户在说」。

    - `reset=True`（默认）：清空历史再重建。适合全新对话。
    - `reset=False`：**保留历史**，只按新 system prompt 重建消息列表。
      续接上次会话就靠它（历史先 `load_history()` 装进来，再走这一步）。

    两种模式都必须调用 `Agent.rebuild()` —— `messages` 是算好的缓存，
    只改 `start_prompts` 不会自动生效。
    '''

    name = "system_prompt"

    def __init__(self, template: Optional[str] = None, reset: bool = True):
        self.template = template
        self.reset = reset

    def execute(self, context: Context) -> None:
        agent = context.get("agent")
        if agent is None:
            raise ValueError(
                "SystemPrompt 需要 Agent：请用 wf.run(agent=...) 运行，"
                "或者包在 ChatWorkflow.run() 里。"
            )
        variables = context.to_dict()
        content = build_system_prompt(
            role=context.get(CHAT_ROLE_KEY, "") or "",
            want=context.get(CHAT_WANT_KEY, "") or "",
            rules=context.get(CHAT_RULES_KEY) or [],
            deliverables=context.get(CHAT_DELIVERABLES_KEY) or [],
            # 关键：`{{basic_info}}` 这类占位符要在这里换成真实值。
            # system prompt 不走 `Prompt.render`，不换就永远是原样的字符。
            variables=variables,
        )
        if self.template is not None:
            from promptkit.prompt import render_text

            content = render_text(self.template, variables, allow_dollar=False)
        _install_system_prompt(agent, content, reset=self.reset)
        context.trace.append(self.name)

    def describe(self) -> str:
        if self.template is None:
            return "SystemPrompt()"
        return "SystemPrompt(自定义模板)"


def _install_system_prompt(agent: Any, content: str, *, reset: bool) -> None:
    '''把 system prompt 装到 agent 上，并让 `messages` 跟上。

    两种模式的区别只在于**历史留不留**：

    - `reset=True`：`agent.reset()` —— 清历史 + 重建
    - `reset=False`：`agent.rebuild()` —— **只重建**，历史留着
    '''
    agent.start_prompts = [{"role": "system", "content": content}]
    if reset:
        agent.reset()
        return
    rebuild = getattr(agent, "rebuild", None)
    if not callable(rebuild):
        raise TypeError(
            "SystemPrompt(reset=False) 需要 agent 实现 rebuild()，"
            f"但 {type(agent).__name__} 没有。"
            "（promptkit.agent.Agent 有；续接会话靠的就是它。）"
        )
    rebuild()


# ============================ 单轮 / 多轮 ============================
class ChatTurn(Step):
    '''一轮对话里的模型调用。

    `tools` / `cautions` 在**执行时**才从 Context 取，所以 `Tools(WEB_SEARCH)`
    写一次就够了，不用在建循环时再传一遍。
    '''

    name = "chat_turn"

    def __init__(
        self,
        *,
        prompt_role: str = "user",
        prompt_content: str = "{{usr_input}}",
        output_key: str = CHAT_REPLY_KEY,
        cautions: Optional[Sequence[Any]] = None,
    ):
        self.prompt_role = prompt_role
        self.prompt_content = prompt_content
        self.output_key = output_key
        self.cautions = list(cautions) if cautions else None

    def execute(self, context: Context) -> None:
        from promptkit.core.steps.actions.llm import LLMStep
        from promptkit.prompt import Prompt

        tools = context.get(CHAT_TOOLS_KEY) or None
        step = LLMStep(
            prompt=Prompt(role=self.prompt_role, content=self.prompt_content),
            output_key=self.output_key,
            cautions=self.cautions,
            tools=tools,
        )
        execute_step(step, context)
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"ChatTurn(-> {self.output_key})"


def ChatLoop(  # noqa: N802 - 跟随 DSL 命名（工厂，返回 Step）
    *,
    input_key: str = CHAT_INPUT_KEY,
    output_key: str = CHAT_REPLY_KEY,
    signal_key: str = CHAT_SIGNAL_KEY,
    read_prompt: str = ">>> ",
    render: str = "markdown",
    exit_words: Optional[Sequence[str]] = None,
    behavior: Sequence[Any] = (),
    after_turn: Optional[Sequence[Any]] = None,
    after_end: Sequence[Any] = (),
    cautions: Optional[Sequence[Any]] = None,
    max_turns: Optional[int] = None,
    turn_template: Optional[str] = None,
) -> Step:
    '''组装多轮对话循环，返回一个 `Workflow`（本身就是 Step）。

    每一轮：读一行 -> 是退出词就结束 -> 跑 `behavior` -> 调模型
    -> 显示回答 -> 跑 `after_turn`（`Deliverable(when="each_turn")` 就挂这儿）。

    用户决定退出后（退出词 / Ctrl+D），跑 `after_end` —— 说句再见、把
    session 存掉、总结这轮对话都放这儿。它跑在循环之后、`ChatWorkflow`
    的写记忆与落盘之前，所以 `after_end` 里仍能读到完整的 Context。

    `ExitWords` 写进 Context 的值会被自动读出来，所以 `exit_words`
    一般不用显式传；显式传的优先。

    `turn_template` 换掉「这一轮发给模型的内容」（默认就是 `{{usr_input}}`）。
    召回记忆之类的每轮上下文就靠它拼进去 —— 别塞进 system prompt：
    那是**一轮**的临时信息，塞进 system 既浪费 token 又容易让模型当成长期设定。
    '''
    from promptkit.core.steps.actions.interactive import Echo, Read
    from promptkit.core.steps.logics.if_step import If
    from promptkit.core.steps.logics.loop import Break
    from promptkit.core.steps.logics.var import NewVar, Set, Var
    from promptkit.core.steps.logics.while_step import While
    from promptkit.core.steps.workflow import Workflow

    # 每轮要跑的收尾步骤。没显式给就默认「把 when='each_turn' 的交付物存掉」——
    # 手写 Workflow 时 `Deliverable(when="each_turn")` 否则会**完全没效果**：
    # 循环里没人管它，循环后的 `SaveDeliverables()` 又按 when="end" 把它过滤掉了。
    resolved_after_turn = (
        list(after_turn)
        if after_turn is not None
        else [SaveDeliverables(when="each_turn")]
    )

    resolved_exit = list(exit_words) if exit_words else None

    # 退出词从哪来：显式传的优先，否则读 `ExitWords(...)` 写进 Context 的那个。
    # 所以判断时**引用变量**（配一个 NewVar 默认值），而不是把列表焊死在
    # `If(...)` 里 —— 焊死的话 `ExitWords("拜拜")` 会被完全忽略。
    exit_expr = (
        list(resolved_exit) if resolved_exit is not None else Var(CHAT_EXIT_KEY)
    )

    loop = While(
        Var(signal_key) != "end", max_iterations=max_turns
    ).Do(
        # on_eof="break"：Ctrl+D / Ctrl+C 当「不想聊了」，跳出循环 ——
        # 这样循环后面的 `SaveDeliverables()` 还能正常跑完。
        # 默认的 `raise` 会让整个工作流抛异常退出，收尾步骤全跳过，
        # 表现就是「聊完了，但桌面上没有文件」。
        Set(Var(input_key), Read(read_prompt, on_eof="break")),
        If(Var(input_key).in_(exit_expr)).Then(
            Set(Var(signal_key), "end"),
            Break(),
        ),
        *coerce_steps(*behavior),
        ChatTurn(
            prompt_content=(
                turn_template
                if turn_template is not None
                else "{{%s}}" % input_key
            ),
            output_key=output_key,
            cautions=cautions,
        ),
        Echo(Var(output_key), render=render),
        *coerce_steps(*resolved_after_turn),
    )

    return Workflow(
        NewVar(signal_key, default="start"),
        NewVar(input_key, default=""),
        NewVar(output_key, default=""),
        NewVar(CHAT_EXIT_KEY, default=list(DEFAULT_EXIT_WORDS)),
        loop,
        # 用户退出后的收尾：退出词和 Ctrl+D 两条路都会走到这儿
        # （`max_turns` 用尽会抛 LoopLimitExceeded，不会走到）。
        *coerce_steps(*after_end),
        name="chat_loop",
    )


# ============================ 落盘 ============================
def save_format(fmt: str) -> str:
    '''交付物的 `format` -> `Save` 的 `format`。

    `Save` 只支持 `text` / `json`。`markdown` 本身就是文本，所以直接写；
    其它格式（xml / html …）暂时也按文本写 —— 至少内容不丢。
    '''
    return "json" if "json" in (fmt or "").lower() else "text"


class MakeFile(Step):
    '''把一个交付物渲染出来写进文件。

    内容从 `context[deliverable.source]` 取（默认 `model_reply`，
    也就是模型这一轮的回答）；`format` 含 `json` 时走 `json.dumps`。

    `save_to` 可以是**文件**也可以是**目录**：给目录时用
    `交付物名字 + 按 format 推的扩展名` 当文件名。所以
    `Deliverable("选题建议", format="markdown", save_to=r"C:\\Users\\me\\Desktop")`
    会写成 `Desktop/选题建议.md` —— 这比直接 `open(Desktop, "w")` 报
    `IsADirectoryError` 有用得多。
    '''

    name = "make_file"

    def __init__(self, deliverable: Any):
        self.deliverable = deliverable

    def execute(self, context: Context) -> None:
        from promptkit.core.steps.actions.interactive import Save

        target = self.target()
        if not target:
            context.trace.append(self.name + ":skip")
            return

        source = getattr(self.deliverable, "source", CHAT_REPLY_KEY) or CHAT_REPLY_KEY
        step = Save(
            VarExpr(source),
            file=target,
            format=save_format(getattr(self.deliverable, "format", "") or ""),
            encoding=getattr(self.deliverable, "encoding", None),
        )
        execute_step(step, context)
        context.trace.append(self.name)

    def target(self) -> Optional[str]:
        '''算出真正要写的文件路径（`save_to` 是目录时自动补文件名）。'''
        raw = getattr(self.deliverable, "save_to", None)
        if not raw:
            return None
        raw = os.path.expanduser(str(raw))
        if not _looks_like_directory(raw):
            return raw
        name = getattr(self.deliverable, "name", "output") or "output"
        extension = extension_for(getattr(self.deliverable, "format", "") or "")
        if name.lower().endswith(extension):
            return os.path.join(raw, name)
        return os.path.join(raw, f"{name}{extension}")

    def describe(self) -> str:
        return (
            f"MakeFile({getattr(self.deliverable, 'name', '?')!r}"
            f" -> {self.target() or '?'!r})"
        )


def _looks_like_directory(path: str) -> bool:
    '''`save_to` 指的是目录吗。

    判据：已经存在且是目录，**或者**最后一段没有扩展名（`outputs` / `Desktop`）。
    后者是为了兼容「目录还没建出来」的情况。
    '''
    if os.path.isdir(path):
        return True
    if path.endswith(("/", "\\")):
        return True
    return not os.path.splitext(os.path.basename(path))[1]


def extension_for(fmt: str) -> str:
    '''按交付物的 `format` 推一个扩展名。'''
    lowered = (fmt or "").lower()
    if "json" in lowered:
        return ".json"
    if "markdown" in lowered or lowered in ("md", "mmd"):
        return ".md"
    if "csv" in lowered:
        return ".csv"
    if "yaml" in lowered or "yml" in lowered:
        return ".yaml"
    if "html" in lowered:
        return ".html"
    return ".txt"


class SaveDeliverables(Step):
    '''把交付物落盘。

    放在**循环之后**用 `when="end"`（默认），放在**循环体内**用
    `when="each_turn"`：

        ChatLoop(),                             # 循环
        SaveDeliverables(),                     # 结束时存 when="end" 的

        # 循环体内每轮存 when="each_turn" 的：
        ChatLoop(after_turn=[SaveDeliverables(when="each_turn")]),

    `items` 显式给的时候**不看 `when`** —— 调用方已经决定要存了；
    只有从 Context 里读（`items=None`）时才按 `when` 过滤，
    因为那种情况下「每轮存」还是「最后存」得靠这个字段区分。
    '''

    name = "save_deliverables"

    def __init__(
        self,
        items: Optional[Sequence[Any]] = None,
        *,
        when: str = "end",
    ):
        self.items = list(items) if items is not None else None
        self.when = when

    def execute(self, context: Context) -> None:
        explicit = self.items is not None
        items = self.items
        if items is None:
            items = context.get(CHAT_DELIVERABLES_KEY) or []
        for item in items:
            if not getattr(item, "save_to", None):
                continue
            if not explicit and getattr(item, "when", "end") != self.when:
                continue
            execute_step(MakeFile(item), context)
        context.trace.append(self.name)

    def describe(self) -> str:
        if self.when != "end":
            return f"SaveDeliverables(when={self.when!r})"
        return "SaveDeliverables()"
