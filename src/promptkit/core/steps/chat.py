# core/steps/chat.py
'''把「聊天机器人」的声明式配置编译成普通 `Workflow`。

    from promptkit.core.steps.chat import ChatWorkflow, Deliverable

    chat = ChatWorkflow(
        role="市场调研分析大赛的导师",
        want="帮参赛者找到合适的选题方向",
        rules=["像导师一样专业、耐心", "每次回复不超过 300 字"],
        deliverables=[
            Deliverable("选题建议", format="markdown", save_to="outputs/选题.md",
                        description="3-5 个可行的方向，含背景、方法、预期结论"),
        ],
        tools=[WEB_SEARCH],
    )
    chat.run(agent=tutor)

**`ChatWorkflow` 只是语法糖，不改核心。** `build()` 返回的就是一个普通
`Workflow` —— 现有的步骤树、事件、Verbosity、测试方法全都适用。
角色/目标/规矩/交付物编译成 system prompt，每轮行为用现成的
`While` / `If` / `LLMStep` / `Echo` 表达。

要长期记忆就给 `memory=` + `memory_config=`（见 README 第 23 节）：

    chat = ChatWorkflow(
        role="课程导师",
        memory=long_term_memory("learner_001"),      # KV + 向量
        memory_config=MemoryConfig(
            read=["level"],                           # 开跑前读
            recall=RecallConfig(output_key="related"),# 每轮召回
            write=["topics_covered"],                 # 结束时写
        ),
    )
'''
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

if TYPE_CHECKING:
    from promptkit.core.context import Context
    from promptkit.core.display_sys import Verbosity
    from promptkit.memory.budget import TokenBudget
    from promptkit.session import Session
    from promptkit.types import AgentLike, JSONValue, MemoryLike, ToolLike

from promptkit.memory.config import MemoryConfig, RecallConfig

from promptkit.core.steps.actions.chat_util import (
    CHAT_INPUT_KEY,
    CHAT_REPLY_KEY,
    DEFAULT_EXIT_WORDS,
    Behavior,
    ChatLoop,
    Deliverables as DeliverablesStep,
    ExitWords,
    MakeFile,
    Role as RoleStep,
    Rules as RulesStep,
    SaveDeliverables,
    SystemPrompt,
    Tools as ToolsStep,
    Want as WantStep,
    build_system_prompt,
)
from promptkit.core.steps.actions.memory import MemoryStep
from promptkit.core.steps.interface import Step, execute_step
from promptkit.core.steps.logics.function import FunctionStep
from promptkit.core.steps.workflow import Workflow
from promptkit.core.tool import Tool

__all__ = [
    "Deliverable",
    "ChatWorkflow",
    "chat_workflow",
]


class _MemoryConfigDict(TypedDict, total=False):
    '''`memory_config=` 接受的那种等价 dict（键与 `MemoryConfig` 同名）。'''

    read: List[str]
    recall: object
    write: List[str]
    budget: Optional["TokenBudget"]
    write_on: str


@dataclass
class Deliverable:
    '''一件要产出的东西。

    | 字段 | 说明 |
    | --- | --- |
    | `name` | 名字，会写进 system prompt |
    | `description` | 要求说明，也会写进 prompt |
    | `format` | `markdown` / `json` / `text`（含 `json` 就走 `json.dumps`） |
    | `save_to` | 存哪儿；`None` 表示只在内存里 |
    | `source` | 从哪个变量取内容，默认 `model_reply`（模型这轮的回答） |
    | `when` | `end` 结束时存 / `each_turn` 每轮都存 |
    | `encoding` | 文件编码，默认跟全局设置 |

    ⚠️ `source` 是**关键**：模型一轮只产出一段回答，所以默认所有交付物
    都从 `model_reply` 取。想让「参考资料」单独出 JSON，得用另一条
    `LLMStep(output_model=...)` 把它抽出来、写到别的变量，再把
    `source` 指向那个变量。
    '''

    name: str
    description: str = ""
    format: str = "markdown"
    save_to: Optional[str] = None
    source: str = CHAT_REPLY_KEY
    when: str = "end"
    encoding: Optional[str] = None

    def __post_init__(self) -> None:
        if self.when not in ("end", "each_turn"):
            raise ValueError(
                f"Deliverable.when 只能是 'end' 或 'each_turn'，得到 {self.when!r}"
            )
        if not self.name:
            raise ValueError("Deliverable 必须有 name")


class _GuardedChatLoop(Step):
    '''`ChatLoop` 外面套一层「失败也要跑收尾」。

    循环结束后的 `MemoryStep(write)` 原本只在正常退出时执行。一旦断网、
    模型调用抛错、Ctrl+C 或 `max_turns` 用尽，循环后面的步骤全被跳过 ——
    已经聊过的内容一条都没写进长期记忆，下次启动就像「记忆全没了」。

    这里把收尾步骤放进 `finally` 语义：无论循环怎么结束，都先写一次记忆
    再继续抛原异常。收尾自身失败时不覆盖原异常（断网时写记忆也很可能失败），
    只在 trace 里留一条记录。
    '''

    name = "chat_loop"

    def __init__(self, loop: Step, on_exit: Sequence[Step]):
        self.loop = loop
        self.on_exit = tuple(on_exit)

    def execute(self, context: Context) -> None:
        failure: Optional[BaseException] = None
        try:
            execute_step(self.loop, context)
        except BaseException as exc:  # 控制流信号也要先收尾再往上抛
            failure = exc

        cleanup_error: Optional[BaseException] = None
        for step in self.on_exit:
            try:
                execute_step(step, context)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                context.trace.append(
                    f"cleanup:{step.name}:error:{type(exc).__name__}"
                )

        if failure is not None:
            raise failure
        if cleanup_error is not None:
            raise cleanup_error

    def describe(self) -> str:
        return f"ChatLoop(失败也会跑 {len(self.on_exit)} 步收尾)"

    def children(self) -> List[Step]:
        return [self.loop, *self.on_exit]

    def __repr__(self) -> str:
        return f"<chat_loop guarded steps={len(self.on_exit)}>"


@dataclass
class ChatWorkflow:
    '''聊天机器人的声明式描述。`build()` 编译成普通 `Workflow`。'''

    role: str = "通用 AI 助手"
    want: str = ""
    rules: List[str] = field(default_factory=list)
    deliverables: List[Deliverable] = field(default_factory=list)
    behavior: List[Step] = field(default_factory=list)
    #: 进入循环前跑**一次**的步骤（开场白：LoadMem、打招呼、读会话……）。
    #: 位置在 SystemPrompt 之后、ChatLoop 之前 —— 这样开场白里的模型调用
    #: 能带上角色设定。手写 Workflow 时直接把步骤放在 ChatLoop 前面即可，
    #: 这个字段只是给声明式写法一个等价的位置。
    opening: List[Step] = field(default_factory=list)
    #: 用户决定退出对话后要跑的步骤（说再见、存 session、总结等）。
    #: 在 `ChatLoop` 之后、写记忆与落盘之前执行。
    after_end: List[Step] = field(default_factory=list)
    tools: List["ToolLike"] = field(default_factory=list)
    #: 让模型自己调用的能力：`FunctionDef`（DSL `Function`）/ `Skill` / 函数。
    #: 编译时会包成 Tool，并自动带上「这次运行用的 agent」。
    skills: List[object] = field(default_factory=list)
    exit_words: List[str] = field(default_factory=lambda: list(DEFAULT_EXIT_WORDS))
    cautions: List[object] = field(default_factory=list)
    max_turns: Optional[int] = None
    #: 对话历史超过多少条消息时自动压缩（None = 不压缩）：
    #: 旧消息总结成一段摘要，最近 `compact_keep` 条原样保留。
    compact_after: Optional[int] = None
    #: 压缩时最近保留多少条消息
    compact_keep: int = 6
    #: 自定义压缩提示词（`{{messages}}` 会被换成旧对话文本）
    compact_prompt: Optional[str] = None
    render: str = "markdown"
    read_prompt: str = ">>> "
    name: str = "chat"
    #: 会话文件路径（或 `Session`）。给了就「开跑前 load、结束时 save」，
    #: 下次用同一个路径就接着上次聊。
    session: Optional[Union[str, "Session"]] = None
    #: 会话目录：`session.json` + `memory.db` 都放这里，一次配好短期 + 长期。
    #: 显式给了 `session` / `memory` 时以显式值为准。
    store: Optional[str] = None
    #: 要跟着会话一起存的 Context 变量名（如 `["basic_info"]`），
    #: 下次 `run()` 自动装回 Context。
    session_variables: Sequence[str] = ()
    #: 压缩出来的摘要是否转存长期记忆。开启后短期上下文被压掉的内容，
    #: 仍能通过语义召回找回来；需要同时配 `compact_after` 和 `memory`。
    promote_compaction: bool = False
    #: 压缩摘要写进记忆时的键前缀（实际键为 `前缀:时间戳`，每次压缩追加一条）
    compact_memory_key: str = "conversation_summary"
    #: 长期记忆（`KVMemory` / `VectorMemory` / 任何 `Memory`），或一个落盘路径
    memory: Optional[Union["MemoryLike", str, os.PathLike[str]]] = None
    #: 记忆配置：读什么、召回什么、写什么（`MemoryConfig` 或等价的 dict）
    memory_config: Optional[Union[MemoryConfig, Dict[str, "JSONValue"]]] = None
    #: 知识库：目录/文件路径、`KnowledgeBase`、现成检索 `Tool`，或它们的列表。
    #: 编译成检索工具合并进 `tools`，让模型自己决定什么时候查资料。
    knowledge: object = None
    #: 命名空间：仅当 `memory` 是路径字符串、需要现建 `KVMemory` 时生效。
    namespace: Optional[str] = None

    def __post_init__(self) -> None:
        '''`memory` 和 `memory_config` 必须成对出现，而且不能是空配置。

        这种错误**必须**在构造时就报：配了记忆却没写配置，跑起来会「一切正常」
        但什么都没记住 —— 比报错难查得多。

        例外：只配 `memory` + `skills`（聊天层不自动读写，记忆只进出 Skill）。
        那时候记忆**确实在用**，不算是「配了没生效」。
        '''
        # 工作区：一个目录装下短期会话和长期记忆，省得手动拼两个路径。
        if self.store:
            root = os.path.abspath(os.path.expanduser(os.fspath(self.store)))
            if self.session is None:
                self.session = os.path.join(root, "session.json")
            if self.memory is None:
                self.memory = os.path.join(root, "memory.db")

        if self.promote_compaction:
            if self.compact_after is None:
                raise ValueError(
                    "promote_compaction=True 需要 compact_after：没有压缩就没有摘要可存"
                )
            if self.memory is None and self.store is None:
                raise ValueError(
                    "promote_compaction=True 需要 memory（或用 store= 自动建一个）"
                )

        # 薄入口：memory 直接给路径字符串 → 现建一个持久化记忆。
        # 默认语义：KV 存结构化值 + HashEmbedding 向量做召回（免密钥、免额外依赖）。
        memory_was_path = isinstance(self.memory, (str, os.PathLike))
        if memory_was_path:
            from promptkit.memory import long_term_memory
            from promptkit.memory.adapters import HashEmbedding

            path = os.fspath(
                cast(Union[str, os.PathLike[str]], self.memory)
            )
            self.memory = long_term_memory(
                self.namespace or "default",
                path=path,
                vector_path=path + ".vectors",
                embed_client=HashEmbedding(),
            )
        # 知识库：路径 / KnowledgeBase / 检索 Tool / 列表 → 合并进 tools
        self._coerce_knowledge()

        if self.memory is None and self.memory_config is None:
            return
        if self.memory is not None and self.memory_config is None and self.skills:
            return
        # 薄入口兜底：memory 是路径、没给 memory_config 时，自动配一个
        # 「每轮按用户输入召回相关记忆」的默认（写哪些键是业务相关的，不猜）。
        if self.memory_config is None and memory_was_path:
            from promptkit.memory import MemoryConfig, RecallConfig

            self.memory_config = MemoryConfig(
                recall=RecallConfig(output_key="related")
            )
        if self.memory is None or self.memory_config is None:
            raise ValueError(
                "ChatWorkflow 的 memory 和 memory_config 要么都给、要么都不给；"
                "只给一个的话记忆不会生效（只让 skills 用记忆时，可以不配 "
                "memory_config，但要先有 skills）"
            )
        from promptkit.memory import MemoryConfig

        if isinstance(self.memory_config, dict):
            try:
                self.memory_config = MemoryConfig(
                    **cast(_MemoryConfigDict, self.memory_config)
                )
            except TypeError as exc:
                raise TypeError(f"memory_config 这个 dict 不对：{exc}") from None
        if not isinstance(self.memory_config, MemoryConfig):
            raise TypeError(
                "memory_config 需要 MemoryConfig（或等价 dict），"
                f"得到 {type(self.memory_config).__name__}"
            )
        if self.memory_config.is_empty():
            raise ValueError(
                "memory_config 里 read / recall / write 一个都没配，"
                "记忆不会有任何读写；要么配一个，要么把 memory 去掉"
            )

    def _coerce_knowledge(self) -> None:
        '''把 `self.knowledge` 统一解析成检索工具，合并进 `self.tools`。

        支持：目录/文件路径、`KnowledgeBase`、现成检索 `Tool`，或这几样的列表。
        列表里多个知识库会自动给工具名编号，避免 function-calling 重名。
        '''
        if self.knowledge is None:
            return
        from promptkit.knowledge import KNOWLEDGE_REFERENCE

        items = (
            list(self.knowledge)
            if isinstance(self.knowledge, (list, tuple))
            else [self.knowledge]
        )
        single = len(items) == 1
        for index, item in enumerate(items, start=1):
            if single:
                tool = KNOWLEDGE_REFERENCE(item)
            else:
                tool = KNOWLEDGE_REFERENCE(item, name=f"search_knowledge_base_{index}")
            self.tools.append(tool)

    # ---------------- 编译 ----------------
    def build_system_prompt(self) -> str:
        '''只拼 system prompt 文本，不构造 Workflow（方便单独检查）。'''
        return build_system_prompt(
            role=self.role,
            want=self.want,
            rules=self.rules,
            deliverables=self.deliverables,
        )

    def memory_parts(self) -> "Tuple[List[Step], List[Step], List[Step]]":
        '''把 `memory_config` 编译成三段步骤：(开跑前读, 每轮召回, 写回)。

        拆成三段是因为它们落在**不同位置**：

            MemoryStep(read)                 ← 循环前（system prompt 能用到）
            While(...) {
                MemoryStep(recall)           ← 每轮、读入输入之后、调模型之前
                主循环
                [MemoryStep(write)]          ← write_on="each_turn" 时
            }
            MemoryStep(write)                ← write_on="end" 时
        '''
        memory = self.memory
        config = self.memory_config
        if memory is None or not isinstance(config, MemoryConfig):
            return [], [], []
        if isinstance(memory, (str, os.PathLike)):
            return [], [], []

        reads: List[Step] = []
        recalls: List[Step] = []
        writes: List[Step] = []

        if config.read:
            reads.append(MemoryStep(memory, read=config.read))
        if config.has_recall():
            recalls.append(
                MemoryStep(
                    memory,
                    recall=config.recall,
                    budget=config.recall_budget(),
                )
            )
        if config.write:
            writes.append(MemoryStep(memory, write=config.write))
        return reads, recalls, writes

    def build(self) -> Workflow:
        '''编译成一个普通 `Workflow`。

        结构很直白：

            Role / Want / Rules / Tools / Deliverables / ExitWords  ← 写进 Context
            MemoryStep(read)                                        ← 有记忆时
            SystemPrompt()                                          ← 编译成 system prompt
            [opening]                                               ← 开场白：进循环前跑一次
            ChatLoop(...)                                           ← 多轮对话（含 after_end）
            SaveDeliverables()                                      ← 结束时落盘

        `write_on="end"` 的写记忆不再单独跟在循环后面，而是包进
        `_GuardedChatLoop` 的收尾里：循环因断网 / 异常 / Ctrl+C 结束时
        也会先写记忆，再抛原异常。
        '''
        declarations: List[Step] = [RoleStep(self.role)]
        if self.want:
            declarations.append(WantStep(self.want))
        if self.rules:
            declarations.append(RulesStep(*self.rules))
        if self.tools or self.skills:
            skill_tools: List[Tool] = []
            skill_step: Optional[Step] = None
            skill_state_step: Optional[Step] = None
            if self.skills:
                from promptkit.skill.chat import AgentHolder, as_tools

                # memory 编译期就知道；agent 要等运行时 —— 一起放进 holder
                holder = AgentHolder(memory=self.memory)
                skill_tools = as_tools(self.skills, holder=holder)
                # 这一轮用哪个 agent，编译期不知道 —— 运行时把 context["agent"] 塞进去
                skill_step = FunctionStep(
                    lambda context: holder.update(context), name="skills"
                )
                # 每轮把状态快照一份，技能里才能「继承状态」
                skill_state_step = FunctionStep(
                    lambda context: holder.snapshot(context), name="skills_state"
                )
            declarations.append(ToolsStep(*self.tools, *skill_tools))
        else:
            skill_step = None
            skill_state_step = None
        if self.deliverables:
            declarations.append(DeliverablesStep(*self.deliverables))
        declarations.append(ExitWords(*self.exit_words))

        behavior = [Behavior(*self.behavior)] if self.behavior else []
        reads, recalls, writes = self.memory_parts()

        # 每轮收尾：记忆写在交付物之前（交付物要用到记忆里的值也不吃亏）
        each_turn: List[Step] = list(writes) if self._writes_each_turn() else []
        each_turn.append(SaveDeliverables(when="each_turn"))
        end_writes = [] if self._writes_each_turn() else list(writes)

        head: List[Step] = [*declarations, *reads]
        if skill_step is not None:
            # 放在 ChatLoop 之前：agent 已经由 Workflow.run 写进 Context 了
            head.append(skill_step)

        # 每轮刷新一次状态快照：技能工具里才能拿到 profile / topics 这些变量
        per_turn = [*recalls, *behavior]
        if skill_state_step is not None:
            per_turn.append(skill_state_step)

        # `when="each_turn"` 的交付物由 `ChatLoop` 自己每轮存，这里不用管
        loop: Step = ChatLoop(
            read_prompt=self.read_prompt,
            render=self.render,
            exit_words=self.exit_words,
            # 召回放在用户行为**之前**：behavior 里的步骤可以接着用召回结果
            behavior=per_turn,
            cautions=self.cautions or None,
            max_turns=self.max_turns,
            after_turn=each_turn,
            after_end=self.after_end,
            turn_template=self.turn_template(),
            compact_after=self.compact_after,
            compact_keep=self.compact_keep,
            compact_prompt=self.compact_prompt,
            # 压缩摘要转存长期记忆：短期窗口滚过去的内容，靠召回还能找回来
            compact_memory=self.memory if self.promote_compaction else None,
            compact_memory_key=self.compact_memory_key,
        )
        if end_writes:
            # 断网 / Ctrl+C / max_turns 用尽时，循环后面的写记忆会被跳过。
            # 套一层 finally 语义，保证已经产生的记忆先落库再抛异常。
            loop = _GuardedChatLoop(loop, end_writes)

        return Workflow(
            *head,
            SystemPrompt(),
            # 开场白：进循环前跑一次（手写 DSL 里就是 ChatLoop 前面的普通步骤）
            *self.opening,
            loop,
            SaveDeliverables(self.end_deliverables()),
            name=self.name,
        )

    def turn_template(self) -> Optional[str]:
        '''配了召回时，每一轮发给模型的内容长什么样。

        召回结果是**每轮**的信息，所以拼在「这一轮的用户消息」里，而不是
        system prompt 里：

            {{usr_input}}

            以下是长期记忆里检索到的相关片段（可能有用，也可能不相关，别硬套）：
            {{related}}

        `recall.raw=True`（要结构化结果）时返回 `None` —— 那种情况下
        调用方得自己组装 prompt，硬塞 `list[dict]` 的 repr 进去只会让模型困惑。
        '''
        config = self.memory_config
        if not isinstance(config, MemoryConfig) or not config.has_recall():
            return None
        recall = config.recall
        if not isinstance(recall, RecallConfig) or recall.raw:
            return None
        return (
            "{{%s}}\n\n以下是长期记忆里检索到的相关片段"
            "（可能有用，也可能不相关，别硬套）：\n{{%s}}"
            % (CHAT_INPUT_KEY, recall.output_key)
        )

    def _writes_each_turn(self) -> bool:
        config = self.memory_config
        return isinstance(config, MemoryConfig) and config.write_on == "each_turn"

    def each_turn_deliverables(self) -> List[Deliverable]:
        return [item for item in self.deliverables if item.when == "each_turn"]

    def end_deliverables(self) -> List[Deliverable]:
        return [item for item in self.deliverables if item.when == "end"]

    # ---------------- 运行 ----------------
    def run(
        self,
        agent: Optional["AgentLike"] = None,
        *,
        verbose: Optional[Union["Verbosity", int, bool]] = None,
        reset: Optional[bool] = None,
        session: Optional[Union[str, "Session"]] = None,
        variables: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> "Context":
        '''编译并运行，返回 Context。

        - `agent` 省略时用 `Agent()` 注册的默认实例
        - `session`：会话文件路径（或 `Session`）。给了就
          **开跑前 load、结束时 save（异常退出也 save）**；
          下次用同一个路径就接着上次聊。`ChatWorkflow(session=...)` 是默认值，
          这里传的优先。
        - `reset`：要不要清空历史。`None`（默认）= 有会话文件就保留、
          没有就清空；显式传就按你传的来。

        实现上就是把编好的 workflow 交给 `Workflow.run(session=...)` ——
        和手写 `Workflow` 调 `wf.run(session=...)` 走的是**同一份代码**，
        两条路不会跑偏。
        '''
        from promptkit.core.display_sys import Verbosity

        if variables is None and self.session_variables:
            variables = list(self.session_variables)

        workflow = self.build()
        return workflow.run(
            agent=agent,
            session=session if session is not None else self.session,
            reset=reset,
            variables=variables,
            verbose=Verbosity.NORMAL if verbose is None else verbose,
            metadata={"role": self.role},
            **kwargs,
        )


def chat_workflow(**kwargs) -> ChatWorkflow:
    '''`ChatWorkflow(...)` 的函数式写法，省得 import 两个名字。'''
    return ChatWorkflow(**kwargs)
