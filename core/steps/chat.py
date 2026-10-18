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
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

if TYPE_CHECKING:
    from promptkit.core.context import Context

from promptkit.core.steps.actions.chat_util import (
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
from promptkit.core.steps.interface import Step
from promptkit.core.steps.workflow import Workflow

__all__ = [
    "Deliverable",
    "ChatWorkflow",
    "chat_workflow",
]


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


@dataclass
class ChatWorkflow:
    '''聊天机器人的声明式描述。`build()` 编译成普通 `Workflow`。'''

    role: str = "通用 AI 助手"
    want: str = ""
    rules: List[str] = field(default_factory=list)
    deliverables: List[Deliverable] = field(default_factory=list)
    behavior: List[Any] = field(default_factory=list)
    tools: List[Any] = field(default_factory=list)
    exit_words: List[str] = field(default_factory=lambda: list(DEFAULT_EXIT_WORDS))
    cautions: List[Any] = field(default_factory=list)
    max_turns: Optional[int] = None
    render: str = "markdown"
    read_prompt: str = ">>> "
    name: str = "chat"
    #: 会话文件路径（或 `Session`）。给了就「开跑前 load、结束时 save」，
    #: 下次用同一个路径就接着上次聊。
    session: Any = None

    # ---------------- 编译 ----------------
    def build_system_prompt(self) -> str:
        '''只拼 system prompt 文本，不构造 Workflow（方便单独检查）。'''
        return build_system_prompt(
            role=self.role,
            want=self.want,
            rules=self.rules,
            deliverables=self.deliverables,
        )

    def build(self) -> Workflow:
        '''编译成一个普通 `Workflow`。

        结构很直白：

            Role / Want / Rules / Tools / Deliverables / ExitWords  ← 写进 Context
            SystemPrompt()                                          ← 编译成 system prompt
            ChatLoop(...)                                           ← 多轮对话
            SaveDeliverables()                                      ← 结束时落盘
        '''
        declarations: List[Step] = [RoleStep(self.role)]
        if self.want:
            declarations.append(WantStep(self.want))
        if self.rules:
            declarations.append(RulesStep(*self.rules))
        if self.tools:
            declarations.append(ToolsStep(*self.tools))
        if self.deliverables:
            declarations.append(DeliverablesStep(*self.deliverables))
        declarations.append(ExitWords(*self.exit_words))

        behavior = [Behavior(*self.behavior)] if self.behavior else []

        return Workflow(
            *declarations,
            SystemPrompt(),
            # `when="each_turn"` 的交付物由 `ChatLoop` 自己每轮存，这里不用管
            ChatLoop(
                read_prompt=self.read_prompt,
                render=self.render,
                exit_words=self.exit_words,
                behavior=behavior,
                cautions=self.cautions or None,
                max_turns=self.max_turns,
            ),
            SaveDeliverables(self.end_deliverables()),
            name=self.name,
        )

    def each_turn_deliverables(self) -> List[Deliverable]:
        return [item for item in self.deliverables if item.when == "each_turn"]

    def end_deliverables(self) -> List[Deliverable]:
        return [item for item in self.deliverables if item.when == "end"]

    # ---------------- 运行 ----------------
    def run(
        self,
        agent: Any = None,
        *,
        verbose: Any = None,
        reset: Any = None,
        session: Any = None,
        variables: Any = None,
        **kwargs: Any,
    ) -> "Context":
        '''编译并运行，返回 Context。

        - `agent` 省略时用 `Agent()` 注册的默认实例
        - `session`：会话文件路径（或 `Session`）。给了就
          **开跑前 load、结束时 save（异常退出也 save）**；
          下次用同一个路径就接着上次聊。`ChatWorkflow(session=...)` 是默认值，
          这里传的优先。
        - `reset`：要不要清空历史。`None`（默认）= 有会话文件就保留、
          没有就清空；显式传就按你传的来。

        实现上就是把编好的 workflow 交给 `session.run_with_session()` ——
        和手写 `Workflow` 调 `wf.run(session=...)` 走的是**同一份代码**，
        两条路不会跑偏。
        '''
        from promptkit.core.display_sys import Verbosity
        from promptkit.session import run_with_session

        workflow = self.build()
        return run_with_session(
            workflow,
            agent,
            session if session is not None else self.session,
            reset=reset,
            variables=variables,
            verbose=Verbosity.NORMAL if verbose is None else verbose,
            agent_required="ChatWorkflow.run()",
            metadata={"role": self.role},
            **kwargs,
        )


def chat_workflow(**kwargs: Any) -> ChatWorkflow:
    '''`ChatWorkflow(...)` 的函数式写法，省得 import 两个名字。'''
    return ChatWorkflow(**kwargs)
