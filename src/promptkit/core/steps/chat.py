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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from promptkit.core.context import Context

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
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.function import FunctionStep
from promptkit.core.steps.workflow import Workflow
from promptkit.tools import Tool

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
    #: 让模型自己调用的能力：`FunctionDef`（DSL `Function`）/ `Skill` / 函数。
    #: 编译时会包成 Tool，并自动带上「这次运行用的 agent」。
    skills: List[Any] = field(default_factory=list)
    exit_words: List[str] = field(default_factory=lambda: list(DEFAULT_EXIT_WORDS))
    cautions: List[Any] = field(default_factory=list)
    max_turns: Optional[int] = None
    render: str = "markdown"
    read_prompt: str = ">>> "
    name: str = "chat"
    #: 会话文件路径（或 `Session`）。给了就「开跑前 load、结束时 save」，
    #: 下次用同一个路径就接着上次聊。
    session: Any = None
    #: 长期记忆（`KVMemory` / `VectorMemory` / 任何 `Memory`）
    memory: Any = None
    #: 记忆配置：读什么、召回什么、写什么（`MemoryConfig` 或等价的 dict）
    memory_config: Any = None

    def __post_init__(self) -> None:
        '''`memory` 和 `memory_config` 必须成对出现，而且不能是空配置。

        这种错误**必须**在构造时就报：配了记忆却没写配置，跑起来会「一切正常」
        但什么都没记住 —— 比报错难查得多。

        例外：只配 `memory` + `skills`（聊天层不自动读写，记忆只进出 Skill）。
        那时候记忆**确实在用**，不算是「配了没生效」。
        '''
        if self.memory is None and self.memory_config is None:
            return
        if self.memory is not None and self.memory_config is None and self.skills:
            return
        if self.memory is None or self.memory_config is None:
            raise ValueError(
                "ChatWorkflow 的 memory 和 memory_config 要么都给、要么都不给；"
                "只给一个的话记忆不会生效（只让 skills 用记忆时，可以不配 "
                "memory_config，但要先有 skills）"
            )
        from promptkit.memory import MemoryConfig

        if isinstance(self.memory_config, dict):
            try:
                self.memory_config = MemoryConfig(**self.memory_config)
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
        if self.memory is None or self.memory_config is None:
            return [], [], []

        config = self.memory_config
        reads: List[Step] = []
        recalls: List[Step] = []
        writes: List[Step] = []

        if config.read:
            reads.append(MemoryStep(self.memory, read=config.read))
        if config.has_recall():
            recalls.append(
                MemoryStep(
                    self.memory,
                    recall=config.recall,
                    budget=config.recall_budget(),
                )
            )
        if config.write:
            writes.append(MemoryStep(self.memory, write=config.write))
        return reads, recalls, writes

    def build(self) -> Workflow:
        '''编译成一个普通 `Workflow`。

        结构很直白：

            Role / Want / Rules / Tools / Deliverables / ExitWords  ← 写进 Context
            MemoryStep(read)                                        ← 有记忆时
            SystemPrompt()                                          ← 编译成 system prompt
            ChatLoop(...)                                           ← 多轮对话
            MemoryStep(write)                                       ← write_on="end"
            SaveDeliverables()                                      ← 结束时落盘
        '''
        declarations: List[Step] = [RoleStep(self.role)]
        if self.want:
            declarations.append(WantStep(self.want))
        if self.rules:
            declarations.append(RulesStep(*self.rules))
        if self.tools or self.skills:
            skill_tools: List[Any] = []
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
        each_turn: List[Any] = list(writes) if self._writes_each_turn() else []
        each_turn.append(SaveDeliverables(when="each_turn"))
        end_writes = [] if self._writes_each_turn() else list(writes)

        head: List[Any] = [*declarations, *reads]
        if skill_step is not None:
            # 放在 ChatLoop 之前：agent 已经由 Workflow.run 写进 Context 了
            head.append(skill_step)

        # 每轮刷新一次状态快照：技能工具里才能拿到 profile / topics 这些变量
        per_turn = [*recalls, *behavior]
        if skill_state_step is not None:
            per_turn.append(skill_state_step)

        return Workflow(
            *head,
            SystemPrompt(),
            # `when="each_turn"` 的交付物由 `ChatLoop` 自己每轮存，这里不用管
            ChatLoop(
                read_prompt=self.read_prompt,
                render=self.render,
                exit_words=self.exit_words,
                # 召回放在用户行为**之前**：behavior 里的步骤可以接着用召回结果
                behavior=per_turn,
                cautions=self.cautions or None,
                max_turns=self.max_turns,
                after_turn=each_turn,
                turn_template=self.turn_template(),
            ),
            *end_writes,
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
        from promptkit.memory.config import RecallConfig

        config = self.memory_config
        if config is None or not config.has_recall():
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
        return bool(
            self.memory_config is not None
            and getattr(self.memory_config, "write_on", "end") == "each_turn"
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
