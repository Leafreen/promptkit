# skill/context.py
'''`SkillContext`：Skill 执行时能拿到的东西。

一个 Skill 不该自己 `Agent()`、自己连数据库、自己 print。需要什么就从 ctx 拿：
`ctx.agent` / `ctx.executor` / `ctx.client` / `ctx.memory` / `ctx.tools` /
`ctx.logger` / `ctx.trace`，以及 `ctx.call_skill()` 去调别的 Skill。
'''
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptkit.skill.errors import SkillError

__all__ = ["SkillContext"]

LOGGER_NAME = "promptkit.skill"


@dataclass
class SkillContext:
    '''Skill 的运行时上下文（也是测试时的注入点）。'''

    #: 干活用的 agent（`ask` / 结构化输出 / 跑工作流都靠它）
    agent: Any = None
    #: 底层厂商客户端（`agent.client`），一般不用直接碰
    client: Any = None
    #: 记忆系统（`promptkit.memory` 的任一实现），按需注入
    memory: Any = None
    #: 其它 Skill 的注册表（`ctx.call_skill` 用）
    skills: Any = None
    #: 这个 Skill 专属的工具：名字 -> Tool / 函数
    tools: Dict[str, Any] = field(default_factory=dict)
    #: 调用方的状态快照（`ChatWorkflow` 每轮更新）：`ctx.get("profile")` 读它
    state: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    #: 执行轨迹：`[{"skill": ..., "status": ..., "duration": ...}, ...]`
    trace: List[Dict[str, Any]] = field(default_factory=list)
    #: 跑工作流时是否打印过程
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.client is None and self.agent is not None:
            self.client = getattr(self.agent, "client", None)
        if self.logger is None:
            self.logger = logging.getLogger(LOGGER_NAME)

    # ---------------- 模型能力 ----------------
    @property
    def model(self) -> Optional[str]:
        return getattr(self.agent, "model", None)

    @property
    def vendor(self) -> Optional[str]:
        return getattr(self.agent, "vendor", None)

    def require_agent(self) -> Any:
        '''拿 agent：没给就用默认的；都没有就明确报错。'''
        if self.agent is not None:
            return self.agent
        from promptkit.agent import Agent

        agent = Agent.default()
        if agent is None:
            raise SkillError(
                "这个 Skill 需要模型，但 ctx 里没有 agent："
                "构造 SkillContext(agent=Agent())，或者先创建一个默认 Agent"
            )
        self.agent = agent
        return agent

    @property
    def executor(self) -> Any:
        '''结构化输出用：`ctx.executor.run(prompt, Model, **vars)`。'''
        from promptkit.executor import Executor

        return Executor(agent=self.require_agent())

    def ask(self, text: str, *, tools: Any = None, **kwargs: Any) -> str:
        '''直接问一句（`text` 是纯文本，不做模板渲染）。'''
        return self.require_agent().ask(text, tools=tools, **kwargs)

    def structured(
        self,
        text: str,
        output_model: Any,
        *,
        tools: Any = None,
        max_retries: Optional[int] = None,
    ) -> Any:
        '''要一个结构化结果（Pydantic 模型），解析失败带错误重试。

        和 `Executor` 的区别：这里的 `text` 是**数据**，不会被当成 `{{模板}}`
        再渲染一遍（技能里经常把用户输入 / 上游结果拼进去）。
        '''
        from promptkit.executor import run_structured

        return run_structured(
            self.require_agent(),
            text,
            output_model,
            tools=tools,
            max_retries=max_retries,
        )

    # ---------------- 组合 ----------------
    def run(self, workflow: Any, **variables: Any) -> Any:
        '''在 Skill 里跑一个完整工作流，返回它的 Context。'''
        started = time.perf_counter()
        context = workflow.run(
            agent=self.require_agent(), verbose=self.verbose, **variables
        )
        self.record(
            "workflow",
            duration=round(time.perf_counter() - started, 3),
            steps=len(getattr(context, "trace", []) or []),
        )
        return context

    def call_skill(self, name: str, /, **arguments: Any) -> Any:
        '''调另一个 Skill（Skill 是组合单元，不是孤立函数）。'''
        if self.skills is None:
            raise SkillError(
                f"ctx 里没有技能注册表，调不了 {name!r}；请用 "
                "SkillContext(skills=registry) 或 registry.execute(..., ctx=ctx)"
            )
        return self.skills.execute(name, arguments, ctx=self)

    def tool(self, name: str) -> Any:
        if name not in self.tools:
            available = ", ".join(sorted(self.tools)) or "（空）"
            raise SkillError(f"没有叫 {name!r} 的工具；可用：{available}")
        return self.tools[name]

    def get(self, key: str, default: Any = None) -> Any:
        '''读调用方继承下来的状态（`ctx.get("profile")` / `ctx.get("topics")`）。'''
        return self.state.get(key, default)

    def record(self, event: str, **data: Any) -> Dict[str, Any]:
        '''记一条轨迹（Skill 执行 / 工作流 / 自定义事件都能用）。'''
        entry = {"event": event, **data}
        self.trace.append(entry)
        return entry

    # ---------------- 记忆 ----------------
    def require_memory(self) -> Any:
        if self.memory is None:
            raise SkillError(
                "这个 Skill 需要记忆，但 ctx 里没有 memory："
                "SkillContext(memory=KVMemory(...))，或 ChatWorkflow(memory=..., "
                "memory_config=...)（后者会自动传进来）"
            )
        return self.memory

    def remember(self, key: str, value: Any) -> None:
        '''写一条长期记忆（键值）。'''
        self.require_memory().write(key, value)

    def read(self, key: str, default: Any = None) -> Any:
        '''读一条长期记忆；没有就返回 `default`。'''
        return self.require_memory().read(key, default)

    def recall(self, query: str, top_k: int = 5) -> Any:
        '''按意思检索记忆片段（需要带 `recall` 的实现，比如 `VectorMemory`）。'''
        memory = self.require_memory()
        if not callable(getattr(memory, "recall", None)):
            raise SkillError(
                f"{type(memory).__name__} 没有 recall；键值记忆只能用 read/write，"
                "要语义检索请用 VectorMemory 或 LayeredMemory"
            )
        try:
            return memory.recall(query, top_k=top_k)
        except NotImplementedError as exc:
            raise SkillError(
                f"{type(memory).__name__} 不支持语义召回：{exc}"
            ) from None

    # ---------------- 构造 ----------------
    @classmethod
    def from_agent(cls, agent: Any, **kwargs: Any) -> "SkillContext":
        return cls(agent=agent, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<SkillContext agent={self.model!r} "
            f"skills={len(self.skills) if self.skills is not None else 0} "
            f"trace={len(self.trace)}>"
        )
