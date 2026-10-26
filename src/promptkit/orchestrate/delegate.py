# orchestrate/delegate.py
'''`DelegateTool`：让一个 Agent 自己决定「派谁去做哪件事」。

这是三种编排里最灵活、也最难控的一种，所以护栏都写在工具里：

- `max_delegations`：总共能派几次
- `max_depth`：能派几层（子代理还能不能再派）
- 子代理**独立**：调用前 `reset()`，不继承父代理的历史
'''
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from promptkit.orchestrate.errors import DelegateError, DepthExceeded
from promptkit.tools import Tool

__all__ = ["DelegateTool"]


class DelegateTool(Tool):
    '''把任务转给一个具名子代理执行，返回它的回答。

        tool = DelegateTool(
            {"researcher": researcher, "coder": coder},
            max_delegations=3,
            max_depth=1,
        )
        main_agent.ask("调研并实现一个爬虫", tools=[tool])
    '''

    def __init__(
        self,
        agents: Optional[Mapping[str, Any]] = None,
        *,
        agent_factory: Optional[Callable[[str], Any]] = None,
        max_delegations: int = 5,
        max_depth: int = 1,
        depth: int = 0,
        sub_tools: Optional[Callable[[str], Sequence[Any]]] = None,
        name: str = "delegate",
    ):
        self.agents: Dict[str, Any] = dict(agents or {})
        if not self.agents and agent_factory is None:
            raise ValueError(
                "DelegateTool 需要 agents={名字: Agent} 或 agent_factory=…… 至少一个"
            )
        self.agent_factory = agent_factory
        self.max_delegations = int(max_delegations)
        self.max_depth = int(max_depth)
        self.depth = int(depth)
        self.sub_tools = sub_tools
        self.used = 0

        if self.max_delegations < 1:
            raise ValueError("max_delegations 至少是 1")
        if self.max_depth < 1:
            raise ValueError("max_depth 至少是 1（否则这个工具没有意义）")
        if self.depth < 0:
            raise ValueError("depth 不能是负数")

        names = ", ".join(sorted(self.agents)) or "（由 agent_factory 按名字创建）"
        super().__init__(
            name=name,
            description=(
                "把一个子任务交给更合适的子代理去做，拿到它的回答。"
                f"可用的子代理：{names}。"
                "适合把「调研 / 写代码 / 校对」这类独立工作分出去；"
                "需要你自己综合判断时不要派。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": f"子代理名字，可选：{names}",
                        **(
                            {"enum": sorted(self.agents)}
                            if self.agents
                            else {}
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": "交给它的完整任务说明（它看不到我们的对话）",
                    },
                },
                "required": ["agent", "task"],
            },
            func=self.run,
        )

    # ---------------- 工具接口 ----------------
    def run(self, agent: str, task: str) -> str:
        try:
            return self._delegate(agent, task)
        except (DelegateError, DepthExceeded) as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001 - 工具总能返回点什么
            return f"错误：{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        '''把「已经派了几次」清零（一次新的主任务开始时调）。'''
        self.used = 0

    # ---------------- 内部 ----------------
    def _delegate(self, agent: str, task: str) -> str:
        name = str(agent or "").strip()
        text = str(task or "").strip()
        if not text:
            raise DelegateError("要派的任务不能是空的")
        if self.depth >= self.max_depth:
            raise DepthExceeded(
                f"委派深度已到上限（max_depth={self.max_depth}）；"
                "子代理不能再往下派了"
            )
        # 先解析目标：名字写错比「次数用完」更该先告诉模型
        target = self._target(name)
        if self.used >= self.max_delegations:
            raise DelegateError(
                f"这批任务已经派了 {self.used} 次（上限 {self.max_delegations}）；"
                "剩下的请自己完成"
            )
        self.used += 1

        reset = getattr(target, "reset", None)
        if callable(reset):
            reset()

        tools: List[Any] = []
        if self.sub_tools is not None:
            tools.extend(list(self.sub_tools(name)))
        if self.depth + 1 < self.max_depth:
            tools.append(
                DelegateTool(
                    self.agents,
                    agent_factory=self.agent_factory,
                    max_delegations=self.max_delegations,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    sub_tools=self.sub_tools,
                    name=self.name,
                )
            )

        reply = target.ask({"role": "user", "content": text}, tools=tools or None)
        return f"【{name}】{reply}"

    def _target(self, name: str) -> Any:
        if name in self.agents:
            return self.agents[name]
        if self.agent_factory is not None:
            target = self.agent_factory(name)
            if target is None:
                raise DelegateError(f"agent_factory 没有造出 {name!r}")
            return target
        known = ", ".join(sorted(self.agents)) or "（空）"
        raise DelegateError(f"没有叫 {name!r} 的子代理；可用：{known}")

    def __repr__(self) -> str:
        return (
            f"<DelegateTool depth={self.depth}/{self.max_depth} "
            f"used={self.used}/{self.max_delegations}>"
        )
