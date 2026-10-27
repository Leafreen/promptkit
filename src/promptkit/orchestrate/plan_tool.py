# orchestrate/plan_tool.py
'''`PlanTool`：让**模型自己**决定「要不要拆、怎么拆」的规划工具。

和 `PlanStep` 的区别只有一点，但很关键：

    PlanStep     工作流替模型决定：到这一步必须规划
    PlanTool     模型自己决定：它可以在推理过程中挑时机调用，也可以不调

工具调用产生 `Plan` 后，会写进绑定的 `Context[output_key]`（默认 `"plan"`），
后面直接 `ExecutePlan(plan=Var("plan"))` 就能接着跑。

    tool = PlanTool(agent)                    # 工具视图，交给模型
    plan = PlanTool(agent, context=ctx)       # 绑了 Context：调一次就落盘
    agent.ask("先想清楚怎么做", tools=[tool])
'''
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from promptkit.orchestrate.agents import as_text
from promptkit.orchestrate.errors import ErrorCode, PlanError
from promptkit.orchestrate.planner import coerce_planner
from promptkit.orchestrate.task import Plan
from promptkit.tools import Tool

__all__ = ["PlanTool"]


class PlanTool(Tool):
    '''把需求拆成任务计划，并把 `Plan` 写进 Context。

        # 1) 工作流里：绑定 Context，模型调完就有 context["plan"]
        PlanWithTool(prompt="先规划：{{usr_input}}")

        # 2) 直接给 Agent 用：结果从 tool.plan 取
        tool = PlanTool(agent, max_calls=2)
        agent.ask("这活儿怎么拆？", tools=[tool])
        tool.plan
    '''

    def __init__(
        self,
        agent: Any = None,
        *,
        planner: Any = None,
        context: Any = None,
        output_key: str = "plan",
        max_tasks: int = 5,
        max_calls: int = 1,
        context_keys: Tuple[str, ...] = (),
        tools: Optional[Sequence[Any]] = None,
        name: str = "plan",
    ):
        if int(max_tasks) < 1:
            raise ValueError("max_tasks 至少是 1")
        if int(max_calls) < 1:
            raise ValueError("max_calls 至少是 1")
        if planner is None and agent is None:
            raise ValueError(
                "PlanTool 需要 agent=...（会包成 LLMPlanner）或 planner=...（Planner 实例）"
            )

        self.agent = agent
        self.planner = planner
        self.context = context
        self.output_key = output_key
        self.max_tasks = int(max_tasks)
        self.max_calls = int(max_calls)
        self.context_keys = tuple(context_keys)
        self.tools = tools
        self.used = 0
        #: 最近一次产出的计划（模型没调工具时是 None）
        self.plan: Optional[Plan] = None

        super().__init__(
            name=name,
            description=(
                "把一个大需求拆成若干可执行的子任务，返回计划和执行波次。"
                "适合任务明显能分头做、或需要先想清楚步骤的时候；"
                "一步一步就能做完的事不要调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "要拆的需求原文：目标、交付物、限制都要写清楚",
                    },
                    "context_keys": {
                        "type": "string",
                        "description": (
                            "可选。要参考的上下文变量名，逗号分隔（如 "
                            "'research,topic'）；不填就只用 task 本身"
                        ),
                    },
                },
                "required": ["task"],
            },
            func=self.run,
        )

    # ---------------- 工具接口 ----------------
    def run(self, task: str = "", context_keys: str = "") -> str:
        '''工具不抛异常：失败也返回一段「错误：……」，让模型自己决定下一步。'''
        try:
            return self._plan(task, context_keys)
        except PlanError as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001 - 工具契约是「总能返回」
            return f"错误：{type(exc).__name__}: {exc}"

    # ---------------- 绑定 / 重置 ----------------
    def bind(self, context: Any) -> "PlanTool":
        '''绑定工作流 Context：之后每次规划都会写进 `context[output_key]`。'''
        self.context = context
        return self

    def reset(self) -> None:
        '''清零调用次数与上次计划（新一轮任务开始时调）。'''
        self.used = 0
        self.plan = None

    # ---------------- 内部 ----------------
    def _plan(self, task: str, context_keys: str) -> str:
        text = str(task or "").strip()
        if not text:
            raise PlanError(
                "要规划的需求不能为空：把目标、交付物、限制写进 task",
                code=ErrorCode.INVALID_REQUEST,
            )
        if self.used >= self.max_calls:
            raise PlanError(
                f"这个任务里最多规划 {self.max_calls} 次，已经用完了；"
                "请按现有计划继续，不要再拆",
                code=ErrorCode.INVALID_REQUEST,
            )

        planner = coerce_planner(
            self.planner if self.planner is not None else self.agent,
            max_tasks=self.max_tasks,
            tools=self.tools,
        )
        plan: Plan = planner.plan(text, self._materials(context_keys))
        self.used += 1
        self.plan = plan
        if self.context is not None:
            self.context[self.output_key] = plan
        return _summary(plan)

    def _materials(self, context_keys: str) -> Dict[str, Any]:
        '''只取模型显式点名的上下文键 —— 不把整个 Context 塞进提示词。'''
        if self.context is None:
            return {}
        keys = [item.strip() for item in str(context_keys or "").split(",") if item.strip()]
        keys = keys or list(self.context_keys)
        if not keys:
            return {}
        data: Mapping[str, Any] = (
            self.context.to_dict()
            if callable(getattr(self.context, "to_dict", None))
            else dict(self.context)
        )
        return {key: as_text(data[key]) for key in keys if key in data}

    def __repr__(self) -> str:
        return (
            f"<PlanTool used={self.used}/{self.max_calls} "
            f"bound={self.context is not None}>"
        )


def _summary(plan: Plan) -> str:
    '''给模型看的计划摘要：任务清单 + 波次，不放完整描述（省 token）。'''
    lines = [f"已生成计划：{len(plan)} 个任务"]
    for index, task in enumerate(plan, start=1):
        depends = f"，依赖 {', '.join(task.depends_on)}" if task.depends_on else ""
        role = f"，角色 {task.role}" if task.role else ""
        lines.append(f"{index}. {task.name}{role}{depends}：{task.description[:80]}")
    waves = "；".join(
        f"第 {index} 波 {'、'.join(item.name for item in layer)}"
        for index, layer in enumerate(plan.waves(), start=1)
    )
    lines.append(f"执行顺序：{waves}")
    return "\n".join(lines)
