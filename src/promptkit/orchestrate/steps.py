# orchestrate/steps.py
'''把编排接进工作流的三个 Step：

    PlanStep      需求 -> 计划（模板 or LLM）
    ExecutePlan   计划 -> 跑 -> Context 里拿到每个任务的结果
    AggregateStep 一堆结果 -> 一份汇总（可以先过 Pydantic 校验）
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from promptkit.core.event_sys.events import bus_of
from promptkit.core.steps.interface import Step
from promptkit.orchestrate.agents import as_text, structured_answer
from promptkit.orchestrate.errors import PlanError, TaskError
from promptkit.orchestrate.orchestrator import Orchestrator, OrchestratorConfig
from promptkit.orchestrate.planner import LLMPlanner, Planner
from promptkit.orchestrate.task import Plan, Task, TaskResult
from promptkit.prompt import render_text

__all__ = [
    "PlanStep",
    "ExecutePlan",
    "AggregateStep",
    "format_task_results",
    "plan_from",
]


# ============================ 规划 ============================
@dataclass
class PlanStep(Step):
    '''把需求拆成计划，写进 Context（是一个 `Plan` 对象，不是字符串）。

        PlanStep(task=Var("usr_input"), planner=planner_agent, max_tasks=5)
    '''

    task: Any = ""
    #: `Planner` 实例 / 带 `plan()` 的对象 / `Agent`（会包成 `LLMPlanner`）
    planner: Any = None
    output_key: str = "plan"
    max_tasks: int = 5
    #: 规划时可以看的上下文键（别把整个 Context 塞进提示词）
    context_keys: Tuple[str, ...] = ()
    tools: Optional[Sequence[Any]] = None

    name: str = "plan"

    def execute(self, context: "Any") -> None:
        task_text = _text_of(self.task, context)
        planner = self._planner()
        materials = {
            key: context[key] for key in self.context_keys if key in context
        }
        plan = planner.plan(task_text, materials)
        context[self.output_key] = plan
        context.trace.append(f"{self.name}:{len(plan)}")

    def _planner(self) -> Any:
        planner = self.planner
        if planner is None:
            raise ValueError(
                "PlanStep 需要 planner=...：一个 Planner，或者一个 Agent（让它规划）"
            )
        if isinstance(planner, Planner):
            return planner
        if hasattr(planner, "plan"):
            return planner
        if hasattr(planner, "ask"):
            return LLMPlanner(planner, max_tasks=self.max_tasks, tools=self.tools)
        raise TypeError(
            f"看不懂的 planner：{type(planner).__name__}；"
            "要 Planner 实例、带 plan() 的对象，或 Agent"
        )

    def describe(self) -> str:
        return f"PlanStep(-> {self.output_key}, 最多 {self.max_tasks} 个任务)"


# ============================ 执行 ============================
@dataclass
class ExecutePlan(Step):
    '''执行计划。

    执行完，Context 里会有：

    - `task_results`：`[{"name","status","output","error","tokens","latency"}, …]`
    - `plan_result`：`PlanResult` 对象（`ok` / `summary()` / `to_context()`）
    - 每个成功任务的 `output_key`（默认任务名）-> 它的输出
    '''

    plan: Any = None
    agent_factory: Optional[Callable[[Task], Any]] = None
    agent: Any = None
    config: Optional[OrchestratorConfig] = None
    output_key: str = "task_results"
    result_key: str = "plan_result"
    #: True = 只要有任务失败就抛 `TaskError`（否则留着结果自己判断）
    strict: bool = False

    name: str = "execute_plan"

    def execute(self, context: "Any") -> None:
        plan = plan_from(_value_of(self.plan, context))
        orchestrator = Orchestrator(
            self.config, agent=self.agent, agent_factory=self.agent_factory
        )
        result = orchestrator.run(plan, context=context.to_dict())

        context[self.output_key] = [item.to_dict() for item in result.results]
        context[self.result_key] = result
        for task, item in zip(plan, result.results):
            if item.ok:
                context[task.key] = item.output
        context.trace.append(
            f"{self.name}:{len(result.succeeded)}/{len(plan)}"
        )
        if self.strict and not result.ok:
            raise TaskError("有任务没有成功：\n" + result.summary(), result=result)

    def describe(self) -> str:
        return f"ExecutePlan(-> {self.output_key})"


# ============================ 汇总 ============================
@dataclass
class AggregateStep(Step):
    '''把子任务结果汇总成一份答复（可选：结构化成 Pydantic 对象）。

        AggregateStep(results_key="task_results", output_model=FinalReport,
                      agent=writer_agent)
    '''

    results_key: str = "task_results"
    output_key: str = "final"
    output_model: Any = None
    agent: Any = None
    tools: Optional[Sequence[Any]] = None
    instructions: str = "把这些子任务的结果合并成一份完整、不重复、不丢关键信息的答复。"
    max_chars: int = 8000

    name: str = "aggregate"

    def execute(self, context: "Any") -> None:
        results = context.get(self.results_key)
        if results is None:
            raise ValueError(
                f"Context 里没有 {self.results_key!r}；先跑 ExecutePlan"
            )
        if isinstance(results, (list, tuple)):
            items = list(results)
        else:
            items = [results]

        agent = self._agent()
        prompt = (
            f"{self.instructions}\n\n"
            f"子任务结果：\n{format_task_results(items, max_chars=self.max_chars)}"
        )
        bus = bus_of(context)
        if self.output_model is not None:
            value = structured_answer(
                agent, prompt, self.output_model, tools=self.tools, bus=bus
            )
        else:
            value = agent.ask(
                {"role": "user", "content": prompt}, tools=self.tools, bus=bus
            )
        context[self.output_key] = value
        context.trace.append(f"{self.name}:{len(items)}")

    def _agent(self) -> Any:
        if self.agent is not None:
            return self.agent
        from promptkit.agent import Agent

        agent = Agent.default()
        if agent is None:
            raise ValueError(
                "AggregateStep 需要一个 agent：传 agent=...，或先设一个默认 Agent"
            )
        return agent

    def describe(self) -> str:
        model = getattr(self.output_model, "__name__", None)
        suffix = f" -> {model}" if model else ""
        return f"AggregateStep({self.results_key} -> {self.output_key}{suffix})"


# ============================ 工具函数 ============================
def plan_from(value: Any) -> Plan:
    '''把 Plan / Task 列表 / `{"tasks": [...]}` 统一成 `Plan`。'''
    if isinstance(value, Plan):
        return value
    if isinstance(value, (list, tuple)):
        return Plan.from_template(value)
    if isinstance(value, dict) and "tasks" in value:
        return Plan.from_template(
            value["tasks"], task=str(value.get("task") or "")
        )
    raise PlanError(
        f"看不懂的计划：{type(value).__name__}；要 Plan、Task 列表，或 {{'tasks': [...]}}"
    )


def format_task_results(
    results: Sequence[Any], *, max_chars: int = 8000
) -> str:
    '''把结果列表拼成给模型看的文本（带状态，失败/跳过也写进去）。'''
    blocks: List[str] = []
    for item in results:
        data = _as_result_dict(item)
        name = data.get("name", "?")
        status = data.get("status", "?")
        body = as_text(data.get("output")) if data.get("output") is not None else ""
        error = data.get("error")
        block = f"## {name} [{status}]\n{body or '（没有输出）'}"
        if error and status != "success":
            block += f"\n（错误：{error}）"
        blocks.append(block)

    text = "\n\n".join(blocks) or "（没有子任务结果）"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（已截断）"

    bad = [
        _as_result_dict(item).get("name")
        for item in results
        if _as_result_dict(item).get("status") != "success"
    ]
    if bad:
        text += f"\n\n注意：这些任务没有成功：{', '.join(map(str, bad))}；"\
                "汇总时不要编造它们的结果。"
    return text


def _as_result_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, TaskResult):
        return item.to_dict()
    if isinstance(item, Mapping):
        return dict(item)
    return {"name": type(item).__name__, "status": "?", "output": item}


def _text_of(value: Any, context: "Any") -> str:
    '''把 `task=` 参数变成文本：字符串当模板渲染，表达式求值。'''
    if isinstance(value, str):
        return render_text(
            value, {key: as_text(item) for key, item in context.to_dict().items()}
        )
    if hasattr(value, "eval"):
        return as_text(value.eval(context))
    return as_text(value)


def _value_of(value: Any, context: "Any") -> Any:
    if hasattr(value, "eval"):
        return value.eval(context)
    return value
