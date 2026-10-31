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
from promptkit.orchestrate.dsl import PlanSpec
from promptkit.orchestrate.errors import PlanError, TaskError
from promptkit.orchestrate.orchestrator import Orchestrator, OrchestratorConfig
from promptkit.orchestrate.plan_tool import PlanTool
from promptkit.orchestrate.planner import coerce_planner
from promptkit.orchestrate.task import Task, TaskResult, plan_from
from promptkit.prompt import render_text

__all__ = [
    "PlanStep",
    "PlanWithTool",
    "PlanTool",
    "ExecutePlan",
    "AggregateStep",
    "format_task_results",
    "plan_from",
]

#: `parallel=True` 且没给 max_workers 时用的保守默认并行度。
#: 线程开太多反而抢锁，给个「够用」的小数字；要精细控制就传 config=。
_DEFAULT_PARALLEL_WORKERS = 4


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
        planner = coerce_planner(
            self.planner, max_tasks=self.max_tasks, tools=self.tools
        )
        materials = {
            key: context[key] for key in self.context_keys if key in context
        }
        plan = planner.plan(task_text, materials)
        context[self.output_key] = plan
        context.trace.append(f"{self.name}:{len(plan)}")

    def describe(self) -> str:
        return f"PlanStep(-> {self.output_key}, 最多 {self.max_tasks} 个任务)"


# ============================ 模型自主规划 ============================
@dataclass
class PlanWithTool(Step):
    '''让模型自己规划：把 `PlanTool` 交给它，产出的 `Plan` 写进 Context。

        Workflow(
            PlanWithTool(prompt="先想清楚怎么做：{{usr_input}}"),
            ExecutePlan(plan=Var("plan")),
        )

    和 `PlanStep` 的区别：`PlanStep` 是「到这一步必须规划」，
    这里模型可以只聊不拆（`required=False`），也可以拆两次（`max_calls=2`）。
    '''

    prompt: Any = ""
    agent: Any = None
    #: `Tool` 产出的 `Plan` 放哪个 key（默认 "plan"）
    output_key: str = "plan"
    #: 模型这一步的自然语言回复放哪个 key
    reply_key: str = "plan_reply"
    planner: Any = None
    max_tasks: int = 5
    max_calls: int = 1
    context_keys: Tuple[str, ...] = ()
    tools: Optional[Sequence[Any]] = None
    #: True = 模型没调规划工具就报错；False = 允许它只回话
    required: bool = True

    name: str = "plan_with_tool"

    def execute(self, context: "Any") -> None:
        if not str(self.prompt or "").strip():
            raise ValueError("PlanWithTool 需要 prompt=...：告诉模型什么时候该规划")
        agent = self.agent or context.get("agent")
        if agent is None:
            raise ValueError(
                "PlanWithTool 需要一个 agent：传 agent=...，或 Workflow.run(agent=...)"
            )

        tool = PlanTool(
            agent,
            planner=self.planner,
            context=context,
            output_key=self.output_key,
            max_tasks=self.max_tasks,
            max_calls=self.max_calls,
            context_keys=self.context_keys,
        )
        reply = agent.ask(
            {"role": "user", "content": _text_of(self.prompt, context)},
            tools=[*(self.tools or []), tool],
        )
        context[self.reply_key] = reply
        context.trace.append(
            f"{self.name}:{len(tool.plan) if tool.plan is not None else 0}"
        )
        if self.required and tool.plan is None:
            raise PlanError(
                "模型没有调用规划工具；把 required=False 允许它只回话，"
                "或在 prompt 里明确要求先拆任务"
            )

    def describe(self) -> str:
        return f"PlanWithTool(-> {self.output_key})"


# ============================ 执行 ============================
@dataclass
class ExecutePlan(Step):
    '''执行计划。

    执行完，Context 里会有：

    - `task_results`：`[{"name","status","output","error","tokens","latency"}, …]`
    - `plan_result`：`PlanResult` 对象（`ok` / `summary()` / `to_context()`）
    - 每个成功任务的 `output_key`（默认任务名）-> 它的输出

    agent 从哪来（按优先级）：`Task.agent` -> `agent_factory` -> `self.agent`
    -> `Workflow.run(agent=...)` 传进来的那个 -> 默认 `Agent`。

    并行：同层任务默认串行；`parallel=True` 用保守默认线程数并行跑，
    `max_workers=N` 显式定线程数。执行引擎本来就是
    `OrchestratorConfig.max_concurrency` + 线程池，这两个参数只是让
    常见写法少一层包装；token 上限 / 超时 / 失败策略还是要 `config=`。
    '''

    plan: Any = None
    agent_factory: Optional[Callable[[Task], Any]] = None
    agent: Any = None
    config: Optional[OrchestratorConfig] = None
    output_key: str = "task_results"
    result_key: str = "plan_result"
    #: True = 只要有任务失败就抛 `TaskError`（否则留着结果自己判断）
    strict: bool = False
    #: True = 同一层（没有依赖关系）的任务用线程池并行跑。
    #: 执行引擎（`OrchestratorConfig.max_concurrency` + 线程池）本来就有，
    #: 这组参数只是省得手写 config。
    parallel: bool = False
    #: 并行线程数；给了就生效（显式给线程数就是想并行，不必再开 parallel）。
    #: 与 `config=` 互斥 —— 同时给报 ValueError。
    max_workers: Optional[int] = None

    name: str = "execute_plan"

    def execute(self, context: "Any") -> None:
        spec = PlanSpec.from_dsl(_value_of(self.plan, context))
        plan = spec.to_plan()
        if self.config is not None and (
            self.parallel or self.max_workers is not None
        ):
            raise ValueError(
                "ExecutePlan 的 config= 和 parallel= / max_workers= 别同时给："
                "要精细控制就只传 config，要省事就只传 parallel / max_workers"
            )
        config = self.config or spec.to_config()
        if self.max_workers is not None:
            workers = int(self.max_workers)
            if workers < 1:
                raise ValueError(f"max_workers 要 >= 1，得到 {self.max_workers!r}")
            config = config or OrchestratorConfig()
            config.max_concurrency = workers
        elif self.parallel:
            config = config or OrchestratorConfig()
            config.max_concurrency = max(
                config.max_concurrency, _DEFAULT_PARALLEL_WORKERS
            )
        factory = self.agent_factory or spec.to_factory()
        # 没显式给 agent 就用 `Workflow.run(agent=...)` 登记在 Context 里的那个，
        # 与 PlanWithTool / LLMStep 的取法保持一致（最后才是 Agent.default()）
        agent = self.agent if self.agent is not None else context.get("agent")
        orchestrator = Orchestrator(
            config, agent=agent, agent_factory=factory
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
# `plan_from` 的唯一实现在 `orchestrate.task`（`Orchestrator` 与 `api` 共用），
# 这里保留导出以维持旧的 `from promptkit.orchestrate.steps import plan_from`。


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
