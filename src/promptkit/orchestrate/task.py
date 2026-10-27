# orchestrate/task.py
'''`Task` / `TaskResult` / `Plan` —— 编排层的三块数据。

这里**只有数据**，不执行任何东西：怎么调度在 `orchestrate.orchestrator`，
怎么规划在 `orchestrate.planner`。这样「任务是什么」和「怎么跑」能分开测。
'''
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from promptkit.orchestrate.errors import ErrorCode, PlanError

__all__ = [
    "ON_FAILURE",
    "Task",
    "TaskResult",
    "Plan",
    "plan_from",
]

#: 任务失败之后怎么办
ON_FAILURE = ("retry", "skip", "abort", "fallback")


@dataclass
class Task:
    '''一个子任务：做什么、谁来干、依赖谁、结果放哪。

        Task("research", "调研市场现状", agent=researcher, output_key="research")
        Task("write", "写报告：{{analysis}}", depends_on=("analyze",),
             role="writer", input_keys=("analysis",))

    字段都显式：没有「自动推断依赖」，也没有「偷偷继承父上下文」。
    '''

    name: str
    description: str = ""
    #: None 时由 `ExecutePlan(agent_factory=...)` / 默认 agent 补上
    agent: Any = None
    #: 这个任务的「角色名」，交给 `RoleFactory` 造 agent 时用
    role: str = ""
    depends_on: Tuple[str, ...] = ()
    #: 允许在 `description` 里用哪些变量（空 = 允许全部可用变量）
    input_keys: Tuple[str, ...] = ()
    #: 结果写进 Context 的哪个 key（空 = 用任务名）
    output_key: str = ""
    max_retries: int = 1
    on_failure: str = "retry"
    fallback_agent: Any = None
    #: 这次任务调用模型时要带的工具
    tools: Optional[Sequence[Any]] = None
    #: 单任务超时（秒）；None = 用 OrchestratorConfig.per_task_timeout
    timeout: Optional[float] = None
    #: 给了就要求输出是这个 Pydantic 模型（走 Executor 的结构化输出）
    output_model: Any = None

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise PlanError("任务必须有名字", code=ErrorCode.TASK_INVALID)
        self.description = str(self.description or "").strip() or self.name
        self.role = str(self.role or "").strip()
        self.depends_on = _unique(self.depends_on)
        self.input_keys = _unique(self.input_keys)
        self.max_retries = int(self.max_retries)
        if self.max_retries < 0:
            raise PlanError(
                f"任务 {self.name!r} 的 max_retries 不能是负数",
                code=ErrorCode.TASK_INVALID,
            )
        self.on_failure = str(self.on_failure or "retry").lower()
        if self.on_failure not in ON_FAILURE:
            raise PlanError(
                f"任务 {self.name!r} 的 on_failure={self.on_failure!r} 不合法；"
                f"可选：{', '.join(ON_FAILURE)}",
                code=ErrorCode.TASK_INVALID,
            )
        if self.on_failure == "fallback" and self.fallback_agent is None:
            raise PlanError(
                f"任务 {self.name!r} 选了 fallback，但没给 fallback_agent",
                code=ErrorCode.TASK_INVALID,
            )
        if self.timeout is not None and float(self.timeout) <= 0:
            raise PlanError(
                f"任务 {self.name!r} 的 timeout 必须大于 0",
                code=ErrorCode.TASK_INVALID,
            )

    @property
    def key(self) -> str:
        '''结果在 Context 里放哪儿。'''
        return self.output_key or self.name

    def to_spec(self) -> Dict[str, Any]:
        '''转成可序列化的样子（规划器 / 日志用）。'''
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "depends_on": list(self.depends_on),
            "input_keys": list(self.input_keys),
            "output_key": self.key,
            "max_retries": self.max_retries,
            "on_failure": self.on_failure,
        }

    def __repr__(self) -> str:
        deps = f" after {list(self.depends_on)}" if self.depends_on else ""
        return f"<Task {self.name!r}{deps}>"


@dataclass
class TaskResult:
    '''一个子任务跑完之后的样子 —— 成功、失败、跳过都是这个形状。'''

    name: str
    status: str = "success"            # success / failed / skipped
    output: Any = None
    error: Optional[str] = None
    tokens: int = 0
    latency: float = 0.0
    attempts: int = 0
    output_key: str = ""
    #: 原始异常对象（只在 orchestrator 内部用，to_dict 不暴露）
    _exception: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "output": _plain(self.output),
            "error": self.error,
            "tokens": self.tokens,
            "latency": round(self.latency, 3),
            "attempts": self.attempts,
            "output_key": self.output_key or self.name,
        }

    def __repr__(self) -> str:
        mark = {"success": "✓", "failed": "✗", "skipped": "·"}.get(self.status, "?")
        return f"<TaskResult {mark} {self.name} {self.latency:.2f}s>"


@dataclass
class Plan:
    '''一组有依赖关系的任务。构造时就把「名字唯一 / 依赖存在 / 无环」验掉。'''

    tasks: Tuple[Task, ...] = ()
    #: 原始需求（LLM 规划时会填；固定模板时一般空着）
    task: str = ""

    def __post_init__(self) -> None:
        self.tasks = tuple(self.tasks)
        if not self.tasks:
            raise PlanError("计划里没有任何任务", code=ErrorCode.PLAN_EMPTY)
        seen: Dict[str, Task] = {}
        for task in self.tasks:
            if not isinstance(task, Task):
                raise PlanError(
                    f"计划里混进了不是 Task 的东西：{task!r}",
                    code=ErrorCode.PLAN_INVALID,
                )
            if task.name in seen:
                raise PlanError(
                    f"任务名重复：{task.name!r}（名字要唯一）",
                    code=ErrorCode.PLAN_DUPLICATE_TASK,
                )
            seen[task.name] = task
        for task in self.tasks:
            for dependency in task.depends_on:
                if dependency == task.name:
                    raise PlanError(
                        f"任务 {task.name!r} 不能依赖自己",
                        code=ErrorCode.PLAN_UNKNOWN_DEPENDENCY,
                    )
                if dependency not in seen:
                    raise PlanError(
                        f"任务 {task.name!r} 依赖的 {dependency!r} 不在计划里",
                        code=ErrorCode.PLAN_UNKNOWN_DEPENDENCY,
                    )
        self.topological()      # 顺带查环

    # ---------------- 基本操作 ----------------
    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __getitem__(self, name: str) -> Task:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(name)

    @property
    def names(self) -> List[str]:
        return [task.name for task in self.tasks]

    def topological(self) -> List[Task]:
        '''拓扑序；同一层里保持**声明顺序**（结果可复现，测试好写）。'''
        done: set = set()
        order: List[Task] = []
        pending = list(self.tasks)
        while pending:
            ready = [
                task for task in pending if all(dep in done for dep in task.depends_on)
            ]
            if not ready:
                left = ", ".join(task.name for task in pending)
                raise PlanError(
                    f"任务依赖有环，转不动了：{left}", code=ErrorCode.PLAN_CYCLE
                )
            for task in ready:
                order.append(task)
                done.add(task.name)
            pending = [task for task in pending if task.name not in done]
        return order

    def waves(self) -> List[List[Task]]:
        '''按依赖分层：同一层的任务彼此无关，可以并发。'''
        done: set = set()
        layers: List[List[Task]] = []
        pending = list(self.tasks)
        while pending:
            ready = [
                task for task in pending if all(dep in done for dep in task.depends_on)
            ]
            if not ready:
                raise PlanError("任务依赖有环，分不出层", code=ErrorCode.PLAN_CYCLE)
            layers.append(ready)
            done.update(task.name for task in ready)
            pending = [task for task in pending if task.name not in done]
        return layers

    # ---------------- 构造 ----------------
    @classmethod
    def from_template(
        cls,
        tasks: Sequence[Any],
        *,
        task: str = "",
        max_tasks: Optional[int] = None,
    ) -> "Plan":
        '''把一串 `Task`（或它们的 dict 写法）变成计划。

        `max_tasks` 只**校验**不截断：模板是你自己写的，超了说明写错了，
        直接报错比悄悄砍掉最后一个任务安全。截断是 LLM 规划那边的事。
        '''
        built = [item if isinstance(item, Task) else Task(**item) for item in tasks]
        if max_tasks is not None and len(built) > int(max_tasks):
            raise PlanError(
                f"计划有 {len(built)} 个任务，超过上限 {int(max_tasks)}；"
                "要么改模板，要么调大 max_tasks",
                code=ErrorCode.PLAN_TOO_MANY_TASKS,
            )
        return cls(tuple(built), task=task)

    def describe(self) -> str:
        order = [task.name for task in self.topological()]
        return f"Plan({len(self)} 个任务: " + " -> ".join(order) + ")"

    def __repr__(self) -> str:
        return self.describe()


def _unique(values: Any) -> Tuple[str, ...]:
    result: List[str] = []
    for item in values or ():
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _plain(value: Any) -> Any:
    '''转成好打印 / 好存的样子（Pydantic 模型转 dict，别的原样）。'''
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:       # noqa: BLE001 - 转换失败就原样给回去
            return value
    return value


def plan_from(value: Any) -> Plan:
    '''把 `Plan` / `Task` 列表 / `{"tasks": [...]}` / `PlanSpec` 统一成 `Plan`。

    这是全项目**唯一**的计划解析入口：`ExecutePlan` / `Orchestrator.run` /
    `orchestrate.api` 都走它，所以 DSL 写法在各处表现一致。
    '''
    if isinstance(value, Plan):
        return value
    if isinstance(value, (list, tuple)):
        return Plan.from_template(value)
    if isinstance(value, dict) and "tasks" in value:
        return Plan.from_template(
            value["tasks"], task=str(value.get("task") or "")
        )
    to_plan = getattr(value, "to_plan", None)      # PlanSpec 等鸭子类型
    if callable(to_plan):
        return to_plan()
    raise PlanError(
        f"看不懂的计划：{type(value).__name__}；"
        "要 Plan、Task 列表、{'tasks': [...]} 或 PlanSpec",
        code=ErrorCode.PLAN_INVALID,
    )


def elapsed(started: float) -> float:
    return time.perf_counter() - started
