# orchestrate/__init__.py
'''多代理编排：把大任务拆开、分头执行、汇总结果、控制成本。

一句话：**`Plan`（拆任务）+ `Orchestrator`（调度）+ `SubAgent`（隔离执行）
+ `Aggregate`（汇总）**。

三个概念别混：

- **任务规划**：把大任务拆成小任务 —— `TemplatePlanner` / `LLMPlanner`
- **子代理**：每个子任务独立执行 —— `sub_agent()` / `RoleFactory`
- **编排**：调度、汇总、重试、预算 —— `Orchestrator` + 三个 Step

    from promptkit.orchestrate import (
        AggregateStep, ExecutePlan, OrchestratorConfig, Plan, PlanStep,
        RoleFactory, Task, TemplatePlanner,
    )

    plan = Plan.from_template([
        Task("research", "调研市场现状", role="researcher", output_key="research"),
        Task("write", "根据 {{research}} 写一份报告", role="writer",
             depends_on=("research",), output_key="report"),
    ])

    workflow = Workflow(
        ExecutePlan(plan=plan, agent_factory=RoleFactory({...})),
        AggregateStep(results_key="task_results", agent=writer),
    )

护栏（默认就开着）：`max_tasks=10`、`max_concurrency=1`、任务级
`on_failure`、可选的 `max_total_tokens`。委派还有 `max_delegations` / `max_depth`。
'''
from promptkit.orchestrate.agents import RoleFactory, as_text, structured_answer, sub_agent
from promptkit.orchestrate.api import (
    API_VERSION,
    SUPPORTED_VERSIONS,
    Request,
    Response,
    handle,
    plan,
    run,
)
from promptkit.orchestrate.delegate import DelegateTool
from promptkit.orchestrate.dsl import PlanSpec
from promptkit.orchestrate.errors import (
    BudgetExceeded,
    DelegateError,
    DepthExceeded,
    ErrorCode,
    OrchestrationError,
    PlanError,
    TaskError,
)
from promptkit.orchestrate.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    PlanResult,
)
from promptkit.orchestrate.plan_tool import PlanTool
from promptkit.orchestrate.planner import (
    LLMPlanner,
    Planner,
    TaskPlanModel,
    TaskSpec,
    TemplatePlanner,
    coerce_planner,
)
from promptkit.orchestrate.steps import (
    AggregateStep,
    ExecutePlan,
    PlanStep,
    PlanWithTool,
    format_task_results,
    plan_from,
)
from promptkit.orchestrate.task import Plan, Task, TaskResult

__all__ = [
    # 数据
    "Task",
    "TaskResult",
    "Plan",
    "PlanResult",
    # 规划
    "Planner",
    "TemplatePlanner",
    "LLMPlanner",
    "coerce_planner",
    "TaskSpec",
    "TaskPlanModel",
    # 调度
    "Orchestrator",
    "OrchestratorConfig",
    # 子代理
    "sub_agent",
    "RoleFactory",
    "structured_answer",
    "as_text",
    # 步骤
    "PlanStep",
    "PlanWithTool",
    "ExecutePlan",
    "AggregateStep",
    "format_task_results",
    "plan_from",
    # 委派 / 工具
    "DelegateTool",
    "PlanTool",
    # 统一接口（v2）
    "API_VERSION",
    "SUPPORTED_VERSIONS",
    "Request",
    "Response",
    "handle",
    "run",
    "plan",
    "PlanSpec",
    "ErrorCode",
    # 异常
    "OrchestrationError",
    "PlanError",
    "TaskError",
    "BudgetExceeded",
    "DepthExceeded",
    "DelegateError",
]
