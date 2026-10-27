# orchestrate/planner.py
'''怎么把「一个需求」变成「一组任务」。

- `TemplatePlanner`：任务拆分写死在代码里（推荐先用这个，可测、成本可预测）
- `LLMPlanner`：让模型输出 JSON 任务列表（灵活，但规划可能错、成本不可控）
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from promptkit.orchestrate.agents import as_text, structured_answer
from promptkit.orchestrate.errors import ErrorCode, PlanError
from promptkit.orchestrate.task import Plan, Task

__all__ = [
    "Planner",
    "TemplatePlanner",
    "LLMPlanner",
    "TaskSpec",
    "TaskPlanModel",
    "coerce_planner",
]


def coerce_planner(
    value: Any,
    *,
    max_tasks: int = 5,
    tools: Optional[Sequence[Any]] = None,
    allow_none: bool = False,
) -> Optional["Planner"]:
    '''把「Planner / 带 plan() 的对象 / Agent / None」统一成规划器。

    全项目**唯一**的规划器归一化入口：`PlanStep`、`orchestrate.api`、
    `PlanTool` 都走它，所以三种写法的行为一致。
    '''
    if value is None:
        if allow_none:
            return None
        raise PlanError(
            "需要一个规划器：Planner 实例、带 plan() 的对象，或 Agent",
            code=ErrorCode.INVALID_REQUEST,
        )
    if isinstance(value, Planner) or callable(getattr(value, "plan", None)):
        return value
    if callable(getattr(value, "ask", None)):
        return LLMPlanner(value, max_tasks=max_tasks, tools=tools)
    raise PlanError(
        f"看不懂的 planner：{type(value).__name__}；"
        "要 Planner 实例、带 plan() 的对象，或 Agent",
        code=ErrorCode.INVALID_REQUEST,
    )


class Planner(ABC):
    '''规划器：只负责产出 `Plan`，不执行。'''

    @abstractmethod
    def plan(self, task: str, context: Mapping[str, Any]) -> Plan:
        '''把 `task`（原始需求）拆成计划。'''
        raise NotImplementedError


class TemplatePlanner(Planner):
    '''固定模板：任务列表是你写好的，规划器只是把它包成 `Plan`。

        TemplatePlanner([
            Task("research", "……", role="researcher", output_key="research"),
            Task("write", "根据 {{research}} 写报告", depends_on=("research",),
                 role="writer", output_key="report"),
        ])
    '''

    def __init__(
        self,
        tasks: Sequence[Any],
        *,
        task: str = "",
        max_tasks: Optional[int] = None,
    ):
        self.tasks = list(tasks)
        self.task = task
        self.max_tasks = max_tasks

    def plan(self, task: str, context: Mapping[str, Any]) -> Plan:
        return Plan.from_template(
            self.tasks, task=task or self.task, max_tasks=self.max_tasks
        )

    def __repr__(self) -> str:
        return f"<TemplatePlanner {len(self.tasks)} 个任务>"


class TaskSpec(BaseModel):
    '''LLM 规划里的一个任务。名字用英文短标识，描述用自然语言。'''

    name: str = Field(description="任务名，英文短标识，如 research / analyze / write")
    description: str = Field(description="交给执行者的完整指令")
    role: str = Field(default="", description="角色名，如 researcher / writer")
    depends_on: List[str] = Field(default_factory=list, description="依赖的任务名")
    input_keys: List[str] = Field(default_factory=list, description="要用到的上下文变量")
    output_key: str = Field(default="", description="结果放哪个 key，空=任务名")
    max_retries: int = Field(default=1, ge=0)
    on_failure: str = Field(default="retry", pattern="^(retry|skip|abort|fallback)$")


class TaskPlanModel(BaseModel):
    '''规划器的输出：一串任务。'''

    tasks: List[TaskSpec] = Field(default_factory=list)
    reasoning: str = Field(default="", description="为什么这么拆（一句话）")


class LLMPlanner(Planner):
    '''让模型规划：输出 JSON 任务列表 -> 校验 -> `Plan`。

    三个护栏：`max_tasks` 截断、依赖必须指向已有任务、`Plan` 构造时查环。
    '''

    def __init__(
        self,
        agent: Any,
        *,
        max_tasks: int = 5,
        tools: Optional[Sequence[Any]] = None,
        max_retries: Optional[int] = None,
    ):
        if agent is None:
            raise ValueError("LLMPlanner 需要一个 agent")
        self.agent = agent
        self.max_tasks = max(1, int(max_tasks))
        self.tools = tools
        self.max_retries = max_retries

    def plan(self, task: str, context: Mapping[str, Any]) -> Plan:
        prompt = self.build_prompt(task, context)
        model: TaskPlanModel = structured_answer(
            self.agent,
            prompt,
            TaskPlanModel,
            tools=self.tools,
            max_retries=self.max_retries,
        )

        specs = list(model.tasks)[: self.max_tasks]
        if not specs:
            raise PlanError("规划器没有给出任何任务")

        tasks: List[Task] = []
        for spec in specs:
            tasks.append(
                Task(
                    name=spec.name,
                    description=spec.description,
                    role=spec.role,
                    depends_on=tuple(spec.depends_on),
                    input_keys=tuple(spec.input_keys),
                    output_key=spec.output_key,
                    max_retries=spec.max_retries,
                    on_failure=spec.on_failure,
                )
            )

        # 截断之后，依赖可能指向已经不存在的任务 —— 去掉它们（而不是让整个计划作废）
        names = {item.name for item in tasks}
        for item in tasks:
            item.depends_on = tuple(
                name for name in item.depends_on if name in names and name != item.name
            )
        return Plan(tuple(tasks), task=task)

    def build_prompt(self, task: str, context: Mapping[str, Any]) -> str:
        materials = "\n".join(
            f"- {key}: {_short(as_text(value))}" for key, value in context.items()
        ) or "（没有额外上下文）"
        return (
            f"请把下面的需求拆成**最多 {self.max_tasks} 个**子任务，供多个子代理分头执行。\n\n"
            f"需求：\n{task}\n\n"
            f"可用上下文：\n{materials}\n\n"
            "规则：\n"
            "- 每个任务名唯一，用英文短标识（如 research / analyze / write）\n"
            "- description 是交给执行者的完整指令，它看不到别的任务的内容\n"
            "- depends_on 只能引用**前面已定义**的任务名；能并行的不要互相依赖\n"
            "- 有 depends_on 的任务会**自动**拿到上游结果，不用在 description 里重复粘贴\n"
            "- role 用一句话的角色名（如 researcher / writer / reviewer）\n"
            "- 单任务不要太大；子任务之间不要重复劳动\n"
            "- 只输出 JSON，形如：\n"
            '{"tasks": [{"name": "research", "description": "……", "role": "researcher", '
            '"depends_on": [], "output_key": "research"}], "reasoning": "一句话"}'
        )

    def __repr__(self) -> str:
        return f"<LLMPlanner max_tasks={self.max_tasks}>"


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
