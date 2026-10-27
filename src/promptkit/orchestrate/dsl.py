# orchestrate/dsl.py
'''DSL 声明式配置：一份 dict 同时描述**任务、调度规则、协同策略**。

这是 planner 与多 agent 协同的**标准解析接入点**，不需要任何适配层：

    from promptkit.orchestrate import PlanSpec

    SPEC = {
        "task": "调研并写一份报告",
        "agents": {
            "researcher": "你是市场调研专家，只给事实和数据来源。",
            "writer": "你是资深报告撰写者。",
        },
        "policy": {"max_concurrency": 2, "per_task_timeout": 60},
        "tasks": [
            {"name": "research", "description": "调研 {{topic}}",
             "role": "researcher", "output_key": "research"},
            {"name": "write", "description": "根据 {{research}} 写报告",
             "role": "writer", "depends_on": ["research"]},
        ],
    }

    # 1) 直接进 DSL 步骤树（最简单）
    Workflow(ExecutePlan(plan=SPEC))

    # 2) 显式解析成规格，自己组装
    spec = PlanSpec.from_dsl(SPEC)
    spec.to_orchestrator(agent=...).run(spec.to_plan(), context=...)

    # 3) 只取一部分（比如复用 policy 给别的计划）
    config = spec.to_config()
    plan = spec.to_plan()

`policy` 的键就是 `OrchestratorConfig` 的字段名，写错会立刻报错（不是静默忽略）。
'''
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

from promptkit.orchestrate.errors import ErrorCode, PlanError
from promptkit.orchestrate.task import Plan, Task, plan_from

__all__ = ["PlanSpec"]

#: dict 里描述任务的键；也接受 `roles` 代替 `agents`
_TASK_KEYS = ("tasks", "task", "policy", "agents", "roles", "default_role")
#: 任务项里允许出现的字段 = `Task` 的字段
_TASK_FIELDS = tuple(item.name for item in fields(Task))


@dataclass(frozen=True)
class PlanSpec:
    '''一个计划的完整声明：任务 + 调度护栏 + 角色表。

    字段都可以留空；`from_dsl()` 是主要构造方式。
    '''

    tasks: Tuple[Any, ...] = ()
    task: str = ""
    policy: Mapping[str, Any] = field(default_factory=dict)
    roles: Mapping[str, str] = field(default_factory=dict)
    default_role: str = ""

    # ---------------- 解析 ----------------
    @classmethod
    def from_dsl(cls, value: Any) -> "PlanSpec":
        '''把 dict / `PlanSpec` / `Plan` / Task 列表统一成规格。'''
        if isinstance(value, PlanSpec):
            return value
        if isinstance(value, Plan):
            return cls(tasks=tuple(value.tasks), task=value.task)
        if isinstance(value, (list, tuple)):
            return cls(tasks=tuple(value))
        if not isinstance(value, Mapping):
            raise PlanError(
                f"DSL 配置需要 dict，得到 {type(value).__name__}",
                code=ErrorCode.INVALID_REQUEST,
            )

        unknown = [key for key in value if key not in _TASK_KEYS]
        if unknown:
            raise PlanError(
                f"DSL 配置里有不认识的键：{', '.join(map(str, unknown))}；"
                f"可用：{', '.join(_TASK_KEYS)}",
                code=ErrorCode.INVALID_REQUEST,
            )
        tasks = value.get("tasks")
        if tasks is None:
            # 只给了需求（task）没给任务：留给规划器去拆（action="plan" 用）
            if not str(value.get("task") or "").strip():
                raise PlanError(
                    "DSL 配置至少要给 tasks 或 task", code=ErrorCode.PLAN_EMPTY
                )
            tasks = ()
        roles = value.get("agents") or value.get("roles") or {}
        return cls(
            tasks=tuple(tasks),
            task=str(value.get("task") or ""),
            policy=dict(value.get("policy") or {}),
            roles={str(key): str(item) for key, item in dict(roles).items()},
            default_role=str(value.get("default_role") or ""),
        )

    # ---------------- 映射 ----------------
    def to_plan(self) -> Plan:
        '''任务声明 -> `Plan`（顺带完成重名 / 依赖 / 环的校验）。'''
        for item in self.tasks:
            if isinstance(item, Mapping):
                unknown = [key for key in item if key not in _TASK_FIELDS]
                if unknown:
                    raise PlanError(
                        f"任务 {item.get('name')!r} 有不认识的字段："
                        f"{', '.join(map(str, unknown))}；"
                        f"可用：{', '.join(_TASK_FIELDS)}",
                        code=ErrorCode.TASK_INVALID,
                    )
        return plan_from({"tasks": list(self.tasks), "task": self.task})

    def to_config(self, base: Any = None) -> Any:
        '''`policy` -> `OrchestratorConfig`；没有 policy 且没给 base 时返回 None。'''
        from promptkit.orchestrate.orchestrator import OrchestratorConfig

        if not self.policy and base is None:
            return None
        config = base or OrchestratorConfig()
        known = {item.name for item in fields(OrchestratorConfig)}
        unknown = [key for key in self.policy if key not in known]
        if unknown:
            raise PlanError(
                f"policy 里有不认识的键：{', '.join(map(str, unknown))}；"
                f"可用：{', '.join(sorted(known))}",
                code=ErrorCode.INVALID_REQUEST,
            )
        for key, value in self.policy.items():
            setattr(config, key, value)
        config.__post_init__()          # 复用既有校验，不重复写规则
        return config

    def to_factory(self, **agent_kwargs: Any) -> Any:
        '''`agents` + `default_role` -> `RoleFactory`；没有角色表时返回 None。'''
        if not self.roles:
            return None
        from promptkit.orchestrate.agents import RoleFactory

        return RoleFactory(
            self.roles, default_system=self.default_role or None, **agent_kwargs
        )

    def to_orchestrator(self, *, agent: Any = None, **kwargs: Any) -> Any:
        '''一步造好编排器（config 与 agent_factory 都来自 DSL）。'''
        from promptkit.orchestrate.orchestrator import Orchestrator

        return Orchestrator(
            self.to_config(), agent=agent, agent_factory=self.to_factory(), **kwargs
        )

    def step(self, **kwargs: Any) -> Any:
        '''包成 `ExecutePlan`，直接放进 `Workflow(...)`。'''
        from promptkit.orchestrate.steps import ExecutePlan

        kwargs.setdefault("plan", self)
        return ExecutePlan(**kwargs)

    def run(self, *, context: Optional[Mapping[str, Any]] = None, agent: Any = None):
        '''直接跑一遍，返回 `PlanResult`。'''
        return self.to_orchestrator(agent=agent).run(self.to_plan(), context=context)

    # ---------------- 序列化 ----------------
    def to_dsl(self) -> Dict[str, Any]:
        '''转回 dict（日志、落盘、测试往返都用得上）。'''
        result: Dict[str, Any] = {
            "tasks": [
                item.to_spec() if isinstance(item, Task) else dict(item)
                for item in self.tasks
            ]
        }
        if self.task:
            result["task"] = self.task
        if self.policy:
            result["policy"] = dict(self.policy)
        if self.roles:
            result["agents"] = dict(self.roles)
        if self.default_role:
            result["default_role"] = self.default_role
        return result

    @property
    def names(self) -> List[str]:
        return [str(item.get("name")) if isinstance(item, Mapping) else item.name
                for item in self.tasks]

    def __repr__(self) -> str:
        return (
            f"<PlanSpec {len(self.tasks)} 个任务"
            f" policy={dict(self.policy) or '{}'}"
            f" roles={list(self.roles) or '[]'}>"
        )
