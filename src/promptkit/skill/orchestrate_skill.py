# skill/orchestrate_skill.py
'''把 `promptkit.orchestrate` 封装成可注册、可直接调用的 Skill 集合。

**一个导入点搞定全部** —— 不需要再手动 `import Orchestrator / PlanSpec / Request /
ErrorCode`，也不需要自己组装 `SkillRegistry`：

    from promptkit.skill import OrchestrateSkillSet, orchestrate_run, register_orchestrate_skills

    # 方式 1：直接调（无模型的离线场景也能跑 plan/validate）
    result = orchestrate_run(
        spec={"tasks": [{"name": "a", "description": "x"}]},
        context={"k": "v"},
    )

    # 方式 2：注册进 SkillRegistry，交给 Agent 或工作流用
    registry = register_orchestrate_skills()
    # 或者：registry = SkillRegistry(); OrchestrateSkillSet().register_to(registry)

三个 Skill 对应 v2 API 的三个动作：

+-----------------------+--------+----------------------------------------+
| Skill 名字            | 动作   | 不需要模型也能跑                        |
+-----------------------+--------+----------------------------------------+
| `orchestrate_run`     | 执行   | 否（ctx 需要 agent）                   |
| `orchestrate_plan`    | 只规划 | 是（spec 里写了 tasks 就直接返回）     |
| `orchestrate_validate`| 校验   | 是（纯结构检查）                       |
+-----------------------+--------+----------------------------------------+

DSL 规范（spec 参数）见 `promptkit.orchestrate.dsl.PlanSpec`；错误码见
`promptkit.orchestrate.errors.ErrorCode`。
'''
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from promptkit.orchestrate import (
    ErrorCode,
    PlanSpec,
    Request,
    Response,
    coerce_planner,
    handle,
    plan_from,
)
from promptkit.skill.context import SkillContext
from promptkit.skill.registry import SkillRegistry
from promptkit.skill.spec import Skill, skill

__all__ = [
    "orchestrate_run",
    "orchestrate_plan",
    "orchestrate_validate",
    "OrchestrateSkillSet",
    "register_orchestrate_skills",
]


# -------------------- 参数 Schema（给模型看） --------------------
# spec 参数是一个嵌套 DSL dict，手写一份 schema 比从函数签名自动推更清楚。
_SPEC_SCHEMA = {
    "type": "object",
    "description": (
        "编排 DSL：一个 dict，描述要拆什么任务、用哪些角色、有什么调度护栏。"
        "顶层键：task / agents(或roles) / default_role / policy / tasks。"
        "tasks 是数组，每个元素有 name / description / role / depends_on / output_key"
        " / timeout / max_retries / on_failure 等字段。"
    ),
    "additionalProperties": False,
    "properties": {
        "task": {"type": "string", "description": "顶层目标描述"},
        "agents": {
            "type": "object",
            "description": "角色表：{角色名: 系统提示词}",
            "additionalProperties": {"type": "string"},
        },
        "roles": {
            "type": "object",
            "description": "agents 的别名",
            "additionalProperties": {"type": "string"},
        },
        "default_role": {"type": "string", "description": "未指定 role 的任务默认角色"},
        "policy": {
            "type": "object",
            "description": "调度护栏：max_concurrency / max_tasks / per_task_timeout 等",
            "additionalProperties": True,
        },
        "tasks": {
            "type": "array",
            "description": "任务列表；每一项是一个 dict",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "role": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "output_key": {"type": "string"},
                    "timeout": {"type": "number"},
                    "max_retries": {"type": "integer"},
                    "on_failure": {
                        "type": "string",
                        "enum": ["retry", "skip", "fallback", "abort"],
                    },
                },
                "additionalProperties": True,
            },
        },
    },
}

_CONTEXT_SCHEMA = {
    "type": "object",
    "description": "任务描述里 {{变量}} 模板渲染用的键值对",
    "additionalProperties": True,
}


# ==================== 1. orchestrate_run ====================
def _orchestrate_run_impl(
    spec: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    ctx: Optional[SkillContext] = None,
) -> Dict[str, Any]:
    '''内部实现：DSL dict → 执行 → 返回可序列化 dict。'''
    if ctx is None:
        ctx = SkillContext()

    # 执行型 skill 必须有 agent（每个任务要真跑）
    try:
        agent = ctx.require_agent()
    except Exception as exc:          # noqa: BLE001 - 转成明确的 Response
        resp = Response(
            ok=False,
            code=ErrorCode.NO_AGENT,
            message=f"orchestrate_run 需要可用的 agent：{exc}",
            version="2.0",
        )
        return resp.to_dict()

    resp = handle(
        Request(
            action="run",
            spec=spec,
            context=context or {},
            agent=agent,
        )
    )
    return resp.to_dict()


orchestrate_run: Skill = skill(
    name="orchestrate_run",
    description=(
        "执行一份编排 DSL：按 tasks 列表并行/串行调用 agent，"
        "收集每个任务的输出，返回整体结果、耗时、token 用量等。"
        "ctx 必须带有 agent；DSL 必须包含 tasks 列表。"
    ),
    tags=["orchestrate", "agent", "execution"],
    version="2.0",
    parameters={
        "type": "object",
        "properties": {
            "spec": _SPEC_SCHEMA,
            "context": _CONTEXT_SCHEMA,
        },
        "required": ["spec"],
    },
    examples=[
        {
            "summary": "两任务串行执行",
            "arguments": {
                "spec": {
                    "agents": {"writer": "资深写手"},
                    "tasks": [
                        {"name": "research", "description": "调研 {{topic}}",
                         "role": "writer", "output_key": "research"},
                        {"name": "write", "description": "根据 {{research}} 写报告",
                         "role": "writer", "depends_on": ["research"]},
                    ],
                },
                "context": {"topic": "AI Agent"},
            },
        },
    ],
)(_orchestrate_run_impl)


# ==================== 2. orchestrate_plan ====================
def _orchestrate_plan_impl(
    spec: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    ctx: Optional[SkillContext] = None,
) -> Dict[str, Any]:
    '''内部实现：只出计划，不执行。'''
    if ctx is None:
        ctx = SkillContext()

    # 两种情况不需要 agent：
    #   (a) DSL 里已经写了 tasks → 直接解析返回
    #   (b) 只有 task 描述但外部传了 planner → planner 自己拆
    # 只有「既没 tasks 也没 planner 也没 agent」才会挂
    spec_obj = PlanSpec.from_dsl(spec)

    if spec_obj.tasks:
        # 情况 (a)：DSL 自带任务，不需要 agent
        resp = handle(Request(action="plan", spec=spec_obj, context=context or {}))
        return resp.to_dict()

    # 情况 (b)：只有 task 描述，需要 planner
    # 显式注入的 planner 优先，否则把 ctx.agent 包成 LLMPlanner（统一走 coerce_planner）
    planner: Any = coerce_planner(
        ctx.tools.get("planner") or ctx.agent, allow_none=True
    )

    if planner is None:
        resp = Response(
            ok=False,
            code=ErrorCode.INVALID_REQUEST,
            message=(
                "orchestrate_plan 在 DSL 没有 tasks 时需要一个 planner："
                "要么在 DSL 里写 tasks 字段，要么给 ctx 注入 agent"
            ),
            version="2.0",
        )
        return resp.to_dict()

    resp = handle(
        Request(
            action="plan",
            spec=spec_obj,
            context=context or {},
            planner=planner,
        )
    )
    return resp.to_dict()


orchestrate_plan: Skill = skill(
    name="orchestrate_plan",
    description=(
        "根据编排 DSL 只做任务拆解和波次排序，不真正执行。"
        "DSL 里写了 tasks 就直接返回计划；只有顶层 task 描述时用 planner（自动用 ctx.agent 包一层）。"
        "不需要模型也能跑（只要 DSL 里有 tasks）。"
    ),
    tags=["orchestrate", "planning"],
    version="2.0",
    parameters={
        "type": "object",
        "properties": {
            "spec": _SPEC_SCHEMA,
            "context": _CONTEXT_SCHEMA,
        },
        "required": ["spec"],
    },
    examples=[
        {
            "summary": "DSL 自带任务 → 直接返回",
            "arguments": {
                "spec": {
                    "tasks": [
                        {"name": "a", "description": "一"},
                        {"name": "b", "description": "二", "depends_on": ["a"]},
                    ],
                },
            },
        },
    ],
)(_orchestrate_plan_impl)


# ==================== 3. orchestrate_validate ====================
def _orchestrate_validate_impl(
    spec: Dict[str, Any],
    ctx: Optional[SkillContext] = None,
) -> Dict[str, Any]:
    '''内部实现：纯结构校验，永远不需要 agent。'''
    if ctx is None:
        ctx = SkillContext()

    # 三重校验：DSL 顶层键 → 任务字段 → 依赖图（重名/环/未知依赖）
    errors: list = []
    try:
        plan_spec = PlanSpec.from_dsl(spec)
    except Exception as exc:          # noqa: BLE001
        code = getattr(exc, "code", ErrorCode.INVALID_REQUEST)
        errors.append({"code": code.value, "message": str(exc), "stage": "dsl"})
        return {
            "ok": False,
            "errors": errors,
            "task_count": 0,
            "waves": [],
        }

    try:
        plan = plan_spec.to_plan()
    except Exception as exc:          # noqa: BLE001
        code = getattr(exc, "code", ErrorCode.PLAN_INVALID)
        errors.append({"code": code.value, "message": str(exc), "stage": "plan"})
        return {
            "ok": False,
            "errors": errors,
            "task_count": len(plan_spec.tasks),
            "waves": [],
        }

    # 额外跑一遍 plan_from（最终校验层），确保没有遗漏
    try:
        plan_from(spec)
    except Exception as exc:          # noqa: BLE001
        code = getattr(exc, "code", ErrorCode.PLAN_INVALID)
        errors.append({"code": code.value, "message": str(exc), "stage": "final"})
        return {
            "ok": False,
            "errors": errors,
            "task_count": len(plan_spec.tasks),
            "waves": [[t.name for t in layer] for layer in plan.waves()],
        }

    return {
        "ok": True,
        "errors": [],
        "task_count": len(plan.tasks),
        "waves": [[t.name for t in layer] for layer in plan.waves()],
        "policy": dict(plan_spec.policy),
        "roles": list(plan_spec.roles),
    }


orchestrate_validate: Skill = skill(
    name="orchestrate_validate",
    description=(
        "纯结构校验：检查编排 DSL 有没有未知键、任务重名、依赖不存在、"
        "循环依赖等问题。永远不需要 agent，离线也能跑。"
        "返回 ok=True + 波次列表，或 ok=False + 具体错误列表。"
    ),
    tags=["orchestrate", "validate", "dsl"],
    version="2.0",
    parameters={
        "type": "object",
        "properties": {
            "spec": _SPEC_SCHEMA,
        },
        "required": ["spec"],
    },
    examples=[
        {
            "summary": "合法 DSL → ok=True",
            "arguments": {
                "spec": {
                    "tasks": [
                        {"name": "a", "description": "一"},
                        {"name": "b", "description": "二", "depends_on": ["a"]},
                    ],
                },
            },
        },
        {
            "summary": "循环依赖 → ok=False + 错误列表",
            "arguments": {
                "spec": {
                    "tasks": [
                        {"name": "a", "depends_on": ["b"]},
                        {"name": "b", "depends_on": ["a"]},
                    ],
                },
            },
        },
    ],
)(_orchestrate_validate_impl)


# ==================== 便捷 bundle ====================
class OrchestrateSkillSet:
    '''把三个 orchestrate Skill 打包成一个可实例化组件。

    用法：

        # 直接调（无注册表也能用）
        bundle = OrchestrateSkillSet()
        bundle.run(spec={...}, context={...})
        bundle.plan(spec={...})
        bundle.validate(spec={...})

        # 注册进注册表（一行代码搞定）
        registry = SkillRegistry()
        bundle.register_to(registry)

        # 或者用模块级 helper
        registry = register_orchestrate_skills()
    '''

    def __init__(self, ctx: Optional[SkillContext] = None):
        self.ctx = ctx or SkillContext()
        # 三个 Skill 对象本身（给需要 Skill 实例的代码用）。
        # 命名以 _skill_ 开头，避免遮蔽下面的 run/plan/validate 实例方法
        # （实例方法会自动把 self.ctx 传进去）。
        self._skill_run = orchestrate_run
        self._skill_plan = orchestrate_plan
        self._skill_validate = orchestrate_validate

    # ---------------- 便捷调用（自动注入 self.ctx） ----------------
    def run(self, spec: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        '''等价于 `orchestrate_run(spec, context=..., ctx=self.ctx)`。'''
        return _orchestrate_run_impl(dict(spec), kwargs.get("context"), self.ctx)

    def plan(self, spec: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return _orchestrate_plan_impl(dict(spec), kwargs.get("context"), self.ctx)

    def validate(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        return _orchestrate_validate_impl(dict(spec), self.ctx)

    # ---------------- 注册 ----------------
    def register_to(
        self,
        registry: SkillRegistry,
        *,
        replace: bool = False,
    ) -> SkillRegistry:
        '''把三个 Skill 全部注册进 registry（已存在则报 SkillAlreadyExists）。'''
        registry.register(orchestrate_run, replace=replace)
        registry.register(orchestrate_plan, replace=replace)
        registry.register(orchestrate_validate, replace=replace)
        return registry

    def __repr__(self) -> str:
        return (
            f"<OrchestrateSkillSet skills=run,plan,validate "
            f"ctx_agent={self.ctx.agent is not None}>"
        )


def register_orchestrate_skills(
    registry: Optional[SkillRegistry] = None,
    *,
    replace: bool = False,
) -> SkillRegistry:
    '''一行代码把三个 orchestrate Skill 注册进 registry。

        registry = register_orchestrate_skills()
        # 或者：
        register_orchestrate_skills(existing_registry)
    '''
    target = registry if registry is not None else SkillRegistry()
    OrchestrateSkillSet().register_to(target, replace=replace)
    return target
