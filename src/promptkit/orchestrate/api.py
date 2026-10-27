# orchestrate/api.py
'''统一的对外接口：请求 / 响应 + 错误码 + 入参校验。

面向「一次调用」的形态（和 REST 的资源动作同构），但不绑定传输层 ——
`handle()` 是纯函数，HTTP / CLI / 工作流步骤都能直接调它：

    from promptkit.orchestrate import Request, handle

    response = handle(Request(spec=SPEC, context={"topic": "运筹学"}))
    response.ok, response.code, response.data["results"]

动作（`Request.action`）：

    run    解析 DSL -> 执行 -> 返回每个任务的结果（默认）
    plan   解析 DSL -> 只出计划，不执行

版本（`Request.version`）：

    "2.0"（默认）失败不抛异常，错误码在 `response.code`
    "1.0"        保留旧行为：失败直接抛 `PlanError` / `TaskError`

调用层级（`handle -> _dispatch -> _planned/_executed -> PlanSpec / Orchestrator`）
不额外包中间层；DSL 解析复用 `PlanSpec`，计划解析复用 `plan_from`。
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from promptkit.orchestrate.dsl import PlanSpec
from promptkit.orchestrate.errors import ErrorCode, PlanError
from promptkit.orchestrate.planner import coerce_planner
from promptkit.orchestrate.task import plan_from

__all__ = [
    "API_VERSION",
    "SUPPORTED_VERSIONS",
    "Request",
    "Response",
    "handle",
    "run",
    "plan",
]

#: 当前版本（新逻辑）
API_VERSION = "2.0"
#: 仍被接受的版本；"1.0" = 旧语义（失败抛异常）
SUPPORTED_VERSIONS = ("1.0", API_VERSION)

_ACTIONS = ("run", "plan")

#: 错误码 -> HTTP 风格状态码（要转 RESTful 响应时直接取）
_STATUS = {
    ErrorCode.OK: 200,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNSUPPORTED_VERSION: 400,
    ErrorCode.PLAN_INVALID: 422,
    ErrorCode.PLAN_EMPTY: 422,
    ErrorCode.PLAN_DUPLICATE_TASK: 422,
    ErrorCode.PLAN_UNKNOWN_DEPENDENCY: 422,
    ErrorCode.PLAN_CYCLE: 422,
    ErrorCode.PLAN_TOO_MANY_TASKS: 422,
    ErrorCode.TASK_INVALID: 422,
    ErrorCode.NO_AGENT: 422,
    ErrorCode.ROLE_UNDEFINED: 422,
    ErrorCode.DEPTH_EXCEEDED: 422,
    ErrorCode.DELEGATE_INVALID: 422,
    ErrorCode.TASK_FAILED: 500,
    ErrorCode.TASK_TIMEOUT: 504,
    ErrorCode.BUDGET_EXCEEDED: 429,
    ErrorCode.INTERNAL_ERROR: 500,
}

#: 异常类型 -> 错误码（异常自身带 code 时优先用它的）
_CODE_BY_TYPE = {
    KeyError: ErrorCode.ROLE_UNDEFINED,
    TimeoutError: ErrorCode.TASK_TIMEOUT,
    TypeError: ErrorCode.INVALID_REQUEST,
    ValueError: ErrorCode.INVALID_REQUEST,
}


@dataclass(frozen=True)
class Request:
    '''一次编排请求。所有字段可选，但 `spec` 必须给。'''

    #: "run"（默认）或 "plan"
    action: str = "run"
    #: DSL dict / `PlanSpec` / `Plan` / Task 列表
    spec: Any = None
    #: 渲染任务描述用的上下文（`{{key}}`）
    context: Mapping[str, Any] = field(default_factory=dict)
    version: str = API_VERSION
    #: 执行时的默认 agent（DSL 里的 agents 优先）
    agent: Any = None
    #: `plan` 动作用来拆任务的规划器（Planner / Agent）
    planner: Any = None
    #: 覆盖 DSL 里的 policy（`OrchestratorConfig`）
    config: Any = None
    #: 覆盖 DSL 里的 agents（`Callable[[Task], Agent]`）
    factory: Any = None

    def validated(self) -> "Request":
        '''入参校验；不合法抛 `PlanError`（带对应错误码）。'''
        if self.version not in SUPPORTED_VERSIONS:
            raise PlanError(
                f"不支持的 version={self.version!r}；"
                f"可选：{', '.join(SUPPORTED_VERSIONS)}",
                code=ErrorCode.UNSUPPORTED_VERSION,
            )
        action = str(self.action or "").strip().lower()
        if action not in _ACTIONS:
            raise PlanError(
                f"不支持的 action={self.action!r}；可选：{', '.join(_ACTIONS)}",
                code=ErrorCode.INVALID_REQUEST,
            )
        if self.spec is None:
            raise PlanError("请求缺少 spec", code=ErrorCode.INVALID_REQUEST)
        if not isinstance(self.context, Mapping):
            raise PlanError(
                f"context 必须是 dict，得到 {type(self.context).__name__}",
                code=ErrorCode.INVALID_REQUEST,
            )
        if action == "plan":
            has_tasks = (
                isinstance(self.spec, PlanSpec)
                and bool(self.spec.tasks)
            ) or (
                isinstance(self.spec, dict)
                and bool(self.spec.get("tasks"))
            )
            if not has_tasks:
                if self.planner is None:
                    raise PlanError(
                        "action='plan' 需要 planner=（只有 task 描述、没写 tasks 时）",
                        code=ErrorCode.INVALID_REQUEST,
                    )
                coerce_planner(self.planner)      # 类型不合法就在这里报错
        if self.config is not None and not hasattr(self.config, "max_tasks"):
            raise PlanError(
                f"config 必须是 OrchestratorConfig，得到 {type(self.config).__name__}",
                code=ErrorCode.INVALID_REQUEST,
            )
        if self.factory is not None and not callable(self.factory):
            raise PlanError(
                f"factory 必须可调用，得到 {type(self.factory).__name__}",
                code=ErrorCode.INVALID_REQUEST,
            )
        return self


@dataclass(frozen=True)
class Response:
    '''统一的响应结构：`ok` / `code` / `message` / `data`。'''

    ok: bool
    code: ErrorCode = ErrorCode.OK
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    version: str = API_VERSION

    @property
    def status(self) -> int:
        '''HTTP 风格状态码（成功 200）。'''
        return 200 if self.ok else _STATUS.get(self.code, 500)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code.value,
            "message": self.message,
            "data": dict(self.data),
            "version": self.version,
        }

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"<Response {self.status} {self.code.value}>"


# ============================ 入口 ============================
def handle(request: Any) -> Response:
    '''执行一个 `Request`（也接受直接传 DSL，自动包成 `Request`）。

    v2 不抛业务异常 —— 失败都在 `Response` 里；
    v1 保持旧行为：先把原异常抛出去（所以两种版本成功时返回类型一致）。
    '''
    req = request if isinstance(request, Request) else Request(spec=request)
    try:
        req.validated()
        return _dispatch(req)
    except Exception as exc:            # noqa: BLE001 - v2 统一转成响应
        if req.version == "1.0":
            raise
        return _failure(exc, req.version)


def run(
    spec: Any,
    *,
    context: Optional[Mapping[str, Any]] = None,
    agent: Any = None,
    config: Any = None,
    factory: Any = None,
    version: str = API_VERSION,
) -> Response:
    '''`handle(Request(action="run", ...))` 的简写。'''
    return handle(
        Request(
            action="run",
            spec=spec,
            context=context or {},
            version=version,
            agent=agent,
            config=config,
            factory=factory,
        )
    )


def plan(
    spec: Any,
    *,
    planner: Any,
    context: Optional[Mapping[str, Any]] = None,
    version: str = API_VERSION,
) -> Response:
    '''`handle(Request(action="plan", ...))` 的简写（只出计划，不执行）。'''
    return handle(
        Request(
            action="plan",
            spec=spec,
            context=context or {},
            version=version,
            planner=planner,
        )
    )


# ============================ 内部 ============================
def _dispatch(req: Request) -> Response:
    '''两个动作共用同一套解析：spec -> PlanSpec。'''
    spec = req.spec if isinstance(req.spec, PlanSpec) else PlanSpec.from_dsl(req.spec)
    if req.action == "plan":
        return _planned(req, spec)
    # run action：必须有 tasks，否则抛 PLAN_EMPTY（由 handle() 转成 Response）
    plan_obj = spec.to_plan() if spec.tasks else plan_from(spec)
    return _executed(req, spec)


def _planned(req: Request, spec: PlanSpec) -> Response:
    '''`plan` 动作：DSL 写了 tasks 就直接回；只给了 task 才让规划器拆。'''
    planner = coerce_planner(req.planner, allow_none=True)
    if spec.tasks:
        plan_obj = spec.to_plan()
    elif planner is not None:
        plan_obj = plan_from(planner.plan(spec.task, dict(req.context)))
    else:
        # 没写 tasks、也没 planner —— validated() 已经拦住了，这里是兜底
        raise PlanError("计划里没有任何任务", code=ErrorCode.PLAN_EMPTY)
    return Response(
        ok=True,
        data={
            "task": plan_obj.task,
            "tasks": [item.to_spec() for item in plan_obj],
            "waves": [[item.name for item in layer] for layer in plan_obj.waves()],
        },
        version=req.version,
    )


def _executed(req: Request, spec: Any) -> Response:
    from promptkit.orchestrate.dsl import PlanSpec
    from promptkit.orchestrate.orchestrator import Orchestrator

    config = req.config or (spec.to_config() if isinstance(spec, PlanSpec) else None)
    factory = req.factory or (
        spec.to_factory() if isinstance(spec, PlanSpec) else None
    )
    orchestrator = Orchestrator(config, agent=req.agent, agent_factory=factory)
    result = orchestrator.run(spec.to_plan(), context=dict(req.context))
    if result.ok:
        return Response(ok=True, data=_payload(result), version=req.version)
    # 有任务没成功（失败或跳过）：对外仍是「未成功」，但结果原样带回
    return Response(
        ok=False,
        code=ErrorCode.TASK_FAILED,
        message=result.summary(),
        data=_payload(result),
        version=req.version,
    )


def _payload(result: Any) -> Dict[str, Any]:
    '''`PlanResult` -> 可序列化的 data。'''
    return {
        "task": getattr(result, "task", ""),
        "ok": result.ok,
        "elapsed": round(result.elapsed, 3),
        "tokens": result.tokens,
        "results": [item.to_dict() for item in result.results],
    }


def _failure(exc: Exception, version: str) -> Response:
    '''异常 -> 统一响应。错误码优先用异常自带的。'''
    code = getattr(exc, "code", None)
    if not isinstance(code, ErrorCode):
        code = _CODE_BY_TYPE.get(type(exc), ErrorCode.INTERNAL_ERROR)

    result = getattr(exc, "result", None)           # TaskError 带着部分结果
    data = _payload(result) if result is not None else {}
    failed_task = getattr(exc, "task", "")
    if failed_task:
        data["failed_task"] = failed_task
    return Response(
        ok=False,
        code=code,
        message=str(exc) or type(exc).__name__,
        data=data,
        version=version,
    )
