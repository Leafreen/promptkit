# orchestrate/errors.py
'''编排层的异常 + 统一错误码。

每个异常自带 `code`（`ErrorCode`），API 层直接把它放进响应的 `code` 字段，
调用方不必解析错误文案。文案可能改，错误码不会。
'''
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    '''对外暴露的错误码（字符串枚举，可直接序列化）。'''

    OK = "OK"
    # 请求本身不合法
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    # 计划不合法
    PLAN_INVALID = "PLAN_INVALID"
    PLAN_EMPTY = "PLAN_EMPTY"
    PLAN_DUPLICATE_TASK = "PLAN_DUPLICATE_TASK"
    PLAN_UNKNOWN_DEPENDENCY = "PLAN_UNKNOWN_DEPENDENCY"
    PLAN_CYCLE = "PLAN_CYCLE"
    PLAN_TOO_MANY_TASKS = "PLAN_TOO_MANY_TASKS"
    # 任务级
    TASK_INVALID = "TASK_INVALID"
    TASK_FAILED = "TASK_FAILED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    # 资源 / 协同
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    NO_AGENT = "NO_AGENT"
    ROLE_UNDEFINED = "ROLE_UNDEFINED"
    # 委派
    DEPTH_EXCEEDED = "DEPTH_EXCEEDED"
    DELEGATE_INVALID = "DELEGATE_INVALID"
    # 兜底
    INTERNAL_ERROR = "INTERNAL_ERROR"


class OrchestrationError(RuntimeError):
    '''编排相关的错误基类；`code` 是稳定的机器可读标识。'''

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(self, message: str, *, code: Optional[ErrorCode] = None):
        super().__init__(message)
        if code is not None:
            self.code = code

    @property
    def detail(self) -> str:
        '''`PLAN_CYCLE: 文案` 形式，日志里一眼定位。'''
        return f"{self.code.value}: {self}"

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "message": str(self)}


class PlanError(OrchestrationError, ValueError):
    '''计划本身不合法：重名、依赖不存在、有环、任务太多。

    同时是 `ValueError`：这些错误几乎都源于「参数写错了」，
    历史上 `PlanStep` 缺 planner 也是抛 `ValueError`，保持兼容。
    '''

    code = ErrorCode.PLAN_INVALID


class TaskError(OrchestrationError):
    '''某个任务失败，而且它的 `on_failure` 要求终止整个计划。

    `result` 带着「到目前为止」的结果（后面的任务标成 skipped），
    这样中止之后你还能看发生了什么。
    '''

    code = ErrorCode.TASK_FAILED

    def __init__(
        self,
        message: str,
        *,
        task: str = "",
        result: Any = None,
        code: Optional[ErrorCode] = None,
    ):
        super().__init__(message, code=code)
        self.task = task
        self.result = result


class BudgetExceeded(OrchestrationError):
    '''token 预算用完了。'''

    code = ErrorCode.BUDGET_EXCEEDED


class DepthExceeded(OrchestrationError):
    '''委派层数超过 `max_depth`。'''

    code = ErrorCode.DEPTH_EXCEEDED


class DelegateError(OrchestrationError):
    '''委派本身不合法：未知的 agent、委派次数超限。'''

    code = ErrorCode.DELEGATE_INVALID
