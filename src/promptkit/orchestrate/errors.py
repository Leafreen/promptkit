# orchestrate/errors.py
'''编排层的异常。都继承 `OrchestrationError`，方便一把捞。'''
from __future__ import annotations

from typing import Any


class OrchestrationError(RuntimeError):
    '''编排相关的错误基类。'''


class PlanError(OrchestrationError):
    '''计划本身不合法：重名、依赖不存在、有环、任务太多。'''


class TaskError(OrchestrationError):
    '''某个任务失败，而且它的 `on_failure` 要求终止整个计划。

    `result` 带着「到目前为止」的结果（后面的任务标成 skipped），
    这样中止之后你还能看发生了什么。
    '''

    def __init__(self, message: str, *, task: str = "", result: Any = None):
        super().__init__(message)
        self.task = task
        self.result = result


class BudgetExceeded(OrchestrationError):
    '''token 预算用完了。'''


class DepthExceeded(OrchestrationError):
    '''委派层数超过 `max_depth`。'''


class DelegateError(OrchestrationError):
    '''委派本身不合法：未知的 agent、委派次数超限。'''
