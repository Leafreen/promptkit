# core/steps/logics/on_error.py
'''`OnError`：把一组步骤的异常收进来，交给 handler / fallback 处理。'''
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Union

from promptkit.core.errors import (
    BreakSignal,
    ContinueSignal,
    GotoSignal,
    ReturnSignal,
)
from promptkit.core.steps.interface import (
    Step,
    StepLike,
    coerce_steps,
    execute_step,
)

if TYPE_CHECKING:  # 只用于类型标注，避免 logics 在运行时依赖 context
    from promptkit.core.context import Context

#: 这些是控制流信号，不是「出错」，OnError 不该吞掉它们
_CONTROL_FLOW = (BreakSignal, ContinueSignal, ReturnSignal, GotoSignal)

#: handler 的签名：`(error, context) -> Any`
ErrorHandler = Callable[["Exception", "Context"], object]


class OnError(Step):
    '''执行一组步骤；出错时写 `output_key`、调 handler、跑 fallback。

        OnError(
            Set("x", Var("a") / Var("b")),
            handler=lambda exc, ctx: 0,
            fallback=Echo("算错了，先跳过"),
        )

    默认**吞掉**异常（工作流继续）；`reraise=True` 时处理完再抛原异常。
    `handler` 的返回值会覆盖 `output_key`（返回 `None` 则保留异常对象）。
    '''

    name = "on_error"

    def __init__(
        self,
        *steps: StepLike,
        handler: Optional[ErrorHandler] = None,
        fallback: Union[StepLike, None] = None,
        output_key: str = "error",
        reraise: bool = False,
        name: str = "on_error",
    ):
        self.steps = coerce_steps(*steps)
        if not self.steps:
            raise ValueError("OnError 至少要包一个步骤")
        if handler is not None and not callable(handler):
            raise TypeError(f"handler 必须可调用，得到 {type(handler).__name__}")
        self.handler = handler
        self.fallback = coerce_steps(fallback) if fallback is not None else []
        self.output_key = output_key
        self.reraise = bool(reraise)
        self.name = name

    def execute(self, context: "Context") -> None:
        try:
            for step in self.steps:
                execute_step(step, context)
        except _CONTROL_FLOW:
            raise
        except Exception as exc:
            context[self.output_key] = exc
            context.trace.append(
                {
                    "step": self.name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if self.handler is not None:
                result = self.handler(exc, context)
                if result is not None:
                    context[self.output_key] = result
            for step in self.fallback:
                execute_step(step, context)
            if self.reraise:
                raise

    def children(self) -> List[Step]:
        return [*self.steps, *self.fallback]

    def describe(self) -> str:
        suffix = f" -> {self.output_key}"
        if self.fallback:
            suffix += " + fallback"
        return f"OnError({len(self.steps)} 步{suffix})"


__all__ = ["OnError", "ErrorHandler"]
