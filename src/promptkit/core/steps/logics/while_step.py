# core/steps/while_step.py
'''While 循环步骤。'''
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Tuple

from promptkit.core.steps.logics.conditions import ConditionLike, as_condition
if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.errors import LoopLimitExceeded
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step, coerce_steps
from promptkit.core.steps.logics.loop import _limit, run_loop_body


@dataclass
class WhileStep(Step):
    '''条件成立时反复执行 Do(...) 里的步骤。

        While(Var("state") != "结束", max_iterations=20).Do(
            Set("n", Var("n") + 1),
        )

    必须有迭代上限兜底：条件一直成立时会抛 LoopLimitExceeded，而不是静默死循环。
    上限省略时用全局的 `settings.max_iterations`（默认 100）。
    '''

    condition: Callable[[Context], bool]
    body: Tuple[Step, ...] = ()
    #: `None` = 用全局设置 `settings.max_iterations`
    max_iterations: Optional[int] = None
    name: str = "while"

    def __post_init__(self) -> None:
        self.condition = as_condition(self.condition)
        if self.max_iterations is not None and (
            not isinstance(self.max_iterations, int) or self.max_iterations < 1
        ):
            raise ValueError(
                f"max_iterations 必须是正整数，得到 {self.max_iterations!r}"
            )

    def Do(self, *steps: Any) -> "WhileStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置循环体。'''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def execute(self, context: Context) -> None:
        limit = _limit(self.max_iterations, settings.max_iterations)
        iterations = 0
        while self.condition(context):
            if iterations >= limit:
                raise LoopLimitExceeded(
                    f"{self.name} 循环超过 {limit} 次仍未结束，"
                    "请检查循环条件里的变量有没有在循环体里被更新"
                )
            iterations += 1
            context.trace.append(f"{self.name}:{iterations}")
            # Break / Continue 由 run_loop_body 统一处理；
            # Continue 之后重新判断条件，正好就是 while 的语义
            if run_loop_body(self.body, context) == "break":
                break
        context.trace.append(f"{self.name}:done")

    def describe(self) -> str:
        return f"While {self.condition!r}"

    def children(self) -> List[Step]:
        return list(self.body)

    def __repr__(self) -> str:
        return f"<WhileStep max_iterations={self.max_iterations} body={len(self.body)}>"


def While(  # noqa: N802
    condition: ConditionLike, *, max_iterations: Optional[int] = None
) -> WhileStep:
    '''While(条件).Do(步骤...)。

    `max_iterations` 省略时用全局的 `settings.max_iterations`（默认 100）。
    '''
    return WhileStep(condition=as_condition(condition), max_iterations=max_iterations)
