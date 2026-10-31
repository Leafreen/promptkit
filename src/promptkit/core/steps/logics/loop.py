# core/steps/loop.py
'''循环相关：ForEach（遍历）、For（C 风格计数循环）、DoWhile、Break / Continue。'''
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    cast,
)

from promptkit.core.steps.logics.conditions import ConditionLike, as_condition
if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.errors import BreakSignal, ContinueSignal, LoopLimitExceeded
from promptkit.core.settings import settings
from promptkit.core.steps.logics.expr_sys.expr import Expr, VarExpr, as_expr
from promptkit.core.steps.interface import Step, StepLike, coerce_steps, execute_step


# ====================== Break / Continue ======================
class BreakStep(Step):
    '''跳出当前循环。'''

    name = "break"

    def execute(self, context: Context) -> None:
        context.trace.append(self.name)
        raise BreakSignal()

    def describe(self) -> str:
        return "Break"


class ContinueStep(Step):
    '''跳过本轮剩下的步骤，进入下一轮。'''

    name = "continue"

    def execute(self, context: Context) -> None:
        context.trace.append(self.name)
        raise ContinueSignal()

    def describe(self) -> str:
        return "Continue"


def Break() -> BreakStep:  # noqa: N802
    '''Break()：跳出当前循环。'''
    return BreakStep()


def Continue() -> ContinueStep:  # noqa: N802
    '''Continue()：跳过本轮剩下的步骤。'''
    return ContinueStep()


def run_loop_body(steps: Tuple[Step, ...], context: Context) -> Optional[str]:
    '''执行一轮循环体。

    返回 "break" / "continue" / None。信号会被嵌套的步骤（比如 If）
    一路透传上来，所以循环体里任何深度的 Break() 都能生效。
    '''
    for step in steps:
        try:
            execute_step(step, context)
        except ContinueSignal:
            return "continue"
        except BreakSignal:
            return "break"
    return None


def _steps(value: object) -> Tuple[Step, ...]:
    '''允许传 None / 单个 Step / Step 列表。'''
    if value is None:
        return ()
    return tuple(coerce_steps(cast(StepLike, value)))


def _check_limit(max_iterations: object, name: str) -> None:
    '''`None` 表示「用全局设置」，只有显式传值时才在这里检查。'''
    if max_iterations is None:
        return
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError(
            f"{name} 的 max_iterations 必须是正整数，得到 {max_iterations!r}"
        )


def _limit(explicit: Optional[int], default: int) -> int:
    '''解析成实际要用的上限：显式值优先，否则取全局设置。

    在 execute 里读全局，所以 `settings.max_iterations = 5` 对**已经构造好的**
    循环也生效，`using_settings(...)` 包住 `wf.run()` 也有意义。
    '''
    value = default if explicit is None else explicit
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"max_iterations 必须是正整数，得到 {value!r}")
    return value


# ====================== ForEach：遍历可迭代对象 ======================
@dataclass
class ForEachStep(Step):
    '''遍历一个可迭代对象，每轮把一个元素绑定到变量上。

        ForEach("i", range(1, 16)).Do(Echo(Var("i")))
        ForEach("name", Var("names")).Do(Echo(Var("name")))

    每轮赋值都会走变量类型转换，所以 ForEach("i", ...) 配 Number("i") 也成立。
    '''

    var: str
    iterable: Expr
    body: Tuple[Step, ...] = ()
    #: `None` = 用全局设置 `settings.max_for_iterations`
    max_iterations: Optional[int] = None
    name: str = "foreach"

    def __post_init__(self) -> None:
        # 变量名也接受 Var("i") / NewVar("i") 这类句柄
        if isinstance(self.var, VarExpr):
            self.var = self.var.var
        if not isinstance(self.var, str) or not self.var:
            raise TypeError(
                f"循环变量名必须是非空字符串或 Var(...) 句柄，得到 {self.var!r}"
            )
        self.iterable = as_expr(self.iterable)
        self.body = _steps(self.body)
        _check_limit(self.max_iterations, self.name)

    def Do(self, *steps: StepLike) -> "ForEachStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置循环体。'''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def execute(self, context: Context) -> None:
        limit = _limit(self.max_iterations, settings.max_for_iterations)
        values = self.iterable.eval(context)
        try:
            iterator = iter(cast(Iterable[object], values))
        except TypeError:
            raise TypeError(
                f"{self.name} 需要一个可迭代对象，得到 {type(values).__name__}"
            ) from None

        for index, value in enumerate(iterator, start=1):
            if index > limit:
                raise LoopLimitExceeded(f"{self.name} 循环超过 {limit} 次")
            context[self.var] = value
            context.trace.append(f"{self.name}:{index}")
            if run_loop_body(self.body, context) == "break":
                break
        context.trace.append(f"{self.name}:done")

    def describe(self) -> str:
        return f"ForEach {self.var!r} in {self.iterable!r}"

    def children(self) -> List[Step]:
        return list(self.body)

    def __repr__(self) -> str:
        return f"<ForEachStep var={self.var!r} body={len(self.body)}>"


def ForEach(  # noqa: N802
    var: str, iterable: object, *, max_iterations: Optional[int] = None
) -> ForEachStep:
    '''ForEach("i", range(1, 10)).Do(...)：遍历 range / 列表 / 表达式。

    `max_iterations` 省略时用全局的 `settings.max_for_iterations`。
    '''
    return ForEachStep(
        var=var, iterable=as_expr(iterable), max_iterations=max_iterations
    )


# ====================== For：C 风格的计数循环 ======================
@dataclass
class ForStep(Step):
    '''C 风格的 for：`for (init; condition; step) { body }`。

        For(Set("i", 0), Var("i") < 5, Inc("i")).Do(Echo(Var("i")))

    对应 C：

        for (i = 0; i < 5; i++) { echo(i); }

    三段都可以省略（`For()` 就等于 `for(;;)`，靠 Break 退出）。
    和 C 一样：`Continue()` 会跳过本轮剩下的循环体，但**仍然执行 step**——
    这正是 for 和 while 的关键区别。
    '''

    init: Tuple[Step, ...] = ()
    condition: Callable[[Context], bool] = None  # type: ignore[assignment]
    step: Tuple[Step, ...] = ()
    body: Tuple[Step, ...] = ()
    #: `None` = 用全局设置 `settings.max_for_iterations`
    max_iterations: Optional[int] = None
    name: str = "for"

    def __post_init__(self) -> None:
        self.init = _steps(self.init)
        self.step = _steps(self.step)
        self.body = _steps(self.body)
        self.condition = as_condition(
            True if self.condition is None else self.condition
        )
        _check_limit(self.max_iterations, self.name)

    def Do(self, *steps: StepLike) -> "ForStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置循环体。'''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def execute(self, context: Context) -> None:
        limit = _limit(self.max_iterations, settings.max_for_iterations)
        for step in self.init:
            execute_step(step, context)

        iterations = 0
        while self.condition(context):
            if iterations >= limit:
                raise LoopLimitExceeded(
                    f"{self.name} 循环超过 {limit} 次仍未结束，"
                    "请检查 condition 和 step 有没有让循环变量发生变化"
                )
            iterations += 1
            context.trace.append(f"{self.name}:{iterations}")

            if run_loop_body(self.body, context) == "break":
                break
            # continue 也要执行 step —— C 里 for 的 increment 就是这么走的
            for step in self.step:
                execute_step(step, context)

        context.trace.append(f"{self.name}:done")

    def describe(self) -> str:
        init = "; ".join(step.describe() for step in self.init)
        post = "; ".join(step.describe() for step in self.step)
        return f"For({init}; {self.condition!r}; {post})"

    def children(self) -> List[Step]:
        return list(self.body)

    def __repr__(self) -> str:
        return (
            f"<ForStep init={len(self.init)} step={len(self.step)} "
            f"body={len(self.body)}>"
        )


def For(  # noqa: N802
    init: Optional[StepLike] = None,
    condition: ConditionLike = True,
    step: Optional[StepLike] = None,
    *,
    max_iterations: Optional[int] = None,
) -> ForStep:
    '''For(初始化, 条件, 步进).Do(...) —— C 风格的 for 循环。

    `max_iterations` 省略时用全局的 `settings.max_for_iterations`。
    '''
    return ForStep(
        init=cast(Tuple[Step, ...], init),
        condition=as_condition(condition),
        step=cast(Tuple[Step, ...], step),
        max_iterations=max_iterations,
    )


# ====================== DoWhile：先执行一次再判断 ======================
@dataclass
class DoWhileStep(Step):
    '''C 风格的 do-while：循环体至少执行一次，之后再判断条件。

        DoWhile(Var("n") < 3).Do(Inc("n"))
    '''

    condition: Callable[[Context], bool] = None  # type: ignore[assignment]
    body: Tuple[Step, ...] = ()
    #: `None` = 用全局设置 `settings.max_iterations`
    max_iterations: Optional[int] = None
    name: str = "do"

    def __post_init__(self) -> None:
        self.condition = as_condition(
            True if self.condition is None else self.condition
        )
        self.body = _steps(self.body)
        _check_limit(self.max_iterations, self.name)

    def Do(self, *steps: StepLike) -> "DoWhileStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置循环体。'''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def execute(self, context: Context) -> None:
        limit = _limit(self.max_iterations, settings.max_iterations)
        iterations = 0
        while True:
            if iterations >= limit:
                raise LoopLimitExceeded(f"{self.name} 循环超过 {limit} 次仍未结束")
            iterations += 1
            context.trace.append(f"{self.name}:{iterations}")

            if run_loop_body(self.body, context) == "break":
                break
            if not self.condition(context):
                break

        context.trace.append(f"{self.name}:done")

    def describe(self) -> str:
        return f"DoWhile({self.condition!r})"

    def children(self) -> List[Step]:
        return list(self.body)

    def __repr__(self) -> str:
        return f"<DoWhileStep body={len(self.body)}>"


def DoWhile(  # noqa: N802
    condition: ConditionLike, *, max_iterations: Optional[int] = None
) -> DoWhileStep:
    '''DoWhile(条件).Do(...)：循环体至少执行一次。

    `max_iterations` 省略时用全局的 `settings.max_iterations`。
    '''
    return DoWhileStep(condition=as_condition(condition), max_iterations=max_iterations)
