# core/steps/while_step.py
'''While 循环步骤：条件成立时反复执行循环体，并暴露 7 个 Hook 扩展点。'''
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Tuple, Union

from promptkit.core.steps.logics.conditions import ConditionLike, as_condition
if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.errors import (
    BreakSignal,
    ContinueSignal,
    GotoSignal,
    LoopLimitExceeded,
    ReturnSignal,
)
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step, coerce_steps
from promptkit.core.steps.logics.loop import _limit, run_loop_body
from promptkit.core.steps.logics.loop_hooks import (
    LOOP_HOOKS_KEY,
    HookPoint,
    LoopHookEvent,
    LoopHookHandle,
    LoopHooks,
    as_hook_point,
    loop_hooks,
)

# ====================== 标准化的 Hook 扩展点 ======================
# 预留给 `While(...).Hook(LOOP_START, fn)` / `hooks.register(BEFORE_ITERATION, fn)`。
# 新增扩展点：在 `HookPoint` 里加一个成员、在 `_iteration` / `execute` 里加一次
# `hooks.trigger(...)`，注册 / 注销 / 派发 / 错误隔离都不需要改。
#
#: 进入循环前触发一次。事件里 `iteration=0`。
LOOP_START = HookPoint.LOOP_START
#: 每轮判条件前触发。返回 `False` 结束整个循环（`reason="hook"`）。
BEFORE_CONDITION = HookPoint.BEFORE_CONDITION
#: 条件成立、跑循环体前触发。返回 `False` 跳过本轮（仍触发 `AFTER_ITERATION`）。
BEFORE_ITERATION = HookPoint.BEFORE_ITERATION
#: 本轮结束后触发（含被跳过 / Break / Continue）。事件带 `signal`、`duration_ms`。
AFTER_ITERATION = HookPoint.AFTER_ITERATION
#: 即将超出 `max_iterations` 时触发，随后照常抛 `LoopLimitExceeded`。
LOOP_LIMIT = HookPoint.LOOP_LIMIT
#: 条件或循环体抛异常时触发，随后照常向上抛。
LOOP_ERROR = HookPoint.LOOP_ERROR
#: 循环正常结束 / Break 退出后触发一次。事件带 `reason`、`iteration`（已完成轮数）。
LOOP_END = HookPoint.LOOP_END

#: 全部扩展点，按执行顺序排列。遍历注册 / 自检时用它，避免漏点。
LOOP_HOOK_POINTS: Tuple[HookPoint, ...] = tuple(HookPoint)

#: 控制流信号不是错误，不该触发 LOOP_ERROR
_CONTROL_FLOW = (BreakSignal, ContinueSignal, ReturnSignal, GotoSignal)


@dataclass
class WhileStep(Step):
    '''条件成立时反复执行 `Do(...)` 里的步骤。

        While(Var("state") != "结束", max_iterations=20).Do(
            Set("n", Var("n") + 1),
        )

    必须有迭代上限兜底：条件一直成立时会抛 `LoopLimitExceeded`，而不是静默死循环。
    上限省略时用全局的 `settings.max_iterations`（默认 100）。

    每轮的执行顺序（方括号里是触发的扩展点）：

        [loop_start]
        [before_condition] -> 条件为假则结束（reason="condition"）
        [before_iteration] -> 返回 False 则本轮跳过（signal="skipped"）
             循环体（Break / Continue 也算本轮结束）
        [after_iteration]  -> 带 signal 与 duration_ms
        超出上限时 [loop_limit] 后抛 LoopLimitExceeded
        条件或循环体抛异常时 [loop_error] 后照常抛
        [loop_end]（仅正常结束 / Break 时触发）

    钩子（详见 `loop_hooks.py`）：

        While(cond).Hook(BEFORE_ITERATION, fn).Do(...)   # 步级，只对本循环
        loop_hooks(ctx).register(AFTER_ITERATION, fn)    # 全局，对本 Context 所有循环
    '''

    #: 循环条件，`Callable[[Context], bool]`。
    #: 每轮判一次，为真才跑循环体。可以传 `Var` 表达式等（会被 `as_condition`
    #: 归一化成 callable），也可以传 `True` / `False` 这样的常量。
    condition: Callable[[Context], bool]
    #: 循环体，`Tuple[Step, ...]`，按顺序执行。
    #: 一般不用直接传，用 `.Do(...)` 设置，它会把任意写法 `coerce_steps` 成步骤。
    body: Tuple[Step, ...] = ()
    #: 迭代上限，`int | None`，默认 `None`。
    #: `None` = 用全局设置 `settings.max_iterations`（默认 100）；必须是正整数，
    #: 否则构造时抛 `ValueError`。超出上限会先触发 `LOOP_LIMIT`，再抛
    #: `LoopLimitExceeded`。
    max_iterations: Optional[int] = None
    #: 步级钩子表，`LoopHooks`，默认空表。
    #: 只对本循环生效；也可以传 `dict`（`{扩展点: 函数或函数列表}`）或
    #: `(扩展点, 函数)` 列表，构造时会自动包装成 `LoopHooks`。
    hooks: LoopHooks = field(default_factory=LoopHooks)
    #: 循环名，`str`，默认 `"while"`。
    #: 出现在 `context.trace`（`f"{name}:{轮次}"`、`f"{name}:done"`）和
    #: `LoopHookEvent.name` 里，多个循环时建议改名以便区分。
    name: str = "while"

    def __post_init__(self) -> None:
        self.condition = as_condition(self.condition)
        if self.max_iterations is not None and (
            not isinstance(self.max_iterations, int) or self.max_iterations < 1
        ):
            raise ValueError(
                f"max_iterations 必须是正整数，得到 {self.max_iterations!r}"
            )
        if not isinstance(self.hooks, LoopHooks):
            self.hooks = LoopHooks(self.hooks)

    def Do(self, *steps: Any) -> "WhileStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置循环体，返回**新的** `WhileStep`（dataclass `replace`，不修改原对象）。

        参数：

        *steps（Any）:
            循环体步骤，写法与 `Workflow(...)` 一样：`Step` 实例、可调用对象、
            步骤列表都可以，内部走 `coerce_steps` 归一化。

        返回：
            WhileStep —— 带循环体的新实例，可继续链式 `.Hook(...)`。
        '''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def Hook(  # noqa: N802 - 跟随 DSL 命名
        self, point: Union[str, HookPoint], fn: Callable[[LoopHookEvent], Any]
    ) -> "WhileStep":
        '''注册一个**步级**钩子（只对这个循环生效），返回自身以便链式调用。

        参数：

        point（str | HookPoint）:
            扩展点，如 `BEFORE_ITERATION` / `"after_iteration"`。大小写不敏感；
            未知名字抛 `ValueError`。
        fn（Callable[[LoopHookEvent], Any]）:
            钩子函数，入参是 `LoopHookEvent`。只有 `BEFORE_CONDITION` /
            `BEFORE_ITERATION` 看返回值（返回 `False` 表示否决），其余忽略。
            钩子内部抛异常不会影响循环。

        返回：
            WhileStep —— `self`（原地注册，`.Hook()` 的改动对已持有的引用可见）。

        异常：
            TypeError: `fn` 不可调用。
            ValueError: `point` 不是已知扩展点。
        '''
        self.hooks.register(point, fn)
        return self

    def Unhook(  # noqa: N802
        self, target: Any, point: Optional[Union[str, HookPoint]] = None
    ) -> bool:
        '''注销一个步级钩子，等价于 `self.hooks.unregister(...)`。

        参数：

        target（LoopHookHandle | str | HookPoint | Callable）:
            - `LoopHookHandle`：注销 `.Hook()` / `register()` 返回的句柄
            - 扩展点名：清掉该扩展点上的全部钩子
            - 函数：按函数注销
        point（str | HookPoint | None，默认 None）:
            仅当 `target` 是函数时有意义：指定只从哪个扩展点删；
            不传则删掉该函数注册过的所有扩展点。

        返回：
            bool —— 真的删掉了为 `True`，本来就没有为 `False`。
        '''
        return self.hooks.unregister(target, point)

    # ---------------- 执行 ----------------
    def execute(self, context: Context) -> None:
        '''跑完整个循环，并在此过程按顺序触发各扩展点。

        参数：

        context（Context）:
            工作流的 Context。钩子可以读写它（改动立即对后续步骤生效）；
            每次进入 / 结束一轮会往 `context.trace` 追加 `"{name}:{轮次}"`，
            循环结束时追加 `"{name}:done"`。

        异常：
            LoopLimitExceeded: 轮数达到上限条件仍成立（触发 `LOOP_LIMIT` 之后）。
            循环体或条件里的任何异常都会原样向上抛（触发 `LOOP_ERROR` 之后）；
            `Break` / `Continue` / `Return` / `Goto` 是控制流，不算错误。
        '''
        limit = _limit(self.max_iterations, settings.max_iterations)
        hooks = self._hooks(context)
        iterations = 0
        reason = "condition"
        hooks.trigger(
            LOOP_START, LoopHookEvent(LOOP_START, self, context)
        )
        while True:
            pending = iterations + 1
            if not hooks.trigger(BEFORE_CONDITION, self._event(BEFORE_CONDITION, context, pending)):
                reason = "hook"
                break
            if not self._condition(context, hooks, pending):
                reason = "condition"
                break
            if iterations >= limit:
                error = LoopLimitExceeded(_limit_message(self.name, limit))
                hooks.trigger(
                    LOOP_LIMIT, self._event(LOOP_LIMIT, context, pending, error=error)
                )
                raise error
            iterations += 1
            context.trace.append(f"{self.name}:{iterations}")
            if not self._iteration(context, hooks, iterations):
                reason = "break"
                break
        hooks.trigger(
            LOOP_END, self._event(LOOP_END, context, iterations, reason=reason)
        )
        context.trace.append(f"{self.name}:done")

    def _condition(self, context: Context, hooks: LoopHooks, index: int) -> bool:
        '''判一次条件；条件本身抛异常时触发 `LOOP_ERROR`（随后照常抛）。'''
        try:
            return bool(self.condition(context))
        except _CONTROL_FLOW:
            raise
        except Exception as exc:
            hooks.trigger(
                LOOP_ERROR, self._event(LOOP_ERROR, context, index, error=exc)
            )
            raise

    def _iteration(self, context: Context, hooks: LoopHooks, index: int) -> bool:
        '''跑一轮循环体。

        返回：
            bool —— `True` 继续下一轮；`False` 表示遇到 `Break`，该退出循环。
        '''
        if not hooks.trigger(BEFORE_ITERATION, self._event(BEFORE_ITERATION, context, index)):
            # 本轮被跳过：仍然触发 after_iteration，事件里 signal="skipped"
            hooks.trigger(
                AFTER_ITERATION,
                self._event(AFTER_ITERATION, context, index, signal="skipped"),
            )
            return True

        started = time.perf_counter()
        try:
            signal = run_loop_body(self.body, context)
        except _CONTROL_FLOW:
            raise
        except Exception as exc:
            hooks.trigger(
                LOOP_ERROR,
                self._event(LOOP_ERROR, context, index, error=exc),
            )
            raise
        hooks.trigger(
            AFTER_ITERATION,
            self._event(
                AFTER_ITERATION,
                context,
                index,
                signal=signal,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
        )
        return signal != "break"

    # ---------------- 钩子表 ----------------
    def _hooks(self, context: Context) -> LoopHooks:
        '''合成这次的执行表：步级钩子在前，Context 里的全局钩子在后。

        没有全局钩子时直接返回 `self.hooks`（零拷贝，正常路径不产生临时对象）。
        '''
        shared = context.get(LOOP_HOOKS_KEY)
        if not isinstance(shared, LoopHooks) or not shared:
            return self.hooks
        merged = self.hooks.copy()
        return merged.extend(shared)

    def _event(self, point: HookPoint, context: Context, iteration: int, **extra: Any):
        '''构造事件：`extra` 里的字段（`signal` / `reason` / `error` / ...）按需补上。'''
        return LoopHookEvent(point, self, context, iteration=iteration, **extra)

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        return f"While {self.condition!r}"

    def children(self) -> List[Step]:
        return list(self.body)

    def __repr__(self) -> str:
        hooks = f" hooks={len(self.hooks)}" if self.hooks else ""
        return (
            f"<WhileStep max_iterations={self.max_iterations} "
            f"body={len(self.body)}{hooks}>"
        )


def _limit_message(name: str, limit: int) -> str:
    return (
        f"{name} 循环超过 {limit} 次仍未结束，"
        "请检查循环条件里的变量有没有在循环体里被更新"
    )


def While(  # noqa: N802
    condition: ConditionLike,
    *,
    max_iterations: Optional[int] = None,
    hooks: Any = None,
) -> WhileStep:
    '''构造 While 循环：`While(条件, ...).Do(步骤...)[.Hook(扩展点, 函数)]`。

    参数：

    condition（ConditionLike）:
        循环条件，每轮求值一次，为真则跑循环体。可以传 `Var` 表达式
        （如 `Var("n") < 3`）、`Callable[[Context], bool]`，或 `True` / `False`。
    max_iterations（int | None，默认 None）:
        迭代上限，防死循环。`None` = 用全局的 `settings.max_iterations`
        （默认 100）。必须是正整数，否则抛 `ValueError`。
        超出时先触发 `LOOP_LIMIT`，再抛 `LoopLimitExceeded`。
    hooks（LoopHooks | dict | list | None，默认 None）:
        初始步级钩子，三种写法等价：

          - `LoopHooks`：**原样复用**（之后 `.Hook()` 的改动对调用方可见）
          - `dict`：`{扩展点: 函数 或 函数列表}`，
            例 `{"before_iteration": log, "loop_end": [save, notify]}`
          - `list/tuple`：`[(扩展点, 函数), ...]`

        扩展点写字符串或 `HookPoint` 均可；传错类型抛 `TypeError`。

    返回：
        WhileStep —— 还没设置循环体，接着调 `.Do(...)`。

    示例：

        While(Var("n") < 3, max_iterations=10).Do(Set("n", Var("n") + 1))

        While(cond, hooks={"before_iteration": log}).Do(...)

        While(cond).Hook(AFTER_ITERATION, print).Do(...)

        While(cond).Hook(BEFORE_CONDITION, budget_ok).Do(...)   # 返回 False 提前结束
    '''
    return WhileStep(
        condition=as_condition(condition),
        max_iterations=max_iterations,
        # 已经是注册表就原样复用（`.Hook()` 的改动对调用方可见）
        hooks=hooks if isinstance(hooks, LoopHooks) else LoopHooks(hooks),
    )


__all__ = [
    "While",
    "WhileStep",
    "LoopHooks",
    "LoopHookEvent",
    "LoopHookHandle",
    "HookPoint",
    "loop_hooks",
    "as_hook_point",
    "LOOP_HOOKS_KEY",
    "LOOP_START",
    "BEFORE_CONDITION",
    "BEFORE_ITERATION",
    "AFTER_ITERATION",
    "LOOP_LIMIT",
    "LOOP_ERROR",
    "LOOP_END",
    "LOOP_HOOK_POINTS",
]
