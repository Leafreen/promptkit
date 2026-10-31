# promptkit/api/hooks.py
'''Hook 扩展 API：在 DSL 步骤执行的前 / 后 / 异常时插入自定义逻辑。

三个生命周期钩子：

    before   前置拦截：返回 False（或抛 HookAbort）→ 这一步不执行
    after    后置处理：步骤正常结束后调用，可以改 Context
    error    异常捕获：返回 True → 吞掉异常，工作流继续

一句接入（`execute_step` 是所有 DSL 步骤的唯一出入口，所以不需要适配层）：

    from promptkit.api import Hooks

    hooks = Hooks()
    hooks.on_before(lambda e: None)                 # 观察
    hooks.on_after(lambda e: print(e.duration_ms))  # 计时
    hooks.on_error(lambda e: e.error is not None)   # 兜底

    Workflow(...).run(agent=agent, hooks=hooks)

钩子的唯一参数是 `HookEvent`（`kind` / `step` / `context` / `error` /
`duration_ms`）。注册返回 `HookHandle`，`handle.off()` 或
`hooks.unregister(fn)` 都能注销。

注意：`before` 钩子会看到 Break / Continue / Goto 这类控制流步骤；把它们的
返回值设成 False 会跳过信号本身，通常不是你想要的效果。`error` 钩子**不会**
收到控制流信号 —— 它们不是错误。
'''
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from promptkit.core.errors import WorkflowError
from promptkit.core.steps.interface import HOOKS_KEY

__all__ = [
    "HookType",
    "HookEvent",
    "HookAbort",
    "HookHandle",
    "Hooks",
    "HOOKS_KEY",
    "use_hooks",
]


class HookType(str, Enum):
    '''钩子的三个生命周期位置。'''

    BEFORE = "before"
    AFTER = "after"
    ERROR = "error"


class HookAbort(WorkflowError):
    '''`before` 钩子里抛出它 → 跳过当前步骤（工作流继续往下走）。'''


@dataclass(frozen=True)
class HookEvent:
    '''一次钩子调用携带的全部信息。'''

    kind: HookType
    step: Any
    context: Any
    error: Optional[BaseException] = None
    duration_ms: float = 0.0

    @property
    def name(self) -> str:
        '''步骤名（`Step.name`）。'''
        return str(getattr(self.step, "name", "")) or type(self.step).__name__


class HookHandle:
    '''一次注册的句柄：`off()` 即注销。'''

    __slots__ = ("hooks", "kind", "fn")

    def __init__(self, hooks: "Hooks", kind: HookType, fn: Callable[[HookEvent], Any]):
        self.hooks = hooks
        self.kind = kind
        self.fn = fn

    def off(self) -> bool:
        '''注销这个钩子；已经注销过返回 False。'''
        return self.hooks.unregister(self)

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return f"<HookHandle {self.kind.value}:{name}>"


class Hooks:
    '''钩子注册表：`on_before` / `on_after` / `on_error` 注册，`run_*` 派发。'''

    __slots__ = ("_before", "_after", "_error")

    def __init__(
        self,
        *,
        before: Iterable[Callable[[HookEvent], Any]] = (),
        after: Iterable[Callable[[HookEvent], Any]] = (),
        error: Iterable[Callable[[HookEvent], Any]] = (),
    ):
        self._before: List[Callable[[HookEvent], Any]] = []
        self._after: List[Callable[[HookEvent], Any]] = []
        self._error: List[Callable[[HookEvent], Any]] = []
        for fn in before:
            self.register(HookType.BEFORE, fn)
        for fn in after:
            self.register(HookType.AFTER, fn)
        for fn in error:
            self.register(HookType.ERROR, fn)

    # ---------------- 注册 / 注销 ----------------
    def register(self, kind: Union[str, HookType], fn: Callable[[HookEvent], Any]) -> HookHandle:
        '''注册一个钩子，返回可 `off()` 的句柄。

        同一个函数重复注册只保留一次。
        '''
        if not callable(fn):
            raise TypeError(f"钩子必须是 callable，得到 {type(fn).__name__}")
        resolved = _as_kind(kind)
        bucket = self._bucket(resolved)
        if fn not in bucket:
            bucket.append(fn)
        return HookHandle(self, resolved, fn)

    def unregister(
        self,
        target: Union[HookHandle, Callable[[HookEvent], Any]],
        kind: Optional[Union[str, HookType]] = None,
    ) -> bool:
        '''注销：句柄 / 函数都能传；函数不指定 kind 时从三类里全删。'''
        kinds: Tuple[HookType, ...]
        if isinstance(target, HookHandle):
            fn = target.fn
            kinds = (_as_kind(target.kind),)
        elif callable(target):
            fn = target
            kinds = (_as_kind(kind),) if kind is not None else tuple(HookType)
        else:
            raise TypeError(
                f"unregister 需要 HookHandle 或 callable，得到 {type(target).__name__}"
            )

        removed = False
        for item in kinds:
            bucket = self._bucket(item)
            if fn in bucket:
                bucket.remove(fn)
                removed = True
        return removed

    def clear(self, kind: Optional[Union[str, HookType]] = None) -> None:
        '''清空全部钩子，或只清某一类。'''
        kinds = tuple(HookType) if kind is None else (_as_kind(kind),)
        for item in kinds:
            self._bucket(item).clear()

    # ---------------- 装饰器写法 ----------------
    def on_before(self, fn: Optional[Callable[[HookEvent], Any]] = None):
        '''`@hooks.on_before` 或 `hooks.on_before(fn)`。'''
        return self._decorate(HookType.BEFORE, fn)

    def on_after(self, fn: Optional[Callable[[HookEvent], Any]] = None):
        '''`@hooks.on_after` 或 `hooks.on_after(fn)`。'''
        return self._decorate(HookType.AFTER, fn)

    def on_error(self, fn: Optional[Callable[[HookEvent], Any]] = None):
        '''`@hooks.on_error` 或 `hooks.on_error(fn)`。'''
        return self._decorate(HookType.ERROR, fn)

    def _decorate(self, kind: HookType, fn: Optional[Callable[[HookEvent], Any]]):
        if fn is None:
            def decorator(inner: Callable[[HookEvent], Any]):
                self.register(kind, inner)
                return inner

            return decorator
        self.register(kind, fn)
        return fn

    # ---------------- 派发（由 execute_step 调用） ----------------
    def run_before(self, step: Any, context: Any) -> bool:
        '''返回 False 表示拦截当前步骤（不执行）。'''
        if not self._before:
            return True
        event = HookEvent(HookType.BEFORE, step, context)
        for fn in tuple(self._before):
            try:
                outcome = fn(event)
            except HookAbort:
                return False
            if outcome is False:
                return False
        return True

    def run_after(self, step: Any, context: Any, duration_ms: float = 0.0) -> None:
        '''步骤正常结束后调用；钩子的返回值被忽略。'''
        if not self._after:
            return
        event = HookEvent(HookType.AFTER, step, context, duration_ms=duration_ms)
        for fn in tuple(self._after):
            fn(event)

    def run_error(self, step: Any, context: Any, error: BaseException) -> bool:
        '''返回 True 表示异常已被处理，工作流继续。'''
        if not self._error:
            return False
        event = HookEvent(HookType.ERROR, step, context, error=error)
        recovered = False
        for fn in tuple(self._error):
            if fn(event) is True:
                recovered = True
        return recovered

    # ---------------- 杂项 ----------------
    def _bucket(self, kind: HookType) -> List[Callable[[HookEvent], Any]]:
        if kind is HookType.BEFORE:
            return self._before
        if kind is HookType.AFTER:
            return self._after
        return self._error

    def counts(self) -> Dict[str, int]:
        '''各类钩子的数量，排查「到底挂上没有」时有用。'''
        return {
            HookType.BEFORE.value: len(self._before),
            HookType.AFTER.value: len(self._after),
            HookType.ERROR.value: len(self._error),
        }

    def __bool__(self) -> bool:
        return bool(self._before or self._after or self._error)

    def __len__(self) -> int:
        return len(self._before) + len(self._after) + len(self._error)

    def __repr__(self) -> str:
        return (
            f"<Hooks before={len(self._before)} "
            f"after={len(self._after)} error={len(self._error)}>"
        )


def _as_kind(kind: Union[str, HookType]) -> HookType:
    if isinstance(kind, HookType):
        return kind
    try:
        return HookType(str(kind).lower())
    except ValueError:
        options = " / ".join(item.value for item in HookType)
        raise ValueError(f"未知的钩子类型 {kind!r}，可选：{options}") from None


def use_hooks(context: Any) -> Hooks:
    '''从 Context 取钩子；没有就建一个挂上去并返回。

    适合在 `FunctionStep` 里拿到本次运行用的钩子表：

        def add_hook(ctx):
            use_hooks(ctx).on_after(my_hook)
    '''
    current = context.get(HOOKS_KEY)
    if isinstance(current, Hooks):
        return current
    created = Hooks()
    context[HOOKS_KEY] = created
    return created
