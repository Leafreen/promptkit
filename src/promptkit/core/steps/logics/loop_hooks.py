# core/steps/logics/loop_hooks.py
'''While 循环的 Hook 扩展 API：扩展点、事件、注册表。

和 `promptkit.api.hooks` 的分工：那边是**步骤级**（每一步前后），
这里是**循环轮级**（每一轮前后、跳过、超限、异常、结束），信息更细。

七个扩展点，执行顺序固定：

    loop_start          进入循环前，一次
      before_condition  每轮判条件前，可返回 False 结束整个循环
        before_iteration  条件成立、跑循环体前，可返回 False 跳过本轮
        after_iteration   本轮结束（含被跳过 / Break / Continue）后，一次
      loop_limit        即将超出 max_iterations 之前（异常随后照常抛）
      loop_error        条件或循环体抛异常时（异常随后照常抛）
    loop_end            循环正常结束 / Break 退出，一次

各扩展点上 `LoopHookEvent` 哪些字段有意义：

| 扩展点 | iteration | condition | signal | reason | error | duration_ms |
| --- | --- | --- | --- | --- | --- | --- |
| `loop_start` | 0（已完成 0 轮） | — | — | — | — | — |
| `before_condition` | 即将执行第几轮 | — | — | — | — | — |
| `before_iteration` | 即将执行第几轮 | — | — | — | — | — |
| `after_iteration` | 刚结束的第几轮 | — | ✓ | — | — | ✓ |
| `loop_limit` | 将被拒绝的第几轮 | — | — | — | ✓ | — |
| `loop_error` | 出错的第几轮 | — | — | — | ✓ | — |
| `loop_end` | 已完成轮数 | — | — | ✓ | — | — |

「—」表示该字段在这个扩展点上取不到值（保持 `None` / 0）。

钩子怎么写（签名只有一个参数）：

    def before_iteration(event: LoopHookEvent) -> Optional[bool]:
        print(event.iteration, event.name)
        return True          # 只有 veto 点看返回值，见 `HookPoint`

错误处理：钩子自己抛异常不会影响主循环 —— 异常被记账进
`LoopHooks.failures`，这个钩子被跳过，其余钩子与循环继续。

扩展示例（新增一个扩展点只需两处：枚举成员 + 一次 trigger）：

    class HookPoint(str, Enum):
        ...
        BEFORE_RETRY = "before_retry"       # 1) 加枚举成员

    hooks.trigger(HookPoint.BEFORE_RETRY, event)   # 2) 在需要的地方触发

注册 / 注销 / 派发 / 错误隔离 / 合并都不需要改。
'''
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

__all__ = [
    "HookPoint",
    "LoopHookEvent",
    "LoopHookFn",
    "LoopHookHandle",
    "HookFailure",
    "LoopHooks",
    "LOOP_HOOKS_KEY",
    "loop_hooks",
    "as_hook_point",
]

#: Context 里存放「全局循环钩子表」的键。
#: 正常用法是调 `loop_hooks(context)`，不需要直接读写这个常量。
LOOP_HOOKS_KEY = "_promptkit_loop_hooks"


class HookPoint(str, Enum):
    '''循环的扩展点枚举。值即字符串（`"loop_start"` 等），可直接当名字用。

    | 扩展点 | 执行时机 | 返回值处理 |
    | --- | --- | --- |
    | `LOOP_START` | 进入循环前，一次 | 忽略 |
    | `BEFORE_CONDITION` | 每轮判断条件前 | `False` → 结束整个循环（`reason="hook"`） |
    | `BEFORE_ITERATION` | 条件成立、跑循环体前 | `False` → 跳过本轮（仍会触发 `AFTER_ITERATION`） |
    | `AFTER_ITERATION` | 本轮结束（含被跳过 / Break / Continue） | 忽略 |
    | `LOOP_LIMIT` | 超出 `max_iterations`、抛异常之前 | 忽略（异常照常抛） |
    | `LOOP_ERROR` | 条件或循环体抛异常时 | 忽略（异常照常抛） |
    | `LOOP_END` | 循环正常结束 / Break | 忽略 |

    返回值规则：**只有 `BEFORE_CONDITION` / `BEFORE_ITERATION` 看返回值**，
    且只认精确的 `False`（`None`、`0`、`""` 都算「不否决」）；其余扩展点
    的返回值一律忽略，可以 `return None`。

    执行顺序：同一扩展点上多个钩子按**注册顺序**串行执行；某个钩子返回
    `False` 时立即短路，后面的钩子不再执行。
    '''

    LOOP_START = "loop_start"
    BEFORE_CONDITION = "before_condition"
    BEFORE_ITERATION = "before_iteration"
    AFTER_ITERATION = "after_iteration"
    LOOP_LIMIT = "loop_limit"
    LOOP_ERROR = "loop_error"
    LOOP_END = "loop_end"


#: 返回值会被当成「否决信号」的扩展点（其余扩展点忽略返回值）
VETO_POINTS: Tuple[HookPoint, ...] = (
    HookPoint.BEFORE_CONDITION,
    HookPoint.BEFORE_ITERATION,
)


@dataclass(frozen=True)
class LoopHookEvent:
    '''一次钩子调用携带的全部信息 —— 钩子函数的**唯一入参**，只读。

    字段（按需取用，取不到的字段保持默认值）：

        point（HookPoint）:
            当前扩展点。一个钩子函数挂到多个扩展点时，用它区分是哪一个。
        loop（WhileStep）:
            触发它的循环步骤。可读 `loop.name` / `loop.max_iterations` /
            `loop.body`，不建议在钩子里改它。
        context（Context）:
            整个工作流的 Context，**可读可写**。在钩子里 `context["x"] = 1`
            会立刻对后续步骤生效。
        iteration（int，默认 0）:
            轮次，从 1 起。`before_*` 是「即将执行的第几轮」，
            `after_iteration` 是「刚结束的第几轮」，`loop_start` 恒为 0，
            `loop_end` 是已完成的轮数。
        condition（bool | None，默认 None）:
            条件结果。`before_condition` 里还是 `None`（尚未求值）；
            `loop_end` 里放下的是**最终**那次条件结果（因 `Break` /
            `hook` 结束时为 `False`）。
        signal（str | None，默认 None）:
            本轮是怎么结束的，仅 `after_iteration` 有值：
            `None`=正常跑完，`"continue"`=本轮遇到 `Continue`，
            `"break"`=本轮遇到 `Break`，`"skipped"`=被 `before_iteration` 否决。
        reason（str | None，默认 None）:
            循环为什么结束，仅 `loop_end` 有值：
            `"condition"`=条件不成立，`"break"`=跳出，`"hook"`=被钩子拦下。
        error（BaseException | None，默认 None）:
            异常对象，仅 `loop_error` / `loop_limit` 有值。读它做日志，
            不要吞掉（异常随后仍会抛给上层）。
        duration_ms（float，默认 0.0）:
            这一轮的耗时（毫秒），仅 `after_iteration` 有值。

    属性：

        name（str）:
            循环名，等于 `loop.name`；循环没有名字时返回 `"loop"`。
        vetoed（bool）:
            当前扩展点是否支持「返回 False 否决」。
    '''

    point: HookPoint
    loop: Any
    context: Any
    iteration: int = 0
    condition: Optional[bool] = None
    signal: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[BaseException] = None
    duration_ms: float = 0.0

    @property
    def name(self) -> str:
        '''循环名（`loop.name`）；循环没设名字时是 `"loop"`。'''
        return str(getattr(self.loop, "name", "")) or "loop"

    @property
    def vetoed(self) -> bool:
        '''当前扩展点是否支持「返回 False 否决」（即是否在 `VETO_POINTS` 里）。'''
        return self.point in VETO_POINTS


#: 钩子函数的类型签名。
#:
#:     参数：event (LoopHookEvent) —— 唯一入参，只读快照 + 可写 context
#:     返回：Optional[bool] —— 只有 veto 点需要返回 False 表示否决；
#:           其余扩展点返回值被忽略，可以不带 return
#:
#: 钩子抛任何 `Exception` 都不会影响主循环（见 `LoopHooks.trigger`）。
LoopHookFn = Callable[[LoopHookEvent], Optional[bool]]


@dataclass(frozen=True)
class HookFailure:
    '''一次钩子执行失败的记录（异常被吞掉，不影响主循环）。

    字段：

        point（HookPoint）:
            钩子注册在哪个扩展点。
        fn（Callable）:
            出错的钩子函数本身。
        error（BaseException）:
            钩子抛出来的异常。
        iteration（int，默认 0）:
            出错时正在跑第几轮。

    属性：

        name（str）:
            钩子的名字（`__name__`，匿名函数是 `"<lambda>"`）。
    '''

    point: HookPoint
    fn: Any
    error: BaseException
    iteration: int = 0

    @property
    def name(self) -> str:
        '''钩子的函数名（`__name__`），拿不到时退化为类型名。'''
        return getattr(self.fn, "__name__", type(self.fn).__name__)

    def __str__(self) -> str:
        return f"{self.point.value}:{self.name} -> {type(self.error).__name__}: {self.error}"


class LoopHookHandle:
    '''一次注册的句柄：持着「注册在哪个扩展点的哪个函数」，`off()` 即注销。'''

    __slots__ = ("hooks", "point", "fn")

    def __init__(self, hooks: "LoopHooks", point: HookPoint, fn: LoopHookFn):
        '''参数：

        hooks（LoopHooks）:
            所属的注册表，注销时要回到它上面删。
        point（HookPoint）:
            注册到的扩展点（已归一化为枚举，不会是字符串）。
        fn（LoopHookFn）:
            被注册的钩子函数。
        '''
        self.hooks = hooks
        self.point = point
        self.fn = fn

    def off(self) -> bool:
        '''注销这个钩子。

        返回：
            bool —— 真的删掉了为 `True`；之前已经注销过为 `False`（幂等）。
        '''
        return self.hooks.unregister(self)

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return f"<LoopHookHandle {self.point.value}:{name}>"


class LoopHooks:
    '''循环钩子注册表：注册 / 注销 / 触发。

        hooks = LoopHooks({"before_iteration": my_fn})
        handle = hooks.register("after_iteration", fn)   # 注册，拿到句柄
        handle.off()                                     # 注销

        @hooks.on("loop_end")                            # 装饰器写法
        def bye(event): ...

    作用域：步级（`While(...).Hook(...)`，只对该循环生效）与全局
    （`loop_hooks(context)`，对同一 Context 下所有循环生效）各存一份；
    执行时**先跑步级、再跑全局**。

    不可变的约束：注册顺序即执行顺序；同一个函数在同一扩展点只保留一次。
    '''

    __slots__ = ("_buckets", "failures", "max_failures")

    def __init__(
        self,
        hooks: Optional[Union["LoopHooks", Mapping[Any, Any], Iterable[Any]]] = None,
        *,
        max_failures: int = 100,
    ):
        '''参数：

        hooks（LoopHooks | dict | list | None，默认 None）:
            初始钩子，三种写法（都不传就是空表）：

              - `LoopHooks`：把另一张表里的钩子复制进来（函数共享）
              - `dict`：`{扩展点: 函数 或 函数列表}`，
                例 `{"before_iteration": fn, "loop_end": [fn1, fn2]}`
              - `list/tuple`：`[(扩展点, 函数), ...]`，等价于 dict 的条目

            扩展点写字符串或 `HookPoint` 都可以，大小写不敏感；未知扩展点报错。
        max_failures（int，默认 100）:
            `failures` 最多保留多少条，防止长时间运行的循环把失败记录堆爆内存。
            超出后新的失败被丢弃（不影响循环本身）。小于 1 会被夹到 1。

        dict 的取值也接受「函数列表」，列表里的函数按顺序注册。
        '''
        self._buckets: Dict[HookPoint, List[LoopHookFn]] = {
            point: [] for point in HookPoint
        }
        #: 钩子执行失败的记录（异常不会影响主循环）
        self.failures: List[HookFailure] = []
        self.max_failures = max(1, int(max_failures))
        self._load(hooks)

    # ---------------- 注册 / 注销 ----------------
    def register(
        self, point: Union[str, HookPoint], fn: LoopHookFn
    ) -> LoopHookHandle:
        '''注册一个钩子。

        参数：

        point（str | HookPoint）:
            扩展点名，如 `"after_iteration"` / `HookPoint.LOOP_END`。
            大小写不敏感；未知名字抛 `ValueError`（错误信息里列出全部可选项）。
        fn（LoopHookFn）:
            钩子函数，签名 `fn(event: LoopHookEvent) -> Optional[bool]`。
            同一函数重复注册到同一扩展点只保留一次（不会重复执行）。

        返回：
            LoopHookHandle —— 句柄，调 `.off()` 注销。

        异常：
            TypeError: `fn` 不是 callable。
            ValueError: `point` 不是已知扩展点。
        '''
        resolved = as_hook_point(point)
        if not callable(fn):
            raise TypeError(f"钩子必须是 callable，得到 {type(fn).__name__}")
        bucket = self._buckets[resolved]
        if fn not in bucket:
            bucket.append(fn)
        return LoopHookHandle(self, resolved, fn)

    def unregister(
        self,
        target: Union[str, HookPoint, LoopHookHandle, LoopHookFn],
        point: Optional[Union[str, HookPoint]] = None,
    ) -> bool:
        '''注销钩子。`target` 有四种含义：

        参数：

        target（LoopHookHandle | str | HookPoint | Callable）:
            - `LoopHookHandle`：注销句柄对应的那一个钩子
            - 扩展点名：清掉该扩展点上的**全部**钩子
            - 函数：按函数删
        point（str | HookPoint | None，默认 None）:
            仅当 `target` 是函数时有意义：指定只从哪个扩展点删；
            不传则从这个函数注册过的**所有**扩展点删。

        返回：
            bool —— 真的删掉了为 `True`，本来就没有为 `False`（不抛异常）。

        异常：
            TypeError: `target` 既不是句柄、也不是扩展点名、也不是 callable。
            ValueError: 传了 `point` 但不是已知扩展点。
        '''
        if isinstance(target, LoopHookHandle):
            return self._drop(target.point, target.fn)
        if isinstance(target, (str, HookPoint)):
            # unregister("before_iteration")：清掉这个扩展点上的全部钩子
            return bool(self.clear(target))
        if not callable(target):
            raise TypeError(
                f"unregister 需要 LoopHookHandle / callable / 扩展点名，"
                f"得到 {type(target).__name__}"
            )
        points = (
            (as_hook_point(point),) if point is not None else tuple(HookPoint)
        )
        removed = False
        for item in points:
            removed = self._drop(item, target) or removed
        return removed

    def clear(self, point: Optional[Union[str, HookPoint]] = None) -> int:
        '''清空钩子。

        参数：

        point（str | HookPoint | None，默认 None）:
            传了只清这一个扩展点；不传清全部七个扩展点。

        返回：
            int —— 被清掉的钩子个数（同一函数跨扩展点注册会各算一次）。
        '''
        points = (as_hook_point(point),) if point is not None else tuple(HookPoint)
        removed = 0
        for item in points:
            removed += len(self._buckets[item])
            self._buckets[item].clear()
        return removed

    # ---------------- 装饰器写法 ----------------
    def on(self, point: Union[str, HookPoint]):
        '''装饰器注册：`@hooks.on("loop_end")`。

        参数：

        point（str | HookPoint）:
            要挂到的扩展点（注册时立即校验，未知名字当场抛 `ValueError`）。

        返回：
            decorator —— 收一个钩子函数，注册它，并**原样返回**该函数
            （所以装饰后仍是普通函数，可以继续被调用 / 被别处注册）。
        '''
        resolved = as_hook_point(point)

        def decorator(fn: LoopHookFn) -> LoopHookFn:
            self.register(resolved, fn)
            return fn

        return decorator

    # ---------------- 派发 ----------------
    def trigger(self, point: Union[str, HookPoint], event: LoopHookEvent) -> bool:
        '''触发一个扩展点，按注册顺序跑该扩展点上的全部钩子。

        参数：

        point（str | HookPoint）:
            要触发的扩展点。通常由循环内部传入，不是 `event.point` 的
            替代品 —— 两者应一致，用参数只是为了让派发不依赖事件构造。
        event（LoopHookEvent）:
            传给每个钩子的事件对象（同一个实例依次传给所有钩子）。

        返回：
            bool —— `True` 表示没被否决；`False` 表示被某个钩子否决
            （只有 `VETO_POINTS` 里的扩展点可能返回 `False`，
            其余扩展点恒为 `True`）。

        「否决」的判定：某个钩子返回**精确的 `False`** 即短路，后面的
        钩子不再执行；返回 `None` / `0` / 空值都不算否决。

        错误处理：某个钩子抛 `Exception` 时，记一条 `HookFailure` 到
        `self.failures`，**跳过这个钩子**继续执行后面的钩子并返回未被否决；
        循环本身不受任何影响。`BaseException`（如 `KeyboardInterrupt`）
        不拦，直接向上抛。

        异常：
            ValueError: `point` 不是已知扩展点。
        '''
        resolved = as_hook_point(point)
        bucket = self._buckets[resolved]
        if not bucket:
            return True
        for fn in tuple(bucket):
            try:
                outcome = fn(event)
            except Exception as exc:      # noqa: BLE001 - 钩子失败不能影响主循环
                self._record(resolved, fn, exc, event.iteration)
                continue
            if outcome is False and resolved in VETO_POINTS:
                return False
        return True

    def run(self, point: Union[str, HookPoint], event: LoopHookEvent) -> bool:
        '''`trigger` 的别名（参数与返回值完全相同），语义更直白。'''
        return self.trigger(point, event)

    # ---------------- 合并 / 查看 ----------------
    def extend(self, other: Optional["LoopHooks"]) -> "LoopHooks":
        '''把另一张表里的钩子**追加**进来，保持各自原有的相对顺序。

        参数：

        other（LoopHooks | None）:
            来源表。`None` / 空表直接返回，不做事。
            已经在本表里的同一个函数不会重复添加。

        返回：
            LoopHooks —— `self`（原地修改，方便链式 `a.extend(b).extend(c)`）。
        '''
        if not other:
            return self
        for point, bucket in other._buckets.items():
            target = self._buckets[point]
            for fn in bucket:
                if fn not in target:
                    target.append(fn)
        return self

    def copy(self) -> "LoopHooks":
        '''复制一份注册表。

        返回：
            LoopHooks —— 新表，钩子函数本身共享（浅拷贝），注册 / 注销互不影响。
            循环执行时就是用它把「步级 + 全局」合成一张临时表，避免污染原表。
        '''
        return LoopHooks().extend(self)

    def counts(self) -> Dict[str, int]:
        '''每个扩展点各挂了几个钩子，用于排查「到底挂上没有」。

        返回：
            dict —— 键是全部七个扩展点的字符串名（没挂的也列出，值为 0）。
        '''
        return {point.value: len(bucket) for point, bucket in self._buckets.items()}

    # ---------------- 内部 ----------------
    def _drop(self, point: HookPoint, fn: LoopHookFn) -> bool:
        bucket = self._buckets[point]
        if fn not in bucket:
            return False
        bucket.remove(fn)
        return True

    def _record(
        self,
        point: HookPoint,
        fn: LoopHookFn,
        error: BaseException,
        iteration: int,
    ) -> None:
        if len(self.failures) >= self.max_failures:
            return
        self.failures.append(HookFailure(point, fn, error, iteration))

    def _load(self, hooks: Optional[Union["LoopHooks", Mapping[Any, Any], Iterable[Any]]]) -> None:
        if hooks is None:
            return
        if isinstance(hooks, LoopHooks):
            self.extend(hooks)
            return
        if isinstance(hooks, Mapping):
            pairs: Iterable[Any] = hooks.items()
        elif isinstance(hooks, (list, tuple)):
            pairs = hooks
        else:
            raise TypeError(
                "hooks 需要 LoopHooks / dict（{扩展点: 函数或函数列表}）/ "
                f"(扩展点, 函数) 列表，得到 {type(hooks).__name__}"
            )
        for point, value in pairs:
            for fn in (value if _is_seq(value) else (value,)):
                self.register(point, fn)

    def __bool__(self) -> bool:
        return any(self._buckets.values())

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items() if v)
        return f"<LoopHooks {counts or 'empty'}>"


def _is_seq(value: Any) -> bool:
    '''`dict` 的值是「函数列表」还是「单个函数」——注意别把 callable 当序列。'''
    return isinstance(value, (list, tuple)) and not callable(value)


def as_hook_point(point: Union[str, HookPoint]) -> HookPoint:
    '''把字符串 / 枚举统一成 `HookPoint`。

    参数：

    point（str | HookPoint）:
        扩展点名。字符串会先转小写再匹配，所以 `"LOOP_END"` 也认。

    返回：
        HookPoint —— 对应的枚举成员。

    异常：
        ValueError: 未知名字；错误信息里列出全部可选项。
    '''
    if isinstance(point, HookPoint):
        return point
    try:
        return HookPoint(str(point).lower())
    except ValueError:
        options = " / ".join(item.value for item in HookPoint)
        raise ValueError(f"未知的循环 Hook 扩展点 {point!r}，可选：{options}") from None


def loop_hooks(context: Any, *, create: bool = True) -> Optional[LoopHooks]:
    '''取（或懒建）Context 里的**全局**循环钩子表 —— 运行时给所有循环挂钩子。

        def add_hook(ctx):
            loop_hooks(ctx).register("after_iteration", my_fn)

    参数：

    context（Context）:
        工作流的 Context。全局钩子挂在它身上，所以只对同一个 Context
        下的循环生效。
    create（bool，默认 True）:
        `True` —— 没有就新建一张空表塞进 Context 再返回；
        `False` —— 只查询，没有就返回 `None`（不产生副作用，适合
        「有没有钩子」的判断）。

    返回：
        LoopHooks | None —— 全局钩子表；`create=False` 且不存在时为 `None`。

    执行顺序：循环先跑步级钩子（`While(...).Hook(...)`），再跑这里的全局钩子。
    '''
    current = context.get(LOOP_HOOKS_KEY)
    if isinstance(current, LoopHooks):
        return current
    if not create:
        return None
    created = LoopHooks()
    context[LOOP_HOOKS_KEY] = created
    return created
