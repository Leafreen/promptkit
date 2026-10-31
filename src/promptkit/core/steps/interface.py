from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Sequence, TypeAlias, Union

from promptkit.core.errors import BreakSignal, ContinueSignal, GotoSignal, ReturnSignal
from promptkit.core.event_sys.events import Event, EventType, bus_of
from promptkit.core.display_sys.tree import Node, format_tree, short

if TYPE_CHECKING:  # 只用于类型标注：interface 是 steps 的最底层，不该依赖 model_action
    from promptkit.core.context import Context

# 这些是控制流信号，不是「出错」，所以不发 ERROR 事件
_CONTROL_FLOW = (BreakSignal, ContinueSignal, ReturnSignal, GotoSignal)

#: Context 里存放 `Hooks` 注册表的键（`promptkit.api.hooks` 用的就是它）
HOOKS_KEY = "_promptkit_hooks"


def execute_step(step: "Step", context: Context) -> None:
    '''执行一步，并在前后发出 STEP_START / STEP_END 事件。

    没有挂事件总线时这是一次普通调用（连 describe() 都不会算），
    所以核心逻辑里可以放心到处用它。

    挂了 `Hooks`（`context[HOOKS_KEY]`）时顺带派发三个生命周期钩子：
    before（返回 False 即拦截）、after（正常结束）、error（返回 True 即吞掉异常）。
    '''
    hooks = context.get(HOOKS_KEY)
    bus = bus_of(context)
    if bus is None and hooks is None:
        step.execute(context)
        return

    info = {"step": step.describe(), "name": step.name}
    if bus is not None:
        bus.emit(Event(EventType.STEP_START, dict(info)))

    started = time.perf_counter()
    if hooks is not None and not hooks.run_before(step, context):
        return  # 前置拦截：这一步不执行，工作流继续

    try:
        step.execute(context)
    except _CONTROL_FLOW:
        raise  # break / continue / return / goto：这一步没有正常跑完
    except Exception as exc:
        if hooks is None or not hooks.run_error(step, context, exc):
            if bus is not None:
                bus.emit(
                    Event(
                        EventType.ERROR,
                        {
                            **info,
                            "error": f"{type(exc).__name__}: {exc}",
                            "duration_ms": _elapsed_ms(started),
                        },
                    )
                )
            raise
        # 异常被 error 钩子吸收：视作正常结束，继续往下走
    else:
        if hooks is not None:
            hooks.run_after(step, context, _elapsed_ms(started))

    if bus is not None:
        bus.emit(
            Event(EventType.STEP_END, {**info, "duration_ms": _elapsed_ms(started)})
        )



def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


class Step(ABC):
    '''工作流中的一步。子类实现 execute()，就地修改 Context。'''

    name: str = "step"

    @abstractmethod
    def execute(self, context: Context) -> None:
        '''执行该步骤，结果写回 context。'''
        raise NotImplementedError

    # ---------------- 展示成步骤树 ----------------
    def describe(self) -> str:
        '''这一行显示成什么。'''
        return repr(self)

    def children(self) -> List["Step"]:
        '''子步骤；没有子步骤的默认返回空。'''
        return []

    def to_node(self) -> Node:
        '''转成展示用的树节点。结构特殊的步骤（If/While/For）会覆盖它。'''
        return Node(self.describe(), [child.to_node() for child in self.children()])

    def render(self) -> str:
        '''把这一步及其子步骤渲染成多行文本。'''
        return format_tree(self.to_node())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


#: 步骤：一个 Step，或一串 Step（含嵌套列表，`coerce_steps` 负责展平）
StepLike: TypeAlias = Union[Step, Sequence[Step]]


def coerce_steps(*items: StepLike) -> List[Step]:
    '''把「Step / Step 列表 / 嵌套列表」展平成一维的 Step 列表。

    这样 `.Then(a, b)`、`.Then([a, b])`、`.Then(workflow)` 都能用。
    '''
    steps: List[Step] = []
    for item in items:
        if isinstance(item, Step):
            steps.append(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            steps.extend(coerce_steps(*item))
        else:
            raise TypeError(
                f"期望 Step 或 Step 列表，得到 {type(item).__name__}: {short(item)}"
            )
    return steps


class SkipStep(Step):
    '''不执行任何操作。相当于Python中的pass'''

    name = "skip"

    def execute(self, context: Context) -> None:
        pass

    def describe(self) -> str:
        return "Skip"
