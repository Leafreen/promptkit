# core/steps/logics/conditions.py
'''If / While 用的条件。'''
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from promptkit.core.steps.logics.expr_sys.expr import Expr, as_expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context


class Condition:
    '''条件。两种用法：

    1) 直接包一个表达式（这一层通常是多余的，If/While 本来就能收 Expr）：

           Condition(Var("state") != "结束")

    2) 子类化并实现 evaluate()，让条件可以复用、有名字：

           class NeedRefund(Condition):
               name = "need_refund"
               def evaluate(self, context):
                   return context["intent"].label == "退款"
    '''

    name: str = "condition"
    _expr = None

    def __init__(self, expr: object = None):
        if isinstance(expr, Condition):
            raise TypeError("Condition 不能嵌套 Condition，直接传表达式即可")
        if expr is not None:
            self._expr = as_expr(expr)

    def evaluate(self, context: Context) -> bool:
        '''判断条件是否成立。'''
        if self._expr is None:
            raise NotImplementedError(f"{type(self).__name__} 需要实现 evaluate()")
        return bool(self._expr.eval(context))

    # 让 Condition 本身也能当普通函数用
    def __call__(self, context: Context) -> bool:
        return self.evaluate(context)

    def __repr__(self) -> str:
        # 包了表达式就把表达式显示出来，否则显示子类名
        if self._expr is not None:
            return f"Condition({self._expr!r})"
        return f"{type(self).__name__}()"


# If / While 能接受的写法：常量、表达式、函数、Condition 对象
# 这里在**赋值**位置用类型别名，`from __future__ import annotations` 管不到，
# 所以 Context 必须写成字符串前向引用（它只在 TYPE_CHECKING 下导入）
ConditionLike = Union[bool, Expr, "Callable[[Context], bool]", Condition]


class BoundCondition:
    '''归一化后的条件：可以直接调用，同时保留可读的文本。

    保留文本是为了 `print(wf)` 时能显示 `if (Var('input') % 15) == 0`，
    而不是一个看不出内容的 `<function <lambda>>`。
    '''

    __slots__ = ("_fn", "text")

    def __init__(self, fn: Callable[[Context], bool], text: str):
        self._fn = fn
        self.text = text

    def __call__(self, context: Context) -> bool:
        return self._fn(context)

    def __repr__(self) -> str:
        return self.text


def as_condition(condition: ConditionLike) -> BoundCondition:
    '''把各种写法统一成 `callable(context) -> bool`。'''
    # 已经归一化过就直接返回，保证 dataclasses.replace() 之后依然正确
    if isinstance(condition, BoundCondition):
        return condition
    if isinstance(condition, Condition):
        return BoundCondition(condition.evaluate, repr(condition))
    if isinstance(condition, Expr):
        return BoundCondition(
            lambda context: bool(condition.eval(context)), repr(condition)
        )
    if isinstance(condition, bool):
        return BoundCondition(lambda _context: condition, repr(condition))
    if callable(condition):
        return BoundCondition(condition, getattr(condition, "__name__", repr(condition)))
    raise TypeError(
        "条件必须是 bool、Expr、Condition 或 callable(context) -> bool，"
        f"得到 {type(condition).__name__}"
    )


class And(Condition):
    '''所有子条件都成立时成立。'''

    def __init__(self, *conditions: ConditionLike):
        self.conditions = [as_condition(c) for c in conditions]
        self.name = "and"

    def evaluate(self, context: Context) -> bool:
        return all(c(context) for c in self.conditions)

    def __repr__(self) -> str:
        return "And(" + ", ".join(repr(c) for c in self.conditions) + ")"


class Or(Condition):
    '''任意一个子条件成立时成立。'''

    def __init__(self, *conditions: ConditionLike):
        self.conditions = [as_condition(c) for c in conditions]
        self.name = "or"

    def evaluate(self, context: Context) -> bool:
        return any(c(context) for c in self.conditions)

    def __repr__(self) -> str:
        return "Or(" + ", ".join(repr(c) for c in self.conditions) + ")"


class Not(Condition):
    '''取反。'''

    def __init__(self, condition: ConditionLike):
        self.condition = as_condition(condition)
        self.name = "not"

    def evaluate(self, context: Context) -> bool:
        return not self.condition(context)

    def __repr__(self) -> str:
        return f"Not({self.condition!r})"


class WhenExpr(Expr):
    '''C 的三元运算符 `cond ? a : b`。

        Set("label", When(Var("n") % 2 == 0, "偶数", "奇数"))

    和 C 一样是**短路**的：只求值被选中的那一边。
    '''

    def __init__(self, condition: ConditionLike, then_value: object, else_value: object):
        self.condition = as_condition(condition)
        self.then_value = as_expr(then_value)
        self.else_value = as_expr(else_value)

    def eval(self, context: Context) -> object:
        chosen = self.then_value if self.condition(context) else self.else_value
        return chosen.eval(context)

    def __repr__(self) -> str:
        return f"When({self.condition!r}, {self.then_value!r}, {self.else_value!r})"


def When(  # noqa: N802
    condition: ConditionLike, then_value: object, else_value: object
) -> WhenExpr:
    '''When(条件, 成立时的值, 不成立时的值)：C 的 `cond ? a : b`。'''
    return WhenExpr(condition, then_value, else_value)
