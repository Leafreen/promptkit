# core/steps/logics/expr_sys/expr.py
'''惰性表达式：在工作流运行时针对 Context 求值。

对 Expr 做比较 / 运算不会立刻计算，而是生成一个新的 Expr：

    Var("state") != "结束"        # -> BinaryExpr，运行时才求值

Python 不能重载 `and` / `or` / `not`，所以用 `&` / `|` / `~` 代替。
'''
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Dict

from promptkit.core.errors import UndefinedVariable

if TYPE_CHECKING:  # 只用于类型标注：运行时只对 context 做鸭子类型调用
    from promptkit.core.context import Context

def _add(left: Any, right: Any) -> Any:
    '''加法。沿用 Python 的语义（str + int 会报错），只是把报错说得更清楚。'''
    try:
        return operator.add(left, right)
    except TypeError:
        raise TypeError(
            f"不能用 + 把 {type(left).__name__} 和 {type(right).__name__} 相加。"
            '要在文本里插入变量，用 Echo("前缀 {{x}}") 或 Echo("前缀 ", Var("x"))；'
            "要拼数字就先用 Call(str, ...) 转成字符串"
        ) from None


# 二元运算符表
_BINOPS: Dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "+": _add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    # and / or 只对 bool 生效，语义上等价于 Python 的 and / or
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
    # 成员判断。`in` 是 Python 关键字，写不成方法名，所以叫 `in_`
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}


class Expr:
    '''所有表达式的基类。'''

    # 重载了 __eq__ 之后默认会变成不可哈希，这里显式恢复
    __hash__ = object.__hash__

    def eval(self, context: Context) -> Any:
        raise NotImplementedError(f"{type(self).__name__} 没有实现 eval()")

    # ---------------- 比较 ----------------
    def __eq__(self, other: Any) -> "BinaryExpr":  # type: ignore[override]
        return BinaryExpr("==", self, other)

    def __ne__(self, other: Any) -> "BinaryExpr":  # type: ignore[override]
        return BinaryExpr("!=", self, other)

    def __lt__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("<", self, other)

    def __le__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("<=", self, other)

    def __gt__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(">", self, other)

    def __ge__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr(">=", self, other)

    # ---------------- 算术 ----------------
    def __add__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("+", self, other)

    def __sub__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("-", self, other)

    def __mul__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("*", self, other)

    def __truediv__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("/", self, other)

    def __mod__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("%", self, other)

    # 反向算术：让 `"前缀：" + Var("x")` 这种写法也能用
    # （str.__add__ 对 Expr 返回 NotImplemented，Python 会退回来找 __radd__）
    def __radd__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("+", other, self)

    def __rsub__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("-", other, self)

    def __rmul__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("*", other, self)

    def __rtruediv__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("/", other, self)

    def __rmod__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("%", other, self)

    # ---------------- 逻辑（替代 and / or / not）----------------
    def __and__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("and", self, other)

    def __or__(self, other: Any) -> "BinaryExpr":
        return BinaryExpr("or", self, other)

    def __invert__(self) -> "UnaryExpr":
        return UnaryExpr("not", self)

    # ---------------- 钻进对象里 ----------------
    def __getitem__(self, key: Any) -> "ItemExpr":
        '''`Var("xs")[0]` / `Var("d")["k"]` / `Var("xs")[1:3]`。'''
        return ItemExpr(self, key)

    def attr(self, name: str) -> "AttrExpr":
        '''取属性。Python 不允许重载 `.`，所以只能写成 `.attr("name")`。'''
        return AttrExpr(self, name)

    def method(self, name: str, *args: Any, **kwargs: Any) -> "CallExpr":
        '''调用方法：`Var("s").method("replace", "a", "b")`。'''
        return CallExpr(AttrExpr(self, name), *args, **kwargs)

    # ---------------- 成员判断 ----------------
    def in_(self, other: Any) -> "BinaryExpr":
        '''`Var("x").in_(["exit", "quit"])` —— 等价于 `x in [...]`。

        `in` 是 Python 关键字，没法当方法名，所以带下划线。
        右边可以是列表 / 集合 / 字符串 / 任何表达式。
        '''
        return BinaryExpr("in", self, other)

    def not_in(self, other: Any) -> "BinaryExpr":
        '''`Var("x").not_in([...])`。'''
        return BinaryExpr("not in", self, other)

    def __iter__(self):
        # 有了 __getitem__ 之后，Python 会退回到「按下标迭代」的老协议上，
        # 而 Expr 永远不会抛 IndexError，那就是死循环。这里显式堵掉。
        raise TypeError(
            "Expr 不能被迭代：请用 For('x', ...) 遍历，或 Call(iter, expr)"
        )

    def __bool__(self) -> bool:
        raise TypeError(
            "Expr 不能用 Python 的 and/or/not 或 if 判断；"
            "请改用 & | ~，或用 Condition(...) 包起来"
        )


def as_expr(value: Any) -> Expr:
    '''把普通值包装成 Expr；已经是 Expr 就原样返回。'''
    if isinstance(value, Expr):
        return value
    # 延迟导入，避免 core.expr <-> core.steps 的循环依赖。
    # VarDecl 既是 Expr 又是 Step，上面那一句已经把它拦下来了。
    from promptkit.core.steps.interface import Step

    if isinstance(value, Step):
        raise TypeError(
            f"{value!r} 是一个步骤，不能当作值来用；"
            "要执行它请直接放进 Workflow.add(...) / .Then(...) / .Do(...)"
        )
    return Const(value)


class Const(Expr):
    '''字面量。'''

    def __init__(self, value: Any):
        self.value = value

    def eval(self, context: Context) -> Any:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)


class VarExpr(Expr):
    '''读取 context 里的一个变量。

    支持点路径：`Var("intent.label")`、`Var("hits.0.text")`。
    精确同名优先——真的有个变量叫 "a.b" 也不会被拆开。
    '''

    def __init__(self, var: str):
        if not isinstance(var, str) or not var:
            raise TypeError(f"变量名必须是非空字符串，得到 {var!r}")
        self.var = var

    def eval(self, context: Context) -> Any:
        if self.var not in context and "." not in self.var:
            raise UndefinedVariable(
                f"变量 {self.var!r} 未定义：请先用 NewVar({self.var!r}) 声明"
            )
        from promptkit.prompt import resolve_variable

        try:
            return resolve_variable(self.var, context.to_dict())
        except KeyError as exc:
            raise UndefinedVariable(
                f"{exc}。变量要先声明：NewVar({self.var!r})"
            ) from None

    def __repr__(self) -> str:
        return f"Var({self.var!r})"


class BinaryExpr(Expr):
    '''二元运算。'''

    def __init__(self, op: str, left: Any, right: Any):
        if op not in _BINOPS:
            raise ValueError(f"不支持的运算符：{op!r}")
        self.op = op
        self.left = as_expr(left)
        self.right = as_expr(right)

    def eval(self, context: Context) -> Any:
        return _BINOPS[self.op](self.left.eval(context), self.right.eval(context))

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


class UnaryExpr(Expr):
    '''一元运算（目前只有 not）。'''

    def __init__(self, op: str, operand: Any):
        if op != "not":
            raise ValueError(f"不支持的一元运算符：{op!r}")
        self.op = op
        self.operand = as_expr(operand)

    def eval(self, context: Context) -> Any:
        return not self.operand.eval(context)

    def __repr__(self) -> str:
        return f"({self.op} {self.operand!r})"


class AgentReplyExpr(Expr):
    '''先执行某个步骤，再读取它的输出。

    由 `SomeStep(...).agent_reply()` 生成，用在 Var(val=...) 里：

        Set("user_answer", AskQuestion(job=job).agent_reply())

    求值时才真正调用模型，所以在 While 循环里每轮都会重新执行。
    '''

    def __init__(self, step: Any, key: str):
        if not hasattr(step, "execute"):
            raise TypeError("AgentReplyExpr 需要一个可执行的步骤")
        self.step = step
        self.key = key

    def eval(self, context: Context) -> Any:
        self.step.execute(context)
        if self.key not in context:
            raise UndefinedVariable(
                f"{self.step!r} 执行后没有写入变量 {self.key!r}。"
                "函数体里要么用 Return(...) 明确返回值，"
                f"要么保证某个步骤把结果写进了 {self.key!r}"
            )
        return context[self.key]

    def __repr__(self) -> str:
        return f"{self.step!r}.agent_reply()"


class CallExpr(Expr):
    '''调用一个函数，参数本身也可以是表达式：

        Call(add, Var("a"), Var("b"))       # 等价于 add(a, b)，运行时求值
        Call(len, Var("text"))
        Call(Var("callback"), 1)            # 函数本身也可以来自变量
    '''

    def __init__(self, func: Any, *args: Any, **kwargs: Any):
        if not callable(func) and not isinstance(func, Expr):
            raise TypeError(
                f"Call 的第一个参数必须是可调用对象或表达式，得到 {func!r}"
            )
        # 统一包成 Expr：普通函数是常量，Var("f") / expr.attr("m") 则运行时求值
        self.func = as_expr(func)
        self.args = [as_expr(a) for a in args]
        self.kwargs = {k: as_expr(v) for k, v in kwargs.items()}

    def eval(self, context: Context) -> Any:
        func = self.func.eval(context)
        if not callable(func):
            raise TypeError(f"{func!r} 不是可调用的")
        return func(
            *(a.eval(context) for a in self.args),
            **{k: v.eval(context) for k, v in self.kwargs.items()},
        )

    def __repr__(self) -> str:
        func = self.func.value if isinstance(self.func, Const) else self.func
        name = getattr(func, "__name__", repr(func))
        parts = [name] + [repr(a) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"Call({', '.join(parts)})"


class ItemExpr(Expr):
    '''下标 / 切片：`Var("xs")[0]`、`Var("d")["k"]`、`Var("xs")[1:3]`。'''

    def __init__(self, target: Any, key: Any):
        self.target = as_expr(target)
        self.key = as_expr(key)

    def eval(self, context: Context) -> Any:
        return self.target.eval(context)[self.key.eval(context)]

    def __repr__(self) -> str:
        return f"{self.target!r}[{self.key!r}]"


class AttrExpr(Expr):
    '''取属性：`Var("person").attr("name")`。

    Python 不能重载 `.`，所以只能有这一种写法。
    '''

    def __init__(self, target: Any, name: str):
        if not isinstance(name, str) or not name:
            raise TypeError(f"属性名必须是非空字符串，得到 {name!r}")
        self.target = as_expr(target)
        self.name = name

    def eval(self, context: Context) -> Any:
        return getattr(self.target.eval(context), self.name)

    def __repr__(self) -> str:
        return f"{self.target!r}.{self.name}"


class TemplateExpr(Expr):
    '''字符串模板：`{{name}}` 在求值时从 Context 取值。

        Echo("情感分数：{{score}}")     # 字符串里带 {{}} 就会被替换

    只认 `{{key}}`，不认 `$key`：普通文本里的 `$` 太容易误伤（例如 "价格 $100"）。
    '''

    def __init__(self, text: str):
        self.text = text

    def eval(self, context: Context) -> str:
        from promptkit.prompt import render_text

        return render_text(self.text, context.to_dict(), allow_dollar=False)

    def __repr__(self) -> str:
        return repr(self.text)


def Call(func: Callable, *args: Any, **kwargs: Any) -> CallExpr:  # noqa: N802
    '''把普通 Python 函数调用变成表达式：`Call(add, Var("a"), Var("b"))`。'''
    return CallExpr(func, *args, **kwargs)


def Len(value: Any) -> CallExpr:  # noqa: N802
    '''长度：`Len(Var("items"))`，等价于 `Call(len, ...)`。

    不能做成 `len(expr)`：Python 的 len() 会立刻调用 __len__，
    而表达式必须等到运行时才能求值。
    '''
    return CallExpr(len, value)
