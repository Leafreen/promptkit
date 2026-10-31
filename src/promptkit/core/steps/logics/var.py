# core/steps/logics/var.py
'''变量：声明（NewVar / Number / Str）、赋值（Set / Var(val=...)）、
复合赋值（Inc / Dec）、引用（Var("x")）。'''
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Optional, Union, cast

from promptkit.core._missing import MISSING
from promptkit.core.errors import UndefinedVariable
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr, ItemExpr, VarExpr, as_expr
from promptkit.core.steps.logics.expr_sys.vartypes import VarType

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context


class VarDecl(VarExpr, Step):
    '''声明一个变量，同时它本身就是一个可用的表达式句柄。

        state = NewVar("state", default="进行中")
        count = Number("count")          # 等价于 NewVar("count", type=Number)

        state != "结束"                  # 直接当表达式用
        wf.add(count)                    # 也可以直接放进工作流，表示「声明它」
    '''

    name = "new_var"

    def __init__(
        self,
        var: str,
        var_type: Optional[VarType] = None,
        default: object = MISSING,
    ):
        VarExpr.__init__(self, var)
        self.var_type = var_type
        self.default = default

    def default_value(self) -> object:
        '''没有显式给 default 时，用类型自带的默认值。'''
        if self.default is not MISSING:
            return self.default
        return None if self.var_type is None else self.var_type.default

    def execute(self, context: Context) -> None:
        if self.var_type is not None:
            context.declare(self.var, self.var_type, self.default_value())
        else:
            context.setdefault(self.var, self.default_value())
        context.trace.append(f"{self.name}:{self.var}")

    def __repr__(self) -> str:
        if self.var_type is None:
            return f"NewVar({self.var!r})"
        return f"{self.var_type.name}({self.var!r})"


class SetStep(Step):
    '''给变量赋值。值可以是字面量，也可以是表达式。

    如果变量声明了类型，赋值时会自动转换（例如字符串 "12" -> 数字 12）。
    '''

    name = "set"

    def __init__(self, var: str, value: object):
        if not isinstance(var, str) or not var:
            raise TypeError(f"变量名必须是非空字符串，得到 {var!r}")
        self.var = var
        self.value = as_expr(value)

    def execute(self, context: Context) -> None:
        # 类型转换发生在 Context.__setitem__ 里
        context[self.var] = self.value.eval(context)
        context.trace.append(f"{self.name}:{self.var}")

    def __repr__(self) -> str:
        return f"Set({self.var!r}, {self.value!r})"


class SetItemStep(Step):
    '''给容器写一个下标：`Set(Var("d")["k"], v)`、`Set(Var("xs")[0], v)`。

    读用 `Var("d")["k"]`（表达式），写用 `Set(Var("d")["k"], v)`（步骤），
    两边写法对称。

        Set(Var("info")["姓名"], Read(">>> "))       # dict
        Set(Var("xs")[0], 42)                        # list
        Set(Var("d")["a"]["b"], 1)                   # 嵌套

    注意这是**就地修改**容器；想每次拿到新对象就用
    `Set("d", Call(merge, Var("d"), "k", v))` 那种写法。
    '''

    name = "set_item"

    def __init__(self, item: ItemExpr, value: object):
        if not isinstance(item, ItemExpr):
            raise TypeError(f"SetItemStep 需要一个下标表达式，得到 {item!r}")
        self.item = item
        self.value = as_expr(value)

    def _key(self, context: Context) -> object:
        key = self.item.key
        return key.eval(context) if isinstance(key, Expr) else key

    def execute(self, context: Context) -> None:
        container = self.item.target.eval(context)
        if not hasattr(container, "__setitem__"):
            raise TypeError(
                f"不能给 {type(container).__name__} 赋下标值"
                f"（{self.item.target!r} 求值成了 {container!r}）"
            )
        container[self._key(context)] = self.value.eval(context)
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"{self.item!r} = {self.value!r}"

    def __repr__(self) -> str:
        return f"SetItem({self.item!r}, {self.value!r})"


def _var_name(name: object) -> str:
    '''既接受字符串，也接受 NewVar(...) / Number(...) 返回的句柄。'''
    if isinstance(name, VarExpr):
        return name.var
    return cast(str, name)


# 复合赋值用到的运算
_COMPOUND_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
}


class CompoundAssignStep(Step):
    '''复合赋值，对应 C 里的 `i++` / `i--` / `i += n`。

        Inc("i")            # i++
        Inc("i", 2)         # i += 2
        Dec("i")            # i--
    '''

    name = "compound"

    def __init__(self, var: str, by: object = 1, op: str = "+"):
        if not isinstance(var, str) or not var:
            raise TypeError(f"变量名必须是非空字符串，得到 {var!r}")
        if op not in _COMPOUND_OPS:
            raise ValueError(
                f"不支持的复合运算符 {op!r}，可选：{', '.join(_COMPOUND_OPS)}"
            )
        self.var = var
        self.by = as_expr(by)
        self.op = op

    def execute(self, context: Context) -> None:
        if self.var not in context:
            raise UndefinedVariable(
                f"变量 {self.var!r} 未定义：不能对它做 {self.op}="
            )
        result = _COMPOUND_OPS[self.op](context[self.var], self.by.eval(context))
        # 写入时按声明的类型转换（Number 变量会保持是数字）
        context[self.var] = result
        context.trace.append(f"{self.name}:{self.var}")

    def describe(self) -> str:
        return f"{self.var} {self.op}= {self.by!r}"

    def __repr__(self) -> str:
        return f"CompoundAssign({self.var!r}, {self.op!r}, {self.by!r})"


def Inc(name: object, by: object = 1) -> CompoundAssignStep:  # noqa: N802
    '''Inc("i") 相当于 i++；Inc("i", 2) 相当于 i += 2。'''
    return CompoundAssignStep(_var_name(name), by, "+")


def Dec(name: object, by: object = 1) -> CompoundAssignStep:  # noqa: N802
    '''Dec("i") 相当于 i--；Dec("i", 2) 相当于 i -= 2。'''
    return CompoundAssignStep(_var_name(name), by, "-")


def NewVar(  # noqa: N802
    name: str,
    type: Optional[VarType] = None,  # noqa: A002 - 跟 DSL 命名保持一致
    default: object = MISSING,
) -> VarDecl:
    '''声明一个变量。

        NewVar("state", default="进行中")     # 不限制类型
        NewVar("count", type=Number)          # 指定类型
        NewVar("name", type=Str, default="")  # 指定类型和默认值
    '''
    if not isinstance(name, str) or not name:
        raise TypeError(f"变量名必须是非空字符串，得到 {name!r}")
    if type is not None and not isinstance(type, VarType):
        raise TypeError(
            f"type 需要是 Number 或 Str，得到 {type!r}；"
            "也可以写成 Number('x') / Str('x')"
        )
    return VarDecl(name, var_type=type, default=default)


def Var(  # noqa: N802
    name: object = None, *, val: object = MISSING
) -> Union[VarExpr, SetStep]:
    '''两种用法：

        Var("state")                    # 引用：一个惰性表达式
        Var(name="state", val=expr)     # 赋值：一个步骤

    也可以传 NewVar(...) / Number(...) 的句柄：Var(state)。
    '''
    if val is not MISSING:
        return SetStep(_var_name(name), val)
    if isinstance(name, VarExpr):
        return name
    return VarExpr(cast(str, name))


def Set(name: object, value: object) -> Union[SetItemStep, SetStep]:  # noqa: N802
    '''给变量赋值。

        Set("state", "结束")                 # 普通变量
        Set("n", Var("n") + 1)
        Set(Var("d")["k"], v)               # 给容器写一个下标
        Set(Var("xs")[0], v)
    '''
    if isinstance(name, ItemExpr):
        return SetItemStep(name, value)
    return SetStep(_var_name(name), value)
