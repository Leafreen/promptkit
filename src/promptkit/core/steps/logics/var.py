# core/steps/logics/var.py
'''变量：声明（NewVar / Number / Str）、赋值（Set / Var(val=...)）、
复合赋值（Inc / Dec）、引用（Var("x")）。'''
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Optional, Union, cast, overload

from promptkit.core._missing import MISSING
from promptkit.core.errors import UndefinedVariable
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import (
    AttrExpr,
    Expr,
    ItemExpr,
    VarExpr,
    as_expr,
)
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

    def default_value(self, context: Context) -> object:
        '''算出声明时用的默认值。

        `default` 传表达式（如 `Var("a")`、`Var("n") + 1`）时在声明这一刻求值，
        和 `Set` 的取值语义保持一致；字面量原样返回。
        '''
        if self.default is MISSING:
            return None if self.var_type is None else self.var_type.default
        if isinstance(self.default, Expr):
            return self.default.eval(context)
        return self.default

    def execute(self, context: Context) -> None:
        # 变量已存在（例如 Workflow.run(n=...) 传进来）时保持原值，
        # 也就不用求值 default，避免表达式引用别的变量时白抛异常
        if self.var_type is not None:
            if self.var in context:
                context.declare(self.var, self.var_type)
            else:
                context.declare(self.var, self.var_type, self.default_value(context))
        elif self.var not in context:
            context.setdefault(self.var, self.default_value(context))
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
        container[self._key(context)] = self.value.eval(context) # type: ignore
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"{self.item!r} = {self.value!r}"

    def __repr__(self) -> str:
        return f"SetItem({self.item!r}, {self.value!r})"


class SetAttrStep(Step):
    '''给对象写一个属性：`Set(Var("cur").attr("next"), v)`。

    读用 `Var("cur").attr("next")`，写用 `Set(Var("cur").attr("next"), v)`，
    两边写法对称。

        Set(Var("node").attr("next"), Var("other"))
        Set(Var("person").attr("name"), "张三")

    和 `SetItemStep` 一样是**就地修改**对象。
    '''

    name = "set_attr"

    def __init__(self, attr: AttrExpr, value: object):
        if not isinstance(attr, AttrExpr):
            raise TypeError(f"SetAttrStep 需要一个属性表达式，得到 {attr!r}")
        self.attr = attr
        self.value = as_expr(value)

    def execute(self, context: Context) -> None:
        target = self.attr.target.eval(context)
        try:
            setattr(target, self.attr.name, self.value.eval(context))
        except AttributeError as exc:
            raise TypeError(
                f"不能给 {type(target).__name__} 设置属性 {self.attr.name!r}"
                f"（{self.attr.target!r} 求值成了 {target!r}）"
            ) from exc
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"{self.attr!r} = {self.value!r}"

    def __repr__(self) -> str:
        return f"SetAttr({self.attr!r}, {self.value!r})"


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
        NewVar("alias", default=Var("name"))  # default 也可以是表达式，声明时求值
    '''
    if not isinstance(name, str) or not name:
        raise TypeError(f"变量名必须是非空字符串，得到 {name!r}")
    if type is not None and not isinstance(type, VarType):
        raise TypeError(
            f"type 需要是 Number 或 Str，得到 {type!r}；"
            "也可以写成 Number('x') / Str('x')"
        )
    return VarDecl(name, var_type=type, default=default)


@overload
def Var(  # noqa: N802 - 跟随 DSL 命名
    name: Union[str, VarExpr],
    *,
    type: "Optional[type]" = None,  # noqa: A002
) -> VarExpr: ...


@overload
def Var(  # noqa: N802 - 跟随 DSL 命名
    *,
    val: object,
) -> SetStep: ...


def Var(  # noqa: N802
    name: object = None,
    *,
    val: object = MISSING,
    type: "Optional[type]" = None,  # noqa: A002 - 跟 DSL 命名保持一致
) -> Union[VarExpr, SetStep]:
    '''两种用法：

        Var("state")                    # 引用：一个惰性表达式
        Var(name="state", val=expr)     # 赋值：一个步骤

    也可以传 NewVar(...) / Number(...) 的句柄：Var(state)。

    `type=` 给引用加运行时类型标注（`Var("img", type=ImagePart)`），
    只在 `Prompt.render()` 解析多模态内容时校验；赋值形式不支持。
    '''
    if val is not MISSING:
        if type is not None:
            raise TypeError("赋值形式 Var(name, val=...) 不支持 type 标注")
        return SetStep(_var_name(name), val)
    if isinstance(name, VarExpr):
        if type is not None:
            raise TypeError("变量句柄已经声明过，不能再加 type 标注")
        return name
    return VarExpr(cast(str, name), type=type)


def Set(  # noqa: N802
    name: object, value: object
) -> Union[SetItemStep, SetAttrStep, SetStep]:
    '''给变量赋值。

        Set("state", "结束")                 # 普通变量
        Set("n", Var("n") + 1)
        Set(Var("d")["k"], v)               # 给容器写一个下标
        Set(Var("xs")[0], v)
        Set(Var("node").attr("next"), v)    # 给对象写一个属性
    '''
    if isinstance(name, ItemExpr):
        return SetItemStep(name, value)
    if isinstance(name, AttrExpr):
        return SetAttrStep(name, value)
    return SetStep(_var_name(name), value)
