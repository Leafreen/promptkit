'''逻辑类步骤：控制流、函数、跳转、变量，以及配套的条件系统。

不调用任何外部能力，只是「怎么算、往哪走」：

    from promptkit.core.steps.logics import If, ForEach, NewVar, Var, Function
'''
from promptkit.core.steps.logics.conditions import (
    And,
    BoundCondition,
    Condition,
    ConditionLike,
    Not,
    Or,
    When,
    WhenExpr,
    as_condition,
)
from promptkit.core.steps.logics.expr_sys import (
    AgentReplyExpr,
    AttrExpr,
    BinaryExpr,
    Call,
    CallExpr,
    Const,
    Expr,
    ItemExpr,
    Len,
    Number,
    Str,
    UnaryExpr,
    VarExpr,
    VarType,
    as_expr,
    var_type_name,
)
from promptkit.core.steps.logics.function import (
    BoundStep,
    Function,
    FunctionDef,
    FunctionStep,
    Return,
    ReturnStep,
)
from promptkit.core.steps.logics.goto import Goto, GotoStep, Label, LabelStep
from promptkit.core.steps.logics.if_step import If, IfStep
from promptkit.core.steps.logics.loop import (
    Break,
    Continue,
    DoWhile,
    DoWhileStep,
    For,
    ForEach,
    ForEachStep,
    ForStep,
)
from promptkit.core.steps.logics.switch import Case, Switch, SwitchStep
from promptkit.core.steps.logics.var import (
    CompoundAssignStep,
    Dec,
    Inc,
    NewVar,
    Set,
    SetItemStep,
    SetStep,
    Var,
    VarDecl,
)
from promptkit.core.steps.logics.while_step import While, WhileStep

__all__ = [
    # conditions.py
    "Condition",
    "BoundCondition",
    "ConditionLike",
    "as_condition",
    "And",
    "Or",
    "Not",
    "When",
    "WhenExpr",
    # expr_sys
    "Expr",
    "Const",
    "VarExpr",
    "BinaryExpr",
    "UnaryExpr",
    "AgentReplyExpr",
    "Call",
    "CallExpr",
    "ItemExpr",
    "AttrExpr",
    "Len",
    "as_expr",
    "VarType",
    "Number",
    "Str",
    "var_type_name",
    # var.py
    "Var",
    "NewVar",
    "Set",
    "VarDecl",
    "SetStep",
    "SetItemStep",
    "Inc",
    "Dec",
    "CompoundAssignStep",
    # if / loops / switch / goto
    "If",
    "IfStep",
    "While",
    "WhileStep",
    "For",
    "ForStep",
    "ForEach",
    "ForEachStep",
    "DoWhile",
    "DoWhileStep",
    "Switch",
    "SwitchStep",
    "Case",
    "Break",
    "Continue",
    "Label",
    "LabelStep",
    "Goto",
    "GotoStep",
    # function.py
    "Function",
    "FunctionDef",
    "BoundStep",
    "FunctionStep",
    "Return",
    "ReturnStep",
]
