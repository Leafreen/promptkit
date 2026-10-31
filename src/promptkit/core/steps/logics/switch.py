# core/steps/switch.py
'''switch-case 分支。'''
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.steps.logics.expr_sys.expr import Expr, as_expr
from promptkit.core.steps.interface import (
    Step,
    StepLike,
    coerce_steps,
    execute_step,
)
from promptkit.core.display_sys.tree import Node


class Case:
    '''一个 case：一组待匹配的值 + 命中的步骤 + 是否往下穿透。'''

    def __init__(
        self,
        values: Iterable[Expr],
        body: Iterable[Step] = (),
        fallthrough: bool = False,
    ):
        self.values: Tuple[Expr, ...] = tuple(values)
        self.body: Tuple[Step, ...] = tuple(body)
        self.fallthrough = fallthrough

    def __repr__(self) -> str:
        values = ", ".join(repr(v) for v in self.values)
        return f"<Case {values} fallthrough={self.fallthrough}>"


@dataclass(eq=False)
class SwitchStep(Step):
    '''按一个值选择分支，对应 C 的 switch-case。

        Switch(Var("grade"))
        .Case("A").Then(Echo("优秀"))
        .Case("B", "C").Then(Echo("及格"))
        .Default(Echo("未知"))

    和 C 的两点区别，都是故意的：

    1. **默认不穿透**。C 里忘了写 break 就会掉进下一个 case，这是经典 bug。
       这里每个 case 只执行自己的步骤；确实想穿透就写 `Case(1, fallthrough=True)`。
    2. **Switch 不吞 `Break()`**。C 里 switch 里的 break 是跳出 switch，
       于是嵌在循环里就成了陷阱。这里 `Break()` 永远作用于最近的**循环**，
       而 case 本来就不需要 break。
    '''

    subject: Expr
    cases: Tuple[Case, ...] = ()
    default_body: Tuple[Step, ...] = ()
    name: str = "switch"

    def __post_init__(self) -> None:
        self.subject = as_expr(self.subject)
        self.cases = tuple(self.cases)
        self.default_body = tuple(self.default_body)

    # ---------------- 链式构建 ----------------
    def Case(  # noqa: N802 - 跟随 DSL 命名
        self, *values: object, fallthrough: bool = False
    ) -> "SwitchStep":
        '''匹配一个（或多个）值。'''
        if not values:
            raise TypeError("Case() 至少要有一个值")
        case = Case([as_expr(v) for v in values], (), fallthrough)
        return replace(self, cases=self.cases + (case,))

    def Then(self, *steps: StepLike) -> "SwitchStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置最后一个 case 要执行的步骤。'''
        if not self.cases:
            raise ValueError("请先写 Case(...) 再写 Then(...)")
        if not steps:
            raise ValueError("Then() 至少要有一个步骤")
        last = self.cases[-1]
        updated = Case(last.values, coerce_steps(*steps), last.fallthrough)
        return replace(self, cases=self.cases[:-1] + (updated,))

    def Default(self, *steps: StepLike) -> "SwitchStep":  # noqa: N802 - 跟随 DSL 命名
        '''所有 case 都不匹配时执行的步骤（可选）。'''
        return replace(self, default_body=tuple(coerce_steps(*steps)))

    # ---------------- 执行 ----------------
    def execute(self, context: Context) -> None:
        # 和 C 一样：被判断的表达式只求值一次
        value = self.subject.eval(context)

        matched = None
        for index, case in enumerate(self.cases):
            if any(value == candidate.eval(context) for candidate in case.values):
                matched = index
                break

        if matched is None:
            context.trace.append(f"{self.name}:default")
            for step in self.default_body:
                execute_step(step, context)
            return

        index = matched
        while index < len(self.cases):
            case = self.cases[index]
            context.trace.append(f"{self.name}:case{index + 1}")
            for step in case.body:
                execute_step(step, context)
            if not case.fallthrough:
                return
            index += 1

        # 一路穿透到末尾 -> 继续执行 default（C 的语义）
        context.trace.append(f"{self.name}:default")
        for step in self.default_body:
            execute_step(step, context)

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        return f"Switch({self.subject!r})"

    def to_node(self) -> Node:
        children = []
        for case in self.cases:
            label = "case " + ", ".join(repr(v) for v in case.values)
            if case.fallthrough:
                label += "  (fallthrough)"
            children.append(Node(label, [step.to_node() for step in case.body]))
        if self.default_body:
            children.append(
                Node("default", [step.to_node() for step in self.default_body])
            )
        return Node(self.describe(), children)

    def __repr__(self) -> str:
        return f"<SwitchStep cases={len(self.cases)} default={len(self.default_body)}>"


def Switch(subject: object) -> SwitchStep:  # noqa: N802 - 跟随 DSL 命名
    '''Switch(值).Case(1).Then(...).Case(2).Then(...).Default(...)。

    值可以是字面量，也可以是表达式（比如 Var("x")），运行时才求值。
    '''
    return SwitchStep(subject=as_expr(subject))
