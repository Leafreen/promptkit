# core/steps/if_step.py
'''条件分支步骤：If(...).Then(...).ElseIf(...).Else(...)。'''
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Tuple

from promptkit.core.steps.interface import Step, coerce_steps, execute_step
from promptkit.core.steps.logics.conditions import ConditionLike, as_condition
from promptkit.core.display_sys.tree import Node

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context

# 一个分支：条件 + 条件成立时要执行的步骤
# 赋值位置的类型别名不受 future annotations 影响，Context 必须写成字符串
Branch = Tuple["Callable[[Context], bool]", Tuple[Step, ...]]


@dataclass
class IfStep(Step):
    '''按条件选择分支执行。

        If(c1).Then(s1).ElseIf(c2).Then(s2).Else(s3)

    - `Then()` 设置的是**最后一个分支**的步骤，所以可以接在 `ElseIf()` 后面
    - 所有分支都不成立时执行 `Else()`
    - Then / ElseIf / Else 都返回新的 IfStep（不修改原对象），可以自由嵌套
    '''

    branches: Tuple[Branch, ...]
    else_steps: Tuple[Step, ...] = ()
    name: str = "if"

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("IfStep 至少要有一个分支")
        # 归一化是幂等的，所以 dataclasses.replace() 之后再跑一次也没问题
        self.branches = tuple(
            (as_condition(condition), tuple(steps))
            for condition, steps in self.branches
        )
        self.else_steps = tuple(self.else_steps)

    def Then(self, *steps: Any) -> "IfStep":  # noqa: N802 - 跟随 DSL 命名
        '''设置最后一个分支的步骤。'''
        if not steps:
            raise ValueError("Then() 至少要有一个步骤")
        condition, _ = self.branches[-1]
        return replace(
            self,
            branches=self.branches[:-1] + ((condition, tuple(coerce_steps(*steps))),),
        )

    def ElseIf(self, condition: ConditionLike) -> "IfStep":  # noqa: N802
        '''前面都不成立时，再判断一个条件。'''
        return replace(self, branches=self.branches + ((condition, ()),))

    def Else(self, *steps: Any) -> "IfStep":  # noqa: N802 - 跟随 DSL 命名
        '''所有分支都不成立时执行的步骤（可选）。'''
        return replace(self, else_steps=tuple(coerce_steps(*steps)))

    def execute(self, context: Context) -> None:
        for index, (condition, steps) in enumerate(self.branches):
            if condition(context):
                label = "then" if index == 0 else f"elif{index}"
                context.trace.append(f"{self.name}:{label}")
                for step in steps:
                    execute_step(step, context)
                return
        context.trace.append(f"{self.name}:else")
        for step in self.else_steps:
            execute_step(step, context)

    def describe(self) -> str:
        return "If"

    def to_node(self) -> Node:
        '''每个分支画成一行，分支里的步骤挂在它下面。'''
        children = []
        for index, (condition, steps) in enumerate(self.branches):
            label = "if" if index == 0 else "elif"
            children.append(
                Node(f"{label} {condition!r}", [s.to_node() for s in steps])
            )
        if self.else_steps:
            children.append(Node("else", [s.to_node() for s in self.else_steps]))
        return Node(self.describe(), children)

    def __repr__(self) -> str:
        return f"<IfStep branches={len(self.branches)} else={len(self.else_steps)}>"


def If(condition: ConditionLike) -> IfStep:  # noqa: N802 - 跟随 DSL 命名
    '''构造条件分支：If(条件).Then(步骤...).ElseIf(...).Else(步骤...)。

    条件可以是 bool、Expr、Condition 实例或 callable(context) -> bool。
    '''
    return IfStep(branches=((as_condition(condition), ()),))
