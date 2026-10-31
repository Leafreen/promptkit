# core/steps/goto.py
'''goto：Label 标记位置，Goto 跳过去。

**只支持跳到工作流顶层的 Label**，原因见 README：
工作流是一棵树（While 的循环体、If 的分支各自是嵌套结构），
没法从外面「跳进」一个嵌套块；但可以随时「跳出去」。
所以这里把标签限制在顶层，语义清晰、实现也简单：

    wf.add(
        Label("retry"),
        DoSomething(),
        If(failed).Then(Goto("retry")),
        Label("done"),
        ...
    )

循环体内的跳转请用 Continue / Break。
'''
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.errors import GotoSignal
from promptkit.core.steps.interface import Step


class LabelStep(Step):
    '''标记一个跳转目标。执行时什么都不做。'''

    name = "label"

    def __init__(self, label: str):
        if not isinstance(label, str) or not label:
            raise TypeError(f"标签名必须是非空字符串，得到 {label!r}")
        self.label = label

    def execute(self, context: Context) -> None:
        context.trace.append(f"label:{self.label}")

    def describe(self) -> str:
        return f"Label({self.label!r})"

    def __repr__(self) -> str:
        return f"Label({self.label!r})"


class GotoStep(Step):
    '''跳到同一条工作流里的某个 Label。'''

    name = "goto"

    def __init__(self, label: str):
        if not isinstance(label, str) or not label:
            raise TypeError(f"标签名必须是非空字符串，得到 {label!r}")
        self.label = label

    def execute(self, context: Context) -> None:
        context.trace.append(f"goto:{self.label}")
        raise GotoSignal(self.label)

    def describe(self) -> str:
        return f"Goto({self.label!r})"

    def __repr__(self) -> str:
        return f"Goto({self.label!r})"


def Label(label: str) -> LabelStep:  # noqa: N802 - 跟随 DSL 命名
    '''Label("name")：标记一个 goto 目标。'''
    return LabelStep(label)


def Goto(label: str) -> GotoStep:  # noqa: N802 - 跟随 DSL 命名
    '''Goto("name")：跳到同一条工作流顶层的 Label("name")。'''
    return GotoStep(label)


def collect_labels(steps: Iterable[object]) -> Dict[str, int]:
    '''收集顶层步骤里的标签 -> 下标。'''
    return {
        step.label: index
        for index, step in enumerate(steps)
        if isinstance(step, LabelStep)
    }
