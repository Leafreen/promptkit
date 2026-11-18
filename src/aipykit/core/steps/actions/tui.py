﻿# core/steps/actions/tui.py
'''TUI 的 DSL 标记步骤。

    from aipykit.core import ChatLoop, EnableTUI

    ChatLoop(
        EnableTUI(),                      # 整个会话用 TUI
        If(Var("usr_input") == "help").Then(Echo("...")),
    )

`EnableTUI()` 和 `TimeStamp()` / `ProactiveAlarms()` 是同一类写法：写进
`ChatLoop(...)` 的位置参数后，工厂会把它**从每轮循环体里摘掉**，改由整个
循环负责起停 —— TUI 是「整段会话」的状态，不是每轮动作。重复执行会重复
挂事件 handler，所以不能当普通步骤跑。

`fullscreen=True` 更进一步：`Workflow.run()` 会把它提升到**整个工作流**
外层，于是 `ChatLoop` 前面的开场白、后面的收尾也都在同一个界面里跑，
模型回复统一显示在对话区。

真正干活的是 `aipykit.tui.start_tui()`，core 只在需要时才导入它：
没装 `aipykit[tui]` 不影响其他功能，只有显式写了 `EnableTUI()` 才会报错。
'''
from __future__ import annotations

from typing import TYPE_CHECKING

from aipykit.core.errors import WorkflowError
from aipykit.core.steps.interface import Step

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 context
    from aipykit.core.context import Context

__all__ = ["EnableTUI", "TuiUnavailableError", "TUI_ACTIVE_KEY", "wants_fullscreen"]

#: 全屏 TUI 正在运行时挂到 Context 上的标记。
#: 全屏应用可能由外层 `Workflow.run()` 启动（这样 ChatLoop 前后的步骤
#: 也在界面里跑），内层 ChatLoop 看到它就不用再启动一次。
TUI_ACTIVE_KEY = "_tui_active"


def wants_fullscreen(step: object) -> bool:
    '''这棵步骤树里有没有「要全屏」的 ChatLoop。

    沿着 `children()` 往下找：`EnableTUI(fullscreen=True)` 写在 ChatLoop
    里，可能又被包在 If / 嵌套 Workflow 里，所以不能只看第一层。
    '''
    stack = [step]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "requests_fullscreen", False):
            return True
        children = getattr(current, "children", None)
        if callable(children):
            try:
                stack.extend(children())
            except Exception:  # noqa: BLE001 - 遍历不该影响主流程
                continue
    return False


class TuiUnavailableError(WorkflowError, ImportError):
    '''写了 `EnableTUI()`，但当前环境跑不了 TUI。

    两种情况都抛这个异常：

    - 没装可选依赖（rich / prompt_toolkit）
    - stdin / stdout 不是终端（CI、重定向、管道）

    刻意不做静默降级：显式要求 TUI 却悄悄退回普通控制台，调用方会以为
    界面已经生效，排查起来更费劲。CI 里要跑带 `EnableTUI()` 的脚本，
    请把它去掉。
    '''


class EnableTUI(Step):
    '''「这个 ChatLoop 要跑 TUI」的标记步骤。

    它本身不执行任何东西 —— `ChatLoop()` 工厂会把它摘出来，交给
    `ChatLoopWorkflow.execute` 在进循环前装好、循环结束后还原。
    '''

    name = "enable_tui"

    def __init__(self, fullscreen: bool = False) -> None:
        '''`fullscreen=False`：增强 REPL（默认）；`True`：Textual 全屏应用。'''
        self.fullscreen = fullscreen

    def execute(self, context: Context) -> None:
        raise WorkflowError(
            "EnableTUI() 只能作为 ChatLoop(...) 的位置参数使用："
            "TUI 的起停由整个循环负责，不能当普通步骤执行"
        )

    def describe(self) -> str:
        return f"EnableTUI(fullscreen={self.fullscreen})"
