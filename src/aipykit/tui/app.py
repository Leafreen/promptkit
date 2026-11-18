﻿# aipykit/tui/app.py
'''增强版终端对话界面：prompt_toolkit 管输入，rich 管状态。

分工是刻意的：

- `rich.Live`（transient）只放**一行状态**：思考中 / 正在跑哪个工具 /
  token 用量。整段对话不塞进 Live —— 那样滚不动、复制不了，长回答还会
  每帧重排 Markdown。
- 正文由 `ChatLoop` 里的 `Echo` 打印，直接落进终端 scrollback。
- 每次读输入前先 `Live.stop()`，避免输入框和状态条抢终端。

入口是 `start_tui()`，由 `ChatLoopWorkflow` 调用，用户只写 `EnableTUI()`。
'''
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from aipykit.core.display_sys.render import get_console
from aipykit.core.event_sys import EVENT_BUS_KEY, EventBus, EventType
from aipykit.core.steps.actions.interactive import INPUT_FN_KEY
from aipykit.core.steps.actions.llm import LLM_STREAM_KEY
from aipykit.core.steps.actions.tui import TuiUnavailableError

if TYPE_CHECKING:  # 只用于类型标注，运行时不导入
    from aipykit.core.context import Context
    from aipykit.core.event_sys import Event
    from rich.console import Console

try:  # 可选依赖：只有在真的启用 TUI 时才要求
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - 依赖齐全时不会走到
    _IMPORT_ERROR = exc

__all__ = ["TuiSession", "start_tui"]

#: 输入历史放用户目录，和具体项目无关
HISTORY_PATH = Path.home() / ".aipykit_history"


def _require_environment() -> None:
    '''检查「依赖齐全 + 是交互式终端」，缺一个就显式失败。'''
    if _IMPORT_ERROR is not None:
        raise TuiUnavailableError(
            "EnableTUI() 需要可选依赖 rich 和 prompt_toolkit："
            'pip install "aipykit[tui]"'
        ) from _IMPORT_ERROR
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise TuiUnavailableError(
            "EnableTUI() 需要交互式终端，但当前 stdin / stdout 不是 TTY"
            "（CI、重定向或管道）。请去掉 EnableTUI() 再运行。"
        )


def _tokens(value: int) -> str:
    '''token 数折成 1.2k 这种短写法。'''
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


class TuiSession:
    '''一次 TUI 启用的状态：事件 handler + 输入函数，退出时还原。

    `stop()` 可以重复调用 —— `ChatLoopWorkflow` 在 finally 里调它，
    异常路径也不会漏掉还原。
    '''

    def __init__(self, *, console: "Console") -> None:
        self._live = Live(
            "", console=console, refresh_per_second=10, transient=True
        )
        self._prompt = PromptSession(
            history=FileHistory(str(HISTORY_PATH))
        )
        self._live_on = False
        self._bus: Optional[EventBus] = None
        self._context: Optional["Context"] = None
        self._handler: Optional[Callable[["Event"], None]] = None
        self._previous_reader: Any = None
        self._previous_stream: Any = None
        self._thinking = 0
        self._streaming = ""
        self._tool = ""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._stopped = False

    # ---------------- 装配 / 还原 ----------------
    def attach(self, context: "Context") -> None:
        '''挂事件 handler，并把自己的输入函数注入 Context。'''
        bus = context.get(EVENT_BUS_KEY)
        if bus is None:
            # 默认 Verbosity.NORMAL 下 run() 不会建总线；TUI 得自己保证有人听
            bus = EventBus()
            context[EVENT_BUS_KEY] = bus
        self._bus = bus
        self._handler = self.on_event
        bus.on(self._handler)
        self._context = context
        self._previous_reader = context.get(INPUT_FN_KEY)
        context[INPUT_FN_KEY] = self.read
        # 让循环里的 LLMStep 走流式，这样状态条能逐字更新
        self._previous_stream = context.get(LLM_STREAM_KEY)
        context[LLM_STREAM_KEY] = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._detach()
        if self._bus is not None and self._handler is not None:
            self._bus.off(self._handler)
        if self._context is not None:
            self._context[INPUT_FN_KEY] = self._previous_reader
            self._context[LLM_STREAM_KEY] = self._previous_stream

    # ---------------- 输入 ----------------
    def read(self, prompt: str) -> str:
        '''`Read` 用的输入函数，形状和内置 `input()` 完全一致。

        EOF / Ctrl+C 原样抛出，交给 `Read` 的 `on_eof` 分流。
        '''
        self._detach()
        return self._prompt.prompt(prompt)

    # ---------------- 事件 ----------------
    def on_event(self, event: "Event") -> None:
        etype = event.type
        if etype == EventType.THINKING_DELTA:
            self._thinking += len(str(event.data.get("text") or ""))
        elif etype == EventType.MESSAGE_DELTA:
            self._streaming += str(event.data.get("text") or "")
        elif etype == EventType.MESSAGE_END:
            self._finish_turn()
            return
        elif etype == EventType.TOOL_CALL_START:
            self._tool = str(event.data.get("name") or "")
        elif etype == EventType.TOOL_RESULT:
            self._tool = ""
        elif etype == EventType.USAGE:
            self._prompt_tokens += int(event.data.get("prompt_tokens") or 0)
            self._completion_tokens += int(event.data.get("completion_tokens") or 0)
        elif etype == EventType.ERROR:
            self._finish_turn()
            return
        else:
            return
        self._render_status()

    def _finish_turn(self) -> None:
        '''一轮结束：收起状态条，把正文留给 `Echo` 打印。'''
        self._thinking = 0
        self._streaming = ""
        self._tool = ""
        self._detach()

    def _render_status(self) -> None:
        if not self._live_on:
            self._live.start(refresh=True)
            self._live_on = True
        self._live.update(
            Spinner("dots", text=Text(self._status_line(), style="dim")),
            refresh=True,
        )

    def _detach(self) -> None:
        if self._live_on:
            self._live.stop()
            self._live_on = False

    def _status_line(self) -> str:
        if self._tool:
            head = f" 运行工具 {self._tool}"
        elif self._streaming:
            head = f" 生成中（{len(self._streaming)} 字）"
        elif self._thinking:
            head = f" 思考中（{self._thinking} 字）"
        else:
            head = " 等待模型"
        if self._prompt_tokens or self._completion_tokens:
            head += (
                f"   ↑{_tokens(self._prompt_tokens)}"
                f" ↓{_tokens(self._completion_tokens)}"
            )
        return head


def start_tui(context: "Context") -> TuiSession:
    '''把 TUI 装到当前运行上；环境不满足时抛 `TuiUnavailableError`。'''
    _require_environment()
    session = TuiSession(console=get_console())
    session.attach(context)
    return session
