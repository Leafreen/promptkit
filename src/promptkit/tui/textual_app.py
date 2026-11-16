# promptkit/tui/textual_app.py
'''Textual 全屏对话界面：状态栏、对话滚动区、底部输入框。

Textual 在主线程跑 asyncio，原有的同步 ChatLoop 则放进 worker 线程。
两边的桥接只有三条：

- `Read` 仍像调用 `input()` 一样阻塞，实际在等底部输入框提交的队列消息。
- 模型事件从 worker 线程经 `call_from_thread` 回到 UI，流式正文按 80ms 节流。
- `Echo` 的输出流暂时换成空设备，避免把最终回复再打印一遍；回复正文
  已由 `MESSAGE_DELTA` / `MESSAGE_END` 显示在滚动区里。

入口是 `run_fullscreen()`，由 `EnableTUI(fullscreen=True)` 触发。
'''
from __future__ import annotations

import queue
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from promptkit.core.display_sys.render import substitute_math
from promptkit.core.event_sys import EVENT_BUS_KEY, EventBus, EventType
from promptkit.core.steps.actions.interactive import INPUT_FN_KEY, OUTPUT_STREAM_KEY
from promptkit.core.steps.actions.llm import LLM_STREAM_KEY
from promptkit.core.steps.actions.tui import TUI_ACTIVE_KEY, TuiUnavailableError

if TYPE_CHECKING:
    from promptkit.core.context import Context
    from promptkit.core.event_sys import Event

try:  # 可选依赖：只有真正启动全屏 TUI 时才要求
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import VerticalScroll
    from textual.message import Message
    from textual.widgets import Markdown, Static, TextArea

    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - 依赖齐全时不会走到
    App = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

__all__ = ["TutorApp", "run_fullscreen"]

_EOF = object()

#: 首次进入全屏界面时显示在对话区顶部的「使用方法」，发出第一条消息后收起
_WELCOME = (
    "[b]使用方法[/b]\n\n"
    "• [b]Enter[/b] 发送　[b]Shift+Enter[/b] / [b]Alt+Enter[/b] 换行\n"
    "• 可直接粘贴多行文本或代码；输入框随行数变高，超过 10 行出现纵向滚动条\n"
    "• 等待模型回复时输入框会锁定，回复结束后自动恢复\n"
    "• [b]Ctrl+C[/b] 退出\n"
)


def _require_environment() -> None:
    '''检查「textual 已安装 + 是交互式终端」，不满足就显式失败。'''
    if _IMPORT_ERROR is not None:
        raise TuiUnavailableError(
            "EnableTUI(fullscreen=True) 需要可选依赖 textual："
            'pip install "aipykit[tui]"'
        ) from _IMPORT_ERROR
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise TuiUnavailableError(
            "EnableTUI(fullscreen=True) 需要交互式终端，但当前 stdin / stdout "
            "不是 TTY（CI、重定向或管道）。请去掉 EnableTUI() 再运行。"
        )


def _tokens(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


class _DevNull:
    '''Echo 的全屏输出去处；保留普通文本流所需的最小接口。'''

    encoding = "utf-8"

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


if App is not None:

    class ChatInput(TextArea):
        '''底部输入框：多行编辑、可粘贴代码，内容超出高度时出现纵向滚动条。

        - `Enter` 发送，`Shift+Enter`（或 `Alt+Enter`）换行；
        - 行数在 1～`MAX_LINES` 之间自动调整高度，再多就滚动；
        - 粘贴整段代码会原样进入输入框（TextArea 原生支持括号粘贴）。
        '''

        #: 最多长到几行，再往上就出现滚动条
        MAX_LINES = 10

        class Submitted(Message):
            '''用户按下 Enter：把整段文本交给 ChatLoop。'''

            def __init__(self, value: str) -> None:
                self.value = value
                super().__init__()

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            '''按文档行数自适应高度（含上下边框，所以 +2）。'''
            lines = min(max(self.document.line_count, 1), self.MAX_LINES)
            self.styles.height = lines + 2

        async def _on_key(self, event: Any) -> None:
            key = event.key
            if key in ("shift+enter", "alt+enter"):
                event.stop()
                event.prevent_default()
                self.insert("\n")
                return
            if key == "enter":
                event.stop()
                event.prevent_default()
                self.post_message(self.Submitted(self.text))
                return
            await super()._on_key(event)

    class TutorApp(App):
        '''一次 ChatLoop 的全屏界面；对话循环退出时应用也随之退出。'''

        CSS = '''
        Screen {
            layout: vertical;
        }
        #status {
            height: 1;
            padding: 0 1;
            background: $boost;
            color: $text;
        }
        #chat {
            height: 1fr;
            padding: 0 1;
        }
        #input {
            height: 3;
            min-height: 3;
            border: round $primary;
            padding: 0 1;
            scrollbar-size-vertical: 1;
        }
        #welcome {
            height: auto;
            margin: 1 0 0 0;
            padding: 0 1;
            color: $text-muted;
        }
        .message {
            height: auto;
            margin: 1 0 0 0;
            padding: 0 1;
        }
        .user {
            background: $boost;
        }
        '''

        # priority=True 才能盖过 App 自带的 `ctrl+c -> quit`（那条也是 priority），
        # 否则应用会立刻退出，而 worker 还阻塞在输入队列上，进程就此卡住。
        BINDINGS = [
            Binding("ctrl+c", "request_exit", "退出", show=False, priority=True)
        ]

        def __init__(
            self,
            context: "Context",
            run_loop: Callable[["Context"], None],
        ) -> None:
            super().__init__()
            self._workflow_context = context
            self._run_loop = run_loop
            self.error: Optional[BaseException] = None

            self._bus: Optional[EventBus] = None
            self._previous_reader: Any = None
            self._previous_output: Any = None
            self._previous_stream: Any = None

            self._inbox: queue.Queue[Any] = queue.Queue()
            self._waiting = False
            self._exit_requested = False
            self._input_prompt = ""
            self._current_reply: Optional[Markdown] = None
            self._welcome_dismissed = False
            self._reply = ""
            self._thinking = 0
            self._tool = ""
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._last_render = 0.0

        def compose(self) -> ComposeResult:
            yield Static("正在启动…", id="status")
            with VerticalScroll(id="chat"):
                yield Static(_WELCOME, id="welcome")
            yield ChatInput(
                placeholder="输入内容，Enter 发送，Shift+Enter 换行", id="input"
            )

        # ---------------- 装配 / 还原 ----------------
        def on_mount(self) -> None:
            bus = self._workflow_context.get(EVENT_BUS_KEY)
            if bus is None:
                bus = EventBus()
                self._workflow_context[EVENT_BUS_KEY] = bus
            self._bus = bus
            bus.on(self._on_event)

            self._previous_reader = self._workflow_context.get(INPUT_FN_KEY)
            self._previous_output = self._workflow_context.get(OUTPUT_STREAM_KEY)
            self._previous_stream = self._workflow_context.get(LLM_STREAM_KEY)
            self._workflow_context[INPUT_FN_KEY] = self._read
            self._workflow_context[OUTPUT_STREAM_KEY] = _DevNull()
            self._workflow_context[LLM_STREAM_KEY] = True

            self.query_one("#input", ChatInput).disabled = True
            self.run_worker(self._run, thread=True)

        def _run(self) -> None:
            '''worker 线程：跑原来的同步循环，保存异常供主线程重新抛出。'''
            try:
                self._run_loop(self._workflow_context)
            except BaseException as exc:
                self.error = exc
            finally:
                self._restore()
                try:
                    self.call_from_thread(self._finish)
                except Exception:  # noqa: BLE001 - 界面已关，没必要再收尾
                    pass

        def on_unmount(self) -> None:
            '''界面因任何原因关闭时，让阻塞中的输入知道自己该退出了。'''
            self._exit_requested = True

        def _restore(self) -> None:
            if self._bus is not None:
                self._bus.off(self._on_event)
            self._workflow_context[INPUT_FN_KEY] = self._previous_reader
            self._workflow_context[OUTPUT_STREAM_KEY] = self._previous_output
            self._workflow_context[LLM_STREAM_KEY] = self._previous_stream

        def _finish(self) -> None:
            self._exit_requested = True
            self._waiting = False
            self.query_one("#input", ChatInput).disabled = True
            self.exit()

        # ---------------- 输入桥 ----------------
        def _read(self, prompt: str) -> str:
            '''worker 线程里的同步输入函数，契约与内置 `input()` 一致。'''
            if self._exit_requested:
                raise EOFError
            requested = self.call_from_thread(self._request_input, prompt)
            if not requested:
                raise EOFError
            # 带超时地等：应用可能因为别的入口（比如 Textual 自己的退出键）
            # 关掉，而没人往队列里放哨兵。只等 get() 会把这个 worker 永久挂住。
            while True:
                try:
                    value = self._inbox.get(timeout=0.1)
                    break
                except queue.Empty:
                    if self._exit_requested:
                        raise EOFError
            if value is _EOF:
                raise EOFError
            return str(value)

        def _request_input(self, prompt: str) -> bool:
            '''UI 线程：开放输入框；返回 False 表示退出请求已捷足先登。'''
            if self._exit_requested:
                return False
            self._input_prompt = prompt
            self._waiting = True
            input_widget = self.query_one("#input", ChatInput)
            input_widget.placeholder = (
                prompt.strip() or "输入内容，Enter 发送，Shift+Enter 换行"
            )
            input_widget.disabled = False
            input_widget.focus()
            self._set_status(self._status_line())
            return True

        def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
            '''Enter 提交：整段文本原样交给 ChatLoop（粘贴的代码不会被拆开）。'''
            if not self._waiting or self._exit_requested:
                return
            text = event.value
            input_widget = self.query_one("#input", ChatInput)
            input_widget.text = ""
            input_widget.disabled = True
            self._waiting = False
            self._dismiss_welcome()
            self._set_status(self._status_line())
            self._inbox.put(text)

        def action_request_exit(self) -> None:
            '''Ctrl+C：唤醒阻塞输入，让 ChatLoop 按 EOF 路径正常收尾。'''
            if self._exit_requested:
                return
            self._exit_requested = True
            self.query_one("#input", ChatInput).disabled = True
            if self._waiting:
                self._waiting = False
                self._inbox.put(_EOF)
            self._set_status("正在结束…")

        # ---------------- 模型事件 ----------------
        def _on_event(self, event: "Event") -> None:
            '''worker 线程收到事件；所有控件操作都转交 UI 线程。'''
            etype = event.type
            if etype == EventType.USER_MESSAGE:
                # 一轮真正发出去的文本（多行输入已经拼好）才渲染成「你」——
                # 按每次提交渲染会把 ml / end 和各分行都变成独立气泡。
                text = str(event.data.get("text") or "")
                if text.strip():
                    self.call_from_thread(
                        lambda value=text: self._mount_message(value, user=True)
                    )
            elif etype == EventType.MESSAGE_DELTA:
                self._reply += str(event.data.get("text") or "")
                now = time.monotonic()
                if now - self._last_render >= 0.08:
                    self._last_render = now
                    self.call_from_thread(self._render_reply, self._reply)
            elif etype == EventType.MESSAGE_END:
                # 非流式调用没有 DELTA，END 里的完整正文也能直接显示。
                self._reply = str(event.data.get("text") or self._reply)
                self.call_from_thread(self._finish_reply, self._reply)
                self._reply = ""
                self._thinking = 0
                self._tool = ""
                self._last_render = 0.0
            elif etype == EventType.THINKING_DELTA:
                self._thinking += len(str(event.data.get("text") or ""))
                self._push_status()
            elif etype == EventType.TOOL_CALL_START:
                self._tool = str(event.data.get("name") or "")
                self._push_status()
            elif etype == EventType.TOOL_RESULT:
                self._tool = ""
                self._push_status()
            elif etype == EventType.USAGE:
                self._prompt_tokens += int(event.data.get("prompt_tokens") or 0)
                self._completion_tokens += int(
                    event.data.get("completion_tokens") or 0
                )
                self._push_status()
            elif etype == EventType.ERROR:
                self._set_status_from_thread("出错了，正在结束…")

        def _render_reply(self, text: str) -> None:
            '''UI 线程：新建或更新当前回复，并滚动到底部。'''
            self._dismiss_welcome()
            if self._current_reply is None:
                self._current_reply = Markdown(
                    self._message_markdown(text, user=False),
                    classes="message assistant",
                )
                self.query_one("#chat", VerticalScroll).mount(self._current_reply)
            else:
                self._current_reply.update(
                    self._message_markdown(text, user=False)
                )
            self._set_status(self._status_line())
            self.call_after_refresh(self._scroll_to_end)

        def _finish_reply(self, text: str) -> None:
            if text or self._current_reply is not None:
                self._render_reply(text)
            self._current_reply = None

        def _mount_message(self, text: str, *, user: bool) -> None:
            self._dismiss_welcome()
            widget = Markdown(
                self._message_markdown(text, user=user),
                classes=f"message {'user' if user else 'assistant'}",
            )
            self.query_one("#chat", VerticalScroll).mount(widget)
            self.call_after_refresh(self._scroll_to_end)

        def _dismiss_welcome(self) -> None:
            '''聊起来之后收掉「使用方法」，把空间还给对话。'''
            if self._welcome_dismissed:
                return
            self._welcome_dismissed = True
            for widget in self.query("#welcome"):
                widget.remove()

        @staticmethod
        def _message_markdown(text: str, *, user: bool) -> str:
            # 公式在进入 Textual 的 Markdown 之前先转成 Unicode；TeXicode 增强
            # 由 settings.prefer_texicode 控制，没开或没装 txc 时自动回退内置。
            return f"**{'你' if user else '助手'}**\n\n{substitute_math(text)}"

        def _scroll_to_end(self) -> None:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

        # ---------------- 状态栏 ----------------
        def _push_status(self) -> None:
            self._set_status_from_thread(self._status_line())

        def _set_status_from_thread(self, text: str) -> None:
            self.call_from_thread(self._set_status, text)

        def _set_status(self, text: str) -> None:
            self.query_one("#status", Static).update(text)

        def _status_line(self) -> str:
            if self._exit_requested:
                head = "正在结束…"
            elif self._waiting:
                head = f"等待输入 · {self._input_prompt.strip()}"
            elif self._tool:
                head = f"运行工具 {self._tool}"
            elif self._reply:
                head = f"生成中（{len(self._reply)} 字）"
            elif self._thinking:
                head = f"思考中（{self._thinking} 字）"
            else:
                head = "等待模型"
            if self._prompt_tokens or self._completion_tokens:
                head += (
                    f"   ↑{_tokens(self._prompt_tokens)}"
                    f" ↓{_tokens(self._completion_tokens)}"
                )
            if self._waiting:
                head += "   Enter 发送 · Shift+Enter 换行 · Ctrl+C 退出"
            return head

else:  # 让 `import promptkit.tui.textual_app` 在缺依赖时仍然成功
    TutorApp = None  # type: ignore[misc,assignment]


def run_fullscreen(
    context: "Context",
    run_loop: Callable[["Context"], None],
) -> None:
    '''运行全屏界面，并在退出后还原现场、重新抛出对话循环的异常。'''
    _require_environment()
    previous = context.get(TUI_ACTIVE_KEY)
    # 内层 ChatLoop 看到这个标记就不再自己启动一个全屏应用。
    context[TUI_ACTIVE_KEY] = True
    app = TutorApp(context, run_loop)
    try:
        app.run()
    finally:
        context[TUI_ACTIVE_KEY] = previous
    if app.error is not None:
        raise app.error
