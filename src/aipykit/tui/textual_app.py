# aipykit/tui/textual_app.py
'''Textual 全屏对话界面：状态栏、对话滚动区、底部输入框。

Textual 在主线程跑 asyncio，原有的同步 ChatLoop 则放进 worker 线程。
两边的桥接只有三条：

- `Read` 仍像调用 `input()` 一样阻塞，实际在等底部输入框提交的队列消息。
- 模型事件从 worker 线程经 `call_from_thread` 回到 UI，流式正文按 100ms 节流。
  流式期间用 Static 纯文本显示，避免 Markdown 解析器因为半截代码块反复重排。
- `Echo` 的输出流暂时换成空设备，避免把最终回复再打印一遍；回复正文
  已由 `MESSAGE_DELTA` / `MESSAGE_END` 显示在滚动区里。

界面分三块：顶部两行状态栏（模型、运行状态、token 用量、快捷键）、
中间的对话滚动区、底部输入框；输入框下面一行把思考等级和上下文进度条
铺满整行（最高档的等级标签带背景高亮）。离开底部时对话区上方会出现「回到最底部」。

思考过程和工具调用默认折叠（鼠标点标题或 Ctrl+R / Ctrl+T 展开），折叠时只
保留一行标题和耗时，展开后才渲染完整内容；历史面板留在 #chat 里可回看。

鼠标拖选对话文本后 Ctrl+C 复制，Ctrl+S 导出 Markdown，Ctrl+Q 退出。

入口是 `run_fullscreen()`，由 `EnableTUI(fullscreen=True)` 触发。
'''
from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from aipykit.core.display_sys.display import set_console_suppressed
from aipykit.core.display_sys.render import (
    HAVE_MERMAID,
    substitute_diagrams,
    substitute_math,
)
from aipykit.core.event_sys import EVENT_BUS_KEY, EventBus, EventType
from aipykit.core.steps.actions.interactive import INPUT_FN_KEY, OUTPUT_STREAM_KEY
from aipykit.core.steps.actions.llm import LLM_STREAM_KEY
from aipykit.core.steps.actions.tui import TUI_ACTIVE_KEY, TuiUnavailableError

if TYPE_CHECKING:
    from aipykit.core.context import Context
    from aipykit.core.event_sys import Event

try:  # 可选依赖：只有真正启动全屏 TUI 时才要求
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, VerticalScroll
    from textual.content import Content
    from textual.message import Message
    from textual.widgets import Collapsible, Markdown, Static, TextArea

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
    "• 鼠标拖选对话内容，[b]Ctrl+C[/b] 复制；[b]Ctrl+S[/b] 导出 Markdown\n"
    "• [b]Ctrl+Q[/b] 退出　[b]End[/b] 回到对话最底部\n"
    "• [b]Ctrl+R[/b] 展开/折叠思考　[b]Ctrl+T[/b] 展开/折叠工具调用\n"
)

#: 按模型名估算上下文窗口，只用于状态栏的百分比显示。
#: 拿不到真实窗口时按 128k 算，宁可显示得保守一点。
_CONTEXT_LIMITS = (
    ("deepseek", 1_000_000),
    ("gemini", 1_000_000),
    ("claude", 200_000),
    ("gpt", 128_000),
)
_DEFAULT_CONTEXT_LIMIT = 128_000


def _context_limit(model: str) -> int:
    '''按模型名猜一个上下文窗口（仅用于展示比例，不参与调用）。'''
    lowered = (model or "").lower()
    for needle, limit in _CONTEXT_LIMITS:
        if needle in lowered:
            return limit
    return _DEFAULT_CONTEXT_LIMIT


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
    '''token 数折成 1.2k / 1.0M 这种短写法。'''
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


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

    class ChatScroll(VerticalScroll):
        '''对话滚动区：滚动位置变化时发消息，用来控制「回到最底部」提示。

        直接监听鼠标滚轮事件不行 —— 滚动容器自己处理完就 `stop()` 了，
        事件不会再冒泡到 App。`scroll_y` 是响应式的，所有滚动来源
        （滚轮 / 键盘 / 代码）都会经过这里。
        '''

        class Scrolled(Message):
            '''滚动位置跨过整行时发出。'''

        def watch_scroll_y(self, old_value: float, new_value: float) -> None:
            super().watch_scroll_y(old_value, new_value)
            if round(old_value) != round(new_value):
                self.post_message(self.Scrolled())

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
            background: $boost;
        }
        #status-left {
            width: 1fr;
            padding: 0 1;
            color: $text;
        }
        #status-right {
            width: auto;
            padding: 0 1;
            color: $text-muted;
        }
        #status-detail {
            height: 1;
            padding: 0 1;
            background: $boost;
            color: $text-muted;
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
        #footer {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }
        #jump-bottom {
            height: 1;
            display: none;
            padding: 0 2;
            background: $boost;
            color: $accent;
            text-align: center;
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
            height: auto;
            margin: 1 0 0 0;
            padding: 0 1;
        }
        Collapsible.tool-panel {
            margin: 1 0 0 0;
            padding: 0 1;
            border: none;
            background: $surface;
        }
        Collapsible.tool-panel > .header {
            color: $text-muted;
        }
        Collapsible.thinking-panel {
            margin: 1 0 0 0;
            padding: 0 1;
            border: none;
            background: $surface;
        }
        Collapsible.thinking-panel > .header {
            color: $text-muted;
        }
        .panel-content {
            padding: 0 0 0 2;
        }
        .panel-args {
            color: $text-muted;
        }
        .panel-result {
            color: $text-muted;
        }
        .stream-reply {
            height: auto;
            margin: 1 0 0 0;
            padding: 0 1;
            color: $text-muted;
        }
        '''

        #: 允许鼠标拖选对话内容；选中后 Ctrl+C 复制（Textual 自带行为）
        ALLOW_SELECT = True

        # 不再拦截 Ctrl+C：选中文本时 Textual 复制，没选中时什么也不做。
        # Ctrl+Q 走「先唤醒阻塞输入、等循环收尾，再关界面」，不要用 Textual
        # 默认的立即退出 —— 那样 worker 里的 after_end / 落盘步骤可能来不及跑。
        BINDINGS = [
            Binding("ctrl+q", "request_exit", "退出", show=False, priority=True),
            Binding("ctrl+r", "toggle_thinking", "展开/折叠思考", show=False),
            Binding("ctrl+t", "toggle_tools", "展开/折叠工具", show=False),
            Binding("ctrl+s", "export_chat", "导出 Markdown", show=False),
            Binding("end", "jump_bottom", "回到最底部", show=False),
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

            # 流式正文：用 Static 累积原始文本，MESSAGE_END 一次性换成 Markdown，
            # 避免 Markdown 解析器在流式过程中因为半截代码块 / 未闭合表格反复重排。
            self._stream_reply: Optional[Static] = None
            self._stream_reply_text = ""
            self._current_reply: Optional[Markdown] = None

            # 思考面板：THINKING_START 挂载，THINKING_DELTA 累积内容，
            # THINKING_END 固定。新用户消息到来时会重置一轮。
            self._thinking_panel: Optional[Collapsible] = None
            self._thinking_content: Optional[Static] = None
            self._thinking_buffer = ""
            #: 本轮思考的起止时间，用来在折叠标题上显示「· 5s」
            self._thinking_started: Optional[float] = None
            self._thinking_seconds = 0.0

            # 工具面板：按 tool_call_id 或 (name+arguments) 分组，
            # 每个面板内含参数 Static 和结果 Static（未返回时显示"执行中…"）。
            # 这些字典跨轮保留：历史面板展开时还能看到完整输出。
            self._tool_panels: dict[str, Collapsible] = {}
            self._tool_contents: dict[str, tuple[Static, Static]] = {}
            #: 每个工具调用的展示信息：name / 格式化参数 / 摘要 / 耗时 / 结果
            self._tool_meta: dict[str, dict[str, Any]] = {}
            self._tool_started: dict[str, float] = {}
            self._tool_current_key: Optional[str] = None
            self._tool_current_args_parts: list[str] = []

            self._welcome_dismissed = False
            self._generating = False
            #: 用户刚提交、模型还没吐第一个 token
            self._pending_reply = False
            self._thinking = 0
            self._tool = ""
            #: 最近一次模型调用占用的上下文（prompt + completion），
            #: 与下面两个累计值不同：它只反映「当前上下文多大」。
            self._context_tokens = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._model: Optional[str] = None
            self._jump_hint: Optional[Static] = None
            self._last_render = 0.0
            #: 高频事件（思考 / 工具参数 / 状态栏）的节流时间戳。
            #: 逐 token 调 call_from_thread 会让终端每秒重绘上百次，
            #: 追不上就表现为「跳字」、甚至局部残影。统一按 ~10fps 节流。
            self._last_thinking_push = 0.0
            self._last_tool_push = 0.0
            self._last_status_push = 0.0

        def compose(self) -> ComposeResult:
            welcome = _WELCOME
            if not HAVE_MERMAID:
                welcome += (
                    "\n[b]图表渲染不可用[/b]：当前环境没装 termaid，"
                    '流程图会原样显示。pip install "aipykit[tui]" 后重启。\n'
                )
            with Horizontal(id="status"):
                yield Static("正在启动…", id="status-left")
                yield Static("", id="status-right")
            yield Static("", id="status-detail")
            with ChatScroll(id="chat"):
                yield Static(welcome, id="welcome")
            yield Static("↓ 回到最底部（End / 点击）", id="jump-bottom")
            yield ChatInput(
                placeholder="输入内容 · Enter 发送 · Shift+Enter 换行 · Ctrl+S 导出",
                id="input",
            )
            # 输入框下面一行：思考等级 + 铺满全宽的上下文进度条
            yield Static("", id="footer")

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
            self._jump_hint = self.query_one("#jump-bottom", Static)
            self._sync_jump_hint()
            self.run_worker(self._run, thread=True)

        def on_resize(self, event) -> None:
            '''终端宽度变了，底部进度条要按新宽度重画。'''
            if self.query("#footer"):
                self._set_status()

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
            self._waiting = True
            input_widget = self.query_one("#input", ChatInput)
            prompt_text = prompt.strip() or "输入内容"
            input_widget.placeholder = (
                f"{prompt_text} · Enter 发送 · Shift+Enter 换行"
            )
            input_widget.disabled = False
            input_widget.focus()
            self._set_status()
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
            self._pending_reply = True
            self._dismiss_welcome()
            self._set_status()
            self._inbox.put(text)

        def action_request_exit(self) -> None:
            '''Ctrl+Q：唤醒阻塞输入，让 ChatLoop 按 EOF 路径收尾后再关界面。'''
            if self._exit_requested:
                return
            self._exit_requested = True
            self.query_one("#input", ChatInput).disabled = True
            if self._waiting:
                self._waiting = False
                self._inbox.put(_EOF)
            self._set_status()

        # ---------------- 折叠快捷键 ----------------
        def action_toggle_thinking(self) -> None:
            '''Ctrl+R：展开 / 折叠所有 ✦ 思考面板。'''
            thinking_panels = list(self.query("Collapsible.thinking-panel"))
            if not thinking_panels:
                return
            # 以第一个为准：如果展开就全折叠，反之亦然。
            all_collapsed = all(panel.collapsed for panel in thinking_panels)
            for panel in thinking_panels:
                panel.collapsed = not all_collapsed
            self.call_after_refresh(self._sync_jump_hint)

        def action_toggle_tools(self) -> None:
            '''Ctrl+T：展开 / 折叠所有 ● 工具调用面板。'''
            tool_panels = list(self.query("Collapsible.tool-panel"))
            if not tool_panels:
                return
            all_collapsed = all(panel.collapsed for panel in tool_panels)
            for panel in tool_panels:
                panel.collapsed = not all_collapsed
            self.call_after_refresh(self._sync_jump_hint)

        # ---------------- 回到最底部 ----------------
        def action_jump_bottom(self) -> None:
            '''End / 点击浮动条：无视当前位置直接跳到底部。'''
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
            self.call_after_refresh(self._sync_jump_hint)

        def _sync_jump_hint(self) -> None:
            '''离开底部时显示「回到最底部」，回到末尾就收起。'''
            if self._jump_hint is None:
                return
            chat = self.query_one("#chat", ChatScroll)
            at_bottom = chat.max_scroll_y - chat.scroll_offset.y <= 1
            self._jump_hint.display = not at_bottom

        def on_chat_scroll_scrolled(self, event: "ChatScroll.Scrolled") -> None:
            self._sync_jump_hint()

        def on_click(self, event: Any) -> None:
            if self._jump_hint is not None and event.widget is self._jump_hint:
                event.stop()
                self.action_jump_bottom()

        def on_collapsible_expanded(self, event: "Collapsible.Expanded") -> None:
            self._on_collapsible_toggled(event.collapsible)

        def on_collapsible_collapsed(self, event: "Collapsible.Collapsed") -> None:
            self._on_collapsible_toggled(event.collapsible)

        def _on_collapsible_toggled(self, collapsible: "Collapsible") -> None:
            '''展开 / 折叠后重画内容：折叠时给预览，展开时给全文。'''
            if collapsible.has_class("thinking-panel"):
                collapsible.title = self._thinking_title()
            elif collapsible.has_class("tool-panel"):
                for key, panel in self._tool_panels.items():
                    if panel is collapsible:
                        self._refresh_tool_panel(key)
                        break
            self.call_after_refresh(self._sync_jump_hint)

        # ---------------- 导出 Markdown ----------------
        def action_export_chat(self) -> None:
            '''Ctrl+S：把当前对话导出到工作目录下的 Markdown 文件。'''
            filename = f"aipykit-chat-{time.strftime('%Y%m%d-%H%M%S')}.md"
            path = Path.cwd() / filename
            try:
                path.write_text(self._transcript_markdown(), encoding="utf-8")
            except OSError as exc:
                self.notify(f"导出失败：{exc}", title="导出失败", severity="error")
                return
            self.notify(f"已导出到 {path}", title="导出完成")

        def _transcript_markdown(self) -> str:
            '''按 #chat 里的挂载顺序还原成 Markdown（不含界面状态栏）。'''
            chat = self.query_one("#chat", ChatScroll)
            blocks = [
                "# aipykit 对话",
                f"- 模型：{self._resolve_model()}\n"
                f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            for widget in chat.children:
                if widget.id == "welcome":
                    continue
                if widget.has_class("stream-reply"):
                    body = self._strip_bullet(str(widget.render()))
                    blocks.append(f"## 助手（生成中）\n\n{body}")
                elif widget.has_class("user"):
                    body = "\n".join(
                        line[2:] if line.startswith("> ") else line
                        for line in str(widget.render()).splitlines()
                    )
                    blocks.append(f"## 你\n\n{body}")
                elif isinstance(widget, Markdown):
                    body = self._strip_bullet(str(widget.source))
                    blocks.append(f"## 助手\n\n{body}")
                elif isinstance(widget, Collapsible):
                    panel = self._panel_markdown(widget)
                    if panel:
                        blocks.append(panel)
            return "\n\n".join(blocks).rstrip() + "\n"

        def _panel_markdown(self, panel: "Collapsible") -> str:
            '''把折叠面板导出成 <details>，保留可折叠的思考 / 工具过程。'''
            if panel.has_class("thinking-panel"):
                content = list(panel.query("Static.panel-content"))
                body = str(content[0].render()).strip() if content else ""
                # 导出文件里不需要「Ctrl+R 展开」这类界面提示
                summary = (
                    panel.title.replace("（Ctrl+R 展开）", "")
                    .replace("（Ctrl+R 折叠）", "")
                    .rstrip()
                )
            elif panel.has_class("tool-panel"):
                meta = self._tool_meta.get(panel.name or "")
                if meta is None:
                    return ""
                summary = panel.title
                args = meta.get("args") or ""
                result = meta.get("result")
                body = args if result is None else f"{args}\n\n{result}"
            else:
                return ""
            return (
                f"<details>\n<summary>{summary}</summary>\n\n"
                f"{body}\n\n</details>"
            )

        @staticmethod
        def _strip_bullet(text: str) -> str:
            '''去掉助手正文前的 ● 圆点，导出时不需要它。'''
            if text.startswith("●\n\n"):
                return text[3:]
            if text.startswith("● "):
                return text[2:]
            return text

        # ---------------- 模型事件 ----------------
        def _on_event(self, event: "Event") -> None:
            '''worker 线程收到事件；所有控件操作都转交 UI 线程。'''
            etype = event.type
            if etype == EventType.USER_MESSAGE:
                text = str(event.data.get("text") or "")
                if text.strip():
                    self._reset_turn_state()
                    self.call_from_thread(
                        lambda value=text: self._mount_message(value, user=True)
                    )
            elif etype == EventType.MESSAGE_DELTA:
                self._generating = True
                self._pending_reply = False
                self._stream_reply_text += str(event.data.get("text") or "")
                now = time.monotonic()
                if now - self._last_render >= 0.1:
                    self._last_render = now
                    self.call_from_thread(self._render_stream_reply)
            elif etype == EventType.MESSAGE_END:
                final = str(event.data.get("text") or self._stream_reply_text)
                self._generating = False
                self._pending_reply = False
                self.call_from_thread(self._finish_reply, final)
                self._stream_reply_text = ""
                self._thinking = 0
                self._tool = ""
                self._last_render = 0.0
                self._push_status(force=True)
            elif etype == EventType.THINKING_START:
                self._thinking_buffer = ""
                self._thinking_started = time.monotonic()
                self._thinking_seconds = 0.0
                self.call_from_thread(self._mount_thinking_panel)
                self._push_status(force=True)
            elif etype == EventType.THINKING_DELTA:
                chunk = str(event.data.get("text") or "")
                self._thinking_buffer += chunk
                self._thinking += len(chunk)
                now = time.monotonic()
                if now - self._last_thinking_push >= 0.12:
                    self._last_thinking_push = now
                    self.call_from_thread(
                        self._update_thinking_panel, self._thinking_buffer
                    )
                    self._push_status()
            elif etype == EventType.THINKING_END:
                final = str(event.data.get("text") or self._thinking_buffer)
                self._thinking_buffer = final
                if self._thinking_started is not None:
                    self._thinking_seconds = (
                        time.monotonic() - self._thinking_started
                    )
                    self._thinking_started = None
                self.call_from_thread(self._finish_thinking_panel, final)
                self._push_status(force=True)
            elif etype == EventType.TOOL_CALL_START:
                name = str(event.data.get("name") or "")
                args = str(event.data.get("arguments") or "")
                self._tool = name
                self._tool_current_key = self._next_tool_key(name, args)
                self._tool_current_args_parts = []
                self._tool_started[self._tool_current_key] = time.monotonic()
                self.call_from_thread(
                    self._mount_tool_panel,
                    self._tool_current_key,
                    name,
                    args,
                )
                self._push_status(force=True)
            elif etype == EventType.TOOL_CALL_DELTA:
                args_piece = str(event.data.get("text") or "")
                self._tool_current_args_parts.append(args_piece)
                key = self._tool_current_key
                now = time.monotonic()
                if key and now - self._last_tool_push >= 0.12:
                    self._last_tool_push = now
                    self.call_from_thread(
                        self._update_tool_panel_args,
                        key,
                        "".join(self._tool_current_args_parts),
                    )
            elif etype == EventType.TOOL_CALL_END:
                args_full = str(event.data.get("arguments") or "")
                key = self._tool_current_key
                if key:
                    self.call_from_thread(
                        self._finish_tool_panel_args,
                        key,
                        args_full or "".join(self._tool_current_args_parts),
                    )
                self._tool_current_args_parts = []
            elif etype == EventType.TOOL_RESULT:
                name = str(event.data.get("name") or "")
                result = str(event.data.get("result") or "")
                key = self._tool_current_key
                if not key:
                    key = self._find_tool_panel_by_name(name)
                if key:
                    started = self._tool_started.pop(key, None)
                    seconds = (
                        time.monotonic() - started if started is not None else None
                    )
                    self.call_from_thread(
                        self._set_tool_result, key, result, seconds
                    )
                self._tool = ""
                self._tool_current_key = None
                self._tool_current_args_parts = []
                self._push_status(force=True)
            elif etype == EventType.USAGE:
                prompt_tokens = int(event.data.get("prompt_tokens") or 0)
                completion_tokens = int(
                    event.data.get("completion_tokens") or 0
                )
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                # 上下文占用看「最新一次调用」而不是累计，否则每轮都会虚高。
                if prompt_tokens or completion_tokens:
                    self._context_tokens = prompt_tokens + completion_tokens
                # 一次调用只发一条 USAGE，直接刷，别被 10fps 节流吞掉。
                self._push_status(force=True)
            elif etype == EventType.NOTIFICATION:
                text = str(event.data.get("text") or "")
                title = str(event.data.get("title") or "通知")
                if text.strip():
                    self.call_from_thread(self._show_notification, text, title)
            elif etype == EventType.ERROR:
                self._generating = False
                self._pending_reply = False
                self._set_status_from_thread()

        def _show_notification(self, text: str, title: str) -> None:
            '''在 UI 线程显示工作流主动发出的通知。'''
            self.notify(text, title=title)

        # ---- 一轮对话状态重置 ----
        def _reset_turn_state(self) -> None:
            '''新用户消息到来时重置上一轮助手状态变量。

            已 mount 的 widget 不动（历史面板留在 #chat 里当可回看记录），
            只重置变量让下一轮挂新面板。
            '''
            self._stream_reply = None
            self._stream_reply_text = ""
            self._current_reply = None
            self._generating = False
            self._pending_reply = True
            self._thinking_panel = None
            self._thinking_content = None
            self._thinking_buffer = ""
            self._thinking_started = None
            self._thinking_seconds = 0.0
            self._tool_current_key = None
            self._tool_current_args_parts = []
            self._tool = ""
            self._thinking = 0
            self._last_render = 0.0
            self._last_thinking_push = 0.0
            self._last_tool_push = 0.0

        def _next_tool_key(self, name: str, args: str) -> str:
            base = name or "tool"
            counter = 0
            key = base
            while key in self._tool_panels:
                counter += 1
                key = f"{base}#{counter}"
            return key

        def _find_tool_panel_by_name(self, name: str) -> Optional[str]:
            if not name:
                return None
            # 从最新往旧找：同名工具应该优先匹配刚发出的那次调用。
            for key in reversed(list(self._tool_panels)):
                meta = self._tool_meta.get(key) or {}
                if meta.get("name") == name:
                    return key
            return None

        # ---- 流式正文（Static，稳定不抖） ----
        def _render_stream_reply(self) -> None:
            '''UI 线程：流式期间用 Static 显示纯文本，避免 Markdown 重排抖动。

            `markup=False`：半截回复里出现 `[` 时不会被当成 Textual 标记吞掉。
            '''
            self._dismiss_welcome()
            text = f"● {self._stream_reply_text}"
            if self._stream_reply is None:
                self._stream_reply = Static(
                    text, classes="stream-reply", markup=False
                )
                self.query_one("#chat", VerticalScroll).mount(self._stream_reply)
            else:
                self._stream_reply.update(text)
            self._set_status()
            self.call_after_refresh(self._scroll_to_end)

        def _finish_reply(self, text: str) -> None:
            '''UI 线程：把流式 Static 替换成 Markdown，收尾。'''
            if self._stream_reply is not None:
                self._detach(self._stream_reply)
                self._stream_reply = None
            if text:
                self._current_reply = Markdown(
                    self._message_markdown(text, user=False),
                    classes="message assistant",
                )
                self.query_one("#chat", VerticalScroll).mount(self._current_reply)
                self._set_status()
                self.call_after_refresh(self._scroll_to_end)
            self._current_reply = None

        # ---- 思考面板 ----
        def _thinking_title(self) -> str:
            '''✦/⚡ 思考 · 5.0s（Ctrl+R 展开）—— 折叠时也看得到耗时。'''
            seconds = self._thinking_seconds
            if self._thinking_started is not None:
                seconds = time.monotonic() - self._thinking_started
            label = "✦ 思考"
            if self._is_max_reasoning():
                # 最高推理等级：思考面板标题也带闪电，和状态栏呼应
                label = "⚡ 思考"
            if seconds:
                label += f" · {seconds:.1f}s"
            expanded = (
                self._thinking_panel is not None
                and not self._thinking_panel.collapsed
            )
            return label + ("（Ctrl+R 折叠）" if expanded else "（Ctrl+R 展开）")

        def _mount_thinking_panel(self) -> None:
            self._dismiss_welcome()
            content_widget = Static(
                "思考中…", classes="panel-content", markup=False
            )
            panel = Collapsible(
                content_widget,
                title=self._thinking_title(),
                collapsed=True,
                classes="thinking-panel",
            )
            self.query_one("#chat", VerticalScroll).mount(panel)
            self._thinking_panel = panel
            self._thinking_content = content_widget
            self.call_after_refresh(self._scroll_to_end)

        def _update_thinking_panel(self, text: str) -> None:
            '''折叠时跳过内容更新：内容不可见，但每次 update 都会重排整块，
            思考阶段几百次重排正是「跳字」的来源。展开时按节流后的频率刷新。
            标题（耗时）始终更新，折叠状态下也能看到进度。'''
            panel = self._thinking_panel
            if panel is None:
                return
            if not panel.collapsed and self._thinking_content is not None:
                self._thinking_content.update(text or "思考中…")
            panel.title = self._thinking_title()

        def _finish_thinking_panel(self, text: str) -> None:
            if self._thinking_panel is not None:
                self._thinking_panel.title = self._thinking_title()
            if self._thinking_content is not None:
                self._thinking_content.update(text or "(空)")

        # ---- 工具面板 ----
        #: 折叠时参数 / 结果各显示几行，其余用「… +N 行」提示
        _TOOL_ARG_LINES = 12
        _TOOL_RESULT_LINES = 8

        def _mount_tool_panel(self, key: str, name: str, args: str) -> None:
            self._dismiss_welcome()
            self._tool_meta[key] = {
                "name": name,
                "args": self._format_tool_args(args),
                "summary": self._tool_summary(args),
                "seconds": None,
                "result": None,
            }
            args_widget = Static(
                "", classes="panel-args panel-content", markup=False
            )
            result_widget = Static(
                "执行中…", classes="panel-result panel-content", markup=False
            )
            panel = Collapsible(
                args_widget,
                result_widget,
                title=self._tool_title(key),
                collapsed=True,
                classes="tool-panel",
                name=key,
            )
            self.query_one("#chat", VerticalScroll).mount(panel)
            self._tool_panels[key] = panel
            self._tool_contents[key] = (args_widget, result_widget)
            self._refresh_tool_panel(key)
            self.call_after_refresh(self._scroll_to_end)

        def _update_tool_panel_args(self, key: str, args: str) -> None:
            '''同上：折叠时没必要频繁重排参数文本，只记下最新值。'''
            meta = self._tool_meta.get(key)
            if meta is None:
                return
            meta["args"] = self._format_tool_args(args)
            meta["summary"] = self._tool_summary(args)
            panel = self._tool_panels.get(key)
            if panel is None or panel.collapsed:
                return
            pair = self._tool_contents.get(key)
            if pair is not None:
                pair[0].update(meta["args"] + "\n（流式中）")

        def _finish_tool_panel_args(self, key: str, args: str) -> None:
            meta = self._tool_meta.get(key)
            if meta is None:
                return
            meta["args"] = self._format_tool_args(args)
            meta["summary"] = self._tool_summary(args)
            self._refresh_tool_panel(key)

        def _set_tool_result(
            self, key: str, result: str, seconds: Optional[float]
        ) -> None:
            meta = self._tool_meta.get(key)
            if meta is None:
                return
            meta["result"] = result
            meta["seconds"] = seconds
            self._refresh_tool_panel(key)

        def _refresh_tool_panel(self, key: str) -> None:
            '''按当前折叠状态重画参数 / 结果：折叠给预览，展开给全文。'''
            panel = self._tool_panels.get(key)
            pair = self._tool_contents.get(key)
            meta = self._tool_meta.get(key)
            if panel is None or pair is None or meta is None:
                return
            expanded = not panel.collapsed
            args_widget, result_widget = pair
            args_widget.update(
                self._preview(
                    meta.get("args") or "",
                    limit=self._TOOL_ARG_LINES,
                    hint="（Ctrl+T 展开）",
                    expanded=expanded,
                )
            )
            if meta.get("result") is None:
                result_widget.update("执行中…")
            else:
                result_widget.update(
                    self._tool_result_text(meta["result"], expanded=expanded)
                )
            panel.title = self._tool_title(key)

        def _tool_title(self, key: str) -> str:
            meta = self._tool_meta.get(key) or {}
            title = f"● {meta.get('name') or 'tool'}"
            summary = meta.get("summary")
            if summary:
                title += f" {summary}"
            seconds = meta.get("seconds")
            if seconds is not None:
                title += f" · {seconds:.1f}s"
            return title

        def _tool_result_text(self, result: str, *, expanded: bool) -> str:
            body = f"结果：\n{result}" if result else "结果：（空）"
            return self._preview(
                body,
                limit=self._TOOL_RESULT_LINES,
                hint="（Ctrl+T 展开）",
                expanded=expanded,
            )

        @staticmethod
        def _preview(text: str, *, limit: int, hint: str, expanded: bool) -> str:
            '''按行截断：折叠时只给前 `limit` 行，展开时原样返回。'''
            lines = text.splitlines()
            if expanded or len(lines) <= limit:
                return text
            hidden = len(lines) - limit
            return "\n".join(lines[:limit]) + f"\n\n… +{hidden} 行{hint}"

        @staticmethod
        def _format_tool_args(args: str) -> str:
            '''工具参数：能当 JSON 就美化，否则原样。

            这段文本进的是 `markup=False` 的 Static，所以 `[` 不会被当成标记；
            也不要套 ``` 围栏，否则终端里会原样看到反引号。
            '''
            if not args:
                return "（无参数）"
            try:
                parsed = json.loads(args.strip())
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                return f"参数：\n{pretty}"
            except (json.JSONDecodeError, ValueError):
                return f"参数：{args.strip()}"

        @staticmethod
        def _tool_summary(args: str, limit: int = 40) -> str:
            '''标题里的一行参数摘要：`query="foo"` / `path=src/app.py`。'''
            text = (args or "").strip()
            if not text:
                return ""
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and parsed:
                parts = []
                for key, value in list(parsed.items())[:2]:
                    if isinstance(value, str):
                        shown = value
                    else:
                        shown = json.dumps(value, ensure_ascii=False)
                    shown = " ".join(str(shown).split())
                    if len(shown) > 24:
                        shown = shown[:24] + "…"
                    parts.append(f"{key}={shown}")
                summary = ", ".join(parts)
            else:
                summary = " ".join(text.split())
            return summary[:limit] + ("…" if len(summary) > limit else "")

        # ---------------- 通用渲染 ----------------
        def _mount_message(self, text: str, *, user: bool) -> None:
            self._dismiss_welcome()
            chat = self.query_one("#chat", VerticalScroll)
            if user:
                widget = Static(self._user_content(text), classes="user")
            else:
                widget = Markdown(
                    self._message_markdown(text, user=False),
                    classes="message assistant",
                )
            chat.mount(widget)
            self.call_after_refresh(self._scroll_to_end)

        @staticmethod
        def _user_content(text: str) -> "Content":
            '''用户消息：橙色加粗、`> ` 前缀，不用气泡背景。'''
            marked = "\n".join(
                f"> {line}" for line in (text.splitlines() or [""])
            )
            return Content.styled(marked, "bold orange1")

        @staticmethod
        def _detach(widget: Any) -> None:
            '''先隐藏，等下一次刷新后再真正移除。

            直接 `remove()` 会让控件立刻失去父节点，但在本轮刷新完成前，
            Textual 的命中测试仍可能返回它。此时鼠标按下会进入 Screen 的
            文本选中逻辑，访问 `container.region` 时因 parent 为 None 崩溃
            （AttributeError: 'NoneType' object has no attribute 'region'）。
            先 `display = False` 让它退出命中范围，下一帧移除就没有窗口期。
            '''
            widget.display = False
            widget.call_after_refresh(widget.remove)

        def _dismiss_welcome(self) -> None:
            '''聊起来之后收掉「使用方法」，把空间还给对话。'''
            if self._welcome_dismissed:
                return
            self._welcome_dismissed = True
            for widget in self.query("#welcome"):
                self._detach(widget)

        @staticmethod
        def _message_markdown(
            text: str, *, user: bool = False, prefer_texicode: Optional[bool] = None
        ) -> str:
            # 先把 ```mermaid / ```dot 画成字符画，再把公式转成 Unicode；
            # TeXicode 增强由 settings.prefer_texicode 控制，没装 txc 自动回退。
            body = substitute_diagrams(text)
            body = substitute_math(body, prefer_texicode=prefer_texicode)
            if user:
                return "\n".join(
                    f"> {line}" for line in (body.splitlines() or [""])
                )
            return TutorApp._with_bullet(body)

        @staticmethod
        def _with_bullet(body: str) -> str:
            '''助手正文前的 ●；正文以代码块 / 列表开头时让圆点单独占一行。'''
            block_starts = ("```", "~~~", "#", ">", "|", "- ", "* ", "+ ", "1. ")
            if body.lstrip().startswith(block_starts):
                return f"●\n\n{body}"
            return f"● {body}"

        def _scroll_to_end(self) -> None:
            '''滚到底部；用户主动往上翻时不抢滚动位置。'''
            chat = self.query_one("#chat", VerticalScroll)
            if chat.max_scroll_y - chat.scroll_offset.y <= 3:
                chat.scroll_end(animate=False)
            self._sync_jump_hint()

        # ---------------- 状态栏 ----------------
        def _push_status(self, *, force: bool = False) -> None:
            '''状态栏刷新。默认 ~10fps 节流；关键节点传 force=True 立即刷。'''
            if not force:
                now = time.monotonic()
                if now - self._last_status_push < 0.12:
                    return
                self._last_status_push = now
            self._set_status_from_thread()

        def _set_status_from_thread(self) -> None:
            self.call_from_thread(self._set_status)

        def _set_status(self) -> None:
            left, right, detail = self._status_parts()
            self.query_one("#status-left", Static).update(left)
            self.query_one("#status-right", Static).update(right)
            self.query_one("#status-detail", Static).update(detail)
            self.query_one("#footer", Static).update(self._footer_text())

        def _resolve_agent(self) -> Optional[Any]:
            '''Context 里的 Agent -> 默认 Agent；都拿不到就 None。'''
            agent = self._workflow_context.get("agent")
            if agent is not None:
                return agent
            try:
                from aipykit.agent import Agent

                return Agent.default()
            except Exception:  # noqa: BLE001 - 纯展示，拿不到就算了
                return None

        def _resolve_model(self) -> str:
            '''模型名：Context 里的 Agent -> 默认 Agent -> 配置默认值。'''
            if self._model:
                return self._model
            agent = self._resolve_agent()
            model = getattr(agent, "model", None) if agent is not None else None
            if not model:
                from aipykit import config

                # 不缓存：TUI 先于 ChatLoop 启动时 Agent 可能还没创建，
                # 下一轮再查一次就能拿到真正的 model。
                return str(config.DEFAULT_MODEL)
            self._model = str(model)
            return self._model

        def _resolve_reasoning(self) -> Optional[str]:
            '''当前推理等级（low / medium / high / off），没设置就不展示。'''
            agent = self._resolve_agent()
            level = (
                getattr(agent, "reasoning_level", None) if agent is not None else None
            )
            return str(level) if level else None

        def _is_max_reasoning(self) -> bool:
            '''是不是厂商声明的最高等级（决定等级标签要不要背景高亮）。'''
            agent = self._resolve_agent()
            return bool(getattr(agent, "reasoning_is_max", False))

        def _footer_text(self) -> str:
            '''输入框下方一行：思考等级 + 铺满宽度的上下文进度条 + 用量。'''
            level = self._resolve_reasoning()
            if level:
                is_max = self._is_max_reasoning()
                # 最高档统一叫 MAX，比 HIGH 更容易一眼看出来
                text = "MAX" if is_max else level.upper()
                if is_max:
                    # 最高档：白字 + 亮紫背景，不做任何定时闪烁
                    label = f"[bold white on bright_magenta] {text} [/]"
                    label_width = len(text) + 2
                else:
                    label = f"[magenta]{text}[/magenta]"
                    label_width = len(text)
            else:
                label_width, label = 2, "[dim]--[/dim]"

            model = self._resolve_model()
            limit = _context_limit(model)
            used = self._context_tokens
            ratio = min(used / limit, 1.0) if limit else 0.0
            usage = f"{_tokens(used)}/{_tokens(limit)} {ratio * 100:.1f}%"

            # 终端宽度 - 左右 padding；再给标签、用量和两个空格让位
            width = max(self.size.width - 2, 20)
            bar_width = max(width - label_width - len(usage) - 2, 5)
            filled = min(max(round(ratio * bar_width), 0), bar_width)
            # 进度条按占用率变色：正常青色 -> 偏高黄色 -> 接近满红色
            color = "cyan" if ratio < 0.6 else ("yellow" if ratio < 0.85 else "red")
            bar = f"[{color}]{'=' * filled}[/][dim]{'-' * (bar_width - filled)}[/]"
            return f"{label} {bar} [bold]{usage}[/]"

        def _state_text(self) -> str:
            if self._exit_requested:
                return "正在结束"
            if self._waiting:
                return "等待输入"
            if self._tool:
                return f"工具 {self._tool}"
            if self._generating:
                return "生成中"
            if self._thinking:
                return "思考中"
            if self._pending_reply:
                return "等待模型"
            if self._prompt_tokens or self._completion_tokens:
                return "就绪"
            return "等待模型"

        def _status_parts(self) -> tuple[str, str, str]:
            '''顶部两行：模型 / 运行状态 + token 用量、快捷键。'''
            model = self._resolve_model()
            left = f"[b]{model}[/b] · {self._state_text()}"
            right = ""
            if self._prompt_tokens or self._completion_tokens:
                right = (
                    f"↑{_tokens(self._prompt_tokens)}"
                    f" ↓{_tokens(self._completion_tokens)}"
                )
            if self._waiting:
                detail = "Enter 发送 · Shift+Enter 换行 · Ctrl+S 导出"
            else:
                detail = "Ctrl+C 复制 · Ctrl+S 导出 · Ctrl+Q 退出"
            return left, right, detail

else:  # 让 `import aipykit.tui.textual_app` 在缺依赖时仍然成功
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
    # Agent(verbose=True) 之类会把思考和工具调用直接写到 stderr；全屏期间
    # 终端归 Textual 所有，这些输出会把界面写花（文字错位叠加）。统一关掉，
    # 需要看过程就展开 TUI 里的面板。
    set_console_suppressed(True)
    app = TutorApp(context, run_loop)
    try:
        app.run()
    finally:
        set_console_suppressed(False)
        context[TUI_ACTIVE_KEY] = previous
    if app.error is not None:
        raise app.error
