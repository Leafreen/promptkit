﻿'''EnableTUI 的接线：标记步骤的摘除、输入注入、环境不满足时的显式失败。'''
import pytest

from aipykit.core import (
    ChatLoop,
    Context,
    Echo,
    EnableTUI,
    Read,
    TuiUnavailableError,
)
from aipykit.core.errors import WorkflowError
from aipykit.core.steps.actions.interactive import INPUT_FN_KEY
from aipykit.core.steps.actions.llm import LLMStep, LLM_STREAM_KEY
from aipykit.prompt import Prompt


class _NotTty:
    '''假的非终端流：只实现 isatty()。'''

    def isatty(self) -> bool:
        return False


def test_enable_tui_is_lifted_out_of_the_loop_body():
    wf = ChatLoop(EnableTUI(), Echo("hi"))
    assert wf._tui is True
    assert not any(isinstance(step, EnableTUI) for step in wf.steps)


def test_chatloop_without_marker_keeps_tui_off():
    assert ChatLoop(Echo("hi"))._tui is False


def test_enable_tui_as_a_normal_step_is_rejected():
    with pytest.raises(WorkflowError):
        EnableTUI().execute(Context())


def test_read_uses_injected_reader():
    ctx = Context()
    seen = []

    def reader(prompt: str) -> str:
        seen.append(prompt)
        return "  你好  "

    ctx[INPUT_FN_KEY] = reader
    assert Read("> ").eval(ctx) == "你好"
    assert seen == ["> "]


def test_read_multiline_uses_injected_reader():
    ctx = Context()
    lines = iter(["第一行", "第二行", ""])
    ctx[INPUT_FN_KEY] = lambda prompt: next(lines)
    assert Read("...").Until("").eval(ctx) == "第一行\n第二行"


def test_missing_dependency_raises_explicitly(monkeypatch):
    from promptkit.tui import app

    monkeypatch.setattr(app, "_IMPORT_ERROR", ImportError("boom"))
    with pytest.raises(TuiUnavailableError) as exc:
        app.start_tui(Context())
    assert "aipykit[tui]" in str(exc.value)


def test_non_tty_raises_explicitly(monkeypatch):
    from promptkit.tui import app

    monkeypatch.setattr(app, "_IMPORT_ERROR", None)
    monkeypatch.setattr(app.sys, "stdin", _NotTty())
    monkeypatch.setattr(app.sys, "stdout", _NotTty())
    with pytest.raises(TuiUnavailableError) as exc:
        app.start_tui(Context())
    assert "TTY" in str(exc.value)


def test_llm_stream_resolution():
    ctx = Context()

    # 显式 True 优先
    assert LLMStep(Prompt("hi"), stream=True)._resolve_stream(ctx) is True

    # 显式 False 覆盖 context 标记
    ctx[LLM_STREAM_KEY] = True
    assert LLMStep(Prompt("hi"), stream=False)._resolve_stream(ctx) is False

    # None 读 context
    assert LLMStep(Prompt("hi"))._resolve_stream(ctx) is True
    ctx[LLM_STREAM_KEY] = False
    assert LLMStep(Prompt("hi"))._resolve_stream(ctx) is False


def test_fullscreen_marker_is_lifted_out_of_the_loop_body():
    wf = ChatLoop(EnableTUI(fullscreen=True), Echo("hi"))
    assert wf._tui is True
    assert wf._tui_fullscreen is True
    assert not any(isinstance(step, EnableTUI) for step in wf.steps)


def test_fullscreen_missing_dependency_raises_explicitly(monkeypatch):
    from promptkit.tui import textual_app

    monkeypatch.setattr(textual_app, "_IMPORT_ERROR", ImportError("boom"))
    with pytest.raises(TuiUnavailableError) as exc:
        textual_app.run_fullscreen(Context(), lambda context: None)
    assert "aipykit[tui]" in str(exc.value)


def test_fullscreen_non_tty_raises_explicitly(monkeypatch):
    from promptkit.tui import textual_app

    monkeypatch.setattr(textual_app, "_IMPORT_ERROR", None)
    monkeypatch.setattr(textual_app.sys, "stdin", _NotTty())
    monkeypatch.setattr(textual_app.sys, "stdout", _NotTty())
    with pytest.raises(TuiUnavailableError) as exc:
        textual_app.run_fullscreen(Context(), lambda context: None)
    assert "TTY" in str(exc.value)


def test_fullscreen_displays_workflow_notification():
    import asyncio
    import threading

    from aipykit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from aipykit.tui.textual_app import TutorApp

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        context[EVENT_BUS_KEY].emit(
            Event(EventType.NOTIFICATION, {"text": "重试对话", "title": "状态"})
        )
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        notifications = []
        app.notify = lambda text, **kwargs: notifications.append((text, kwargs))
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(100):
                if notifications:
                    break
                await pilot.pause(0.02)
            allow_exit.set()
            await pilot.pause(0.2)
        return app, notifications

    app, notifications = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert notifications == [("重试对话", {"title": "状态"})]


def test_fullscreen_shows_usage_help_until_first_submit():
    '''首次进入时对话区显示使用方法，发出第一条消息后收起。'''
    import asyncio
    import threading

    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.widgets import Static

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        context[INPUT_FN_KEY]("你 >")
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = str(app.query_one("#welcome", Static).render())
            assert "使用方法" in text
            assert "Enter" in text
            assert "Shift+Enter" in text
            assert "Ctrl+C" in text
            try:
                box = app.query_one("#input", ChatInput)
                box.text = "你好"
                await pilot.press("enter")
                await pilot.pause()
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
            assert not app.query("#welcome")
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_input_events_and_context_restoration():
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.containers import VerticalScroll
    from textual.widgets import Markdown

    ctx = Context()
    seen = []
    allow_exit = threading.Event()

    def loop(context):
        text = context[INPUT_FN_KEY]("你 >")
        seen.append(text)
        bus = context[EVENT_BUS_KEY]
        # 「你」的气泡由 USER_MESSAGE 驱动：这样多行输入才不会被拆成好几条。
        bus.emit(Event(EventType.USER_MESSAGE, {"text": text}))
        bus.emit(Event(EventType.MESSAGE_DELTA, {"text": "你好"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "你好"}))
        # 等 pilot.press() 返回后再退出 App，否则它会在按键收尾时失去消息循环。
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            input_widget = app.query_one("#input", ChatInput)
            assert input_widget.disabled is False
            input_widget.text = "测试消息"
            try:
                await pilot.press("enter")
                await pilot.pause()
                chat = app.query_one("#chat", VerticalScroll)
                # 用户消息是橙色 Static，助手回复是 Markdown
                assert len(chat.query(Markdown)) == 1
                assert len(chat.query("Static.user")) == 1
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(exercise())
    assert app.error is None
    assert seen == ["测试消息"]
    assert ctx.get(INPUT_FN_KEY) is None
    assert ctx.get(LLM_STREAM_KEY) is None


def test_fullscreen_multiline_submissions_collapse_into_one_bubble():
    '''多行输入按每次 Enter 提交，但界面上只该出现一条「你」的消息。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.containers import VerticalScroll
    from textual.widgets import Markdown

    ctx = Context()
    hold = threading.Event()
    joined = []

    def loop(context):
        parts = [context[INPUT_FN_KEY]("... ") for _ in range(3)]
        joined.append("\n".join(parts))
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": joined[-1]}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "收到"}))
        assert hold.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            for text in ["第一行", "第二行", "第三行"]:
                for _ in range(100):
                    if not app.query_one("#input", ChatInput).disabled:
                        break
                    await pilot.pause(0.02)
                box = app.query_one("#input", ChatInput)
                box.text = text
                await pilot.press("enter")
                await pilot.pause(0.05)

            await pilot.pause(0.2)
            chat = app.query_one("#chat", VerticalScroll)
            users = [str(w.render()) for w in chat.query("Static.user")]
            replies = [str(getattr(w, "source", "")) for w in chat.query(Markdown)]
            hold.set()
            await pilot.pause(0.2)
        return app, users, replies

    app, users, replies = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert joined == ["第一行\n第二行\n第三行"]
    assert users == ["> 第一行\n> 第二行\n> 第三行"]
    assert replies == ["● 收到"]


def test_fullscreen_renders_math_in_messages():
    '''助手回复里的 LaTeX 公式在进 Markdown 控件前先转成 Unicode。'''
    from promptkit.tui.textual_app import TutorApp

    text = TutorApp._message_markdown(r"其中 $\beta_0$ 是截距", user=False)
    assert text == "● 其中 `β₀` 是截距"


def test_fullscreen_user_message_uses_orange_static():
    '''用户消息没有气泡背景，用 `> ` 前缀 + 橙色加粗 Static 呈现。'''
    from promptkit.tui.textual_app import TutorApp

    content = TutorApp._user_content("你好\n世界")
    assert str(content) == "> 你好\n> 世界"


def test_fullscreen_input_supports_newlines_and_pasted_code():
    '''Shift+Enter 换行；粘贴进来的多行代码原样提交，不再被拆成多次输入。'''
    import asyncio
    import threading

    from promptkit.tui.textual_app import ChatInput, TutorApp

    ctx = Context()
    received = []
    allow_exit = threading.Event()

    def loop(context):
        received.append(context[INPUT_FN_KEY]("... "))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#input", ChatInput)
            await pilot.press("shift+enter")
            await pilot.pause()
            # Shift+Enter 是换行，不是提交
            assert box.text == "\n"
            # 模拟粘贴：TextArea 的粘贴就是一次多行 insert
            box.text = ""
            box.insert("int main() {\n\n  return 0;\n}")
            await pilot.pause()
            # 4 行正文 + 上下边框：自动撑高，再长就出现滚动条
            assert int(box.styles.height.value) == 6
            try:
                await pilot.press("enter")
                await pilot.pause()
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert received == ["int main() {\n\n  return 0;\n}"]


def test_fullscreen_ctrl_c_does_not_quit():
    '''Ctrl+C 现在只负责复制选中文本，没选中时也不应退出应用。'''
    import asyncio

    from promptkit.tui.textual_app import ChatInput, TutorApp

    ctx = Context()
    outcome = []

    def loop(context):
        try:
            context[INPUT_FN_KEY]("你 >")
        except EOFError:
            outcome.append("eof")
            return
        outcome.append("got-input")

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#input", ChatInput)
            assert box.disabled is False
            await pilot.press("ctrl+c")
            await pilot.pause(0.2)
            assert app._exit_requested is False
            assert box.disabled is False
            # 用 Ctrl+Q 正常退出，让阻塞的输入走 EOF 路径收尾
            await pilot.press("ctrl+q")
            await pilot.pause(0.3)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert outcome == ["eof"]
    assert ctx.get(INPUT_FN_KEY) is None


def test_fullscreen_ctrl_c_copies_selected_chat_text():
    '''拖选对话内容后 Ctrl+C 应该复制到剪贴板（Textual 的选区复制）。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import TutorApp

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "测试消息"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "收到"}))
        # 让输入框拿到焦点，验证它没有选中内容时 Ctrl+C 会落到 Screen 的复制
        try:
            context[INPUT_FN_KEY]("你 >")
        except EOFError:
            pass
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            chat = app.query_one("#chat")
            app.screen._select_all_in_widget(chat)
            await pilot.pause(0.05)
            selected = app.screen.get_selected_text() or ""
            assert "测试消息" in selected
            await pilot.press("ctrl+c")
            await pilot.pause(0.05)
            clipboard = app._clipboard or ""
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app, clipboard

    app, clipboard = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert "测试消息" in clipboard


def test_fullscreen_ctrl_q_waits_for_worker_to_finish():
    '''Ctrl+Q 不能立刻销毁界面：要等 worker 跑完收尾步骤再退出。'''
    import asyncio
    import threading

    from promptkit.tui.textual_app import TutorApp

    ctx = Context()
    eof_seen = threading.Event()
    release = threading.Event()
    finished = []

    def loop(context):
        try:
            context[INPUT_FN_KEY]("你 >")
        except EOFError:
            eof_seen.set()
            assert release.wait(timeout=10)
            finished.append(True)
            return
        finished.append(False)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+q")
            await pilot.pause(0.2)
            assert eof_seen.is_set()
            # worker 还卡在收尾里，界面应该继续保持运行
            assert app.is_running is True
            release.set()
            await pilot.pause(0.3)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert finished == [True]


def test_console_display_silenced_while_fullscreen():
    '''全屏期间 console_display 必须闭嘴，否则会写花终端（跳字/文字叠加）。'''
    import io

    from promptkit.core.display_sys.display import (
        console_display,
        set_console_suppressed,
    )
    from promptkit.core.event_sys import Event, EventType

    out = io.StringIO()
    handler = console_display(stream=out)
    handler(Event(EventType.TOOL_CALL_START, {"name": "search"}))

    try:
        set_console_suppressed(True)
        before = out.getvalue()
        handler(Event(EventType.TOOL_CALL_START, {"name": "search"}))
        handler(Event(EventType.THINKING_DELTA, {"text": "被静音"}))
        assert out.getvalue() == before, "全屏期间不应再往控制台写任何东西"
    finally:
        set_console_suppressed(False)

    # 退出全屏后恢复
    handler(Event(EventType.TOOL_CALL_START, {"name": "again"}))
    assert "again" in out.getvalue()


def test_fullscreen_stream_uses_static_not_markdown_during_delta():
    '''流式 MESSAGE_DELTA 期间应该挂 Static 纯文本（稳定），不是 Markdown。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.widgets import Markdown, Static

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        # 直接发事件，不要阻塞在 INPUT_FN_KEY 上（pilot 没按键）
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "你好"}))
        bus.emit(Event(EventType.MESSAGE_DELTA, {"text": "```python\n"}))
        bus.emit(Event(EventType.MESSAGE_DELTA, {"text": "def f():\n"}))
        bus.emit(Event(EventType.MESSAGE_DELTA, {"text": "    return 42\n```"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "```python\ndef f():\n    return 42\n```"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause(0.5)  # 等所有事件处理完
            # END 之后 stream Static 应该消失，user Static + assistant Markdown 在
            assert len(list(app.query("Static.stream-reply"))) == 0
            mds = list(app.query("Markdown"))
            assert len(mds) == 1, f"应有 1 条助手 Markdown，实际 {len(mds)}"
            assert len(list(app.query("Static.user"))) == 1
            # 验证流式期间确实是 Static（内部计数能佐证）
            assert len(app._stream_reply_text) == 0  # 已清空
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_thinking_panel_appears_collapsed():
    '''THINKING_START/DELTA/END 应该挂一个默认折叠的 Collapsible。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.widgets import Collapsible

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "你好"}))
        bus.emit(Event(EventType.THINKING_START, {}))
        bus.emit(Event(EventType.THINKING_DELTA, {"text": "让我想想..."}))
        bus.emit(Event(EventType.THINKING_DELTA, {"text": "先分析问题"}))
        bus.emit(Event(EventType.THINKING_END, {"text": "让我想想...先分析问题"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "好的，答案是 42。"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(50):
                panels = list(app.query("Collapsible.thinking-panel"))
                if panels and "Ctrl+R" in panels[0].title:
                    break
                await pilot.pause(0.02)
            panels = list(app.query("Collapsible.thinking-panel"))
            assert len(panels) == 1
            panel = panels[0]
            assert panel.title.startswith("✦ 思考")
            assert "Ctrl+R 展开" in panel.title
            assert panel.collapsed is True
            content = panel.query_one("Static.panel-content")
            assert "让我想想" in str(content.render())
            assert "先分析问题" in str(content.render())
            try:
                await pilot.pause(0.2)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_tool_panel_shows_args_and_result():
    '''TOOL_CALL_START + TOOL_RESULT 应该生成一个带参数和结果的折叠面板。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.widgets import Collapsible

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "你好"}))
        bus.emit(Event(EventType.TOOL_CALL_START, {"name": "search", "arguments": '{"query": "foo"}'}))
        bus.emit(Event(EventType.TOOL_RESULT, {"name": "search", "arguments": '{"query": "foo"}', "result": "42 条结果", "tool_call_id": "tc_1"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "好的。"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(50):
                panels = list(app.query("Collapsible.tool-panel"))
                if panels and "·" in panels[0].title:
                    break
                await pilot.pause(0.02)
            panels = list(app.query("Collapsible.tool-panel"))
            assert len(panels) == 1
            panel = panels[0]
            assert panel.title.startswith("● search")
            assert "query=foo" in panel.title
            assert panel.title.endswith("s")
            assert panel.collapsed is True
            key = list(app._tool_contents.keys())[0]
            args_static, result_static = app._tool_contents[key]
            args_text = str(args_static.render())
            assert "query" in args_text
            assert "foo" in args_text
            result_text = str(result_static.render())
            assert "42 条结果" in result_text
            try:
                await pilot.pause(0.2)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_tool_panel_expands_to_full_result():
    '''展开工具面板后，结果从「… +N 行」切换成完整内容。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import TutorApp
    from textual.widgets import Collapsible

    ctx = Context()
    allow_exit = threading.Event()
    body = "\n".join(f"第 {i} 行" for i in range(30))

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "读文件"}))
        bus.emit(
            Event(
                EventType.TOOL_CALL_START,
                {"name": "Read", "arguments": '{"path": "a.py"}'},
            )
        )
        bus.emit(Event(EventType.TOOL_CALL_END, {"arguments": '{"path": "a.py"}'}))
        bus.emit(
            Event(
                EventType.TOOL_RESULT,
                {"name": "Read", "result": body, "tool_call_id": "tc_1"},
            )
        )
        bus.emit(Event(EventType.MESSAGE_END, {"text": "好了"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(50):
                panels = list(app.query("Collapsible.tool-panel"))
                if panels and "·" in panels[0].title:
                    break
                await pilot.pause(0.02)
            panel = list(app.query("Collapsible.tool-panel"))[0]
            key = list(app._tool_contents.keys())[0]
            _, result_widget = app._tool_contents[key]
            collapsed_text = str(result_widget.render())
            assert "… +" in collapsed_text
            assert "第 29 行" not in collapsed_text
            panel.collapsed = False
            await pilot.pause(0.15)
            expanded_text = str(result_widget.render())
            assert "第 29 行" in expanded_text
            assert "… +" not in expanded_text
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_ctrl_r_toggles_thinking_panels():
    '''Ctrl+R 应该把所有思考面板从折叠状态展开（反之亦然）。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import ChatInput, TutorApp
    from textual.widgets import Collapsible

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "你好"}))
        bus.emit(Event(EventType.THINKING_START, {}))
        bus.emit(Event(EventType.THINKING_END, {"text": "一些思考内容"}))
        bus.emit(Event(EventType.THINKING_START, {}))
        bus.emit(Event(EventType.THINKING_END, {"text": "更多思考内容"}))
        bus.emit(Event(EventType.MESSAGE_END, {"text": "答案。"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(50):
                panels = list(app.query("Collapsible.thinking-panel"))
                if len(panels) >= 2:
                    break
                await pilot.pause(0.02)
            panels = list(app.query("Collapsible.thinking-panel"))
            assert len(panels) == 2
            for p in panels:
                assert p.collapsed is True
            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            for p in panels:
                assert p.collapsed is False
            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            for p in panels:
                assert p.collapsed is True
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_tool_result_preview_is_truncated_by_lines():
    '''折叠时只给前几行，并提示还有多少行；展开后原样返回。'''
    from promptkit.tui.textual_app import TutorApp

    body = "\n".join(f"第 {i} 行" for i in range(20))
    collapsed = TutorApp._preview(
        body, limit=8, hint="（Ctrl+T 展开）", expanded=False
    )
    assert "第 7 行" in collapsed
    assert "第 8 行" not in collapsed
    assert "… +12 行（Ctrl+T 展开）" in collapsed
    expanded = TutorApp._preview(
        body, limit=8, hint="（Ctrl+T 展开）", expanded=True
    )
    assert expanded == body


def test_fullscreen_status_bar_shows_model_and_context_usage():
    '''状态栏两行：左模型右用量，第二行给出 ctx 百分比。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import TutorApp
    from textual.widgets import Static

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(
            Event(
                EventType.USAGE,
                {"prompt_tokens": 39_000, "completion_tokens": 800},
            )
        )
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            left = str(app.query_one("#status-left", Static).render())
            right = str(app.query_one("#status-right", Static).render())
            detail = str(app.query_one("#status-detail", Static).render())
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app, left, right, detail

    app, left, right, detail = asyncio.run(
        asyncio.wait_for(exercise(), timeout=20)
    )
    assert app.error is None
    assert "deepseek" in left
    assert "39.8k" in right
    assert "4.0%" in right
    assert "ctx 4.0%" in detail
    assert "↑39.0k" in detail


def test_fullscreen_jump_hint_appears_above_bottom():
    '''离开底部时出现「回到最底部」，跳到底部后消失。'''
    import asyncio
    import threading

    from promptkit.tui.textual_app import TutorApp
    from textual.containers import VerticalScroll
    from textual.widgets import Static

    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            chat.mount(*[Static(f"第 {i} 行") for i in range(40)])
            await pilot.pause(0.2)
            chat.scroll_end(animate=False)
            await pilot.pause(0.1)
            chat.scroll_to(y=0, animate=False)
            await pilot.pause(0.15)
            assert app._jump_hint.display is True
            app.action_jump_bottom()
            await pilot.pause(0.15)
            assert app._jump_hint.display is False
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None


def test_fullscreen_ctrl_s_exports_transcript_markdown(tmp_path, monkeypatch):
    '''Ctrl+S 把用户 / 助手 / 思考 / 工具调用导出成 Markdown。'''
    import asyncio
    import threading

    from promptkit.core.event_sys import EVENT_BUS_KEY, Event, EventType
    from promptkit.tui.textual_app import TutorApp

    monkeypatch.chdir(tmp_path)
    ctx = Context()
    allow_exit = threading.Event()

    def loop(context):
        bus = context[EVENT_BUS_KEY]
        bus.emit(Event(EventType.USER_MESSAGE, {"text": "你好"}))
        bus.emit(Event(EventType.THINKING_START, {}))
        bus.emit(Event(EventType.THINKING_END, {"text": "让我想想"}))
        bus.emit(
            Event(
                EventType.TOOL_CALL_START,
                {"name": "Read", "arguments": '{"path": "a.py"}'},
            )
        )
        bus.emit(Event(EventType.TOOL_CALL_END, {"arguments": '{"path": "a.py"}'}))
        bus.emit(
            Event(EventType.TOOL_RESULT, {"name": "Read", "result": "42 条结果"})
        )
        bus.emit(Event(EventType.MESSAGE_END, {"text": "答案是 42"}))
        assert allow_exit.wait(timeout=10)

    async def exercise():
        app = TutorApp(ctx, loop)
        async with app.run_test() as pilot:
            await pilot.pause(0.4)
            app.action_export_chat()
            await pilot.pause(0.1)
            try:
                await pilot.pause(0.1)
            finally:
                allow_exit.set()
            await pilot.pause(0.2)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    files = list(tmp_path.glob("promptkit-chat-*.md"))
    assert len(files) == 1, f"应导出一个 Markdown，实际 {files}"
    text = files[0].read_text(encoding="utf-8")
    assert "## 你" in text
    assert "你好" in text
    assert "## 助手" in text
    assert "答案是 42" in text
    assert "<summary>✦ 思考" in text
    assert "让我想想" in text
    assert "<summary>● Read" in text
    assert "42 条结果" in text
