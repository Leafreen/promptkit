'''EnableTUI 的接线：标记步骤的摘除、输入注入、环境不满足时的显式失败。'''
import pytest

from promptkit.core import (
    ChatLoop,
    Context,
    Echo,
    EnableTUI,
    Read,
    TuiUnavailableError,
)
from promptkit.core.errors import WorkflowError
from promptkit.core.steps.actions.interactive import INPUT_FN_KEY
from promptkit.core.steps.actions.llm import LLMStep, LLM_STREAM_KEY
from promptkit.prompt import Prompt


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
                assert len(chat.query(Markdown)) == 2
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
            sources = [str(getattr(w, "source", "")) for w in chat.query(Markdown)]
            hold.set()
            await pilot.pause(0.2)
        return app, sources

    app, sources = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert joined == ["第一行\n第二行\n第三行"]
    assert sources == [
        "**你**\n\n第一行\n第二行\n第三行",
        "**助手**\n\n收到",
    ]


def test_fullscreen_renders_math_in_messages():
    '''助手回复里的 LaTeX 公式在进 Markdown 控件前先转成 Unicode。'''
    from promptkit.tui.textual_app import TutorApp

    text = TutorApp._message_markdown(r"其中 $\beta_0$ 是截距", user=False)
    assert text == "**助手**\n\n其中 `β₀` 是截距"


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


def test_fullscreen_ctrl_c_wakes_blocked_input():
    '''Ctrl+C 必须走 EOF 路径唤醒阻塞的输入，而不是让应用直接退出把 worker 挂死。'''
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
            assert app.query_one("#input", ChatInput).disabled is False
            await pilot.press("ctrl+c")
            await pilot.pause(0.3)
        return app

    app = asyncio.run(asyncio.wait_for(exercise(), timeout=20))
    assert app.error is None
    assert outcome == ["eof"]
    assert ctx.get(INPUT_FN_KEY) is None
