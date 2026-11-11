'''时间戳 / 计时器：`GET_TIME`、`TIMER`、到点提醒插进对话。

`ALARMS` 是进程级队列，所以每个用例结束都要清空 —— 否则一个用例里
没到点的计时器会在另一个用例调模型时突然冒出提醒。
'''
import re
import time
from unittest import mock

import pytest

from promptkit.agent import Agent
from promptkit.core.context import Context
from promptkit.core.steps.actions.chat_util import CHAT_TIME_KEY, ChatLoop
from promptkit.core.steps.actions.clock import (
    Now,
    ProactiveAlarms,
    StartTimer,
    StopTimer,
    TimeStamp,
    WaitAlarm,
)
from promptkit.core.tool import (
    ALARMS,
    GET_TIME,
    AlarmRunner,
    clock_text,
    drain_alarms,
    now_text,
    running_timers,
    start_timer,
    stop_all_timers,
    stop_timer,
    timer,
)

_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 周[一二三四五六日]$")
_ALARM_RE = re.compile(r"^<\d{2}:\d{2}>到了。$")


@pytest.fixture(autouse=True)
def _clean_timers():
    ALARMS.clear()
    stop_all_timers()
    yield
    stop_all_timers()
    ALARMS.clear()


def test_get_time_reports_date_and_weekday():
    assert _TIME_RE.match(GET_TIME.call())


def test_clock_text_is_short():
    assert re.match(r"^\d{2}-\d{2} \d{2}:\d{2}$", clock_text())
    assert "周" in now_text()


def test_timer_fires_a_reminder_message(capsys):
    start_timer(0.05, name="quick")
    deadline = time.monotonic() + 2
    while not ALARMS.peek() and time.monotonic() < deadline:
        time.sleep(0.01)

    messages = drain_alarms()
    assert len(messages) == 1
    assert _ALARM_RE.match(messages[0])
    assert messages[0] in capsys.readouterr().out


def test_stop_cancels_the_reminder():
    start_timer(5, name="later")
    assert "已停止" in stop_timer("later")
    time.sleep(0.05)
    assert drain_alarms() == []


def test_custom_message_is_used():
    timer("start", seconds=0.05, name="custom", message="时间到了，交卷！")
    deadline = time.monotonic() + 2
    while not ALARMS.peek() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert drain_alarms() == ["时间到了，交卷！"]


def test_timer_elapsed_and_list():
    timer("start", seconds=5, name="work")
    assert "已过" in timer("elapsed", name="work")
    assert "work" in timer("list")
    timer("stop", name="work")
    assert timer("list") == "没有正在运行的计时器"


def test_timer_rejects_unknown_action():
    with pytest.raises(ValueError):
        timer("stopp")


def test_agent_injects_pending_alarms_before_the_prompt():
    agent = Agent(api_key="k")
    ALARMS.push("<15:30>到了。")
    agent._drain_alarms()
    agent._append_message({"role": "user", "content": "我刚回来"})
    assert [message["content"] for message in agent.history] == [
        "<15:30>到了。",
        "我刚回来",
    ]


def test_agent_drains_alarms_once():
    agent = Agent(api_key="k")
    ALARMS.push("<15:30>到了。")
    agent._drain_alarms()
    agent._drain_alarms()
    assert len(agent.history) == 1


class _EchoAgent:
    vendor = "fake"
    model = "echo"

    def __init__(self):
        self.prompts = []

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        self.prompts.append(prompt)
        return "答"


def _run_loop(*steps, **kwargs):
    agent = _EchoAgent()
    step = ChatLoop(*steps, **kwargs)
    with mock.patch("builtins.input", side_effect=["你好", "exit"]):
        step.run(agent=agent)
    return agent, step


def test_chat_loop_stamps_user_messages():
    agent, step = _run_loop(timestamps=True)
    content = agent.prompts[0]["content"]
    assert re.match(r"^\[\d{2}-\d{2} \d{2}:\d{2}\] 你好$", content)
    assert step._chat_turn.prompt_content == "[{{chat_time}}] {{usr_input}}"


def test_chat_loop_without_timestamps_keeps_prompt_plain():
    agent, _ = _run_loop()
    assert agent.prompts[0]["content"] == "你好"


def test_chat_loop_accepts_timestamp_step():
    agent, step = _run_loop(TimeStamp())
    assert step._chat_turn.prompt_content == "[{{chat_time}}] {{usr_input}}"
    assert re.match(r"^\[\d{2}-\d{2} \d{2}:\d{2}\] 你好$", agent.prompts[0]["content"])


def test_chat_loop_custom_timestamp_key_keeps_prompt_plain():
    agent, step = _run_loop(TimeStamp("turn_time"))
    assert step._chat_turn.prompt_content == "{{usr_input}}"
    assert agent.prompts[0]["content"] == "你好"


def test_chat_loop_accepts_proactive_alarm_step():
    step = ChatLoop(ProactiveAlarms())
    assert step._proactive is True
    assert step._chat_turn.prompt_content == "{{usr_input}}"


def test_chat_time_key_is_exported():
    assert CHAT_TIME_KEY == "chat_time"


# ============================ 主动推送 ============================
def test_alarm_board_wait_returns_a_message():
    ALARMS.push("<15:30>到了。")
    assert ALARMS.wait(0.1) == "<15:30>到了。"
    assert ALARMS.wait(0.01) is None


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_alarm_runner_pushes_a_model_turn_immediately():
    agent = _EchoAgent()
    runner = AlarmRunner(agent, poll_interval=0.01).start()
    try:
        ALARMS.push("<15:30>到了。")
        assert _wait_for(lambda: agent.prompts)
    finally:
        runner.stop()
    assert agent.prompts == ["<15:30>到了。"]


def test_alarm_runner_stops_without_leaking_turns():
    agent = _EchoAgent()
    runner = AlarmRunner(agent, poll_interval=0.01).start()
    runner.stop()
    assert runner.running is False
    ALARMS.push("<15:31>到了。")
    time.sleep(0.1)
    assert agent.prompts == []


def test_alarm_runner_can_wrap_the_prompt():
    agent = _EchoAgent()
    runner = AlarmRunner(
        agent, prompt_template="（系统提醒）{alarm}", poll_interval=0.01
    ).start()
    try:
        ALARMS.push("<15:32>到了。")
        assert _wait_for(lambda: agent.prompts)
    finally:
        runner.stop()
    assert agent.prompts == ["（系统提醒）<15:32>到了。"]


def test_chat_loop_proactive_alarms_flag_is_propagated():
    step = ChatLoop(proactive_alarms=True)
    assert step._proactive is True
    assert ChatLoop()._proactive is False


# ============================ DSL 原语 ============================
def test_timestamp_step_writes_short_time():
    ctx = Context()
    TimeStamp().execute(ctx)
    assert re.match(r"^\d{2}-\d{2} \d{2}:\d{2}$", ctx[CHAT_TIME_KEY])
    assert ctx.trace[-1] == "timestamp"


def test_now_step_writes_full_time():
    ctx = Context()
    Now("clock").execute(ctx)
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 周[一二三四五六日]$", ctx["clock"])


def test_start_and_stop_timer_steps():
    ctx = Context()
    StartTimer(5, name="dsl", output_key="started").execute(ctx)
    assert "dsl" in ctx["started"]
    assert [item["name"] for item in running_timers()] == ["dsl"]

    StopTimer("dsl", output_key="stopped").execute(ctx)
    assert "已停止" in ctx["stopped"]
    assert running_timers() == []


def test_wait_alarm_step_receives_a_reminder():
    ctx = Context()
    StartTimer(0.02, name="dsl-wait", message="DSL 提醒。").execute(ctx)
    WaitAlarm(timeout=1, output_key="alarm").execute(ctx)
    assert ctx["alarm"] == "DSL 提醒。"
    assert ctx.trace[-1] == "wait_alarm"


def test_wait_alarm_step_times_out_quietly():
    ctx = Context()
    WaitAlarm(timeout=0.01, output_key="alarm").execute(ctx)
    assert ctx["alarm"] == ""
    assert ctx.trace[-1] == "wait_alarm:timeout"


def test_clock_steps_are_exported_from_promptkit_core():
    from promptkit.core import (
        Now as CoreNow,
        ProactiveAlarms as CoreProactive,
        StartTimer as CoreStart,
        StopTimer as CoreStop,
        TimeStamp as CoreStamp,
        WaitAlarm as CoreWait,
    )

    assert (CoreNow, CoreStart, CoreStop, CoreStamp, CoreWait, CoreProactive) == (
        Now,
        StartTimer,
        StopTimer,
        TimeStamp,
        WaitAlarm,
        ProactiveAlarms,
    )
