# core/steps/actions/clock.py
'''时间与计时器的 DSL 原语。

    from promptkit.core import Now, TimeStamp, StartTimer, StopTimer, WaitAlarm

    Workflow(
        TimeStamp(),                    # -> chat_time = "11-09 15:30"
        StartTimer(600, name="contest"),# 10 分钟后提醒
        ChatLoop(proactive_alarms=True),# 到点立刻让模型说一句
        StopTimer("contest"),
    )

这些步骤只是把 `GET_TIME` / `TIMER` 背后的能力搬进步骤树：不依赖模型是否
愿意调用工具，写在工作流里就一定执行。

| 原语 | 做什么 |
| --- | --- |
| `TimeStamp(output_key="chat_time")` | 写短时间戳（`11-09 15:30`），适合塞进 prompt |
| `Now(output_key="now")` | 写完整时间（`2026-11-09 15:30:12 周日`） |
| `StartTimer(seconds, name=..., message=...)` | 起计时器；到点提醒进 `ALARMS` |
| `StopTimer(name=...)` | 停计时器 |
| `WaitAlarm(timeout=None, output_key="alarm")` | 阻塞等一条提醒（工作流版「等闹钟」） |
'''
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 context
    from promptkit.core.context import Context

from promptkit.core.steps.actions.chat_util import CHAT_TIME_KEY
from promptkit.core.steps.interface import Step
from promptkit.core.tool.builtin.clock import (
    ALARMS,
    clock_text,
    get_time,
    start_timer,
    stop_timer,
)

__all__ = [
    "Now",
    "ProactiveAlarms",
    "TimeStamp",
    "StartTimer",
    "StopTimer",
    "WaitAlarm",
]


class TimeStamp(Step):
    '''把「现在几点」写进 Context（短格式，适合塞进 prompt）。

        TimeStamp()                 # chat_time = "11-09 15:30"
        TimeStamp("turn_time")      # 换个变量名

    放在 `ChatLoop` 的每轮步骤里、配合 `turn_template="{{chat_time}} ..."`，
    就是「每条消息都带时间」的手写版。
    '''

    name = "timestamp"

    def __init__(self, output_key: str = CHAT_TIME_KEY):
        self.output_key = output_key

    def execute(self, context: Context) -> None:
        context[self.output_key] = clock_text()
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"TimeStamp(-> {self.output_key})"


class Now(Step):
    '''把完整的当前时间（含星期）写进 Context。

        Now()                       # now = "2026-11-09 15:30:12 周日"
        Now("now", timezone="Asia/Tokyo")
    '''

    name = "now"

    def __init__(self, output_key: str = "now", timezone: str = ""):
        self.output_key = output_key
        self.timezone = timezone

    def execute(self, context: Context) -> None:
        context[self.output_key] = get_time(self.timezone)
        context.trace.append(self.name)

    def describe(self) -> str:
        suffix = f", {self.timezone}" if self.timezone else ""
        return f"Now(-> {self.output_key}{suffix})"


class StartTimer(Step):
    '''起一个计时器；到点后提醒进入 `ALARMS`。

        StartTimer(600, name="contest")          # 10 分钟
        StartTimer(60, name="tea", message="茶泡好了。")

    - `name` 是计时器名字（同名会覆盖旧的），不是步骤名。
    - `output_key` 给了就把「已启动…」这句话写进 Context。
    - 想让模型**立刻**收到提醒，用 `ChatLoop(proactive_alarms=True)`
      或 `WaitAlarm()` 等它。
    '''

    name = "start_timer"

    def __init__(
        self,
        seconds: float,
        name: str = "default",
        message: str = "",
        output_key: Optional[str] = None,
    ):
        self.seconds = seconds
        self.timer_name = name
        self.message = message
        self.output_key = output_key

    def execute(self, context: Context) -> None:
        result = start_timer(
            self.seconds, name=self.timer_name, message=self.message
        )
        if self.output_key is not None:
            context[self.output_key] = result
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"StartTimer({self.timer_name!r}, {self.seconds:g}s)"


class StopTimer(Step):
    '''停掉一个计时器（不存在的名字不算错）。

        StopTimer("contest")
    '''

    name = "stop_timer"

    def __init__(self, name: str = "default", output_key: Optional[str] = None):
        self.timer_name = name
        self.output_key = output_key

    def execute(self, context: Context) -> None:
        result = stop_timer(self.timer_name)
        if self.output_key is not None:
            context[self.output_key] = result
        context.trace.append(self.name)

    def describe(self) -> str:
        return f"StopTimer({self.timer_name!r})"


class WaitAlarm(Step):
    '''阻塞等一条到点提醒，把提醒文本写进 Context。

        StartTimer(2, name="nap",
                   message="两秒到了。")
        WaitAlarm(timeout=5, output_key="alarm")   # alarm = "两秒到了。"
        Echo(Var("alarm"))

    `timeout=None` 表示一直等；超时没等到就写空串（**不抛异常**，
    工作流照常往下走），trace 里记一条 `wait_alarm:timeout`。
    '''

    name = "wait_alarm"

    def __init__(
        self,
        timeout: Optional[float] = None,
        output_key: str = "alarm",
    ):
        self.timeout = timeout
        self.output_key = output_key

    def execute(self, context: Context) -> None:
        alarm = ALARMS.wait(self.timeout)
        context[self.output_key] = alarm or ""
        if alarm is None:
            context.trace.append(f"{self.name}:timeout")
        else:
            context.trace.append(self.name)

    def describe(self) -> str:
        wait = "直到到点" if self.timeout is None else f"最多 {self.timeout:g}s"
        return f"WaitAlarm({wait} -> {self.output_key})"


class ProactiveAlarms(Step):
    '''标记步骤：对话期间到点就立刻让模型说一句。

        ChatLoop(ProactiveAlarms())              # 等价于 proactive_alarms=True
        ChatWorkflow(behavior=[ProactiveAlarms()])

    它本身每轮什么都不做（只留一条 trace）；`ChatLoop` 在组装时会认出它，
    把「起停后台 `AlarmRunner`」挂到整个循环上。放在循环体里只是为了
    写法统一 —— 和 `Compact(...)` 一样，用步骤表达配置。
    '''

    name = "proactive_alarms"

    def execute(self, context: Context) -> None:
        context.trace.append(self.name)

    def describe(self) -> str:
        return "ProactiveAlarms()"
