'''时间与计时器：让模型知道「现在几点」，也能定时把自己叫醒。

- `GET_TIME`：查当前日期 / 时间 / 星期（模型可调用）
- `TIMER`：起一个计时器；到点会往对话里插一条 `<时间>到了。`
- `ALARMS`：到点消息的队列，`Agent.ask()` 每次调模型前会把它取空，
  把里面的消息作为 user 消息插进历史（所以模型能在下一轮「被提醒」）
- `AlarmRunner`：后台线程版的「主动推送」—— 一到点立刻让模型跑一轮，
  不用等用户下一次输入（`ChatLoop(proactive_alarms=True)` 会自动用它）

不想让模型自己调也行，Python 侧直接用：

    from promptkit.core.tool import AlarmRunner, start_timer, clock_text

    start_timer(300, name="contest")      # 5 分钟后提醒
    AlarmRunner(agent).start()            # 到点立刻让模型说一句
'''
from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Dict, List, Optional

from promptkit.core.tool.base import Tool

_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def clock_text(moment: Optional[datetime] = None) -> str:
    '''`11-09 15:30`：聊天气泡 / 历史里的时间戳。'''
    return f"{moment or datetime.now():%m-%d %H:%M}"


def now_text(moment: Optional[datetime] = None) -> str:
    '''`2026-11-09 15:30:12 周日`：给模型看的完整时间。'''
    moment = moment or datetime.now()
    return f"{moment:%Y-%m-%d %H:%M:%S} {_WEEKDAYS[moment.weekday()]}"


def get_time(timezone: str = "") -> str:
    '''查询当前的日期、时间和星期。

    Args:
        timezone: IANA 时区名，例如 "Asia/Shanghai"；留空用本机时区。
    '''
    moment = datetime.now(_zone(timezone)) if timezone else datetime.now()
    return now_text(moment)


def _zone(name: str) -> tzinfo:
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


GET_TIME = Tool.from_function(get_time, name="get_time")


# ============================ 到点消息队列 ============================
class AlarmBoard:
    '''到点消息的线程安全队列：定时器线程写，模型调用前读空。'''

    def __init__(self) -> None:
        self._items: List[str] = []
        self._lock = threading.Lock()
        #: 有消息时唤醒正在 `wait()` 的后台线程（AlarmRunner）
        self._arrived = threading.Condition(self._lock)

    def push(self, text: str) -> None:
        with self._arrived:
            self._items.append(text)
            self._arrived.notify_all()

    def drain(self) -> List[str]:
        with self._arrived:
            items, self._items = self._items, []
        return items

    def peek(self) -> List[str]:
        with self._arrived:
            return list(self._items)

    def wait(self, timeout: Optional[float] = None) -> Optional[str]:
        '''阻塞等到一条到点消息；超时返回 `None`。

        和 `drain()` 二选一：谁先取到就归谁，提醒不会重复投递。
        '''
        with self._arrived:
            if not self._items:
                self._arrived.wait(timeout)
            if self._items:
                return self._items.pop(0)
            return None

    def clear(self) -> None:
        with self._arrived:
            self._items.clear()


#: 全局队列（单进程 CLI 够用；需要隔离就自己 new 一个 AlarmBoard）
ALARMS = AlarmBoard()


def drain_alarms() -> List[str]:
    '''取走到点的提醒消息（`Agent.ask` 用它插进对话）。'''
    return ALARMS.drain()


# ============================ 计时器 ============================
_TIMERS: Dict[str, dict] = {}
_TIMERS_LOCK = threading.RLock()


def _alarm_text(moment: datetime) -> str:
    return f"<{moment:%H:%M}>到了。"


def _fire(name: str, message: str) -> None:
    '''计时器到点：移出登记表，写进队列，顺手打印让用户立刻看到。'''
    with _TIMERS_LOCK:
        _TIMERS.pop(name, None)
    text = message or _alarm_text(datetime.now())
    ALARMS.push(text)
    print(f"\n{text}", flush=True)


def start_timer(seconds: float, name: str = "default", message: str = "") -> str:
    '''启动（或重设）一个计时器；到点往对话里插一条提醒。'''
    if seconds <= 0:
        raise ValueError(f"seconds 必须大于 0，得到 {seconds!r}")
    expected = datetime.now() + timedelta(seconds=seconds)
    text = message or _alarm_text(expected)
    with _TIMERS_LOCK:
        old = _TIMERS.pop(name, None)
        if old is not None:
            old["handle"].cancel()
        handle = threading.Timer(seconds, _fire, args=(name, message))
        handle.daemon = True
        _TIMERS[name] = {
            "handle": handle,
            "started": _time.monotonic(),
            "seconds": seconds,
            "message": text,
        }
        handle.start()
    return f"计时器 {name!r} 已启动：{seconds:g} 秒后提醒（{text}）"


def stop_timer(name: str = "default") -> str:
    '''停掉一个计时器（不会发出提醒）。'''
    with _TIMERS_LOCK:
        entry = _TIMERS.pop(name, None)
    if entry is None:
        return f"没有正在运行的计时器 {name!r}"
    entry["handle"].cancel()
    return f"计时器 {name!r} 已停止（已过 {elapsed_seconds(name, entry):.1f} 秒）"


def elapsed_seconds(name: str = "default", entry: Optional[dict] = None) -> float:
    '''这个计时器已经走了多少秒。'''
    with _TIMERS_LOCK:
        entry = entry or _TIMERS.get(name)
        if entry is None:
            raise KeyError(name)
        return _time.monotonic() - entry["started"]


def running_timers() -> List[dict]:
    '''正在跑的计时器：`[{name, remaining, seconds}]`。'''
    now = _time.monotonic()
    with _TIMERS_LOCK:
        return [
            {
                "name": name,
                "remaining": max(entry["seconds"] - (now - entry["started"]), 0.0),
                "seconds": entry["seconds"],
            }
            for name, entry in _TIMERS.items()
        ]


def stop_all_timers() -> None:
    '''清空所有计时器（测试 / 退出时用）。'''
    with _TIMERS_LOCK:
        entries = list(_TIMERS.values())
        _TIMERS.clear()
    for entry in entries:
        entry["handle"].cancel()


def timer(
    action: str,
    seconds: float = 0,
    name: str = "default",
    message: str = "",
) -> str:
    '''设置一个计时器；到点会往对话里插入一条「<时间>到了。」的提醒。

    Args:
        action: start 启动 / stop 停止 / elapsed 已过秒数 / list 查看在跑的
        seconds: start 时用，提醒前等待的秒数
        name: 计时器名字（同名会覆盖旧的）
        message: 自定义到点消息；留空用「<时间>到了。」
    '''
    if action == "start":
        return start_timer(seconds, name=name, message=message)
    if action == "stop":
        return stop_timer(name)
    if action == "elapsed":
        try:
            return f"计时器 {name!r} 已过 {elapsed_seconds(name):.1f} 秒"
        except KeyError:
            return f"没有正在运行的计时器 {name!r}"
    if action == "list":
        items = running_timers()
        if not items:
            return "没有正在运行的计时器"
        return "\n".join(
            f"{item['name']}: 还剩 {item['remaining']:.0f} 秒 / 共 {item['seconds']:g} 秒"
            for item in items
        )
    raise ValueError(f"不认识的 action：{action!r}（start / stop / elapsed / list）")


TIMER = Tool.from_function(timer, name="timer")


# ============================ 主动推送 ============================
class AlarmRunner:
    '''后台守护线程：一到点就立刻让模型跑一轮，不等用户输入。

        runner = AlarmRunner(agent).start()
        ...                                  # 用户该干嘛干嘛
        runner.stop()

    `ChatLoop(proactive_alarms=True)` 会在对话期间自动起停它。

    - 线程只做一件事：从 `ALARMS` 取一条提醒，调 `agent.ask(提醒)`；
      真正的并发保护在 `Agent` 的回合锁里（同一时刻只有一轮模型调用）。
    - `prompt_template` 可换发给模型的内容，`{alarm}` 是提醒原文。
    - `Agent.ask()` 本来就会先取空队列，所以提醒**不会重复**：
      主线程先取到就由主线程带上，后台线程先取到就由后台线程推送。
    '''

    def __init__(
        self,
        agent: Any,
        *,
        prompt_template: Optional[str] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        poll_interval: float = 0.2,
    ) -> None:
        self.agent = agent
        self.prompt_template = prompt_template
        self.on_error = on_error
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> "AlarmRunner":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="promptkit-alarm-runner", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        '''停掉后台线程（正在跑的那一轮模型调用会先跑完）。'''
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            alarm = ALARMS.wait(self.poll_interval)
            if alarm is None:
                continue
            prompt = (
                self.prompt_template.format(alarm=alarm)
                if self.prompt_template
                else alarm
            )
            try:
                self.agent.ask(prompt)
            except BaseException as exc:  # 后台线程的异常不能吞掉主流程
                if self.on_error is not None:
                    self.on_error(exc)
                else:
                    print(
                        f"\n[计时器提醒失败] {type(exc).__name__}: {exc}",
                        flush=True,
                    )

    def __repr__(self) -> str:
        return f"<AlarmRunner running={self.running}>"


__all__ = [
    "ALARMS",
    "GET_TIME",
    "TIMER",
    "AlarmBoard",
    "AlarmRunner",
    "clock_text",
    "drain_alarms",
    "elapsed_seconds",
    "get_time",
    "now_text",
    "running_timers",
    "start_timer",
    "stop_all_timers",
    "stop_timer",
    "timer",
]
