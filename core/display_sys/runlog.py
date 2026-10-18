# core/display_sys/runlog.py
'''把过程写进文件 —— 终端给人看，日志给排查用。

同一份事件可以挂两个处理器，互不影响：

    wf.run(
        agent=agent,
        verbose=Verbosity.NORMAL,                 # 终端：只有对话
        on_event=log_display("logs/run.log"),     # 文件：全量
    )

`log_display` 返回的是普通的事件处理器（`Callable[[Event], None]`），
所以也能直接挂到 `Agent.on(...)` 上，或者用 `EventBus.on_all` 挂一串。

**为什么不往每个 Step 里塞 `logger.debug`**：那样每加一个步骤就要记得加日志，
而且拿不到流式增量。事件总线本来就带全了（步骤、工具、思考、用量），
在这里集中格式化一次就够了。
'''
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence, Set

from promptkit.core.event_sys.events import Event, EventType

__all__ = [
    "LOGGER_NAME",
    "get_logger",
    "setup_logging",
    "close_logging",
    "log_display",
    "DEFAULT_LOG_PATH",
    "DEFAULT_FORMAT",
]

LOGGER_NAME = "promptkit"

#: 不传路径时的默认日志位置
DEFAULT_LOG_PATH = os.path.join("logs", "promptkit.log")

#: 日志行格式：时间和级别，方便 grep
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

#: 事件 -> 日志级别
_LEVELS: Dict[Any, int] = {
    EventType.ERROR: logging.ERROR,
    EventType.TOOL_RESULT: logging.INFO,
    EventType.MESSAGE_END: logging.INFO,
    EventType.USAGE: logging.INFO,
    EventType.TOOL_CALL_END: logging.DEBUG,
    EventType.STEP_START: logging.DEBUG,
    EventType.STEP_END: logging.DEBUG,
    EventType.THINKING_END: logging.DEBUG,
}

#: 默认**不**记的（太吵，而且内容已经包含在别的行里）
_NOISY = {
    EventType.THINKING_START,
    EventType.THINKING_DELTA,
    EventType.MESSAGE_DELTA,
    EventType.TOOL_CALL_START,
    EventType.TOOL_CALL_DELTA,
}


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    '''拿 promptkit 的 logger。没配过也能用（会走 lastResort，什么也不打）。'''
    return logging.getLogger(name)


def setup_logging(
    path: str = DEFAULT_LOG_PATH,
    *,
    level: int = logging.DEBUG,
    fmt: str = DEFAULT_FORMAT,
    logger: Optional[logging.Logger] = None,
    replace: bool = True,
    encoding: str = "utf-8",
) -> logging.Logger:
    '''把日志写到 `path`（父目录自动创建），返回 logger。

    默认 `replace=True`：先把之前 `setup_logging` 加的处理器摘掉再挂新的，
    免得同一个文件被写两遍、或者重复调用时处理器越挂越多。
    '''
    target = logger if logger is not None else get_logger()
    target.setLevel(level)

    if replace:
        for handler in list(target.handlers):
            if getattr(handler, "_promptkit_file", False):
                target.removeHandler(handler)
                handler.close()

    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    handler = logging.FileHandler(path, encoding=encoding)
    handler.setFormatter(logging.Formatter(fmt))
    handler._promptkit_file = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    return target


def close_logging(logger: Optional[logging.Logger] = None) -> None:
    '''关掉并摘掉 `setup_logging` 加的处理器（测试收尾用）。'''
    target = logger if logger is not None else get_logger()
    for handler in list(target.handlers):
        if getattr(handler, "_promptkit_file", False):
            target.removeHandler(handler)
            handler.close()


def _format(event: Event) -> str:
    '''把一个事件变成一行日志。'''
    etype = event.type
    data = event.data or {}

    if etype == EventType.STEP_START:
        return f"step_start name={data.get('name', '?')}"
    if etype == EventType.STEP_END:
        parts = [f"step_end name={data.get('name', '?')}"]
        if data.get("duration_ms") is not None:
            parts.append(f"duration_ms={data['duration_ms']}")
        if data.get("error"):
            parts.append(f"error={data['error']}")
        return " ".join(parts)
    if etype == EventType.TOOL_CALL_END:
        return f"tool_call name={data.get('name', '?')} args={data.get('arguments', '')}"
    if etype == EventType.TOOL_RESULT:
        text = str(data.get("result", ""))
        return f"tool_result name={data.get('name', '?')} chars={len(text)}"
    if etype == EventType.MESSAGE_END:
        return f"message_end chars={len(str(data.get('text', '')))}"
    if etype == EventType.THINKING_END:
        return f"thinking chars={len(str(data.get('text', '')))}"
    if etype == EventType.USAGE:
        return (
            "usage "
            f"in={data.get('prompt_tokens')} "
            f"out={data.get('completion_tokens')} "
            f"total={data.get('total_tokens')}"
        )
    if etype == EventType.ERROR:
        return f"error where={data.get('name', '?')} {data.get('error', '')}"

    # 其余事件：把 data 里的标量列出来
    flat = " ".join(
        f"{key}={value}"
        for key, value in sorted(data.items())
        if isinstance(value, (str, int, float, bool, type(None)))
    )
    return f"{etype.value} {flat}".strip()


def log_display(
    path: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    skip: Sequence[Any] = (),
    include_thinking: bool = False,
) -> Callable[[Event], None]:
    '''返回一个把事件写进日志文件的处理器。

    - `path`：日志文件；给了就顺手 `setup_logging(path)`，不给就用现有 logger
    - `skip`：额外想跳过的事件类型（默认已经跳过高频的流式增量）
    - `include_thinking`：连思考的逐字增量也记（很大，默认关）
    '''
    if path is not None:
        setup_logging(path, level=level)

    active = logger if logger is not None else get_logger()
    ignored: Set[Any] = set(_NOISY) | {EventType.TOOL_CALL_START}
    if not include_thinking:
        ignored |= {EventType.THINKING_DELTA}
    ignored -= set(skip) if skip else set()
    explicit_skip = set(skip)

    def handler(event: Event) -> None:
        etype = event.type
        if etype in explicit_skip:
            return
        if etype in ignored:
            return
        active.log(_LEVELS.get(etype, logging.DEBUG), _format(event))

    return handler
