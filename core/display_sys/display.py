# core/display_sys/display.py
'''开箱即用的事件处理器。

核心库不 print；「要不要显示」全在这里，按需挂到 EventBus 上：

    bus = EventBus()
    bus.on(console_display(show_thinking=True, show_tools=True))
'''
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from promptkit.core.event_sys.events import Event, EventType

__all__ = [
    "console_display",
    "json_display",
    "collect_events",
    "filter_events",
]


# 终端颜色
_DIM = "\033[90m"
_RESET = "\033[0m"


def _short(value: Any, limit: int = 200) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def console_display(
    show_thinking: bool = True,
    show_tools: bool = True,
    show_message: bool = True,
    show_steps: bool = False,
    show_errors: bool = True,
    stream: Any = None,
) -> Callable[[Event], None]:
    '''返回一个打印到控制台的事件处理器。

    - `show_thinking`：灰色的思考过程
    - `show_tools`：工具名、参数、结果
    - `show_message`：最终回答（流式就边到边打）
    - `show_steps`：每个 Step 的开始 / 结束
    - `show_errors`：ERROR 事件
    - `stream`：默认写到 sys.stdout，可以换成别的流
    '''
    out = stream if stream is not None else sys.stdout
    # 只有真的在终端上才用颜色，重定向到文件 / 管道时不撒转义字符
    color = bool(getattr(out, "isatty", None) and out.isatty())
    dim, reset = (_DIM, _RESET) if color else ("", "")
    thinking_open = False  # 思考块是否正在输出（用来配颜色）
    streamed = False  # 这一条消息是否已经流式打过了

    def write(text: str) -> None:
        out.write(text)
        out.flush()

    def handler(event: Event) -> None:
        nonlocal thinking_open, streamed
        etype = event.type
        data = event.data

        if etype == EventType.THINKING_START:
            if show_thinking and not thinking_open:
                write(dim)
                thinking_open = True

        elif etype == EventType.THINKING_DELTA:
            if show_thinking:
                if not thinking_open:
                    write(dim)
                    thinking_open = True
                write(str(data.get("text", "")))

        elif etype == EventType.THINKING_END:
            if show_thinking and thinking_open:
                write(reset + "\n")
                thinking_open = False

        elif etype == EventType.TOOL_CALL_START:
            if show_tools:
                if thinking_open:
                    write(reset + "\n")
                    thinking_open = False
                print(f"\n🔧 调用工具: {data.get('name', '?')}", file=out)

        elif etype == EventType.TOOL_CALL_END:
            # 参数在 END 才是完整的（流式是一段段拼出来的），只在这里打一次
            if show_tools and data.get("arguments"):
                print(f"   参数: {data['arguments']}", file=out)

        elif etype == EventType.TOOL_RESULT:
            if show_tools:
                print(f"   结果: {_short(data.get('result', ''))}", file=out)

        elif etype == EventType.MESSAGE_DELTA:
            streamed = True
            if show_message:
                write(str(data.get("text", "")))

        elif etype == EventType.MESSAGE_END:
            if show_message and not streamed:
                # 非流式：内容是一次性给的，这里补打
                print(data.get("text", ""), file=out)
            elif show_message and streamed:
                write("\n")
            streamed = False

        elif etype == EventType.STEP_START:
            if show_steps:
                print(f"▶ {data.get('step', '?')}", file=out)

        elif etype == EventType.STEP_END:
            if show_steps:
                print(f"✔ {data.get('step', '?')}", file=out)

        elif etype == EventType.ERROR:
            if show_errors:
                if thinking_open:
                    write(_RESET + "\n")
                    thinking_open = False
                print(f"✖ {data.get('step', '')} {data.get('error', '')}", file=out)


    return handler


def collect_events() -> Tuple[Callable[[Event], None], List[Event]]:
    '''返回 (handler, events)：把事件收集到列表里，测试 / 断言用。'''
    events: List[Event] = []

    def handler(event: Event) -> None:
        events.append(event)

    return handler, events


def json_display(stream: Any = None, indent: Optional[int] = None) -> Callable[[Event], None]:
    '''一行一条 JSON，适合写日志或转发到 WebSocket / SSE。'''
    out = stream if stream is not None else sys.stdout

    def handler(event: Event) -> None:
        payload: Dict[str, Any] = {"type": event.type.value, **event.data}
        out.write(json.dumps(payload, ensure_ascii=False, indent=indent) + "\n")
        out.flush()

    return handler


def filter_events(*types: Any) -> Callable[[Event], bool]:
    '''判断一条事件是不是给定类型之一（`types` 可以是 EventType 或字符串）。'''
    wanted = {EventType(t) if not isinstance(t, EventType) else t for t in types}
    return lambda event: event.type in wanted
