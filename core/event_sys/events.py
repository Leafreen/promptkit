# core/event_sys/events.py
'''事件定义。

核心库只负责 **发事件**，「要不要显示、怎么显示」由调用方的 handler 决定。
所以这里没有任何 print / sys.stdout。
'''
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from promptkit.core.context import Context


class EventType(str, Enum):
    '''事件类型。

    刻意继承 `str`：这样 `event.type == EventType.TOOL_RESULT` 和
    `event.type == "tool_result"` 都能成立——两种写法都很自然，
    少一个「忘了 .value 所以永远不相等」的坑。
    '''

    # ---- 模型的思考过程（DeepSeek 的 reasoning_content / o1 的 reasoning）----
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"

    # ---- 工具调用 ----
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"

    # ---- 给用户的回答 ----
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"

    # ---- 工作流步骤 ----
    STEP_START = "step_start"
    STEP_END = "step_end"

    # ---- 用量 ----
    #: 一次模型调用的 token 用量。终端默认不显示，日志处理器会记下来。
    USAGE = "usage"

    # ---- 出错 ----
    ERROR = "error"


# 也提供裸常量写法：from ...events import TOOL_RESULT
THINKING_START = EventType.THINKING_START
THINKING_DELTA = EventType.THINKING_DELTA
THINKING_END = EventType.THINKING_END
TOOL_CALL_START = EventType.TOOL_CALL_START
TOOL_CALL_DELTA = EventType.TOOL_CALL_DELTA
TOOL_CALL_END = EventType.TOOL_CALL_END
TOOL_RESULT = EventType.TOOL_RESULT
MESSAGE_DELTA = EventType.MESSAGE_DELTA
MESSAGE_END = EventType.MESSAGE_END
STEP_START = EventType.STEP_START
STEP_END = EventType.STEP_END
USAGE = EventType.USAGE
ERROR = EventType.ERROR

# 事件总线挂在 Context 里的键名
EVENT_BUS_KEY = "_event_bus"


@dataclass
class Event:
    '''一条事件：类型 + 数据。'''

    type: EventType
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 允许 Event("tool_result", {...}) 这种字符串写法
        if not isinstance(self.type, EventType):
            try:
                self.type = EventType(self.type)
            except ValueError:
                raise ValueError(
                    f"未知的事件类型 {self.type!r}；"
                    f"可用：{', '.join(t.value for t in EventType)}"
                ) from None

    def __repr__(self) -> str:
        return f"Event({self.type.value}, {self.data!r})"


def extract_reasoning(delta: Any) -> Optional[str]:
    '''从流式 delta 里取思考内容，兼容几种字段名：

    - DeepSeek：`reasoning_content`
    - OpenAI o 系列：`reasoning`
    - Claude extended thinking：`thinking`
    '''
    for attr in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def bus_of(context: Context) -> Any:
    '''取出挂在 Context 上的事件总线；没有就返回 None。'''
    getter = getattr(context, "get", None)
    return getter(EVENT_BUS_KEY) if callable(getter) else None


def emit(context: Context, event_type: Any, **data: Any) -> bool:
    '''往 context 上的总线发一条事件。

    没有总线时**什么都不做**，所以核心逻辑里可以放心到处写 emit。
    返回是否真的发出去了。
    '''
    bus = bus_of(context)
    if bus is None:
        return False
    bus.emit(Event(event_type, data))
    return True
