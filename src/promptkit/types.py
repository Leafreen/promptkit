'''跨层共享的类型别名与协议。

这里只放「不依赖任何子包」的基础类型，避免循环导入：各层从这里取，
具体领域的 TypedDict（工具 schema、技能规格、编排任务）留在各自的包里。

约定
----
- JSON 边界用 `JSONValue`，不再用 `Any`。
- 编译期确实无法确定类型的动态边界用 `object`（Python 里最接近
  TypeScript `unknown` 的东西），取用时必须 `isinstance` 收窄或 `cast`。
- 鸭子类型协作者（记忆 / agent / 工具）用 `Protocol`，不要硬造假精确类型。
'''
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
)

from promptkit.core.event_sys import EventBus, EventHandler

if TYPE_CHECKING:  # 只用于类型标注，避免 types -> core.tool 的运行时循环
    from promptkit.core.tool.base import Tool

__all__ = [
    "JSONScalar",
    "JSONValue",
    "MessageRole",
    "ContentPart",
    "MessageContent",
    "ToolCallFunction",
    "ToolCall",
    "Message",
    "PromptMessage",
    "PromptInput",
    "RecallHit",
    "SessionPayload",
    "ToolLike",
    "SupportsWrite",
    "MemoryLike",
    "AgentLike",
]

# ---------------- JSON ----------------
JSONScalar: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]

# ---------------- 消息 ----------------
MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool", "function"]
ContentPart: TypeAlias = Dict[str, JSONValue]
MessageContent: TypeAlias = Union[str, None, List[ContentPart]]


class ToolCallFunction(TypedDict):
    '''工具调用里的函数部分：名字 + JSON 字符串参数。'''

    name: str
    arguments: str


class ToolCall(TypedDict):
    '''一条工具调用（OpenAI Chat Completions 形状）。'''

    id: str
    type: str
    function: ToolCallFunction


class _MessageRequired(TypedDict):
    role: MessageRole
    content: MessageContent


class Message(_MessageRequired, total=False):
    '''一条对话消息。

    `role` / `content` 必有；其余是可选键 —— 用 `total=False` 继承，
    因为 Python 3.10 还没有 `NotRequired`。
    '''

    tool_calls: List[ToolCall]
    tool_call_id: str
    name: str
    reasoning_content: str


class PromptMessage(TypedDict, total=False):
    '''调用方传进来的「一条提示」：角色 / 内容都可以省。'''

    role: MessageRole
    content: MessageContent


#: `ask()` 接受的提示：纯文本，或一条（字段可省的）消息
PromptInput: TypeAlias = Union[str, Mapping[str, object]]

# ---------------- 记忆 / 会话 ----------------
class RecallHit(TypedDict):
    '''一条召回结果。'''

    text: str
    score: float


class SessionPayload(TypedDict, total=False):
    '''会话文件的结构（读的时候宽松处理，所以全部可选）。'''

    version: int
    messages: List[Message]
    variables: Dict[str, JSONValue]
    saved_at: str
    turns: int
    system_prompt: str
    history: List[Message]


# ---------------- 协作者协议 ----------------
#: 工具：`Tool` 实例，或任何普通可调用对象
ToolLike: TypeAlias = Union["Tool", Callable[..., object]]


class SupportsWrite(Protocol):
    '''只要求「能写」的最小协作者。'''

    def write(self, key: str, value: object) -> None: ...


class MemoryLike(SupportsWrite, Protocol):
    '''记忆的鸭子类型：KV 读 / 召回 / 写。'''

    def read(self, key: str, default: object = None) -> object: ...

    def recall(self, query: str, top_k: int = 5) -> Sequence[Dict[str, object]]: ...


class AgentLike(Protocol):
    '''Agent 的鸭子类型。

    参数名与 `Agent.ask` 保持一致 —— 协议是按关键字匹配的，
    名字对不上就不可赋值。签名里用 `Message` / `ToolLike`，
    也是为了让真正的 `Agent` 能原样满足它。
    '''

    history: List[Message]
    start_prompts: List[Message]

    def ask(
        self,
        prompt: PromptInput,
        tools: Optional[Sequence[ToolLike]] = None,
        max_tool_rounds: Optional[int] = None,
        bus: Optional[EventBus] = None,
        on_event: Optional[EventHandler] = None,
        verbose: bool = False,
        stream: bool = False,
    ) -> str: ...

    def reset(self) -> None: ...

    def get_history(self) -> List[Message]: ...

    def load_history(self, records: object, *, replace: bool = True) -> int: ...
