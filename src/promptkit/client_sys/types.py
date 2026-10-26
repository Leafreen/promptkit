# client_sys/types.py
'''厂商无关的「Chat Completions 方言」数据模型。

promptkit 内部**只用一个方言**描述对话：OpenAI Chat Completions 的字段名
（`messages` / `tools` / `tool_calls` / `usage` / `choices[0].delta`）。
所以「支持多厂商」不是再造一套中间格式，而是让每家厂商都往这个方言上靠：
谁不认识这套字段，谁就在 client 层翻译进来、翻译出去。

这样 `Agent` 里那套「工具循环 + 事件 + 流式」的代码**一行都不用按厂商分支**。

这两套类（`Completion` / `Chunk`）专门给**非 OpenAI** 厂商当翻译目标用，
字段名和 openai SDK 返回的对象逐字段对齐：

    completion.choices[0].message.content
    completion.choices[0].message.tool_calls[0].function.arguments
    completion.choices[0].message.reasoning_content
    completion.usage.prompt_tokens
    chunk.choices[0].delta.content

`OpenAIClient` 走的是另一条路：它把 SDK 的原始对象**直接透出去**
（形状完全一样，没必要多拷贝一层，也不丢 `refusal` / `logprobs` 这些字段）。
所以「透传的对象」和「翻译出来的对象」在 Agent 眼里没有区别。
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "FunctionCall",
    "ToolCall",
    "Message",
    "Choice",
    "Usage",
    "Completion",
    "Chunk",
]


@dataclass
class FunctionCall:
    '''工具调用的函数部分（名字 + 参数）。

    `arguments` 是**字符串**，不是 dict —— OpenAI 的约定就是这样，
    参数最终由 `Agent` 用 `json.loads` 解析，解析失败也交给模型自己处理。
    流式分片时它可能只是半截 JSON（`'{"city"'`），靠 `Agent` 负责拼。
    '''

    name: str = ""
    arguments: str = ""

    def model_dump(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    '''一次工具调用。

    `index` 只有流式分片才有意义（第几个工具调用），非流式时为 None。
    `model_dump()` 输出 OpenAI 的形状，`Agent` 记历史时直接用它。
    '''

    id: str = ""
    function: FunctionCall = field(default_factory=FunctionCall)
    type: str = "function"
    index: Optional[int] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass
class Message:
    '''一条消息 / 一个流式增量。

    同一套字段干两件事：非流式时是 `Choice.message`（完整回复），
    流式时是 `Choice.delta`（一个小片段）。OpenAI 那边也是这么设计的，
    所以 `Agent` 的解析代码能共用。
    '''

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    #: 思考过程。DeepSeek 叫 reasoning_content，这里统一成这个名字
    reasoning_content: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.reasoning_content:
            data["reasoning_content"] = self.reasoning_content
        return data

    @property
    def text(self) -> str:
        '''正文（None 当空字符串，省得每处都写 `or ""`）。'''
        return self.content or ""


@dataclass
class Choice:
    '''一个候选回复。非流式只有 `message`，流式只有 `delta`。'''

    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    '''token 用量。'''

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_anthropic(cls, data: Dict[str, Any]) -> "Usage":
        '''Anthropic 的 input_tokens / output_tokens -> 这边的三件套。'''
        prompt = int(data.get("input_tokens") or 0)
        completion = int(data.get("output_tokens") or 0)
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


@dataclass
class Completion:
    '''一次非流式回复，形状和 openai SDK 的 `ChatCompletion` 一致。'''

    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    model: Optional[str] = None
    id: Optional[str] = None
    #: 厂商原始响应（排查问题时有用）
    raw: Any = None

    @property
    def message(self) -> Optional[Message]:
        return self.choices[0].message if self.choices else None

    @property
    def text(self) -> str:
        message = self.message
        return message.text if message is not None else ""


@dataclass
class Chunk:
    '''一个流式分片。`choices` 为空表示只带用量 / 心跳。'''

    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    model: Optional[str] = None

    @property
    def delta(self) -> Optional[Message]:
        return self.choices[0].delta if self.choices else None
