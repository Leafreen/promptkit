# conversation/session.py
'''纯状态的会话容器。

这里**绝对不导入任何 LLM 相关模块**（LLMClient / Prompt / Executor ……）：
`Session` 只负责「消息按顺序排好、能增删改查、能克隆」，发不发请求、
用哪个厂商，都由上层决定。

按 id 操作一律先 `index_of(id)` 再定位，不依赖调用方传入的位置；
位置型操作（`insert` / `slice` / `truncate`）才显式接受下标。
'''
from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from promptkit.conversation.exceptions import SessionError
from promptkit.conversation.message import (
    JSONValue,
    Message,
    MessageContent,
    MessageRole,
)

if TYPE_CHECKING:  # 只用于类型标注，运行时由上层注入 store
    from promptkit.conversation.store import SessionStore


@dataclass
class SessionState:
    '''可持久化的会话快照（store 层只认这个结构，不直接碰 Session 私有字段）。'''

    session_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    compressed_upto: int = 0


class Session:
    '''按顺序保存消息的纯状态容器。

    store 省略时永远是纯内存的：进程结束即消失。传入 store 时：

    - 构造时尝试 `store.load(session_id)` 恢复已有状态；
    - 每次写操作后尝试 `store.save(state)`；
    - **持久化失败不会抛出**，内存里的会话照常可用。

    `clone()` 用 `copy.deepcopy` 做完全隔离，并丢弃 store、换新 session_id，
    保证分叉出来的会话不会和父会话共享任何可变对象、也不会写到同一个持久化目标。
    '''

    def __init__(
        self,
        messages: Sequence[Message] | None = None,
        *,
        session_id: str | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        compressed_upto: int = 0,
        store: SessionStore | None = None,
    ) -> None:
        self._session_id: str = session_id or uuid.uuid4().hex
        self._messages: list[Message] = list(messages or [])
        self._metadata: dict[str, JSONValue] = dict(metadata or {})
        self._compressed_upto: int = compressed_upto
        self._validate_compressed(self._compressed_upto)
        self._store = store
        if store is not None:
            try:
                state = store.load(self._session_id)
            except Exception:  # noqa: BLE001 - 持久化失败不影响内存操作
                state = None
            if state is not None:
                self._apply_state(state)

    # ---------------- 只读属性 ----------------
    @property
    def session_id(self) -> str:
        '''会话标识；纯内存会话也会生成一个 uuid。'''
        return self._session_id

    @property
    def messages(self) -> list[Message]:
        '''当前消息列表的浅拷贝（消息本身不可变，外部改列表不会影响会话）。'''
        return list(self._messages)

    @property
    def metadata(self) -> dict[str, JSONValue]:
        '''会话级元信息的浅拷贝。'''
        return dict(self._metadata)

    @property
    def compressed_upto(self) -> int:
        '''前多少条消息已被压缩进摘要（回退边界）。'''
        return self._compressed_upto

    @property
    def length(self) -> int:
        '''消息条数。'''
        return len(self._messages)

    @property
    def is_persistent(self) -> bool:
        '''是否绑定了 store。'''
        return self._store is not None

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"<Session id={self._session_id!r} messages={len(self._messages)}"
            f" compressed_upto={self._compressed_upto}>"
        )

    # ---------------- 20 个 CRUD ----------------
    def append(self, message: Message) -> Message:
        '''在末尾追加一条消息，返回它。'''
        self._messages.append(message)
        self._touch()
        return message

    def append_user(self, content: MessageContent, *, name: str | None = None) -> Message:
        '''追加一条 user 消息。'''
        return self.append(self._build("user", content, name=name))

    def append_assistant(
        self,
        content: MessageContent,
        *,
        tool_calls: Sequence[Mapping[str, JSONValue]] | None = None,
        reasoning_content: str | None = None,
        name: str | None = None,
    ) -> Message:
        '''追加一条 assistant 消息（可带 tool_calls / reasoning_content）。'''
        return self.append(
            self._build(
                "assistant",
                content,
                name=name,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
        )

    def append_system(self, content: MessageContent, *, name: str | None = None) -> Message:
        '''追加一条 system 消息。'''
        return self.append(self._build("system", content, name=name))

    def append_tool(
        self,
        content: MessageContent,
        tool_call_id: str,
        *,
        name: str | None = None,
    ) -> Message:
        '''追加一条 tool 消息（必须带 tool_call_id）。'''
        return self.append(
            Message(
                role="tool",
                content=content,
                metadata=self._compact_metadata(
                    {"tool_call_id": tool_call_id, "name": name}
                ),
            )
        )

    def insert(self, index: int, message: Message) -> Message:
        '''在指定下标插入一条消息（`0 <= index <= length`）。'''
        if index < 0 or index > len(self._messages):
            raise IndexError(f"插入位置越界：{index}，当前长度 {len(self._messages)}")
        self._messages.insert(index, message)
        self._touch()
        return message

    def insert_before(self, message_id: str, message: Message) -> Message:
        '''按 id 找到目标消息，插在它前面。'''
        return self.insert(self.index_of(message_id), message)

    def insert_after(self, message_id: str, message: Message) -> Message:
        '''按 id 找到目标消息，插在它后面。'''
        return self.insert(self.index_of(message_id) + 1, message)

    def remove(self, message_id: str) -> Message:
        '''按 id 删除一条消息，返回被删掉的那条。'''
        index = self.index_of(message_id)
        message = self._messages.pop(index)
        self._touch()
        return message

    def remove_last(self) -> Message | None:
        '''删除最后一条消息；会话为空时返回 None。'''
        if not self._messages:
            return None
        message = self._messages.pop()
        self._touch()
        return message

    def replace(self, message_id: str, message: Message) -> Message:
        '''按 id 用新消息替换旧的，返回新消息。'''
        index = self.index_of(message_id)
        self._messages[index] = message
        self._touch()
        return message

    def update_content(
        self, message_id: str, content: MessageContent
    ) -> Message:
        '''按 id 替换消息内容（id / metadata 保持不变）。'''
        return self.replace(message_id, self.get(message_id).with_content(content))

    def update_metadata(self, message_id: str, **values: JSONValue) -> Message:
        '''按 id 合并元信息（内容 / id 保持不变）。'''
        return self.replace(message_id, self.get(message_id).with_metadata(**values))

    def get(self, message_id: str) -> Message:
        '''按 id 取消息；不存在时抛 SessionError。'''
        return self._messages[self.index_of(message_id)]

    def find(self, message_id: str) -> Message | None:
        '''按 id 取消息；不存在时返回 None。'''
        for message in self._messages:
            if message.id == message_id:
                return message
        return None

    def index_of(self, message_id: str) -> int:
        '''按 id 返回下标；不存在时抛 SessionError。'''
        for index, message in enumerate(self._messages):
            if message.id == message_id:
                return index
        raise SessionError(f"没有 id 为 {message_id!r} 的消息")

    def slice(self, start: int = 0, end: int | None = None) -> list[Message]:
        '''按下标切片，返回消息列表的副本。'''
        return list(self._messages[start:end])

    def replace_all(self, messages: Sequence[Message]) -> None:
        '''用一批消息整体替换当前内容（保留 session_id / metadata / 压缩边界）。'''
        self._messages = list(messages)
        self._compressed_upto = min(self._compressed_upto, len(self._messages))
        self._touch()

    def truncate(self, length: int) -> list[Message]:
        '''只保留前 `length` 条，返回被截掉的消息。'''
        if length < 0:
            raise ValueError(f"truncate 长度不能为负：{length}")
        removed = self._messages[length:]
        if removed:
            self._messages = self._messages[:length]
            self._compressed_upto = min(self._compressed_upto, len(self._messages))
            self._touch()
        return list(removed)

    def clear(self) -> None:
        '''清空全部消息（压缩边界同步归零，并触发一次持久化）。'''
        if self._messages:
            self._messages = []
            self._compressed_upto = 0
            self._touch()

    # ---------------- 状态 / 克隆 ----------------
    def state(self) -> SessionState:
        '''导出可持久化的状态（store 层只接收这个结构）。'''
        return SessionState(
            session_id=self._session_id,
            messages=list(self._messages),
            metadata=dict(self._metadata),
            compressed_upto=self._compressed_upto,
        )

    def set_compressed_upto(self, count: int) -> int:
        '''设置压缩边界（前 `count` 条消息视为已进摘要），返回它。'''
        self._validate_compressed(count)
        self._compressed_upto = count
        self._touch()
        return count

    def clone(self) -> Session:
        '''用 deepcopy 做完全独立的副本；store 被丢弃，session_id 换新。'''
        cloned = copy.deepcopy(self)
        cloned._store = None
        cloned._session_id = uuid.uuid4().hex
        return cloned

    # ---------------- 内部 ----------------
    def _apply_state(self, state: SessionState) -> None:
        self._messages = list(state.messages)
        self._metadata = dict(state.metadata)
        self._compressed_upto = min(state.compressed_upto, len(self._messages))

    def _touch(self) -> None:
        '''每次写操作后尝试持久化；失败只吞掉，绝不影响内存状态。'''
        if self._store is None:
            return
        try:
            self._store.save(self.state())
        except Exception:  # noqa: BLE001 - 持久化失败不得影响内存操作
            return

    def _validate_compressed(self, count: int) -> None:
        if count < 0 or count > len(self._messages):
            raise ValueError(
                f"compressed_upto 必须在 0..{len(self._messages)} 之间，得到 {count}"
            )

    @staticmethod
    def _compact_metadata(values: Mapping[str, object]) -> dict[str, JSONValue]:
        '''去掉值为 None 的键，避免历史里出现 `"name": null`。'''
        return {
            key: cast(JSONValue, value)
            for key, value in values.items()
            if value is not None
        }

    def _build(
        self,
        role: MessageRole,
        content: MessageContent,
        *,
        name: str | None = None,
        tool_calls: Sequence[Mapping[str, JSONValue]] | None = None,
        reasoning_content: str | None = None,
    ) -> Message:
        metadata = self._compact_metadata(
            {
                "name": name,
                "tool_calls": list(tool_calls) if tool_calls else None,
                "reasoning_content": reasoning_content,
            }
        )
        return Message(role=role, content=content, metadata=metadata)


# 本文件核心职责：实现纯状态 Session（20 个 CRUD、按 id 定位、deepcopy clone、可选 store 持久化）。
