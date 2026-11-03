# conversation/steps.py
'''会话 CRUD 的 DSL 包装。

所有 Step 都从 `context["_session"]` 拿到会话实例，执行结果追加到
`context.trace`；这里不碰任何 LLM，也不直接读写 `Session` 的私有字段。
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from promptkit.conversation.exceptions import SessionError
from promptkit.conversation.message import (
    JSONValue,
    Message,
    MessageContent,
    MessageRole,
)
from promptkit.conversation.session import Session
from promptkit.core.context import Context
from promptkit.core.steps.interface import Step

__all__ = [
    "SESSION_KEY",
    "SessionAppend",
    "SessionClear",
    "SessionInsert",
    "SessionPop",
    "SessionRemove",
    "SessionTruncate",
    "SessionUpdate",
    "session_from_context",
]

#: Context 里存放会话实例的键；所有会话 Step 都从这里取
SESSION_KEY = "_session"


def session_from_context(context: Context) -> Session:
    '''取出当前会话；没挂或挂错类型时给出明确的 SessionError。'''
    value = context.data.get(SESSION_KEY)
    if not isinstance(value, Session):
        raise SessionError(
            f'context[{SESSION_KEY!r}] 必须是 Session，得到 {type(value).__name__}；'
            "运行工作流时传 `_session=agent.session`"
        )
    return value


def _trace(context: Context, **fields: object) -> None:
    '''把一次会话操作记进 trace。'''
    context.trace.append(fields)


@dataclass
class SessionAppend(Step):
    '''在会话末尾追加一条消息。'''

    role: MessageRole = "user"
    content: MessageContent = ""
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    name: str = "session.append"

    def execute(self, context: Context) -> None:
        '''追加消息，并把新消息 id 写进 trace。'''
        session = session_from_context(context)
        message = Message(
            role=self.role, content=self.content, metadata=dict(self.metadata)
        )
        session.append(message)
        _trace(context, step=self.name, message_id=message.id, length=session.length)

    def describe(self) -> str:
        return f"SessionAppend({self.role!r}, {self.content!r})"


@dataclass
class SessionInsert(Step):
    '''在指定下标插入一条消息。'''

    index: int = 0
    role: MessageRole = "user"
    content: MessageContent = ""
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    name: str = "session.insert"

    def execute(self, context: Context) -> None:
        '''插入消息，并把下标 / 新 id 写进 trace。'''
        session = session_from_context(context)
        message = Message(
            role=self.role, content=self.content, metadata=dict(self.metadata)
        )
        session.insert(self.index, message)
        _trace(
            context,
            step=self.name,
            message_id=message.id,
            index=self.index,
            length=session.length,
        )

    def describe(self) -> str:
        return f"SessionInsert({self.index}, {self.role!r}, {self.content!r})"


@dataclass
class SessionRemove(Step):
    '''按 id 删除一条消息。'''

    message_id: str = ""
    name: str = "session.remove"

    def execute(self, context: Context) -> None:
        '''删除消息，并把被删消息的 id / 角色写进 trace。'''
        session = session_from_context(context)
        removed = session.remove(self.message_id)
        _trace(
            context,
            step=self.name,
            message_id=removed.id,
            role=removed.role,
            length=session.length,
        )

    def describe(self) -> str:
        return f"SessionRemove({self.message_id!r})"


@dataclass
class SessionPop(Step):
    '''弹出会话末尾的一条消息。

    `output_key` 有值时，被弹掉的消息（`{role, content, ...}` dict，不含
    内部 id）会写进 `context[output_key]`，方便后续 Step 直接引用；
    会话为空时写 `None`。

    会话为空时默认什么都不做、只在 trace 里记一条 `popped=None`；
    传 `raise_if_empty=True` 可以让空会话直接抛 `SessionError` —— 适合你
    明确知道会话里应该有东西、为空就是 bug 的场景。
    '''

    output_key: str | None = None
    raise_if_empty: bool = False
    name: str = "session.pop"

    def execute(self, context: Context) -> None:
        '''弹掉最后一条消息，并把被弹消息写进 context / trace。'''
        session = session_from_context(context)
        if session.length == 0:
            if self.raise_if_empty:
                raise SessionError("SessionPop：会话是空的，没东西可弹")
            if self.output_key is not None:
                context[self.output_key] = None
            popped_extra: dict[str, object] = {"popped": None}
            if self.output_key is not None:
                popped_extra["output_key"] = self.output_key
            _trace(context, step=self.name, length=0, **popped_extra)
            return
        removed = session.remove_last()
        # remove_last 在 length>0 时一定返回非 None，这里只是类型收窄
        assert removed is not None
        payload: dict[str, JSONValue] = removed.to_dict()
        if self.output_key is not None:
            context[self.output_key] = payload
        trace_extra: dict[str, object] = {}
        if self.output_key is not None:
            trace_extra["output_key"] = self.output_key
        _trace(
            context,
            step=self.name,
            message_id=removed.id,
            role=removed.role,
            length=session.length,
            **trace_extra,
        )

    def describe(self) -> str:
        parts: list[str] = []
        if self.output_key is not None:
            parts.append(f"-> {self.output_key}")
        if self.raise_if_empty:
            parts.append("raise_if_empty=True")
        suffix = (" " + " ".join(parts)) if parts else ""
        return f"SessionPop({suffix.strip()})"


@dataclass
class SessionTruncate(Step):
    '''只保留前 `length` 条消息。'''

    length: int = 0
    name: str = "session.truncate"

    def execute(self, context: Context) -> None:
        '''截断会话，并把被移除的消息 id 列表写进 trace。'''
        session = session_from_context(context)
        removed = session.truncate(self.length)
        _trace(
            context,
            step=self.name,
            removed=[message.id for message in removed],
            length=session.length,
        )

    def describe(self) -> str:
        return f"SessionTruncate({self.length})"


@dataclass
class SessionUpdate(Step):
    '''按 id 更新消息内容 / 元信息。

    `content` / `metadata` 为 None 表示「这一项不改」；两者都为 None 属于
    无效操作，会在执行时抛 ValueError。
    '''

    message_id: str = ""
    content: MessageContent | None = None
    metadata: Mapping[str, JSONValue] | None = None
    name: str = "session.update"

    def execute(self, context: Context) -> None:
        '''按需更新内容与元信息，并把最终消息 id 写进 trace。'''
        if self.content is None and self.metadata is None:
            raise ValueError("SessionUpdate 至少要提供 content 或 metadata 之一")
        session = session_from_context(context)
        message = session.get(self.message_id)
        if self.content is not None:
            message = session.update_content(self.message_id, self.content)
        if self.metadata is not None:
            message = session.update_metadata(self.message_id, **dict(self.metadata))
        _trace(
            context,
            step=self.name,
            message_id=message.id,
            changed=_changed_fields(self.content, self.metadata),
        )

    def describe(self) -> str:
        return f"SessionUpdate({self.message_id!r})"


@dataclass
class SessionClear(Step):
    '''清空会话里的全部消息。'''

    name: str = "session.clear"

    def execute(self, context: Context) -> None:
        '''清空会话，并把清空前的条数写进 trace。'''
        session = session_from_context(context)
        before = session.length
        session.clear()
        _trace(context, step=self.name, cleared=before)


def _changed_fields(
    content: MessageContent | None, metadata: Mapping[str, JSONValue] | None
) -> list[str]:
    '''算出本次更新动了哪些字段（只用于 trace）。'''
    changed: list[str] = []
    if content is not None:
        changed.append("content")
    if metadata is not None:
        changed.append("metadata")
    return changed


# 本文件核心职责：实现 SessionAppend / Insert / Remove / Pop / Truncate / Update / Clear 七个 CRUD Step。
