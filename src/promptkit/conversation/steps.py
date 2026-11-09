# conversation/steps.py
'''会话 CRUD 的 DSL 包装。

所有 Step 都从 `context["_session"]` 拿到会话实例，执行结果追加到
`context.trace`；这里不碰任何 LLM，也不直接读写 `Session` 的私有字段。
'''
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

from promptkit.conversation.exceptions import SessionError
from promptkit.conversation.message import (
    JSONValue,
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
    "SessionRollback",
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
        message_id = session.append(
            self.role, self.content, metadata=dict(self.metadata)
        )
        _trace(context, step=self.name, message_id=message_id, length=session.length)

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
        message_id = session.insert(
            self.index, self.role, self.content, metadata=dict(self.metadata)
        )
        _trace(
            context,
            step=self.name,
            message_id=message_id,
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
        index = session.index_of(self.message_id)
        removed = session.at(index)
        session.remove(self.message_id)
        _trace(
            context,
            step=self.name,
            message_id=self.message_id,
            role=(removed.role if removed is not None else None),
            length=session.length,
        )

    def describe(self) -> str:
        return f"SessionRemove({self.message_id!r})"


def _content_text(content: MessageContent) -> str:
    '''把消息内容压成回滚 Step 返回的字符串。'''
    return content if isinstance(content, str) else str(content)


@dataclass
class SessionRollback(Step):
    '''回滚最近若干轮对话（一轮 = 一条 user + 其后到下一轮之前的消息）。

    用法：

        SessionRollback(turns=1)               # 删 1 轮的 assistant，保留 user
        SessionRollback(turns=1, keep="user")  # 同上（默认）
        SessionRollback(turns=2, keep="both")  # 连 user 一起删掉 2 轮

    语义：

    - `keep="user"`：保留所有 user 消息，只删本轮之后的 assistant / tool /
      system 消息 —— 适合"让用户重新说一遍"。
    - `keep="both"`：从最早被回滚的 user 开始，整段删掉。

    返回值（写进 `context[output_key]`）是被回滚的 user 消息内容 `str`；
    会话里没有 user 消息时写 `None`。同时往 `session.trace` 记一条
    `{"step": ..., "message_ids": [...], ...}`。
    '''

    turns: int = 1
    keep: str = "user"
    output_key: str | None = None
    raise_if_empty: bool = False
    name: str = "session.rollback"

    def __post_init__(self) -> None:
        if self.turns < 1:
            raise ValueError(f"turns 必须 >= 1，得到 {self.turns}")
        if self.keep not in ("user", "both"):
            raise ValueError(f"keep 只能是 'user' 或 'both'，得到 {self.keep!r}")

    def execute(self, context: Context) -> None:
        '''按轮回滚，并把被回滚的 user 内容写进 context / trace。'''
        session = session_from_context(context)
        user_indices = [
            index
            for index, message in enumerate(session.messages)
            if message.role == "user"
        ]
        if not user_indices:
            if self.raise_if_empty:
                raise SessionError("SessionRollback：会话里没有 user 消息可回滚")
            if self.output_key is not None:
                context[self.output_key] = None
            _trace(context, step=self.name, turns=self.turns, keep=self.keep, length=0)
            return

        # 最近 turns 条 user 里的最早那条，就是回滚边界
        boundary = user_indices[-self.turns]
        boundary_message = session.at(boundary)
        content = (
            _content_text(boundary_message.content)
            if boundary_message is not None
            else ""
        )

        # 先记下将要删掉的 id（删完就取不到了）
        if self.keep == "both":
            doomed = [
                message
                for message in session.slice(boundary)
                if message.role != "system"
            ]
        else:
            doomed = [
                message
                for message in session.slice(boundary + 1)
                if message.role != "user" and message.role != "system"
            ]

        for message in doomed:
            session.remove(message.id)

        removed_ids = [message.id for message in doomed]
        session.record_trace(
            {
                "step": self.name,
                "turns": self.turns,
                "keep": self.keep,
                "message_ids": removed_ids,
                "user_content": content,
            }
        )
        _trace(
            context,
            step=self.name,
            turns=self.turns,
            keep=self.keep,
            message_ids=removed_ids,
            user_content=content,
            length=session.length,
        )
        if self.output_key is not None:
            context[self.output_key] = content

    def describe(self) -> str:
        suffix = f", keep={self.keep!r}" if self.keep != "user" else ""
        out = f" -> {self.output_key}" if self.output_key is not None else ""
        return f"SessionRollback(turns={self.turns}{suffix}){out}"


@dataclass
class SessionPop(Step):
    '''【已弃用】弹出会话末尾的一条消息。

    阶段 2 起请改用 `SessionRollback(turns=1, keep="user")`：
    旧 `SessionPop` 的语义（"弹一条"）看不清到底弹 user 还是 assistant，
    现在统一按"回滚一轮、保留 user"处理，并发出 `DeprecationWarning`。
    '''

    output_key: str | None = None
    raise_if_empty: bool = False
    name: str = "session.pop"

    def execute(self, context: Context) -> None:
        '''转发给 `SessionRollback(turns=1, keep="user")`，并发出弃用警告。'''
        warnings.warn(
            "SessionPop 已弃用（阶段 2）：请改用 "
            'SessionRollback(turns=1, keep="user")',
            DeprecationWarning,
            stacklevel=2,
        )
        SessionRollback(
            turns=1,
            keep="user",
            output_key=self.output_key,
            raise_if_empty=self.raise_if_empty,
            name=self.name,
        ).execute(context)

    def describe(self) -> str:
        return "SessionPop(...)【已弃用：改用 SessionRollback】"


@dataclass
class SessionTruncate(Step):
    '''只保留前 `length` 条消息。'''

    length: int = 0
    name: str = "session.truncate"

    def execute(self, context: Context) -> None:
        '''截断会话，并把被移除的消息 id 列表写进 trace。'''
        session = session_from_context(context)
        removed = session.slice(self.length)
        session.truncate(self.length)
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
        session.update(
            self.message_id, content=self.content, metadata=self.metadata
        )
        _trace(
            context,
            step=self.name,
            message_id=self.message_id,
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
