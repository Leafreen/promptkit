# conversation/store.py
'''会话存储层：抽象接口 + 内存实现 + SQLite 实现。

Session 只把 `SessionState` 交给 store，自己不认识 SQL / 文件；反过来
store 也不碰 Session 的私有字段。两者唯一的契约就是
`SessionState(session_id, messages, metadata, compressed_upto)`。
'''
from __future__ import annotations

import copy
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import closing
from typing import cast

from promptkit.conversation.message import (
    JSONValue,
    Message,
    MessageContent,
    MessageRole,
)
from promptkit.conversation.session import SessionState

__all__ = ["InMemorySessionStore", "SQLiteSessionStore", "SessionStore"]

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS conversation_sessions (
        session_id TEXT PRIMARY KEY,
        metadata TEXT NOT NULL,
        compressed_upto INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT NOT NULL,
        FOREIGN KEY (session_id)
            REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
        ON conversation_messages(session_id, position)
    """,
)


class SessionStore(ABC):
    '''会话持久化接口。

    四个抽象方法都只处理 `SessionState`；实现方不需要理解 Session 的
    增删改查逻辑，也不需要负责异常策略（Session 会吞掉持久化错误）。
    '''

    @abstractmethod
    def save(self, state: SessionState) -> None:
        '''整体写入一份会话状态（覆盖同 id 的旧状态）。'''
        raise NotImplementedError

    @abstractmethod
    def load(self, session_id: str) -> SessionState | None:
        '''读取会话状态；不存在返回 None。'''
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        '''删除会话；删到了返回 True。'''
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> list[str]:
        '''列出全部会话 id。'''
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    '''进程内存储：写入/读出都做 deepcopy，避免调用方拿到同一份可变对象。'''

    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}

    def save(self, state: SessionState) -> None:
        '''保存一份状态副本。'''
        self._states[state.session_id] = _copy_state(state)

    def load(self, session_id: str) -> SessionState | None:
        '''返回一份状态副本；不存在返回 None。'''
        state = self._states.get(session_id)
        return None if state is None else _copy_state(state)

    def delete(self, session_id: str) -> bool:
        '''删除会话。'''
        return self._states.pop(session_id, None) is not None

    def list_ids(self) -> list[str]:
        '''列出全部会话 id。'''
        return list(self._states)


class SQLiteSessionStore(SessionStore):
    '''SQLite 存储。

    表结构：

    - `conversation_sessions(session_id PK, metadata JSON, compressed_upto, updated_at)`
    - `conversation_messages(message_id PK, session_id FK, position, role,
      content JSON, metadata JSON)`

    `content` / `metadata` 都按 JSON 文本存；每次操作临时打开连接，
    所以 store 对象本身没有「必须 close 的连接」，deepcopy 也是安全的。
    '''

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("SQLiteSessionStore 需要非空的数据库路径")
        self.path = os.path.abspath(os.path.expanduser(path))
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            for statement in _SCHEMA:
                conn.execute(statement)

    # ---------------- 接口实现 ----------------
    def save(self, state: SessionState) -> None:
        '''把状态整体写进数据库（先删旧消息，再插新消息）。'''
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO conversation_sessions"
                " (session_id, metadata, compressed_upto, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                " metadata = excluded.metadata,"
                " compressed_upto = excluded.compressed_upto,"
                " updated_at = excluded.updated_at",
                (
                    state.session_id,
                    _dumps(state.metadata),
                    state.compressed_upto,
                    stamp,
                ),
            )
            conn.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?",
                (state.session_id,),
            )
            conn.executemany(
                "INSERT INTO conversation_messages"
                " (message_id, session_id, position, role, content, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        message.id,
                        state.session_id,
                        position,
                        message.role,
                        _dumps(message.content),
                        _dumps(dict(message.metadata)),
                    )
                    for position, message in enumerate(state.messages)
                ],
            )

    def load(self, session_id: str) -> SessionState | None:
        '''按 id 读取状态；不存在返回 None。'''
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT metadata, compressed_upto FROM conversation_sessions"
                " WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            message_rows = conn.execute(
                "SELECT message_id, role, content, metadata"
                " FROM conversation_messages WHERE session_id = ?"
                " ORDER BY position",
                (session_id,),
            ).fetchall()

        messages = [
            Message(
                role=cast(MessageRole, str(role)),
                content=cast(MessageContent, _loads(str(content))),
                id=str(message_id),
                metadata=cast("dict[str, JSONValue]", _loads(str(metadata))),
            )
            for message_id, role, content, metadata in message_rows
        ]
        return SessionState(
            session_id=session_id,
            messages=messages,
            metadata=cast("dict[str, JSONValue]", _loads(str(row[0]))),
            compressed_upto=int(row[1]),
        )

    def delete(self, session_id: str) -> bool:
        '''删除会话（消息通过外键级联删除）。'''
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

    def list_ids(self) -> list[str]:
        '''列出全部会话 id，按更新时间倒序。'''
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT session_id FROM conversation_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [str(row[0]) for row in rows]

    # ---------------- 内部 ----------------
    def _connect(self) -> sqlite3.Connection:
        '''打开一个短连接，并启用外键约束。'''
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _dumps(value: object) -> str:
    '''JSON 序列化（中文不转义）。'''
    return json.dumps(value, ensure_ascii=False)


def _loads(raw: str) -> JSONValue:
    '''JSON 反序列化，结果按 JSONValue 使用。'''
    return cast(JSONValue, json.loads(raw))


def _copy_state(state: SessionState) -> SessionState:
    '''深拷贝一份状态（InMemory 用）。'''
    return copy.deepcopy(state)


# 本文件核心职责：定义 SessionStore 抽象基类，并实现内存 / SQLite 两种 JSON 持久化。
