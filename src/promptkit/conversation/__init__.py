# conversation/__init__.py
'''对话状态层：消息、会话、存储、DSL Step 与语义糖。

这里只依赖标准库与 Pydantic（本层实际连 Pydantic 都不需要），
不导入任何 LLM 客户端 / Prompt / Executor 模块。
'''
from promptkit.conversation.config import RewindConfig
from promptkit.conversation.exceptions import (
    BranchLimitExceeded,
    RewindBeyondCompression,
    RewindBeyondLimit,
    SessionError,
    SnapshotReadonly,
)
from promptkit.conversation.message import Message
from promptkit.conversation.semantic import (
    Branch,
    Branches,
    Fork,
    Merge,
    Restore,
    Snapshot,
)
from promptkit.conversation.session import Session, SessionState
from promptkit.conversation.steps import (
    SESSION_KEY,
    SessionAppend,
    SessionClear,
    SessionInsert,
    SessionPop,
    SessionRemove,
    SessionTruncate,
    SessionUpdate,
)
from promptkit.conversation.store import (
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
)

__all__ = [
    "SESSION_KEY",
    "Branch",
    "BranchLimitExceeded",
    "Branches",
    "Fork",
    "InMemorySessionStore",
    "Merge",
    "Message",
    "Restore",
    "RewindBeyondCompression",
    "RewindBeyondLimit",
    "RewindConfig",
    "SQLiteSessionStore",
    "Session",
    "SessionAppend",
    "SessionClear",
    "SessionError",
    "SessionInsert",
    "SessionPop",
    "SessionRemove",
    "SessionState",
    "SessionStore",
    "SessionTruncate",
    "SessionUpdate",
    "Snapshot",
    "SnapshotReadonly",
]


# 本文件核心职责：汇总导出对话状态层的公开 API（消息 / 会话 / 存储 / Step / 配置 / 异常）。
