'''CRUD Step 单元测试：从 ctx["_session"] 取会话、结果写 trace。'''
import pytest

from promptkit.conversation.exceptions import SessionError
from promptkit.conversation.message import Message
from promptkit.conversation.session import Session
from promptkit.conversation.steps import (
    SESSION_KEY,
    SessionAppend,
    SessionClear,
    SessionInsert,
    SessionRemove,
    SessionTruncate,
    SessionUpdate,
)
from promptkit.core.context import Context


def make_context() -> Context:
    session = Session()
    return Context(data={SESSION_KEY: session})


def test_append_writes_message_and_trace():
    context = make_context()
    SessionAppend("user", "你好").execute(context)
    session: Session = context[SESSION_KEY]
    assert session.messages[0].content == "你好"
    assert context.trace[-1]["step"] == "session.append"
    assert context.trace[-1]["message_id"] == session.messages[0].id


def test_insert_uses_index_and_trace():
    context = make_context()
    session: Session = context[SESSION_KEY]
    session.append_user("a")
    SessionInsert(1, "user", "b").execute(context)
    assert [m.content for m in session.messages] == ["a", "b"]
    assert context.trace[-1]["index"] == 1


def test_remove_by_id():
    context = make_context()
    session: Session = context[SESSION_KEY]
    first = session.append_user("a")
    session.append_user("b")
    SessionRemove(first.id).execute(context)
    assert [m.content for m in session.messages] == ["b"]
    assert context.trace[-1]["message_id"] == first.id


def test_truncate():
    context = make_context()
    session: Session = context[SESSION_KEY]
    for text in ("a", "b", "c"):
        session.append_user(text)
    SessionTruncate(1).execute(context)
    assert [m.content for m in session.messages] == ["a"]
    assert len(context.trace[-1]["removed"]) == 2


def test_update_content_and_metadata():
    context = make_context()
    session: Session = context[SESSION_KEY]
    target = session.append_user("旧")
    SessionUpdate(target.id, content="新").execute(context)
    assert session.get(target.id).content == "新"

    SessionUpdate(target.id, metadata={"tag": "updated"}).execute(context)
    assert session.get(target.id).metadata["tag"] == "updated"
    assert context.trace[-1]["changed"] == ["metadata"]


def test_update_requires_at_least_one_field():
    context = make_context()
    with pytest.raises(ValueError):
        SessionUpdate("anything").execute(context)


def test_clear():
    context = make_context()
    session: Session = context[SESSION_KEY]
    session.append_user("a")
    SessionClear().execute(context)
    assert session.length == 0
    assert context.trace[-1]["cleared"] == 1


def test_missing_session_raises():
    context = Context()
    with pytest.raises(SessionError):
        SessionAppend("user", "x").execute(context)


def test_append_with_metadata_is_kept():
    context = make_context()
    SessionAppend(
        "assistant", "好了", metadata={"reasoning_content": "想想"}
    ).execute(context)
    session: Session = context[SESSION_KEY]
    assert session.messages[0].metadata["reasoning_content"] == "想想"
    assert isinstance(session.messages[0], Message)
