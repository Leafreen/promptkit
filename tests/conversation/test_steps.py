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
    SessionPop,
    SessionRemove,
    SessionRollback,
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
    updated = session.get(target.id)
    assert updated is not None and updated.content == "新"

    SessionUpdate(target.id, metadata={"tag": "updated"}).execute(context)
    found = session.get(target.id)
    assert found is not None and found.metadata["tag"] == "updated"
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


def test_rollback_one_turn_keeps_user():
    '''Back：问题1 → 答案1 → 回滚，usr_input 回到"问题1"，会话只剩 1 条。'''
    context = make_context()
    session: Session = context[SESSION_KEY]
    session.append_user("问题1")
    session.append_assistant("答案1")

    SessionRollback(turns=1, output_key="usr_input").execute(context)

    assert context["usr_input"] == "问题1"
    assert session.count() == 1
    assert [m.content for m in session.messages] == ["问题1"]
    assert context.trace[-1]["step"] == "session.rollback"
    assert context.trace[-1]["message_ids"]
    assert session.trace[-1]["user_content"] == "问题1"


def test_rollback_two_turns_deletes_both():
    '''turns=2, keep="both"：两条 user 与它们的回复全部删掉。'''
    context = make_context()
    session: Session = context[SESSION_KEY]
    for index in (1, 2):
        session.append_user(f"问题{index}")
        session.append_assistant(f"答案{index}")

    SessionRollback(turns=2, keep="both").execute(context)

    assert session.count() == 0
    assert context.trace[-1]["keep"] == "both"


def test_rollback_empty_session_is_tolerated_or_raises():
    context = make_context()
    SessionRollback(output_key="usr_input").execute(context)
    assert context["usr_input"] is None

    context = make_context()
    with pytest.raises(SessionError):
        SessionRollback(raise_if_empty=True).execute(context)


def test_session_pop_is_deprecated_but_still_works():
    context = make_context()
    session: Session = context[SESSION_KEY]
    session.append_user("问题1")
    session.append_assistant("答案1")

    with pytest.warns(DeprecationWarning, match="SessionRollback"):
        SessionPop(output_key="usr_input").execute(context)

    assert context["usr_input"] == "问题1"
    assert session.count() == 1


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
