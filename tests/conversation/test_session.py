'''Session 单元测试：20 个 CRUD、按 id 定位、clone 隔离、持久化容错。'''
import pytest

from promptkit.conversation.exceptions import SessionError
from promptkit.conversation.message import Message
from promptkit.conversation.session import Session, SessionState
from promptkit.conversation.store import InMemorySessionStore, SessionStore


class BrokenStore(SessionStore):
    '''写盘永远失败的 store，用来验证「持久化失败不影响内存」。'''

    def save(self, state: SessionState) -> None:
        raise RuntimeError("disk full")

    def load(self, session_id: str) -> SessionState | None:
        return None

    def delete(self, session_id: str) -> bool:
        return False

    def list_ids(self) -> list[str]:
        return []


def test_append_and_directional_appends():
    session = Session()
    assert session.append_user("你好").role == "user"
    assert session.append_assistant("你好呀").role == "assistant"
    assert session.append_system("规则").role == "system"
    tool = session.append_tool("晴", "c1", name="weather")
    assert tool.role == "tool"
    assert tool.metadata["tool_call_id"] == "c1"
    assert session.length == 4


def test_insert_and_id_based_insert_before_after():
    session = Session()
    first = session.append_user("a")
    last = session.append_user("c")
    session.insert_before(last.id, Message(role="user", content="b"))
    assert [m.content for m in session.messages] == ["a", "b", "c"]
    session.insert_after(first.id, Message(role="user", content="a2"))
    assert [m.content for m in session.messages] == ["a", "a2", "b", "c"]


def test_insert_bounds():
    session = Session()
    with pytest.raises(IndexError):
        session.insert_message(1, Message(role="user", content="x"))
    with pytest.raises(IndexError):
        session.insert_message(-1, Message(role="user", content="x"))


def test_get_find_index_of_and_remove_by_id_not_position():
    session = Session()
    first = session.append_user("a")
    second = session.append_user("b")
    session.insert_message(0, Message(role="user", content="x"))
    # second 的位置变了，但按 id 仍然找得到
    assert session.index_of(second.id) == 2
    assert session.get(second.id) is second
    assert session.find(first.id) is first
    assert session.find("missing") is None
    session.remove(first.id)
    assert first.content == "a"
    assert [m.content for m in session.messages] == ["x", "b"]


def test_remove_last_replace_update_content_and_metadata():
    session = Session()
    session.append_user("a")
    target = session.append_assistant("b")
    session.update_content(target.id, "B")
    session.update_metadata(target.id, reasoning_content="想了想")
    updated = session.get(target.id)
    assert updated is not None
    assert updated.content == "B"
    assert updated.metadata["reasoning_content"] == "想了想"

    replacement = Message(role="assistant", content="c")
    session.replace(target.id, replacement)
    found = session.get(replacement.id)
    assert found is not None and found.content == "c"

    assert session.remove_last() is replacement
    assert session.length == 1


def test_slice_replace_all_truncate_and_clear():
    session = Session()
    for text in ("a", "b", "c", "d"):
        session.append_user(text)
    assert [m.content for m in session.slice(1, 3)] == ["b", "c"]
    removed = session.slice(2)
    session.truncate(2)
    assert [m.content for m in removed] == ["c", "d"]
    assert [m.content for m in session.messages] == ["a", "b"]

    session.replace_all([Message(role="user", content="z")])
    assert [m.content for m in session.messages] == ["z"]

    with pytest.raises(ValueError):
        session.truncate(-1)

    session.clear()
    assert session.length == 0


def test_clone_is_deepcopy_and_completely_isolated():
    session = Session(metadata={"owner": "parent"})
    session.append_user("a")
    cloned = session.clone()

    cloned.append_user("b")
    cloned.update_metadata(cloned.messages[0].id, tag="clone")
    cloned.clear()

    assert session.length == 1
    assert session.messages[0].content == "a"
    assert "tag" not in session.messages[0].metadata
    assert session.metadata == {"owner": "parent"}
    assert cloned.session_id != session.session_id
    assert cloned.is_persistent is False


def test_snapshot_round_trip():
    '''snapshot() → from_snapshot() 能原样恢复消息 id / 内容 / 元信息。'''
    session = Session(metadata={"owner": "snap"})
    user = session.append_user("问题1")
    assistant = session.append_assistant("答案1")
    session.record_trace({"step": "demo", "message_ids": [user.id]})

    snap = session.snapshot()
    restored = Session.from_snapshot(snap)

    assert restored.session_id == session.session_id
    assert restored.count() == 2
    assert restored.at(0).id == user.id  # type: ignore[union-attr]
    assert restored.at(1).content == "答案1"  # type: ignore[union-attr]
    assert restored.metadata == {"owner": "snap"}
    assert restored.trace == session.trace
    assert session.trace[-1]["message_ids"] == [user.id]
    assert assistant.created_at > 0


def test_from_snapshot_rejects_bad_data():
    with pytest.raises(TypeError):
        Session.from_snapshot({"messages": "not-a-list"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Session.from_snapshot({"messages": [42]})  # type: ignore[list-item]


def test_to_list_and_last_count():
    session = Session()
    for text in ("a", "b", "c"):
        session.append_user(text)
    assert session.count() == 3
    assert [m.content for m in session.last(2)] == ["b", "c"]
    assert session.last(0) == []
    assert [item["content"] for item in session.to_list()] == ["a", "b", "c"]


def test_compressed_boundary_is_validated():
    session = Session()
    session.append_user("a")
    assert session.set_compressed_upto(1) == 1
    with pytest.raises(ValueError):
        session.set_compressed_upto(2)
    session.clear()
    assert session.compressed_upto == 0


def test_missing_message_returns_none_or_raises():
    session = Session()
    assert session.get("missing") is None
    assert session.at(99) is None
    with pytest.raises(SessionError):
        session.remove("missing")


def test_persistence_failure_does_not_break_memory_operations():
    session = Session(store=BrokenStore())
    message = session.append_user("仍然可用")
    assert session.get(message.id) is message
    assert session.length == 1


def test_in_memory_store_round_trip():
    store = InMemorySessionStore()
    session = Session(store=store)
    session.append_user("a")
    session.append_assistant("b")

    restored = Session(session_id=session.session_id, store=store)
    assert [m.content for m in restored.messages] == ["a", "b"]
    assert restored.session_id == session.session_id
