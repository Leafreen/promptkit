'''Store 单元测试：InMemory / SQLite 的读写、删除、JSON 序列化与隔离。'''
import sqlite3

from promptkit.conversation.message import Message
from promptkit.conversation.session import Session, SessionState
from promptkit.conversation.store import InMemorySessionStore, SQLiteSessionStore


def test_in_memory_store_saves_copies():
    store = InMemorySessionStore()
    state = SessionState(
        session_id="s1",
        messages=[Message(role="user", content="a")],
        metadata={"k": "v"},
    )
    store.save(state)
    state.messages.clear()

    loaded = store.load("s1")
    assert loaded is not None
    assert [m.content for m in loaded.messages] == ["a"]
    assert loaded.metadata == {"k": "v"}


def test_in_memory_delete_and_list():
    store = InMemorySessionStore()
    store.save(SessionState(session_id="a"))
    store.save(SessionState(session_id="b"))
    assert set(store.list_ids()) == {"a", "b"}
    assert store.delete("a") is True
    assert store.delete("missing") is False
    assert store.list_ids() == ["b"]


def test_sqlite_creates_expected_tables(tmp_path):
    path = str(tmp_path / "nested" / "conversation.db")
    SQLiteSessionStore(path)
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "conversation_sessions" in tables
    assert "conversation_messages" in tables


def test_sqlite_round_trip_with_multimodal_content(tmp_path):
    store = SQLiteSessionStore(str(tmp_path / "conversation.db"))
    parts = [{"type": "text", "text": "看图"}, {"type": "image_url", "image_url": {"url": "u"}}]
    state = SessionState(
        session_id="s1",
        messages=[
            Message(role="user", content=parts, id="m1"),
            Message(role="tool", content="晴", id="m2", metadata={"tool_call_id": "c1"}),
        ],
        metadata={"owner": "tester"},
        compressed_upto=1,
    )
    store.save(state)

    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.metadata == {"owner": "tester"}
    assert loaded.compressed_upto == 1
    assert loaded.messages[0].id == "m1"
    assert loaded.messages[0].content == parts
    assert loaded.messages[1].metadata["tool_call_id"] == "c1"


def test_sqlite_persists_across_store_instances(tmp_path):
    path = str(tmp_path / "conversation.db")
    first = SQLiteSessionStore(path)
    first.save(SessionState(session_id="s1", messages=[Message(role="user", content="a")]))

    second = SQLiteSessionStore(path)
    loaded = second.load("s1")
    assert loaded is not None
    assert loaded.messages[0].content == "a"


def test_sqlite_load_missing_returns_none(tmp_path):
    store = SQLiteSessionStore(str(tmp_path / "conversation.db"))
    assert store.load("missing") is None


def test_sqlite_delete_and_list_ids(tmp_path):
    store = SQLiteSessionStore(str(tmp_path / "conversation.db"))
    store.save(SessionState(session_id="a"))
    store.save(SessionState(session_id="b"))
    assert set(store.list_ids()) == {"a", "b"}
    assert store.delete("a") is True
    assert store.delete("missing") is False
    assert store.list_ids() == ["b"]


def test_session_writes_through_sqlite_store(tmp_path):
    path = str(tmp_path / "conversation.db")
    store = SQLiteSessionStore(path)
    session = Session(store=store)
    session.append_user("a")
    session.append_assistant("b")

    restored = Session(session_id=session.session_id, store=SQLiteSessionStore(path))
    assert [m.content for m in restored.messages] == ["a", "b"]


def test_session_clear_is_persisted(tmp_path):
    path = str(tmp_path / "conversation.db")
    store = SQLiteSessionStore(path)
    session = Session(store=store)
    session.append_user("a")
    session.clear()

    restored = Session(session_id=session.session_id, store=SQLiteSessionStore(path))
    assert restored.length == 0
