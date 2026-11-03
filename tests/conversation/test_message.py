'''Message 单元测试：不可变、id、content 形状、dict 互转。'''
import dataclasses

import pytest

from promptkit.conversation.message import Message


def test_id_is_uuid4_hex_and_unique():
    first = Message(role="user", content="a")
    second = Message(role="user", content="a")
    assert len(first.id) == 32
    int(first.id, 16)  # 必须是纯十六进制
    assert first.id != second.id


def test_message_is_frozen():
    message = Message(role="user", content="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "改了"  # type: ignore[misc]


def test_content_supports_multimodal_list():
    parts = [{"type": "text", "text": "看图"}, {"type": "image_url", "image_url": {"url": "x"}}]
    message = Message(role="user", content=parts)
    parts.append({"type": "text", "text": "后来的"})
    assert message.content == [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]


def test_metadata_is_defensively_copied():
    metadata = {"tool_calls": [{"id": "c1"}]}
    message = Message(role="assistant", content="", metadata=metadata)
    metadata["tool_calls"].append({"id": "c2"})
    assert message.metadata["tool_calls"] == [{"id": "c1"}]


def test_with_content_returns_new_message_with_same_id():
    message = Message(role="user", content="旧")
    changed = message.with_content("新")
    assert changed.id == message.id
    assert changed.content == "新"
    assert message.content == "旧"


def test_with_metadata_merges_without_mutating_original():
    message = Message(role="tool", content="晴", metadata={"tool_call_id": "c1"})
    changed = message.with_metadata(name="weather")
    assert changed.metadata == {"tool_call_id": "c1", "name": "weather"}
    assert message.metadata == {"tool_call_id": "c1"}


def test_to_dict_shape_matches_old_interface():
    message = Message(
        role="tool",
        content="晴",
        metadata={"tool_call_id": "c1", "name": "weather"},
    )
    payload = message.to_dict()
    assert payload["role"] == "tool"
    assert payload["content"] == "晴"
    assert payload["tool_call_id"] == "c1"
    assert payload["name"] == "weather"
    assert "id" not in payload


def test_from_dict_puts_extra_keys_into_metadata():
    message = Message.from_dict(
        {
            "role": "assistant",
            "content": "好了",
            "tool_calls": [{"id": "c1"}],
            "reasoning_content": "想想",
        }
    )
    assert message.role == "assistant"
    assert message.metadata["tool_calls"] == [{"id": "c1"}]
    assert message.metadata["reasoning_content"] == "想想"


def test_from_dict_normalises_none_content_and_unknown_role():
    message = Message.from_dict({"role": "nope", "content": None})
    assert message.role == "user"
    assert message.content == ""


def test_invalid_content_type_is_rejected():
    with pytest.raises(TypeError):
        Message(role="user", content=123)  # type: ignore[arg-type]
