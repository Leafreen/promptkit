'''ContentPart：构造、to_base64、序列化与部件推断。'''
import base64

import pytest

from promptkit.core.content import (
    AudioPart,
    FilePart,
    ImagePart,
    TextPart,
    VideoPart,
    part_from_any,
    part_from_dict,
    part_from_string,
)


def test_text_part_defaults():
    part = TextPart(text="你好")
    assert part.type == "text"
    assert part.source == "你好"  # source 是 text 的别名
    assert part.source_type == "inline"
    assert part.media_type == "text/plain"
    assert not hasattr(part, "to_base64")  # 文本没有二进制内容


def test_to_base64_only_on_image_and_file():
    assert hasattr(ImagePart(source="x"), "to_base64")
    assert hasattr(FilePart(source="x"), "to_base64")
    assert not hasattr(AudioPart(source="x"), "to_base64")
    assert not hasattr(VideoPart(source="x"), "to_base64")


def test_to_base64_passthrough():
    part = ImagePart(source="aGk=", source_type="base64", media_type="image/png")
    assert part.to_base64() == "aGk="


def test_to_base64_reads_file(tmp_path):
    payload = tmp_path / "a.png"
    payload.write_bytes(b"hi")
    part = ImagePart(source=str(payload), source_type="file", media_type="image/png")
    assert base64.b64decode(part.to_base64()) == b"hi"


def test_to_base64_rejects_url():
    part = FilePart(source="https://example.com/a.pdf")
    with pytest.raises(ValueError, match="没有本地字节"):
        part.to_base64()


def test_to_dict_roundtrip():
    parts = [
        TextPart(text="看图"),
        ImagePart(
            source="https://example.com/a.png",
            source_type="url",
            media_type="image/png",
            detail="high",
        ),
        AudioPart(source="aGk=", source_type="base64", media_type="audio/mpeg"),
        VideoPart(source="/tmp/v.mp4", source_type="file", media_type="video/mp4"),
        FilePart(
            source="https://example.com/a.pdf",
            source_type="url",
            media_type="application/pdf",
            filename="a.pdf",
        ),
    ]
    for part in parts:
        assert part_from_dict(part.to_dict()) == part


def test_to_dict_shape():
    data = ImagePart(
        source="u", source_type="url", media_type="image/png", detail="low"
    ).to_dict()
    assert data == {
        "type": "image",
        "source": "u",
        "source_type": "url",
        "media_type": "image/png",
        "detail": "low",
    }
    # detail 为 None 时不输出；TextPart 不输出 source / source_type / media_type
    assert "detail" not in ImagePart(source="u").to_dict()
    assert TextPart(text="hi").to_dict() == {"type": "text", "text": "hi"}


def test_part_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError, match="未知的内容部件类型"):
        part_from_dict({"type": "hologram", "source": "x"})


def test_part_from_dict_requires_source_for_media():
    with pytest.raises(KeyError):
        part_from_dict({"type": "image"})


def test_part_from_string_infers_by_extension():
    cases = {
        "a.png": ImagePart,
        "b.jpg": ImagePart,
        "c.mp3": TextPart,  # 音频没有扩展名表，按文本处理
        "d.mp4": TextPart,
        "e.pdf": FilePart,
        "f.txt": TextPart,
    }
    for path, cls in cases.items():
        assert isinstance(part_from_string(path), cls)


def test_part_from_string_infers_url_and_mime():
    part = part_from_string("https://example.com/pic.webp")
    assert isinstance(part, ImagePart)
    assert part.source_type == "url"
    assert part.media_type == "image/webp"


def test_part_from_string_local_file_is_file_source():
    part = part_from_string("cat.png")
    assert isinstance(part, ImagePart)
    assert part.source_type == "file"


def test_part_from_any():
    image = ImagePart(source="u")
    assert part_from_any(image) is image
    assert isinstance(part_from_any("a.png"), ImagePart)
    assert part_from_any("随便一句话") == TextPart(text="随便一句话")
    with pytest.raises(TypeError):
        part_from_any(123)
    with pytest.raises(TypeError):
        part_from_any({"type": "text", "text": "hi"})  # dict 走 part_from_dict


def test_parts_are_frozen():
    part = TextPart(text="hi")
    with pytest.raises(AttributeError):
        part.text = "changed"  # type: ignore[misc]
