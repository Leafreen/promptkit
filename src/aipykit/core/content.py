# core/content.py
'''多模态内容部件：一条消息的 content 可以是若干「部件」的列表。

薄、显式、零魔法：

- 五个部件类型，全部是 ``@dataclass(frozen=True)``，用 ``Literal`` 标 ``type``，
  不引入枚举
- 每个部件有统一的四个字段：``type`` / ``source`` / ``source_type`` / ``media_type``
  （文本的「来源」就是它自己，``source`` 是 ``text`` 的别名，``source_type`` 恒为
  ``"inline"``）
- ``ImagePart`` 额外有 ``detail``，``FilePart`` 额外有 ``filename``
- ``to_dict()`` 输出 JSON 可序列化的字典 —— Session 快照和厂商转换都基于它；
  文本部件的字典形状和 OpenAI / Claude 的 text block 完全一致，不需要再翻译
- ``part_from_string()`` 按扩展名推断部件类型：``.png`` -> ``ImagePart``，
  ``.pdf`` -> ``FilePart``，其他 -> ``TextPart``

厂商相关的字段映射（image_url / video_url / document …）不在这里做，
由各 ``LLMClient._to_vendor_part`` 负责。
'''
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union
from typing_extensions import Literal

__all__ = [
    "SourceType",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "VideoPart",
    "FilePart",
    "ContentPart",
    "CONTENT_PART_TYPES",
    "part_from_string",
    "part_from_dict",
    "part_from_any",
]

#: 媒体来源：远程 URL / 本地文件 / 已经编码好的 base64
SourceType = Literal["url", "file", "base64"]

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_FILE_EXTS = {".pdf"}


def _source_to_base64(source: str, source_type: str) -> str:
    '''base64 原样返回；file 读取编码；url 没有本地字节，拒绝。'''
    if source_type == "base64":
        return source
    if source_type == "file":
        with open(source, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    raise ValueError(
        f"来源类型是 {source_type!r}，没有本地字节可编码；"
        "请先用 file / base64 来源，或直接把 URL 传给支持它的厂商"
    )


def _guess_media_type(name: str) -> Optional[str]:
    guessed, _ = mimetypes.guess_type(name)
    return guessed


@dataclass(frozen=True)
class TextPart:
    '''一段文本。模板替换只发生在文本部件上。'''

    text: str
    type: Literal["text"] = "text"
    source_type: Literal["inline"] = "inline"
    media_type: str = "text/plain"

    @property
    def source(self) -> str:
        '''统一字段：文本的「来源」就是文本本身。'''
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        '''和 OpenAI / Claude 的 text block 形状一致，可直接上线。'''
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImagePart:
    '''一张图片。``detail`` 透传给支持的厂商（low / high / auto）。'''

    source: str
    source_type: SourceType = "url"
    media_type: Optional[str] = None
    detail: Optional[str] = None
    type: Literal["image"] = "image"

    def to_base64(self) -> str:
        '''拿到 base64 编码的内容。url 来源没有本地字节，抛 ``ValueError``。'''
        return _source_to_base64(self.source, self.source_type)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "image",
            "source": self.source,
            "source_type": self.source_type,
            "media_type": self.media_type,
        }
        if self.detail is not None:
            data["detail"] = self.detail
        return data


@dataclass(frozen=True)
class AudioPart:
    '''一段音频。'''

    source: str
    source_type: SourceType = "url"
    media_type: Optional[str] = None
    type: Literal["audio"] = "audio"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "audio",
            "source": self.source,
            "source_type": self.source_type,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class VideoPart:
    '''一段视频。'''

    source: str
    source_type: SourceType = "url"
    media_type: Optional[str] = None
    type: Literal["video"] = "video"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "video",
            "source": self.source,
            "source_type": self.source_type,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class FilePart:
    '''一个文件（PDF 等）。``filename`` 给需要的厂商接口用。'''

    source: str
    source_type: SourceType = "url"
    media_type: Optional[str] = None
    filename: Optional[str] = None
    type: Literal["file"] = "file"

    def to_base64(self) -> str:
        '''拿到 base64 编码的内容。url 来源没有本地字节，抛 ``ValueError``。'''
        return _source_to_base64(self.source, self.source_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "file",
            "source": self.source,
            "source_type": self.source_type,
            "media_type": self.media_type,
            "filename": self.filename,
        }


#: 全部内容部件的联合类型
ContentPart = Union[TextPart, ImagePart, AudioPart, VideoPart, FilePart]

#: type 字符串 -> 部件类，反序列化用
CONTENT_PART_TYPES: Dict[str, Type[ContentPart]] = {
    "text": TextPart,
    "image": ImagePart,
    "audio": AudioPart,
    "video": VideoPart,
    "file": FilePart,
}


def part_from_string(value: str) -> ContentPart:
    '''把字符串按扩展名推断成内容部件（``Var`` 解析出字符串时的规则）。

    - ``.png`` / ``.jpg`` 等图片扩展名 -> ``ImagePart``
    - ``.pdf`` -> ``FilePart``
    - 其他 -> ``TextPart``

    ``http(s)://`` 开头的来源记为 ``url``，其余记为 ``file``。
    '''
    text = value.strip()
    source_type: SourceType = (
        "url" if text.startswith(("http://", "https://")) else "file"
    )
    ext = os.path.splitext(text.split("?")[0])[1].lower()
    if ext in _IMAGE_EXTS:
        return ImagePart(
            source=value, source_type=source_type, media_type=_guess_media_type(text)
        )
    if ext in _FILE_EXTS:
        return FilePart(
            source=value,
            source_type=source_type,
            media_type=_guess_media_type(text),
            filename=os.path.basename(text) or None,
        )
    return TextPart(text=value)


def part_from_dict(data: Dict[str, Any]) -> ContentPart:
    '''把 ``to_dict()`` 的输出还原成部件（Session 快照反序列化用）。'''
    kind = data.get("type")
    cls = CONTENT_PART_TYPES.get(str(kind))
    if cls is None:
        raise ValueError(f"未知的内容部件类型：{kind!r}")
    if cls is TextPart:
        return TextPart(text=str(data.get("text") or ""))
    if cls is ImagePart:
        return ImagePart(
            source=str(data["source"]),
            source_type=data.get("source_type") or "url",
            media_type=data.get("media_type"),
            detail=data.get("detail"),
        )
    if cls is FilePart:
        return FilePart(
            source=str(data["source"]),
            source_type=data.get("source_type") or "url",
            media_type=data.get("media_type"),
            filename=data.get("filename"),
        )
    if cls is AudioPart:
        return AudioPart(
            source=str(data["source"]),
            source_type=data.get("source_type") or "url",
            media_type=data.get("media_type"),
        )
    return VideoPart(
        source=str(data["source"]),
        source_type=data.get("source_type") or "url",
        media_type=data.get("media_type"),
    )


def part_from_any(value: object) -> ContentPart:
    '''``ContentPart`` 原样返回；字符串按扩展名推断；其他类型拒绝。'''
    if isinstance(value, (TextPart, ImagePart, AudioPart, VideoPart, FilePart)):
        return value
    if isinstance(value, str):
        return part_from_string(value)
    raise TypeError(
        f"无法把 {type(value).__name__} 当成内容部件："
        "需要 ContentPart 或字符串（按扩展名推断）"
    )


# 职责：定义多模态内容部件（TextPart/ImagePart/AudioPart/VideoPart/FilePart）及其序列化、字符串推断
