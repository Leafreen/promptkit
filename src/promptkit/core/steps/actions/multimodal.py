# core/steps/actions/multimodal.py
'''多模态生成 / 理解步骤：文生图、文生视频、语音转文字、文字转语音。

四个步骤都是薄壳：

1. 从 Context 读输入（提示词 / 音频 / 文本）
2. 能力协商：`client.supports(modality, operation)` 不满足抛
   `UnsupportedModality`，不静默降级
3. 调 `BaseMultimodalClient`（统一多模态网关），Step 层不直接发 HTTP
4. 把结果写成 `ContentPart`（或文本）放回 Context

`client=None` 时在执行时 lazy 造默认多模态客户端（`make_multimodal_client()`），
方便测试时注入 FakeClient。
'''
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

from promptkit.client_sys.client import UnsupportedModality
from promptkit.core.content import AudioPart, ImagePart, VideoPart
from promptkit.core.steps.interface import Step

if TYPE_CHECKING:  # 只用于类型标注，运行时惰性 import
    from promptkit.client_sys.multimodal import BaseMultimodalClient, Media
    from promptkit.core.context import Context

__all__ = ["ImageGenStep", "VideoGenStep", "STTStep", "TTSStep"]

#: `_part_of` 的输出部件类型：只可能是三种媒体部件
P = TypeVar("P", ImagePart, AudioPart, VideoPart)


def _resolve_client(client: Optional["BaseMultimodalClient"]) -> "BaseMultimodalClient":
    '''构造传入的优先；否则造一个默认多模态客户端（通用网关）。'''
    if client is not None:
        return client
    from promptkit.client_sys.multimodal import make_multimodal_client

    return make_multimodal_client()


def _require(
    client: "BaseMultimodalClient", modality: str, operation: str
) -> None:
    '''能力协商：不支持的模态在发请求前抛 `UnsupportedModality`。'''
    if not client.supports(modality, operation):
        raise UnsupportedModality(client.vendor, f"{modality}:{operation}")


def _part_of(media: "Media", part_cls: Type[P]) -> P:
    '''网关返回的 Media -> 内容部件（url / 本地文件 / base64 三态映射）。'''
    if media.url:
        return part_cls(source=media.url, source_type="url", media_type=media.mime)
    if media.path:
        return part_cls(source=media.path, source_type="file", media_type=media.mime)
    payload = base64.b64encode(media.data or b"").decode("ascii")
    return part_cls(source=payload, source_type="base64", media_type=media.mime)


def _media_of(value: Any) -> "Media":
    '''Context 里的音频输入 -> Media：ContentPart 显式转换，其余交给 Media.from_any。'''
    from promptkit.client_sys.multimodal import Media

    if isinstance(value, (AudioPart, ImagePart, VideoPart)):
        if value.source_type == "url":
            return Media.from_url(value.source, mime=value.media_type)
        if value.source_type == "file":
            return Media.from_path(value.source, mime=value.media_type)
        return Media.from_bytes(
            base64.b64decode(value.source), mime=value.media_type
        )
    return Media.from_any(value)


def _media_of_result(result: Any, what: str) -> "Media":
    '''取生成结果的媒体；网关只回了文本 / 空结果时报错而不是静默放空。'''
    media = getattr(result, "media", None)
    if media is None:
        from promptkit.client_sys.client import ClientError

        raise ClientError(f"{what}没有返回媒体：{getattr(result, 'raw', result)!r}")
    return media


@dataclass
class ImageGenStep(Step):
    '''文生图：`context[output_key] = ImagePart`。

        ImageGenStep("prompt", "image", size="1024x1024")
    '''

    prompt_key: str
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    size: Optional[str] = None
    name: str = "image_gen"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "image", "generate")
        params: Dict[str, Any] = {"size": self.size} if self.size else {}
        result = client.generate("image", str(context[self.prompt_key]), **params)
        context[self.output_key] = _part_of(
            _media_of_result(result, "文生图"), ImagePart
        )
        context.trace.append(self.name)


@dataclass
class VideoGenStep(Step):
    '''文生视频：`context[output_key] = VideoPart`。

        VideoGenStep("prompt", "video", duration=5)
    '''

    prompt_key: str
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    duration: Optional[float] = None
    name: str = "video_gen"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "video", "generate")
        params: Dict[str, Any] = (
            {"duration": self.duration} if self.duration is not None else {}
        )
        result = client.generate("video", str(context[self.prompt_key]), **params)
        context[self.output_key] = _part_of(
            _media_of_result(result, "文生视频"), VideoPart
        )
        context.trace.append(self.name)


@dataclass
class STTStep(Step):
    '''语音转文字：`context[output_key] = str`。

    `audio_key` 指向的值可以是 AudioPart / 路径 / URL / bytes。
    '''

    audio_key: str
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    name: str = "stt"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "audio", "understand")
        result = client.understand("audio", _media_of(context[self.audio_key]))
        context[self.output_key] = result.text or ""
        context.trace.append(self.name)


@dataclass
class TTSStep(Step):
    '''文字转语音：`context[output_key] = AudioPart`。

        TTSStep("text", "voice_over", voice="alloy")
    '''

    text_key: str
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    voice: Optional[str] = None
    name: str = "tts"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "audio", "generate")
        params: Dict[str, Any] = {"voice": self.voice} if self.voice else {}
        result = client.generate(
            "audio", str(context[self.text_key]), output_modality="audio", **params
        )
        context[self.output_key] = _part_of(
            _media_of_result(result, "文字转语音"), AudioPart
        )
        context.trace.append(self.name)

# 职责：多模态生成/理解步骤（文生图、文生视频、STT、TTS），统一走 BaseMultimodalClient
