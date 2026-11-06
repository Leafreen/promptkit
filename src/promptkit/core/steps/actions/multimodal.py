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
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

from promptkit.client_sys.client import UnsupportedModality
from promptkit.core.content import (
    AudioPart,
    FilePart,
    ImagePart,
    VideoPart,
    part_from_dict,
)
from promptkit.core.steps.interface import Step

if TYPE_CHECKING:  # 只用于类型标注，运行时惰性 import
    from promptkit.client_sys.multimodal import BaseMultimodalClient, Media
    from promptkit.core.context import Context
    from promptkit.prompt import Prompt

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

    if isinstance(value, (AudioPart, ImagePart, VideoPart, FilePart)):
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


def _prompt_of(
    value: object,
    context: "Context",
    step_cautions: object = None,
    *,
    use_defaults: bool = True,
) -> "Prompt":
    '''把步骤输入统一成「带约束的 Prompt」。

    - `Prompt(...)`：原样使用
    - 其它（字符串等）：包成 `Prompt(content=str(value))`

    然后和 LLMStep 一样套约束：全局默认 -> Workflow -> Step，拼到末尾。
    '''
    from promptkit.core.model_action.caution_related.caution import (
        WORKFLOW_CAUTIONS_KEY,
        apply_cautions,
    )
    from promptkit.prompt import Prompt

    prompt = value if isinstance(value, Prompt) else Prompt(content=str(value))
    return apply_cautions(
        prompt,
        step_cautions,
        context.get(WORKFLOW_CAUTIONS_KEY),
        use_defaults=use_defaults,
    )


def _prompt_input(
    value: object, context: "Context"
) -> Tuple[str, Tuple["Media", ...]]:
    '''把步骤的输入拆成「文本提示词 + 参考媒体」。

    - 普通值（字符串等）：原样转成文本，没有参考媒体（老行为）
    - `Prompt(...)`：按 Context 渲染，文本部件拼成 prompt，图片 / 视频等
      媒体部件转成 `Media` 作为 `inputs`，从而实现以图生图 / 以图生视频
    '''
    from promptkit.prompt import Prompt

    if not isinstance(value, Prompt):
        return str(value), ()

    rendered = value.render(**context.to_dict())
    content = rendered.get("content")
    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        return "" if content is None else str(content), ()

    texts = []
    inputs = []
    for part in content:
        if part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
            continue
        media = _media_of(part_from_dict(part))
        # 只把图片 / 视频当参考输入；音频等其它部件也原样带上，交由客户端判断
        inputs.append(media)
    return "\n".join(texts), tuple(inputs)


@dataclass
class ImageGenStep(Step):
    '''文生图 / 以图生图：`context[output_key] = ImagePart`。

        ImageGenStep("prompt", "image", size="1024x1024")

        # 以图生图：Prompt 里的图片部件会作为参考图传给多模态客户端
        ImageGenStep(
            Prompt(TextPart("把这只猫变成水彩画"), ImagePart(Var("cat_image"))),
            "result",
        )

        # 约束和 LLMStep 同一套语法：cautions=[...]，全局 / Workflow 级也生效
        ImageGenStep(
            Prompt("user", "给我生成一只猫"),
            "image",
            cautions=[Style("水彩画"), NoText()],
        )

    `prompt_key` 既可以是 Context 里的变量名，也可以直接是一个 `Prompt`
    （变量用 `Var(...)` 引用，渲染时从 Context 取值）。
    '''

    prompt_key: Union[str, "Prompt"]
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    size: Optional[str] = None
    #: 约束：Caution / 字符串 / 它们的列表 / 组合，与 LLMStep 语义一致
    cautions: Optional[Sequence[object]] = None
    #: 是否合并全局默认约束（set_default_cautions 设的）
    use_default_cautions: bool = True
    name: str = "image_gen"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "image", "generate")
        source = (
            self.prompt_key
            if not isinstance(self.prompt_key, str)
            else context[self.prompt_key]
        )
        prompt_obj = _prompt_of(
            source, context, self.cautions, use_defaults=self.use_default_cautions
        )
        prompt, inputs = _prompt_input(prompt_obj, context)
        params: Dict[str, Any] = {"size": self.size} if self.size else {}
        if inputs:
            params["inputs"] = inputs
        result = client.generate("image", prompt, **params)
        context[self.output_key] = _part_of(
            _media_of_result(result, "生图"), ImagePart
        )
        context.trace.append(self.name)


@dataclass
class VideoGenStep(Step):
    '''文生视频 / 以图生视频：`context[output_key] = VideoPart`。

        VideoGenStep("prompt", "video", duration=5)

        # 以图生视频：首帧图片从 Context 取
        VideoGenStep(
            Prompt(TextPart("让猫动起来"), ImagePart(Var("cat_image"))),
            "result",
            duration=5,
            cautions=[Style("电影感"), Quality("高清")],
        )

    `prompt_key` 同样既可以是变量名，也可以直接是一个 `Prompt`；
    `cautions` 与 LLMStep 语义一致（全局默认 / Workflow / Step 三层合并）。
    '''

    prompt_key: Union[str, "Prompt"]
    output_key: str
    client: Optional["BaseMultimodalClient"] = None
    duration: Optional[float] = None
    #: 约束：Caution / 字符串 / 它们的列表 / 组合，与 LLMStep 语义一致
    cautions: Optional[Sequence[object]] = None
    #: 是否合并全局默认约束（set_default_cautions 设的）
    use_default_cautions: bool = True
    name: str = "video_gen"

    def execute(self, context: "Context") -> None:
        client = _resolve_client(self.client)
        _require(client, "video", "generate")
        source = (
            self.prompt_key
            if not isinstance(self.prompt_key, str)
            else context[self.prompt_key]
        )
        prompt_obj = _prompt_of(
            source, context, self.cautions, use_defaults=self.use_default_cautions
        )
        prompt, inputs = _prompt_input(prompt_obj, context)
        params: Dict[str, Any] = (
            {"duration": self.duration} if self.duration is not None else {}
        )
        if inputs:
            params["inputs"] = inputs
        result = client.generate("video", prompt, **params)
        context[self.output_key] = _part_of(
            _media_of_result(result, "生视频"), VideoPart
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
