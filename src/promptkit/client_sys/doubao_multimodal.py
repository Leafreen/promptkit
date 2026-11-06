# client_sys/doubao_multimodal.py
'''豆包（火山方舟）多模态客户端 —— 文生图 / 文生视频 / 图片理解。

豆包方舟统一端点 `https://ark.cn-beijing.volces.com/api/v3` 是 OpenAI 兼容的，
所以直接复用 `openai.OpenAI` SDK（和 DoubaoClient 一样），不需要自己写 HTTP。

豆包特有 API：

| 能力 | OpenAI 兼容？ | 端点 | SDK 方法 |
|---|---|---|---|
| 文生图 | ✅ | `/images/generations` | `sdk.images.generate()` |
| 文生视频 | ❌（异步任务） | `/contents/generations/tasks` | # TODO |
| 图片理解 | ✅ | `/chat/completions` | `sdk.chat.completions.create()` with vision |
| 语音转文字 | ✅ | `/audio/transcriptions` | `sdk.audio.transcriptions.create()` |
| 文字转语音 | ✅ | `/audio/speech` | `sdk.audio.speech.create()` |

generate("image", ...) / generate("text", ...) / understand("image", ...) 都走
这个客户端；video generate 留 TODO，因为方舟 Seedance 的异步任务接口还需要确认。

注册到 `multimodal._FACTORIES["doubao"]` 后，`make_multimodal_client("doubao")` 会拿到
这个客户端。ImageGenStep 默认用 generic，但显式传 `client=make_multimodal_client("doubao")`
就走豆包。
'''
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import base64
import mimetypes

from promptkit import config as config_module
from promptkit.client_sys.multimodal import (
    BaseMultimodalClient,
    Media,
    MultimodalError,
    MultimodalRequest,
    MultimodalResult,
)

__all__ = ["DoubaoMultimodalClient"]

logger = logging.getLogger(__name__)


class DoubaoMultimodalClient(BaseMultimodalClient):
    '''豆包方舟多模态客户端。

    所有模态共用 OpenAI SDK —— 方舟端点和 Chat Completions 一样。
    '''

    vendor = "doubao"

    #: 默认文生图模型（Seedream 5.0 flash，最低 960x960 像素）
    DEFAULT_IMAGE_MODEL = config_module.DOUBAO_IMAGE_MODEL

    #: 默认 Seedance 视频模型
    DEFAULT_VIDEO_MODEL = config_module.DOUBAO_VIDEO_MODEL

    #: 默认文本/对话模型（图片理解用）
    DEFAULT_TEXT_MODEL = config_module.DOUBAO_TEXT_MODEL

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or config_module.DOUBAO_BASE_URL,
            spec=spec,
            timeout=timeout,
        )
        # 惰性 import —— openai 包是可选的（但 DoubaoClient 已经依赖了）
        from openai import OpenAI

        self.sdk = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # BaseMultimodalClient 接口
    # ------------------------------------------------------------------

    def run(self, request: MultimodalRequest) -> MultimodalResult:
        '''按 modality × operation 分发到具体实现。'''
        modality = request.modality
        operation = request.operation.value if hasattr(request.operation, "value") else str(request.operation)

        try:
            if operation == "generate":
                if modality == "image":
                    return self._generate_image(request)
                if modality == "video":
                    return self._generate_video(request)
                if modality in ("audio", "speech"):
                    return self._generate_audio(request)
                if modality in ("text",):
                    return self._generate_text(request)
                raise MultimodalError(f"豆包不支持 generate:{modality}")

            if operation == "understand":
                if modality == "image":
                    return self._understand_image(request)
                if modality in ("audio", "speech"):
                    return self._understand_audio(request)
                if modality == "video":
                    return self._understand_video(request)
                if modality in ("text",):
                    return self._understand_text(request)
                raise MultimodalError(f"豆包不支持 understand:{modality}")

            raise MultimodalError(f"未知操作 {operation!r}")
        except MultimodalError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK 异常统一包成 MultimodalError
            raise MultimodalError(f"豆包多模态请求失败：{type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    # generate: image —— POST /images/generations（OpenAI 兼容）
    # ------------------------------------------------------------------

    def _generate_image(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_IMAGE_MODEL
        params = dict(request.params or {})

        # size / n / response_format 透传
        response_format = params.pop("response_format", "url")

        logger.info("豆包文生图: model=%s prompt=%r...", model, (request.prompt or "")[:40])

        # 以图生图：Seedream 的参考图参数是 image（URL 或 base64 data URL）
        refs = [
            media.url if media.url else media.to_data_url()
            for media in request.inputs
        ]
        extra_body = dict(params.pop("extra_body", {}) or {})
        if refs:
            extra_body["image"] = refs[0] if len(refs) == 1 else refs

        resp = self.sdk.images.generate(
            model=model,
            prompt=request.prompt or "",
            size=params.pop("size", "1024x1024"),
            n=params.pop("n", 1),
            response_format=response_format,
            extra_body=extra_body or None,
            **params,
        )

        # 返回第一张图的 URL
        url = resp.data[0].url if resp.data else None
        if url is None and resp.data and resp.data[0].b64_json:
            import base64
            data = base64.b64decode(resp.data[0].b64_json)
            media = Media(data=data, mime="image/png")
        elif url:
            media = Media(url=url, mime="image/png")
        else:
            raise MultimodalError(f"豆包文生图没有返回图片：{resp}")

        # 可选落盘
        if request.output_path:
            media.save(request.output_path)

        return MultimodalResult(
            operation="generate",
            modality="image",
            model=model,
            media=media,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # generate: video —— Seedance 异步任务（TODO：确认请求格式）
    # ------------------------------------------------------------------

    def _generate_video(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_VIDEO_MODEL
        logger.warning(
            "豆包文生视频 Seedance 异步任务 API 格式待确认 —— "
            "临时用 Chat Completions 模拟，不要用于生产。model=%s", model
        )
        # TODO: 确认 Seedance /contents/generations/tasks 请求格式（任务创建 / 轮询 / 分辨率参数）
        raise MultimodalError(
            "豆包 Seedance 文生视频 API 格式待确认 —— 请先实现 "
            "_generate_video() 的异步任务创建与轮询逻辑"
        )

    # ------------------------------------------------------------------
    # generate: text —— 走 Chat Completions（纯文本生成）
    # ------------------------------------------------------------------

    def _generate_text(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_TEXT_MODEL
        resp = self.sdk.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt or ""}],
        )
        text = resp.choices[0].message.content or ""
        return MultimodalResult(
            operation="generate",
            modality="text",
            model=model,
            text=text,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # generate: audio —— 文字转语音 /audio/speech
    # ------------------------------------------------------------------

    def _generate_audio(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or "doubao-tts-1-240615"
        voice = (request.params or {}).get("voice", "alloy")
        resp = self.sdk.audio.speech.create(
            model=model,
            input=request.prompt or "",
            voice=voice,
        )
        # resp.content 是 bytes
        return MultimodalResult(
            operation="generate",
            modality="audio",
            model=model,
            media=Media(data=resp.content, mime="audio/mpeg"),
            raw=resp,
        )

    # ------------------------------------------------------------------
    # understand: image —— Chat Completions with vision input
    # ------------------------------------------------------------------

    def _understand_image(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_TEXT_MODEL
        # 把 inputs 转成 image_url block
        inputs: list[Media] = list(request.inputs or [])
        content: list = [{"type": "text", "text": request.prompt or "描述这张图片"}]
        for media in inputs:
            if isinstance(media, Media):
                url = media.to_data_url()  # 本地文件转 data URL，远程 URL 原样
                content.append({"type": "image_url", "image_url": {"url": url}})

        resp = self.sdk.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        text = resp.choices[0].message.content or ""
        return MultimodalResult(
            operation="understand",
            modality="image",
            model=model,
            text=text,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # understand: audio —— 语音转文字 /audio/transcriptions
    # ------------------------------------------------------------------

    def _understand_audio(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or "whisper-1"
        media = (request.inputs or [None])[0]
        if not isinstance(media, Media):
            raise MultimodalError("understand:audio 需要 Media 输入")

        # 本地文件开 File object，URL / bytes 转 NamedTempFile
        if media.path and os.path.exists(media.path):
            with open(media.path, "rb") as handle:
                resp = self.sdk.audio.transcriptions.create(model=model, file=handle)
        else:
            raise MultimodalError("音频理解只支持本地文件路径输入")

        return MultimodalResult(
            operation="understand",
            modality="audio",
            model=model,
            text=resp.text,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # understand: video —— Chat Completions with video_url input
    # ------------------------------------------------------------------

    def _understand_video(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_TEXT_MODEL
        inputs: list[Media] = list(request.inputs or [])
        content: list = [{"type": "text", "text": request.prompt or "描述这段视频"}]
        for media in inputs:
            if isinstance(media, Media) and media.url:
                content.append({"type": "video_url", "video_url": {"url": media.url}})

        resp = self.sdk.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        text = resp.choices[0].message.content or ""
        return MultimodalResult(
            operation="understand",
            modality="video",
            model=model,
            text=text,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # understand: text —— 纯文本理解（就是对话）
    # ------------------------------------------------------------------

    def _understand_text(self, request: MultimodalRequest) -> MultimodalResult:
        model = request.model or self.DEFAULT_TEXT_MODEL
        resp = self.sdk.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt or ""}],
        )
        text = resp.choices[0].message.content or ""
        return MultimodalResult(
            operation="understand",
            modality="text",
            model=model,
            text=text,
            raw=resp,
        )
# 职责：豆包方舟多模态客户端 —— 文生图/文生视频（TODO）/图片理解/语音转文字/文字转语音，复用 OpenAI SDK
