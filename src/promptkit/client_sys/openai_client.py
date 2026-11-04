# client_sys/openai_client.py
'''OpenAI 家族客户端：OpenAI 官方、DeepSeek、以及任何「OpenAI 兼容」端点。

这类厂商本来就是 Chat Completions 协议，所以这里**几乎不做翻译**：

- 参数原样交给 SDK（`reasoning_effort`、`extra_body`、`temperature`、
  `response_format`……都行，SDK 不认识的会自己报错）
- 返回值直接把 SDK 的对象透出去（同一个形状：`choices[0].message` /
  `choices[0].delta` / `usage`），不重新拷贝一遍
- 重试、超时、连接池交给官方 SDK

想接 Moonshot / 通义 / Ollama / vLLM / 中转站，只要它们支持
`/v1/chat/completions`，写个 `base_url` 就能用（见 `vendor="openai-compatible"`）。
'''
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from promptkit.client_sys.client import (
    BaseClient,
    ClientError,
    ClientSpec,
    UnsupportedModality,
    part_data_url,
)

__all__ = ["OpenAIClient"]


class OpenAIClient(BaseClient):
    '''用官方 `openai` SDK 说话的客户端。

        client = OpenAIClient(api_key="sk-xxx", base_url="https://api.deepseek.com")
        client.create(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}])
        # 等价于 client.chat.completions.create(...)

    `sdk=` 用来注入一个已经建好的 `openai.OpenAI`（测试、或你想自己配
    `http_client` / 代理 / 重试策略的时候）。
    '''

    vendor = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[ClientSpec] = None,
        timeout: Optional[float] = None,
        sdk: Any = None,
    ):
        super().__init__(
            api_key=api_key, base_url=base_url, spec=spec, timeout=timeout
        )
        if sdk is None:
            try:
                from openai import OpenAI
            except ImportError:  # pragma: no cover - openai 是必需依赖，正常装不上才会到这
                raise ClientError(
                    "这个厂商要用 OpenAI 兼容协议，需要先装 openai 包："
                    "`pip install openai`"
                ) from None
            options: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                options["base_url"] = base_url
            if timeout is not None:
                options["timeout"] = timeout
            sdk = OpenAI(**options)
        self.sdk = sdk

    def create(
        self,
        *,
        model: str,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **params: Any,
    ) -> Any:
        '''发一次 Chat Completions 请求。返回值是 SDK 的原始对象。'''
        body: Dict[str, Any] = dict(params)
        body["model"] = model
        body["messages"] = [self._translate_message(m) for m in messages]
        body["stream"] = stream
        if tools:
            body["tools"] = list(tools)
        return self.sdk.chat.completions.create(**body)

    # ---------------- 多模态 ----------------
    def supports(self, modality: str) -> bool:
        '''OpenAI 兼容端点：文本 + 图片 + 音频 + 文件（视频不支持）。'''
        return modality in ("text", "image", "audio", "file")

    def _to_vendor_part(self, part: Mapping[str, Any]) -> Dict[str, Any]:
        '''标准部件 -> OpenAI content part。'''
        kind = part.get("type")
        if kind == "image":
            url = part_data_url(part)
            image_url: Dict[str, Any] = {"url": url}
            if part.get("detail"):
                image_url["detail"] = part["detail"]
            return {"type": "image_url", "image_url": image_url}
        if kind == "audio":
            if part.get("source_type") == "url":
                raise ClientError(
                    "OpenAI 的音频输入只支持 base64，不支持 URL；请先下载"
                )
            media_type = str(part.get("media_type") or "audio/mpeg")
            fmt = media_type.split("/")[-1].replace("mpeg", "mp3")
            return {
                "type": "input_audio",
                "input_audio": {"data": part_data_url(part).split(",", 1)[-1],
                                "format": fmt},
            }
        if kind == "file":
            if part.get("source_type") == "url":
                raise ClientError(
                    "OpenAI 的文件输入只支持 base64，不支持 URL；请先下载"
                )
            return {
                "type": "file",
                "file": {
                    "filename": part.get("filename") or "file",
                    "file_data": part_data_url(part),
                },
            }
        raise UnsupportedModality(self.vendor, str(kind))

    def close(self) -> None:
        '''关掉 SDK 的 HTTP 连接池（有这个方法才关，没有就算了）。'''
        closer = getattr(self.sdk, "close", None)
        if callable(closer):
            closer()
