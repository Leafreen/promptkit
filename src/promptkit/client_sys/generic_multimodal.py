# client_sys/generic_multimodal.py
'''通用多模态网关：所有模态走同一个 JSON 接口。

客户端只做四件事：参数校验、输入格式标准化、POST 到同一个端点、
把网关返回的 JSON 映射回 `MultimodalResult`。这里没有按模态分支，
也没有任何厂商专属的解析 / 生成 / 格式转换逻辑 —— 模型路由由请求里的
`model` 与 `output` 参数决定，具体实现完全在网关侧。

请求信封：

    {
      "operation": "generate" | "understand",
      "modality": "text" | "image" | ...,
      "model": "xxx",                       # 可选，由调用方指定
      "prompt": "...",
      "inputs": [{"type": "image", "mime": "image/png",
                  "name": "cat.png", "data_base64": "..."}],
      "output": {"modality": "image", "format": "png"},
      "params": {...}
    }

响应信封（字段全部可选，未知字段原样保留在 `raw`）：

    {
      "text": "...",
      "media": {"url": "...", "data_base64": "...", "mime": "..."},
      "usage": {...},
      "task_id": "..."
    }

HTTP 仍走标准库 urllib；`transport=` 可注入假传输层，测试和离线例程
不需要网络。日志写到 `promptkit.multimodal`，请求体里的 base64
不会出现在 INFO 日志中。
'''
from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from promptkit import config as config_module
from promptkit.client_sys.multimodal import (
    GENERIC,
    BaseMultimodalClient,
    Media,
    MultimodalError,
    MultimodalRequest,
    MultimodalResult,
    MultimodalSpec,
)

__all__ = ["GenericMultimodalClient", "HttpTransport", "REQUEST_LOGGER"]

#: (method, url, headers, body, timeout) -> (status, headers, body)
HttpTransport = Callable[
    [str, str, Mapping[str, str], Optional[bytes], float],
    Tuple[int, Mapping[str, str], bytes],
]

#: 多模态请求日志（不配置 handler 时由应用自己的 logging 接管）
REQUEST_LOGGER = logging.getLogger("promptkit.multimodal")


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
    timeout: float,
) -> Tuple[int, Mapping[str, str], bytes]:
    '''默认传输层：标准库 urllib。HTTPError 也把响应体读回来。'''
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        raise MultimodalError(
            f"{method} {url} -> HTTP {exc.code}: "
            f"{payload.decode('utf-8', 'replace')[:500]}"
        ) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MultimodalError(
                f"{method} {url} 超时（{timeout:.1f}s）"
            ) from None
        raise MultimodalError(f"{method} {url} 连接失败：{reason}") from None
    except (TimeoutError, socket.timeout) as exc:
        raise MultimodalError(f"{method} {url} 超时（{timeout:.1f}s）") from exc


class GenericMultimodalClient(BaseMultimodalClient):
    '''通用多模态客户端。

        client = GenericMultimodalClient(api_key="...", base_url="https://gw/v1")
        client.generate("image", "一只猫在月球上", save_to="cat.png")
        client.generate("audio", "朗读这段话", output_format="mp3")
        client.understand("rich_text", Media.from_text("<h1>你好</h1>", "text/html"),
                          prompt="总结")

    模态名不做白名单校验：只要网关认识，客户端就原样转发。
    '''

    vendor = "generic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[MultimodalSpec] = None,
        timeout: Optional[float] = None,
        endpoint: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            spec=spec or GENERIC,
            timeout=timeout,
        )
        self.base_url = (base_url or config_module.MULTIMODAL_BASE_URL).rstrip("/")
        self.endpoint = endpoint or config_module.MULTIMODAL_ENDPOINT
        self.transport: HttpTransport = transport or _urllib_transport

    # ---------------- 统一入口 ----------------
    def run(self, request: MultimodalRequest) -> MultimodalResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()

        try:
            payload = self._build_payload(request)
        except ValueError as exc:
            REQUEST_LOGGER.error(
                "multimodal request %s status=invalid error=%s", request_id, exc
            )
            raise MultimodalError(f"多模态请求参数错误：{exc}") from exc

        timeout = self._timeout(request)
        REQUEST_LOGGER.info(
            "multimodal request %s operation=%s modality=%s model=%s "
            "output=%s format=%s inputs=%d timeout=%.1fs params=%s prompt=%s",
            request_id,
            request.operation,
            request.modality,
            payload.get("model") or "-",
            payload["output"]["modality"],
            request.output_format or "-",
            len(request.inputs),
            timeout,
            payload.get("params") or "-",
            _preview(request.prompt),
        )
        if REQUEST_LOGGER.isEnabledFor(logging.DEBUG):
            REQUEST_LOGGER.debug(
                "multimodal payload %s: %s", request_id, _safe_payload(payload)
            )

        try:
            result = self._call(payload, request, request_id, timeout)
            if request.output_path and result.media is not None:
                result.media = self._materialize(
                    result.media, request.output_path, request_id
                )
        except MultimodalError as exc:
            REQUEST_LOGGER.error(
                "multimodal response %s status=error elapsed=%.3fs error=%s",
                request_id,
                time.monotonic() - started,
                exc,
            )
            raise
        except Exception as exc:  # 最后一道兜底：不让网关异常穿透成裸异常
            REQUEST_LOGGER.error(
                "multimodal response %s status=error elapsed=%.3fs error=%s",
                request_id,
                time.monotonic() - started,
                exc,
            )
            raise MultimodalError(f"多模态接口调用失败：{exc}") from exc

        REQUEST_LOGGER.info(
            "multimodal response %s status=ok elapsed=%.3fs text=%s "
            "media=%s task_id=%s",
            request_id,
            time.monotonic() - started,
            _preview(result.text),
            result.media,
            result.task_id or "-",
        )
        return result

    # ---------------- 请求组装 ----------------
    def _url(self) -> str:
        if self.endpoint.startswith(("http://", "https://")):
            return self.endpoint
        path = self.endpoint.strip("/")
        if path and self.base_url:
            return f"{self.base_url}/{path}"
        return self.base_url or ""

    def _timeout(self, request: Optional[MultimodalRequest] = None) -> float:
        if request is not None and request.timeout is not None:
            return float(request.timeout)
        return float(self.timeout or config_module.MULTIMODAL_TIMEOUT)

    def _build_payload(self, request: MultimodalRequest) -> Dict[str, Any]:
        prompt = request.prompt or ""
        if not prompt.strip() and not request.inputs:
            raise ValueError("prompt 与 inputs 不能同时为空")

        model = request.model or (
            self.spec.model_for(request.modality) if self.spec is not None else None
        )
        output: Dict[str, Any] = {
            "modality": request.desired_output_modality,
        }
        if request.output_format:
            output["format"] = request.output_format

        payload: Dict[str, Any] = {
            "operation": request.operation,
            "modality": request.modality,
            "prompt": prompt,
            "inputs": [self._describe_media(media) for media in request.inputs],
            "output": output,
        }
        if model:
            payload["model"] = model
        if request.params:
            payload["params"] = dict(request.params)
        return payload

    @staticmethod
    def _describe_media(media: Media) -> Dict[str, Any]:
        '''把所有输入统一成「type + mime + url/data_base64」的描述符。'''
        mime = media.content_type
        kind = mime.split("/", 1)[0] if mime else "file"
        descriptor: Dict[str, Any] = {"type": kind}
        if mime:
            descriptor["mime"] = mime
        if media.url:
            descriptor["url"] = media.url
            return descriptor
        if media.path:
            descriptor["name"] = os.path.basename(media.path)
        try:
            payload = media.read_bytes()
        except OSError as exc:
            raise ValueError(f"读取输入媒体 {media.path!r} 失败：{exc}") from exc
        descriptor["data_base64"] = base64.b64encode(payload).decode("ascii")
        return descriptor

    # ---------------- HTTP ----------------
    def _call(
        self,
        payload: Dict[str, Any],
        request: MultimodalRequest,
        request_id: str,
        timeout: float,
    ) -> MultimodalResult:
        url = self._url()
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            status, _, raw = self.transport("POST", url, headers, body, timeout)
        except MultimodalError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise MultimodalError(f"多模态接口 {url} 超时（{timeout:.1f}s）") from exc
        except Exception as exc:
            raise MultimodalError(f"多模态接口 {url} 调用失败：{exc}") from exc

        if status >= 400:
            raise MultimodalError(
                f"POST {url} -> HTTP {status}: "
                f"{raw.decode('utf-8', 'replace')[:500]}"
            )
        if not raw:
            data: Mapping[str, Any] = {}
        else:
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MultimodalError(
                    f"多模态接口 {url} 返回的不是 JSON：{exc}"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise MultimodalError(
                    f"多模态接口 {url} 返回的 JSON 不是对象：{type(decoded).__name__}"
                )
            data = decoded

        REQUEST_LOGGER.debug("multimodal raw response %s: %s", request_id, data)
        return self._parse_result(request, data)

    def _download(self, url: str, request_id: str) -> bytes:
        '''下载生成结果；签名 URL 不携带网关密钥。'''
        timeout = self._timeout()
        try:
            status, _, raw = self.transport("GET", url, {}, None, timeout)
        except MultimodalError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise MultimodalError(f"下载 {url} 超时（{timeout:.1f}s）") from exc
        except Exception as exc:
            raise MultimodalError(f"下载 {url} 失败：{exc}") from exc
        if status >= 400:
            raise MultimodalError(
                f"下载 {url} -> HTTP {status}（请求 {request_id}）"
            )
        return raw

    # ---------------- 响应映射 ----------------
    def _parse_result(
        self, request: MultimodalRequest, data: Mapping[str, Any]
    ) -> MultimodalResult:
        media = self._parse_media(data.get("media"))
        text = data.get("text")
        if text is not None and not isinstance(text, str):
            text = str(text)
        if text is None:
            content = data.get("content")
            if isinstance(content, str):
                text = content
        usage = data.get("usage")
        task_id = data.get("task_id")
        if task_id is None and isinstance(data.get("id"), str):
            task_id = data["id"]
        model = data.get("model")
        return MultimodalResult(
            operation=request.operation,
            modality=request.modality,
            model=model if isinstance(model, str) else request.model,
            text=text,
            media=media,
            usage=usage if isinstance(usage, Mapping) else None,
            raw=data,
            task_id=task_id if isinstance(task_id, str) else None,
        )

    @staticmethod
    def _parse_media(value: object) -> Optional[Media]:
        if not isinstance(value, Mapping):
            return None
        mime = value.get("mime")
        mime_text = mime if isinstance(mime, str) and mime else None
        url = value.get("url")
        if isinstance(url, str) and url:
            return Media(url=url, mime=mime_text)

        data_base64 = value.get("data_base64") or value.get("data")
        if isinstance(data_base64, str) and data_base64:
            try:
                payload = base64.b64decode(data_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise MultimodalError(f"网关返回的 media.data_base64 非法：{exc}") from exc
            return Media(data=payload, mime=mime_text)

        path = value.get("path")
        if isinstance(path, str) and path:
            return Media(path=path, mime=mime_text)
        raise MultimodalError(f"网关返回的 media 无法识别：{dict(value)!r}")

    def _materialize(
        self, media: Media, output_path: str, request_id: str
    ) -> Media:
        '''把结果落盘到调用方指定的路径（不做任何格式转换）。'''
        target = os.path.abspath(os.path.expanduser(str(output_path)))
        folder = os.path.dirname(target)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if media.data is not None:
            payload = media.data
        elif media.path:
            if os.path.abspath(media.path) == target:
                return media
            with open(media.path, "rb") as handle:
                payload = handle.read()
        elif media.url:
            payload = self._download(media.url, request_id)
        else:
            raise MultimodalError("生成结果没有可保存的内容")

        with open(target, "wb") as handle:
            handle.write(payload)
        return Media(
            path=target,
            mime=media.mime or mimetypes.guess_type(target)[0],
        )


def _preview(value: Optional[str], limit: int = 80) -> str:
    '''日志里的短预览：空值给 "-"，过长截断。'''
    if value is None:
        return "-"
    text = " ".join(str(value).split())
    if not text:
        return "-"
    return text if len(text) <= limit else f"{text[:limit]}..."


def _safe_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    '''DEBUG 日志用：保留信封结构，但把 base64 换成长度。'''
    safe = dict(payload)
    inputs = payload.get("inputs")
    if isinstance(inputs, list):
        items: list[Dict[str, Any]] = []
        for item in inputs:
            if not isinstance(item, Mapping):
                items.append({"value": str(item)})
                continue
            copy = dict(item)
            data = copy.pop("data_base64", None)
            if isinstance(data, str):
                copy["data_base64"] = f"<{len(data)} chars base64>"
            items.append(copy)
        safe["inputs"] = items
    return safe
