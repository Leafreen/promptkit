# client_sys/doubao_multimodal.py
'''豆包方舟（Ark）多模态实现：文本 / 图片 / 视频。

三种操作各走一条官方接口，但对外都收成 `MultimodalRequest`：

    文本生成 / 理解     POST /chat/completions
    图片理解           POST /chat/completions（content 里带 image_url）
    视频理解           POST /chat/completions（content 里带 video_url）
    图片生成           POST /images/generations（Seedream）
    视频生成           POST /contents/generations/tasks（Seedance）
                        GET  /contents/generations/tasks/{id}（轮询）

默认模型来自 `promptkit/config.py`；`request.model` 可以临时覆盖。
HTTP 走标准库 urllib，不额外引入依赖。`transport=` 可以注入一个假传输层，
测试和离线例程因此不需要网络。
'''
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from promptkit import config as config_module
from promptkit.client_sys.multimodal import (
    DOUBAO,
    BaseMultimodalClient,
    Media,
    Modality,
    MultimodalError,
    MultimodalRequest,
    MultimodalResult,
    MultimodalSpec,
    Operation,
)

__all__ = ["DoubaoMultimodalClient", "HttpTransport"]

#: (method, url, headers, body, timeout) -> (status, headers, body)
HttpTransport = Callable[
    [str, str, Mapping[str, str], Optional[bytes], float],
    Tuple[int, Mapping[str, str], bytes],
]

#: 视频任务轮询的终态
_DONE_STATUS = frozenset({"succeeded", "failed", "cancelled", "expired"})


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
        raise MultimodalError(f"{method} {url} 连接失败：{exc.reason}") from None


class DoubaoMultimodalClient(BaseMultimodalClient):
    '''豆包方舟多模态客户端。

        client = DoubaoMultimodalClient(api_key="...")       # 默认端点
        client.generate("image", "一只猫在月球上", save_to="cat.png")
        client.generate("video", "镜头缓慢推进", save_to="demo.mp4")
        client.understand("video", "demo.mp4", prompt="描述这段视频")
    '''

    vendor = "doubao"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[MultimodalSpec] = None,
        timeout: Optional[float] = None,
        transport: Optional[HttpTransport] = None,
        poll_interval: float = 2.0,
        poll_timeout: float = 180.0,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            spec=spec or DOUBAO,
            timeout=timeout,
        )
        self.base_url = (base_url or config_module.DOUBAO_BASE_URL).rstrip("/")
        self.transport: HttpTransport = transport or _urllib_transport
        self.poll_interval = max(0.0, float(poll_interval))
        self.poll_timeout = float(poll_timeout)

    # ---------------- 统一入口 ----------------
    def run(self, request: MultimodalRequest) -> MultimodalResult:
        self.require(request.modality, request.operation)
        if request.operation == Operation.UNDERSTAND:
            return self._understand(request)
        if request.modality == Modality.TEXT:
            return self._text(request)
        if request.modality == Modality.IMAGE:
            return self._image(request)
        if request.modality == Modality.VIDEO:
            return self._video(request)
        raise MultimodalError(f"豆包实现暂不支持模态 {request.modality!r}")

    def _model(self, request: MultimodalRequest) -> str:
        model = request.model or (
            self.spec.model_for(request.modality) if self.spec is not None else None
        )
        if not model:
            model = {
                Modality.TEXT: config_module.DOUBAO_TEXT_MODEL,
                Modality.IMAGE: config_module.DOUBAO_IMAGE_MODEL,
                Modality.VIDEO: config_module.DOUBAO_VIDEO_MODEL,
            }.get(request.modality)
        if not model:
            raise MultimodalError(
                f"模态 {request.modality!r} 没有默认模型，请显式传 model="
            )
        return model

    # ---------------- HTTP ----------------
    def _timeout(self, request: Optional[MultimodalRequest] = None) -> float:
        if request is not None and request.timeout:
            return float(request.timeout)
        return float(self.timeout or 60.0)

    def _json(
        self,
        method: str,
        path: str,
        request: Optional[MultimodalRequest] = None,
        body: Optional[Dict[str, Any]] = None,
        *,
        auth: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = None if body is None else json.dumps(body).encode("utf-8")

        status, _, raw = self.transport(
            method, url, headers, payload, self._timeout(request)
        )
        if status >= 400:
            raise MultimodalError(
                f"{method} {url} -> HTTP {status}: "
                f"{raw.decode('utf-8', 'replace')[:500]}"
            )
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MultimodalError(f"{method} {url} 返回的不是 JSON：{exc}") from None

    def _download(self, url: str, request: Optional[MultimodalRequest] = None) -> bytes:
        '''下载生成结果。签名 URL 不携带方舟密钥。'''
        status, _, raw = self.transport(
            "GET", url, {}, None, self._timeout(request)
        )
        if status >= 400:
            raise MultimodalError(f"下载 {url} -> HTTP {status}")
        return raw

    def _materialize(
        self, media: Media, output_path: Optional[str], request: MultimodalRequest
    ) -> Media:
        '''给了 output_path 就把结果落盘，返回本地 Media。'''
        if not output_path:
            return media
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
            payload = self._download(media.url, request)
        else:
            raise MultimodalError("生成结果没有可保存的内容")

        with open(target, "wb") as handle:
            handle.write(payload)
        return Media(path=target, mime=media.content_type)

    def _media_url(self, media: Media) -> str:
        '''远程 URL 原样传；本地 / 内存转 data URL。'''
        return media.url or media.to_data_url()

    # ---------------- 文本 / 理解 ----------------
    def _text(self, request: MultimodalRequest) -> MultimodalResult:
        body: Dict[str, Any] = {
            "model": self._model(request),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt or ""},
                    ],
                }
            ],
        }
        body.update(request.params)
        data = self._json("POST", "/chat/completions", request, body)
        text = _extract_text(data)
        return MultimodalResult(
            operation=request.operation,
            modality=request.modality,
            model=str(body.get("model")),
            text=text,
            usage=data.get("usage"),
            raw=data,
        )

    def _understand(self, request: MultimodalRequest) -> MultimodalResult:
        parts: List[Dict[str, Any]] = []
        if request.prompt:
            parts.append({"type": "text", "text": request.prompt})
        for media in request.inputs:
            if request.modality == Modality.IMAGE:
                parts.append(
                    {"type": "image_url", "image_url": {"url": self._media_url(media)}}
                )
            elif request.modality == Modality.VIDEO:
                parts.append(
                    {"type": "video_url", "video_url": {"url": self._media_url(media)}}
                )
            elif request.modality == Modality.TEXT:
                parts.append({"type": "text", "text": self._media_url(media)})
            else:
                parts.append(
                    {"type": "text", "text": f"{request.modality}: {media}"}
                )
        if not parts:
            raise MultimodalError("理解请求需要 prompt 或至少一个输入媒体")

        body: Dict[str, Any] = {
            "model": self._model(request),
            "messages": [{"role": "user", "content": parts}],
        }
        body.update(request.params)
        data = self._json("POST", "/chat/completions", request, body)
        text = _extract_text(data)
        return MultimodalResult(
            operation=request.operation,
            modality=request.modality,
            model=str(body.get("model")),
            text=text,
            usage=data.get("usage"),
            raw=data,
        )

    # ---------------- 图片生成 ----------------
    def _image(self, request: MultimodalRequest) -> MultimodalResult:
        body: Dict[str, Any] = {
            "model": self._model(request),
            "prompt": request.prompt or "",
            "response_format": "url",
        }
        if request.inputs:
            body["image"] = [self._media_url(item) for item in request.inputs]
        body.update(request.params)

        data = self._json("POST", "/images/generations", request, body)
        items = data.get("data") or []
        if not items:
            raise MultimodalError(f"图片生成没有返回 data：{data!r}")
        first = items[0]

        if first.get("b64_json"):
            payload = base64.b64decode(first["b64_json"])
            media = Media(data=payload, mime="image/png")
        elif first.get("url"):
            media = Media(url=str(first["url"]), mime="image/png")
        else:
            raise MultimodalError(f"图片生成结果既没有 url 也没有 b64_json：{first!r}")

        media = self._materialize(media, request.output_path, request)
        return MultimodalResult(
            operation=request.operation,
            modality=request.modality,
            model=str(body.get("model")),
            media=media,
            usage=data.get("usage"),
            raw=data,
        )

    # ---------------- 视频生成 ----------------
    def _video(self, request: MultimodalRequest) -> MultimodalResult:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": request.prompt or ""}
        ]
        for media in request.inputs:
            content.append(
                {"type": "image_url", "image_url": {"url": self._media_url(media)}}
            )

        body: Dict[str, Any] = {"model": self._model(request), "content": content}
        body.update(request.params)

        created = self._json(
            "POST", "/contents/generations/tasks", request, body
        )
        task_id = created.get("id") or created.get("task_id")
        if not task_id:
            raise MultimodalError(f"视频任务创建失败，没有返回 id：{created!r}")

        final = self._poll_video(str(task_id), request)
        video_url = _extract_video_url(final)
        if not video_url:
            raise MultimodalError(f"视频任务 {task_id} 成功但没有 video_url：{final!r}")

        media = Media(url=video_url, mime="video/mp4")
        media = self._materialize(media, request.output_path, request)
        return MultimodalResult(
            operation=request.operation,
            modality=request.modality,
            model=str(body.get("model")),
            media=media,
            usage=final.get("usage"),
            raw=final,
            task_id=str(task_id),
        )

    def _poll_video(
        self, task_id: str, request: MultimodalRequest
    ) -> Dict[str, Any]:
        '''轮询 Seedance 任务，直到成功 / 失败 / 超时。'''
        started = time.monotonic()
        while True:
            data = self._json(
                "GET", f"/contents/generations/tasks/{task_id}", request
            )
            status = str(data.get("status") or "").lower()
            if status in _DONE_STATUS:
                if status == "succeeded":
                    return data
                raise MultimodalError(
                    f"视频任务 {task_id} 结束于 {status}："
                    f"{data.get('error') or data}"
                )
            if time.monotonic() - started >= self.poll_timeout:
                raise MultimodalError(
                    f"视频任务 {task_id} 轮询超时（{self.poll_timeout}s），"
                    f"最后状态：{status or 'unknown'}"
                )
            if self.poll_interval:
                time.sleep(self.poll_interval)


def _extract_text(data: Mapping[str, Any]) -> str:
    '''从 Chat Completions 响应里取出文本；content 是列表时拼起来。'''
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping)
        )
    return "" if content is None else str(content)


def _extract_video_url(data: Mapping[str, Any]) -> Optional[str]:
    '''Seedance 查询结果的 video_url 可能在 content 里，也可能在顶层。'''
    content = data.get("content")
    if isinstance(content, Mapping):
        url = content.get("video_url")
        if url:
            return str(url)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, Mapping) and part.get("video_url"):
                return str(part["video_url"])
    url = data.get("video_url")
    return str(url) if url else None
