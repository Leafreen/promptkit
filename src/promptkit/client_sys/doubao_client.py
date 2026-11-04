# client_sys/doubao_client.py
'''豆包（火山方舟 Ark）客户端。

Ark 的对话接口是 OpenAI 兼容的，文本和图片理解直接复用 `OpenAIClient`；
差别在三处：

1. 密钥环境变量是 `DOUBAO_API_KEY`（回退 `ARK_API_KEY`）
2. 图片之外还支持视频输入（`video_url` 块）
3. 视频生成（Seedance）走统一多模态网关：`generate_video()`

在 `client_sys/registry.py` 注册后，`Agent(vendor="doubao")` 即可使用。
'''
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import os

from promptkit import config as config_module
from promptkit.client_sys.client import (
    ClientError,
    UnsupportedModality,
    part_data_url,
)
from promptkit.client_sys.openai_client import OpenAIClient

if TYPE_CHECKING:  # 类型标注用；运行时由方法内惰性 import 拿到真身
    from promptkit.client_sys.multimodal import Media

__all__ = ["DoubaoClient"]


class DoubaoClient(OpenAIClient):
    '''豆包（火山方舟）客户端：OpenAI 兼容 + 视频能力。

        client = DoubaoClient(api_key="...")   # 或环境变量 DOUBAO_API_KEY
        client.create(model="doubao-seed-1-6-250615", messages=[...])
        media = client.generate_video("一只猫在草地上跑", duration=5)
    '''

    vendor = "doubao"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        '''密钥缺省按 `DOUBAO_API_KEY` -> `ARK_API_KEY` 环境变量读，端点默认火山方舟。

        走 `make_client("doubao")` / `Agent(vendor="doubao")` 时注册表已经
        把密钥和端点解决好了，这里的兜底只服务「直接 `DoubaoClient()`」的场景。
        '''
        if api_key is None:
            for env_name in (config_module.DOUBAO_API_KEY_ENV, "ARK_API_KEY"):
                api_key = os.environ.get(env_name)
                if api_key:
                    break
        if not api_key:
            raise ClientError(
                "缺少豆包 API Key：传 api_key=，或设置环境变量 "
                f"{config_module.DOUBAO_API_KEY_ENV} / ARK_API_KEY"
            )
        super().__init__(
            api_key=api_key,
            base_url=base_url or config_module.DOUBAO_BASE_URL,
            timeout=timeout,
            **kwargs,
        )

    def supports(self, modality: str) -> bool:
        '''豆包支持文本 / 图片 / 视频（视频输入理解 + Seedance 生成）。'''
        return modality in ("text", "image", "video")

    def _to_vendor_part(self, part: Mapping[str, Any]) -> Dict[str, Any]:
        '''标准部件 -> Ark content part：视频走 `video_url`，图片沿用 OpenAI 形状。

        没声明支持的模态（audio / file）在这里抛 `UnsupportedModality`，
        不偷偷套 OpenAI 的形状。
        '''
        kind = str(part.get("type"))
        if kind == "video":
            return {
                "type": "video_url",
                "video_url": {"url": part_data_url(part)},
            }
        if not self.supports(kind):
            raise UnsupportedModality(self.vendor, kind)
        return super()._to_vendor_part(part)

    def generate_video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        save_to: Optional[str] = None,
        **params: Any,
    ) -> "Media":
        '''文生视频（Seedance）。

        走统一多模态网关（`client_sys/multimodal.py`），端点从
        `MULTIMODAL_BASE_URL` 读；没配置会给出清晰报错而不是瞎猜地址。

        # TODO: 确认 Seedance 直连 API 的请求格式（任务创建 / 轮询 / 分辨率
        # 参数），确认后在这里补一条不走网关的直连路径。
        '''
        from promptkit.client_sys.multimodal import make_multimodal_client

        client = make_multimodal_client("doubao", api_key=self.api_key)
        if duration is not None:
            params["duration"] = duration
        result = client.generate(
            "video",
            prompt,
            model=model or config_module.DOUBAO_VIDEO_MODEL,
            save_to=save_to,
            **params,
        )
        if result.media is None:
            raise ClientError(f"视频生成没有返回媒体：{result.raw!r}")
        return result.media
# 职责：豆包（火山方舟）客户端 —— OpenAI 兼容的对话/图片理解 + 视频输入/生成
