# examples/multimodal.py
'''多模态端到端例程：文本 -> 图片生成 -> 图片理解 -> 视频生成 -> 视频理解。

运行：

    python examples/multimodal.py

有 `ARK_API_KEY`（豆包方舟）时会走真实 Seedance / Seedream / Seed 模型；
没有密钥时自动使用同一套客户端 + 内置离线传输层，照样把完整工作流跑通，
方便在本地验证类型系统、DSL 步骤和工作流集成。

环境变量：

    ARK_API_KEY         豆包方舟密钥（不填则离线演示）
    DEEPSEEK_API_KEY    仅用于构造 Agent；本流程不调用对话模型
'''
import base64
import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

from promptkit.agent import Agent
from promptkit.client_sys import DoubaoMultimodalClient
from promptkit.core import (
    Bool,
    Echo,
    File,
    ImageGen,
    ImageUnderstand,
    Set,
    TextGen,
    Time,
    Var,
    VideoGen,
    VideoUnderstand,
    Workflow,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, ".tmp", "multimodal-demo")
IMAGE_PATH = os.path.join(OUTPUT_DIR, "cat.png")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "demo.mp4")

#: 1x1 像素 PNG：离线模式用来生成一个真实可打开的小图
_OFFLINE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def offline_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
    timeout: float,
) -> Tuple[int, Mapping[str, str], bytes]:
    '''模拟豆包方舟的三类端点，不访问网络。'''
    payload: Dict[str, Any] = json.loads(body) if body else {}

    if url.endswith("/chat/completions"):
        parts = payload["messages"][0]["content"]
        has_image = any(part.get("type") == "image_url" for part in parts)
        has_video = any(part.get("type") == "video_url" for part in parts)
        if has_video:
            text = "（离线演示）视频理解完成：画面从静帧开始缓慢推进，主体保持稳定。"
        elif has_image:
            text = "（离线演示）图片理解完成：画面主体清晰，构图居中，背景干净。"
        else:
            text = "（离线演示）文本生成完成：多模态工作流把三种模态串在同一条流水线上。"
        return 200, {}, json.dumps(
            {"choices": [{"message": {"content": text}}]}, ensure_ascii=False
        ).encode("utf-8")

    if url.endswith("/images/generations"):
        return 200, {}, json.dumps(
            {"data": [{"b64_json": base64.b64encode(_OFFLINE_PNG).decode("ascii")}]}
        ).encode("utf-8")

    if method == "POST" and url.endswith("/contents/generations/tasks"):
        return 200, {}, b'{"id": "cgt-offline-demo"}'

    if method == "GET" and "/contents/generations/tasks/" in url:
        return 200, {}, json.dumps(
            {
                "status": "succeeded",
                "content": {"video_url": "https://offline.local/demo.mp4"},
            }
        ).encode("utf-8")

    if method == "GET" and url.endswith("/demo.mp4"):
        return 200, {}, b"OFFLINE-DEMO-MP4-BYTES"

    raise AssertionError(f"离线传输层没有实现：{method} {url}")


def build_multimodal_client() -> Tuple[DoubaoMultimodalClient, str]:
    key = (
        os.environ.get("ARK_API_KEY")
        or os.environ.get("DOUBAO_API_KEY")
        or os.environ.get("VOLCENGINE_API_KEY")
    )
    if key:
        return DoubaoMultimodalClient(api_key=key), "online"
    return (
        DoubaoMultimodalClient(
            api_key="offline-demo",
            transport=offline_transport,
            poll_interval=0.0,
            poll_timeout=5.0,
        ),
        "offline",
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 本流程只走多模态入口；构造 Agent 时给一个占位 key，避免没有
    # DEEPSEEK_API_KEY 的机器在演示开始前就失败。
    agent = Agent(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or "offline-demo",
        register_as_default=False,
    )
    client, mode = build_multimodal_client()
    agent.multimodal_client = client

    wf = Workflow(
        # ---- 变量声明：覆盖新增的 File / Bool / Time 类型 ----
        File("image"),
        File("video"),
        Bool("ok"),
        Time("started"),
        Set("started", Time.now()),

        # ---- 文本模态 ----
        TextGen(
            "用一句话说明多模态工作流的价值",
            output_key="intro",
        ),

        # ---- 图片生成 + 图片理解 ----
        ImageGen(
            "一只在月球上弹吉他的猫，电影感灯光，超清细节",
            output_key="image",
            save_to=IMAGE_PATH,
            size="1024x1024",
        ),
        ImageUnderstand(
            Var("image"),
            prompt="描述画面主体、构图和光线",
            output_key="image_description",
        ),

        # ---- 视频生成 + 视频理解 ----
        VideoGen(
            "镜头从猫的正面缓慢推进，吉他弦轻微震动",
            inputs=[Var("image")],
            output_key="video",
            save_to=VIDEO_PATH,
            duration=5,
            ratio="16:9",
            watermark=False,
        ),
        VideoUnderstand(
            Var("video"),
            prompt="描述镜头运动、主体动作和整体节奏",
            output_key="video_description",
        ),

        # ---- 类型系统收尾：File 校验 + Time 计时 ----
        Set(
            "ok",
            Bool.and_(File.exists(Var("image")), File.exists(Var("video"))),
        ),
        Set("elapsed", Time.diff(Time.now(), Var("started"))),
        Echo(
            "文本：{{intro}}\n"
            "图片描述：{{image_description}}\n"
            "视频描述：{{video_description}}\n"
            "文件检查：{{ok}}，耗时 {{elapsed}} 秒"
        ),
    )

    context = wf.run(agent=agent, verbose=False)

    print(f"模式：{mode}（默认豆包方舟 / Seedance）")
    print(f"图片：{context['image']} ({os.path.getsize(IMAGE_PATH)} bytes)")
    print(f"视频：{context['video']} ({os.path.getsize(VIDEO_PATH)} bytes)")
    print(f"文本：{context['intro']}")
    print(f"图片描述：{context['image_description']}")
    print(f"视频描述：{context['video_description']}")
    print(f"File/Bool/Time 校验：{context['ok']}，耗时 {context['elapsed']:.3f} 秒")


if __name__ == "__main__":
    main()
