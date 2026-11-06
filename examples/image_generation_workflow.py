# examples/image_generation_workflow.py
'''生图工作流完整样板 — 覆盖三种常见场景。

## 场景 A：纯文生图（ImageGenStep，不走 Agent）
固定流程：按 prompt 生成图 → 下载到本地

## 场景 B：生成 + Agent 看图写注（ImageGenStep → LLMStep）
两步 Workflow：文生图 → Agent 看图生成描述

## 场景 C：Agent 自主调用生图 Tool
把"生图"包成 Tool，Agent 自己判断要不要调

## 运行前

1. 在 `d:\projects\promptkit\.env` 配 `DOUBAO_API_KEY=你的key`
2. 或者在 PowerShell 里 `$env:DOUBAO_API_KEY = "你的key"`
3. 模型名可按需改（下面用的是 Seedream 4.5）

## 跑法

    python examples/image_generation_workflow.py

三种场景依次跑。
'''
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 先加载 .env —— resolve_multimodal_api_key 会读到
from promptkit import config as _cfg
_cfg.load_env_file()

# =============================================================================
# 统一入口：豆包多模态客户端
# =============================================================================
from promptkit.client_sys.multimodal import make_multimodal_client

doubao_mm = make_multimodal_client("doubao")
print(f"✓ 豆包多模态客户端已就绪（vendor={doubao_mm.vendor}）")
print(f"  支持 image:generate = {doubao_mm.supports('image', 'generate')}")
print(f"  支持 image:understand = {doubao_mm.supports('image', 'understand')}")

# =============================================================================
# 场景 A：纯文生图 —— ImageGenStep（不走 Agent）
# =============================================================================

print()
print("=" * 60)
print("场景 A：纯文生图（ImageGenStep）")
print("=" * 60)

from promptkit.core.steps.actions.multimodal import ImageGenStep
from promptkit.core.steps.workflow import Workflow as _WorkflowA

workflow_a = _WorkflowA(
    ImageGenStep(
        prompt_key="image_prompt",
        output_key="generated_image",
        client=doubao_mm,           # 显式用豆包
        size="1024x1024",
    ),
)

ctx_a = workflow_a.run(
    image_prompt="一只橘猫趴在机械键盘上睡觉，柔和晨光，电影级光影，8K",
)

img_a = ctx_a["generated_image"]
print(f"✓ 生成成功")
print(f"  类型：{type(img_a).__name__}")
print(f"  URL：{img_a.source[:80]}..." if len(img_a.source) > 80 else f"  URL：{img_a.source}")
print(f"  source_type：{img_a.source_type}")
print(f"  media_type：{img_a.media_type}")

# =============================================================================
# 场景 B：文生图 → Agent 看图写注 —— 两步 Workflow
# =============================================================================

print()
print("=" * 60)
print("场景 B：生成 + 理解（ImageGenStep → LLMStep）")
print("=" * 60)

from promptkit.agent import Agent
from promptkit.core.content import TextPart
from promptkit.core.steps.actions.llm import LLMStep
from promptkit.core.steps.logics.var import Var
from promptkit.prompt import Prompt

agent = Agent(
    vendor="doubao",
    model="doubao-seed-2-1-pro-260915",  # 支持 vision 的豆包模型
)

workflow_b = _WorkflowA(
    # Step 1：文生图（不走 agent，走多模态 client）
    ImageGenStep(
        prompt_key="image_prompt",
        output_key="cat_image",
        client=doubao_mm,
        size="1024x1024",
    ),
    # Step 2：Agent 看图写注（走 agent.chat.completions with vision）
    LLMStep(
        Prompt(
            role="user",
            content=[
                TextPart(text="用 50 字以内给这张图写一句有趣的注脚："),
                Var("cat_image"),  # ImagePart —— Prompt 自动识别并传给 vision model
            ],
        ),
        output_key="caption",
    ),
)

ctx_b = workflow_b.run(
    agent=agent,                              # 第二步要 agent
    image_prompt="一只橘猫趴在机械键盘上睡觉，柔和晨光",
)

print(f"✓ 图生成：{ctx_b['cat_image'].source[:80]}...")
print(f"✓ 图注：{ctx_b['caption']}")

# =============================================================================
# 场景 C：Agent 自主调用生图 Tool
# =============================================================================

print()
print("=" * 60)
print("场景 C：Agent 自主调用 Tool（不硬编码 Step）")
print("=" * 60)

from promptkit.core.tool.base import Tool
from promptkit.core.tool.registry import ToolRegistry


def _do_generate_image(prompt: str, size: str = "1024x1024") -> str:
    '''Tool 的真实实现 — 内部调豆包多模态 client。'''
    result = doubao_mm.generate("image", prompt, size=size)
    if result.media and result.media.url:
        return f"成功！图片 URL：{result.media.url}"
    return f"生成失败：{result.text or '未知错误'}"


image_tool = Tool(
    name="generate_image",
    description="""根据文本描述生成图片（文生图）。参数：
  - prompt: 图片内容的详细描述（必填）
  - size: 尺寸，可选 "512x512" / "1024x1024" / "2K"（默认 "1024x1024"）""",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "图片内容描述，越详细越好（风格、主体、光影、构图）",
            },
            "size": {
                "type": "string",
                "enum": ["512x512", "1024x1024", "2K"],
                "default": "1024x1024",
            },
        },
        "required": ["prompt"],
    },
    func=_do_generate_image,
)

registry = ToolRegistry([image_tool])

# 测试 1：Agent 判断"生成图" → 调 Tool
reply_c1 = agent.ask(
    "画一张赛博朋克风格的橘猫，霓虹灯光，雨夜街道背景",
    tools=registry.all(),
)
print(f"✓ Agent 自主调 tool：")
print(f"  {reply_c1[:120]}...")

# 测试 2：Agent 判断"不需要生成" → 不调 Tool
reply_c2 = agent.ask(
    "文生图一般用什么算法？Stable Diffusion 和 DALL-E 有什么区别？",
    tools=registry.all(),
)
print()
print(f"✓ Agent 判断不调 tool（纯问答）：")
print(f"  {reply_c2[:120]}...")

# =============================================================================
# 附加：保存图片到本地
# =============================================================================

print()
print("=" * 60)
print("附加：保存图片到本地")
print("=" * 60)

import os as _os
from urllib.request import urlopen

save_dir = _os.path.join(_os.path.expanduser("~"), "Desktop")
save_path = _os.path.join(save_dir, "generated_cat.png")

img = ctx_a["generated_image"]
if img.source_type == "url":
    with urlopen(img.source) as resp:  # noqa: S310 - 用户显式 URL
        with open(save_path, "wb") as f:
            f.write(resp.read())
    print(f"✓ 图片已保存：{save_path}")
else:
    print(f"图片来源是 {img.source_type}，需要自行处理")

print()
print("🏁 三种场景全部跑完！")
print()
print("总结：")
print("  • 固定流程 → ImageGenStep（不走 Agent）")
print("  • 生成 + 理解 → ImageGenStep + LLMStep（需要 Agent）")
print("  • 灵活/Agent 自主 → Tool 包函数（ToolRegistry + agent.ask(tools=...)）")
