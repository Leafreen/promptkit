'''LLMStep 能力协商：客户端不支持的模态抛 UnsupportedModality，不静默降级。'''
from types import SimpleNamespace

import pytest

from promptkit.client_sys.client import UnsupportedModality
from promptkit.core.content import ImagePart, TextPart
from promptkit.core.context import Context
from promptkit.core.steps.actions.llm import LLMStep
from promptkit.prompt import Prompt

from .conftest import client_of, make_agent


def _multimodal_prompt() -> Prompt:
    return Prompt(
        role="user",
        content=[
            TextPart(text="描述这张图"),
            ImagePart(
                source="https://example.com/cat.png",
                source_type="url",
                media_type="image/png",
            ),
        ],
    )


def test_unsupported_modality_raises():
    agent = make_agent(modalities=("text",))
    step = LLMStep(_multimodal_prompt(), output_key="out", agent=agent)
    with pytest.raises(UnsupportedModality, match="不支持 image"):
        step.execute(Context())


def test_supported_modality_passes_and_reaches_client():
    agent = make_agent(replies=["一只猫坐在窗台上"], modalities=("text", "image"))
    step = LLMStep(_multimodal_prompt(), output_key="out", agent=agent)
    context = Context()
    step.execute(context)
    assert context["out"] == "一只猫坐在窗台上"
    # 图片部件真的发给了客户端，没被拍平成文本（messages[0] 是 system，用户消息在最后）
    content = client_of(agent).calls[0]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image" for part in content)


def test_text_prompt_skips_check():
    '''纯文本 prompt 不做任何模态检查，老行为不变。'''
    agent = make_agent(modalities=())  # 连 text 都不声明
    step = LLMStep(Prompt(role="user", content="你好"), output_key="out", agent=agent)
    context = Context()
    step.execute(context)
    assert context["out"] == "好的"


def test_legacy_client_without_supports_not_blocked():
    '''老式假客户端（没有 supports 方法）不拦，保持兼容。'''
    rendered = {
        "role": "user",
        "content": [
            {"type": "image", "source": "x", "source_type": "url", "media_type": None}
        ],
    }
    legacy_agent = SimpleNamespace(client=SimpleNamespace())
    LLMStep._check_modalities(legacy_agent, rendered)  # 不抛即通过


def test_str_content_skips_check():
    agent = make_agent(modalities=())
    LLMStep._check_modalities(agent, {"role": "user", "content": "纯文本"})
