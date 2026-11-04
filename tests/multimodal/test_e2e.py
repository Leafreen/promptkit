'''端到端：图片理解 -> 生成报告（FakeClient，不调真实 API）。'''
import pytest

from promptkit.client_sys.client import UnsupportedModality
from promptkit.core.content import TextPart
from promptkit.core.display_sys.render import Verbosity
from promptkit.core.steps.actions.llm import LLMStep
from promptkit.core.steps.logics.var import Var
from promptkit.core.steps.workflow import Workflow
from promptkit.prompt import Prompt

from .conftest import client_of, make_agent


def test_image_understanding_to_report():
    agent = make_agent(
        replies=["一只橘猫趴在键盘上", "# 图片报告\n\n图中是一只橘猫……"],
        modalities=("text", "image"),
    )
    workflow = Workflow(
        LLMStep(
            Prompt(
                role="user",
                content=[TextPart(text="用一句话描述这张图片。"), Var("image")],
            ),
            output_key="description",
        ),
        LLMStep(
            Prompt(role="user", content="根据这段描述写一份简短报告：{{description}}"),
            output_key="report",
        ),
    )

    context = workflow.run(agent=agent, verbose=Verbosity.SILENT, image="cat.png")

    assert context["description"] == "一只橘猫趴在键盘上"
    assert context["report"].startswith("# 图片报告")

    # 第一轮：Var("image") 按扩展名推断成 ImagePart，真的发给了客户端
    # （messages[0] 是 system 提示，用户消息在最后一条）
    first_content = client_of(agent).calls[0]["messages"][-1]["content"]
    assert first_content[0] == {"type": "text", "text": "用一句话描述这张图片。"}
    assert first_content[1]["type"] == "image"
    assert first_content[1]["source"] == "cat.png"

    # 第二轮：纯文本，带上了第一轮的描述
    second_content = client_of(agent).calls[1]["messages"][-1]["content"]
    assert "一只橘猫趴在键盘上" in second_content


def test_unsupported_image_fails_fast():
    '''客户端不支持 image 时，第一步就抛 UnsupportedModality，不发请求。'''
    agent = make_agent(modalities=("text",))
    workflow = Workflow(
        LLMStep(
            Prompt(role="user", content=[TextPart(text="看图"), Var("image")]),
            output_key="description",
        ),
    )
    with pytest.raises(UnsupportedModality, match="不支持 image"):
        workflow.run(agent=agent, verbose=Verbosity.SILENT, image="cat.png")
    assert client_of(agent).calls == []  # 协商失败在发请求前
