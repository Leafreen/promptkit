'''Prompt 多模态渲染：部件列表、Var 解析、模板替换。'''
import pytest

from promptkit.core.content import ImagePart, TextPart, part_from_any
from promptkit.core.steps.logics.var import Var
from promptkit.prompt import Prompt, render_messages


def test_str_content_behavior_unchanged():
    prompt = Prompt(role="user", content="你好，{{name}}")
    rendered = prompt.render(name="世界")
    assert rendered == {"role": "user", "content": "你好，世界"}


def test_list_content_renders_parts():
    prompt = Prompt(
        role="user",
        content=[
            TextPart(text="用一句话描述这张{{kind}}图"),
            ImagePart(
                source="https://example.com/cat.png",
                source_type="url",
                media_type="image/png",
            ),
        ],
    )
    rendered = prompt.render(kind="动物")
    content = rendered["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "用一句话描述这张动物图"}
    assert content[1] == {
        "type": "image",
        "source": "https://example.com/cat.png",
        "source_type": "url",
        "media_type": "image/png",
    }


def test_var_resolves_string_by_extension():
    prompt = Prompt(role="user", content=["看这张图：", Var("img")])
    rendered = prompt.render(img="cat.png")
    content = rendered["content"]
    assert content[0] == {"type": "text", "text": "看这张图："}
    assert content[1]["type"] == "image"
    assert content[1]["source"] == "cat.png"
    assert content[1]["source_type"] == "file"


def test_var_resolves_part_instance_as_is():
    image = ImagePart(source="aGk=", source_type="base64", media_type="image/png")
    prompt = Prompt(role="user", content=[Var("img")])
    rendered = prompt.render(img=image)
    assert rendered["content"] == [image.to_dict()]


def test_var_resolves_string_without_extension_as_text():
    prompt = Prompt(role="user", content=[Var("note")])
    rendered = prompt.render(note="随便一句话")
    assert rendered["content"] == [{"type": "text", "text": "随便一句话"}]


def test_var_missing_raises_key_error():
    prompt = Prompt(role="user", content=[Var("img")])
    with pytest.raises(KeyError):
        prompt.render()


def test_render_messages_multimodal():
    prompt = Prompt(
        role="user",
        content=[TextPart(text="描述它"), ImagePart(
            source="https://example.com/a.png",
            source_type="url",
            media_type="image/png",
        )],
    )
    messages = render_messages([prompt])
    assert isinstance(messages[0]["content"], list)


def test_part_from_any_string_pdf_becomes_file_part():
    part = part_from_any("report.pdf")
    assert part.type == "file"
    assert part.source_type == "file"
