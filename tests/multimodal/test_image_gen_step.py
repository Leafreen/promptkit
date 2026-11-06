'''多模态生成 / 理解步骤：ImageGenStep / VideoGenStep / STTStep / TTSStep。'''
import pytest

from promptkit.client_sys.client import UnsupportedModality
from promptkit.client_sys.multimodal import Media
from promptkit.core.content import AudioPart, ImagePart, TextPart, VideoPart
from promptkit.core.context import Context
from promptkit.core.steps.actions.multimodal import (
    ImageGenStep,
    STTStep,
    TTSStep,
    VideoGenStep,
)
from promptkit.core.steps.logics.var import Var
from promptkit.prompt import Prompt

from .conftest import FakeMultimodalClient


def test_image_gen_step_writes_image_part():
    client = FakeMultimodalClient(
        {("image", "generate")},
        generate_media=Media.from_url("https://example.com/cat.png"),
    )
    context = Context(data={"prompt": "一只猫"})
    ImageGenStep("prompt", "image", client=client).execute(context)

    image = context["image"]
    assert isinstance(image, ImagePart)
    assert image.source == "https://example.com/cat.png"
    assert image.source_type == "url"
    assert context.trace == ["image_gen"]


def test_image_gen_step_passes_size():
    client = FakeMultimodalClient({("image", "generate")})
    context = Context(data={"prompt": "一只猫"})
    ImageGenStep("prompt", "image", client=client, size="1024x1024").execute(context)

    kind, modality, prompt, params = client.requests[0]
    assert (kind, modality, prompt) == ("generate", "image", "一只猫")
    assert params["size"] == "1024x1024"


def test_image_gen_step_requires_capability():
    client = FakeMultimodalClient()  # 什么能力都没有
    context = Context(data={"prompt": "一只猫"})
    with pytest.raises(UnsupportedModality, match="不支持"):
        ImageGenStep("prompt", "image", client=client).execute(context)


def test_image_gen_step_accepts_prompt_with_image_input():
    '''以图生图：Prompt 的文本当提示词，图片部件当参考输入。'''
    client = FakeMultimodalClient({("image", "generate")})
    context = Context(data={"ref": "https://example.com/ref.png"})
    step = ImageGenStep(
        Prompt(TextPart(text="把这只猫变成水彩画"), ImagePart(Var("ref"))),
        "image",
        client=client,
    )
    step.execute(context)

    _, modality, prompt, params = client.requests[0]
    assert (modality, prompt) == ("image", "把这只猫变成水彩画")
    assert isinstance(params["inputs"], tuple)
    assert params["inputs"][0].url == "https://example.com/ref.png"


def test_image_gen_step_accepts_prompt_in_context():
    '''prompt_key 指向 Context 里的 Prompt 也可以。'''
    client = FakeMultimodalClient({("image", "generate")})
    context = Context(
        data={
            "ref": "https://example.com/ref.png",
            "spec": Prompt(ImagePart(Var("ref")), TextPart(text="改成油画风格")),
        }
    )
    ImageGenStep("spec", "out", client=client).execute(context)

    _, _, prompt, params = client.requests[0]
    assert prompt == "改成油画风格"
    assert params["inputs"][0].url == "https://example.com/ref.png"


def test_video_gen_step_writes_video_part():
    client = FakeMultimodalClient(
        {("video", "generate")},
        generate_media=Media.from_url("https://example.com/cat.mp4"),
    )
    context = Context(data={"prompt": "猫在草地上跑"})
    VideoGenStep("prompt", "video", client=client, duration=5).execute(context)

    video = context["video"]
    assert isinstance(video, VideoPart)
    assert video.source == "https://example.com/cat.mp4"
    _, _, _, params = client.requests[0]
    assert params["duration"] == 5


def test_video_gen_step_accepts_prompt_with_image_input():
    '''以图生视频：图片部件作为首帧 / 参考图。'''
    client = FakeMultimodalClient(
        {("video", "generate")},
        generate_media=Media.from_url("https://example.com/cat.mp4"),
    )
    context = Context(data={"ref": "https://example.com/ref.png"})
    VideoGenStep(
        Prompt(TextPart(text="让猫动起来"), ImagePart(Var("ref"))),
        "video",
        client=client,
        duration=5,
    ).execute(context)

    _, modality, prompt, params = client.requests[0]
    assert (modality, prompt) == ("video", "让猫动起来")
    assert params["duration"] == 5
    assert params["inputs"][0].url == "https://example.com/ref.png"


def test_stt_step_writes_text():
    client = FakeMultimodalClient(
        {("audio", "understand")}, understand_text="你好，世界"
    )
    context = Context(
        data={
            "audio": AudioPart(
                source="aGk=", source_type="base64", media_type="audio/mpeg"
            )
        }
    )
    STTStep("audio", "text", client=client).execute(context)

    assert context["text"] == "你好，世界"
    kind, modality, source, _, _ = client.requests[0]
    assert (kind, modality) == ("understand", "audio")
    assert isinstance(source, Media)
    assert source.data == b"hi"  # base64 被还原成字节


def test_tts_step_writes_audio_part():
    client = FakeMultimodalClient(
        {("audio", "generate")},
        generate_media=Media.from_url("https://example.com/voice.mp3"),
    )
    context = Context(data={"text": "你好"})
    TTSStep("text", "voice", client=client, voice="alloy").execute(context)

    audio = context["voice"]
    assert isinstance(audio, AudioPart)
    assert audio.source == "https://example.com/voice.mp3"
    _, _, _, params = client.requests[0]
    assert params["voice"] == "alloy"
    assert params["output_modality"] == "audio"
