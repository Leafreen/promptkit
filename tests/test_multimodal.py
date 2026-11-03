'''多模态模块的离线测试：变量类型、默认豆包客户端、DSL 步骤。'''
import base64
import json
import os
import tempfile
import unittest
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from promptkit.agent import Agent
from promptkit.client_sys import (
    BaseMultimodalClient,
    DoubaoMultimodalClient,
    Media,
    Modality,
    MultimodalRequest,
    MultimodalResult,
    MultimodalSpec,
    Operation,
    get_multimodal_spec,
    make_multimodal_client,
    register_multimodal_provider,
)
from promptkit.client_sys.doubao_multimodal import HttpTransport
from promptkit.core import (
    Bool,
    Echo,
    File,
    FileRef,
    ImageGen,
    ImageUnderstand,
    PyObj,
    Set,
    TextGen,
    Time,
    Var,
    VideoGen,
    VideoUnderstand,
    Workflow,
)
from promptkit.core.errors import VarTypeError

#: 1x1 透明 PNG（有效图片，方便断言确实写进了文件）
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class FakeArkTransport:
    '''模拟方舟端点，记录每次请求，供断言请求格式。'''

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Dict[str, str], Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
        timeout: float,
    ) -> Tuple[int, Mapping[str, str], bytes]:
        payload: Dict[str, Any] = json.loads(body) if body else {}
        self.calls.append((method, url, dict(headers), payload))
        if url.endswith("/chat/completions"):
            parts = payload["messages"][0]["content"]
            text = " ".join(
                part.get("text", "")
                for part in parts
                if part.get("type") == "text"
            )
            return 200, {}, _json(
                {"choices": [{"message": {"content": f"ark:{text}"}}],
                 "usage": {"total_tokens": 3}}
            )
        if url.endswith("/images/generations"):
            return 200, {}, _json(
                {"data": [{"b64_json": PNG_B64}], "usage": {"images": 1}}
            )
        if method == "POST" and url.endswith("/contents/generations/tasks"):
            return 200, {}, _json({"id": "cgt-demo"})
        if method == "GET" and "/contents/generations/tasks/" in url:
            return 200, {}, _json(
                {
                    "status": "succeeded",
                    "content": {"video_url": "https://demo.local/v.mp4"},
                    "usage": {"completion_tokens": 1},
                }
            )
        if method == "GET" and url.endswith("/v.mp4"):
            return 200, {}, b"fake-mp4"
        raise AssertionError(f"unexpected request: {method} {url}")


def _agent_with_fake_client(
    transport: Optional[HttpTransport] = None,
) -> Agent:
    agent = Agent(api_key="test-key", register_as_default=False)
    agent.multimodal_client = DoubaoMultimodalClient(
        api_key="test-key",
        transport=transport or FakeArkTransport(),
        poll_interval=0,
        poll_timeout=1,
    )
    return agent


class TestVarTypes(unittest.TestCase):
    def test_file_type_roundtrip_and_ops(self) -> None:
        root = tempfile.mkdtemp()
        path = os.path.join(root, "note.txt")
        wf = Workflow(
            File("src"),
            Set("src", path),
            Set("write_result", File.write(Var("src"), "你好，多模态")),
            Set("content", File.read(Var("src"))),
            Set("exists", File.exists(Var("src"))),
            Set("name", File.filename(Var("src"))),
            Set("suffix", File.suffix(Var("src"))),
            Set("size", File.size(Var("src"))),
        )
        ctx = wf.run()
        self.assertIsInstance(ctx["src"], FileRef)
        self.assertEqual(ctx["content"], "你好，多模态")
        self.assertTrue(ctx["exists"])
        self.assertEqual(ctx["name"], "note.txt")
        self.assertEqual(ctx["suffix"], ".txt")
        self.assertGreater(ctx["size"], 0)

    def test_file_type_rejects_non_path(self) -> None:
        wf = Workflow(File("src"), Set("src", 123))
        with self.assertRaises(VarTypeError):
            wf.run()

    def test_bool_type(self) -> None:
        wf = Workflow(
            Bool("ok"),
            Set("ok", "是"),
            Set("back", Bool.not_(Var("ok"))),
            Set("both", Bool.and_(Var("ok"), True)),
            Set("one", Bool.or_(False, Var("back"))),
        )
        ctx = wf.run()
        self.assertIs(ctx["ok"], True)
        self.assertIs(ctx["back"], False)
        self.assertIs(ctx["both"], True)
        self.assertIs(ctx["one"], False)
        with self.assertRaises(VarTypeError):
            Workflow(Bool("bad"), Set("bad", "maybe")).run()

    def test_time_type(self) -> None:
        wf = Workflow(
            Time("t0"),
            Time("t1"),
            Set("t0", "2026-11-03T12:00:00"),
            Set("t1", Time.add(Var("t0"), days=1)),
            Set("year", Time.year(Var("t1"))),
            Set("iso", Time.format(Var("t1"), "%Y-%m-%d")),
            Set("diff", Time.diff(Var("t1"), Var("t0"))),
        )
        ctx = wf.run()
        self.assertEqual(ctx["t1"].year, 2026)
        self.assertEqual(ctx["year"], 2026)
        self.assertEqual(ctx["iso"], "2026-11-04")
        self.assertEqual(ctx["diff"], 86400.0)

    def test_time_default_is_now(self) -> None:
        ctx = Workflow(Time("started")).run()
        self.assertIsInstance(ctx["started"], datetime)

    def test_pyobj_type_and_ops(self) -> None:
        class Box:
            value: int = 7

            def double(self) -> int:
                return self.value * 2

        box = Box()
        wf = Workflow(
            PyObj("box"),
            Set("box", box),
            Set("value", PyObj.attr(Var("box"), "value")),
            Set("double", PyObj.call(PyObj.attr(Var("box"), "double"))),
            Set("type", PyObj.type_name(Var("box"))),
        )
        ctx = wf.run()
        self.assertIs(ctx["box"], box)
        self.assertEqual(ctx["value"], 7)
        self.assertEqual(ctx["double"], 14)
        self.assertEqual(ctx["type"], "Box")

        mapping = Workflow(
            PyObj("data"),
            Set("data", {"a": 1}),
            Set("got", PyObj.get(Var("data"), "a", 0)),
            Set("keys", PyObj.keys(Var("data"))),
        ).run()
        self.assertEqual(mapping["got"], 1)
        self.assertEqual(mapping["keys"], ("a",))


class TestDoubaoMultimodalClient(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp: str = tempfile.mkdtemp()
        self.transport = FakeArkTransport()
        self.client = DoubaoMultimodalClient(
            api_key="test-key",
            transport=self.transport,
            poll_interval=0,
            poll_timeout=1,
        )

    def test_default_provider_is_doubao(self) -> None:
        spec = get_multimodal_spec(None)
        self.assertEqual(spec.name, "doubao")
        self.assertEqual(spec.model_for(Modality.VIDEO), "doubao-seedance-1-0-pro-250528")
        self.assertTrue(spec.supports(Modality.VIDEO, Operation.GENERATE))

    def test_text_generate(self) -> None:
        result = self.client.generate("text", "你好")
        self.assertEqual(result.text, "ark:你好")
        method, url, headers, payload = self.transport.calls[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "你好")

    def test_image_understand_sends_data_url(self) -> None:
        path = os.path.join(self.tmp, "in.png")
        with open(path, "wb") as handle:
            handle.write(b"png-bytes")

        result = self.client.understand("image", path, prompt="描述")
        self.assertEqual(result.text, "ark:描述")
        payload = self.transport.calls[-1][3]
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "描述"})
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_image_generate_and_save(self) -> None:
        target = os.path.join(self.tmp, "cat.png")
        result = self.client.generate(
            "image", "一只猫", save_to=target, size="1024x1024"
        )
        self.assertTrue(os.path.isfile(target))
        self.assertIsNotNone(result.media)
        assert result.media is not None
        self.assertEqual(result.media.path, target)
        payload = self.transport.calls[-1][3]
        self.assertEqual(payload["prompt"], "一只猫")
        self.assertEqual(payload["size"], "1024x1024")

    def test_video_generate_polls_and_downloads(self) -> None:
        target = os.path.join(self.tmp, "demo.mp4")
        result = self.client.generate(
            "video", "镜头缓慢推进", save_to=target, duration=5, ratio="16:9"
        )
        self.assertEqual(result.task_id, "cgt-demo")
        self.assertIsNotNone(result.media)
        assert result.media is not None
        self.assertEqual(result.media.path, target)
        self.assertTrue(os.path.isfile(target))
        with open(target, "rb") as handle:
            self.assertEqual(handle.read(), b"fake-mp4")

        created = [call for call in self.transport.calls if call[0] == "POST"][0]
        self.assertTrue(created[1].endswith("/contents/generations/tasks"))
        self.assertEqual(created[3]["model"], "doubao-seedance-1-0-pro-250528")
        self.assertEqual(created[3]["duration"], 5)
        self.assertEqual(created[3]["ratio"], "16:9")

    def test_unsupported_modality_rejected_before_http(self) -> None:
        with self.assertRaises(Exception) as caught:
            self.client.generate("audio", "hello")
        self.assertIn("不支持", str(caught.exception))
        self.assertEqual(self.transport.calls, [])

    def test_registry_custom_provider_supports_new_modality(self) -> None:
        class AudioClient(BaseMultimodalClient):
            vendor = "audio-lab"

            def run(self, request: MultimodalRequest) -> MultimodalResult:
                return MultimodalResult(
                    operation=request.operation,
                    modality=request.modality,
                    text=f"synthesized:{request.prompt}",
                )

        spec = MultimodalSpec(
            name="audio-lab",
            label="Audio Lab",
            kind="custom",
            models={"audio": "audio-1"},
            capabilities=frozenset(
                {(Modality.TEXT, Operation.GENERATE), ("audio", Operation.GENERATE)}
            ),
        )
        register_multimodal_provider(spec, AudioClient)
        client = make_multimodal_client("audio-lab", api_key="x")
        result = client.generate("audio", "你好")
        self.assertEqual(result.modality, "audio")
        self.assertEqual(result.text, "synthesized:你好")
        self.assertEqual(get_multimodal_spec("audio-lab").model_for("audio"), "audio-1")


class TestMultimodalSteps(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp: str = tempfile.mkdtemp()
        self.transport = FakeArkTransport()
        self.agent = _agent_with_fake_client(self.transport)

    def test_full_workflow_end_to_end(self) -> None:
        image = os.path.join(self.tmp, "cat.png")
        video = os.path.join(self.tmp, "demo.mp4")
        wf = Workflow(
            File("image"),
            File("video"),
            Bool("ok"),
            Time("started"),
            Set("started", Time.now()),
            TextGen("你好", output_key="intro"),
            ImageGen("一只猫", output_key="image", save_to=image),
            ImageUnderstand(Var("image"), prompt="描述画面", output_key="image_desc"),
            VideoGen(
                "让画面动起来",
                inputs=[Var("image")],
                output_key="video",
                save_to=video,
                duration=5,
            ),
            VideoUnderstand(Var("video"), prompt="描述视频", output_key="video_desc"),
            Set("ok", Bool.and_(File.exists(Var("image")), File.exists(Var("video")))),
            Set("elapsed", Time.diff(Time.now(), Var("started"))),
            Echo("{{intro}} | {{image_desc}} | {{video_desc}} | {{ok}}"),
        )
        ctx = wf.run(agent=self.agent, verbose=False)
        self.assertIsInstance(ctx["image"], FileRef)
        self.assertIsInstance(ctx["video"], FileRef)
        self.assertEqual(ctx["image"].path, image)
        self.assertEqual(ctx["video"].path, video)
        self.assertTrue(ctx["ok"])
        self.assertGreaterEqual(ctx["elapsed"], 0)
        self.assertTrue(ctx["intro"].startswith("ark:"))
        self.assertTrue(ctx["image_desc"].startswith("ark:"))

        # 视频生成请求里应该带着上一张图片作为首帧
        video_call = [
            call for call in self.transport.calls
            if call[0] == "POST" and "contents/generations/tasks" in call[1]
        ][0]
        content = video_call[3]["content"]
        self.assertEqual(content[0]["text"], "让画面动起来")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_generic_step_supports_custom_modality(self) -> None:
        class AudioClient(BaseMultimodalClient):
            vendor = "test-audio"

            def run(self, request: MultimodalRequest) -> MultimodalResult:
                return MultimodalResult(
                    operation=request.operation,
                    modality=request.modality,
                    text="audio-ok",
                )

        self.agent.multimodal_client = AudioClient()
        from promptkit.core import Generate

        ctx = Workflow(
            Generate("audio", "hello", output_key="audio_result")
        ).run(agent=self.agent)
        self.assertEqual(ctx["audio_result"], "audio-ok")

    def test_step_without_agent_reports_clear_error(self) -> None:
        agent = Agent(api_key="test-key", register_as_default=False)
        Agent.set_default(None)
        with self.assertRaises(ValueError):
            Workflow(ImageGen("x")).run(agent=None)


if __name__ == "__main__":
    unittest.main()
