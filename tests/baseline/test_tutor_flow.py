# tests/baseline/test_tutor_flow.py
'''基线：课程学习助手（tutor 场景）不依赖真实 API 的核心流程。'''
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

from promptkit.core import Verbosity
from promptkit.memory.base import InMemoryMemory
from tests.baseline.fake_agent import FakeAgent

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "course_assistant.py"


@pytest.fixture()
def tutor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    '''加载 examples/course_assistant.py，并把副作用（会话 / 记忆）关进 tmp。'''
    spec = importlib.util.spec_from_file_location(
        "baseline_course_assistant", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "memory", InMemoryMemory())
    monkeypatch.setattr(module, "SESSION", str(tmp_path / "session.json"))
    return module


def test_tutor_single_turn(tutor: object) -> None:
    '''一轮问答：模型收到输入、回复写回 Context。'''
    agent = FakeAgent(replies=["这一节的要点是：先定义，再举例。"])
    workflow = tutor.build().build()  # type: ignore[attr-defined]

    with mock.patch("builtins.input", side_effect=["这一节讲什么？", "exit"]):
        context = workflow.run(agent=agent, verbose=Verbosity.SILENT)

    assert context.get("model_reply") == "这一节的要点是：先定义，再举例。"
    assert len(agent.calls) == 1
    assert context.get("chat_tools")


def test_tutor_multi_turn_uses_latest_reply(tutor: object) -> None:
    '''两轮问答：每轮都会调模型，最终结果取最后一轮。'''
    agent = FakeAgent(replies=["答案1", "答案2"])
    workflow = tutor.build().build()  # type: ignore[attr-defined]

    with mock.patch("builtins.input", side_effect=["问题1", "问题2", "exit"]):
        context = workflow.run(agent=agent, verbose=Verbosity.SILENT)

    assert len(agent.calls) == 2
    assert context.get("model_reply") == "答案2"
