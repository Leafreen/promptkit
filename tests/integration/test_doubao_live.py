'''豆包真实 API 集成测试。

需要环境变量 `DOUBAO_API_KEY`（或 `ARK_API_KEY`）；没配置时整体跳过。
跑法：`pytest tests/integration/ -v`
'''
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DOUBAO_API_KEY") and not os.getenv("ARK_API_KEY"),
    reason="需要 DOUBAO_API_KEY / ARK_API_KEY 环境变量",
)


def test_doubao_supports_modalities():
    from promptkit.client_sys import DoubaoClient

    client = DoubaoClient()
    assert client.supports("text")
    assert client.supports("image")
    assert client.supports("video")
    assert not client.supports("audio")
    assert not client.supports("file")


def test_doubao_text_roundtrip():
    from promptkit import Agent

    agent = Agent(vendor="doubao")
    reply = agent.ask("用一句话介绍你自己")
    assert isinstance(reply, str)
    assert reply.strip()
