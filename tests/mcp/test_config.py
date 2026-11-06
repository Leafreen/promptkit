# tests/mcp/test_config.py
'''load_mcp_from_config 单测 — 用 asyncio.run 手动驱动 async，不用 pytest-asyncio。

真实 MCP SDK 未安装也能跑 —— 把 load_mcp_from_config 内部调的 load_mcp_tools
替换成同步 fake，不连 Server。
'''
from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from promptkit.core.tool.registry import ToolRegistry
from promptkit.mcp.config import load_mcp_from_config
from promptkit.mcp.exceptions import MCPConnectionError
from promptkit.mcp.tool import build_mcp_tools
from tests.mcp.conftest import FakeMCPToolDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_with_fake(config, fake_load, monkeypatch):
    '''在 monkeypatch 替换 load_mcp_tools 后跑 load_mcp_from_config。'''
    import promptkit.mcp.config as config_mod
    monkeypatch.setattr(config_mod, "load_mcp_tools", fake_load)
    registry = ToolRegistry()
    registered = await load_mcp_from_config(registry, config)
    return registry, registered


async def _fake_load_2tools(registry, server_url, server_name, transport="http", *, allowed_tools=None):
    '''每个 server 伪造 2 个工具。'''
    tools = [
        FakeMCPToolDef(
            name=f"tool_{i}", description="",
            input_schema={"type": "object", "properties": {}},
        )
        for i in range(2)
    ]
    if allowed_tools:
        allowed = set(allowed_tools)
        tools = [t for t in tools if t.name in allowed]
    mt = build_mcp_tools(server_name, server_url, tools)
    registered: list[str] = []
    for t in mt:
        if registry.register(t):
            registered.append(t.name)
    return registered


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_from_config_registers_all(monkeypatch):
    '''两个 server 都能加载，共 4 个工具。'''
    config = {
        "mcp_servers": [
            {"name": "fs", "url": "http://x/mcp"},
            {"name": "gh", "url": "http://y/mcp"},
        ],
    }
    import promptkit.mcp.config as config_mod
    monkeypatch.setattr(config_mod, "load_mcp_tools", _fake_load_2tools)

    registry, registered = asyncio.run(_load_with_fake(config, _fake_load_2tools, monkeypatch))
    assert len(registered) == 4
    assert registry.contains("fs__tool_0")
    assert registry.contains("gh__tool_1")


def test_load_from_config_allowed_tools(monkeypatch):
    '''allowed_tools 白名单生效。'''
    config = {
        "mcp_servers": [
            {
                "name": "fs", "url": "http://x/mcp",
                "allowed_tools": ["tool_0"],
            },
        ],
    }
    import promptkit.mcp.config as config_mod
    monkeypatch.setattr(config_mod, "load_mcp_tools", _fake_load_2tools)

    registry, registered = asyncio.run(_load_with_fake(config, _fake_load_2tools, monkeypatch))
    assert registered == ["fs__tool_0"]


def test_load_from_config_empty():
    '''配置里没有 mcp_servers —— 返回空列表，不报错。'''
    registry = ToolRegistry()
    registered = asyncio.run(load_mcp_from_config(registry, {}))
    assert registered == []
    registered2 = asyncio.run(load_mcp_from_config(registry, {"other": []}))
    assert registered2 == []


def test_load_from_config_skips_bad_server(monkeypatch):
    '''缺 name / url 的 server 配置跳过，其他照常。'''
    config = {
        "mcp_servers": [
            {"name": "", "url": "http://x"},      # 缺 name
            {"name": "fs"},                        # 缺 url
            {"name": "gh", "url": "http://y"},     # 正常
        ],
    }
    import promptkit.mcp.config as config_mod
    monkeypatch.setattr(config_mod, "load_mcp_tools", _fake_load_2tools)

    registry, registered = asyncio.run(_load_with_fake(config, _fake_load_2tools, monkeypatch))
    assert all(n.startswith("gh__") for n in registered)
    assert len(registered) == 2


def test_load_from_config_server_failure_continues(monkeypatch, caplog):
    '''某个 server 连接失败 —— warning 记录，继续下一个。'''
    import promptkit.mcp.config as config_mod

    call_count = [0]

    async def fake_with_one_fail(registry, server_url, server_name, transport="http", *, allowed_tools=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise MCPConnectionError("Connection refused")
        tools = [FakeMCPToolDef(name="tool", description="",
                               input_schema={"type": "object", "properties": {}})]
        mt = build_mcp_tools(server_name, server_url, tools)
        for t in mt:
            registry.register(t)
        return [t.name for t in mt]

    monkeypatch.setattr(config_mod, "load_mcp_tools", fake_with_one_fail)

    config = {
        "mcp_servers": [
            {"name": "bad", "url": "http://x"},
            {"name": "good", "url": "http://y"},
        ],
    }
    registry = ToolRegistry()
    registered = asyncio.run(load_mcp_from_config(registry, config))
    assert registered == ["good__tool"]
    assert registry.contains("good__tool")
    assert not registry.contains("bad__tool")
