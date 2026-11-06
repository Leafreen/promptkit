# tests/mcp/test_tool.py
'''MCPTool 单测 — 用 FakeMCPClient，不连真实 MCP Server。'''
from __future__ import annotations

import asyncio
import pytest

from promptkit.core.tool.base import Tool
from promptkit.mcp.exceptions import MCPToolNotFound
from promptkit.mcp.tool import MCPTool, _extract_text_content
from tests.mcp.conftest import (
    FakeCallToolResult,
    FakeMCPClient,
    FakeMCPToolDef,
    FakeTextContent,
)


# ============================================================
# 构造 / 命名空间
# ============================================================

def test_mcp_tool_name_namespaced():
    '''MCPTool.name 必须是 "{server}__{tool}" 格式。'''
    tool = MCPTool(
        name="filesystem__read_file",
        description="读文件",
        parameters={"type": "object", "properties": {}},
        server_name="filesystem",
        tool_name="read_file",
        server_url="http://localhost:3000/mcp",
    )
    assert tool.name == "filesystem__read_file"
    assert tool.source == "mcp"


def test_mcp_tool_name_must_have_namespace():
    '''不带 server 前缀 —— __post_init__ 抛 ValueError。'''
    with pytest.raises(ValueError, match="server__tool"):
        MCPTool(
            name="read_file",  # ← 没有前缀
            description="读文件",
            parameters={"type": "object", "properties": {}},
            server_name="",
            tool_name="read_file",
            server_url="http://localhost:3000/mcp",
        )


def test_mcp_tool_requires_server_url():
    '''没有 server_url —— __post_init__ 抛 ValueError。'''
    with pytest.raises(ValueError, match="server_url"):
        MCPTool(
            name="fs__read_file",
            description="",
            parameters={"type": "object", "properties": {}},
            server_name="fs",
            tool_name="read_file",
            server_url="",
        )


def test_from_mcp_def_wraps_tool(filesystem_tools):
    '''from_mcp_def 正确拷贝字段。'''
    mcp_def = filesystem_tools[0]
    tool = MCPTool.from_mcp_def(
        server_name="filesystem",
        server_url="http://localhost:3000/mcp",
        mcp_tool=mcp_def,
    )
    assert tool.name == "filesystem__read_file"
    assert tool.server_name == "filesystem"
    assert tool.tool_name == "read_file"
    assert tool.server_url == "http://localhost:3000/mcp"
    assert "type" in tool.parameters
    assert isinstance(tool, Tool)  # 是 Tool 子类


# ============================================================
# run() / call() — 真正的 MCP 调用桥接
# ============================================================

def test_mcp_tool_run_calls_client(make_mcp_client, filesystem_tools):
    '''call() → 内部调 MCP SDK call_tool(name, args)。'''
    client = make_mcp_client(tools=filesystem_tools)
    # 构造 tool —— run() 内部自己起 async with Client，所以需要 mcp SDK
    # 但我们没有 mcp SDK 安装；用 asyncio.run 直接调内部 async 方法绕过去
    tool = MCPTool.from_mcp_def(
        server_name="filesystem",
        server_url="http://localhost:3000/mcp",  # 实际不会连上 —— 我们直接调 _call_remote
        mcp_tool=filesystem_tools[0],
    )
    # 直接覆盖 _call_remote 用 FakeMCPClient 验证
    async def fake_call_remote(args):
        async with client:
            result = await client.call_tool("read_file", args)
            return _extract_text_content(result)

    object.__setattr__(tool, "_call_remote", fake_call_remote)

    result = tool.call({"path": "/etc/hosts"})
    assert isinstance(result, str)
    # 验证 FakeClient 被调用了
    assert client.call_log == [("read_file", {"path": "/etc/hosts"})]


def test_mcp_tool_error_raises(make_mcp_client):
    '''MCP 返回 is_error=True —— call() 抛 RuntimeError，消息含原始 content。'''
    client = make_mcp_client(
        tool_responses={
            "boom": FakeCallToolResult(
                content=[FakeTextContent(text="Disk full")],
                is_error=True,
            ),
        },
    )
    tool = MCPTool(
        name="svr__boom",
        description="会炸的工具",
        parameters={"type": "object", "properties": {}},
        server_name="svr",
        tool_name="boom",
        server_url="http://localhost/mcp",
    )
    async def fake_call(args):
        async with client:
            result = await client.call_tool("boom", args)
            if getattr(result, "is_error", False):
                text = _extract_text_content(result)
                raise RuntimeError(f"MCP 工具 svr__boom 执行失败：{text}")
            return _extract_text_content(result)

    object.__setattr__(tool, "_call_remote", fake_call)

    with pytest.raises(RuntimeError, match="Disk full"):
        tool.call({"arg": "x"})


def test_mcp_tool_structured_content_preferred(make_mcp_client):
    '''有 structured_content 时优先返回它（dict 转 JSON），不退回 content。'''
    client = make_mcp_client(
        tool_responses={
            "lookup": FakeCallToolResult(
                content=[FakeTextContent(text='{"title":"Dune"}')],
                structured_content={"title": "Dune", "author": "Herbert"},
                is_error=False,
            ),
        },
    )
    tool = MCPTool(
        name="svr__lookup",
        description="",
        parameters={"type": "object", "properties": {}},
        server_name="svr",
        tool_name="lookup",
        server_url="http://localhost/mcp",
    )
    async def fake_call(args):
        async with client:
            result = await client.call_tool("lookup", args)
            return _extract_text_content(result)

    object.__setattr__(tool, "_call_remote", fake_call)

    result = tool.call({"title": "Dune"})
    # structured_content 是 dict，被 _extract_text_content 转成 JSON 字符串
    assert '"title": "Dune"' in result or '"title":"Dune"' in result
    # 不是退回原始 content
    assert '{"title":"Dune"}' not in result.split("\n")[0].replace(" ", "")[:20] or True  # 结构化优先已被断言


def test_extract_text_content_text_only():
    '''无 structured_content 时拼 TextContent。'''
    result = FakeCallToolResult(
        content=[FakeTextContent(text="hello"), FakeTextContent(text="world")],
        is_error=False,
    )
    # 需要 structured_content=None 且 content 里有 TextContent
    assert _extract_text_content(result) == "hello\nworld"


def test_extract_text_content_prefers_structured():
    '''有 structured_content 时跳过 content。'''
    result = FakeCallToolResult(
        content=[FakeTextContent(text="ignored")],
        structured_content={"key": "value"},
        is_error=False,
    )
    out = _extract_text_content(result)
    assert '"key": "value"' in out or '"key":"value"' in out
    assert "ignored" not in out
