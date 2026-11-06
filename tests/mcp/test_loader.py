# tests/mcp/test_loader.py
'''load_mcp_tools / build_mcp_tools 单测 — FakeMCPClient，不连真实 MCP Server。

注意：测试用 build_mcp_tools 构造 MCPTool 实例 + ToolRegistry.register 集成验证，
跳过真实 MCP SDK 的 async 连接（SDK 未安装也能跑）。
'''
from __future__ import annotations

import pytest

from promptkit.core.tool.base import Tool
from promptkit.core.tool.registry import ToolRegistry
from promptkit.mcp.tool import MCPTool, build_mcp_tools
from tests.mcp.conftest import FakeMCPToolDef


# ============================================================
# build_mcp_tools — 构造工具列表
# ============================================================

def test_build_tools_wraps_all(filesystem_tools):
    '''build_mcp_tools 正确包装所有 MCPToolDef，命名空间隔离。'''
    tools = build_mcp_tools("filesystem", "http://localhost:3000/mcp", filesystem_tools)
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert "filesystem__read_file" in names
    assert "filesystem__list_dir" in names

    tool = next(t for t in tools if t.tool_name == "read_file")
    assert isinstance(tool, MCPTool)
    assert isinstance(tool, Tool)
    assert tool.server_name == "filesystem"
    assert tool.server_url == "http://localhost:3000/mcp"
    assert tool.parameters["type"] == "object"


def test_build_tools_single_failure_continues(caplog):
    '''一个工具构造失败打 warning，其他继续。'''
    import logging
    caplog.set_level(logging.WARNING)

    # mock from_mcp_def 让第二个抛
    original = MCPTool.from_mcp_def
    call_count = [0]

    def patched(server_name, server_url, mcp_tool):
        call_count[0] += 1
        if call_count[0] == 2:
            raise ValueError("simulated constructor failure")
        return original(server_name, server_url, mcp_tool)

    MCPTool.from_mcp_def = patched  # type: ignore[assignment]
    try:
        tools = build_mcp_tools(
            "svr",
            "http://x",
            [
                FakeMCPToolDef(name="good1", description="ok",
                               input_schema={"type": "object", "properties": {}}),
                FakeMCPToolDef(name="broken", description="bad",
                               input_schema={"type": "object", "properties": {}}),
                FakeMCPToolDef(name="good2", description="ok",
                               input_schema={"type": "object", "properties": {}}),
            ],
        )
        assert [t.tool_name for t in tools] == ["good1", "good2"]
        # warning 被记录
        assert any("构造失败" in rec.message for rec in caplog.records)
    finally:
        MCPTool.from_mcp_def = original  # type: ignore[assignment]


# ============================================================
# ToolRegistry.register — 重名不覆盖
# ============================================================

def test_registry_register_skip_on_duplicate():
    '''重名默认跳过，override=True 才覆盖。'''
    registry = ToolRegistry()
    t1 = MCPTool(
        name="fs__read", description="第一版",
        parameters={"type": "object", "properties": {}},
        server_name="fs", tool_name="read",
        server_url="http://v1/mcp",
    )
    t2 = MCPTool(
        name="fs__read", description="第二版",
        parameters={"type": "object", "properties": {}},
        server_name="fs", tool_name="read",
        server_url="http://v2/mcp",
    )
    assert registry.register(t1) is True
    assert registry.register(t2) is False  # 重名跳过
    assert len(registry) == 1
    assert registry.get("fs__read").server_url == "http://v1/mcp"
    # override=True 覆盖
    assert registry.register(t2, override=True) is True
    assert registry.get("fs__read").server_url == "http://v2/mcp"


def test_registry_len_and_iter():
    '''ToolRegistry 支持 len / iter / contains。'''
    registry = ToolRegistry([
        MCPTool(
            name="svr__a", description="",
            parameters={"type": "object", "properties": {}},
            server_name="svr", tool_name="a", server_url="http://x",
        ),
        MCPTool(
            name="svr__b", description="",
            parameters={"type": "object", "properties": {}},
            server_name="svr", tool_name="b", server_url="http://x",
        ),
    ])
    assert len(registry) == 2
    assert registry.contains("svr__a")
    assert "svr__b" in registry
    assert registry.get("nonexistent") is None
    names = [t.name for t in registry]
    assert names == ["svr__a", "svr__b"]


# ============================================================
# allowed_tools 白名单
# ============================================================

def test_allowed_tools_filters_before_build():
    '''build_mcp_tools 本身不做过滤（loader 里做）；验证手动过滤。'''
    mcp_defs = [
        FakeMCPToolDef(name="read", description="", input_schema={"type": "object", "properties": {}}),
        FakeMCPToolDef(name="write", description="", input_schema={"type": "object", "properties": {}}),
        FakeMCPToolDef(name="delete", description="", input_schema={"type": "object", "properties": {}}),
    ]
    allowed = {"read", "delete"}
    filtered = [t for t in mcp_defs if t.name in allowed]
    tools = build_mcp_tools("fs", "http://x", filtered)
    assert {t.tool_name for t in tools} == {"read", "delete"}
