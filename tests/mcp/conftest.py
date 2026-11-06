# tests/mcp/conftest.py
'''MCP 测试共享 fixture — FakeMCPClient / FakeMCPToolDef。'''
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

# 确保 promptkit 可 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ---------------------------------------------------------------------------
# FakeMCPToolDef — mimicking mcp.types.Tool
# ---------------------------------------------------------------------------
@dataclass
class FakeMCPToolDef:
    '''模仿 mcp.types.Tool — list_tools() 返回的工具定义。'''
    name: str
    description: str = ""
    title: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FakeCallToolResult — mimicking mcp.types.CallToolResult
# ---------------------------------------------------------------------------
@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""


@dataclass
class FakeCallToolResult:
    '''模仿 mcp.types.CallToolResult — call_tool() 的返回值。'''
    content: List[Any] = field(default_factory=list)
    structured_content: Optional[Dict[str, Any]] = None
    is_error: bool = False


# ---------------------------------------------------------------------------
# FakeListToolsResult
# ---------------------------------------------------------------------------
@dataclass
class FakeListToolsResult:
    tools: List[FakeMCPToolDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FakeMCPClient — mimicking mcp.Client
# ---------------------------------------------------------------------------
class FakeMCPClient:
    '''MCP Client 假实现 — 不连真实 Server，硬编码工具清单和调用结果。

    关键行为：
    - `async with FakeMCPClient(...) as c:` 可正常进入/退出
    - `await c.list_tools()` → 预配置的工具清单
    - `await c.call_tool(name, args)` → 按 self.tool_responses 或默认返回
    - `raise_on_connect=True` 时 async with 里抛异常（模拟连接失败）
    - `tool_responses[name] = FakeCallToolResult` 可预设某个工具的返回
    '''

    def __init__(
        self,
        tools: Optional[List[FakeMCPToolDef]] = None,
        *,
        tool_responses: Optional[Dict[str, FakeCallToolResult]] = None,
        raise_on_connect: bool = False,
        default_tool_result: Optional[FakeCallToolResult] = None,
    ):
        self.tools = list(tools) if tools else []
        self.tool_responses = dict(tool_responses) if tool_responses else {}
        self.default_tool_result = default_tool_result or FakeCallToolResult(
            content=[FakeTextContent(text="ok")],
            structured_content=None,
            is_error=False,
        )
        self.raise_on_connect = raise_on_connect
        self.call_log: List[tuple] = []  # (name, args)
        self.connected = False

    async def __aenter__(self):
        if self.raise_on_connect:
            raise ConnectionRefusedError("Connection refused (fake)")
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.connected = False

    async def list_tools(self, cursor: Optional[str] = None) -> FakeListToolsResult:
        return FakeListToolsResult(tools=list(self.tools))

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> FakeCallToolResult:
        self.call_log.append((name, arguments))
        if name in self.tool_responses:
            return self.tool_responses[name]
        # Unknown tool 返回 is_error=True（不抛异常 — 跟真实 MCP SDK 行为一致）
        return FakeCallToolResult(
            content=[FakeTextContent(text=f"Unknown tool: {name}")],
            structured_content=None,
            is_error=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def filesystem_tools() -> List[FakeMCPToolDef]:
    return [
        FakeMCPToolDef(
            name="read_file",
            description="读取文件内容",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
        ),
        FakeMCPToolDef(
            name="list_dir",
            description="列出目录下的文件",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
    ]


@pytest.fixture
def fake_filesystem_client(filesystem_tools) -> FakeMCPClient:
    return FakeMCPClient(tools=filesystem_tools)


@pytest.fixture
def make_mcp_client():
    '''工厂：按需组装 fake client。'''
    def _make(**kwargs) -> FakeMCPClient:
        return FakeMCPClient(**kwargs)
    return _make
