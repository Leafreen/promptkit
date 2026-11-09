# examples/mcp_agent.py
'''MCP Client 示例：FakeMCPClient 跑通 Agent → MCPTool 调用流程。

核心展示：
1. ToolRegistry 注册 MCPTool
2. Agent 自主调用（tool_calls 循环零改动）
3. MCPTool.call() 内部 asyncio.run → 每次新起 async with Client

依赖：
    pip install promptkit[mcp]   # 需要 mcp SDK 才能连真实 Server
    python examples/mcp_agent.py  # FakeClient 不需要 SDK

跑真实 Server 时，把 FakeMCPTool 替换成：
    import asyncio
    from promptkit.mcp import load_mcp_tools
    asyncio.run(load_mcp_tools(registry, "http://localhost:3000/mcp", "filesystem"))
'''
from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import dataclass

from promptkit.agent import Agent
from promptkit.core.tool.registry import ToolRegistry
from promptkit.core.tool.base import Tool


# ---------------------------------------------------------------------------
# 1. Fake 层 — 不连真实 MCP Server
# ---------------------------------------------------------------------------

@dataclass
class FakeMCPTool(Tool):
    '''完全在本地跑的假 MCP 工具 —— 演示 Agent 怎么通过 Tool 协议调它。

    继承 Tool 协议，额外加 server_name/tool_name/source 字段（MCPTool 真实实现
    会有这些字段）。
    '''

    func: Any = lambda **_: None  # Tool 必需；真正执行走 call()
    server_name: str = ""
    tool_name: str = ""
    source: str = "mcp"

    def call(self, arguments: Optional[Dict[str, object]] = None) -> str:
        args = arguments or {}
        if self.tool_name == "read_file":
            path = args.get("path", "")
            return f"[fake read_file] 读取 {path} 的内容：Hello World!"
        if self.tool_name == "list_dir":
            path = args.get("path", ".")
            return f"[fake list_dir] {path} 下有：main.py, config.yaml, README.md"
        return f"[fake {self.tool_name}] args={args}"


def make_fake_mcp_tool(server_name: str, tool_name: str, description: str,
                       parameters: Dict[str, Any]) -> FakeMCPTool:
    return FakeMCPTool(
        name=f"{server_name}__{tool_name}",
        description=description,
        parameters=parameters,
        server_name=server_name,
        tool_name=tool_name,
    )


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    # 注册 MCPTool（fake 版）
    registry = ToolRegistry([
        make_fake_mcp_tool(
            "filesystem", "read_file", "读取文件内容",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        make_fake_mcp_tool(
            "filesystem", "list_dir", "列出目录下的文件",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ])

    # Agent 集成方式（注释）：
    #   agent = Agent(api_key="<你的密钥>", vendor="doubao")
    #   agent.ask("读一下 config.yaml 里有什么", tools=registry.all())
    # Agent 循环零改动 —— MCPTool 和内置 Tool 走同一套 tool_calls 流程

    # 直接演示 ToolRegistry 的用法
    print("=" * 50)
    print("MCP Tool Registry 里的工具：")
    for t in registry.all():
        print(f"  • {t.name}: {t.description}")
        print(f"    parameters = {t.parameters}")

    print()
    print("Registry 操作：")
    print(f"  len = {len(registry)}")
    print(f"  contains 'filesystem__read_file' = {registry.contains('filesystem__read_file')}")
    print(f"  get nonexistent = {registry.get('nonexistent')}")

    print()
    print("直接调 read_file：")
    tool = registry.get("filesystem__read_file")
    print(f"  → {tool.call({'path': './config.yaml'})}")

    print()
    print("直接调 list_dir：")
    tool2 = registry.get("filesystem__list_dir")
    print(f"  → {tool2.call({'path': './'})}")

    print()
    print("MCPTool 继承 Tool 协议 ✓ —— Agent 循环零改动 ✓")


if __name__ == "__main__":
    main()
