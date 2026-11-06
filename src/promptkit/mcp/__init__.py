# mcp/__init__.py
'''MCP Client 支持 —— 把 MCP Server 的工具接进 promptkit 的 Tool 体系。

只做 Client，不做 Server。核心入口：

    from promptkit.mcp import (
        MCPTool, load_mcp_tools, load_mcp_from_config,
        HookRegistry, register_mcp_security,
    )

Agent 循环零改动 — MCPTool 实现 Tool 协议，和内置 Tool 走同一套 tool_calls 流程。

MCP SDK 是可选依赖：`pip install promptkit[mcp]`
'''
from __future__ import annotations

from promptkit.mcp.exceptions import MCPConnectionError, MCPError, MCPToolNotFound

__all__ = [
    "MCPError",
    "MCPConnectionError",
    "MCPToolNotFound",
    # 第一层
    "MCPTool",
    "build_mcp_tools",
    "load_mcp_tools",
    "load_mcp_from_config",
    # 第二轮：Hook + Security
    "HookRegistry",
    "ToolHookResult",
    "get_default_hook_registry",
    "set_default_hook_registry",
    "mcp_permission_gate",
    "register_mcp_security",
]


def __getattr__(name: str):
    '''延迟 import，让 SDK 未安装时也能 import 本模块拿异常类。'''
    if name in ("MCPTool", "build_mcp_tools"):
        from promptkit.mcp.tool import MCPTool, build_mcp_tools
        return locals()[name]
    if name == "load_mcp_tools":
        from promptkit.mcp.loader import load_mcp_tools
        return load_mcp_tools
    if name == "load_mcp_from_config":
        from promptkit.mcp.config import load_mcp_from_config
        return load_mcp_from_config
    if name in ("HookRegistry", "ToolHookResult",
                "get_default_hook_registry", "set_default_hook_registry"):
        from promptkit.mcp.hooks import (
            HookRegistry, ToolHookResult,
            get_default_hook_registry, set_default_hook_registry,
        )
        return locals()[name]
    if name in ("mcp_permission_gate", "register_mcp_security"):
        from promptkit.mcp.security import (
            mcp_permission_gate, register_mcp_security,
        )
        return locals()[name]
    raise AttributeError(f"promptkit.mcp 没有属性 {name!r}")
# 职责：MCP Client 包入口，延迟 import 保持 SDK 未安装时也能 import 异常类
