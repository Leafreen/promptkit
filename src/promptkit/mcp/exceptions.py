# mcp/exceptions.py
'''MCP Client 层专用异常。'''
from __future__ import annotations

__all__ = ["MCPError", "MCPConnectionError", "MCPToolNotFound"]


class MCPError(Exception):
    '''MCP 客户端层所有异常的基类。'''


class MCPConnectionError(MCPError):
    '''连接 MCP Server 失败（URL 不可达 / 握手超时 / 协议不匹配）。'''


class MCPToolNotFound(MCPError):
    '''MCP Server 上没有请求的那个工具（Server 返回 Unknown tool）。'''
# 职责：MCP 客户端层专用异常（MCPError + 连接 + 未找到工具）
