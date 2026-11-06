# mcp/loader.py
'''连接 MCP Server → 发现工具 → 注册到 ToolRegistry。

只做 Streamable HTTP（规格书约束），不实现 stdio 子进程。

用法（在 async 上下文里）：

    registry = ToolRegistry()
    registered = await load_mcp_tools(registry, "http://localhost:3000/mcp", "filesystem")
    print(registered)  # ["filesystem__read_file", "filesystem__list_dir"]

MCP SDK 未安装时 import 会抛清晰 ImportError，引导用户装 `pip install promptkit[mcp]`。
'''
from __future__ import annotations

import logging
from typing import List, Optional

from promptkit.core.tool.registry import ToolRegistry

try:
    from mcp import Client as MCPClient
except ImportError:  # pragma: no cover - 运行时依赖未安装
    MCPClient = None  # type: ignore[assignment,misc]

from promptkit.mcp.exceptions import MCPConnectionError
from promptkit.mcp.tool import MCPTool, build_mcp_tools

__all__ = ["load_mcp_tools"]

logger = logging.getLogger(__name__)


async def load_mcp_tools(
    registry: ToolRegistry,
    server_url: str,
    server_name: str,
    transport: str = "http",
    *,
    allowed_tools: Optional[List[str]] = None,
) -> List[str]:
    '''连接 MCP Server，发现工具并注册到 ToolRegistry。

    Args:
        registry: 要注册到的 ToolRegistry。
        server_url: MCP Server 的 Streamable HTTP endpoint（比如
            `"http://localhost:3000/mcp"`）。
        server_name: Server 名字，用于命名空间隔离 —— 注册出来的工具名是
            `{server_name}__{tool.name}`。
        transport: 传输协议，当前只支持 `"http"`（Streamable HTTP）。
        allowed_tools: 非空时只注册列表内的裸工具名（不含 server_name 前缀）；
            None / 空列表 = 注册全部。

    Returns:
        成功注册的工具名列表（带 server_name 前缀）。重名被跳过的不算。

    Raises:
        ImportError: `mcp` SDK 未安装（会提示 `pip install promptkit[mcp]`）。
        MCPConnectionError: Server 不可达或握手失败。
        ValueError: `transport` 不是支持的值。
    '''
    if MCPClient is None:
        raise ImportError(
            "MCP 可选依赖未安装：请执行 `pip install promptkit[mcp]`"
        )

    if transport != "http":
        raise ValueError(
            f"当前只支持 transport='http'（Streamable HTTP），得到 {transport!r}"
        )

    # 连接 + list_tools（连接用完立即关闭，MCPTool 自己会按需重连）
    try:
        async with MCPClient(server_url) as client:
            result = await client.list_tools()
            raw_tools = getattr(result, "tools", [])
    except ImportError:
        raise  # SDK 缺失，上面已处理
    except Exception as exc:
        raise MCPConnectionError(
            f"连接 MCP Server {server_name!r} ({server_url}) 失败：{exc}"
        ) from exc

    # allowed_tools 过滤（裸工具名匹配，不含前缀）
    if allowed_tools:
        allowed = set(allowed_tools)
        raw_tools = [t for t in raw_tools if t.name in allowed]

    # 构造 MCPTool + 注册（单个失败 warning 跳过）
    tools = build_mcp_tools(server_name, server_url, raw_tools)

    registered: List[str] = []
    for tool in tools:
        if registry.register(tool):
            registered.append(tool.name)

    logger.info(
        "MCP Server %r 发现 %d 个工具，注册 %d 个",
        server_name, len(raw_tools), len(registered),
    )
    return registered
# 职责：async load_mcp_tools — 连接 MCP Server 发现工具并注册到 ToolRegistry；单个失败不中断整体
