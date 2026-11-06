# mcp/config.py
'''从配置字典批量加载 MCP Server。

配置格式：

    mcp_servers:
      - name: filesystem
        url: http://localhost:3000/mcp
        transport: http
        allowed_tools: ["read_file", "list_dir"]
      - name: github
        url: http://localhost:3001/mcp

`allowed_tools` 为空表示全部允许；非空则只注册列表内的工具。
单个 Server 失败不影响其他 Server，记录 warning 继续。
'''
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from promptkit.core.tool.registry import ToolRegistry

from promptkit.mcp.exceptions import MCPConnectionError
from promptkit.mcp.loader import load_mcp_tools

__all__ = ["load_mcp_from_config"]

logger = logging.getLogger(__name__)


async def load_mcp_from_config(
    registry: ToolRegistry,
    config: Dict[str, Any],
    *,
    servers_key: str = "mcp_servers",
) -> List[str]:
    '''从配置字典批量加载 MCP Server。

    Args:
        registry: 要注册到的 ToolRegistry。
        config: 配置字典，`config[servers_key]` 是 Server 列表。每个元素是 dict：
            - `name` (str): Server 名字（必填）。
            - `url` (str): Server URL（必填）。
            - `transport` (str, default "http"): 传输协议。
            - `allowed_tools` (list[str], default None): 非空时白名单注册。
        servers_key: Server 列表在 config 里的 key，默认 `"mcp_servers"`。

    Returns:
        所有成功注册的工具名列表（带 `server__` 前缀）。

    异常策略：单个 Server 失败（连接失败 / 配置缺 name 或 url）打 warning，
    继续加载下一个，不整体中断。
    '''
    servers = config.get(servers_key)
    if not servers:
        logger.warning("配置里 %r 为空，没有加载任何 MCP Server", servers_key)
        return []

    all_registered: List[str] = []

    for idx, server_cfg in enumerate(servers):
        if not isinstance(server_cfg, dict):
            logger.warning(
                "Server 配置 #%d 不是 dict，跳过：%r", idx, server_cfg
            )
            continue

        name = server_cfg.get("name")
        url = server_cfg.get("url")
        if not name or not url:
            logger.warning(
                "Server 配置 #%d 缺少 name 或 url，跳过：%r", idx, server_cfg
            )
            continue

        try:
            registered = await load_mcp_tools(
                registry,
                server_url=str(url),
                server_name=str(name),
                transport=str(server_cfg.get("transport", "http")),
                allowed_tools=(
                    list(server_cfg["allowed_tools"])
                    if server_cfg.get("allowed_tools")
                    else None
                ),
            )
            all_registered.extend(registered)
        except MCPConnectionError as exc:
            logger.warning("Server %r 连接失败：%s", name, exc)
        except Exception as exc:
            logger.warning("Server %r 加载异常：%s", name, exc)

    logger.info(
        "从配置加载 %d 个 MCP Server，共注册 %d 个工具",
        len(servers), len(all_registered),
    )
    return all_registered
# 职责：从 dict 配置批量加载 MCP Server；单个失败 warning 跳过，整体继续
