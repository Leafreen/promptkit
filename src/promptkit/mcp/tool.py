# mcp/tool.py
'''把 MCP Server 的一个工具包装成 promptkit 的 Tool —— Agent 循环零改动。

设计要点：

1. **命名空间隔离**：`name = "{server_name}__{tool_name}"`，不同 Server 上的
   同名工具不会撞车。
2. **异步桥接只在这一层**：`call()` 内部用 `asyncio.run()` 调 MCP SDK 的
   `call_tool()`，Agent / Step 层保持同步。
3. **连接按调用建**：MCP SDK 的 `Client` 不能跨 `async with` 复用 ——
   每个 MCPTool 持有 `server_url`，`call()` 时临时起 `async with Client(url)`。
4. **错误转译**：MCP 返回 `is_error=True` 时抛 `RuntimeError`（带原始 content
   让 Agent 知道怎么重试）；Server 返回 Unknown tool 时抛 `MCPToolNotFound`。
5. **结果策略**：优先 `structured_content`，没有则取 `content` 里第一个
   TextContent 的文本。
'''
from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from promptkit.core.tool.base import Tool, ToolSchema
from promptkit.mcp.exceptions import MCPToolNotFound

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool as MCPToolDef

__all__ = ["MCPTool", "build_mcp_tools"]

logger = logging.getLogger(__name__)

# MCP SDK 在 import 失败时的延迟引用 —— ToolRegistry 层可以正常 import，
# 只在真正构造 MCPTool 时才要求 SDK 已安装
MCPClient = None  # type: ignore[assignment]


def _ensure_mcp_sdk():
    '''延迟 import MCP SDK；未安装时抛清晰 ImportError。'''
    global MCPClient
    if MCPClient is not None:
        return MCPClient
    try:
        from mcp import Client as _Client
    except ImportError as exc:
        raise ImportError(
            "MCP 可选依赖未安装：请执行 `pip install promptkit[mcp]`"
        ) from exc
    MCPClient = _Client
    return _Client


def _extract_text_content(result: "CallToolResult") -> str:
    '''从 MCP CallToolResult 拼可读文本：优先 structured_content，退回 TextContent。'''
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        if isinstance(sc, str):
            return sc
        return _json.dumps(sc, ensure_ascii=False)

    texts: List[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    return "\n".join(texts)


@dataclass
class MCPTool(Tool):
    '''把 MCP Server 上的一个工具适配成 promptkit 的 Tool。

    Agent 循环一行不改 —— MCPTool 实现 Tool 协议，ToolRegistry 统一管理。

    示例：

        tool = MCPTool(
            name="filesystem__read_file",
            description="读取文件内容",
            parameters={"type": "object", "properties": {"path": {...}}},
            server_name="filesystem",
            tool_name="read_file",
            server_url="http://localhost:3000/mcp",
        )
        tool.call({"path": "/etc/hosts"})
        # → 内部: async with Client(server_url) as c: await c.call_tool("read_file", {...})
    '''

    # Tool dataclass 字段顺序: name, description, parameters, func
    # func 在 Tool 里没有默认值，这里覆盖成默认 lambda —— 真正执行走 call()，不用它
    func: Callable[..., object] = field(default=lambda **_: None)

    # MCP 特有字段
    server_name: str = ""
    tool_name: str = ""
    server_url: str = ""
    source: str = "mcp"
    hooks: Any = None  # HookRegistry 实例；None = 用默认单例（延迟 import）

    def __post_init__(self) -> None:
        '''反推 server_name / tool_name（允许构造时只给 name）；校验 server_url。'''
        if "__" in self.name and not self.server_name:
            self.server_name, self.tool_name = self.name.split("__", 1)
        if not self.server_name or not self.tool_name:
            raise ValueError(
                f"MCPTool.name 必须是 'server__tool' 格式（得到 {self.name!r}）"
            )
        if not self.server_url:
            raise ValueError(f"MCPTool {self.name!r} 需要 server_url")

    # ---- Tool 协议：call() 是 Agent 调的入口 ----

    def _get_hooks(self):
        '''懒拿 HookRegistry（延迟 import 避免循环依赖）。'''
        if self.hooks is not None:
            return self.hooks
        from promptkit.mcp.hooks import get_default_hook_registry
        return get_default_hook_registry()

    def call(self, arguments: Optional[Dict[str, object]] = None) -> str:
        '''同步桥接：HookRegistry.before → asyncio.run → HookRegistry.after。

        before hook 返回 deny 时抛 PermissionError，不会真调 MCP Server。
        '''
        args = cast(Dict[str, Any], arguments or {})
        hooks = self._get_hooks()

        # ---- before hook ----
        try:
            gate_result = hooks.fire_before(self.name, args)
        except Exception:
            gate_result = None
        if gate_result is not None and getattr(gate_result, "is_denied", False):
            raise PermissionError(
                getattr(gate_result, "reason", "MCP 工具被拒绝")
            )

        # ---- 真正调用 ----
        t0 = time.perf_counter()
        result_text = ""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                raise RuntimeError(
                    f"MCPTool.call() 在已有事件循环里不能用（比如 FastAPI / Starlette）。"
                    f"请在 async 函数里直接用 MCPTool.acall()，或提供 Agent 的 executor。"
                )

            result_text = asyncio.run(self._call_remote(args))
            return result_text
        finally:
            duration = time.perf_counter() - t0
            try:
                hooks.fire_after(self.name, args, result_text, duration)
            except Exception:
                pass

    async def acall(self, arguments: Optional[Dict[str, Any]] = None) -> str:
        '''async 入口：已有事件循环时用这个。'''
        return await self._call_remote(arguments or {})

    async def _call_remote(self, args: Dict[str, Any]) -> str:
        '''真正的 async 调用：每次起一个 Client，避免长连接管理复杂度。'''
        ClientCls = _ensure_mcp_sdk()

        async with ClientCls(self.server_url) as client:
            from mcp.exceptions import McpError

            try:
                result = await client.call_tool(self.tool_name, args)
            except McpError as exc:
                message = str(exc) or str(getattr(exc, "__cause__", ""))
                if "Unknown tool" in message or "-32601" in message:
                    raise MCPToolNotFound(
                        f"MCP Server {self.server_name!r} 上没有工具 {self.tool_name!r}"
                    ) from exc
                raise RuntimeError(
                    f"MCP Server {self.server_name!r} 调用 {self.tool_name!r} 失败：{message}"
                ) from exc

            if getattr(result, "is_error", False):
                text = _extract_text_content(result)
                raise RuntimeError(
                    f"MCP 工具 {self.name!r} 执行失败：{text or '(Server 未返回错误详情)'}"
                )

            return _extract_text_content(result)

    # ---- 便捷构造 ----

    @classmethod
    def from_mcp_def(
        cls,
        server_name: str,
        server_url: str,
        mcp_tool: "MCPToolDef",
    ) -> "MCPTool":
        '''从 MCP SDK 的 Tool 定义构造 MCPTool。

        不绑定长生命周期 Client —— 每次 call() 内部自己起 async with Client(url)。
        '''
        parameters = cast(
            ToolSchema,
            mcp_tool.input_schema or {"type": "object", "properties": {}},
        )
        desc = (
            mcp_tool.title
            if mcp_tool.title and not mcp_tool.description
            else mcp_tool.description or mcp_tool.title or ""
        )
        return cls(
            name=f"{server_name}__{mcp_tool.name}",
            description=desc,
            parameters=parameters,
            server_name=server_name,
            tool_name=mcp_tool.name,
            server_url=server_url,
        )


def build_mcp_tools(
    server_name: str,
    server_url: str,
    tools: List["MCPToolDef"],
) -> List[MCPTool]:
    '''批量包装：单个失败打 warning 跳过，不中断整体。'''
    out: List[MCPTool] = []
    for mcp_tool in tools:
        try:
            out.append(MCPTool.from_mcp_def(server_name, server_url, mcp_tool))
        except Exception as exc:
            logger.warning(
                "MCP 工具 %r 构造失败：%s",
                getattr(mcp_tool, "name", "<unknown>"),
                exc,
            )
    return out
# 职责：MCPTool（继承 Tool），持有 server_url，call() 内部每次起 async with Client(url)，支持 acall() 给 async 上下文
