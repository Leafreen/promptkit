# mcp/security.py
'''MCP 工具安全层 —— 默认拒绝所有 MCP 工具，必须显式在 allowlist 里才放行。

为什么要默认拒绝：MCP Server 是外部进程，MCPTool 能做什么完全取决于 Server 开了哪些
工具。不做 allowlist 相当于把 Agent 的权限全开 —— 跟"外部进程不可信"的基本假设冲突。

注册方式：

    from promptkit.mcp.security import register_mcp_security
    from promptkit.mcp.hooks import get_default_hook_registry

    register_mcp_security(get_default_hook_registry(), {"fs__read_file", "fs__list_dir"})

之后所有 MCPTool.call() 都会先经过 allowlist 检查。不在 allowlist 里的直接抛
PermissionError，不会真调 MCP Server。
'''
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from promptkit.mcp.hooks import HookRegistry, ToolHookResult

__all__ = ["mcp_permission_gate", "register_mcp_security"]

logger = logging.getLogger(__name__)


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    '''简单脱敏：把看起来像密钥 / token / key 的值替换成 ***。'''
    out: Dict[str, Any] = {}
    sensitive_keys = {"key", "api_key", "token", "secret", "password", "auth"}
    for k, v in args.items():
        if any(s in k.lower() for s in sensitive_keys):
            out[k] = "***"
        else:
            out[k] = v
    return out


def mcp_permission_gate(
    allowlist: Optional[Set[str]],
    tool_name: str,
    args: Dict[str, Any],
) -> ToolHookResult:
    '''核心 gate：allowlist 为 None / 空 → 默认拒绝；工具在 allowlist 里 → 放行。

    Args:
        allowlist: 允许调用的 MCPTool.name 集合，None 或空集 = 全部拒绝。
        tool_name: 形如 `"filesystem__read_file"` 的命名空间全名。
        args: 工具调用参数（已脱敏）。

    Returns:
        ToolHookResult，deny 时 reason 含工具名和 allowlist 快照（截断到前 10 个）。
    '''
    if not allowlist:
        preview = "empty"
    else:
        listed = sorted(allowlist)
        preview = ", ".join(listed[:10])
        if len(listed) > 10:
            preview += f", ... (+{len(listed) - 10} more)"

    if not allowlist or tool_name not in allowlist:
        logger.warning(
            "MCP 工具 %r 被默认安全层拒绝（allowlist=%s, args=%s）",
            tool_name, preview, _sanitize_args(args),
        )
        return ToolHookResult(
            decision="deny",
            reason=f"MCP 工具未授权: {tool_name!r}（allowlist={preview}）",
        )

    return ToolHookResult(decision="allow")


def register_mcp_security(
    hook_registry: HookRegistry,
    allowlist: Set[str],
) -> None:
    '''把安全 gate 注册到 HookRegistry 的 before_tool_call。

    Args:
        hook_registry: 目标 HookRegistry（通常是 HookRegistry.default）。
        allowlist: 允许调用的 MCPTool.name 集合。**允许空集**（全部拒绝）；
            要"全部允许"请手动清空 hook_registry 或不调本函数。

    注册后 MCPTool.call() 每次执行前都会跑这个 gate —— 不在 allowlist 里的抛 PermissionError。
    '''
    if allowlist is None:
        allowlist = set()
    if not isinstance(allowlist, set):
        allowlist = set(allowlist)

    # 闭包捕获 allowlist —— hook_registry.register_before 存的是这个闭包
    def gate(tool_name: str, args: Dict[str, Any]) -> ToolHookResult:
        return mcp_permission_gate(allowlist, tool_name, args)

    hook_registry.register_before(gate)
    logger.info(
        "MCP 安全 gate 已注册，allowlist 含 %d 个工具：%s",
        len(allowlist), ", ".join(sorted(allowlist)[:10]) or "(empty)",
    )
# 职责：MCP 默认安全 gate —— allowlist 为空时全部拒绝；register_mcp_security 把 gate 挂到 HookRegistry.before_tool_call
