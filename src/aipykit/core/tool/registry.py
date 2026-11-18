# core/tool/registry.py
'''轻量工具注册表：集中管理 Tool，供 Agent 和 MCP 发现层共用。

为什么单独一个 registry：
- Agent 只收 List[Tool]，没有中央查找点
- MCP loader 需要 `registry.register(tool)` + `registry.get(name)`
- 后续 HookRegistry（第二轮）要按名字审计，registry 是天然索引

## _ctx 注入协议

``execute(name, args, ctx=None)`` 会检查目标 Tool 的 ``func`` 签名：
只有声明了 ``_ctx`` 参数的 Tool 才会被注入运行时上下文，其他 Tool
行为**完全不变**（仍然只传业务参数）。这样 SummonSubAgentTool 等
需要访问运行时状态的工具可以拿到 ctx，而现有 Tool / 函数调用零改动。
'''
from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, cast

from aipykit.core.tool.base import Tool, _func_has_ctx_param

__all__ = ["ToolRegistry"]

logger = logging.getLogger(__name__)


#: 兼容别名，统一从 base._func_has_ctx_param 来。
_func_has_ctx_param = _func_has_ctx_param


class ToolRegistry:
    '''工具注册表：`register` / `get` / `all` / `contains` / `remove`。

    内部用 dict 保证 O(1) 查找；遍历顺序按注册先后。

    重名时默认 warning 并跳过，只有 `override=True` 才覆盖。
    '''

    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools is not None:
            for t in tools:
                self.register(t)

    # ---- 基本操作 ----

    def register(self, tool: Tool, *, override: bool = False) -> bool:
        '''注册一个工具。返回 True 表示新注册，False 表示跳过（重名未覆盖）。'''
        existing = self._tools.get(tool.name)
        if existing is not None and not override:
            logger.warning(
                "工具 %r 已存在，跳过注册（override=True 可覆盖）", tool.name
            )
            return False
        if existing is not None:
            logger.info("工具 %r 已覆盖", tool.name)
        self._tools[tool.name] = tool
        return True

    def get(self, name: str) -> Optional[Tool]:
        '''按名字查找工具，找不到返回 None。'''
        return self._tools.get(name)

    def remove(self, name: str) -> Optional[Tool]:
        '''移除一个工具，返回被移除的实例（找不到返回 None）。'''
        return self._tools.pop(name, None)

    def contains(self, name: str) -> bool:
        '''是否已注册某名字。'''
        return name in self._tools

    # ---- 批量 ----

    def all(self) -> List[Tool]:
        '''返回所有已注册工具（按注册顺序）。'''
        return list(self._tools.values())

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._tools))
        return f"<ToolRegistry [{names}]>"

    # ------------------------------------------------------------------ #
    # 执行（可选 _ctx 注入）
    # ------------------------------------------------------------------ #

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> str:
        """按名字执行工具，返回统一的字符串结果。

        这是**唯一**会自动注入 ``_ctx`` 的入口。它会检查目标 Tool 的
        ``func`` 签名：

        - 声明了 ``_ctx`` 参数 → 调用时注入 ``ctx``；
        - 没声明 → 行为和直接 ``tool.call(args)`` **完全一致**。

        这样 SummonSubAgentTool 等需要运行时上下文的工具可以拿到 ctx，
        而其它 Tool / 普通函数零改动、零性能影响（inspect 结果会缓存）。

        Args:
            name: 已注册的工具名；不存在时抛 ``KeyError``。
            args: 传给工具的业务参数字典。
            ctx: 运行时上下文（通常是 Workflow 的 ctx 或 Agent._ctx）；
                为 ``None`` 时仍然正常执行，只是 Tool 内部的 ``_ctx``
                会拿到 ``None``。

        Returns:
            工具返回值的字符串形式（和 ``Tool.call`` 一致）。

        Raises:
            KeyError: ``name`` 未注册。
        """
        tool = self._tools[name]
        if _func_has_ctx_param(tool.func):
            return tool.call(args, ctx=ctx)
        return tool.call(args)


# 职责：轻量工具注册表（register/get/all）+ execute 可选 _ctx 注入，
# 供 Agent / Orchestrator / MCP loader 共用
