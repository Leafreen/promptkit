# core/tool/registry.py
'''轻量工具注册表：集中管理 Tool，供 Agent 和 MCP 发现层共用。

为什么单独一个 registry：
- Agent 只收 List[Tool]，没有中央查找点
- MCP loader 需要 `registry.register(tool)` + `registry.get(name)`
- 后续 HookRegistry（第二轮）要按名字审计，registry 是天然索引
'''
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, cast

from promptkit.core.tool.base import Tool

__all__ = ["ToolRegistry"]

logger = logging.getLogger(__name__)


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
# 职责：轻量工具注册表（register/get/all），供 Agent 和 MCP 发现层共用
