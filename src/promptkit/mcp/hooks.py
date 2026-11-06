# mcp/hooks.py
'''MCP 工具调用 Hook 系统 —— before_tool_call / after_tool_call。

设计：

- 轻量，不混进 promptkit 现有 EventBus（EventBus 是对话级别的事件，
  这里是工具调用级别的，粒度不同）
- MCPTool.call() 内部调 HookRegistry.fire_before / fire_after，
  Agent 循环零改动
- 全局单例 `HookRegistry.default`，或每个 Tool 可单独绑定

Hook 签名：

    before_hook(tool_name: str, args: dict) -> dict | None
    after_hook(tool_name: str, args: dict, result: str, duration: float) -> None

before_hook 可以返回 `{"decision": "deny", "reason": "..."}` —— 调用方（MCPTool）
看到 deny 就抛 PermissionError，不再执行工具。
'''
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ToolHookResult",
    "HookRegistry",
    "get_default_hook_registry",
    "set_default_hook_registry",
]

logger = logging.getLogger(__name__)

# 默认单例 —— MCPTool.call() 在没显式绑定 registry 时用这个
_default_registry: Optional["HookRegistry"] = None


def get_default_hook_registry() -> "HookRegistry":
    '''拿全局默认 HookRegistry（首次调用时构造）。'''
    global _default_registry
    if _default_registry is None:
        _default_registry = HookRegistry()
    return _default_registry


def set_default_hook_registry(registry: "HookRegistry") -> None:
    '''替换全局默认 HookRegistry（测试用）。'''
    global _default_registry
    _default_registry = registry


@dataclass
class ToolHookResult:
    '''before_hook 的返回值。

    - `{"decision": "allow"}` 或 None — 放行
    - `{"decision": "deny", "reason": "..."}` — 拒绝
    '''

    decision: str = "allow"
    reason: str = ""

    @property
    def is_denied(self) -> bool:
        return self.decision == "deny"


# Hook 类型别名
BeforeHook = Callable[[str, Dict[str, Any]], Optional[ToolHookResult]]
AfterHook = Callable[[str, Dict[str, Any], str, float], None]


class HookRegistry:
    '''工具调用钩子注册表。

    - `register_before(name, hook)` —— MCPTool.call() 之前跑
    - `register_after(name, hook)` —— MCPTool.call() 之后跑
    - `fire_before(name, args)` —— 触发所有 before hooks，返回第一个 deny 或最后一个非 None 结果
    - `fire_after(name, args, result, duration)` —— 触发所有 after hooks，忽略异常
    '''

    def __init__(self) -> None:
        self._before: List[BeforeHook] = []
        self._after: List[AfterHook] = []

    # ---- 注册 ----

    def register_before(self, hook: BeforeHook) -> None:
        '''注册一个 before_tool_call hook。'''
        self._before.append(hook)

    def register_after(self, hook: AfterHook) -> None:
        '''注册一个 after_tool_call hook。'''
        self._after.append(hook)

    # ---- 触发 ----

    def fire_before(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[ToolHookResult]:
        '''依次调所有 before hooks。第一个返回 deny 的直接短路返回。

        Hook 抛异常会被捕获、打 warning、继续下一个 —— 不让一个坏 hook 挡住整个调用。
        '''
        for hook in self._before:
            try:
                result = hook(tool_name, args)
            except Exception as exc:  # noqa: BLE001
                logger.warning("before hook %r 抛异常，跳过：%s", hook, exc)
                continue
            if result is not None and getattr(result, "is_denied", False):
                return result
        return None  # 默认 allow

    def fire_after(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str,
        duration: float,
    ) -> None:
        '''依次调所有 after hooks。每个 hook 的异常被捕获。'''
        for hook in self._after:
            try:
                hook(tool_name, args, result, duration)
            except Exception as exc:  # noqa: BLE001
                logger.warning("after hook %r 抛异常，跳过：%s", hook, exc)

    # ---- 清理 ----

    def clear(self) -> None:
        self._before.clear()
        self._after.clear()

    def __repr__(self) -> str:
        return f"<HookRegistry before={len(self._before)} after={len(self._after)}>"
# 职责：MCP 工具调用钩子注册表（before_tool_call / after_tool_call），支持默认单例
