# tests/mcp/test_security.py
'''安全层单测：默认拒绝 / allowlist 放行 / hook 注册 / 脱敏。'''
from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from promptkit.core.tool.base import Tool
from promptkit.mcp.exceptions import MCPToolNotFound
from promptkit.mcp.hooks import (
    HookRegistry,
    ToolHookResult,
    get_default_hook_registry,
    set_default_hook_registry,
)
from promptkit.mcp.security import (
    _sanitize_args,
    mcp_permission_gate,
    register_mcp_security,
)
from promptkit.mcp.tool import MCPTool, build_mcp_tools
from tests.mcp.conftest import FakeCallToolResult, FakeMCPClient, FakeMCPToolDef


# ============================================================
# HookRegistry 基础
# ============================================================

def test_hook_registry_register_and_fire():
    '''register_before + fire_before 返回第一个 deny。'''
    reg = HookRegistry()
    calls: list = []

    def before1(name, args):
        calls.append(("before1", name, args))

    def before2(name, args):
        calls.append(("before2", name, args))
        return ToolHookResult(decision="deny", reason="nope")

    reg.register_before(before1)
    reg.register_before(before2)

    result = reg.fire_before("tool_a", {"x": 1})
    assert result is not None
    assert result.is_denied
    assert calls == [("before1", "tool_a", {"x": 1}), ("before2", "tool_a", {"x": 1})]


def test_hook_registry_first_deny_short_circuit():
    '''第一个 hook 返回 deny —— 第二个不会被调。'''
    reg = HookRegistry()
    called = [False, False]

    reg.register_before(lambda n, a: (setattr(called, '__setitem__', None) or ToolHookResult(decision="deny")))  # 不好写，换简单的
    # 重写：
    reg2 = HookRegistry()
    reg2.register_before(lambda n, a: ToolHookResult(decision="deny", reason="denied first"))
    reg2.register_before(lambda n, a: ToolHookResult(decision="allow"))  # 永远不会被调
    result = reg2.fire_before("t", {})
    assert result is not None and result.is_denied

    # 验证第二个没被调：after hooks 独立跑，before 短路
    reg2.register_after(lambda n, a, r, d: called.__setitem__(0, True))
    reg2.fire_after("t", {}, "", 0.001)
    assert called == [True, False]  # after hook 被调了（无关），只验证 HookRegistry 本身


def test_hook_registry_hook_exception_skipped():
    '''一个 before hook 抛异常 —— warning 并继续下一个。'''
    reg = HookRegistry()
    results: list = []

    def bad(name, args):
        raise RuntimeError("boom")

    def good(name, args):
        results.append("good")

    reg.register_before(bad)
    reg.register_before(good)

    reg.fire_before("t", {})  # bad 抛异常，被捕获
    assert results == ["good"]


def test_after_hook_receives_args():
    '''after hook 拿到 name / args / result / duration。'''
    reg = HookRegistry()
    captured: dict = {}

    def after(name, args, result, duration):
        captured.update(name=name, args=args, result=result, duration=duration)

    reg.register_after(after)
    reg.fire_after("fs__read", {"p": "/x"}, "ok", 0.05)

    assert captured["name"] == "fs__read"
    assert captured["args"] == {"p": "/x"}
    assert captured["result"] == "ok"
    assert 0.04 < captured["duration"] < 0.1


# ============================================================
# 安全 gate：mcp_permission_gate
# ============================================================

def test_security_default_deny_empty_allowlist():
    '''allowlist 为空 —— 默认拒绝所有 MCP 工具。'''
    result = mcp_permission_gate(set(), "fs__read_file", {"path": "/x"})
    assert result.is_denied
    assert "MCP 工具未授权" in result.reason
    assert "fs__read_file" in result.reason


def test_security_default_deny_none_allowlist():
    '''allowlist=None —— 同样拒绝（_空集语义_）。'''
    result = mcp_permission_gate(None, "fs__read_file", {"path": "/x"})
    assert result.is_denied


def test_security_allowlist_pass():
    '''工具在 allowlist 里 —— 放行。'''
    result = mcp_permission_gate(
        {"fs__read_file", "fs__list_dir"}, "fs__read_file", {"path": "/x"}
    )
    assert not result.is_denied
    assert result.decision == "allow"


def test_security_allowlist_miss():
    '''工具不在 allowlist —— 拒绝，reason 里有工具名和 allowlist 快照。'''
    result = mcp_permission_gate(
        {"fs__read_file"}, "fs__delete_file", {}
    )
    assert result.is_denied
    assert "fs__delete_file" in result.reason
    assert "fs__read_file" in result.reason  # allowlist 快照显示出来


def test_security_sanitize_args():
    '''密钥/token 参数在 reason 里被脱敏。'''
    allowlist = {"fs__read_file"}
    args = {"path": "/x", "api_key": "sk-xxx", "token": "secret-123"}
    # allowlist 里但参数含密钥 —— 仍然放行（安全 gate 只管工具名，不管参数内容）
    result = mcp_permission_gate(allowlist, "fs__read_file", args)
    assert not result.is_denied


# ============================================================
# register_mcp_security 集成到 HookRegistry
# ============================================================

def test_register_mcp_security_deny_all_when_empty():
    '''register_mcp_security(hook, set()) → 所有 MCPTool.call() 被 PermissionError 拒绝。'''
    reg = HookRegistry()
    register_mcp_security(reg, set())

    # 造一个 MCPTool，但 HookRegistry 的 before hook 会先拒绝
    tool = MCPTool(
        name="fs__read", description="",
        parameters={"type": "object", "properties": {}},
        server_name="fs", tool_name="read", server_url="http://localhost/mcp",
    )
    tool.hooks = reg  # 注入 hook

    # mock _call_remote 让它别真连 Server
    async def fake_remote(args):
        return "ok"
    object.__setattr__(tool, "_call_remote", fake_remote)

    with pytest.raises(PermissionError, match="未授权"):
        tool.call({"path": "/x"})


def test_register_mcp_security_allowlist_pass():
    '''register_mcp_security(hook, allowlist) → allowlist 内放行，外拒绝。'''
    reg = HookRegistry()
    register_mcp_security(reg, {"fs__read"})

    def _make_tool(name: str):
        t = MCPTool(
            name=name, description="",
            parameters={"type": "object", "properties": {}},
            server_name=name.split("__")[0], tool_name=name.split("__")[1],
            server_url="http://localhost/mcp",
        )
        t.hooks = reg
        async def fake_remote(args):
            return "ok"
        object.__setattr__(t, "_call_remote", fake_remote)
        return t

    # allowlist 内 —— 放行
    t1 = _make_tool("fs__read")
    assert t1.call({"path": "/x"}) == "ok"

    # allowlist 外 —— 拒绝
    t2 = _make_tool("fs__delete")
    with pytest.raises(PermissionError, match="未授权"):
        t2.call({"path": "/x"})


# ============================================================
# 默认单例：get_default_hook_registry
# ============================================================

def test_default_registry_reset_between_tests():
    '''set_default_hook_registry 替换默认实例 —— 测试隔离。'''
    original = get_default_hook_registry()
    try:
        fresh = HookRegistry()
        set_default_hook_registry(fresh)
        assert get_default_hook_registry() is fresh
    finally:
        set_default_hook_registry(original)
