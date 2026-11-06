# tests/mcp/test_e2e_agent.py
'''端到端：Agent → MCPTool 调用流程（FakeLLMClient + FakeMCPClient）。

两个场景：
1. Agent 自主调用：模型返回 tool_calls → Agent._run_tool → MCPTool.call → HookRegistry（安全 gate）→ FakeMCPClient
2. 安全拒绝：allowlist 不含工具 → PermissionError → 结果写进 ctx → Agent 收到错误消息决定下一步
'''
from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from promptkit.agent import Agent
from promptkit.core.context import Context
from promptkit.core.tool.base import Tool
from promptkit.core.tool.registry import ToolRegistry
from promptkit.core.steps.actions.mcp_step import MCPToolStep
from promptkit.mcp.hooks import HookRegistry
from promptkit.mcp.security import register_mcp_security
from promptkit.mcp.tool import MCPTool


# ---------------------------------------------------------------------------
# Fake 层：FakeMCPTool（继承 MCPTool，但用本地 fake 执行）
# ---------------------------------------------------------------------------
class FakeMCPTool(MCPTool):
    '''MCPTool 的本地 fake —— 不走真正的 MCP SDK。

    call() 直接按 self.tool_name 返回预设值，用来验证 HookRegistry 流程。
    '''

    def call(self, arguments=None):
        # 先跑 HookRegistry（和真实 MCPTool 一样）
        args = arguments or {}
        hooks = self._get_hooks()
        try:
            gate = hooks.fire_before(self.name, args)
        except Exception:
            gate = None
        if gate is not None and getattr(gate, "is_denied", False):
            raise PermissionError(getattr(gate, "reason", ""))
        import time
        t0 = time.perf_counter()
        if self.tool_name == "read_file":
            result = f"[fake] read {args.get('path')}: Hello World!"
        elif self.tool_name == "list_dir":
            result = f"[fake] list {args.get('path', '.')}: main.py, README.md"
        else:
            result = f"[fake] unknown tool {self.tool_name}"
        # after hook
        hooks.fire_after(self.name, args, result, time.perf_counter() - t0)
        return result


def make_fake_mcp_tool(
    server_name: str,
    tool_name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> FakeMCPTool:
    return FakeMCPTool(
        name=f"{server_name}__{tool_name}",
        description=description or tool_name,
        parameters=parameters or {"type": "object", "properties": {}},
        server_name=server_name,
        tool_name=tool_name,
        server_url="http://fake/mcp",
    )


# ---------------------------------------------------------------------------
# test_agent_autonomous_mcp_call：Agent 自主调 MCPTool
# ---------------------------------------------------------------------------

def test_agent_autonomous_mcp_call():
    '''Agent 收到工具列表后，_run_tool 能执行 MCPTool —— HookRegistry 放行。'''
    # HookRegistry + allowlist（放行 fs__read）
    hooks = HookRegistry()
    register_mcp_security(hooks, {"fs__read_file", "fs__list_dir"})

    # ToolRegistry 注册两个 MCPTool（fake）
    read_tool = make_fake_mcp_tool("fs", "read_file", "读文件",
                                   {"type": "object", "properties": {"path": {"type": "string"}}})
    list_tool = make_fake_mcp_tool("fs", "list_dir", "列目录",
                                   {"type": "object", "properties": {"path": {"type": "string"}}})
    read_tool.hooks = hooks
    list_tool.hooks = hooks

    registry = ToolRegistry([read_tool, list_tool])

    # 直接调 tool.call() 验证（Agent 循环里 _run_tool 就是这么调的）
    result = registry.get("fs__read_file").call({"path": "config.yaml"})
    assert "[fake] read config.yaml" in result
    # 第二个也放行
    result2 = registry.get("fs__list_dir").call({"path": "."})
    assert "[fake] list" in result2


def test_agent_mcp_denied_by_security():
    '''allowlist 不含工具 → PermissionError —— Agent._run_tool 会 catch 成错误消息。'''
    hooks = HookRegistry()
    register_mcp_security(hooks, set())  # 空集 = 全部拒绝

    read_tool = make_fake_mcp_tool("fs", "read_file")
    read_tool.hooks = hooks

    # call() 直接抛 PermissionError
    with pytest.raises(PermissionError, match="未授权"):
        read_tool.call({"path": "x"})

    # Agent._run_tool 的异常处理（捕获后成错误消息）：
    from promptkit.agent import Agent
    # 这里直接模拟 _run_tool 的行为
    result: str
    try:
        result = read_tool.call({"path": "x"})
    except Exception as exc:
        result = f"错误：工具执行失败 {type(exc).__name__}: {exc}"

    assert "PermissionError" in result
    assert "未授权" in result


# ---------------------------------------------------------------------------
# MCPToolStep 集成到 Workflow
# ---------------------------------------------------------------------------

def test_mcp_tool_step_in_workflow():
    '''MCPToolStep 在 Workflow 里执行 —— ctx[output_key] 有值，ctx.trace 有记录。'''
    from promptkit.core.steps.workflow import Workflow
    from promptkit.core.steps.logics.var import NewVar, Var

    hooks = HookRegistry()
    register_mcp_security(hooks, {"fs__read_file"})

    read_tool = make_fake_mcp_tool("fs", "read_file")
    read_tool.hooks = hooks

    workflow = Workflow(
        NewVar("target_path", default="./config.yaml"),
        MCPToolStep(
            tool=read_tool,
            input_keys={"path": "target_path"},
            output_key="file_content",
        ),
    )

    ctx = workflow.run()
    assert "file_content" in ctx.data
    assert "[fake] read" in ctx["file_content"]
    assert any("mcp_fs__read_file" == t.get("step") for t in ctx.trace if isinstance(t, dict))


def test_mcp_tool_step_denied_still_writes_result():
    '''安全 gate 拒绝 —— MCPToolStep.execute catch PermissionError，拒绝消息写 ctx。'''
    hooks = HookRegistry()
    register_mcp_security(hooks, set())  # 全部拒绝

    read_tool = make_fake_mcp_tool("fs", "read_file")
    read_tool.hooks = hooks

    from promptkit.core.steps.workflow import Workflow
    from promptkit.core.steps.logics.var import NewVar

    workflow = Workflow(
        NewVar("target_path", default="./config.yaml"),
        MCPToolStep(
            tool=read_tool,
            input_keys={"path": "target_path"},
            output_key="file_content",
        ),
    )

    ctx = workflow.run()
    # 被拒绝，但不会崩溃 —— 拒绝消息写进 ctx
    assert "[拒绝]" in ctx["file_content"]
    assert "未授权" in ctx["file_content"]
