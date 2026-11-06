# tests/mcp/test_mcp_step.py
'''MCPToolStep 单测 —— 显式调用 + Context 交互 + trace 脱敏。'''
from __future__ import annotations

import pytest

from promptkit.core.context import Context
from promptkit.core.tool.base import Tool
from promptkit.core.steps.actions.mcp_step import MCPToolStep, _sanitize_args


# ---------------------------------------------------------------------------
# FakeTool — 假 Tool，可预设返回值和记录调用
# ---------------------------------------------------------------------------
import dataclasses

@dataclasses.dataclass
class FakeTool(Tool):
    call_args: list = dataclasses.field(default_factory=list)

    def call(self, arguments=None):
        self.call_args.append(arguments or {})
        return f"[fake {self.name}] args={arguments}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mcp_step_writes_context():
    '''执行后 ctx[output_key] 有值。'''
    tool = FakeTool(
        name="fs__read", description="",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        func=lambda **_: None,
    )
    ctx = Context(data={"target_path": "./config.yaml"})
    step = MCPToolStep(
        tool=tool,
        input_keys={"path": "target_path"},
        output_key="file_content",
    )
    step.execute(ctx)

    assert "file_content" in ctx.data
    assert "[fake fs__read]" in ctx["file_content"]
    # 工具收到正确参数
    assert tool.call_args == [{"path": "./config.yaml"}]


def test_mcp_step_writes_trace():
    '''ctx.trace 有 {step, tool, args, duration} 记录。'''
    tool = FakeTool(
        name="fs__read", description="",
        parameters={"type": "object", "properties": {}},
        func=lambda **_: None,
    )
    ctx = Context(data={"p": "/x"})
    step = MCPToolStep(tool=tool, input_keys={"path": "p"}, output_key="out")
    step.execute(ctx)

    assert len(ctx.trace) == 1
    entry = ctx.trace[0]
    assert entry["tool"] == "fs__read"
    assert entry["args"] == {"path": "/x"}
    assert entry["duration"] >= 0
    assert entry["step"] == "mcp_fs__read"


def test_mcp_step_default_name():
    '''没给 name —— 默认 "mcp_{tool.name}"。'''
    tool = FakeTool(
        name="filesystem__read_file", description="",
        parameters={"type": "object", "properties": {}},
        func=lambda **_: None,
    )
    step = MCPToolStep(tool=tool, input_keys={}, output_key="out")
    assert step.name == "mcp_filesystem__read_file"


def test_mcp_step_missing_context_key_raises():
    '''input_keys 里的 Context 变量不存在 —— 抛 KeyError。'''
    tool = FakeTool(
        name="fs__t", description="",
        parameters={"type": "object", "properties": {}},
        func=lambda **_: None,
    )
    ctx = Context(data={})  # 空！没有 "p"
    step = MCPToolStep(tool=tool, input_keys={"path": "nonexistent_key"}, output_key="out")

    with pytest.raises(KeyError, match="nonexistent_key"):
        step.execute(ctx)


def test_mcp_step_sanitize_args():
    '''_sanitize_args 把密钥/token 值换成 ***。'''
    args = {
        "path": "/etc/hosts",
        "api_key": "sk-xxx",
        "user_token": "secret-123",
        "auth": "Bearer abc",
        "description": "public info",
    }
    out = _sanitize_args(args)
    assert out["path"] == "/etc/hosts"
    assert out["api_key"] == "***"
    assert out["user_token"] == "***"
    assert out["auth"] == "***"
    assert out["description"] == "public info"


def test_mcp_step_requires_tool_and_output_key():
    '''构造时 tool=None 或 output_key="" —— __post_init__ 抛 ValueError。'''
    with pytest.raises(ValueError, match="tool"):
        MCPToolStep(tool=None, input_keys={}, output_key="")

    tool = FakeTool(
        name="t", description="",
        parameters={"type": "object", "properties": {}},
        func=lambda **_: None,
    )
    with pytest.raises(ValueError, match="output_key"):
        MCPToolStep(tool=tool, input_keys={}, output_key="")
