# examples/mcp_demo.py
'''MCP Client 完整演示：三种场景，全 Fake（不连真实 Server）。

场景：
1. ToolRegistry 怎么装 MCP 工具
2. Agent 自主调用（模型决定调什么 —— 安全 gate 放行）
3. Workflow 显式调用（Step 编排，工具被 Security gate 拒绝 → 消息写进 ctx）
4. allowlist 空 → 默认全部拒绝（安全默认）

跑法：`python examples/mcp_demo.py`
'''
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# 假 MCP 工具（真实项目里用 load_mcp_tools 从 Server 发现）
# ---------------------------------------------------------------------------
from promptkit.mcp.tool import MCPTool


def make_fake_fs_tool():
    '''本地文件系统 Server —— 两个工具。'''

    @dataclass
    class FakeFSTool(MCPTool):
        '''Fake MCPTool —— call() 不走真正的 MCP SDK，按 tool_name 硬返回。'''

        def call(self, arguments=None):
            args = arguments or {}
            # 先跑 HookRegistry（和真实 MCPTool 完全一样）
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
                path = args.get("path", "")
                return f"[fake-fs] 读取 {path}：\nname: promptkit\nversion: 0.2.0\nlicense: MIT"
            if self.tool_name == "list_dir":
                return "[fake-fs] ./ 下有：src, tests, examples, pyproject.toml"
            return f"[fake-fs] 未知工具 {self.tool_name}"

    return [
        FakeFSTool(
            name="fs__read_file", description="读取文件内容",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            server_name="fs", tool_name="read_file",
            server_url="http://fake-fs/mcp",
        ),
        FakeFSTool(
            name="fs__list_dir", description="列出目录下的文件",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="fs", tool_name="list_dir",
            server_url="http://fake-fs/mcp",
        ),
    ]


def make_fake_github_tool():
    '''GitHub Server —— 两个工具。'''

    @dataclass
    class FakeGithubTool(MCPTool):
        def call(self, arguments=None):
            args = arguments or {}
            hooks = self._get_hooks()
            gate = hooks.fire_before(self.name, args)
            if gate is not None and getattr(gate, "is_denied", False):
                raise PermissionError(getattr(gate, "reason", ""))

            if self.tool_name == "search_issues":
                return (
                    "[fake-gh] 找到 3 个 open issue：\n"
                    "  #42  MCP Tool HookRegistry 集成问题\n"
                    "  #38  多模态 Prompt 类型注解补全\n"
                    "  #27  安全 gate 审计日志脱敏"
                )
            if self.tool_name == "create_issue":
                return f"[fake-gh] issue #99 已创建：{args.get('title', '')}"
            return f"[fake-gh] 未知工具 {self.tool_name}"

    return [
        FakeGithubTool(
            name="gh__search_issues", description="搜索 GitHub issues",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="gh", tool_name="search_issues",
            server_url="http://fake-gh/mcp",
        ),
        FakeGithubTool(
            name="gh__create_issue", description="创建新 issue",
            parameters={"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}},
            server_name="gh", tool_name="create_issue",
            server_url="http://fake-gh/mcp",
        ),
    ]


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

def main():
    from promptkit.core.tool.registry import ToolRegistry
    from promptkit.mcp.hooks import HookRegistry, get_default_hook_registry, set_default_hook_registry
    from promptkit.mcp.security import register_mcp_security

    print("=" * 60)
    print("场景 0：HookRegistry 安全 gate —— 默认拒绝一切")
    print("=" * 60)

    hooks = HookRegistry()
    # 注册空 allowlist —— 拒绝全部
    register_mcp_security(hooks, set())

    fs_tools = make_fake_fs_tool()
    gh_tools = make_fake_github_tool()

    # 绑定 hook
    for t in fs_tools + gh_tools:
        t.hooks = hooks

    # 尝试调用 —— 全部 PermissionError
    for t in fs_tools + gh_tools:
        try:
            t.call({"path": "/x"})
            print(f"  ✗ {t.name}: 不应放行但放了")
        except PermissionError as exc:
            print(f"  ✓ {t.name}: 被安全层拒绝 — {exc}")

    print()
    print("=" * 60)
    print("场景 1：ToolRegistry + 部分 allowlist 放行")
    print("=" * 60)

    # 新 HookRegistry —— 只放行 3 个工具
    hooks2 = HookRegistry()
    register_mcp_security(hooks2, {"fs__read_file", "fs__list_dir", "gh__search_issues"})

    registry = ToolRegistry()
    for t in fs_tools + gh_tools:
        t.hooks = hooks2
        registry.register(t)

    print(f"注册后 registry 有 {len(registry)} 个工具：")
    for t in registry.all():
        print(f"  • {t.name}: {t.description}")

    print()
    print("逐个调用：")
    for name in ["fs__read_file", "fs__list_dir", "gh__search_issues", "gh__create_issue"]:
        tool = registry.get(name)
        try:
            result = tool.call({"path": "./config.yaml", "query": "mcp"})
            print(f"  ✓ {name}: 放行 — {result[:60]}...")
        except PermissionError as exc:
            print(f"  ✗ {name}: 拒绝 — {exc}")

    print()
    print("=" * 60)
    print("场景 2：Workflow 显式调用（MCPToolStep）")
    print("=" * 60)

    from promptkit.core.steps.workflow import Workflow
    from promptkit.core.steps.logics.var import NewVar, Var
    from promptkit.core.steps.actions.mcp_step import MCPToolStep

    workflow = Workflow(
        NewVar("target_path", default="./pyproject.toml"),
        NewVar("issue_title", default="文件不存在问题"),
        # ✅ 在 allowlist 里 —— 放行
        MCPToolStep(
            tool=registry.get("fs__read_file"),
            input_keys={"path": "target_path"},
            output_key="file_content",
        ),
        # ❌ 不在 allowlist 里 —— PermissionError 被 Step catch，写进 ctx
        MCPToolStep(
            tool=registry.get("gh__create_issue"),
            input_keys={"title": "issue_title"},
            output_key="issue_result",
        ),
    )

    ctx = workflow.run()

    print(f"  file_content: {ctx['file_content'][:50]}...")
    print(f"  issue_result: {ctx['issue_result']}")
    print(f"  trace 记录数: {len([t for t in ctx.trace if isinstance(t, dict)])}")
    for t in ctx.trace:
        if isinstance(t, dict):
            print(f"    - {t['step']}: {t['tool']} (duration={t['duration']}s)")

    print()
    print("=" * 60)
    print("场景 3：异步 load_mcp_tools（真实 MCP Server 流程）")
    print("=" * 60)

    print("  from promptkit.mcp import load_mcp_tools, register_mcp_security")
    print()
    print("  asyncio.run(load_mcp_tools(registry, 'http://localhost:3000/mcp', 'filesystem'))")
    print("  # → 发现 fs__read_file, fs__list_dir 等工具，已注册到 ToolRegistry")
    print()
    print("  register_mcp_security(hooks, {'fs__read_file', 'fs__list_dir'})")
    print("  # → allowlist 设好，不在列表里的调用自动 PermissionError")
    print()
    print("  agent = Agent(vendor='doubao')")
    print("  agent.ask('读一下 config.yaml', tools=registry.all())")
    print("  # → Agent 循环零改动，MCPTool 和内置 Tool 走同一套 tool_calls 流程")

    print()
    print("✓ MCP Client 集成演示完成")


if __name__ == "__main__":
    main()
