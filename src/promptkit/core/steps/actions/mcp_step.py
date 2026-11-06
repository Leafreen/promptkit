# core/steps/actions/mcp_step.py
'''在工作流中显式调用 MCP 工具的 Step —— 不经过 Agent 循环，直接按映射参数。

为什么要单独一个 Step：
- Agent 自主调用 MCP 工具是"让模型自己决定调什么"
- MCPToolStep 是"工作流硬编排，明确调哪个工具、输入参数从哪来、结果写到哪"
- 两者互补，都走 MCPTool.call()（同一个安全 gate）
'''
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from promptkit.core.tool.base import Tool
from promptkit.core.steps.interface import Step

if TYPE_CHECKING:
    from promptkit.core.context import Context

__all__ = ["MCPToolStep"]

logger = logging.getLogger(__name__)


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    '''脱敏：把明显是密钥/密码的参数值替换成 ***。'''
    sensitive_keys = {"key", "api_key", "token", "secret", "password", "auth"}
    return {
        k: "***" if any(s in k.lower() for s in sensitive_keys) else v
        for k, v in args.items()
    }


@dataclass
class MCPToolStep(Step):
    '''在工作流中显式调用一个工具（通常是 MCPTool），结果写 Context。

    Args:
        tool: 要调用的 Tool（任何 Tool 都可以，但主要用于 MCPTool）。
        input_keys: `{工具参数名: Context 变量名}` 映射。比如 `{"path": "target_path"}`
            表示调工具时 `path=ctx["target_path"]`。
        output_key: 结果写到 `ctx[output_key]`。
        name: Step 名字（默认 `f"mcp_{tool.name}"`），用于 `ctx.trace`。

    示例：

        workflow = Workflow(
            NewVar("target_path", default="./config.yaml"),
            MCPToolStep(
                tool=registry.get("filesystem__read_file"),
                input_keys={"path": "target_path"},
                output_key="file_content",
            ),
            Echo(Var("file_content")),
        )
    '''

    tool: Any = None  # Tool 实例（主要是 MCPTool，但任何 Tool 都行）
    input_keys: Dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if self.tool is None:
            raise ValueError("MCPToolStep 需要 tool 实例")
        if not self.output_key:
            raise ValueError(f"MCPToolStep {self.name!r} 需要 output_key")
        if not self.name:
            self.name = f"mcp_{getattr(self.tool, 'name', 'tool')}"

    def execute(self, ctx: "Context") -> None:
        '''从 ctx 按 input_keys 收集参数 → 调 tool.call() → 写 ctx[output_key]。

        执行过程和结果（脱敏）写进 ctx.trace。
        '''
        # 收集参数
        args: Dict[str, Any] = {}
        for tool_param, ctx_key in self.input_keys.items():
            if ctx_key not in ctx:
                raise KeyError(
                    f"MCPToolStep {self.name!r}: Context 里没有变量 {ctx_key!r}，"
                    f"无法给工具参数 {tool_param!r} 赋值"
                )
            args[tool_param] = ctx[ctx_key]

        # 执行
        t0 = time.perf_counter()
        try:
            result = self.tool.call(args)
        except PermissionError as exc:
            # 安全 gate 拒绝 —— 结果写进 ctx，错误信息让用户/模型看到
            result = f"[拒绝] {exc}"
            logger.warning("MCPToolStep %s 被拒绝：%s", self.name, exc)

        duration = time.perf_counter() - t0

        # 写 Context
        ctx[self.output_key] = result

        # 写 trace（脱敏）
        ctx.trace.append({
            "step": self.name,
            "tool": getattr(self.tool, "name", "<unknown>"),
            "args": _sanitize_args(args),
            "duration": round(duration, 4),
        })
# 职责：工作流里显式调用 Tool 的 Step —— 按 input_keys 从 Context 收集参数，结果写 output_key，trace 记录脱敏 args 和 duration
