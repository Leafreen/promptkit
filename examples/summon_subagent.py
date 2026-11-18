"""SummonSubAgentTool 示例：主 Agent 通过 tool_calls 动态召唤子 Agent。

演示的重点：
- 主 Agent 如何挂载 SummonSubAgentTool
- 子 Agent 有独立 Session，不继承主 Agent 的 history
- 三个护栏（max_depth / max_summon / max_total_tokens）在 Tool 内部检查
- 所有错误返回字符串，主 Agent 能看到并自行决定下一步

这个 example 用 FakeClient 脚本化回复，不需要真实 API key 就能跑通。

运行：
    python examples/summon_subagent.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

# 让 promptkit 包可以被导入：把 src/ 放进 sys.path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)

# =====================================================================
# FakeClient：脚本化回复，不连真实 API
# =====================================================================

class _FakeToolCall(SimpleNamespace):
    """模仿 SDK 的 ToolCall：有 id / function / model_dump。"""

    def model_dump(self) -> dict:  # type: ignore[override]
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeCompletions:
    """按预设顺序返回的假 chat.completions.create。"""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeClient 没有更多预设回复了")
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            message = SimpleNamespace(content=reply, tool_calls=None)
        else:
            tc_raw = reply.get("tool_calls") or []
            tool_calls = [
                _FakeToolCall(
                    id=tc.id,
                    type="function",
                    function=SimpleNamespace(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in tc_raw
            ]
            message = SimpleNamespace(
                content=reply.get("content", ""),
                tool_calls=tool_calls or None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=42),
        )


class FakeClient:
    def __init__(self, replies: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(replies))


# =====================================================================
# 构建主 / 子 Agent
# =====================================================================

from promptkit.agent import Agent  # noqa: E402
from promptkit.agent_factory import SubAgentFactory  # noqa: E402
from promptkit.core.tool.subagent import SummonSubAgentTool  # noqa: E402

# 角色模板
factory = SubAgentFactory(
    api_key="sk-fake",
    roles={
        "researcher": "你是市场调研专家。请在 100 字内总结要点。",
        "analyst": "你是数据分析师。请在 100 字内给出洞察。",
        "writer": "你是写作专家。请在 100 字内写出结论。",
    },
)

# 主 Agent 的预设回复（模拟它调用工具再总结）
main_replies: list[Any] = [
    # 第 1 轮：主 Agent 决定召唤 researcher 和 analyst
    {
        "content": None,
        "tool_calls": [
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(
                    name="summon_subagent",
                    arguments='{"role": "researcher", "task": "调研 2026 年 AI Agent 市场规模"}',
                ),
            ),
            SimpleNamespace(
                id="call_2",
                type="function",
                function=SimpleNamespace(
                    name="summon_subagent",
                    arguments='{"role": "analyst", "task": "分析 2026 年 AI Agent 增长趋势"}',
                ),
            ),
        ],
    },
    # 第 2 轮：主 Agent 看到子 Agent 结果，调用 writer 整合
    {
        "content": None,
        "tool_calls": [
            SimpleNamespace(
                id="call_3",
                type="function",
                function=SimpleNamespace(
                    name="summon_subagent",
                    arguments='{"role": "writer", "task": "根据调研和分析结果写一份简报"}',
                ),
            ),
        ],
    },
    # 第 3 轮：主 Agent 给出最终总结
    "好的，我已经完成任务。报告如下：\n\n"
    "## 2026 年 AI Agent 市场简报\n\n"
    "- 市场规模预计 500 亿美元\n"
    "- 年增长率 45%\n"
    "- 主要玩家：OpenAI、Anthropic、国产厂商",
]

main_agent = Agent(
    api_key="sk-fake",
    register_as_default=False,
    start_prompts=[
        {
            "role": "system",
            "content": (
                "你是项目主管。需要专人处理时，调用 summon_subagent。"
                "可用角色：researcher（调研）、analyst（分析）、writer（写作）。"
            ),
        }
    ],
)
main_agent.client = FakeClient(main_replies)

# 给主 Agent 挂 SummonSubAgentTool
summon = SummonSubAgentTool(
    factory,
    max_depth=2,
    max_summon=10,
    max_total_tokens=100000,
    timeout=10.0,  # 示例里短一点
)


# =====================================================================
# 子 Agent 的预设回复
# =====================================================================

def _child_replies() -> list[str]:
    """子 Agent 的 FakeClient 回复池。"""
    return [
        "调研结果：2026 年 AI Agent 市场规模约 500 亿美元。",
        "分析结果：年增长率 45%，主要驱动是企业级应用。",
        "简报结论：AI Agent 将成为下一轮生产力革命的核心技术。",
    ]


def _make_child(role: str) -> Agent:
    """构造一个子 Agent，挂上 FakeClient。"""
    child = factory.create(role)
    child.client = FakeClient(_child_replies())
    return child


# =====================================================================
# 演示：主 Agent 自主召唤子 Agent
# =====================================================================

def main() -> None:
    print("=" * 60)
    print("SummonSubAgentTool 演示：主 Agent 自主召唤子 Agent")
    print("=" * 60)

    # 替换 factory.create 让子 Agent 也用 FakeClient
    original_create = factory.create

    def patched_create(role: str, **kwargs: Any) -> Agent:
        child = original_create(role, **kwargs)
        child.client = FakeClient(_child_replies())
        return child

    factory.create = patched_create  # type: ignore[assignment]

    ctx: dict[str, Any] = {}
    main_agent._ctx = ctx

    # 执行
    print("\n[用户] 帮我调研 2026 年 AI Agent 市场，写一份报告")
    reply = main_agent.ask(
        "帮我调研 2026 年 AI Agent 市场，写一份报告",
        tools=[summon],
        max_tool_rounds=6,
    )

    print(f"\n[主 Agent 最终回复]\n{reply}")
    print(f"\n[ctx 护栏计数] _depth={ctx.get('_depth')}, "
          f"_summon_count={ctx.get('_summon_count')}, "
          f"_total_tokens={ctx.get('_total_tokens')}")
    print(f"\n[trace 记录（{len(ctx.get('_trace', []))} 条）]")
    for i, entry in enumerate(ctx.get("_trace", []), 1):
        print(f"  {i}. [{entry['status']}] {entry['role']} — {entry['task_preview']}")
    print("\n✓ 演示完成：主 Agent 共召唤了 3 个子 Agent（researcher / analyst / writer），"
          "子 Agent 有独立 Session，结果成功回流。")


if __name__ == "__main__":
    main()
