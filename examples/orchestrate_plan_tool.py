"""`PlanTool`：模型自己规划，结果落进 Context（离线可跑）。

    python examples/orchestrate_plan_tool.py

两个用法：
    1. 只给模型工具 —— 它自己决定要不要拆（`max_calls` 护栏）
    2. 放进工作流 —— `PlanWithTool` 规划，`ExecutePlan` 接着跑
"""
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.core import Var, Workflow  # noqa: E402
from promptkit.orchestrate import (  # noqa: E402
    ExecutePlan,
    Plan,
    PlanTool,
    PlanWithTool,
    Task,
    TemplatePlanner,
)


class FakeAgent:
    """假 agent：默认拿到工具就调一次；`call_tools=False` 时只回话。"""

    vendor = "fake"
    model = "scripted"

    def __init__(self, reply="好的", arguments=None, call_tools=True):
        self.reply = reply
        self.arguments = arguments or {"task": "写一份 AI 报告"}
        self.call_tools = call_tools
        self.tool_calls = []

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        if tools and self.call_tools:
            self.tool_calls.append(tools[-1].call(self.arguments))
        return self.reply


TEMPLATE = TemplatePlanner([
    Task("research", "调研 {{topic}} 的现状"),
    Task("write", "根据 {{research}} 写报告", depends_on=("research",)),
])

# ============================================================
# 1. 裸工具：模型调一次，plan 就进了 context
# ============================================================
print("=" * 58)
print("1. PlanTool：模型自己调，结果写进 Context")
print("=" * 58)

context = {}
tool = PlanTool(planner=TEMPLATE, context=context, max_calls=2)
agent = FakeAgent(reply="我先拆一下")
agent.ask("这活儿怎么拆？", tools=[tool])

print(f"工具返回给模型:\n{agent.tool_calls[0]}\n")
print(f"Context 里的 plan: {context['plan'].names}")
print(f"已用次数: {tool.used}/{tool.max_calls}")

assert isinstance(context["plan"], Plan)
assert context["plan"].names == ["research", "write"]

# 次数护栏：不会让模型无限拆
again = tool.call({"task": "再拆一次"})
tool.call({"task": "再拆一次"})
print(f"第三次调用: {tool.call({'task': '再拆一次'})}")
assert "最多规划 2 次" in tool.call({"task": "再拆一次"})

# ============================================================
# 2. 工作流：PlanWithTool -> ExecutePlan
# ============================================================
print()
print("=" * 58)
print("2. PlanWithTool + ExecutePlan：规划完直接跑")
print("=" * 58)

planner_agent = FakeAgent(reply="规划好了")
ctx = Workflow(
    PlanWithTool(
        prompt="先想清楚怎么做：{{usr_input}}",
        planner=TEMPLATE,
        max_tasks=5,
    ),
    ExecutePlan(
        plan=Var("plan"),
        agent_factory=lambda task: FakeAgent(reply=f"{task.name} 的产出"),
    ),
).run(agent=planner_agent, usr_input="写一份 AI 报告", topic="AI Agent")

print(f"模型回复: {ctx['plan_reply']}")
print(f"计划: {ctx['plan'].names}")
print(f"任务结果: {[(r['name'], r['status']) for r in ctx['task_results']]}")
print(f"trace: {ctx.trace}")

assert [r["name"] for r in ctx["task_results"]] == ["research", "write"]
assert ctx["plan_result"].ok

# ============================================================
# 3. 模型选择不规划（required=False）
# ============================================================
print()
print("=" * 58)
print("3. required=False：模型可以只回话，不拆任务")
print("=" * 58)

ctx2 = Workflow(
    PlanWithTool(prompt="看情况规划", planner=TEMPLATE, required=False)
).run(agent=FakeAgent(reply="这题一步就够", call_tools=False))
print(f"回复: {ctx2['plan_reply']}  有没有 plan: {'plan' in ctx2}")
assert "plan" not in ctx2

print()
print("ALL PLAN TOOL CHECKS PASSED")
