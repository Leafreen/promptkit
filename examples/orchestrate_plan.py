"""plan（只规划、不执行）的完整示例 —— 离线可跑，不需要 API Key。

    python examples/orchestrate_plan.py

覆盖 4 种用法：
    1. action="plan" + DSL 自带 tasks   —— 最简，直接出波次（waves）
    2. PlanSpec.from_dsl 显式解析       —— 拆开用：to_plan / to_config / to_dsl
    3. LLMPlanner 让模型拆任务          —— 用假 agent 模拟，演示结构化输出
    4. 错误处理                         —— 环 / 未知键 / 空计划的错误码
"""
from pathlib import Path
import json
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.orchestrate import (  # noqa: E402
    ErrorCode,
    LLMPlanner,
    Plan,
    PlanSpec,
    Request,
    Task,
    handle,
)

# ============================================================
# 1. 最简形态：DSL 里已经写好了 tasks，直接要计划
# ============================================================
print("=" * 60)
print("1. action='plan' + DSL 自带 tasks（不执行，只出计划）")
print("=" * 60)

SPEC = {
    "task": "调研并写一份 AI Agent 行业报告",
    "agents": {
        "researcher": "你是市场调研专家，只给事实与数据来源。",
        "writer": "你是资深报告撰写者。",
    },
    "policy": {"max_concurrency": 2, "max_tasks": 10},
    "tasks": [
        {
            "name": "research",
            "description": "调研 {{topic}} 的市场规模与主要玩家",
            "role": "researcher",
            "output_key": "research",
        },
        {
            "name": "outline",
            "description": "根据 {{research}} 拟定报告提纲",
            "role": "writer",
            "depends_on": ["research"],
            "output_key": "outline",
        },
        {
            "name": "write",
            "description": "根据 {{outline}} 写出完整报告",
            "role": "writer",
            "depends_on": ["outline"],
        },
    ],
}

response = handle(
    Request(action="plan", spec=SPEC, context={"topic": "AI Agent"})
)

print(f"ok      : {response.ok}")
print(f"code    : {response.code.value}（HTTP {response.status}）")
print(f"任务数  : {len(response.data['tasks'])}")
for wave_index, wave in enumerate(response.data["waves"], start=1):
    print(f"第 {wave_index} 波（可并行）: {wave}")

assert response.ok
assert response.data["waves"] == [["research"], ["outline"], ["write"]]

# ============================================================
# 2. 显式解析：PlanSpec -> Plan / Config / dict
# ============================================================
print()
print("=" * 60)
print("2. PlanSpec.from_dsl：拆开用，想拿什么拿什么")
print("=" * 60)

spec = PlanSpec.from_dsl(SPEC)
print(f"PlanSpec     : {spec}")

plan: Plan = spec.to_plan()
print(f"Plan 任务数  : {len(plan)}")
for task in plan:
    print(f"  - {task.name:<8} role={task.role:<10} depends_on={list(task.depends_on)}")

config = spec.to_config()
print(f"OrchestratorConfig: max_concurrency={config.max_concurrency}, "
      f"max_tasks={config.max_tasks}")

print("转回 dict（可落盘 / 日志）:")
print(json.dumps(spec.to_dsl(), ensure_ascii=False, indent=2)[:300] + " ...")

# 往返一致
assert PlanSpec.from_dsl(spec.to_dsl()).names == spec.names

# 也可以不用 DSL，直接用 Task 列表
manual = PlanSpec.from_dsl([
    Task("a", "第一步"),
    Task("b", "第二步", depends_on=("a",)),
])
print(f"Task 列表直接建spec: {manual.names}")

# ============================================================
# 3. LLMPlanner：只给需求，让模型拆任务
# ============================================================
print()
print("=" * 60)
print("3. LLMPlanner：只给 task 描述，模型输出 JSON 任务列表")
print("=" * 60)


class FakeAgent:
    """假 agent：模拟模型返回 JSON（离线演示用，不联网）。"""

    vendor = "fake"
    model = "scripted"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        self.calls += 1
        return self.reply


planner_reply = json.dumps(
    {
        "reasoning": "先调研再写，两步最省",
        "tasks": [
            {
                "name": "research",
                "description": "调研 {{topic}} 的现状与玩家",
                "role": "researcher",
                "depends_on": [],
                "output_key": "research",
            },
            {
                "name": "write",
                "description": "根据调研写 800 字报告",
                "role": "writer",
                "depends_on": ["research"],
                "output_key": "report",
            },
        ],
    },
    ensure_ascii=False,
)

llm_planner = LLMPlanner(FakeAgent(planner_reply), max_tasks=5)

response2 = handle(
    Request(
        action="plan",
        spec={"task": "写一篇 {{topic}} 的行业观察"},   # 只给需求，没有 tasks
        context={"topic": "AI Agent"},
        planner=llm_planner,
    )
)

print(f"ok   : {response2.ok}")
print(f"波次 : {response2.data['waves']}")
assert response2.ok
assert response2.data["waves"] == [["research"], ["write"]]

# 也可以直接用 planner 拿 Plan（不经过 handle）
plan2 = llm_planner.plan("写一篇 AI Agent 行业观察", {"topic": "AI Agent"})
print(f"planner.plan 直接得到: {[t.name for t in plan2]}")

# ============================================================
# 4. 错误处理：环 / 未知键 / 空计划
# ============================================================
print()
print("=" * 60)
print("4. 错误处理（v2：不抛异常，错误码在 response.code）")
print("=" * 60)


def show_failure(title: str, request: Request) -> None:
    resp = handle(request)
    print(f"{title:<20} ok={resp.ok}  code={resp.code.value:<24} {resp.message}")
    assert not resp.ok


show_failure(
    "循环依赖",
    Request(action="plan", spec={"tasks": [
        {"name": "a", "depends_on": ["b"]},
        {"name": "b", "depends_on": ["a"]},
    ]}),
)

show_failure(
    "未知顶层键",
    Request(action="plan", spec={
        "taks": [],                                  # 拼错了，应该写 tasks
        "tasks": [{"name": "a", "description": "x"}],
    }),
)

show_failure(
    "未知任务字段",
    Request(action="plan", spec={"tasks": [{"name": "a", "description": "写错"}]}),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                    )

show_failure(
    "空计划（无 tasks 无 planner）",
    Request(action="plan", spec={"task": "做点什么"}),
)

show_failure(
    "不支持的版本",
    Request(action="plan", spec=SPEC, version="3.0"),
)

# 错误码可以直接用于分支
resp = handle(Request(action="plan", spec={"tasks": [
    {"name": "a", "depends_on": ["ghost"]},
]}))
if resp.code is ErrorCode.PLAN_UNKNOWN_DEPENDENCY:
    print(f"\n命中错误码分支：{resp.code.value} -> HTTP {resp.status}")

print()
print("=" * 60)
print("ALL PLAN EXAMPLE CHECKS PASSED")
print("=" * 60)
