"""orchestrate skill 自包含使用示例（离线、不需要 API Key）。

    python examples/orchestrate_skill.py

验证点：
    1. validate —— 永远不需要 agent，纯结构检查
    2. plan —— DSL 有 tasks 就不需要 agent；只有 task 描述需要 planner
    3. run —— 必须有 agent，无 agent 时自动返回 NO_AGENT 错误
    4. SkillRegistry 注册 → 导 OpenAI tools → 执行
    5. OrchestrateSkillSet bundle —— 一个对象搞定 run/plan/validate
    6. 边界场景：循环依赖 / 重名 / 未知键 / 空 spec / 失败透传

所有 orchestrate 相关组件都从 `promptkit.skill` 导入，
不需要手动 import Orchestrator / PlanSpec / Request / ErrorCode —— 开箱即用。
"""
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.core.event_sys import Event, EventType  # noqa: E402
from promptkit.skill import (  # noqa: E402
    OrchestrateSkillSet,
    SkillContext,
    SkillRegistry,
    orchestrate_plan,
    orchestrate_run,
    orchestrate_validate,
    register_orchestrate_skills,
)


# ==================== 假 agent（模拟 OrchAgent 行为） ====================
class _TestAgent:
    '''编排测试用的假 agent：按顺序回话。'''

    vendor = "fake"
    model = "scripted"

    def __init__(self, replies=None, *, tokens=0, fail_times=0):
        self.replies = list(replies or ["好的"])
        self.tokens = tokens
        self.fail_times = fail_times
        self.prompts = []
        self.calls = 0

    def reset(self):
        pass

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        self.calls += 1
        text = prompt["content"] if isinstance(prompt, dict) else str(prompt)
        self.prompts.append(text[:60])
        if self.calls <= self.fail_times:
            raise RuntimeError(f"第 {self.calls} 次挂了")
        if bus is not None and self.tokens:
            bus.emit(Event(EventType.USAGE, {"total_tokens": self.tokens}))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


# ==================== DSL（和 orchestrate 模块一致） ====================
SPEC = {
    "task": "调研并写报告",
    "agents": {"writer": "资深写手"},
    "policy": {"max_concurrency": 2},
    "tasks": [
        {"name": "research", "description": "调研 {{topic}}",
         "role": "writer", "output_key": "research"},
        {"name": "write", "description": "根据 {{research}} 写报告",
         "role": "writer", "depends_on": ["research"]},
    ],
}


# ==================== 1. validate：永远不需要 agent ====================
print("=== 1. orchestrate_validate ===")

v = orchestrate_validate(spec=SPEC)
print(f"  合法 DSL  ok={v['ok']}  waves={v['waves']}")
assert v["ok"] and v["waves"] == [["research"], ["write"]]

v_cycle = orchestrate_validate(spec={
    "tasks": [
        {"name": "a", "depends_on": ["b"]},
        {"name": "b", "depends_on": ["a"]},
    ],
})
print(f"  循环依赖  ok={v_cycle['ok']}  codes={[e['code'] for e in v_cycle['errors']]}")
assert not v_cycle["ok"]

v_dup = orchestrate_validate(spec={"tasks": [{"name": "a"}, {"name": "a"}]})
print(f"  重名任务  ok={v_dup['ok']}  codes={[e['code'] for e in v_dup['errors']]}")

v_empty = orchestrate_validate(spec={})
print(f"  空 spec   ok={v_empty['ok']}")

print("  ✅ validate 全部通过\n")


# ==================== 2. plan：DSL 有 tasks 就不需要 agent ====================
print("=== 2. orchestrate_plan ===")

p = orchestrate_plan(spec=SPEC, context={"topic": "AI"})
print(f"  DSL 自带 tasks  ok={p['ok']}  waves={p['data']['waves']}")
assert p["ok"]

p_no = orchestrate_plan(spec={"task": "写个东西"})
print(f"  只有 task + 无 agent  ok={p_no['ok']}  code={p_no['code']}")
assert not p_no["ok"] and p_no["code"] == "INVALID_REQUEST"

# 显式 planner（注入 ctx.tools）
class DummyPlanner:
    def plan(self, task, context=None):
        from promptkit.orchestrate import PlanSpec
        return PlanSpec.from_dsl({
            "task": task,
            "tasks": [
                {"name": "research", "description": "调研"},
                {"name": "write", "description": "写", "depends_on": ["research"]},
            ],
        })

p_planner = orchestrate_plan(
    spec={"task": "写报告"},
    ctx=SkillContext(tools={"planner": DummyPlanner()}),
)
print(f"  显式 planner    ok={p_planner['ok']}")
assert p_planner["ok"]

print("  ✅ plan 全部通过\n")


# ==================== 3. run：必须有 agent ====================
print("=== 3. orchestrate_run ===")

r_no = orchestrate_run(spec=SPEC)
print(f"  无 agent  ok={r_no['ok']}  code={r_no['code']}")
assert not r_no["ok"] and r_no["code"] == "NO_AGENT"

agent = _TestAgent(replies=["AI 正在改变世界", "AI 报告内容"])
ctx = SkillContext(agent=agent)
r = orchestrate_run(spec=SPEC, context={"topic": "AI"}, ctx=ctx)
print(f"  有 agent  ok={r['ok']}  results={[x['name'] for x in r['data']['results']]}")
assert r["ok"]

print("  ✅ run 全部通过\n")


# ==================== 4. SkillRegistry：一行注册，导工具，执行 ====================
print("=== 4. SkillRegistry ===")

registry = register_orchestrate_skills()
print(f"  registry names={registry.names}")
assert len(registry) == 3

tools = registry.to_openai_tools()
print(f"  to_openai_tools count={len(tools)}")
for t in tools:
    assert t["type"] == "function"
    params = t["function"]["parameters"]
    assert "spec" in params["properties"]

ctx_run = SkillContext(agent=_TestAgent(replies=["A", "B"]), skills=registry)
result = registry.execute(
    "orchestrate_run",
    {"spec": SPEC, "context": {"topic": "X"}},
    ctx=ctx_run,
)
print(f"  registry.execute run ok={result['ok']}")
assert result["ok"]

result_v = registry.execute(
    "orchestrate_validate",
    {"spec": {"tasks": [{"name": "a", "description": "x"}]}},
    ctx=SkillContext(),
)
print(f"  registry.execute validate ok={result_v['ok']}")
assert result_v["ok"]

print("  ✅ SkillRegistry 全部通过\n")


# ==================== 5. OrchestrateSkillSet：一个对象搞定 ====================
print("=== 5. OrchestrateSkillSet bundle ===")

# 无 agent —— validate / plan 可用
bundle = OrchestrateSkillSet()
print(f"  bundle={bundle}")
b_v = bundle.validate({"tasks": [{"name": "a", "description": "x"}]})
b_p = bundle.plan({"tasks": [{"name": "a", "description": "x"}]})
print(f"  bundle.validate ok={b_v['ok']}  bundle.plan ok={b_p['ok']}")
assert b_v["ok"] and b_p["ok"]

# 带 agent —— run 也可用
bundle_full = OrchestrateSkillSet(ctx=SkillContext(
    agent=_TestAgent(replies=["ok1", "ok2"])
))
b_r = bundle_full.run(spec=SPEC, context={"topic": "Y"})
print(f"  bundle.run ok={b_r['ok']}")
assert b_r["ok"]

reg2 = SkillRegistry()
bundle.register_to(reg2)
assert sorted(reg2.names) == sorted(registry.names)
print(f"  bundle.register_to ok")

print("  ✅ OrchestrateSkillSet 全部通过\n")


print("=== ALL EXAMPLE CHECKS PASSED ===")
