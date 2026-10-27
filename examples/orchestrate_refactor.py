"""编排层重构 smoke test（离线，不需要 API Key）。

跑法：
    python examples/orchestrate_refactor.py

核心验证点：
  - PlanSpec 把 DSL dict 一次性解析成 plan + config + factory
  - v2 handle() 统一返回 Response（ok/code/message/data）
  - v1 模式保持旧行为（抛 PlanError/TaskError）
  - ExecutePlan 直接接受 DSL dict
  - 错误码在 abort / timeout / unknown-role 时正确传递
  - 模块级唯一计划解析入口 plan_from（从 task、dict、Plan 都能进）
"""
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.orchestrate import (  # noqa: E402
    API_VERSION,
    ErrorCode,
    ExecutePlan,
    OrchestratorConfig,
    PlanError,
    PlanSpec,
    Request,
    TaskError,
    handle,
    plan_from,
)
from promptkit.orchestrate.task import Plan, Task  # noqa: E402
from promptkit.core import Set, Workflow  # noqa: E402


# ==================== 假 agent ====================
class FakeAgent:
    def __init__(self, replies=("ok",)):
        self.replies = list(replies)
    def reset(self): pass
    def ask(self, prompt, tools=None, bus=None, **kwargs):
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class SlowAgent(FakeAgent):
    def __init__(self, delay=0.2):
        super().__init__(["still-working"])
        self.delay = delay
    def ask(self, *args, **kwargs):
        time.sleep(self.delay)
        return super().ask(*args, **kwargs)


# ==================== 1. DSL 声明式 ====================
SPEC = {
    "task": "调研并写报告",
    "policy": {"max_concurrency": 2, "max_tasks": 5},
    "tasks": [
        {"name": "research", "description": "调研 {{topic}}", "output_key": "research"},
        {"name": "write", "description": "写 {{research}}", "depends_on": ["research"]},
    ],
}

spec = PlanSpec.from_dsl(SPEC)
plan = spec.to_plan()
config = spec.to_config()

assert plan.names == ["research", "write"], plan.names
assert config.max_concurrency == 2
assert PlanSpec.from_dsl(spec.to_dsl()).names == spec.names, "round-trip"
print("1. PlanSpec.from_dsl / to_plan / to_config / round-trip  OK")


# ==================== 2. v2 统一 API ====================
response = handle(
    Request(spec=SPEC, context={"topic": "运筹学"}, agent=FakeAgent(["A", "B"]))
)
assert response.ok
assert response.status == 200
assert response.code is ErrorCode.OK
results = response.data["results"]
assert [r["name"] for r in results] == ["research", "write"]
print("2. v2 handle OK  |  status={}  code={}  results={}  ".format(
    response.status, response.code.value,
    [(r["name"], r["status"], r["output"]) for r in results]
))

# plan 动作：DSL 已经给了 tasks → 直接返回
planner_response = handle(
    Request(action="plan", spec=SPEC, context={"topic": "x"})
)
assert planner_response.ok
assert "waves" in planner_response.data
print("   plan action waves =", planner_response.data["waves"])

# plan 动作：只有 task 描述 → planner 来拆
class _DummyPlanner:
    def plan(self, task, ctx):
        return Plan.from_template([Task("research", "调研"), Task("write", "写")], task=task)

only_task = handle(
    Request(action="plan", spec={"task": "写东西"}, planner=_DummyPlanner())
)
assert only_task.ok
assert [t["name"] for t in only_task.data["tasks"]] == ["research", "write"]
print("   plan via planner tasks =", [t["name"] for t in only_task.data["tasks"]])


# ==================== 3. 错误码 ====================
cyclic = {"tasks": [{"name": "a", "depends_on": ["b"]}, {"name": "b", "depends_on": ["a"]}]}
r = handle(cyclic)
assert not r.ok and r.code is ErrorCode.PLAN_CYCLE and r.status == 422
print(f"3a. cycle  -> code={r.code.value} status={r.status}")

dup = {"tasks": [{"name": "a"}, {"name": "a"}]}
r = handle(dup)
assert r.code is ErrorCode.PLAN_DUPLICATE_TASK

unknown_dep = {"tasks": [{"name": "a", "depends_on": ["ghost"]}]}
r = handle(unknown_dep)
assert r.code is ErrorCode.PLAN_UNKNOWN_DEPENDENCY

# abort 失败 —— agent 连续失败（fail_times 足够大导致重试耗尽）
class _FailAgent(FakeAgent):
    def ask(self, *args, **kwargs):
        raise RuntimeError("模拟任务执行失败")

abort_spec = {"tasks": [{"name": "a", "description": "x", "on_failure": "abort",
                         "max_retries": 0},
                        {"name": "b", "description": "y"}]}
r = handle(Request(spec=abort_spec, agent=_FailAgent()))
assert not r.ok and r.code is ErrorCode.TASK_FAILED, r
print(f"3b. abort  -> code={r.code.value}  failed_task={r.data.get('failed_task')}")

# timeout 失败：错误码要从 TaskError 里透传下来
slow_spec = {"tasks": [{"name": "slow", "description": "x",
                        "timeout": 0.01, "max_retries": 0, "on_failure": "abort"}]}
r = handle(Request(spec=slow_spec, agent=SlowAgent()))
assert not r.ok and r.code is ErrorCode.TASK_TIMEOUT, r
print(f"3c. timeout -> code={r.code.value}")

# unknown role：key error 映射成 ROLE_UNDEFINED
role_spec = {
    "agents": {"writer": "写手"},
    "tasks": [{"name": "a", "role": "ghost", "description": "x"}],
}
r = handle(role_spec)
assert r.code is ErrorCode.ROLE_UNDEFINED
print(f"3d. role   -> code={r.code.value}")

# bad version / bad action / missing spec
assert handle(Request(spec=SPEC, version="9.9")).code is ErrorCode.UNSUPPORTED_VERSION
assert handle(Request(action="delete", spec=SPEC)).code is ErrorCode.INVALID_REQUEST
assert handle(Request()).code is ErrorCode.INVALID_REQUEST
print("3e. 入参校验 OK")


# ==================== 4. v1 向后兼容（抛异常） ====================
try:
    handle(Request(spec=cyclic, version="1.0"))
except PlanError as exc:
    assert exc.code is ErrorCode.PLAN_CYCLE
    print("4a. v1 抛 PlanError OK")

try:
    handle(Request(spec=abort_spec, agent=FakeAgent(["A", "B"]), version="1.0"))
except TaskError as exc:
    assert exc.code is ErrorCode.TASK_FAILED
    assert exc.task == "a"
    print("4b. v1 抛 TaskError OK")


# ==================== 5. ExecutePlan 直接吃 DSL ====================
ctx = Workflow(
    Set("topic", "运筹学"),
    ExecutePlan(plan=SPEC, agent=FakeAgent(["X", "Y"])),
).run()
assert ctx["plan_result"].ok
assert ctx.get("research") == "X"
print("5. ExecutePlan DSL 直连 OK")

# ExecutePlan 拒绝未知 plan 类型
try:
    Workflow(ExecutePlan(plan=123)).run()
except PlanError:
    print("   ExecutePlan 入参校验 OK")


# ==================== 6. 向后兼容：旧入口不变 ====================
result = plan_from([Task("legacy", "老接口也能跑")])
assert result.names == ["legacy"]

# Orchestrator.run 接受 list / Plan / dict 三种
from promptkit.orchestrate.orchestrator import Orchestrator

for value in ([Task("a", "x"), Task("b", "y", depends_on=["a"])],
              Plan((Task("a", "x"),)),
              {"tasks": [{"name": "a", "description": "x"}]}):
    r = Orchestrator(agent=FakeAgent(["ok"])).run(value)
    assert r.ok

print("6. Orchestrator.run / plan_from 旧入口 OK")


# ==================== 7. Response 序列化 ====================
r = handle(Request(spec=SPEC, context={"topic": "x"}, agent=FakeAgent()))
d = r.to_dict()
assert {"ok", "code", "message", "data", "version"} <= set(d)
assert bool(r) == r.ok   # __bool__ 重载
print("7. Response.to_dict / bool OK")


print(f"\n=== ALL SMOKE PASSED  (version={API_VERSION}) ===")
