"""While 循环的 Hook 机制：手写 DSL 时怎么定义和使用 Hook。

    python examples/while_hooks.py

七个扩展点（按执行顺序）：

    loop_start        进入循环前，一次
      before_condition 每轮判条件前，返回 False 结束循环
        before_iteration 条件成立、跑循环体前，返回 False 跳过本轮
        after_iteration  本轮结束（含被跳过 / Break / Continue）
      loop_limit       即将超出 max_iterations 之前
      loop_error       条件或循环体抛异常时（之后照常抛）
    loop_end           循环正常结束 / Break，一次
"""
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.core import (  # noqa: E402
    AFTER_ITERATION,
    BEFORE_CONDITION,
    BEFORE_ITERATION,
    LOOP_END,
    LOOP_ERROR,
    LOOP_HOOK_POINTS,
    LOOP_LIMIT,
    LOOP_START,
    FunctionStep,
    HookPoint,
    LoopHooks,
    NewVar,
    Set,
    Var,
    While,
    Workflow,
    loop_hooks,
)

# ============================================================
# 1. 七个扩展点的执行时机
# ============================================================
print("=" * 58)
print("1. 七个扩展点的执行时机")
print("=" * 58)

order = []


def watch(point):
    def fn(event):
        order.append(f"{point.value}@{event.iteration}")

    return fn


loop = While(Var("n") < 3, max_iterations=10).Do(Set("n", Var("n") + 1))
for point in LOOP_HOOK_POINTS:
    if point not in (LOOP_LIMIT, LOOP_ERROR):   # 正常跑不会触发这两个
        loop.Hook(point, watch(point))

Workflow(NewVar("n", default=0), loop).run()
for line in order:
    print("  " + line)

assert order[:4] == [
    "loop_start@0",
    "before_condition@1",
    "before_iteration@1",
    "after_iteration@1",
]
assert order[-2:] == ["before_condition@4", "loop_end@3"]

# ============================================================
# 2. before_iteration 返回 False：跳过这一轮
# ============================================================
print()
print("=" * 58)
print("2. before_iteration 返回 False → 跳过本轮（仍触发 after_iteration）")
print("=" * 58)

signals = []


def skip_odd(event):
    '''奇数轮返回 False → 这一轮不跑循环体。'''
    return event.iteration % 2 == 0


ctx = Workflow(
    NewVar("n", default=0),
    While(Var("n") < 4, max_iterations=10)
    .Hook(BEFORE_ITERATION, skip_odd)
    .Hook(AFTER_ITERATION, lambda e: signals.append(e.signal))
    .Do(Set("n", Var("n") + 1)),
).run()

print(f"每轮信号: {signals}")
print(f"n = {ctx['n']}")
assert signals == ["skipped", None, "skipped", None, "skipped", None, "skipped", None]
assert ctx["n"] == 4

# ============================================================
# 3. before_condition 返回 False：提前结束循环
# ============================================================
print()
print("=" * 58)
print("3. before_condition 返回 False → 结束循环（reason='hook'）")
print("=" * 58)

reasons = []
ctx = Workflow(
    NewVar("n", default=0),
    While(True, max_iterations=100)
    .Hook(BEFORE_CONDITION, lambda e: e.iteration <= 3)
    .Hook(LOOP_END, lambda e: reasons.append((e.reason, e.iteration)))
    .Do(Set("n", Var("n") + 1)),
).run()

print(f"结束原因: {reasons}   n = {ctx['n']}")
assert reasons == [("hook", 3)]

# ============================================================
# 4. 注册 / 注销：句柄、函数、扩展点名
# ============================================================
print()
print("=" * 58)
print("4. 注册 / 注销")
print("=" * 58)

hooks = LoopHooks()
handle = hooks.register(AFTER_ITERATION, lambda e: None)
hooks.register("loop_end", lambda e: None)
print(f"注册后: {hooks}")
print(f"counts: {hooks.counts()}")

handle.off()                                     # 句柄注销
print(f"handle.off() 后: {hooks}")

hooks.unregister("loop_end")                     # 按扩展点名注销
print(f"清空某点后: {hooks} | 空表? {not hooks}")

# 装饰器写法
@hooks.on(BEFORE_ITERATION)
def log_start(event):
    print(f"    [装饰器] 第 {event.iteration} 轮开始，name={event.name}")


Workflow(
    NewVar("n", default=0),
    While(Var("n") < 2, max_iterations=5, hooks=hooks).Do(Set("n", Var("n") + 1)),
).run()

# ============================================================
# 5. 全局钩子：运行时通过 Context 挂到所有循环上
# ============================================================
print()
print("=" * 58)
print("5. loop_hooks(context)：运行时注册，对所有循环生效")
print("=" * 58)

global_hits = []


def add_global(ctx):
    loop_hooks(ctx).register(
        AFTER_ITERATION,
        lambda e: global_hits.append((e.name, e.iteration)),
    )


ctx = Workflow(
    NewVar("a", default=0),
    FunctionStep(add_global, name="add_hook"),
    While(Var("a") < 2, max_iterations=5).Do(Set("a", Var("a") + 1)),
).run()
print(f"全局钩子收到: {global_hits}")
assert global_hits == [("while", 1), ("while", 2)]

# ============================================================
# 6. 错误隔离：Hook 抛异常不影响主循环
# ============================================================
print()
print("=" * 58)
print("6. 错误隔离")
print("=" * 58)

bad = LoopHooks({"after_iteration": lambda e: 1 / 0})
ctx = Workflow(
    NewVar("n", default=0),
    While(Var("n") < 3, max_iterations=10, hooks=bad).Do(Set("n", Var("n") + 1)),
).run()
print(f"主循环照常跑完: n = {ctx['n']}")
print(f"记录到的失败: {len(bad.failures)} 次")
for failure in bad.failures[:2]:
    print(f"  {failure}")

assert ctx["n"] == 3
assert len(bad.failures) == 3
assert bad.failures[0].point is HookPoint.AFTER_ITERATION

# ============================================================
# 7. loop_limit / loop_error
# ============================================================
print()
print("=" * 58)
print("7. loop_limit 与 loop_error")
print("=" * 58)

limit_seen = []
try:
    Workflow(
        NewVar("n", default=0),
        While(Var("n") < 100, max_iterations=2)
        .Hook(LOOP_LIMIT, lambda e: limit_seen.append((e.iteration, type(e.error).__name__)))
        .Do(Set("n", Var("n") + 1)),
    ).run()
except Exception as exc:
    print(f"抛出: {type(exc).__name__}")
print(f"loop_limit 钩子看到: {limit_seen}")
assert limit_seen == [(3, "LoopLimitExceeded")]


def boom(ctx):
    raise RuntimeError("循环体里的错")


errors = []
try:
    Workflow(
        NewVar("n", default=0),
        While(True, max_iterations=5)
        .Hook(LOOP_ERROR, lambda e: errors.append(str(e.error)))
        .Do(FunctionStep(boom, name="boom")),
    ).run()
except RuntimeError as exc:
    print(f"异常照常抛出: {exc}")
print(f"loop_error 钩子看到: {errors}")
assert errors == ["循环体里的错"]

print()
print("=" * 58)
print("ALL WHILE HOOK CHECKS PASSED")
print("=" * 58)
