'''Hook 扩展 + DSL 断言的最小示例（离线，不需要 API Key）。

运行：
    python examples/hooks_and_assertions.py

三件事：
1. 用 `Hooks` 挂三类生命周期钩子：前置拦截 / 后置处理 / 异常捕获
2. 用 `Assert` 把断言写进步骤树，失败可抛错，也可只记录
3. 用自定义 `AssertionAdapter` 让断言认识自己的规则类型

接入方式只有一种：`Workflow(...).run(hooks=hooks)`。
`execute_step` 是所有 DSL 步骤的唯一出入口，所以嵌套的 If / While /
子 Workflow 里的每一步都会被同一个钩子表看到，不需要额外适配层。
'''
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.api import (  # noqa: E402
    Assert,
    AssertionAdapter,
    AssertionFailed,
    AssertionResult,
    Hooks,
    use_hooks,
)
from promptkit.core import (  # noqa: E402
    FunctionStep,
    If,
    Inc,
    Number,
    Set,
    Var,
    While,
    Workflow,
)

# ============================ 1. 三类钩子 ============================
hooks = Hooks()

# 前置拦截：返回 False 就跳过这一步（这里跳过所有名为 "debug" 的步骤）
hooks.on_before(
    lambda event: False if event.name == "debug" else None
)

# 后置处理：步骤正常结束后拿到耗时，顺手往 Context 里记一笔
def record_timing(event):
    event.context.setdefault("timings", {})[event.name] = event.duration_ms

hooks.on_after(record_timing)

# 异常捕获：返回 True 表示「这个异常我处理了」，工作流继续
def swallow_assertion(event):
    if isinstance(event.error, AssertionFailed):
        event.context.setdefault("warnings", []).append(str(event.error))
        return True
    return False

hooks.on_error(swallow_assertion)


# ============================ 2. 断言与步骤树 ============================
def build_workflow() -> Workflow:
    return Workflow(
        Number("i"),
        Set("i", 0),
        Assert(Var("i") == 0, message="i 应该从 0 开始"),
        # 循环里也能用断言：i 每次 +1，但不超过 3
        While(Var("i") < 5, max_iterations=10).Do(
            Inc("i"),
            Assert(Var("i") <= 3, message="i 超过 3 了"),
        ),
    ).add(
        # 只记录不抛错：把 AssertionResult 存进 Context，供后续步骤读取
        Assert(False, variable="verdict", raise_error=False),
    )


def main() -> None:
    ctx = build_workflow().run(hooks=hooks)

    print("i =", ctx["i"])                       # 循环在断言失败时被钩子兜住后继续
    print("timings =", ctx.get("timings"))
    print("warnings =", ctx.get("warnings"))
    print("verdict =", ctx.get("verdict"))
    print()

    # ======================== 3. 自定义断言适配器 ========================
    class RangeRule(AssertionAdapter):
        '''规则长这样：{"key": "score", "min": 0, "max": 100}。'''

        name = "range-rule"

        def supports(self, rule):
            return isinstance(rule, dict) and "key" in rule

        def check(self, rule, context):
            value = context.get(rule["key"])
            low, high = rule.get("min"), rule.get("max")
            ok = value is not None and (low is None or value >= low) and (
                high is None or value <= high
            )
            message = "" if ok else f"{rule['key']}={value!r} 不在 [{low}, {high}] 内"
            return AssertionResult(ok, message, rule)

    from promptkit.api import Assertions

    registry = Assertions()
    registry.register(RangeRule())

    ctx2 = Workflow(
        Number("score"),
        Set("score", 120),
        Assert(
            {"key": "score", "min": 0, "max": 100},
            message="分数必须在 0~100 之间",
            adapters=registry,     # 用私有注册表，不影响全局
            raise_error=False,
            variable="score_check",
        ),
    ).run()
    print("score_check =", ctx2["score_check"])

    # ======================== 4. 在步骤里动态挂钩子 ========================
    def install_dynamic_hook(context):
        use_hooks(context).on_after(
            lambda event: context.setdefault("trace_names", []).append(event.name)
        )

    ctx3 = Workflow(
        FunctionStep(install_dynamic_hook, name="install"),
        Set("x", 1),
    ).run()
    print("dynamic trace =", ctx3["trace_names"])


if __name__ == "__main__":
    main()
