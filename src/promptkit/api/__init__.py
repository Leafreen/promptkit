# promptkit/api/__init__.py
'''轻量扩展 API：Hook 生命周期 + DSL 断言。

一句话接入现有 DSL（不需要适配层）：

    from promptkit.api import Hooks, Assert

    hooks = Hooks(
        before=[lambda e: None],
        after=[lambda e: print(e.name, e.duration_ms)],
        error=[lambda e: isinstance(e.error, AssertionFailed)],
    )

    wf = Workflow(
        Assert(Var("count") > 0, message="count 必须大于 0"),
        Assert(lambda ctx: bool(ctx.get("answer"))),
    )
    ctx = wf.run(agent=agent, hooks=hooks)

模块很小、零额外依赖：
  - `hooks.py`      三个生命周期钩子 + 注册 / 注销
  - `assertions.py` 断言步骤 + 可插拔适配器
'''
from promptkit.api.assertions import (
    Assert,
    AssertionAdapter,
    AssertionFailed,
    AssertionResult,
    Assertions,
    AssertStep,
    CallableAssertionAdapter,
    assertion_adapters,
)
from promptkit.api.hooks import (
    HOOKS_KEY,
    HookAbort,
    HookEvent,
    HookHandle,
    HookType,
    Hooks,
    use_hooks,
)

__all__ = [
    # hooks.py
    "Hooks",
    "HookEvent",
    "HookType",
    "HookHandle",
    "HookAbort",
    "HOOKS_KEY",
    "use_hooks",
    # assertions.py
    "Assert",
    "AssertStep",
    "AssertionFailed",
    "AssertionResult",
    "AssertionAdapter",
    "CallableAssertionAdapter",
    "Assertions",
    "assertion_adapters",
]
