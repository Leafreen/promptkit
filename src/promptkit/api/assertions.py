# promptkit/api/assertions.py
'''DSL 断言：`Assert(rule)` 步骤 + 可插拔的断言适配器。

规则写法与 `If` / `While` 完全一致（bool / Expr / Condition / callable），
所以「DSL 里能写的条件」都能直接当断言用：

    from promptkit.api import Assert

    Workflow(
        Assert(Var("count") > 0, message="count 必须大于 0"),
        Assert(lambda ctx: ctx.get("answer"), message="answer 不能为空"),
    ).run()

自定义规则类型（比如 pydantic 模型、字典规则）注册一个适配器即可：

    from promptkit.api import AssertionAdapter, assertion_adapters

    class DictRule(AssertionAdapter):
        name = "dict-rule"
        def supports(self, rule): return isinstance(rule, dict) and "op" in rule
        def check(self, rule, context):
            ok = context.get(rule["key"]) == rule.get("value")
            return AssertionResult(ok, f"{rule['key']} 不符合预期")

    assertion_adapters.register(DictRule())

适配器是全局注册表；想隔离就在 `Assert(..., adapters=Assertions(...))` 里
传私有的。断言失败抛 `AssertionFailed`（`WorkflowError` 子类），因此
`error` 钩子可以统一捕获。
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Union

from promptkit.core.errors import WorkflowError
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.conditions import as_condition

__all__ = [
    "AssertionResult",
    "AssertionAdapter",
    "CallableAssertionAdapter",
    "Assertions",
    "AssertionFailed",
    "AssertStep",
    "Assert",
    "assertion_adapters",
]


@dataclass(frozen=True)
class AssertionResult:
    '''一次断言的结果。`bool(result)` 等价于 `result.ok`。'''

    ok: bool
    message: str = ""
    rule: Any = None

    def __bool__(self) -> bool:
        return self.ok


class AssertionAdapter(ABC):
    '''断言适配器：把一种「规则」翻译成 `AssertionResult`。

    标准接口只有两个方法：
      - `supports(rule)`：这个规则归不归我管
      - `check(rule, context)`：断言的真正实现

    `AssertStep` 按注册顺序找第一个 `supports()` 为真的适配器。
    '''

    name: str = "assertion"

    def supports(self, rule: Any) -> bool:
        '''这个适配器是否认识这种规则。'''
        return False

    @abstractmethod
    def check(self, rule: Any, context: Any) -> AssertionResult:
        '''执行断言；实现里不该抛异常来表达「不成立」。'''
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class CallableAssertionAdapter(AssertionAdapter):
    '''默认适配器：bool / Expr / Condition / callable(context)。'''

    name = "condition"

    def supports(self, rule: Any) -> bool:
        return (
            isinstance(rule, bool)
            or callable(rule)
            or hasattr(rule, "eval")          # Expr
            or hasattr(rule, "evaluate")      # Condition
        )

    def check(self, rule: Any, context: Any) -> AssertionResult:
        condition = as_condition(rule)
        ok = bool(condition(context))
        text = getattr(condition, "text", None) or repr(rule)
        return AssertionResult(
            ok=ok,
            message="" if ok else f"断言不成立：{text}",
            rule=rule,
        )


class Assertions:
    '''适配器注册表；`check()` 负责挑选并执行。'''

    __slots__ = ("_adapters",)

    def __init__(self, *adapters: AssertionAdapter):
        self._adapters: List[AssertionAdapter] = []
        for adapter in adapters or (CallableAssertionAdapter(),):
            self.register(adapter)

    def register(self, adapter: AssertionAdapter) -> AssertionAdapter:
        '''注册适配器（插到最前，自定义的会先于内置的被尝试）。'''
        if not isinstance(adapter, AssertionAdapter):
            raise TypeError(
                f"适配器必须是 AssertionAdapter 子类，得到 {type(adapter).__name__}"
            )
        if adapter not in self._adapters:
            self._adapters.insert(0, adapter)
        return adapter

    def unregister(self, adapter: Union[AssertionAdapter, type]) -> bool:
        '''按实例或类注销；返回是否真的删掉了。'''
        is_type = isinstance(adapter, type)
        matched = [
            item
            for item in self._adapters
            if (isinstance(item, adapter) if is_type else item is adapter)
        ]
        for item in matched:
            self._adapters.remove(item)
        return bool(matched)

    def check(self, rule: Any, context: Any) -> AssertionResult:
        '''找第一个支持该规则的适配器并执行。'''
        for adapter in tuple(self._adapters):
            if adapter.supports(rule):
                return adapter.check(rule, context)
        raise TypeError(
            f"没有适配器认识这种断言规则：{type(rule).__name__}。"
            "请注册一个 AssertionAdapter，或改用 bool / Expr / Condition / callable"
        )

    def __call__(self, rule: Any, context: Any) -> AssertionResult:
        return self.check(rule, context)

    def __len__(self) -> int:
        return len(self._adapters)

    def __repr__(self) -> str:
        names = ", ".join(item.name for item in self._adapters)
        return f"<Assertions [{names}]>"


class AssertionFailed(WorkflowError):
    '''断言不成立。`result` 里有原始规则与消息。'''

    def __init__(self, result: AssertionResult):
        super().__init__(result.message or "断言失败")
        self.result = result


class AssertStep(Step):
    '''把断言写进步骤树。

    - `message=` 覆盖失败时的默认文案
    - `variable=` 把 `AssertionResult` 存进 Context，供后续步骤 / 钩子读取
    - `raise_error=False` 只记录不抛错（配合 `before/after` 钩子上报）
    '''

    name = "assert"

    def __init__(
        self,
        rule: Any,
        *,
        message: Optional[str] = None,
        variable: Optional[str] = None,
        raise_error: bool = True,
        adapters: Optional[Assertions] = None,
    ):
        if adapters is not None and not isinstance(adapters, Assertions):
            raise TypeError(
                f"adapters 必须是 Assertions，得到 {type(adapters).__name__}"
            )
        self.rule = rule
        self.message = message
        self.variable = variable
        self.raise_error = bool(raise_error)
        self.adapters = adapters

    def check(self, context: Any) -> AssertionResult:
        '''只求值，不抛错；钩子 / 自定义流程可以直接用。'''
        registry = self.adapters or assertion_adapters
        result = registry.check(self.rule, context)
        if not result.ok and self.message:
            return replace(result, message=str(self.message))
        return result

    def execute(self, context: Any) -> None:
        result = self.check(context)
        context.trace.append(f"{self.name}:{'ok' if result.ok else 'fail'}")
        if self.variable:
            context[self.variable] = result
        if not result.ok and self.raise_error:
            raise AssertionFailed(result)

    def describe(self) -> str:
        return f"Assert({self.rule!r})"

    def __repr__(self) -> str:
        return f"<AssertStep rule={self.rule!r}>"


def Assert(rule: Any, **kwargs: Any) -> AssertStep:  # noqa: N802 - 跟随 DSL 命名
    '''`Assert(规则, message=..., variable=..., raise_error=..., adapters=...)`。

    规则与 `If` / `While` 的条件同构：bool / Expr / Condition / callable。
    '''
    return AssertStep(rule, **kwargs)


#: 全局默认注册表（`Assert` 不带 `adapters=` 时用它）
assertion_adapters = Assertions()
