# coding/loop.py
'''`FixLoop`：测试失败 → 带着错误再改一轮。

和 `While` 的区别不是能力，而是**意图**：

    While(Var("round") < 3).Do(...)                    # 你得自己数轮数、自己 Break
    FixLoop(body, until=Var("test_result")["passed"])   # 说清「什么算修好了」

它还负责两件容易忘的事：

- 把当前轮次写进 `round_key`（默认 `fix_round`，从 1 开始），提示词里可以引用
- 轮数用完还没通过时**明确标记**（`fix_exhausted=True`），而不是假装成功；
  `on_exhausted="raise"` 可以让它直接抛错，逼人工介入
'''
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from promptkit.core.errors import LoopLimitExceeded
from promptkit.core.steps.interface import Step, coerce_steps
from promptkit.core.steps.logics.conditions import as_condition
from promptkit.core.steps.logics.loop import run_loop_body

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context

__all__ = ["FixLoop"]


@dataclass
class FixLoop(Step):
    '''反复执行 `body`，直到 `until` 成立（或轮数用完）。

        FixLoop(
            [
                CodingStep(task=Var("task"), error=Var("test_result"), path="src/a.py"),
                ApplyDiff("src/a.py", diff=Var("patch")),
                RunTests("python -m unittest -q"),
            ],
            until=Var("test_result")["passed"],
            max_rounds=3,
        )

    - `until`：成功条件（`Expr` / `Condition` / `callable(context)`）
    - `max_rounds`：最多几轮
    - `on_exhausted`：`"stop"`（默认，标记后继续走）或 `"raise"`（抛 `LoopLimitExceeded`）
    - 循环体里的 `Break()` 会提前结束（视为「不想再修了」），`Continue()` 跳过本轮剩余步骤
    '''

    body: Tuple[Step, ...] = ()
    until: Any = None
    max_rounds: int = 3
    round_key: Optional[str] = "fix_round"
    exhausted_key: Optional[str] = "fix_exhausted"
    on_exhausted: str = "stop"

    name: str = "fix_loop"

    def __post_init__(self) -> None:
        self.body = tuple(coerce_steps(self.body))
        if self.until is None:
            raise ValueError(
                "FixLoop 需要 until=（成功条件，比如 Var('test_result')['passed']）"
            )
        self.until = as_condition(self.until)
        if not isinstance(self.max_rounds, int) or self.max_rounds < 1:
            raise ValueError(
                f"max_rounds 必须是正整数，得到 {self.max_rounds!r}"
            )
        if self.on_exhausted not in ("stop", "raise"):
            raise ValueError(
                f"on_exhausted 只能是 'stop' 或 'raise'，得到 {self.on_exhausted!r}"
            )
        if self.round_key is not None and not isinstance(self.round_key, str):
            raise TypeError(
                f"round_key 需要字符串或 None，得到 {type(self.round_key).__name__}"
            )
        if self.exhausted_key is not None and not isinstance(
            self.exhausted_key, str
        ):
            raise TypeError(
                "exhausted_key 需要字符串或 None，"
                f"得到 {type(self.exhausted_key).__name__}"
            )

    def Do(self, *steps: Any) -> "FixLoop":  # noqa: N802 - 跟随 DSL 命名
        '''链式写法：`FixLoop(until=...).Do(...)`。'''
        return replace(self, body=tuple(coerce_steps(*steps)))

    def execute(self, context: "Context") -> None:
        if not self.body:
            raise ValueError(
                "FixLoop 的 body 是空的：构造时传 body=[...]，或者用 .Do(...) 补上"
            )
        last_round = 0
        for round_number in range(1, self.max_rounds + 1):
            last_round = round_number
            if self.round_key:
                context[self.round_key] = round_number
            context.trace.append(f"{self.name}:{round_number}")

            if run_loop_body(self.body, context) == "break":
                # 显式 Break：调用方自己决定停，不算 exhausted
                if self.exhausted_key:
                    context[self.exhausted_key] = False
                context.trace.append(f"{self.name}:done")
                return

            if self.until(context):
                if self.exhausted_key:
                    context[self.exhausted_key] = False
                context.trace.append(f"{self.name}:done")
                return

        if self.exhausted_key:
            context[self.exhausted_key] = True
        context.trace.append(f"{self.name}:exhausted")
        if self.on_exhausted == "raise":
            raise LoopLimitExceeded(
                f"{self.name} 跑了 {last_round} 轮仍然没有满足条件（{self.until!r}）；"
                "把 on_exhausted 改成 'stop' 可以只标记不抛错"
            )

    def describe(self) -> str:
        return f"FixLoop(until {self.until!r}, max_rounds={self.max_rounds})"

    def children(self) -> List[Step]:
        return list(self.body)
