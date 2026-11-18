# core/steps/parallel.py
'''并行执行多个子工作流，各自 Context 隔离，输出按 output_key 合并。

语法：

    # 1) 显式标 output_key（安全，key 冲突可控）
    Parallel(
        Workflow(Set("a", "hello")) >> Var("a"),
        Workflow(Set("b", "world")) >> Var("b"),
    )

    # 2) 裸 Step：子 Context 里所有非内部 key 自动 merge 回父 Context
    Parallel(
        Workflow(Set("a", "hello")),
        Workflow(Set("b", "world")),
    )

    # 3) 直接构造 Branch
    Parallel(
        Branch(Workflow(Set("a", "hello")), output_key="a"),
    )

Parallel 执行完后，父 Context 里会有 a="hello", b="world"，
然后正常写 `Set(Var("final"), Var("a") + Var("b"))` 就行。
'''
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from aipykit.core.context import Context
from aipykit.core.steps.interface import Step, execute_step

if TYPE_CHECKING:
    from aipykit.core.steps.logics.expr_sys.expr import VarExpr
    from aipykit.core.steps.logics.var import VarDecl

#: 裸 Step 时表示「子 Context 里所有非内部 key 都 merge 回父 Context」
_AUTO_MERGE = object()


@dataclass
class Branch:
    '''Parallel 的一个并行分支：Step + 可选的 output_key。

    step 执行完后，从子 Context 里取 output_key 对应的值写回父 Context。
    output_key=None 时表示自动 merge 所有非内部 key。
    '''
    step: Step
    output_key: Optional[str] = None  # None = 自动 merge

    def __repr__(self) -> str:
        if self.output_key is None:
            return f"Branch({self.step!r}, auto_merge)"
        return f"Branch({self.step!r}, output_key={self.output_key!r})"

    def describe(self) -> str:
        if self.output_key is None:
            return "auto_merge"
        return f"→ {self.output_key!r}"


class Parallel(Step):
    '''并行执行多个子工作流。

    每个分支拿到父 Context 的浅拷贝（data + types），执行完后：
    - 显式 output_key 的分支：只把子 ctx 中 output_key 对应的值写回父 ctx
    - 裸 Step 分支：把子 ctx 中所有非下划线开头的 key 写回父 ctx（后跑完的覆盖先跑完的同名 key）

    默认一个分支失败不影响其他分支继续；失败的分支不会写回父 Context 任何值。
    设 stop_on_error=True 会让第一个失败触发剩余分支取消并整体抛异常。
    '''

    name = "parallel"

    def __init__(
        self,
        *branches: Union[Step, Branch],
        max_workers: Optional[int] = None,
        stop_on_error: bool = False,
    ):
        self._branches: List[Branch] = []
        for item in branches:
            if isinstance(item, Branch):
                self._branches.append(item)
            elif isinstance(item, Step):
                self._branches.append(Branch(item))  # output_key=None → 自动 merge
            else:
                raise TypeError(
                    f"Parallel 需要 Step 或 Branch，拿到 {type(item).__name__}"
                )
        if not self._branches:
            raise ValueError("Parallel 至少需要一个分支")
        self.max_workers = max_workers
        self.stop_on_error = stop_on_error

    # ---------------- execute ----------------
    def execute(self, context: Context) -> None:
        workers = self.max_workers or len(self._branches)

        if len(self._branches) == 1:
            self._run_one(self._branches[0], context)
            return

        errors: List[tuple] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._run_one_safe, branch, context): branch
                for branch in self._branches
            }

            for future in as_completed(futures):
                branch = futures[future]
                exc = future.exception()
                if exc is not None:
                    errors.append((branch, exc))
                    if self.stop_on_error:
                        for f in futures:
                            f.cancel()

        if errors and self.stop_on_error:
            branch, exc = errors[0]
            raise RuntimeError(
                f"Parallel 分支 {branch.describe()} 失败（stop_on_error=True）"
            ) from exc

        context.trace.append(f"{self.name}:{len(self._branches)}")

    def _run_one_safe(self, branch: Branch, parent_ctx: Context) -> None:
        '''供线程池调用的安全版：吞掉顶层异常，让 execute() 统一处理。'''
        try:
            self._run_one(branch, parent_ctx)
        except Exception:  # noqa: BLE001 —— 在线程池里只做捕获，不处理
            raise

    def _run_one(self, branch: Branch, parent_ctx: Context) -> None:
        '''真正跑一个分支 + 子 ctx → 父 ctx 的 merge。'''
        sub_ctx = Context(
            data=dict(parent_ctx.data),
            trace=[],
            types=dict(parent_ctx.types),
        )
        # function_stack 不拷贝：子 Workflow 有自己的函数作用域

        execute_step(branch.step, sub_ctx)

        if branch.output_key is not None:
            # 显式标 key：优先取 Export() 存的 _export，其次取子 ctx 同名 key
            if "_export" in sub_ctx.data:
                parent_ctx[branch.output_key] = sub_ctx["_export"]
            elif branch.output_key in sub_ctx.data:
                parent_ctx[branch.output_key] = sub_ctx[branch.output_key]
        else:
            # 裸 Step：merge 所有非内部 key（_export 是内部协议，不对外 merge）
            for key, value in sub_ctx.data.items():
                if key.startswith("_"):
                    continue
                parent_ctx[key] = value

    # ---------------- 展示 ----------------
    def children(self) -> List[Step]:
        return [b.step for b in self._branches]

    def describe(self) -> str:
        keys = [
            b.output_key if b.output_key is not None else "auto"
            for b in self._branches
        ]
        suffix = f", max_workers={self.max_workers}" if self.max_workers else ""
        return f"Parallel({', '.join(map(str, keys))}{suffix})"

    def __repr__(self) -> str:
        return (
            f"Parallel({', '.join(b.describe() for b in self._branches)}, "
            f"max_workers={self.max_workers!r}, stop_on_error={self.stop_on_error})"
        )
