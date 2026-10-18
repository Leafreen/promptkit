# core/steps/function.py
'''步骤工厂：Function(args=...).Do(...)、Return(...)，以及最基础的 FunctionStep。'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.errors import ArgumentError, ReturnSignal
from promptkit.core.steps.logics.expr_sys.expr import AgentReplyExpr, Expr, as_expr
from promptkit.core.steps.interface import Step, coerce_steps, execute_step
from promptkit.core.steps.actions.llm import LLMStep

if TYPE_CHECKING:  # 仅为类型标注，避免 core 强依赖 pydantic
    from pydantic import BaseModel

from promptkit.prompt import Prompt

_MISSING = object()


@dataclass
class FunctionStep(Step):
    '''执行一个 Python 函数，可选地把返回值写回 Context。

        FunctionStep(lambda ctx: print(ctx["text"]))              # 只产生副作用
        FunctionStep(lambda ctx: "可以退款。", output_key="reply")  # 写回变量
    '''

    func: Callable[[Context], Any]
    output_key: Optional[str] = None
    name: str = "function"

    def execute(self, context: Context) -> None:
        result = self.func(context)
        if self.output_key is not None:
            context[self.output_key] = result
        context.trace.append(self.name)

    def describe(self) -> str:
        func_name = getattr(self.func, "__name__", type(self.func).__name__)
        suffix = f" -> {self.output_key}" if self.output_key else ""
        return f"FunctionStep({func_name}){suffix}"


class FunctionDef:
    '''`Function(args=...).Do(...)` 的结果：一个可复用的、带参数的步骤定义。

        Ask = Function(args=("job",), name="ask").Do(
            Prompt(role="user", content="面试一个{{job}}岗位，问一个问题。"),
        )
        Ask(job="后端工程师")          # -> BoundStep，可以放进 Workflow
    '''

    def __init__(
        self,
        args: tuple = (),
        *,
        name: str = "function",
        output_key: Optional[str] = None,
        output_model: Optional[type] = None,
        tools: Optional[List[Any]] = None,
    ):
        self.args = tuple(args)
        if any(not isinstance(a, str) or not a for a in self.args):
            raise ArgumentError(f"args 必须是非空字符串序列，得到 {args!r}")
        if len(set(self.args)) != len(self.args):
            raise ArgumentError(f"args 有重复：{self.args!r}")

        self.name = name
        self.output_key = output_key or name
        self.output_model = output_model
        self.tools = list(tools) if tools else None
        self.body: List[Step] = []

    def Do(self, *body: Any) -> "FunctionDef":  # noqa: N802 - 跟随 DSL 命名
        '''设置步骤体：可以是一个 Prompt，也可以是若干步骤。'''
        steps: List[Step] = []
        for item in body:
            if isinstance(item, Prompt):
                steps.append(
                    LLMStep(
                        item,
                        self.output_model,
                        self.output_key,
                        tools=self.tools,
                        name=self.name,
                    )
                )
            else:
                steps.extend(coerce_steps(item))
        if not steps:
            raise ArgumentError(f"{self.name}: Do() 至少要有一个步骤或 Prompt")
        self.body = steps
        return self

    def __call__(self, **kwargs: Any) -> "BoundStep":
        '''绑定参数，得到一个可执行 / 可当表达式用的步骤。'''
        unknown = sorted(set(kwargs) - set(self.args))
        if unknown:
            raise ArgumentError(
                f"{self.name}() 不接受的参数 {unknown}；"
                f"它声明的参数是 {list(self.args)}"
            )
        missing = [a for a in self.args if a not in kwargs]
        if missing:
            raise ArgumentError(
                f"{self.name}() 缺少参数 {missing}；"
                f"它声明的参数是 {list(self.args)}"
            )
        return BoundStep(self, dict(kwargs))

    def __repr__(self) -> str:
        return f"<Function {self.name}({', '.join(self.args)})>"


class ReturnStep(Step):
    '''结束当前函数体，并把一个值作为该函数的返回值。

        Add = Function(args=("a", "b"), name="add").Do(
            Return(Var("a") + Var("b")),
        )
        Echo("和 = ", Add(a=1, b=2).result())

    返回值既可以用 `.result()` 当表达式读，也会写进函数的 output_key，
    所以 `Var("add")` 也能读到。
    '''

    name = "return"

    def __init__(self, value: Any = None):
        self.value = as_expr(value)

    def execute(self, context: Context) -> None:
        context.trace.append(self.name)
        raise ReturnSignal(self.value.eval(context))

    def describe(self) -> str:
        return f"Return({self.value!r})"

    def __repr__(self) -> str:
        return f"Return({self.value!r})"


def Return(value: Any = None) -> ReturnStep:  # noqa: N802 - 跟随 DSL 命名
    '''Return(表达式)：结束函数体并返回这个值。'''
    return ReturnStep(value)


class BoundStep(Step):
    '''Function 定义绑定参数后的结果。

    - 当作 Step 用：执行函数体（参数临时注入 Context，执行完恢复）
    - 当作表达式用：`.result()`（别名 `.agent_reply()`）得到
      「运行它，然后取它的返回值」
    '''

    def __init__(self, definition: FunctionDef, kwargs: Dict[str, Any]):
        self.definition = definition
        self.kwargs = kwargs
        self.name = definition.name
        self.output_key = definition.output_key

    def execute(self, context: Context) -> None:
        # 参数只在函数体内部可见（词法作用域），执行完恢复原值
        saved: Dict[str, Any] = {}
        for key, raw in self.kwargs.items():
            # 参数可以是字面量，也可以是表达式：表达式在每次执行时重新求值，
            # 所以 While 循环里传 Var("x") 能拿到每一轮最新的值。
            value = raw.eval(context) if isinstance(raw, Expr) else raw
            saved[key] = context.get(key, _MISSING)
            context[key] = value

        returned: Any = _MISSING
        try:
            for step in self.definition.body:
                execute_step(step, context)
        except ReturnSignal as signal:
            returned = signal.value
        finally:
            for key, old in saved.items():
                if old is _MISSING:
                    context.data.pop(key, None)
                else:
                    context[key] = old

        # 写回是在参数恢复之后做的，避免 output_key 和参数同名时被覆盖
        if returned is not _MISSING:
            context[self.output_key] = returned

        context.trace.append(self.name)

    @property
    def result(self) -> AgentReplyExpr:
        '''作为表达式使用：先执行本函数，再取它的返回值。'''
        return AgentReplyExpr(self, self.output_key)

    def agent_reply(self) -> AgentReplyExpr:
        '''`result()` 的别名（针对以提示词为函数体的场景）。'''
        return self.result

    def describe(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.name}({args})"

    def children(self) -> List[Step]:
        return list(self.definition.body)

    def __repr__(self) -> str:
        return self.describe()


def Function(  # noqa: N802 - 跟随 DSL 命名
    args: tuple = (),
    *,
    name: str = "function",
    output_key: Optional[str] = None,
    output_model: Optional[type] = None,
    tools: Optional[List[Any]] = None,
) -> FunctionDef:
    '''定义一个带参数的步骤：Function(args=("job",)).Do(Prompt(...))。

    传了 tools 时，这个函数里的模型调用可以自己调用工具（function calling）。
    '''
    return FunctionDef(
        args=args,
        name=name,
        output_key=output_key,
        output_model=output_model,
        tools=tools,
    )
