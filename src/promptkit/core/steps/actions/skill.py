# core/steps/actions/skill.py
'''`SkillStep`：在 DSL 工作流里调用一个 Skill。

`FunctionStep` 是 Step（调用约定是 `func(context)`，只有一个参数），Skill 是
「按名字 + 参数调用」的能力单元 —— 所以 `@skill` 套 `FunctionStep` 不成立
（Step 不可调用，也没有参数签名）。要在工作流里用 Skill，用这个 Step：

    Workflow(
        SkillStep(summarize, text=Var("text"), output_key="summary"),
        Echo("摘要：{{summary}}"),
    )

它会把 workflow 的 agent 传进 `SkillContext`，所以 Skill 内部的 `ctx.ask` /
`ctx.structured` / `ctx.run` 用的是同一个模型，而不是「没有 agent」。
'''
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Mapping,
    Optional,
    cast,
)

from promptkit.core.steps.interface import Step
from promptkit.skill.context import SkillContext
from promptkit.skill.errors import SkillNotFound
from promptkit.skill.schema import validate_args
from promptkit.skill.spec import Skill, build_spec

if TYPE_CHECKING:      # 只为类型标注，不在运行时依赖
    from promptkit.core.context import Context

__all__ = ["SkillStep"]

#: 这几个关键字是 SkillStep 自己的参数，不能当 Skill 的入参
_RESERVED = ("inputs", "output_key", "registry", "ctx", "name", "skill")


class SkillStep(Step):
    '''把一个 Skill 当成工作流里的一步。

        SkillStep(summarize, text=Var("text"), output_key="summary")
        SkillStep("summarize", text=Var("text"), registry=registry)   # 按名字
        SkillStep(research, **{"query": Var("q")})                    # 参数名撞车时用 dict

    - `skill`：`Skill`（`@skill` 装饰过的）/ 普通函数 / 注册表里的名字（配合 `registry=`）
    - 其余关键字都是 Skill 的入参：值可以是字面量，也可以是表达式
      （`Var("text")`，每一轮重新求值）
    - `output_key`：结果写回哪儿；不给就用 Skill 的名字
    - `registry`：给了就按名字解析，并走 `registry.execute`（校验 + 轨迹都在那边）
    - `ctx`：想自己控制 `SkillContext` 时才传；默认从当前 Context 的 agent 造一个
    '''

    name: str = "skill"

    def __init__(
        self,
        skill: object = None,
        *,
        inputs: Optional[Mapping[str, object]] = None,
        output_key: Optional[str] = None,
        registry: object = None,
        ctx: object = None,
        name: str = "skill",
        **arguments,
    ):
        if skill is None:
            raise ValueError("SkillStep 需要 skill=...（Skill、函数，或注册表里的名字）")
        duplicated = sorted(set(arguments) & set(inputs or {}))
        if duplicated:
            raise ValueError(
                f"这些参数同时出现在 inputs= 和关键字里：{duplicated}；留一处就行"
            )
        self.skill = skill
        self.inputs: Dict[str, object] = {**dict(inputs or {}), **arguments}
        self.output_key = output_key
        self.registry = registry
        self.ctx: Optional[SkillContext] = cast(Optional[SkillContext], ctx)
        self.name = name

    # ---------------- 执行 ----------------
    def execute(self, context: "Context") -> None:
        skill = self._resolve()
        values = {
            key: (value.eval(context) if hasattr(value, "eval") else value)
            for key, value in self.inputs.items()
        }

        if self.registry is not None:
            result = getattr(self.registry, "execute")(
                skill.name, values, ctx=self._context(context)
            )
        else:
            validate_args(skill.spec, values)
            result = skill.spec.invoke(self._context(context), **values)

        context[self.output_key or skill.name] = result
        context.trace.append(f"{self.name}:{skill.name}")

    # ---------------- 内部 ----------------
    def _resolve(self) -> Skill:
        if isinstance(self.skill, Skill):
            return self.skill
        if isinstance(self.skill, str):
            if self.registry is None:
                raise SkillNotFound(
                    f"SkillStep 用名字 {self.skill!r} 调 Skill，但没给 registry="
                )
            return getattr(self.registry, "get")(self.skill)
        if callable(self.skill):
            return Skill(build_spec(cast(Callable[..., object], self.skill)))
        raise TypeError(
            "SkillStep 的 skill 只能是 Skill / 函数 / 名字，"
            f"得到 {type(self.skill).__name__}"
        )

    def _context(self, context: "Context") -> SkillContext:
        if self.ctx is not None:
            return self.ctx
        return SkillContext(agent=context.get("agent"), skills=self.registry)

    def describe(self) -> str:
        label = getattr(self.skill, "name", self.skill)
        args = ", ".join(f"{key}={_short(value)}" for key, value in self.inputs.items())
        return f"SkillStep({label}({args}) -> {self.output_key or label})"

    def __repr__(self) -> str:
        return self.describe()


def _short(value: object) -> str:
    '''显示成 `text=source` 这种短样子（表达式就用它的变量名 / 表达式文本）。'''
    if isinstance(value, str):
        return repr(value)
    for attribute in ("var", "name", "key", "text"):
        text = getattr(value, attribute, None)
        if isinstance(text, str) and text:
            return text
    return repr(value)
