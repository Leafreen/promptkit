# skill/chat.py
'''把 Skill / DSL `Function` 挂到 `ChatWorkflow(skills=[...])` 上。

聊天里模型只能看到「工具」：名字 + 描述 + 参数 Schema。所以这里做两件事：

1. 把 `FunctionDef`（DSL 函数）和 `Skill`（`@skill` 装饰过的）统一包成 `Tool`
2. 工具执行时注入**这一次运行用的 agent** —— 通过 `AgentHolder`，
   由 `ChatWorkflow.build()` 里的一步 `FunctionStep` 把 `context["agent"]` 放进去
'''
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from promptkit.core.context import Context
from promptkit.core.steps.logics.function import BoundStep, FunctionDef
from promptkit.skill.context import SkillContext
from promptkit.skill.errors import SkillError
from promptkit.skill.schema import validate_args
from promptkit.skill.spec import Skill, build_spec
from promptkit.skill.tools import result_text
from promptkit.tools import Tool

__all__ = ["AgentHolder", "as_tools", "function_parameters", "inherited_state"]

#: 这几个 key 不参与「状态继承」：要么由框架注入，要么是内部状态
INHERIT_SKIP = ("agent", "memory", "chat_tools", "_verbosity")


class AgentHolder:
    '''在「编译期建的工具」和「运行时才知道的 agent / memory」之间传一手。

    默认兜底用 `Agent.default()`，所以 `wf.run()` 不带 agent 也能用。
    `memory` 由 `ChatWorkflow(memory=...)` 在编译时塞进来 —— 工具执行时拿不到
    Context，只能靠这个中转。
    '''

    def __init__(self, agent: Any = None, memory: Any = None, state: Any = None):
        self.agent = agent
        self.memory = memory
        #: 调用方的状态快照（`ChatWorkflow` 每轮更新一次）
        self.state: Dict[str, Any] = dict(state or {})

    def update(self, context: Any) -> Any:
        agent = context.get("agent") if hasattr(context, "get") else None
        if agent is not None:
            self.agent = agent
        memory = context.get("memory") if hasattr(context, "get") else None
        if memory is not None:
            self.memory = memory
        return self.agent

    def snapshot(self, context: Any) -> Dict[str, Any]:
        '''把调用方的 Context 记一份，供技能「继承状态」使用。'''
        if hasattr(context, "to_dict"):
            data = context.to_dict()
        elif isinstance(context, dict):
            data = dict(context)
        else:
            data = {}
        self.state = {
            key: value
            for key, value in data.items()
            if key not in INHERIT_SKIP
        }
        return self.state

    def peek(self) -> Any:
        '''不强制要 agent：纯函数 Skill 用不上模型。'''
        return self.agent

    def get(self) -> Any:
        if self.agent is None:
            from promptkit.agent import Agent

            self.agent = Agent.default()
        if self.agent is None:
            raise SkillError(
                "调用这个 Skill 需要模型，但拿不到 agent："
                "用 wf.run(agent=Agent())，或者先创建一个默认 Agent"
            )
        return self.agent


def as_tools(skills: Iterable[Any], *, holder: Optional[AgentHolder] = None) -> List[Tool]:
    '''把 `FunctionDef` / 已绑定参数的 `BoundStep` / `Skill` / 函数变成 `Tool`。'''
    resolved = holder or AgentHolder()
    tools: List[Tool] = []
    for item in _flatten(skills):
        if isinstance(item, Tool):
            tools.append(item)
        elif isinstance(item, FunctionDef):
            tools.append(_FunctionTool(item, resolved))
        elif isinstance(item, BoundStep):
            tools.append(_BoundFunctionTool(item, resolved))
        elif isinstance(item, Skill):
            tools.append(_SkillTool(item, resolved))
        elif callable(item):
            tools.append(_SkillTool(Skill(build_spec(item)), resolved))
        else:
            raise TypeError(
                "skills 里只能是 FunctionDef / BoundStep / Skill / 函数 / Tool，"
                f"得到 {type(item).__name__}"
            )
    return tools


def inherited_state(inherit: Any, state: Mapping[str, Any]) -> Dict[str, Any]:
    '''按 `inherit` 的声明，从状态快照里挑出要继承的键。

    - `True`：除框架内部键外全部继承
    - `("profile", "topics")`：只继承这些（没有的跳过）
    - `False`：一个都不继承
    '''
    if not inherit:
        return {}
    if inherit is True:
        return {key: value for key, value in state.items() if key not in INHERIT_SKIP}
    keys = (inherit,) if isinstance(inherit, str) else tuple(inherit)
    return {key: state[key] for key in keys if key in state}


def function_parameters(definition: FunctionDef) -> Dict[str, Any]:
    '''DSL `Function(args=("a", "b"))` 的参数 Schema：都是必填字符串。'''
    properties = {
        name: {"type": "string", "description": f"参数 {name}"}
        for name in definition.args
    }
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if definition.args:
        schema["required"] = list(definition.args)
    return schema


class _FunctionTool(Tool):
    '''`Function(args=...).Do(...)` 的 Tool 视图。'''

    def __init__(self, definition: FunctionDef, holder: AgentHolder):
        self.definition = definition
        self.holder = holder
        super().__init__(
            name=definition.name,
            description=definition.description
            or f"{definition.name}（参数：{', '.join(definition.args) or '无'}）",
            parameters=function_parameters(definition),
            func=self.run,
        )

    def run(self, **arguments: Any) -> str:
        try:
            bound = self.definition(**arguments)      # 缺参 / 多参在这里就报
            return _run_bound(bound, self.holder)
        except SkillError as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001 - 工具的契约是「总能返回」
            return f"错误：{type(exc).__name__}: {exc}"


class _BoundFunctionTool(Tool):
    '''已经绑好参数的 `Function(...)`：变成一个无参工具。'''

    def __init__(self, bound: BoundStep, holder: AgentHolder):
        self.bound = bound
        self.holder = holder
        definition = bound.definition
        super().__init__(
            name=definition.name,
            description=definition.description or f"{definition.name}（参数已绑定）",
            parameters={"type": "object", "properties": {}},
            func=self.run,
        )

    def run(self) -> str:
        try:
            return _run_bound(self.bound, self.holder)
        except SkillError as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001
            return f"错误：{type(exc).__name__}: {exc}"


def _run_bound(bound: BoundStep, holder: AgentHolder) -> str:
    definition = bound.definition
    context = Context()
    context.data.update(inherited_state(definition.inherit, holder.state))
    context["agent"] = holder.get()
    if holder.memory is not None:
        # 函数体里的 FunctionStep 可以直接 ctx["memory"] 用长期记忆
        context["memory"] = holder.memory
    bound.execute(context)
    value = context.get(definition.output_key)
    if value is None:
        value = _last_body_output(definition, context)
    return result_text(value) if value is not None else "（没有返回值）"


class _SkillTool(Tool):
    '''`@skill` 函数的 Tool 视图：执行时用 holder 里的 agent 造 SkillContext。'''

    def __init__(self, skill: Skill, holder: AgentHolder):
        self.skill = skill
        self.holder = holder
        super().__init__(
            name=skill.name,
            description=skill.description,
            parameters=skill.parameters,
            func=self.run,
        )

    def run(self, **arguments: Any) -> str:
        try:
            validate_args(self.skill.spec, arguments)
            skill_context = SkillContext(
                agent=self.holder.peek(),
                memory=self.holder.memory,
                state=dict(self.holder.state),
            )
            return result_text(self.skill.spec.invoke(skill_context, **arguments))
        except SkillError as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001
            return f"错误：{type(exc).__name__}: {exc}"


def _last_body_output(definition: FunctionDef, context: Context) -> Any:
    '''函数体自己指定了 `output_key`（不是函数名）时，取最后一个写进去的值。

        Function(args=..., name="make_lab").Do(LLMStep(..., output_key="output"))

    `Return(...)` 那条路已经写进 `definition.output_key`，不会走到这儿。
    '''
    for step in reversed(getattr(definition, "body", []) or []):
        key = getattr(step, "output_key", None)
        if isinstance(key, str) and key in context.data:
            return context.data[key]
    return None


def _flatten(items: Iterable[Any]) -> Iterator[Any]:
    for item in items or ():
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item
