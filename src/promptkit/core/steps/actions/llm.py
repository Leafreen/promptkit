# core/steps/llm.py
'''调用大模型、并把结果写回 Context 的步骤。'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, List, Optional, Sequence, Type, TypeVar, cast

from promptkit.core.model_action.caution_related.caution import (
    Caution,
    WORKFLOW_CAUTIONS_KEY,
    join_cautions,
    resolve_cautions,
)
if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.event_sys.events import bus_of
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step
from promptkit.prompt import Prompt
from promptkit.types import ToolLike

if TYPE_CHECKING:  # 仅为类型标注，避免 core 强依赖 pydantic
    from pydantic import BaseModel

    from promptkit.core.event_sys import EventBus
    from promptkit.core.model_action.caution_related.formatter import Formatter
    from promptkit.types import AgentLike

#: 输出类型参数：`LLMStep[Intent](...)`，让 IDE / mypy 知道这一步产出什么
T = TypeVar("T", bound="BaseModel")


@dataclass
class LLMStep(Step, Generic[T]):
    '''执行一个 Prompt，把结果写进 context[output_key]。

        # 纯文本：context["output"] = 模型回复
        LLMStep(Prompt(role="user", content="你好"))

        # 结构化输出：context["intent"] = Intent 实例
        LLMStep[Intent](
            Prompt(role="user", content="判断意图：{{text}}"),
            Intent,
            output_key="intent",
            cautions=[JSON_ONLY],
        )

    三件事各管一段：

    - `prompt`：任务本身
    - `cautions`：事前引导（追加到 user 消息末尾，降低格式出错率）
    - `output_model` + `formatter`：事后解析与校验（错了就重试，兜底）

    提示词里的变量 {{text}} 会从 Context 里取值。Agent 的获取顺序：
    构造时传入的 agent= ，否则用 context["agent"]（可写在 Workflow.run(agent=...)）。
    '''

    prompt: Prompt
    output_model: Optional[Type["BaseModel"]] = None
    output_key: str = "output"
    agent: object = None
    reset: bool = False
    max_retries: Optional[int] = None      # None = 用 settings.max_retries
    tools: Optional[Sequence[ToolLike]] = None  # 工具：Tool / 函数 / API 原生 dict

    #: 约束：Caution / 字符串 / 它们的列表 / 组合，都会被展平
    cautions: Optional[Sequence[object]] = None
    #: 事后解析器：Formatter 实例，或 "json" / "xml" / "yaml" / "regex:<pattern>"
    formatter: object = None
    #: 自动把 output_model 的 JSON Schema 作为约束加进去（会多花 token）
    #: None = 用 settings.auto_schema
    auto_schema: Optional[bool] = None
    #: 是否合并全局默认约束（set_default_cautions 设的）
    use_default_cautions: bool = True

    name: str = "llm"

    def auto_schema_enabled(self) -> bool:
        '''这次要不要自动注入 JSON Schema（显式值优先，否则取全局设置）。'''
        return settings.auto_schema if self.auto_schema is None else self.auto_schema

    def retry_count(self) -> int:
        '''实际生效的重试次数。'''
        return settings.max_retries if self.max_retries is None else self.max_retries

    # ---------------- 提示词 ----------------
    def build_prompt(self, context: Optional[Context] = None) -> Prompt:
        '''把约束追加到 Prompt 内容末尾，返回一个新的 Prompt。

        优先级（低 -> 高，越靠后越「就近」）：全局默认 -> Workflow -> Step。
        文本完全相同的约束会去重，所以「全局 + 预置组合」撞车也不会重复。
        '''
        workflow_cautions = (
            context.get(WORKFLOW_CAUTIONS_KEY) if context is not None else None
        )
        cautions = resolve_cautions(
            self.cautions,
            workflow_cautions,
            use_defaults=self.use_default_cautions,
        )

        if self.auto_schema_enabled() and self.output_model is not None:
            from promptkit.cautions.builtin import JsonSchemaHint

            cautions = [*cautions, JsonSchemaHint(self.output_model)]

        text = join_cautions(cautions)
        if not text:
            return self.prompt
        return Prompt(
            role=self.prompt.role,
            content=f"{self.prompt.content}\n\n{text}",
        )

    # 旧名字，保留可用
    _build_prompt = build_prompt

    def resolved_cautions(self, context: Optional[Context] = None) -> List[Caution]:
        '''真正会被拼进去的约束列表（调试 / 测试用）。'''
        workflow_cautions = (
            context.get(WORKFLOW_CAUTIONS_KEY) if context is not None else None
        )
        cautions = resolve_cautions(
            self.cautions,
            workflow_cautions,
            use_defaults=self.use_default_cautions,
        )
        if self.auto_schema_enabled() and self.output_model is not None:
            from promptkit.cautions.builtin import JsonSchemaHint

            cautions = [*cautions, JsonSchemaHint(self.output_model)]
        return cautions

    # ---------------- 执行 ----------------
    def execute(self, context: Context) -> None:
        # 延迟导入：让 core 在没有安装 pydantic / openai 时也能正常 import
        from promptkit.executor import Executor

        agent = self.agent if self.agent is not None else context.get("agent")
        if agent is None:
            # 兜底：用 Agent() 自动注册的默认实例
            try:
                from promptkit.agent import Agent

                agent = Agent.default()
            except ImportError:
                agent = None
        if agent is None:
            raise ValueError(
                "LLMStep 需要一个 agent：先执行 `agent = Agent()`，"
                "或构造时传 agent=，或 Workflow.run(agent=...)"
            )

        variables = context.to_dict()

        if self.reset:
            getattr(agent, "reset")()

        # 把工作流上的事件总线交给模型层，这样「思考/工具调用/回复」
        # 也能在 Workflow.run(bus=...) 这一层看到
        bus: Optional["EventBus"] = bus_of(context)

        prompt = self.build_prompt(context)

        if self.output_model is None:
            value: object = getattr(agent, "ask")(
                prompt.render(**variables), tools=self.tools, bus=bus
            )
        else:
            value = Executor(
                agent=cast("AgentLike", agent),
                max_retries=self.retry_count(),
                formatter=cast("Formatter | str | None", self.formatter),
            ).run(prompt, self.output_model, tools=self.tools, bus=bus, **variables)

        context[self.output_key] = value
        context.trace.append(self.name)

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        from promptkit.core.display_sys.tree import short

        model = f" -> {self.output_model.__name__}" if self.output_model else ""
        tools = ""
        if self.tools:
            names = ", ".join(
                str(getattr(t, "name", None) or getattr(t, "__name__", "?"))
                for t in self.tools
            )
            tools = f" [tools: {names}]"

        bits = []
        count = len(list(self.cautions or ()))
        if count:
            bits.append(f"{count} caution" + ("s" if count > 1 else ""))
        if self.auto_schema_enabled() and self.output_model is not None:
            bits.append("auto-schema")
        if self.formatter is not None:
            from promptkit.core.model_action.caution_related.formatter import get_formatter

            bits.append(get_formatter(self.formatter).name)
        extra = f" [{' + '.join(bits)}]" if bits else ""

        return (
            f"LLM({short(self.prompt.content)}){model}{tools}{extra}"
            f" -> {self.output_key}"
        )

__all__ = ["LLMStep", "T"]
