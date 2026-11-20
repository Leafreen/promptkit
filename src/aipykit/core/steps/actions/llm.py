# core/steps/llm.py
'''调用大模型、并把结果写回 Context 的步骤。'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, List, Mapping, Optional, Sequence, Type, TypeVar, cast

from aipykit.core.model_action.caution_related.caution import (
    Caution,
    WORKFLOW_CAUTIONS_KEY,
    apply_cautions,
    resolve_cautions,
)
if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from aipykit.core.context import Context
from aipykit.core.event_sys.events import bus_of
from aipykit.core.settings import settings
from aipykit.core.steps.interface import Step
from aipykit.prompt import Prompt

if TYPE_CHECKING:  # 仅为类型标注，避免 core 强依赖 pydantic
    from pydantic import BaseModel

    from aipykit.core.event_sys import EventBus
    from aipykit.core.model_action.caution_related.formatter import Formatter
    from aipykit.types import AgentLike, ToolLike

#: 输出类型参数：`LLMStep[Intent](...)`，让 IDE / mypy 知道这一步产出什么
T = TypeVar("T", bound="BaseModel")

#: Context 里的流式开关键名：`EnableTUI()` 会把它设成 True，
#: `LLMStep.stream=None` 时读这个值决定要不要走流式。
LLM_STREAM_KEY = "_llm_stream"


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
    #: 是否流式调用模型：True=流式，False=一次性，None=读 Context 里的
    #: `_llm_stream`（`EnableTUI()` 会自动设成 True）。只有纯文本输出
    #: 生效，`output_model` 结构化输出不支持流式。
    stream: Optional[bool] = None
    #: 临时步骤：True 表示这次调用**不写进对话历史**。开场白、总结、意图识别
    #: 这类「工作流自己说的话」不该出现在主对话里 —— 否则模型下一轮会看到
    #: 一串伪造的 user 消息。回复仍照常写进 `output_key`，只是历史被还原。
    temp: bool = False

    name: str = "llm"

    def auto_schema_enabled(self) -> bool:
        '''这次要不要自动注入 JSON Schema（显式值优先，否则取全局设置）。'''
        return settings.auto_schema if self.auto_schema is None else self.auto_schema

    def retry_count(self) -> int:
        '''实际生效的重试次数。'''
        return settings.max_retries if self.max_retries is None else self.max_retries

    def _resolve_stream(self, context: Optional[Context]) -> bool:
        '''真正生效的流式开关：显式值 -> Context 标记 -> False。'''
        if self.stream is not None:
            return self.stream
        if context is not None:
            return bool(context.get(LLM_STREAM_KEY))
        return False

    # ---------------- 提示词 ----------------
    def build_prompt(self, context: Optional[Context] = None) -> Prompt:
        '''把约束追加到 Prompt 内容末尾，返回一个新的 Prompt。

        优先级（低 -> 高，越靠后越「就近」）：全局默认 -> Workflow -> Step。
        文本完全相同的约束会去重，所以「全局 + 预置组合」撞车也不会重复。
        '''
        workflow_cautions = (
            context.get(WORKFLOW_CAUTIONS_KEY) if context is not None else None
        )
        extra: List[object] = []
        if self.auto_schema_enabled() and self.output_model is not None:
            from aipykit.cautions.builtin import JsonSchemaHint

            extra.append(JsonSchemaHint(self.output_model))
        return apply_cautions(
            self.prompt,
            self.cautions,
            workflow_cautions,
            use_defaults=self.use_default_cautions,
            extra=extra,
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
            from aipykit.cautions.builtin import JsonSchemaHint

            cautions = [*cautions, JsonSchemaHint(self.output_model)]
        return cautions

    # ---------------- 执行 ----------------
    def execute(self, context: Context) -> None:
        # 延迟导入：让 core 在没有安装 pydantic / openai 时也能正常 import
        from aipykit.executor import Executor

        agent = self.agent if self.agent is not None else context.get("agent")
        if agent is None:
            # 兜底：用 Agent() 自动注册的默认实例
            try:
                from aipykit.agent import Agent

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
        rendered = prompt.render(**variables)
        self._check_modalities(agent, rendered)

        # temp：执行完把历史还原到调用前（回复本身不受影响）。
        # 放在 try/finally 里，模型报错 / 解析失败也能还原，不留半条消息。
        snapshot = _snapshot_history(agent) if self.temp else None
        try:
            if self.output_model is None:
                value: object = getattr(agent, "ask")(
                    rendered,
                    tools=self.tools,
                    bus=bus,
                    stream=self._resolve_stream(context),
                )
            else:
                value = Executor(
                    agent=cast("AgentLike", agent),
                    max_retries=self.retry_count(),
                    formatter=cast("Formatter | str | None", self.formatter),
                ).run(prompt, self.output_model, tools=self.tools, bus=bus, **variables)
        finally:
            if snapshot is not None:
                _restore_history(agent, snapshot)

        context[self.output_key] = value
        context.trace.append(self.name)

    # ---------------- 能力协商 ----------------
    @staticmethod
    def _check_modalities(agent: object, rendered: Mapping[str, Any]) -> None:
        '''多模态协商：消息里每个内容部件都要求 client 支持对应模态。

        不支持就抛 `UnsupportedModality`，不静默降级成文本。
        content 是纯字符串（老用法）时什么都不做。
        '''
        content = rendered.get("content")
        if not isinstance(content, list):
            return
        client = getattr(agent, "client", None)
        supports = getattr(client, "supports", None)
        if not callable(supports):
            # 老式假客户端（SimpleNamespace 之类）没有 supports，不拦；
            # 真实客户端都继承 BaseClient，默认只声明 text
            return
        from aipykit.client_sys.client import UnsupportedModality

        for part in content:
            if not (isinstance(part, dict) and "source" in part):
                continue
            modality = str(part.get("type") or "")
            if modality and not supports(modality):
                vendor = getattr(client, "vendor", None) or type(client).__name__
                raise UnsupportedModality(str(vendor), modality)

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        from aipykit.core.display_sys.tree import short

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
            from aipykit.core.model_action.caution_related.formatter import get_formatter

            bits.append(get_formatter(self.formatter).name)
        if self.stream is True:
            bits.append("stream")
        if self.temp:
            bits.append("temp")
        extra = f" [{' + '.join(bits)}]" if bits else ""

        content = self.prompt.content
        summary = content if isinstance(content, str) else f"[多模态 {len(content)} 部件]"
        return f"LLM({short(summary)}){model}{tools}{extra} -> {self.output_key}"


def _snapshot_history(agent: object) -> Optional[List[object]]:
    '''给 temp 步骤拍一份历史快照，执行完用它回滚。

    优先用 `agent.session.messages`：那是会话层的不可变 Message，回滚后
    **id 不变**，不会让外面已经持有的消息引用失效。没有 session 的轻量
    Agent（测试里的假 agent）退回 `get_history()` / 普通 list `history`。
    '''
    session = getattr(agent, "session", None)
    messages = getattr(session, "messages", None) if session is not None else None
    if isinstance(messages, list):
        return list(messages)
    getter = getattr(agent, "get_history", None)
    if callable(getter):
        return list(getter())
    history = getattr(agent, "history", None)
    if isinstance(history, list):
        return list(history)
    return None


def _restore_history(agent: object, snapshot: List[object]) -> None:
    '''把历史还原成 snapshot（`Agent.history` 的 setter 会同步 messages 缓存）。'''
    try:
        setattr(agent, "history", snapshot)
    except (AttributeError, TypeError):
        # agent 没有可写 history：无法回滚，但别让 temp 反而把调用搞崩
        return


__all__ = ["LLMStep", "LLM_STREAM_KEY", "T"]
