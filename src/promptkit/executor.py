# executor.py
'''把 Prompt 交给 Agent 执行，并把模型输出解析成 Pydantic 对象。'''
import json
from typing import Any, Optional, Sequence, Type

from pydantic import BaseModel, ValidationError

from promptkit.core.model_action.caution_related.formatter import (
    Formatter,
    FormatterError,
    get_formatter,
)
from promptkit.core.settings import settings
from promptkit.prompt import Prompt


class OutputValidationError(FormatterError):
    '''模型的输出无法解析成目标模型（`FormatterError` 的别名语义，保留旧名字）。'''


class MaxRetriesExceeded(Exception):
    '''重试若干次后仍然没有得到合法输出。'''


def run_structured(
    agent: Any,
    text: str,
    output_model: Type[BaseModel],
    *,
    tools: Optional[Sequence[Any]] = None,
    bus: Any = None,
    max_retries: Optional[int] = None,
) -> BaseModel:
    '''把一段**已经是数据**的文本发给 agent，解析成 `output_model`。

    和 `Executor.run` 的区别只有一个，但很关键：**不经过 `Prompt.render()`**。
    编排层和技能层经常把上游结果 / 用户输入拼进提示词，那些文本里可能带
    `{{...}}`，再渲染一遍会被当成模板变量而抛 `KeyError`。

    解析失败会带着错误重试，重试次数默认取 `settings.max_retries`。
    '''
    formatter = get_formatter(None)
    limit = settings.max_retries if max_retries is None else int(max_retries)
    correction: Optional[str] = None
    last_error: Optional[Exception] = None

    for _ in range(limit + 1):
        raw = agent.ask(
            {"role": "user", "content": correction or text},
            tools=tools,
            bus=bus,
        )
        try:
            return output_model.model_validate(formatter.parse(raw))
        except Exception as exc:      # noqa: BLE001 - 解析/校验失败都值得重试
            last_error = exc
            correction = (
                f"你上一次的输出无法使用：{exc}\n"
                f"请只输出{formatter.name}格式的内容，不要解释、不要代码块。"
            )

    raise MaxRetriesExceeded(f"重试 {limit} 次后仍未得到合法输出：{last_error}")


class Executor:
    '''「发提示词 -> 解析 -> 校验 -> 失败重试」的小循环。'''

    def __init__(
        self,
        agent: Any,
        max_retries: int = 2,
        formatter: Any = None,
    ):
        self.agent = agent
        self.max_retries = max_retries
        self.formatter = formatter

    def _resolve_formatter(self, formatter: Any = None) -> Formatter:
        return get_formatter(self.formatter if formatter is None else formatter)

    def run(
        self,
        prompt: Prompt,
        output_model: Type[BaseModel],
        tools: Optional[Sequence[Any]] = None,
        bus: Optional[Any] = None,
        formatter: Any = None,
        **kwargs: Any,
    ) -> BaseModel:
        '''运行一个 Prompt，返回一个 BaseModel 对象。

        - `tools` 会透传给 Agent，所以「先调工具拿数据，再输出结构化结果」也能用
        - `bus` 用来把执行过程中的事件接出去
        - `formatter` 决定怎么把原始文本变成 dict，默认 `JsonFormatter`
        '''
        message = prompt.render(**kwargs)
        active = self._resolve_formatter(formatter)
        correction = None
        last_error: Optional[Exception] = None

        for _ in range(self.max_retries + 1):
            # 第一次发送渲染好的消息；之后发送纠错提示，
            # 由 Agent 负责把上一轮的错误输出和纠错提示都记入上下文。
            raw = self.agent.ask(
                message if correction is None else correction,
                tools=tools,
                bus=bus,
            )
            try:
                return output_model.model_validate(active.parse(raw))
            except (FormatterError, ValidationError) as exc:
                last_error = exc
                correction = (
                    f"你上一次的输出无法使用：{exc}\n"
                    f"请只输出{active.name}格式的内容，"
                    "不要包含 Markdown 代码块、解释或多余文字。"
                )

        raise MaxRetriesExceeded(
            f"重试 {self.max_retries} 次后仍未得到合法输出：{last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> Any:
        '''从模型输出里尽力抽取 JSON（默认格式化器的快捷方式）。'''
        try:
            return get_formatter(None).parse(text)
        except FormatterError as exc:
            raise OutputValidationError(str(exc)) from None

    @staticmethod
    def _try_loads(text: str):
        '''保留的旧工具函数。'''
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, exc
