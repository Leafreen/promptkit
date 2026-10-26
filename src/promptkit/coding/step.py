# coding/step.py
'''`CodingStep`：让模型产出**可执行的代码变更**，而不是给人看的文本。

它和 `LLMStep` 的分工：

| | `LLMStep` | `CodingStep` |
| --- | --- | --- |
| 输入 | 提示词 | 任务 + 相关文件 + 上次的失败 + 规范 |
| 输出 | 文本 / Pydantic 对象 | diff / 完整文件 / 多文件变更（**下一步会执行**） |
| 校验 | 解析格式 | 结构 + 可应用性 + 语法 |
| 重试 | 格式不对 | 校验不过 / 干跑装不上 |

关键就在「干跑」：给了 `path=` 时，`CodingStep` 会先读目标文件、在内存里
把补丁应用一遍，装不上就把错误塞回提示词重试 —— 而不是等到 `ApplyDiff`
把文件改坏才发现。
'''
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from promptkit.coding.errors import CodeFormatError, DiffError
from promptkit.coding.formats import CodeOutput, get_output
from promptkit.coding.sandbox import Sandbox
from promptkit.core.event_sys.events import bus_of
from promptkit.core.errors import UndefinedVariable
from promptkit.core.model_action.caution_related.caution import (
    join_cautions,
    resolve_cautions,
)
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context

__all__ = ["CodingStep"]


@dataclass
class CodingStep(Step):
    '''生成代码变更，校验之后写进 Context。

        CodingStep(
            task=Var("task"),
            context={"src/main.py": Var("main_source")},
            error=Var("test_result"),
            output_format="diff",          # diff / full_file / json
            output_key="patch",
            path="src/main.py",            # 给了就干跑校验
        )

    - `task`：任务描述（字符串或表达式）
    - `context`：相关文件，`{路径: 内容}`（内容可以是 `Var(...)`）
    - `error`：上一次的失败信息（测试输出 / 补丁错误），dict 也接受
    - `output_format`：`diff` 改现有文件 / `full_file` 新建或重写 / `json` 多文件
    - `path`：目标文件。给了就能在写盘前验证补丁装不装得上
    - `cautions`：编码规范（`Custom("不要改测试文件")` 之类）
    - `max_retries`：校验不过时最多再试几次（省略时用 `settings.max_retries`）
    '''

    task: Any
    context: Any = None
    error: Any = None
    output_format: str = "diff"
    #: 直接给一个现成的 `CodeOutput`（想自定义语法检查时用）
    output: Optional[Any] = None
    output_key: str = "patch"
    path: Optional[Any] = None
    base_dir: str = "."
    language: str = "python"
    cautions: Optional[Sequence[Any]] = None
    max_retries: Optional[int] = None
    #: 给了 path 时是否干跑校验（关掉就只做结构校验）
    verify: bool = True
    #: 塞进提示词的上次错误最多多少字符
    max_error_chars: int = 4000
    #: 相关文件总共最多塞多少字符（防止把整个仓库喂进去）
    max_context_chars: int = 40_000
    agent: Any = None
    name: str = "coding"

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = get_output(self.output_format)
        if not isinstance(self.output, CodeOutput):
            raise TypeError(
                f"output 需要 CodeOutput 实例，得到 {type(self.output).__name__}"
            )
        self.output_format = self.output.name
        if self.max_retries is not None and (
            not isinstance(self.max_retries, int) or self.max_retries < 0
        ):
            raise ValueError(
                f"max_retries 必须是非负整数，得到 {self.max_retries!r}"
            )
        if not self.output_key:
            raise ValueError("output_key 不能为空")

    # ---------------- 取值 ----------------
    def _agent(self, context: "Context") -> Any:
        agent = self.agent if self.agent is not None else context.get("agent")
        if agent is None:
            try:
                from promptkit.agent import Agent

                agent = Agent.default()
            except ImportError:  # pragma: no cover
                agent = None
        if agent is None:
            raise ValueError(
                "CodingStep 需要一个 agent：构造时传 agent=，"
                "或 Workflow.run(agent=...)，或先创建一个 Agent()"
            )
        return agent

    @staticmethod
    def _render(value: Any, context: "Context") -> Any:
        return value.eval(context) if isinstance(value, Expr) else value

    def _task_text(self, context: "Context") -> str:
        value = self._render(self.task, context)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return "" if value is None else str(value)

    def target(self, context: "Context") -> Optional[str]:
        '''这次改哪个文件（相对 `base_dir`）。'''
        if self.path is None:
            return None
        value = self._render(self.path, context)
        return None if value is None else str(value)

    def _error_text(self, context: "Context") -> str:
        if self.error is None:
            return ""
        try:
            value = self._render(self.error, context)
        except UndefinedVariable:
            # 第一次跑的时候还没有 test_result，这不是错误
            return ""
        if value is None:
            return ""
        if isinstance(value, dict):
            # 直接把 RunTests 的结果丢进来也行：优先取 output
            text = value.get("output") or value.get("stderr") or json.dumps(
                value, ensure_ascii=False, indent=2
            )
        elif isinstance(value, (list, tuple)):
            text = "\n".join(str(item) for item in value)
        else:
            text = str(value)
        text = text.strip()
        if len(text) > self.max_error_chars:
            text = text[: self.max_error_chars] + "\n…（错误信息已截断）"
        return text

    def _files(self, context: "Context") -> List[Tuple[str, str]]:
        '''把 `context=` 归一成 [(名字, 内容)]。'''
        value = self._render(self.context, context)
        if value is None:
            return []
        if isinstance(value, dict):
            pairs = [(str(name), item) for name, item in value.items()]
        else:
            pairs = [("（任务相关材料）", value)]

        files: List[Tuple[str, str]] = []
        for name, item in pairs:
            item = self._render(item, context)
            if isinstance(item, (dict, list)):
                text = json.dumps(item, ensure_ascii=False, indent=2)
            else:
                text = "" if item is None else str(item)
            if text.strip():
                files.append((name, text))
        return files

    # ---------------- 提示词 ----------------
    def build_prompt(self, context: "Context") -> str:
        '''拼出这次要发给模型的东西（单独暴露，方便测试 / 排查）。'''
        parts: List[str] = [
            "你是一名严谨的软件工程师。你输出的内容会被程序直接执行，"
            "不是给人读的解释 —— 不要寒暄、不要总结、不要「好的，以下是……」"
        ]
        parts.append(f"## 任务\n{self._task_text(context)}")

        files = self._files(context)
        if files:
            budget = self.max_context_chars
            chunks: List[str] = []
            for name, text in files:
                if budget <= 0:
                    chunks.append(f"### {name}\n（超出上下文预算，未提供）")
                    continue
                if len(text) > budget:
                    text = text[:budget] + "\n…（已截断）"
                budget -= len(text)
                chunks.append(
                    f"### {name}\n```{self.language}\n{text.rstrip()}\n```"
                )
            parts.append("## 相关文件\n" + "\n\n".join(chunks))

        error = self._error_text(context)
        if error:
            parts.append(
                "## 上一次尝试失败了\n```text\n"
                + error
                + "\n```\n先分析原因，再给出修正后的输出；不要重复同样的错误。"
            )

        cautions = resolve_cautions(self.cautions, None, use_defaults=False)
        text = join_cautions(cautions)
        if text:
            parts.append(f"## 约束\n{text}")

        parts.append(f"## 输出格式\n{self.output.instruction()}")
        return "\n\n".join(parts)

    # ---------------- 执行 ----------------
    def execute(self, context: "Context") -> None:
        agent = self._agent(context)
        target = self.target(context)
        sandbox = (
            Sandbox(self.base_dir, allow_write=False)
            if (target is not None and self.verify) or self.output.name == "json"
            else None
        )

        prompt = self.build_prompt(context)
        retries = settings.max_retries if self.max_retries is None else self.max_retries
        attempts = retries + 1
        last_error: Optional[Exception] = None
        bus = bus_of(context)

        for attempt in range(1, attempts + 1):
            raw = agent.ask({"role": "user", "content": prompt}, bus=bus)
            try:
                value = self.output.parse(
                    raw,
                    sandbox=sandbox,
                    path=target,
                    language=self.language,
                )
            except (CodeFormatError, DiffError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                prompt = (
                    self.build_prompt(context)
                    + "\n\n## 你上一次的输出不能用\n```text\n"
                    + str(exc)
                    + "\n```\n请严格按上面的输出格式重来一次，不要解释、不要加代码块围栏。"
                )
                continue

            context[self.output_key] = value
            context.trace.append(f"{self.name}:{attempt}")
            return

        raise CodeFormatError(
            f"CodingStep 连续 {attempts} 次都没有拿到可用的代码变更；"
            f"最后一次错误：{last_error}"
        )

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        from promptkit.core.display_sys.tree import short

        target = f" -> {self.path}" if self.path is not None else ""
        return (
            f"Coding({self.output.name}, {short(self.task)}{target}"
            f" -> {self.output_key})"
        )
