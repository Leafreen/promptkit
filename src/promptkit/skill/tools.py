# skill/tools.py
'''把 Skill 包成 `promptkit.tools.Tool`，这样 `Agent` 就能自己挑着调。'''
from __future__ import annotations

import json
from typing import Any

from promptkit.skill.errors import SkillError
from promptkit.skill.schema import validate_args
from promptkit.tools import Tool

__all__ = ["SkillTool", "result_text"]


class SkillTool(Tool):
    '''一个 Skill 的 Tool 视图：模型看到描述和参数，调用就执行 handler。

        tool = SkillTool(registry.get("classify_sentiment"), ctx=ctx)
        agent.ask("这句话是正面还是负面？", tools=[tool])
    '''

    def __init__(self, skill: Any, *, ctx: Any, registry: Any = None):
        self.skill = skill
        self.ctx = ctx
        self.registry = registry
        super().__init__(
            name=skill.name,
            description=skill.description,
            parameters=skill.parameters,
            func=self.run,
        )

    def run(self, **arguments: Any) -> str:
        '''工具不抛异常：错误变成字符串交给模型自己处理。'''
        try:
            if self.registry is not None:
                result = self.registry.execute(
                    self.skill.name, arguments, ctx=self.ctx
                )
            else:
                validate_args(self.skill.spec, arguments)
                result = self.skill.spec.invoke(self.ctx, **arguments)
        except SkillError as exc:
            return f"错误：{exc}"
        except Exception as exc:      # noqa: BLE001 - 工具的契约是「总能返回」
            return f"错误：{type(exc).__name__}: {exc}"
        return result_text(result)


def result_text(result: Any) -> str:
    '''工具结果统一成文本：Pydantic 转 JSON，别糊一层 `<Order object at ...>`。'''
    if isinstance(result, str):
        return result
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        try:
            return json.dumps(dump(), ensure_ascii=False)
        except Exception:      # noqa: BLE001
            return str(result)
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)
