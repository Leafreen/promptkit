# core/tool/skill.py
'''把 Skill 包成 `Tool`，这样 `Agent` 就能自己挑着调。'''
from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from promptkit.core.tool.base import Tool, ToolSchema

if TYPE_CHECKING:  # 避免 core.tool -> skill -> core.tool 的运行时套圈
    from promptkit.skill.spec import Skill

__all__ = ["SkillTool", "result_text"]


class SkillTool(Tool):
    '''一个 Skill 的 Tool 视图：模型看到描述和参数，调用就执行 handler。

        tool = SkillTool(registry.get("classify_sentiment"), ctx=ctx)
        agent.ask("这句话是正面还是负面？", tools=[tool])
    '''

    def __init__(self, skill: "Skill", *, ctx: object, registry: object = None):
        self.skill = skill
        self.ctx = ctx
        self.registry = registry
        super().__init__(
            name=skill.name,
            description=skill.description,
            parameters=cast(ToolSchema, skill.parameters),
            func=self.run,
        )

    def run(self, **arguments) -> str:
        '''工具不抛异常：错误变成字符串交给模型自己处理。'''
        # 延迟导入：`skill` 包本身会反过来引用本模块，模块级导入会套圈。
        from promptkit.skill.errors import SkillError
        from promptkit.skill.schema import validate_args

        try:
            if self.registry is not None:
                result = getattr(self.registry, "execute")(
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



def result_text(result: object) -> str:
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
