# skill/__init__.py
'''Skill：把「提示词 + 控制流 + 工具 + 输出契约」封装成一个可注册、可描述、
可执行的单元。

**不引入新引擎** —— Skill 内部就是普通 Python（可以调 `Workflow` / `LLMStep` /
`Tool` / `Memory`），只是多了一层描述：

    from promptkit.skill import SkillContext, skill

    @skill(name="classify_sentiment", description="给一段文本做情感分类",
           tags=["nlp"], output_model=Sentiment)
    def classify_sentiment(text: str, ctx: SkillContext) -> Sentiment:
        return ctx.structured(f"判断情感：{text}", Sentiment)

`@skill` 会从**函数签名**生成 `parameters`（JSON Schema），所以：

    registry = SkillRegistry()
    registry.register(classify_sentiment)

    registry.to_openai_tools()      # 给模型看：名称 / 用途 / 参数
    registry.to_tools(ctx)          # 给 Agent 真能调：Tool 列表
    registry.execute("classify_sentiment", {"text": "很好"}, ctx=ctx)

Skill 可以调 Skill（`ctx.call_skill`）；第三方包用 entry points 装进来
（`registry.load_entry_points()`）。
'''
from promptkit.skill.context import SkillContext
from promptkit.skill.errors import (
    SkillAlreadyExists,
    SkillError,
    SkillExecutionError,
    SkillNotFound,
    SkillValidationError,
)
from promptkit.skill.registry import SkillRegistry
from promptkit.skill.schema import build_parameters, type_schema, validate_args
from promptkit.skill.spec import Skill, SkillSpec, build_spec, skill
from promptkit.skill.steps import SkillStep
from promptkit.skill.tools import SkillTool, result_text

__all__ = [
    # 定义
    "skill",
    "build_spec",
    "SkillSpec",
    "Skill",
    # 注册 / 执行
    "SkillRegistry",
    "SkillContext",
    "SkillStep",
    "SkillTool",
    "result_text",
    # schema
    "build_parameters",
    "type_schema",
    "validate_args",
    # 异常
    "SkillError",
    "SkillValidationError",
    "SkillAlreadyExists",
    "SkillNotFound",
    "SkillExecutionError",
]
