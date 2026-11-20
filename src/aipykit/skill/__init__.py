# skill/__init__.py
'''Skill：把「提示词 + 控制流 + 工具 + 输出契约」封装成一个可注册、可描述、
可执行的单元。

**不引入新引擎** —— Skill 内部就是普通 Python（可以调 `Workflow` / `LLMStep` /
`Tool` / `Memory`），只是多了一层描述：

    from aipykit.skill import SkillContext, skill

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
from aipykit.skill.context import SkillContext
from aipykit.skill.errors import (
    DependencyInstallError,
    DuplicateToolError,
    InvalidSkillPackageError,
    PathEscapeError,
    PythonInterpreterError,
    ResourceNotFoundError,
    ScriptNotFoundError,
    SkillAlreadyExists,
    SkillDependencyError,
    SkillError,
    SkillExecutionError,
    SkillNotFound,
    SkillValidationError,
)
from aipykit.skill.markdown import MarkdownSkillTool, as_skill, as_skills
from aipykit.skill.orchestrate_skill import (
    OrchestrateSkillSet,
    orchestrate_plan,
    orchestrate_run,
    orchestrate_validate,
    register_orchestrate_skills,
)
from aipykit.skill.package import (
    ScriptSpec,
    SkillPackage,
    as_skill_package,
    load_skills_from_dir,
    merge_tools,
)
from aipykit.skill.python import ENV_VAR as PYTHON_ENV_VAR
from aipykit.skill.python import resolve_python
from aipykit.skill.registry import SkillRegistry
from aipykit.skill.schema import build_parameters, type_schema, validate_args
from aipykit.skill.spec import Skill, SkillSpec, build_spec, skill

# `SkillStep` 在 `core.steps.actions.skill`、工具视图在 `core.tool.skill`；
# 延迟导入避免 `core` 与 `skill` 在包初始化时互相套圈。
_LAZY = {
    "SkillStep": ("aipykit.core.steps.actions.skill", "SkillStep"),
    "SkillTool": ("aipykit.core.tool.skill", "SkillTool"),
    "result_text": ("aipykit.core.tool.skill", "result_text"),
}


def __getattr__(name: str):
    '''按需导入搬家到 core 的 Skill 步骤 / 工具。'''
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    # 定义
    "skill",
    "build_spec",
    "SkillSpec",
    "Skill",
    # 外部 Markdown 技能
    "as_skill",
    "as_skills",
    "MarkdownSkillTool",
    # 目录式技能包（SKILL.md + references/assets/templates + scripts）
    "as_skill_package",
    "SkillPackage",
    "ScriptSpec",
    "load_skills_from_dir",
    "merge_tools",
    # 解释器自动解析
    "resolve_python",
    "PYTHON_ENV_VAR",
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
    "SkillDependencyError",
    # 技能包（skill_spec v1）
    "InvalidSkillPackageError",
    "DependencyInstallError",
    "ScriptNotFoundError",
    "ResourceNotFoundError",
    "PathEscapeError",
    "DuplicateToolError",
    "PythonInterpreterError",
    # 编排 Skill（自包含）
    "orchestrate_run",
    "orchestrate_plan",
    "orchestrate_validate",
    "OrchestrateSkillSet",
    "register_orchestrate_skills",
]
