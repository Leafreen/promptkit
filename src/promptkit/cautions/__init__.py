'''约束（Caution）：追加到提示词末尾的指令对象。

**原子约束**（单个要求）：

    from promptkit.cautions import OutputFormat, NoMarkdown, Citations

    cautions=[OutputFormat("JSON"), NoMarkdown()]

**预置约束组**（一整套，拿来就用）：

    from promptkit.cautions import JSON_STRICT, GROUNDED, tutor, analyst

    cautions=[JSON_STRICT]
    cautions=grounded()                    # 基于资料回答 + 标出处 + 不编
    cautions=tutor("市场调研大赛的导师")
    cautions=GROUNDED + Tone("耐心")        # 组还能继续叠加
'''
from promptkit.cautions.builtin import (
    AskFirst,
    Citations,
    Concise,
    Custom,
    Grounded,
    JsonSchemaHint,
    Language,
    MaxLength,
    NoExtraText,
    NoFabrication,
    NoMarkdown,
    OutputFormat,
    Role,
    SchemaHint,
    StepByStep,
    Tone,
)
from promptkit.cautions.presets import (
    ANALYST,
    CODE_ONLY,
    GROUNDED,
    JSON_ONLY,
    JSON_STRICT,
    PLAIN_TEXT,
    SOCRATIC,
    TUTOR,
    analyst,
    code_only,
    concise_text,
    grounded,
    json_only,
    json_strict,
    json_with_schema,
    plain_text,
    polite_reply,
    socratic,
    translator,
    tutor,
)

__all__ = [
    # ---- 原子约束 ----
    # 格式
    "OutputFormat",
    "NoExtraText",
    "NoMarkdown",
    "Language",
    "MaxLength",
    # 内容
    "NoFabrication",
    "Citations",
    "Grounded",
    "Concise",
    "StepByStep",
    "AskFirst",
    # 语气 / 角色
    "Tone",
    "Role",
    # 结构
    "SchemaHint",
    "JsonSchemaHint",
    # 兜底
    "Custom",
    # ---- 预置组：通用 ----
    "json_only",
    "json_strict",
    "json_with_schema",
    "plain_text",
    "concise_text",
    "code_only",
    "polite_reply",
    # ---- 预置组：场景 ----
    "grounded",
    "tutor",
    "socratic",
    "analyst",
    "translator",
    # ---- 预置组：常量 ----
    "JSON_ONLY",
    "JSON_STRICT",
    "PLAIN_TEXT",
    "CODE_ONLY",
    "GROUNDED",
    "TUTOR",
    "SOCRATIC",
    "ANALYST",
]
