'''预置约束组：常用的一整套约束，拿名字就能用。

    from promptkit.cautions import JSON_STRICT, GROUNDED, tutor

    LLMStep(..., cautions=[JSON_STRICT])
    LLMStep(..., cautions=grounded())            # 配合知识库 / 搜索
    LLMStep(..., cautions=tutor("市场调研大赛导师"))

每个都是 `CautionGroup`，所以还能继续叠加：

    cautions=GROUNDED + Tone("耐心")
'''
from typing import Any

from promptkit.cautions.builtin import (
    AskFirst,
    Citations,
    Concise,
    Grounded,
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
from promptkit.core.model_action.caution_related.caution import CautionGroup

__all__ = [
    # 通用
    "json_only",
    "json_strict",
    "json_with_schema",
    "plain_text",
    "concise_text",
    "code_only",
    "polite_reply",
    # 场景
    "grounded",
    "tutor",
    "socratic",
    "analyst",
    "translator",
    # 常量
    "JSON_ONLY",
    "JSON_STRICT",
    "PLAIN_TEXT",
    "CODE_ONLY",
    "GROUNDED",
    "TUTOR",
    "SOCRATIC",
    "ANALYST",
]


# ============================ 通用 ============================
def json_only(lang: str = "中文") -> CautionGroup:
    '''只要 JSON，不要废话。'''
    return CautionGroup([OutputFormat("JSON"), NoExtraText(), Language(lang)])


def json_strict(lang: str = "中文") -> CautionGroup:
    '''严格 JSON：不要 Markdown 包裹，字段不确定就如实说。

    比 `json_only` 多两条：`NoMarkdown`（模型很爱包一层 ```json）
    和 `NoFabrication`（宁缺毋滥，缺字段好过编字段）。
    '''
    return CautionGroup(
        [
            OutputFormat("JSON"),
            NoMarkdown(),
            NoExtraText(),
            NoFabrication(),
            Language(lang),
        ]
    )


def json_with_schema(schema: Any, lang: str = "中文") -> CautionGroup:
    '''只要 JSON，并且把字段说明写清楚。'''
    return CautionGroup(
        [OutputFormat("JSON"), SchemaHint(schema), NoExtraText(), Language(lang)]
    )


def plain_text(max_chars: int = 200, lang: str = "中文") -> CautionGroup:
    '''纯文本回答，限制长度。'''
    return CautionGroup([NoExtraText(), MaxLength(max_chars), Language(lang)])


def concise_text(max_chars: int = 120, lang: str = "中文") -> CautionGroup:
    '''简短纯文本：不要 Markdown、不要铺垫、有长度上限。'''
    return CautionGroup(
        [NoMarkdown(), Concise(), MaxLength(max_chars), Language(lang)]
    )


def code_only(language: str = "Python") -> CautionGroup:
    '''只要代码，不要解释。'''
    return CautionGroup([OutputFormat("代码"), Language(language), NoExtraText()])


def polite_reply(max_chars: int = 200) -> CautionGroup:
    '''客服礼貌回复。'''
    return CautionGroup([*plain_text(max_chars), Tone("礼貌、耐心")])


# ============================ 场景 ============================
def grounded(lang: str = "中文") -> CautionGroup:
    '''**只根据给定资料回答**，标出处，不知道就说不知道。

    配合知识库（`KnowledgeBase`）或联网搜索（`WEB_SEARCH`）用 ——
    这三条正好治住这类场景最常见的毛病：编资料、不给出处、硬答。
    '''
    return CautionGroup(
        [Grounded(), Citations("[编号]"), NoFabrication(), Language(lang)]
    )


def tutor(
    persona: str = "一位经验丰富的导师",
    lang: str = "中文",
) -> CautionGroup:
    '''导师风格：专业、分步骤、不编。

        tutor("市场调研分析大赛的导师")
    '''
    return CautionGroup(
        [
            Role(persona),
            Tone("专业、准确、有耐心"),
            StepByStep(),
            NoFabrication(),
            Language(lang),
        ]
    )


def socratic(lang: str = "中文") -> CautionGroup:
    '''苏格拉底式：先反问，再一步步引导，不要直接给答案。'''
    return CautionGroup(
        [
            Role("引导者"),
            AskFirst(),
            StepByStep(),
            Tone("启发式、不直接给结论"),
            Language(lang),
        ]
    )


def analyst(lang: str = "中文") -> CautionGroup:
    '''数据分析师：给依据、量化、简短。'''
    return CautionGroup(
        [Role("数据分析师"), StepByStep(), Concise(), NoFabrication(), Language(lang)]
    )


def translator(lang: str = "英文") -> CautionGroup:
    '''翻译：只输出译文，不要解释、不要 Markdown。'''
    return CautionGroup(
        [OutputFormat("译文"), NoExtraText(), NoMarkdown(), Language(lang)]
    )


# ============================ 常量 ============================
#: 最常用的那个
JSON_ONLY = json_only()

#: 严格 JSON（推荐替代 JSON_ONLY）
JSON_STRICT = json_strict()

#: 纯文本（默认 200 字）
PLAIN_TEXT = plain_text()

#: 只要代码
CODE_ONLY = code_only()

#: 基于资料回答（配合知识库 / 搜索）
GROUNDED = grounded()

#: 导师风格
TUTOR = tutor()

#: 追问引导
SOCRATIC = socratic()

#: 数据分析
ANALYST = analyst()
