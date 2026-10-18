'''内置约束。每个都是个小对象，`to_text()` 给出要追加的那句话。'''
import json
from typing import Any, Dict

from promptkit.core.model_action.caution_related.caution import Caution, Custom

# 这些键对模型理解字段没帮助，只是白烧 token（递归清理）
_NOISY_SCHEMA_KEYS = ("title", "additionalProperties", "default", "examples", "$schema")


def _strip_noise(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_noise(value)
            for key, value in node.items()
            if key not in _NOISY_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_strip_noise(item) for item in node]
    return node

__all__ = [
    "OutputFormat",
    "NoExtraText",
    "NoMarkdown",
    "NoFabrication",
    "Citations",
    "Grounded",
    "Concise",
    "StepByStep",
    "AskFirst",
    "Language",
    "MaxLength",
    "Tone",
    "SchemaHint",
    "JsonSchemaHint",
    "Role",
    "Custom",
]


class OutputFormat(Caution):
    '''强制输出格式。

        OutputFormat("JSON")                 -> 请以 JSON 格式输出。
        OutputFormat("JSON", extra="字段名用英文")  -> 请以 JSON 格式输出，字段名用英文。
    '''

    def __init__(self, format: str = "JSON", extra: str = ""):
        self.format = format
        self.extra = extra

    def to_text(self) -> str:
        text = f"请以 {self.format} 格式输出"
        if self.extra:
            text += f"，{self.extra}"
        return text + "。"


class NoExtraText(Caution):
    '''不要生成解释、前后缀、Markdown 包装。'''

    def to_text(self) -> str:
        return "除非另有指令，不要生成任何解释、前后缀或其他内容。"


class NoMarkdown(Caution):
    '''不要用 Markdown 标记，输出纯文本。'''

    def to_text(self) -> str:
        return "不要使用 Markdown 标记（**、#、-、表格等），直接输出纯文本。"


class NoFabrication(Caution):
    '''不知道就说不知道，不要编。

    配合搜索 / 知识库用尤其重要 —— 资料里没有的东西，模型很容易"补"一个出来。
    '''

    def to_text(self) -> str:
        return "如果信息不足或不确定，直接说不知道，不要编造或猜测。"


class Citations(Caution):
    '''引用资料时必须标出处。

        Citations()            -> 引用资料时用 [编号] 标出出处。
        Citations("[来源: 标题]")
    '''

    def __init__(self, style: str = "[编号]"):
        self.style = style

    def to_text(self) -> str:
        return f"引用资料时用 {self.style} 标出出处，不要给出来源不明的结论。"


class Grounded(Caution):
    '''只根据给定资料回答。'''

    def to_text(self) -> str:
        return "只根据上面提供的资料回答；资料里没有的内容不要补充。"


class Concise(Caution):
    '''简短直接，不要铺垫。'''

    def to_text(self) -> str:
        return "回答尽量简短直接，不要铺垫、复述问题或重复自己说过的话。"


class StepByStep(Caution):
    '''分步骤说明，每步给依据。'''

    def to_text(self) -> str:
        return "请分步骤说明，每一步都给出理由或依据。"


class AskFirst(Caution):
    '''信息不足时先反问，不要硬答。'''

    def __init__(self, at_most: int = 3):
        self.at_most = at_most

    def to_text(self) -> str:
        return (
            f"如果缺少必要信息，先问我（一次最多 {self.at_most} 个问题），"
            "不要自己假设。"
        )


class Language(Caution):
    '''指定回答语言。'''

    def __init__(self, lang: str = "中文"):
        self.lang = lang

    def to_text(self) -> str:
        return f"请使用{self.lang}回答。"


class MaxLength(Caution):
    '''限制回答长度。'''

    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars

    def to_text(self) -> str:
        return f"回答不超过 {self.max_chars} 字。"


class Tone(Caution):
    '''指定语气。'''

    def __init__(self, tone: str = "专业"):
        self.tone = tone

    def to_text(self) -> str:
        return f"语气要求：{self.tone}。"


class Role(Caution):
    '''指定角色 / 人设。'''

    def __init__(self, role: str):
        self.role = role

    def to_text(self) -> str:
        return f"你现在扮演：{self.role}。"


class SchemaHint(Caution):
    '''把字段说明写进提示词。

        SchemaHint({"label": "退款/咨询/投诉"})
        -> 输出字段说明：{"label": "退款/咨询/投诉"}
    '''

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def to_text(self) -> str:
        return f"输出字段说明：{json.dumps(self.schema, ensure_ascii=False)}"


class JsonSchemaHint(Caution):
    '''直接把 Pydantic 模型的 JSON Schema 贴进去。

        JsonSchemaHint(Intent)                      # 从模型类生成
        JsonSchemaHint(Intent.model_json_schema())  # 或者直接给 schema
    '''

    def __init__(self, model_or_schema: Any):
        schema = model_or_schema
        generator = getattr(model_or_schema, "model_json_schema", None)
        if callable(generator):
            schema = generator()
        self.schema = schema

    def to_text(self) -> str:
        return f"输出字段说明：{json.dumps(_strip_noise(self.schema), ensure_ascii=False)}"
