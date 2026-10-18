'''约束（Caution）与格式化器（Formatter）—— 大模型调用前后的两层。

- `caution`：**事前**把要求追加进提示词，降低格式出错率
- `formatter`：**事后**把原始文本解析成结构化数据，交给 Pydantic 校验

    from promptkit.core.model_action.caution_related import Caution, JsonFormatter
'''
from promptkit.core.model_action.caution_related.caution import (
    Caution,
    CautionGroup,
    Custom,
    WORKFLOW_CAUTIONS_KEY,
    as_caution,
    flatten_cautions,
    join_cautions,
    resolve_cautions,
)
from promptkit.core.model_action.caution_related.formatter import (
    Formatter,
    FormatterError,
    JsonFormatter,
    RegexFormatter,
    XmlFormatter,
    YamlFormatter,
    get_formatter,
)

__all__ = [
    # caution.py
    "Caution",
    "CautionGroup",
    "Custom",
    "WORKFLOW_CAUTIONS_KEY",
    "as_caution",
    "flatten_cautions",
    "join_cautions",
    "resolve_cautions",
    # formatter.py
    "Formatter",
    "FormatterError",
    "JsonFormatter",
    "XmlFormatter",
    "RegexFormatter",
    "YamlFormatter",
    "get_formatter",
]
