'''跟「模型输出」有关的运行时：全局默认约束 + 约束/格式化器。

`Context`（状态容器）和 `Workflow`（步骤容器）已经分别搬到
`promptkit.core.context` 和 `promptkit.core.steps.workflow` ——
它们是步骤层的基础件，不属于「模型输出控制」。

    from promptkit.core.model_action import Caution, JsonFormatter
'''
from promptkit.core.model_action.caution_related import (
    Caution,
    CautionGroup,
    Custom,
    Formatter,
    FormatterError,
    JsonFormatter,
    RegexFormatter,
    WORKFLOW_CAUTIONS_KEY,
    XmlFormatter,
    YamlFormatter,
    as_caution,
    flatten_cautions,
    get_formatter,
    join_cautions,
    resolve_cautions,
)
from promptkit.core.model_action.defaults import (
    add_default_caution,
    clear_default_cautions,
    get_default_cautions,
    set_default_cautions,
    using_default_cautions,
)

__all__ = [
    # defaults.py
    "set_default_cautions",
    "add_default_caution",
    "get_default_cautions",
    "clear_default_cautions",
    "using_default_cautions",
    # caution_related
    "Caution",
    "CautionGroup",
    "Custom",
    "WORKFLOW_CAUTIONS_KEY",
    "as_caution",
    "flatten_cautions",
    "join_cautions",
    "resolve_cautions",
    "Formatter",
    "FormatterError",
    "JsonFormatter",
    "XmlFormatter",
    "RegexFormatter",
    "YamlFormatter",
    "get_formatter",
]
