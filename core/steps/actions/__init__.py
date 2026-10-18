'''「动作」类步骤：会跟外部打交道的步骤（模型 / 用户 / 文件 / 消息）。

    from promptkit.core.steps.actions import LLMStep, Echo, ReadFile, Save
'''
from promptkit.core.steps.actions.interactive import (
    DEFAULT_ENCODING,
    OUTPUT_STREAM_KEY,
    Echo,
    EchoStep,
    Read,
    ReadExpr,
    ReadFile,
    ReadFileExpr,
    Save,
    SaveHistory,
    SaveJSON,
    SaveHistoryStep,
    SaveStep,
    output_stream,
)
from promptkit.core.steps.actions.llm import LLMStep, T
from promptkit.core.steps.actions.message import MessageStep
from promptkit.core.steps.actions.chat_util import (
    CHAT_ROLE_KEY,
    CHAT_WANT_KEY,
    CHAT_RULES_KEY,
    CHAT_DELIVERABLES_KEY,
    CHAT_TOOLS_KEY,
    CHAT_EXIT_KEY,
    CHAT_SIGNAL_KEY,
    CHAT_INPUT_KEY,
    CHAT_REPLY_KEY,
    DEFAULT_EXIT_WORDS,
    Role,
    Want,
    Rules,
    Tools,
    Deliverables,
    ExitWords,
    Behavior,
    SystemPrompt,
    ChatLoop,
    ChatTurn,
    MakeFile,
    SaveDeliverables,
    build_system_prompt,
    save_format,
    chat_context_vars,
)

__all__ = [
    # llm.py
    "LLMStep",
    "T",
    # interactive.py：读
    "Read",
    "ReadExpr",
    "ReadFile",
    "ReadFileExpr",
    # interactive.py：写
    "Save",
    "SaveStep",
    "SaveJSON",
    "SaveHistory",
    "SaveHistoryStep",
    # interactive.py：打印
    "Echo",
    "EchoStep",
    "OUTPUT_STREAM_KEY",
    "DEFAULT_ENCODING",
    "output_stream",
    # message.py
    "MessageStep",    "CHAT_ROLE_KEY",
    "CHAT_WANT_KEY",
    "CHAT_RULES_KEY",
    "CHAT_DELIVERABLES_KEY",
    "CHAT_TOOLS_KEY",
    "CHAT_EXIT_KEY",
    "CHAT_SIGNAL_KEY",
    "CHAT_INPUT_KEY",
    "CHAT_REPLY_KEY",
    "DEFAULT_EXIT_WORDS",
    "Role",
    "Want",
    "Rules",
    "Tools",
    "Deliverables",
    "ExitWords",
    "Behavior",
    "SystemPrompt",
    "ChatLoop",
    "ChatTurn",
    "MakeFile",
    "SaveDeliverables",
    "build_system_prompt",
    "save_format",
    "chat_context_vars",

]
