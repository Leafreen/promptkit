'''「动作」类步骤：会跟外部打交道的步骤（模型 / 用户 / 文件 / 消息）。

    from promptkit.core.steps.actions import LLMStep, Echo, ReadFile, Save
'''
from promptkit.core.steps.actions.interactive import (
    DEFAULT_ENCODING,
    OUTPUT_STREAM_KEY,
    Choose,
    ChooseExpr,
    Confirm,
    ConfirmExpr,
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
from promptkit.core.steps.actions.memory import (
    ForgetMemory,
    MemoryStep,
    ReadMemory,
    UseMemory,
    WriteMemory,
)
from promptkit.core.steps.actions.message import MessageStep
from promptkit.core.steps.actions.recall import RecallMemory, format_recalled
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
    DEFAULT_COMPACT_PROMPT,
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
    Compact,
    CompactStep,
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
    "Choose",
    "ChooseExpr",
    "Confirm",
    "ConfirmExpr",
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
    # memory.py：长期记忆的 IO 原语
    "ReadMemory",
    "WriteMemory",
    "ForgetMemory",
    "MemoryStep",
    "UseMemory",
    # recall.py：语义召回
    "RecallMemory",
    "format_recalled",
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
    "DEFAULT_COMPACT_PROMPT",
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
    "Compact",
    "CompactStep",
    "MakeFile",
    "SaveDeliverables",
    "build_system_prompt",
    "save_format",
    "chat_context_vars",
    # orchestrate.py / coding.py / skill.py：依赖上层包，按需导入
    "PlanStep",
    "PlanWithTool",
    "ExecutePlan",
    "AggregateStep",
    "format_task_results",
    "CodingStep",
    "SkillStep",
]

#: 这几类 Step 依赖上层包（orchestrate / skill / coding），只在真正取用时导入，
#: 保证 `import promptkit.core` 不会把它们一起拖进来。
_LAZY = {
    "PlanStep": ("promptkit.core.steps.actions.orchestrate", "PlanStep"),
    "PlanWithTool": ("promptkit.core.steps.actions.orchestrate", "PlanWithTool"),
    "ExecutePlan": ("promptkit.core.steps.actions.orchestrate", "ExecutePlan"),
    "AggregateStep": ("promptkit.core.steps.actions.orchestrate", "AggregateStep"),
    "format_task_results": (
        "promptkit.core.steps.actions.orchestrate",
        "format_task_results",
    ),
    "CodingStep": ("promptkit.core.steps.actions.coding", "CodingStep"),
    "SkillStep": ("promptkit.core.steps.actions.skill", "SkillStep"),
}


def __getattr__(name: str):
    '''按需导入依赖上层包的那几类 Step。'''
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
