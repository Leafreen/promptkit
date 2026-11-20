'''工具（function calling）的统一入口。

最小的用法就是直接传 Python 函数，签名和 docstring 会自动变成 JSON Schema：

    def get_weather(city: str) -> str:
        """查询某个城市的天气。"""
        return f"{city}：晴，25℃"

    agent.ask("北京天气怎么样？", tools=[get_weather])

需要自定义名字 / 描述 / 参数时，用 Tool 显式构造：

    Tool("get_weather", "查询天气", {"type": "object", "properties": {...}}, get_weather)

自带工具在 `builtin/` 里，`from aipykit.core.tool import WEB_SEARCH` 就能用。
'''
from aipykit.core.tool.base import (
    Tool,
    as_tool,
    as_tools,
    schema_from_signature,
    tool,
)
from aipykit.core.tool.builtin import (
    ALARMS,
    DEFAULT_ALLOWED_EXTENSIONS,
    GET_TIME,
    MAX_CONTENT_CHARS,
    RUN_COMMAND,
    TIMER,
    WEB_SEARCH,
    WRITE_FILE,
    AlarmBoard,
    AlarmRunner,
    WebSearch,
    WriteFile,
    clock_text,
    drain_alarms,
    elapsed_seconds,
    now_text,
    running_timers,
    start_timer,
    stop_all_timers,
    stop_timer,
    timer,
)

#: 这几组工具依赖上层包（coding / skill / memory），只在真正取用时导入，
#: 保证 `import aipykit.core` 不会把它们一起拖进来。
_LAZY = {
    "ReadFileTool": ("aipykit.core.tool.coding", "ReadFileTool"),
    "WriteFileTool": ("aipykit.core.tool.coding", "WriteFileTool"),
    "ApplyDiffTool": ("aipykit.core.tool.coding", "ApplyDiffTool"),
    "ListDirTool": ("aipykit.core.tool.coding", "ListDirTool"),
    "SearchCodeTool": ("aipykit.core.tool.coding", "SearchCodeTool"),
    "RunCommandTool": ("aipykit.core.tool.coding", "RunCommandTool"),
    "RunTestsTool": ("aipykit.core.tool.coding", "RunTestsTool"),
    "default_coding_tools": ("aipykit.core.tool.coding", "default_coding_tools"),
    "SkillTool": ("aipykit.core.tool.skill", "SkillTool"),
    "result_text": ("aipykit.core.tool.skill", "result_text"),
    "RecallMemoryTool": ("aipykit.core.tool.memory", "RecallMemoryTool"),
    "WriteMemoryTool": ("aipykit.core.tool.memory", "WriteMemoryTool"),
    "MEMORY_REFERENCE": ("aipykit.core.tool.memory", "MEMORY_REFERENCE"),
    "WRITE_MEMORY": ("aipykit.core.tool.memory", "WRITE_MEMORY"),
    "SummonSubAgentTool": ("aipykit.core.tool.subagent", "SummonSubAgentTool"),
}


def __getattr__(name: str):
    '''按需导入依赖上层包的那几组工具。'''
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    # 定义
    "Tool",
    "tool",
    "as_tool",
    "as_tools",
    "schema_from_signature",
    # 自带工具
    "RUN_COMMAND",
    "WEB_SEARCH",
    "WebSearch",
    "WriteFile",
    "WRITE_FILE",
    "DEFAULT_ALLOWED_EXTENSIONS",
    "MAX_CONTENT_CHARS",
    # 时间 / 定时器
    "GET_TIME",
    "TIMER",
    "ALARMS",
    "AlarmBoard",
    "AlarmRunner",
    "clock_text",
    "now_text",
    "drain_alarms",
    "start_timer",
    "stop_timer",
    "stop_all_timers",
    "elapsed_seconds",
    "running_timers",
    "timer",
    # 编码 / Skill / 记忆工具（按需导入）
    "ReadFileTool",
    "WriteFileTool",
    "ApplyDiffTool",
    "ListDirTool",
    "SearchCodeTool",
    "RunCommandTool",
    "RunTestsTool",
    "default_coding_tools",
    "SkillTool",
    "result_text",
    "RecallMemoryTool",
    "WriteMemoryTool",
    "MEMORY_REFERENCE",
    "WRITE_MEMORY",
    # 子 Agent 召唤工具（按需导入，见 _LAZY）
    "SummonSubAgentTool",
]
