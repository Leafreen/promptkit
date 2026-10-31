'''工具（function calling）的统一入口。

最小的用法就是直接传 Python 函数，签名和 docstring 会自动变成 JSON Schema：

    def get_weather(city: str) -> str:
        """查询某个城市的天气。"""
        return f"{city}：晴，25℃"

    agent.ask("北京天气怎么样？", tools=[get_weather])

需要自定义名字 / 描述 / 参数时，用 Tool 显式构造：

    Tool("get_weather", "查询天气", {"type": "object", "properties": {...}}, get_weather)

自带工具在 `builtin/` 里，`from promptkit.core.tool import WEB_SEARCH` 就能用。
'''
from promptkit.core.tool.base import (
    Tool,
    as_tool,
    as_tools,
    schema_from_signature,
    tool,
)
from promptkit.core.tool.builtin import (
    DEFAULT_ALLOWED_EXTENSIONS,
    MAX_CONTENT_CHARS,
    RUN_COMMAND,
    WEB_SEARCH,
    WRITE_FILE,
    WebSearch,
    WriteFile,
)

#: 这几组工具依赖上层包（coding / skill / memory），只在真正取用时导入，
#: 保证 `import promptkit.core` 不会把它们一起拖进来。
_LAZY = {
    "ReadFileTool": ("promptkit.core.tool.coding", "ReadFileTool"),
    "WriteFileTool": ("promptkit.core.tool.coding", "WriteFileTool"),
    "ApplyDiffTool": ("promptkit.core.tool.coding", "ApplyDiffTool"),
    "ListDirTool": ("promptkit.core.tool.coding", "ListDirTool"),
    "SearchCodeTool": ("promptkit.core.tool.coding", "SearchCodeTool"),
    "RunCommandTool": ("promptkit.core.tool.coding", "RunCommandTool"),
    "RunTestsTool": ("promptkit.core.tool.coding", "RunTestsTool"),
    "default_coding_tools": ("promptkit.core.tool.coding", "default_coding_tools"),
    "SkillTool": ("promptkit.core.tool.skill", "SkillTool"),
    "result_text": ("promptkit.core.tool.skill", "result_text"),
    "RecallMemoryTool": ("promptkit.core.tool.memory", "RecallMemoryTool"),
    "WriteMemoryTool": ("promptkit.core.tool.memory", "WriteMemoryTool"),
    "MEMORY_REFERENCE": ("promptkit.core.tool.memory", "MEMORY_REFERENCE"),
    "WRITE_MEMORY": ("promptkit.core.tool.memory", "WRITE_MEMORY"),
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
]
