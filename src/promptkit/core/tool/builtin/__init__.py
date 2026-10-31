'''自带工具：能直接挂到 `LLMStep(tools=[...])` 或 `Tools(...)` 上。

- `RUN_COMMAND`：跑一条 shell 命令
- `WEB_SEARCH`：联网搜索（免密钥，有 API Key 会自动升级后端）
- `WriteFile` / `WRITE_FILE`：让**模型自己决定**把成果写进文件

    from promptkit.core.tool import WriteFile
    from promptkit.core import Tools

    writer = WriteFile("outputs")       # 沙箱：只能写 outputs/ 下面
    Tools(writer, WEB_SEARCH)

⚠️ 三个都会把**外部内容**喂给模型（命令输出、网页文本、文件路径），
这些内容里可能有「忽略之前的指令」之类的注入文本，别当指令执行。
`WRITE_FILE` 更是直接**由模型驱动写磁盘**，所以默认带沙箱（根目录限制 +
扩展名白名单 + 大小上限）；别为了省事把 `sandbox=False` 打开。
'''
from promptkit.core.tool.builtin.run_command import RUN_COMMAND
from promptkit.core.tool.builtin.web_search import WEB_SEARCH, WebSearch
from promptkit.core.tool.builtin.write_file import (
    DEFAULT_ALLOWED_EXTENSIONS,
    MAX_CONTENT_CHARS,
    WRITE_FILE,
    WriteFile,
)

__all__ = [
    "RUN_COMMAND",
    "WEB_SEARCH",
    "WebSearch",
    "WriteFile",
    "WRITE_FILE",
    "DEFAULT_ALLOWED_EXTENSIONS",
    "MAX_CONTENT_CHARS",
]
