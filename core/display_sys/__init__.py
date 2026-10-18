'''显示与输出：把内部状态变成给人看的文本。

这一层是**叶子** —— 只依赖 `event_sys`（要认识事件类型），
不依赖 `steps` / `model_action`，所以谁都能用，也不会形成环。

| 模块 | 干什么 |
| --- | --- |
| `display.py` | 事件 -> 终端（`console_display` / `json_display`） |
| `render.py` | Markdown / 数学公式 -> 终端；`Verbosity` 分级 |
| `runlog.py` | 事件 -> 日志文件（`log_display` / `setup_logging`） |
| `tree.py` | 步骤树 -> 文本（`Node` / `format_tree`） |

分层约定：**结果走 stdout，过程走 stderr，日志走文件**。

    from promptkit.core.display_sys import Verbosity, log_display, render_markdown
'''
from promptkit.core.display_sys.display import (
    collect_events,
    console_display,
    filter_events,
    json_display,
)
from promptkit.core.display_sys.render import (
    HAVE_RICH,
    HAVE_SYMPY,
    RENDERERS,
    Verbosity,
    as_verbosity,
    get_console,
    install_hint,
    latex_to_unicode,
    looks_like_math,
    needs_art,
    plain_text,
    render_markdown,
    render_math,
    render_plain,
    strip_markdown,
)
from promptkit.core.display_sys.runlog import (
    DEFAULT_FORMAT,
    DEFAULT_LOG_PATH,
    LOGGER_NAME,
    close_logging,
    get_logger,
    log_display,
    setup_logging,
)
from promptkit.core.display_sys.tree import Node, format_tree, short

__all__ = [
    # display.py：事件 -> 终端
    "console_display",
    "json_display",
    "collect_events",
    "filter_events",
    # render.py：Markdown / 公式 / 分级
    "Verbosity",
    "as_verbosity",
    "render_markdown",
    "render_math",
    "render_plain",
    "plain_text",
    "strip_markdown",
    "latex_to_unicode",
    "looks_like_math",
    "needs_art",
    "get_console",
    "HAVE_RICH",
    "HAVE_SYMPY",
    "install_hint",
    "RENDERERS",
    # runlog.py：事件 -> 文件
    "setup_logging",
    "close_logging",
    "get_logger",
    "log_display",
    "LOGGER_NAME",
    "DEFAULT_LOG_PATH",
    "DEFAULT_FORMAT",
    # tree.py：步骤树 -> 文本
    "Node",
    "format_tree",
    "short",
]
