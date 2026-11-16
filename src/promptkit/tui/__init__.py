'''增强版终端对话界面（可选依赖）。

    pip install "aipykit[tui]"

不写 `EnableTUI()` 就完全不会用到。没装依赖时 `import promptkit.tui`
仍然成功，真正启动时才抛 `TuiUnavailableError`。
'''
from promptkit.core.steps.actions.tui import TuiUnavailableError
from promptkit.tui.app import TuiSession, start_tui

__all__ = ["start_tui", "TuiSession", "TuiUnavailableError"]
