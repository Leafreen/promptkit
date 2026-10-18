# core/steps/actions/message.py
'''把一条消息追加进 context["messages"]。'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from promptkit.core.steps.interface import Step

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context


@dataclass
class MessageStep(Step):
    '''把一条消息追加到 context["messages"] 里。'''

    role: str
    content: str
    name: str = "message"

    def execute(self, context: Context) -> None:
        messages = context.setdefault("messages", [])
        messages.append({"role": self.role, "content": self.content})
        context.trace.append(self.name)

    def describe(self) -> str:
        from promptkit.core.display_sys.tree import short

        return f"Message({self.role!r}, {short(self.content)})"
