# memory/session.py
'''短期记忆：把当前对话（`Agent.history`）当成一块可读的记忆。

**不引入任何新存储** —— 它只是给 `agent.history` 套一层 `read()`，
所以工作流里可以像读记忆那样读上下文：

    wf = Workflow(
        ReadMemory(SessionMemory(), "history", output_key="so_far"),
        LLMStep(Prompt(role="user", content="接着聊：{{so_far}}")),
    )

它**只读**：写会话历史的正主是 `Agent`（`ask()` 会追加）和
`Agent.load_history()` / `Agent.reset()`。记忆系统不参与对话流转，
这样「谁改历史」始终只有一个答案。

没有 `recall()`：对话历史的语义召回得靠 `VectorMemory`（把历史 chunk 化
再存进去）。这里保持薄，不做一个「大致能用」的假召回。
'''
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from promptkit.memory.base import Memory, MemoryError

if TYPE_CHECKING:  # 只用于类型标注：memory 包不在 import 时依赖 Agent
    from promptkit.agent import Agent

__all__ = ["SessionMemory"]

_ROLE_LABELS = {
    "user": "用户",
    "assistant": "助手",
    "system": "系统",
    "tool": "工具",
    "function": "工具",
}


class SessionMemory(Memory):
    '''当前会话的只读视图。

    支持这些键（其它键返回 `default`）：

    | key | 值 |
    | --- | --- |
    | `history` | 对话历史 `[{role, content, ...}, ...]`（不含 system） |
    | `messages` | 真正发给模型的消息（含 system） |
    | `turns` | 大致几轮（`len(history) // 2`） |
    | `last` | 最后一条消息（dict） |
    | `last_user` / `last_assistant` | 最后一条该角色消息的正文 |
    | `system` | system prompt 文本 |
    | `text` | 整个历史拼成的多行文本（给人看 / 喂模型） |

    `agent` 省略时用 `Agent.default()`（`Agent()` 会自动注册），
    所以 `SessionMemory()` 放在模块级声明里也能用 —— 真正取值发生在运行时。
    '''

    namespace = "session"

    def __init__(self, agent: Optional["Agent"] = None, *, include_system: bool = False):
        self.agent = agent
        #: `history` / `text` 要不要把 system prompt 也算进去
        self.include_system = include_system

    # ---------------- 解析目标 ----------------
    def target(self) -> Any:
        '''真正要读的 agent：显式传的，否则默认 agent。'''
        if self.agent is not None:
            return self.agent
        from promptkit.agent import Agent

        agent = Agent.default()
        if agent is None:
            raise MemoryError(
                "SessionMemory 还没有绑定 agent：先执行 `agent = Agent()`"
                "（会自动注册为默认），或者写成 `SessionMemory(agent=agent)`"
            )
        return agent

    # ---------------- 读 ----------------
    def read(self, key: str = "history", default: Any = None) -> Any:
        agent = self.target()
        name = str(key)

        if name == "messages":
            return [dict(message) for message in getattr(agent, "messages", [])]
        if name == "system":
            return self._system_prompt(agent) or default
        if name == "turns":
            return len(getattr(agent, "history", []) or []) // 2
        if name == "text":
            return self._as_text(agent)

        history = self._messages(agent)
        if name == "history":
            return history
        if name == "last":
            return history[-1] if history else default
        if name in ("last_user", "last_assistant"):
            role = "user" if name.endswith("user") else "assistant"
            for message in reversed(history):
                if message.get("role") == role:
                    return message.get("content") or ""
            return default
        return default

    # ---------------- 只读 ----------------
    def write(self, key: str, value: Any) -> None:
        raise MemoryError(
            "SessionMemory 是只读的：改对话历史请用 agent.ask() / "
            "agent.load_history() / agent.reset()，"
            "要写长期记忆请用 KVMemory 或 VectorMemory"
        )

    # ---------------- 内部 ----------------
    def _messages(self, agent: Any) -> List[Dict[str, Any]]:
        history = [dict(message) for message in getattr(agent, "history", [])]
        if not self.include_system:
            return history
        system = self._system_prompt(agent)
        if not system:
            return history
        return [{"role": "system", "content": system}, *history]

    @staticmethod
    def _system_prompt(agent: Any) -> str:
        for message in getattr(agent, "start_prompts", []) or []:
            if isinstance(message, dict) and message.get("role") == "system":
                return str(message.get("content") or "")
        return ""

    def _as_text(self, agent: Any) -> str:
        lines: List[str] = []
        for message in self._messages(agent):
            role = str(message.get("role", "?"))
            content = message.get("content")
            label = _ROLE_LABELS.get(role, role)
            lines.append(f"{label}：{'' if content is None else content}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        agent = self.agent
        return f"<SessionMemory agent={'默认' if agent is None else type(agent).__name__}>"
