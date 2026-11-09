# tests/baseline/fake_agent.py
'''基线测试用的假 Agent：不联网、可脚本化回复、记录每次提问。'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeAgent:
    '''最小的 Agent 替身。

    只实现工作流真正会用到的那几个成员：`vendor` / `model` / `history` /
    `start_prompts` / `ask()` / `reset()`。`ask()` 按脚本顺序返回，
    脚本用完后重复最后一条，避免测试因轮数变化而崩。
    '''

    replies: list[str] = field(default_factory=lambda: ["（假模型回复）"])
    vendor: str = "fake"
    model: str = "scripted"
    history: list[dict[str, Any]] = field(default_factory=list)
    start_prompts: list[dict[str, str]] = field(
        default_factory=lambda: [{"role": "system", "content": "假 Agent"}]
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    def ask(
        self,
        prompt: object,
        tools: object = None,
        bus: object = None,
        **kwargs: object,
    ) -> str:
        '''记录一次提问并返回脚本化的回复。'''
        index = len(self.calls)
        self.calls.append(
            {"prompt": prompt, "tools": tools, "bus": bus, "kwargs": dict(kwargs)}
        )
        self.history.append({"role": "user", "content": str(prompt)})
        reply = self.replies[min(index, len(self.replies) - 1)]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        '''清空历史（SystemPrompt(reset=True) 会调用）。'''
        self.history = []

    def rebuild(self) -> None:
        '''SystemPrompt(reset=False) 需要它；这里只要不清历史就够。'''
