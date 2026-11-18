# conversation/config.py
'''回退 / 分支行为的配置。

`RewindConfig` 是冻结数据类，可以安全地作为默认值共享；需要不同策略时
显式构造一份传给 `Restore` / `Fork` / `Branch`，或预先放进
`context["_rewind_config"]`。
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OnBeyondLimit = Literal["error", "clamp"]

_ON_BEYOND_LIMIT_VALUES = frozenset({"error", "clamp"})


@dataclass(frozen=True)
class RewindConfig:
    '''回退与分支的约束。

    - `allow_rewind_into_summary`：是否允许回退到压缩（摘要）之前的消息。
      `False` 时命中压缩区间会抛 `RewindBeyondCompression`。
    - `max_rewind_turns`：一次最多回退多少轮（一轮 = 一条 user 消息）。
    - `on_beyond_limit`：超出轮次上限时的策略。
      `"error"` 抛 `RewindBeyondLimit`；`"clamp"` 只回退到允许的最早位置。
    - `max_branches`：同一工作流里最多允许创建多少个分支。
    - `max_concurrency`：仅保留接口，当前实现永远串行执行分支。
    '''

    allow_rewind_into_summary: bool = False
    max_rewind_turns: int = 3
    on_beyond_limit: OnBeyondLimit = "error"
    max_branches: int = 4
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.max_rewind_turns < 0:
            raise ValueError("max_rewind_turns 必须 >= 0")
        if self.on_beyond_limit not in _ON_BEYOND_LIMIT_VALUES:
            raise ValueError(
                "on_beyond_limit 只能是 'error' 或 'clamp'，"
                f"得到 {self.on_beyond_limit!r}"
            )
        if self.max_branches < 1:
            raise ValueError("max_branches 必须 >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency 必须 >= 1（当前实现始终串行）")


# 本文件核心职责：定义 RewindConfig（压缩边界 / 回退轮次 / 超限策略 / 分支上限）。
