# memory/config.py
'''声明式记忆配置：给 `ChatWorkflow` / `MemoryStep` 用的两个小对象。

    memory_config = MemoryConfig(
        read=["level", "preferred_style"],
        recall=RecallConfig(query_key="usr_input", output_key="relevant_history", top_k=5),
        write=["topics_covered", "new_weak_points"],
        budget=TokenBudget(2000),
        write_on="end",
    )

它们只是**普通数据**，不碰存储，也不依赖 core（免得循环 import）。
真正干活的是被它们编译出来的 `ReadMemory` / `RecallMemory` / `WriteMemory`。
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from promptkit.memory.budget import TokenBudget

__all__ = ["RecallConfig", "MemoryConfig", "WRITE_ON_VALUES"]

#: `write_on` 的合法值：循环结束后写 / 每轮写
WRITE_ON_VALUES = ("end", "each_turn")


def _as_list(items: Union[str, Sequence[str], None]) -> List[str]:
    '''`read="level"` 和 `read=["level"]` 都当列表处理。'''
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    return [str(item) for item in items]


@dataclass
class RecallConfig:
    '''一次语义召回的参数。

    `query_key` 是 **Context 里的变量名**（默认 `usr_input`，也就是聊天循环
    刚读进来的那句话）。想用别的东西当查询，写那个变量名就行。
    '''

    #: 拿哪个变量的值当查询
    query_key: str = "usr_input"
    #: 召回结果写进哪个变量
    output_key: str = "recalled"
    #: 要几条
    top_k: int = 5
    #: 单独的预算；不给就用 `MemoryConfig.budget`
    budget: Optional[TokenBudget] = None
    #: 低于这个分数的干脆不要（0.0 ~ 1.0，具体量纲看你的 embedding）
    min_score: Optional[float] = None
    #: True = 输出结构化 `list[dict]`；False（默认）= 输出拼好的文本
    raw: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k 必须是正整数，得到 {self.top_k!r}")
        if not self.query_key:
            raise ValueError("query_key 不能为空")
        if not self.output_key:
            raise ValueError("output_key 不能为空")
        if self.min_score is not None and not isinstance(self.min_score, (int, float)):
            raise ValueError(f"min_score 必须是数字，得到 {self.min_score!r}")


@dataclass
class MemoryConfig:
    '''一个会话该读什么、召回什么、写什么。

    - `read`：开跑前从长期记忆读出来的**键名**，写进同名变量
    - `recall`：每轮按用户输入做一次语义召回（`RecallConfig` / dict / `RecallMemory`）
    - `write`：结束后（或每轮）把**上下文里同名变量**写回长期记忆
    - `budget`：`recall` 没单独给预算时用这个
    - `write_on`：`"end"` 循环结束后写一次 / `"each_turn"` 每轮写

    ⚠️ `write` 里的名字必须是 Context 里**已经存在**的变量。模型的回答默认只
    写进 `model_reply`，想把「这轮聊了什么」写进记忆，得先用
    `LLMStep(output_model=...)` 把它抽成结构化字段 —— 或者说，**想写什么
    由工作流显式决定**，不是由模型随手决定。
    '''

    read: List[str] = field(default_factory=list)
    recall: Optional[object] = None
    write: List[str] = field(default_factory=list)
    budget: Optional[TokenBudget] = None
    write_on: str = "end"

    def __post_init__(self) -> None:
        self.read = _as_list(self.read)
        self.write = _as_list(self.write)
        if self.write_on not in WRITE_ON_VALUES:
            raise ValueError(
                f"write_on 只能是 {WRITE_ON_VALUES}，得到 {self.write_on!r}"
            )
        if isinstance(self.recall, dict):
            self.recall = RecallConfig(**self.recall)
        if self.recall is not None and not self.has_recall():
            raise TypeError(
                "recall 需要 RecallConfig / dict / RecallMemory 之一，"
                f"得到 {type(self.recall).__name__}"
            )
        if self.budget is not None and not isinstance(self.budget, TokenBudget):
            raise TypeError(
                f"budget 需要 TokenBudget，得到 {type(self.budget).__name__}"
            )

    def has_recall(self) -> bool:
        '''`recall` 是不是一个能用的东西。

        这里用鸭子类型而不是 isinstance：`RecallMemory` 在 core 里，
        memory 包不 import 它（否则循环依赖）。
        '''
        return self.recall is not None and (
            isinstance(self.recall, RecallConfig)
            or callable(getattr(self.recall, "execute", None))
        )

    def recall_budget(self) -> Optional[TokenBudget]:
        '''实际生效的召回预算（`recall` 自己的优先）。'''
        if isinstance(self.recall, RecallConfig) and self.recall.budget is not None:
            return self.recall.budget
        return self.budget

    def is_empty(self) -> bool:
        '''什么都没配。'''
        return not self.read and not self.write and not self.has_recall()

    def __repr__(self) -> str:
        return (
            f"MemoryConfig(read={self.read!r}, recall={self.recall!r}, "
            f"write={self.write!r}, write_on={self.write_on!r}, budget={self.budget!r})"
        )
