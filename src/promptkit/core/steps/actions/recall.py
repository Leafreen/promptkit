# core/steps/actions/recall.py
'''检索原语：`RecallMemory` —— 语义召回 + 预算裁剪 + 拼成可读文本。

为什么召回单独一个原语、不塞进 `ReadMemory`
-------------------------------------------
按键读是**确定**的（没有就是没有），语义召回是**模糊**的（总有一堆「差不多」的
结果，只是分数高低）。两者的参数、失败模式、成本都不一样，塞进一个步骤会
变成一堆 `if`。所以：

    ReadMemory(kv, "level")                      # IO 层
    RecallMemory(vec, query=Var("usr_input"))    # 检索层

召回结果默认**拼成文本**（`[1] …` 这种），因为它的主要去处是 prompt；
要结构化数据就 `raw=True`，拿到的是 `[{text, score, metadata}, ...]`。
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context
    from promptkit.memory.budget import TokenBudget

__all__ = ["RecallMemory", "format_recalled"]


def format_recalled(items: Sequence[object], *, max_chars: int = 0) -> str:
    '''把召回结果拼成文本（`RecallMemory` 的默认输出格式）。

        [1] 退款政策：7 天无理由……
        [2] 退款需要订单号……

    实现放在 `memory.base.format_memories` 里（`VectorMemory` 之类的调用方
    也要用），这里**延迟导入**：`core` 不该在 import 时就把整个 memory 包
    拖进来 —— 步骤层只认「有 read/write/recall 的对象」，不认 memory 包。
    '''
    from promptkit.memory.base import format_memories

    return format_memories(items, max_chars=max_chars)


def _score_of(item: object) -> float:
    '''一条结果的分数（dict / MemoryItem / 别的都兜住）。'''
    raw = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class RecallMemory(Step):
    '''从长期记忆里召回相关内容，写进 Context。

        RecallMemory(vec, query=Var("usr_input"), output_key="related", top_k=5)
        RecallMemory(vec, query_key="usr_input", budget=TokenBudget(1500))
        RecallMemory(vec, query="退款怎么处理", raw=True)   # 查询写死 + 要结构化

    - `query=`：表达式（`Var(...)`）或字面量字符串
    - `query_key=`：Context 里的变量名（和 `query` 二选一）
    - `output_key`：默认 `"recalled"`
    - `top_k`：要几条（重排器可能再截一次）
    - `budget`：召回后按 token 裁剪，**强烈建议给**，否则一条超长笔记就能
      把上下文吃掉
    - `min_score`：低于这个分数的直接丢
    - `raw=True`：输出 `list[dict]` 而不是文本
    '''

    memory: object
    #: 查询（表达式或字面量）
    query: object = None
    #: 从 Context 的哪个变量取查询
    query_key: Optional[str] = None
    output_key: str = "recalled"
    top_k: int = 5
    budget: Optional["TokenBudget"] = None
    min_score: Optional[float] = None
    #: True = 输出 list[dict]；False = 输出拼好的文本
    raw: bool = False
    #: 文本模式下每条最多多少字（0 = 不截）
    max_chars: int = 0

    name: str = "recall_memory"

    def __post_init__(self) -> None:
        if not callable(getattr(self.memory, "recall", None)):
            raise TypeError(
                "RecallMemory 需要实现了 recall() 的记忆对象，"
                f"得到 {type(self.memory).__name__}"
            )
        if self._is_pure_kv():
            raise TypeError(
                f"{type(self.memory).__name__} 是纯 KV 记忆，没有语义召回（recall）。"
                "要召回请用 VectorMemory / LayeredMemory，"
                "或者这个字段改用 ReadMemory 按键读"
            )
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k 必须是正整数，得到 {self.top_k!r}")
        if self.query is None and self.query_key is None:
            raise ValueError(
                "RecallMemory 需要 query=（表达式 / 字面量）或 query_key=（变量名）"
            )

    def _is_pure_kv(self) -> bool:
        '''这个记忆是不是「有 recall 这个名字、但其实没实现」的纯 KV 记忆。

        构造时就查，比跑到对话中间才抛 `NotImplementedError` 好得多。
        `KVMemory` / `InMemoryMemory` 继承的是 `Memory.recall`（那个只会抛错）。
        '''
        try:
            from promptkit.memory.base import Memory
        except ImportError:  # pragma: no cover - 没有 memory 包时无从比较
            return False
        method = getattr(type(self.memory), "recall", None)
        return method is Memory.recall

    # ---------------- 查询文本 ----------------
    def query_text(self, context: "Context") -> str:
        '''这次到底拿什么去召回。'''
        if self.query_key is not None:
            value = context.get(self.query_key)
        else:
            value = self.query.eval(context) if isinstance(self.query, Expr) else self.query
        return "" if value is None else str(value).strip()

    # ---------------- 执行 ----------------
    def execute(self, context: "Context") -> None:
        text = self.query_text(context)
        if not text:
            # 空查询不召回：既能省一次 embedding，也避免「用空串召回一堆噪音」
            context[self.output_key] = [] if self.raw else ""
            context.trace.append(f"{self.name}:skip")
            return

        items = getattr(self.memory, "recall")(text, top_k=self.top_k)
        if self.min_score is not None:
            items = [item for item in items if _score_of(item) >= self.min_score]
        if self.budget is not None:
            items = self.budget.trim(items)

        context[self.output_key] = (
            items
            if self.raw
            else format_recalled(items, max_chars=self.max_chars)
        )
        context.trace.append(f"{self.name}:{len(items)}")

    def describe(self) -> str:
        query = (
            f"key={self.query_key!r}"
            if self.query_key is not None
            else repr(self.query)
        )
        budget = f", {self.budget!r}" if self.budget is not None else ""
        return (
            f"RecallMemory({type(self.memory).__name__}, {query}, "
            f"top_k={self.top_k}{budget} -> {self.output_key})"
        )
