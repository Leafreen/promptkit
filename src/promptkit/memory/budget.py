# memory/budget.py
'''召回内容的 token 预算。

召回 20 条笔记很容易把上下文吃光。所以**召回之后必须过一道预算**：
按 token 算，超了就丢（或截断）最不相关的那几条。

没有 tokenizer 也能用：内置一个粗糙但方向正确的估算 ——
ASCII 约 4 字符 1 token，非 ASCII（中文 / emoji）约 1 字符 1 token。
真要精确就把 `tokenizer=` 传进来（任何 `text -> int` 的可调用对象）。
'''
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, TypeVar

from promptkit.memory.base import item_text, replace_item_text

T = TypeVar("T")

__all__ = ["estimate_tokens", "TokenBudget"]


def estimate_tokens(text: str) -> int:
    '''不用 tokenizer 的粗略估算（宁可高估，别低估）。

    - 非 ASCII：1 字符 ≈ 1 token（中文大致就是这样，略偏保守）
    - ASCII：4 字符 ≈ 1 token（英文单词平均 4~5 字符）

    空字符串算 0；非空至少算 1（不然一堆空字符串能无限塞）。
    '''
    text = "" if text is None else str(text)
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4) + other_chars)


class TokenBudget:
    '''按 token 裁剪一串召回结果。

        budget = TokenBudget(max_tokens=2000)
        items = budget.trim(items)      # 输入输出都是 list[dict] / list[MemoryItem]

    规则很简单、也刻意好懂：

    1. 按**给定顺序**累加，放得下就留（顺序就是相关度，重排之后的顺序）
    2. 放不下的到此为止
    3. 但第一条就超预算时**截断第一条**而不是返回空 ——
       预算特别小时，一条被截断的答案也比什么都没有强

    `tokenizer` 可以是 `tiktoken` 的编码器（`lambda t: len(enc.encode(t))`）、
    模型的 tokenizer，或者任何 `text -> int`。
    '''

    def __init__(
        self,
        max_tokens: int = 2000,
        tokenizer: Optional[Callable[[str], int]] = None,
    ):
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens 必须是正整数，得到 {max_tokens!r}")
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    # ---------------- 计数 ----------------
    def count(self, text: object) -> int:
        '''一段文本要花多少 token。'''
        content = "" if text is None else str(text)
        if self.tokenizer is not None:
            try:
                return max(0, int(self.tokenizer(content)))
            except Exception as exc:  # noqa: BLE001 - 自定义 tokenizer 出错不该炸掉工作流
                raise TypeError(
                    f"tokenizer {self.tokenizer!r} 对文本求值失败：{exc}"
                ) from None
        return estimate_tokens(content)

    def count_items(self, items: Sequence[object]) -> int:
        '''一串召回结果总共要花多少 token（调试 / 测试用）。'''
        return sum(self.count(item_text(item)) for item in items or ())

    # ---------------- 裁剪 ----------------
    def trim(
        self, items: Sequence[T], *, max_tokens: Optional[int] = None
    ) -> List[T]:
        '''裁到预算之内，返回**同类型**的列表。'''
        limit = self.max_tokens if max_tokens is None else int(max_tokens)
        kept: List[T] = []
        used = 0

        for index, item in enumerate(items or ()):
            text = item_text(item)
            cost = self.count(text)
            if used + cost <= limit:
                kept.append(item)
                used += cost
                continue
            if index == 0:
                # 第一条就放不下：截断它，别让调用方拿到空结果
                kept.append(self._shorten(item, text, limit - used))
            break

        return kept

    # ---------------- 内部 ----------------
    def _shorten(self, item: T, text: str, room: int) -> T:
        '''把正文截到 `room` 个 token 以内（二分猜字符数，对自定义 tokenizer 也适用）。'''
        if not text:
            return replace_item_text(item, "")
        if room <= 1:
            # 连一个字都放不下：给个省略号，至少让模型知道「这里本来有东西」
            return replace_item_text(item, "…")

        # 给结尾的「…」留一个 token，否则截完还是超
        allowance = room - 1
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count(text[:middle]) <= allowance:
                low = middle
            else:
                high = middle - 1
        if low >= len(text):
            return item
        return replace_item_text(item, text[: max(1, low)] + "…")

    def __repr__(self) -> str:
        name = "default" if self.tokenizer is None else getattr(
            self.tokenizer, "__name__", "custom"
        )
        return f"TokenBudget(max_tokens={self.max_tokens}, tokenizer={name})"
