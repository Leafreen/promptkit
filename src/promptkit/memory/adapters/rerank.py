# memory/adapters/rerank.py
'''`Reranker` 的现成适配器：粗排召回之后的精排。

| 适配器 | 依赖 | 说明 |
| --- | --- | --- |
| `NoopReranker` | 无 | 什么都不做（只想统一接口时用） |
| `LlmReranker` | 无（用你自己的 Agent） | 让模型给候选排序；最贵也最省事 |
| `CohereReranker` | `cohere` | Cohere Rerank API |
| `CrossEncoderReranker` | `sentence-transformers` | 本地 BGE / bge-reranker 等 |

重排的价值在于「粗排只看向量距离，精排看 query 和文档的交互」——
代价是每条候选都要过一次模型，所以 `top_k * 2` 到 `top_k * 4` 的候选量比较合适。
'''
from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional, Sequence, Union, cast

from promptkit.memory.base import MemoryItem, item_text
from promptkit.memory.vector import Reranker

__all__ = [
    "NoopReranker",
    "LlmReranker",
    "CohereReranker",
    "CrossEncoderReranker",
]


class NoopReranker(Reranker):
    '''原样返回（截到 top_k）。想统一调用路径又不想多花钱时用。'''

    def rerank(
        self, query: str, items: Sequence[MemoryItem], top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        result = list(items)
        return result[:top_k] if top_k else result

    def __repr__(self) -> str:
        return "<NoopReranker>"


class LlmReranker(Reranker):
    '''让模型给候选排序（一次 ask，输出 JSON 下标数组）。

    好处：零新依赖，中文语义排序通常比纯向量好；坏处：慢、要 token、
    结果依赖模型听话。**一定要给它设上限**（`max_candidates`）——
    否则召回 50 条就把 50 条全塞给模型了。
    '''

    def __init__(
        self,
        agent: object = None,
        *,
        max_chars: int = 200,
        max_candidates: int = 20,
    ):
        self.agent = agent
        self.max_chars = int(max_chars)
        self.max_candidates = int(max_candidates)

    def rerank(
        self, query: str, items: Sequence[MemoryItem], top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        candidates = list(items)[: max(1, self.max_candidates)]
        if len(candidates) <= 1:
            return candidates[:top_k] if top_k else candidates

        agent = self.agent
        if agent is None:
            from promptkit.agent import Agent

            agent = Agent.default()
        if agent is None:
            raise ValueError(
                "LlmReranker 没有可用的 agent：传 agent= 或者先 `Agent()`"
            )

        listing = "\n".join(
            f"{index}. {item_text(item)[: self.max_chars]}"
            for index, item in enumerate(candidates)
        )
        prompt = (
            f"问题：{query}\n\n候选资料：\n{listing}\n\n"
            "请按「对回答这个问题有多相关」从高到低排序，"
            '只输出 JSON，形如 {"order": [2, 0, 1]}，不要解释。'
        )
        raw = getattr(agent, "ask")({"role": "user", "content": prompt})
        order = self._order(raw, len(candidates))
        return self._apply(candidates, order, top_k)

    # ---------------- 内部 ----------------
    @staticmethod
    def _order(raw: object, size: int) -> List[int]:
        '''从模型输出里抠出下标数组（JSON 优先，抠不出来就找数字）。'''
        text = "" if raw is None else str(raw)
        values: object = None
        try:
            from promptkit.core.model_action.caution_related.formatter import (
                get_formatter,
            )

            parsed = get_formatter(None).parse(text)
            if isinstance(parsed, dict):
                values = parsed.get("order") or parsed.get("ranking") or parsed.get("indexes")
            elif isinstance(parsed, list):
                values = parsed
        except Exception:  # noqa: BLE001 - 解析失败还有下面的兜底
            values = None

        if not isinstance(values, (list, tuple)):
            values = re.findall(r"\d+", text)

        order: List[int] = []
        for value in cast(Sequence[object], values or ()):
            try:
                index = int(cast(Union[str, int, float], value))
            except (TypeError, ValueError):
                continue
            if 0 <= index < size and index not in order:
                order.append(index)
        # 模型漏掉的按原顺序补在后面，保证一条不丢
        order.extend(index for index in range(size) if index not in order)
        return order

    @staticmethod
    def _apply(
        items: Sequence[MemoryItem], order: Sequence[int], top_k: Optional[int]
    ) -> List[MemoryItem]:
        total = len(order)
        reranked = [
            # 分数按名次倒着给：1.0, 0.95, 0.90…，让下游能看出顺序
            replace(items[index], score=1.0 - (rank / max(total, 1)) * 0.5)
            for rank, index in enumerate(order)
        ]
        return reranked[:top_k] if top_k else reranked


class CohereReranker(Reranker):
    '''Cohere Rerank（v3 / v4 都行）。

        import cohere
        CohereReranker(cohere.Client(), model="rerank-multilingual-v3.0")

    `client=` 也可以是一个只实现了 `rerank(...)` 的对象（方便测试 / 走代理）。
    '''

    def __init__(
        self,
        client: object = None,
        *,
        model: str = "rerank-multilingual-v3.0",
        api_key: Optional[str] = None,
    ):
        if client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "CohereReranker 需要 cohere：pip install cohere；"
                    "或者把已经建好的 client 用 client= 传进来"
                ) from None
            client = cohere.Client(api_key=api_key) if api_key else cohere.Client()
        self.client = client
        self.model = str(model)

    def rerank(
        self, query: str, items: Sequence[MemoryItem], top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        candidates = list(items)
        if not candidates:
            return []
        documents = [item_text(item) for item in candidates]
        response = getattr(self.client, "rerank")(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or len(documents),
        )
        rows = getattr(response, "results", response) or []
        result: List[MemoryItem] = []
        for row in rows:
            index = int(getattr(row, "index", -1))
            if not 0 <= index < len(candidates):
                continue
            score = float(getattr(row, "relevance_score", 0.0) or 0.0)
            result.append(replace(candidates[index], score=score))
        return result[:top_k] if top_k else result

    def __repr__(self) -> str:
        return f"<CohereReranker model={self.model!r}>"


class CrossEncoderReranker(Reranker):
    '''本地 cross-encoder（BGE reranker 系列都走这个）。

        CrossEncoderReranker("BAAI/bge-reranker-base")

    `model=` 也可以是一个已经加载好、带 `.predict(pairs)` 的对象。
    '''

    def __init__(
        self,
        model: object = None,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
    ):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "CrossEncoderReranker 需要 sentence-transformers：\n"
                    "  pip install sentence-transformers"
                ) from None
            model = CrossEncoder(model_name, device=device)
        if not callable(getattr(model, "predict", None)):
            raise TypeError(
                f"CrossEncoderReranker 需要带 .predict(pairs) 的模型，得到 {model!r}"
            )
        self.model = model

    def rerank(
        self, query: str, items: Sequence[MemoryItem], top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        candidates = list(items)
        if not candidates:
            return []
        pairs = [(query, item_text(item)) for item in candidates]
        scores = getattr(self.model, "predict")(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        result = [replace(item, score=float(score)) for item, score in ranked]
        return result[:top_k] if top_k else result

    def __repr__(self) -> str:
        return f"<CrossEncoderReranker {type(self.model).__name__}>"
