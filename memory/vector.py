# memory/vector.py
'''向量记忆：切分 → 向量化 → 存储 → 召回 → 重排。

`VectorMemory` 自己不碰任何数据库、不碰任何模型，它只做**编排**：

    VectorMemory(EmbeddingClient, VectorStore, Reranker?)

这三个都是**接口**，实现放在 `memory/adapters/` 里（OpenAI / BGE / Chroma /
Qdrant / Cohere…）。于是：

- 核心零依赖：`HashEmbedding` + `InMemoryVectorStore` 就能跑通整条链路
- 换后端不动业务代码：同一份 `VectorMemory`，换一个 store 就是另一套基础设施
- 好测：测试里注入假的 embedding / store，不联网、不装库

召回之后一定过 `TokenBudget`（如果给了）—— 拼进 prompt 之前没人想要 20 条笔记。
'''
from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from promptkit.memory.base import Memory, MemoryError, MemoryItem, empty
from promptkit.memory.budget import TokenBudget

__all__ = [
    "EmbeddingClient",
    "VectorStore",
    "Reranker",
    "InMemoryVectorStore",
    "VectorMemory",
    "chunk_text",
    "cosine",
]


# ============================ 接口 ============================
class EmbeddingClient(ABC):
    '''把文本变成向量。

    只需要实现 `embed()`；`embed_many()` 默认逐个调，批量接口（OpenAI 支持
    一次传一批）可以覆盖它来省网络往返。
    '''

    #: 向量维度（知道就填，方便存储层校验；不知道就留 None）
    dimension: Optional[int] = None

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        '''一段文本 -> 一个向量。'''

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        '''多段文本 -> 多个向量（默认就是循环）。'''
        return [self.embed(text) for text in texts]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} dimension={self.dimension}>"


class VectorStore(ABC):
    '''向量存储。按 `namespace` 隔离多用户 / 多主题。

    四个方法就够：`upsert` / `search` / `delete` / `count`。
    '''

    @abstractmethod
    def upsert(self, namespace: str, item: MemoryItem) -> None:
        '''写入或覆盖一条（同 namespace 同 id 就是覆盖）。'''

    @abstractmethod
    def search(
        self, namespace: str, vector: Sequence[float], top_k: int = 5
    ) -> List[MemoryItem]:
        '''粗排：返回最相似的 `top_k` 条，`score` 已填好。'''

    def delete(
        self,
        namespace: str,
        *,
        ids: Optional[Sequence[str]] = None,
        key: Optional[str] = None,
    ) -> int:
        '''删记录：按 id 删，或按 `metadata["key"]` 删（逻辑键的所有 chunk）。

        返回删掉了几条。默认抛错 —— 「悄悄删不掉」比「报错」危险得多。
        '''
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 delete()；"
            "可遗忘是记忆的基本要求，请补上这个方法"
        )

    def count(self, namespace: str) -> int:
        '''这个 namespace 里有多少条（调试 / 测试用）。'''
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 count()"
        )

    def clear(self, namespace: str) -> None:
        '''清空一个 namespace。'''
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 clear()"
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class Reranker(ABC):
    '''精排：粗排回来的候选，用更强的模型重新打分。'''

    @abstractmethod
    def rerank(
        self, query: str, items: Sequence[MemoryItem], top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        '''返回重排后的列表（`score` 换成重排分数）。'''


# ============================ 工具函数 ============================
def chunk_text(text: str, size: int = 500, overlap: int = 60) -> List[str]:
    '''按字符切 chunk，带 overlap。

    为什么按字符而不是按词：中文没有空格，按词切需要分词器（又一个依赖）；
    按字符切对中英混排最稳。代价是可能把一个词切成两半 —— 所以才有
    overlap：相邻 chunk 重叠一段，切边上的信息不至于两边都丢。

        chunk_text("abc" * 300, size=500, overlap=60)   # 每段 500 字，重叠 60
    '''
    text = "" if text is None else str(text)
    if not text:
        return []
    if size < 1:
        raise ValueError(f"size 必须是正整数，得到 {size!r}")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"overlap 必须满足 0 <= overlap < size（当前 size={size}, overlap={overlap}）；"
            "重叠不小于块长会导致死循环"
        )

    step = size - overlap
    chunks: List[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        if piece.strip():
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    '''余弦相似度（0 向量算 0 分，别造出 NaN）。'''
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError(
            f"向量维度不一致：{len(left)} vs {len(right)}。"
            "换过 embedding 模型的话，旧向量要重建"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _as_text(value: Any) -> str:
    '''写进向量记忆的内容统一成文本（dict / pydantic 转 JSON）。'''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), ensure_ascii=False)
    return str(value)


# ============================ 零依赖的默认实现 ============================
class InMemoryVectorStore(VectorStore):
    '''进程内的向量存储：一个 dict + 线性扫描。

    用途：测试、小规模笔记（几千条以内）、以及抄它的骨架去写自己的后端。
    **重启就没了** —— 要持久化请用 Chroma / Qdrant，或者自己实现 `VectorStore`。
    '''

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], MemoryItem] = {}

    def upsert(self, namespace: str, item: MemoryItem) -> None:
        self._items[(str(namespace), str(item.id))] = replace(
            item, metadata=dict(item.metadata)
        )

    def search(
        self, namespace: str, vector: Sequence[float], top_k: int = 5
    ) -> List[MemoryItem]:
        namespace = str(namespace)
        scored: List[MemoryItem] = []
        for (stored_ns, _), item in self._items.items():
            if stored_ns != namespace:
                continue
            score = cosine(vector, item.vector or ())
            scored.append(replace(item, score=score))
        scored.sort(key=lambda entry: entry.score, reverse=True)
        return scored[: max(1, int(top_k))]

    def delete(
        self,
        namespace: str,
        *,
        ids: Optional[Sequence[str]] = None,
        key: Optional[str] = None,
    ) -> int:
        namespace = str(namespace)
        doomed: List[Tuple[str, str]] = []
        for (stored_ns, stored_id), item in self._items.items():
            if stored_ns != namespace:
                continue
            if ids is not None and stored_id not in {str(i) for i in ids}:
                continue
            if key is not None and item.key != str(key):
                continue
            doomed.append((stored_ns, stored_id))
        for entry in doomed:
            del self._items[entry]
        return len(doomed)

    def count(self, namespace: str) -> int:
        namespace = str(namespace)
        return sum(1 for stored_ns, _ in self._items if stored_ns == namespace)

    def clear(self, namespace: str) -> None:
        '''清空一个 namespace（两个过滤条件都不给 = 这个 namespace 全删）。'''
        self.delete(str(namespace))

    def __len__(self) -> int:
        return len(self._items)


# ============================ 向量记忆 ============================
class VectorMemory(Memory):
    '''长期记忆（非结构化）：写入时切块向量化，召回时相似度 + 重排 + 预算。

        # 进程内（测试 / 原型）
        memory = VectorMemory(embed_client=OpenAIEmbedding(), namespace="learner_001")

        # 存进一个文件夹（持久化，零依赖）
        memory = VectorMemory(embed_client=..., namespace="learner_001",
                              path="~/.promptkit/memory/vector/learner_001")

        # 换成向量库，接口不变
        memory = VectorMemory(embed_client=..., store=ChromaVectorStore(collection),
                              namespace="learner_001", reranker=CohereReranker(client),
                              budget=TokenBudget(2000))

        memory.write("notes", "...很长的笔记...")
        memory.recall("退款怎么处理", top_k=3)
        memory.close()          # 落盘（FolderVectorStore 需要；内存库是空操作）

    **只有 `write` / `recall`**：向量库没有「按键读原文」的语义
    （原文被切成一块一块了），要 KV 就用 `KVMemory`。这条刻意留着
    `NotImplementedError`，而不是返回第一块假装能读。
    '''

    def __init__(
        self,
        embed_client: EmbeddingClient,
        store: Optional[VectorStore] = None,
        namespace: str = "default",
        reranker: Optional[Reranker] = None,
        *,
        path: Optional[str] = None,
        budget: Optional[TokenBudget] = None,
        chunk_size: int = 500,
        overlap: int = 60,
        max_age: Optional[float] = None,
        decay_half_life: Optional[float] = None,
        min_score: Optional[float] = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if embed_client is None:
            raise ValueError("VectorMemory 需要 embed_client（EmbeddingClient 实例）")
        if store is not None and path is not None:
            raise ValueError(
                "store= 和 path= 只能给一个：path= 只是 store=FolderVectorStore(path) 的简写"
            )
        if path is not None:
            from promptkit.memory.folder_store import FolderVectorStore

            store = FolderVectorStore(path)
        self.embed_client = embed_client
        self.store: VectorStore = store if store is not None else InMemoryVectorStore()
        self.reranker = reranker
        self.namespace = str(namespace)
        self.budget = budget
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        #: 超过这么多秒的记录直接不召回（按 `metadata["ts"]` 算）
        self.max_age = max_age
        #: 每过这么多秒，分数减半（越旧越往后排）；None = 不降权
        self.decay_half_life = decay_half_life
        #: 相似度低于它的结果直接丢。默认 0.0 = 把「毫不相关」（余弦 <= 0）的噪音挡掉；
        #: 想让阈值更严就设 0.2 / 0.3，`None` 表示什么都留
        self.min_score = min_score
        #: 每条记录都带上的固定 metadata（`{"source": "notes", "owner": "u1"}`）
        self.metadata: Dict[str, Any] = dict(metadata or {})

        # 构造时就校验 chunk 参数，别等第一次 write 才炸
        chunk_text("x", self.chunk_size, self.overlap)

    # ---------------- 写 ----------------
    def write(self, key: str, value: Any) -> None:
        '''把一条内容切块后写进去；同 key 重写会**覆盖**旧的所有 chunk。'''
        key = str(key)
        text = _as_text(value)
        if empty(text):
            # 空内容不写：它以后会被召回出来、占预算、还什么都不说
            self.store.delete(self.namespace, key=key)
            return

        chunks = chunk_text(text, self.chunk_size, self.overlap)
        vectors = self.embed_client.embed_many(chunks)
        if len(vectors) != len(chunks):
            raise MemoryError(
                f"embed_many 返回了 {len(vectors)} 个向量，但文本有 {len(chunks)} 段"
            )

        # 先清旧 chunk：否则「原来 10 段、现在 3 段」时会留下 7 段孤儿
        self.store.delete(self.namespace, key=key)
        now = time.time()
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.store.upsert(
                self.namespace,
                MemoryItem(
                    id=f"{key}#{index}",
                    text=chunk,
                    vector=list(vector),
                    metadata={
                        # 用户 metadata 先放，系统字段后放：`key` / `ts` 不能被覆盖，
                        # 否则 forget() 和过期判断会莫名其妙失效
                        **self.metadata,
                        "key": key,
                        "chunk": index,
                        "chunks": len(chunks),
                        "ts": now,
                    },
                ),
            )

    # ---------------- 读 / 遗忘 ----------------
    def read(self, key: str, default: Any = None) -> Any:
        '''向量记忆不支持按键读原文（原文已经切块了）。'''
        raise NotImplementedError(
            "VectorMemory 没有 read()：内容被切成 chunk 了，读不回原文。"
            "要按键读结构化值请用 KVMemory；要语义召回请用 recall()"
        )

    def forget(self, key: str) -> bool:
        '''删掉一个逻辑键的所有 chunk。'''
        return self.store.delete(self.namespace, key=str(key)) > 0

    # ---------------- 召回 ----------------
    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''粗排 -> （可选）重排 -> （可选）预算裁剪 -> 公开格式。'''
        query = "" if query is None else str(query).strip()
        if not query:
            return []
        top_k = max(1, int(top_k))

        vector = self.embed_client.embed(query)
        # 多取一些候选给重排器用；没有重排器时多拿的会被下面截掉
        candidates = self.store.search(self.namespace, vector, top_k=top_k * 2)
        candidates = self._apply_age(candidates)
        if self.min_score is not None:
            candidates = [
                item for item in candidates if item.score > float(self.min_score)
            ]

        if self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k)

        if self.budget is not None:
            candidates = self.budget.trim(candidates)

        return [item.to_dict() for item in candidates[:top_k]]

    # ---------------- 内部 ----------------
    def _apply_age(self, items: Sequence[MemoryItem]) -> List[MemoryItem]:
        '''过期（max_age）与降权（decay_half_life）。'''
        if self.max_age is None and not self.decay_half_life:
            return list(items)

        now = time.time()
        kept: List[MemoryItem] = []
        for item in items:
            ts = _timestamp(item.metadata.get("ts"), now)
            age = max(0.0, now - ts)
            if self.max_age is not None and age > float(self.max_age):
                continue
            if self.decay_half_life:
                factor = 0.5 ** (age / float(self.decay_half_life))
                item = replace(item, score=item.score * factor)
            kept.append(item)
        kept.sort(key=lambda entry: entry.score, reverse=True)
        return kept

    def count(self) -> int:
        '''这个 namespace 里存了多少 chunk。'''
        return self.store.count(self.namespace)

    def clear(self) -> None:
        '''清空这个 namespace。'''
        self.store.clear(self.namespace)

    # ---------------- 生命周期 ----------------
    def close(self) -> None:
        '''把存储层落盘。

        `FolderVectorStore` 会在这里压实写盘；内存库、Chroma、Qdrant
        没有 `close()` 就什么都不做。
        '''
        closer = getattr(self.store, "close", None)
        if callable(closer):
            closer()

    def flush(self) -> None:
        '''立刻落盘（不想等 `close()` 时用）。'''
        flusher = getattr(self.store, "flush", None)
        if callable(flusher):
            flusher()

    def __enter__(self) -> "VectorMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        bits = [f"namespace={self.namespace!r}", f"store={type(self.store).__name__}"]
        if self.reranker is not None:
            bits.append(f"reranker={type(self.reranker).__name__}")
        if self.budget is not None:
            bits.append(repr(self.budget))
        return f"<VectorMemory {' '.join(bits)}>"


def _timestamp(raw: Any, fallback: float) -> float:
    '''metadata["ts"] 可能是 float、字符串、甚至没有 —— 都要能算。'''
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback
