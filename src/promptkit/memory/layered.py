# memory/layered.py
'''把 KV 和向量叠成「一份长期记忆」。

前面三个实现各管一摊：`KVMemory` 只会读写、`VectorMemory` 只会写和召回。
但真实场景里它们往往是**同一份记忆的两面**：

    画像 / 偏好 / 进度   -> 要按键精确读出来     -> KV
    笔记 / 历史 / 长文档 -> 要按语义找出来       -> 向量

`LayeredMemory` 就是把这两面拼起来，对外仍然是那三个方法：

    memory = long_term_memory("learner_001")     # 一行拿到「KV + 向量」
    memory.read("level")                          # 走 KV
    memory.recall("退款怎么处理")                  # 走向量
    memory.write("notes", "……")                   # 两边都写（mirror=True）

`mirror=True` 意味着**每条写入都会再算一次向量**（要花钱 / 花时间）。
想省下来就 `mirror=False`，需要被召回的内容自己用 `VectorMemory` 直接写。
'''
from __future__ import annotations

import os

from typing import TYPE_CHECKING, Dict, List, Optional, cast

from promptkit.memory.base import Memory
from promptkit.memory.kv import KVMemory
from promptkit.memory.vector import EmbeddingClient, Reranker, VectorMemory, VectorStore

if TYPE_CHECKING:
    from promptkit.memory.budget import TokenBudget

__all__ = ["LayeredMemory", "long_term_memory"]


class LayeredMemory(Memory):
    '''KV + 向量的组合记忆。

    分工是固定的、也是好预测的：

    | 方法 | 去哪 |
    | --- | --- |
    | `read` | KV（没有 KV 后端就抛 `NotImplementedError`） |
    | `write` | KV；`mirror=True` 时**也**写向量 |
    | `recall` | 向量（没有向量后端就抛 `NotImplementedError`） |
    | `forget` | 两边都删 |

    「一边有、一边没有」是允许的：只有 KV 就是普通的 KV 记忆，
    只有向量就是普通的向量记忆 —— 只是统一在同一个对象上。
    '''

    def __init__(
        self,
        kv: Optional[Memory] = None,
        vector: Optional[Memory] = None,
        *,
        namespace: Optional[str] = None,
        mirror: bool = True,
    ):
        if kv is None and vector is None:
            raise ValueError(
                "LayeredMemory 至少要有 kv= 或 vector= 之一"
            )
        self.kv = kv
        self.vector = vector
        self.mirror = bool(mirror)
        self.namespace = str(
            namespace
            or getattr(kv, "namespace", None)
            or getattr(vector, "namespace", None)
            or "default"
        )

    # ---------------- 读 ----------------
    def read(self, key: str, default: object = None) -> object:
        if self.kv is None:
            raise NotImplementedError(
                "LayeredMemory 没有 KV 后端，读不了键；"
                "要按键读请建 LayeredMemory(kv=KVMemory(...), ...)"
            )
        return self.kv.read(key, default)

    # ---------------- 写 ----------------
    def write(self, key: str, value: object) -> None:
        if self.kv is not None:
            self.kv.write(key, value)
        if self.vector is not None and self.mirror:
            self.vector.write(key, value)

    # ---------------- 召回 ----------------
    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        if self.vector is None:
            raise NotImplementedError(
                "LayeredMemory 没有向量后端，召回不了；"
                "要语义召回请建 LayeredMemory(..., vector=VectorMemory(...))"
            )
        return self.vector.recall(query, top_k=top_k)

    # ---------------- 遗忘 ----------------
    def forget(self, key: str) -> bool:
        removed = False
        if self.kv is not None:
            removed = bool(self.kv.forget(key)) or removed
        if self.vector is not None:
            removed = bool(self.vector.forget(key)) or removed
        return removed

    # ---------------- 生命周期 ----------------
    def close(self) -> None:
        '''把两个后端都关掉（KV 关连接、向量库落盘）。'''
        for backend in (self.kv, self.vector):
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> "LayeredMemory":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        # 注意用 `is None` 判断：`KVMemory` 有 `__len__`，空库是「假值」，
        # 写成 `if self.kv` 会把「空的 KV 后端」显示成 None
        return (
            f"<LayeredMemory namespace={self.namespace!r} "
            f"kv={type(self.kv).__name__ if self.kv is not None else None} "
            f"vector={type(self.vector).__name__ if self.vector is not None else None} "
            f"mirror={self.mirror}>"
        )


def long_term_memory(
    namespace: str = "default",
    *,
    path: Optional[str] = None,
    vector_path: Optional[str] = None,
    embed_client: Optional[EmbeddingClient] = None,
    store: Optional[VectorStore] = None,
    reranker: Optional[Reranker] = None,
    budget: Optional["TokenBudget"] = None,
    mirror: bool = True,
    chunk_size: int = 500,
    overlap: int = 60,
    min_score: Optional[float] = 0.0,
) -> LayeredMemory:
    '''一行组装「长期记忆」：SQLite 做 KV，向量做召回。

        from promptkit.memory import long_term_memory
        from promptkit.memory.adapters import OpenAIEmbedding

        # 全用默认位置（<data_dir>/memory/{kv,vector}/learner_001）
        memory = long_term_memory(
            "learner_001",
            embed_client=OpenAIEmbedding(model="text-embedding-3-small"),
        )

        # 自己指定：KV 一个文件，向量一个文件夹
        memory = long_term_memory(
            "learner_001",
            path="~/data/tutor.db",
            vector_path="~/data/tutor-vectors",
            embed_client=...,
        )

    `embed_client` 省略时只建 KV 后端（召回会明确报错并提示怎么补），
    这样「先不装向量、只用结构化记忆」也是一条正常路径。
    `store=` 给了就用它（Chroma / Qdrant …），此时 `vector_path` 不起作用。
    '''
    kv = KVMemory(path, namespace=namespace)
    if embed_client is None:
        return LayeredMemory(kv=kv, namespace=namespace, mirror=False)

    if store is not None:
        vector = VectorMemory(
            embed_client,
            store,
            namespace=namespace,
            reranker=reranker,
            budget=budget,
            chunk_size=chunk_size,
            overlap=overlap,
            min_score=min_score,
        )
    else:
        from promptkit.memory.paths import default_vector_dir

        vector = VectorMemory(
            embed_client,
            namespace=namespace,
            path=vector_path or default_vector_dir(namespace),
            reranker=reranker,
            budget=budget,
            chunk_size=chunk_size,
            overlap=overlap,
            min_score=min_score,
        )
    return LayeredMemory(kv=kv, vector=vector, namespace=namespace, mirror=mirror)


def as_memory(
    source: object,
    *,
    namespace: Optional[str] = None,
    embed_client: Optional[EmbeddingClient] = None,
    **kwargs,
) -> Memory:
    '''把「Memory 对象 / 路径 / 命名空间」统一成一份长期记忆。

    `MEMORY_REFERENCE(...)` / `WRITE_MEMORY(...)` / `Compact(...).To(...)`
    都用这一套规则，所以三处的写法可以一样：

        as_memory(memory)               # 已经是记忆对象 → 原样返回
        as_memory("learner_001")        # 命名空间 → 默认数据目录
        as_memory("data/tutor.db")      # 路径 → KV 落在这个文件，
                                        # 向量落在同级 <路径>.vectors/

    路径判定与工厂一致：含分隔符，或以 `.db` / `.sqlite` / `.db3` / `.jsonl`
    结尾。`embed_client` 省略时用本地 `HashEmbedding`（免密钥、不联网）。

    注意：同一个路径写两次会建出**两个实例**——KV（SQLite）在磁盘上互通，
    向量索引是进程内的；要「写了立刻能召回」，请只建一份再传给各处。
    '''
    if source is None:
        raise TypeError("as_memory 需要 Memory 对象 / 路径 / 命名空间，得到 None")
    if callable(getattr(source, "write", None)) or callable(
        getattr(source, "recall", None)
    ):
        return cast(Memory, source)

    if isinstance(source, (str, os.PathLike)):
        if embed_client is None:
            from promptkit.memory.adapters import HashEmbedding

            embed_client = HashEmbedding(dimension=512)

        text = os.fspath(source)
        is_path = (
            os.sep in text
            or (os.altsep is not None and os.altsep in text)
            or text.endswith((".db", ".sqlite", ".db3", ".jsonl"))
        )
        if is_path:
            identity = namespace or (
                os.path.splitext(os.path.basename(text))[0] or "default"
            )
            return long_term_memory(
                identity,
                path=text,
                vector_path=text + ".vectors",
                embed_client=embed_client,
                **kwargs,
            )
        return long_term_memory(namespace or text, embed_client=embed_client, **kwargs)

    raise TypeError(
        "as_memory 需要 Memory 对象、路径或命名空间，"
        f"得到 {type(source).__name__}"
    )
