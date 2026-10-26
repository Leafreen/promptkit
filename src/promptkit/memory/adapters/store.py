# memory/adapters/store.py
'''`VectorStore` 的现成适配器。

| 适配器 | 依赖 | 说明 |
| --- | --- | --- |
| `InMemoryVectorStore` | 无 | 进程内、线性扫描；测试 / 小规模 |
| `ChromaVectorStore` | `chromadb` | 包一个**已经建好的** Collection（或 `at()` 自己开） |
| `QdrantVectorStore` | `qdrant-client` | 包一个已经建好 collection 的 Client |

还有一个**文件夹型**的零依赖实现：`promptkit.memory.folder_store.FolderVectorStore`
（`VectorMemory(..., path=...)` 用的就是它）。

两个原则：

1. **只包不建**：collection / 索引由你自己创建（维度、距离函数、持久化位置
   都是运维决策，不该藏在记忆层里）。
2. **namespace 走 metadata 过滤**：一个 collection 里可以放很多用户
   （`metadata.namespace`），召回和删除都带上它 —— 多用户隔离不是靠
   「记得换个 collection」。

FAISS / pgvector 没写：FAISS 是**索引**不是库，删记录要重建，得自己维护
id ↔ 文本的映射；pgvector 直接用 `psycopg` 写四个方法就行。两者都可以照着
`InMemoryVectorStore` 的骨架实现 `upsert / search / delete / count`。
'''
from __future__ import annotations

import json
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Sequence

from promptkit.memory.base import MemoryItem
from promptkit.memory.vector import InMemoryVectorStore, VectorStore

__all__ = ["InMemoryVectorStore", "ChromaVectorStore", "QdrantVectorStore"]


def _scalarize(metadata: Dict[str, Any]) -> Dict[str, Any]:
    '''Chroma / Qdrant 的 metadata 只放标量；复杂值 JSON 成字符串。

    干脆丢掉不是不行，但「写进去的元数据查不出来」更难排查 ——
    至少留一个能看的字符串。
    '''
    result: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = json.dumps(value, ensure_ascii=False)
    return result


class ChromaVectorStore(VectorStore):
    '''把 `chromadb` 的 Collection 包成 `VectorStore`。

        import chromadb
        client = chromadb.PersistentClient(path="chroma")
        collection = client.get_or_create_collection(
            "notes", metadata={"hnsw:space": "cosine"},
        )
        store = ChromaVectorStore(collection)

    ⚠️ 相似度换算成 `score = 1 - distance`，只有 collection 建的时候用了
    `hnsw:space="cosine"` 才是 0~1 的相似度。用默认的 l2 的话分数只用来排序，
    别拿去跟阈值比。
    '''

    def __init__(self, collection: Any):
        if collection is None:
            raise ValueError("ChromaVectorStore 需要 chromadb 的 Collection 实例")
        self.collection = collection

    @classmethod
    def at(
        cls,
        path: Optional[str] = None,
        collection: str = "promptkit",
        *,
        url: Optional[str] = None,
    ) -> "ChromaVectorStore":
        '''自己开一个（本地持久化目录或 chroma 服务），不用先建 collection。

            ChromaVectorStore.at("~/.promptkit/memory/vector/user_001")
            ChromaVectorStore.at(url="http://localhost:8000", collection="tutor")
        '''
        if not path and not url:
            raise ValueError("ChromaVectorStore.at 需要 path=（本地目录）或 url=（服务地址）")
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaVectorStore 需要 chromadb：pip install chromadb"
            ) from None

        if url:
            parsed = urllib.parse.urlparse(str(url))
            client = chromadb.HttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or (443 if parsed.scheme == "https" else 8000),
                ssl=parsed.scheme == "https",
            )
        else:
            from promptkit.memory.paths import expand_path

            client = chromadb.PersistentClient(path=expand_path(path))
        # cosine 必须在这里指定：默认是 l2，score 就不是「相似度」了
        return cls(
            client.get_or_create_collection(
                str(collection), metadata={"hnsw:space": "cosine"}
            )
        )

    def upsert(self, namespace: str, item: MemoryItem) -> None:
        metadata = {"namespace": str(namespace), **_scalarize(item.metadata)}
        self.collection.upsert(
            ids=[str(item.id)],
            embeddings=[list(item.vector or [])],
            documents=[item.text],
            metadatas=[metadata],
        )

    def search(
        self, namespace: str, vector: Sequence[float], top_k: int = 5
    ) -> List[MemoryItem]:
        result = self.collection.query(
            query_embeddings=[list(vector)],
            n_results=max(1, int(top_k)),
            where={"namespace": str(namespace)},
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        items: List[MemoryItem] = []
        for index, item_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            metadata.pop("namespace", None)
            distance = float(distances[index]) if index < len(distances) else 1.0
            items.append(
                MemoryItem(
                    id=str(item_id),
                    text="" if text is None else str(text),
                    metadata=metadata,
                    score=1.0 - distance,
                )
            )
        return items

    def delete(
        self,
        namespace: str,
        *,
        ids: Optional[Sequence[str]] = None,
        key: Optional[str] = None,
    ) -> int:
        if ids is not None:
            target = [str(item) for item in ids]
            if target:
                self.collection.delete(ids=target)
            return len(target)

        where: Dict[str, Any] = {"namespace": str(namespace)}
        if key is not None:
            where["key"] = str(key)
        found = self.collection.get(where=where, include=[])
        target = [str(item) for item in (found or {}).get("ids") or []]
        if target:
            self.collection.delete(ids=target)
        return len(target)

    def count(self, namespace: str) -> int:
        found = self.collection.get(where={"namespace": str(namespace)}, include=[])
        return len((found or {}).get("ids") or [])

    def clear(self, namespace: str) -> None:
        self.delete(str(namespace))

    def __repr__(self) -> str:
        name = getattr(self.collection, "name", "collection")
        return f"<ChromaVectorStore {name!r}>"


class QdrantVectorStore(VectorStore):
    '''把 `qdrant_client.QdrantClient` 包成 `VectorStore`。

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(path="qdrant")
        client.create_collection(
            "notes", vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(client, "notes")

    字符串 id 会被映射成稳定的 UUID（Qdrant 只收 int / UUID）；namespace 和
    `metadata.key` 都放 payload，删除和召回都靠 payload 过滤。

    `models=` 是为了测试 / 依赖注入：不传就 `import qdrant_client.models`。
    '''

    def __init__(self, client: Any, collection_name: str, *, models: Any = None):
        if client is None:
            raise ValueError("QdrantVectorStore 需要 QdrantClient 实例")
        self.client = client
        self.collection_name = str(collection_name)
        self.models = models if models is not None else _qdrant_models()

    def upsert(self, namespace: str, item: MemoryItem) -> None:
        payload = {
            "text": item.text,
            **_scalarize(item.metadata),
            "key": item.key,
            "namespace": str(namespace),
        }
        point = self.models.PointStruct(
            id=_point_id(item.id), vector=list(item.vector or []), payload=payload
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(
        self, namespace: str, vector: Sequence[float], top_k: int = 5
    ) -> List[MemoryItem]:
        condition = self._condition(namespace)
        limit = max(1, int(top_k))
        query_points = getattr(self.client, "query_points", None)
        if callable(query_points):  # qdrant-client >= 1.10
            result = query_points(
                collection_name=self.collection_name,
                query=list(vector),
                limit=limit,
                query_filter=condition,
                with_payload=True,
            )
            points = getattr(result, "points", result)
        else:  # 老版本
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=list(vector),
                limit=limit,
                query_filter=condition,
                with_payload=True,
            )
        return [_from_point(point) for point in points or ()]

    def delete(
        self,
        namespace: str,
        *,
        ids: Optional[Sequence[str]] = None,
        key: Optional[str] = None,
    ) -> int:
        selector = self._condition(namespace, key)
        if ids is not None:
            selector = self.models.Filter(
                must=[
                    self.models.HasIdCondition(has_id=[_point_id(item) for item in ids])
                ]
            )
        self.client.delete(
            collection_name=self.collection_name, points_selector=selector
        )
        # Qdrant 的 delete 不返回条数；这里如实返回 -1（未知）
        return -1 if ids is None else len(list(ids))

    def count(self, namespace: str) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._condition(namespace),
            exact=True,
        )
        return int(getattr(result, "count", result) or 0)

    def _condition(self, namespace: str, key: Optional[str] = None) -> Any:
        must = [
            self.models.FieldCondition(
                key="namespace",
                match=self.models.MatchValue(value=str(namespace)),
            )
        ]
        if key is not None:
            must.append(
                self.models.FieldCondition(
                    key="key", match=self.models.MatchValue(value=str(key))
                )
            )
        return self.models.Filter(must=must)

    def __repr__(self) -> str:
        return f"<QdrantVectorStore {self.collection_name!r}>"


def _point_id(raw: str) -> str:
    '''字符串 -> 稳定的 UUID（同一个 id 永远映射到同一个点）。'''
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw)))


def _from_point(point: Any) -> MemoryItem:
    payload = dict(getattr(point, "payload", None) or {})
    text = payload.pop("text", "")
    payload.pop("namespace", None)
    return MemoryItem(
        id=str(getattr(point, "id", "")),
        text="" if text is None else str(text),
        metadata=payload,
        score=float(getattr(point, "score", 0.0) or 0.0),
    )


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError:
        raise ImportError(
            "QdrantVectorStore 需要 qdrant-client：pip install qdrant-client"
        ) from None
    return models
