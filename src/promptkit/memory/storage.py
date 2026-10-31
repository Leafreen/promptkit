# memory/storage.py
'''「存哪」的统一配置：`StorageConfig` + `create_memory()` 工厂。

为什么要有这一层
----------------
`SessionMemory` 不需要文件（它包 `Agent.history`），`InMemoryMemory` 也不需要；
但 `KVMemory` 需要 `.db`，`VectorMemory` 需要文件夹或服务地址。把这些**位置**
塞进构造函数里会到处硬编码，塞进核心库的默认值又会把文件丢进用户的工作目录。

所以位置单独抽出来：

    from promptkit.memory import StorageConfig, create_memory

    memory = create_memory(StorageConfig(backend="sqlite", namespace="user_001"))
    # -> KVMemory("~/.promptkit/memory/kv/user_001.db", namespace="user_001")

    memory = create_memory(
        StorageConfig(backend="jsonl", namespace="user_001"),   # 目录型向量记忆
        embed_client=HashEmbedding(),
    )

`namespace` 是**必填**的：默认值会让两个用户的记忆悄悄串到一起。
`path` / `url` 都可以覆盖，默认值见 `memory.paths`。

支持的后端
----------
| backend | 得到 | 位置 |
| --- | --- | --- |
| `sqlite` | `KVMemory` | `<data_dir>/memory/kv/<ns>.db` 或 `path` |
| `jsonl`（别名 `folder` / `vector`） | `VectorMemory` + `FolderVectorStore` | `<data_dir>/memory/vector/<ns>/` 或 `path` |
| `chroma` | `VectorMemory` + `ChromaVectorStore` | `path`（本地）或 `url`（服务） |
| `qdrant` | `VectorMemory` + `QdrantVectorStore` | `url` |
| `memory` | `InMemoryMemory` | 进程内 |
| `session` | `SessionMemory` | 当前 `Agent` |
| `layered` | `LayeredMemory`（KV + 向量） | 上面两者组合 |

要接自己的实现（Redis、公司网关、FAISS…），注册一下就行：

    register_memory_backend("redis", lambda config, **kw: MyRedisMemory(config.url))
'''
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Mapping, Optional, Sequence, Union, cast

from promptkit.memory.base import InMemoryMemory, Memory
from promptkit.memory.kv import KVMemory
from promptkit.memory.layered import LayeredMemory
from promptkit.memory.paths import (
    default_kv_path,
    default_vector_dir,
    expand_path,
    safe_namespace_name,
)
from promptkit.memory.session import SessionMemory
from promptkit.memory.vector import EmbeddingClient, VectorMemory, VectorStore
from promptkit.types import JSONValue

__all__ = [
    "StorageConfig",
    "create_memory",
    "register_memory_backend",
    "BACKENDS",
]

#: 只能走网络的后端（必须给 url）
_REMOTE_BACKENDS = frozenset({"qdrant"})

#: 本地 / 混合后端（不会用 url）
_LOCAL_BACKENDS = frozenset({"sqlite", "jsonl", "folder", "vector", "memory", "session", "layered"})

#: 后端名 -> 工厂：`(config, *, embed_client=None, **kwargs) -> Memory`
BACKENDS: Dict[str, Callable[..., Memory]] = {}


@dataclass
class StorageConfig:
    '''「存哪 + 隔离给谁 + 用哪个后端」的一个小对象。

        StorageConfig(backend="sqlite", namespace="user_001")            # 默认目录
        StorageConfig(backend="sqlite", namespace="user_001",
                      path="~/data/tutor.db")                            # 指定文件
        StorageConfig(backend="jsonl", namespace="user_001",
                      path="~/data/vectors")                             # 指定文件夹
        StorageConfig(backend="qdrant", namespace="user_001",
                      url="http://localhost:6333",
                      options={"collection": "tutor"})

    `namespace` **没有默认值**：这是刻意的。`"default"` 一不小心就会变成
    「所有用户共用一份记忆」，而这种事上线之后才发现就很难拆了。
    '''

    backend: str
    #: 用户 / 租户 / 会话的隔离键，必填
    namespace: str
    #: 本地文件（sqlite）或文件夹（向量后端）
    path: Optional[str] = None
    #: 远程服务地址（qdrant / chroma）
    url: Optional[str] = None
    #: 后端自己的选项：collection / api_key / vector_backend /
    #: vector_path / vector_url / mirror …（`layered` 用后三个决定向量那边怎么存）
    options: Dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError(f"backend 必须是非空字符串，得到 {self.backend!r}")
        self.backend = self.backend.strip().lower().replace("-", "_")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError(
                "StorageConfig 必须显式给 namespace（比如 'user_001'）："
                "没有默认值，是为了不让不同用户的记忆串到一起"
            )
        self.namespace = self.namespace.strip()
        if self.path is not None:
            self.path = expand_path(str(self.path))
        if self.url is not None:
            self.url = str(self.url).strip() or None
        if not isinstance(self.options, dict):
            raise TypeError(
                f"options 需要 dict，得到 {type(self.options).__name__}"
            )

        if self.backend in _REMOTE_BACKENDS and not self.url:
            raise ValueError(
                f"backend={self.backend!r} 是远程后端，需要 url=（比如 "
                "http://localhost:6333）"
            )
        if self.url and self.backend in _LOCAL_BACKENDS:
            raise ValueError(
                f"backend={self.backend!r} 是本地后端，不用 url；要指定位置请用 path="
            )
        if self.backend == "chroma" and not (self.path or self.url):
            raise ValueError(
                "backend='chroma' 需要 path=（本地持久化目录）或 url=（chroma 服务）"
            )

    # ---------------- 位置解析 ----------------
    def kv_path(self) -> str:
        '''KV 数据库文件：显式 `path` 优先，否则默认数据目录。'''
        return self.path or default_kv_path(self.namespace)

    def vector_path(self) -> str:
        '''向量记忆的文件夹：显式 `path` 优先，否则默认数据目录。'''
        return self.path or default_vector_dir(self.namespace)

    def collection_name(self, prefix: str = "promptkit") -> str:
        '''向量库里的 collection 名（默认按 namespace 起名）。'''
        return str(
            self.options.get("collection")
            or f"{prefix}_{safe_namespace_name(self.namespace)}"
        )

    def describe(self) -> str:
        '''一行描述（日志 / 排查「到底存的哪儿」）。'''
        where = self.url or (self.path or "默认目录")
        return f"{self.backend}({where}, namespace={self.namespace!r})"

    def __repr__(self) -> str:
        return (
            f"StorageConfig(backend={self.backend!r}, namespace={self.namespace!r}, "
            f"path={self.path!r}, url={self.url!r}, options={self.options!r})"
        )


# ============================ 工厂 ============================
def register_memory_backend(
    name: str, factory: Callable[..., Memory], *, aliases: Sequence[str] = ()
) -> None:
    '''注册一个后端：`factory(config, embed_client=None, **kwargs) -> Memory`。

    内置后端也是用这个注册的 —— 所以扩展和「官方支持」走的是同一条路。
    '''
    if not isinstance(name, str) or not name.strip():
        raise ValueError("后端名必须是非空字符串")
    key = name.strip().lower().replace("-", "_")
    BACKENDS[key] = factory
    for alias in aliases:
        BACKENDS[str(alias).strip().lower().replace("-", "_")] = factory


def create_memory(
    config: Union[StorageConfig, Mapping[str, object]],
    *,
    embed_client: Optional[EmbeddingClient] = None,
    **kwargs,
) -> Memory:
    '''按 `StorageConfig` 造一个记忆对象。

    额外参数原样传给具体实现（`budget=` / `reranker=` / `chunk_size=` /
    `timeout=` …），所以工厂不是「只能给默认值」的阉割版。
    '''
    if isinstance(config, Mapping):
        config = StorageConfig(
            backend=cast(str, config["backend"]),
            namespace=cast(str, config["namespace"]),
            path=cast(Optional[str], config.get("path")),
            url=cast(Optional[str], config.get("url")),
            options=cast(Dict[str, JSONValue], config.get("options", {})),
        )
    if not isinstance(config, StorageConfig):
        raise TypeError(
            f"create_memory 需要 StorageConfig（或等价的 dict），得到 {type(config).__name__}"
        )

    factory = BACKENDS.get(config.backend)
    if factory is None:
        raise ValueError(
            f"不认识的后端 {config.backend!r}；已注册：{', '.join(sorted(BACKENDS))}。"
            "要接自己的实现可以用 register_memory_backend()"
        )
    return factory(config, embed_client=embed_client, **kwargs)


# ---------------- 内置后端 ----------------
def _kv_memory(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    path = config.kv_path()
    if os.path.isdir(path):
        raise ValueError(
            f"sqlite 的 path 要指向一个 .db 文件，得到的是文件夹：{path!r}；"
            "想用文件夹存向量请用 backend='jsonl'，或者把 path 写成 "
            "os.path.join(folder, 'memory.db')"
        )
    return KVMemory(path, namespace=config.namespace, **kwargs)


def _in_memory(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    return InMemoryMemory(namespace=config.namespace, **kwargs)


def _session_memory(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    return SessionMemory(**kwargs)


def _require_embedder(
    config: StorageConfig, embed_client: Optional[EmbeddingClient]
) -> EmbeddingClient:
    if embed_client is None:
        raise ValueError(
            f"backend={config.backend!r} 是向量记忆，需要 embed_client="
            "（OpenAIEmbedding / SentenceTransformerEmbedding / HashEmbedding …）"
        )
    return embed_client


def _vector_memory(
    config: StorageConfig,
    *,
    store: VectorStore,
    embed_client: EmbeddingClient,
    **kwargs,
) -> Memory:
    return VectorMemory(
        embed_client,
        store,
        namespace=config.namespace,
        **kwargs,
    )


def _folder_vectors(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    from promptkit.memory.folder_store import FolderVectorStore

    embedder = _require_embedder(config, embed_client)
    store = FolderVectorStore(config.vector_path())
    return _vector_memory(config, store=store, embed_client=embedder, **kwargs)


def _chroma_vectors(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    from promptkit.memory.adapters.store import ChromaVectorStore

    embedder = _require_embedder(config, embed_client)
    store = ChromaVectorStore.at(
        config.path, config.collection_name(), url=config.url
    )
    return _vector_memory(config, store=store, embed_client=embedder, **kwargs)


def _qdrant_vectors(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    from promptkit.memory.adapters.store import QdrantVectorStore

    embedder = _require_embedder(config, embed_client)
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError(
            "backend='qdrant' 需要 qdrant-client：pip install qdrant-client"
        ) from None
    client = QdrantClient(
        url=config.url,
        api_key=cast(Optional[str], config.options.get("api_key")) or None,
    )
    store = QdrantVectorStore(client, config.collection_name())
    return _vector_memory(config, store=store, embed_client=embedder, **kwargs)


def _layered_memory(
    config: StorageConfig, *, embed_client: Optional[EmbeddingClient] = None, **kwargs
) -> Memory:
    '''KV + 向量：KV 存结构化值，向量存能被召回的内容。'''
    kv = KVMemory(config.kv_path(), namespace=config.namespace)

    vector_backend = config.options.get("vector_backend") or (
        "jsonl" if embed_client is not None else None
    )
    if vector_backend is None:
        return LayeredMemory(kv=kv, namespace=config.namespace, mirror=False)

    vector_config = replace(
        config,
        backend=str(vector_backend),
        path=cast(Optional[str], config.options.get("vector_path")),
        url=cast(Optional[str], config.options.get("vector_url")),
    )
    vector = create_memory(vector_config, embed_client=embed_client, **kwargs)
    return LayeredMemory(
        kv=kv,
        vector=vector,
        namespace=config.namespace,
        mirror=bool(config.options.get("mirror", True)),
    )


register_memory_backend("sqlite", _kv_memory, aliases=("kv",))
register_memory_backend("jsonl", _folder_vectors, aliases=("folder", "vector"))
register_memory_backend("chroma", _chroma_vectors)
register_memory_backend("qdrant", _qdrant_vectors)
register_memory_backend("memory", _in_memory)
register_memory_backend("session", _session_memory)
register_memory_backend("layered", _layered_memory)
