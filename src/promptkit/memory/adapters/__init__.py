# memory/adapters/__init__.py
'''`EmbeddingClient` / `VectorStore` / `Reranker` 的具体实现。

**这个包不会被 `promptkit.memory` 自动导入** —— 是故意的：

- 核心（`memory.base` / `kv` / `vector` / `budget`）零依赖，import 就完事
- 适配器可能拉进 `openai` / `chromadb` / `qdrant-client` / `sentence-transformers`，
  只有你真的 import 这个包时才碰它们
- 每个适配器的可选依赖还是**懒加载**的（构造时才 import，报错带安装提示）

    from promptkit.memory.adapters import HashEmbedding, ChromaVectorStore

    from promptkit.memory.adapters.store import QdrantVectorStore
'''
from promptkit.memory.adapters.embedding import (
    CallableEmbedding,
    HashEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)
from promptkit.memory.adapters.rerank import (
    CohereReranker,
    CrossEncoderReranker,
    LlmReranker,
    NoopReranker,
)
from promptkit.memory.adapters.store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
)

__all__ = [
    # embedding
    "CallableEmbedding",
    "HashEmbedding",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    # store
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",
    # rerank
    "NoopReranker",
    "LlmReranker",
    "CohereReranker",
    "CrossEncoderReranker",
]
