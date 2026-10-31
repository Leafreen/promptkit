# memory/adapters/embedding.py
'''`EmbeddingClient` 的现成适配器。

| 适配器 | 依赖 | 用在哪 |
| --- | --- | --- |
| `HashEmbedding` | 无 | 跑通链路 / 测试 / 小规模词法召回 |
| `CallableEmbedding` | 无 | 包一层你自己的函数（Ollama、公司网关…） |
| `OpenAIEmbedding` | `openai` | OpenAI、以及所有兼容 `/v1/embeddings` 的端点 |
| `SentenceTransformerEmbedding` | `sentence-transformers` | 本地模型，含 BGE / m3e / gte 等 |

可选依赖都是**懒加载**的：没装 `sentence-transformers` 也能 `import`
这个模块，只有真的构造对应适配器时才报错，并告诉你怎么装。
'''
from __future__ import annotations

import math
import zlib
from typing import Callable, List, Optional, Sequence

from promptkit.memory.vector import EmbeddingClient

__all__ = [
    "CallableEmbedding",
    "HashEmbedding",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
]

#: 常见 OpenAI 向量模型的维度（拿来填 `dimension`，省一次试错）
_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class CallableEmbedding(EmbeddingClient):
    '''把任意 `text -> list[float]` 的函数包成 `EmbeddingClient`。

        embedder = CallableEmbedding(
            lambda text: requests.post("http://localhost:11434/api/embeddings",
                                       json={"model": "nomic-embed-text", "prompt": text}
                                       ).json()["embedding"]
        )
    '''

    def __init__(
        self, func: Callable[[str], Sequence[float]], dimension: Optional[int] = None
    ):
        if not callable(func):
            raise TypeError(f"CallableEmbedding 需要可调用对象，得到 {func!r}")
        self.func = func
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        return [float(value) for value in self.func(text)]


class HashEmbedding(EmbeddingClient):
    '''零依赖的「词法级」向量：字符 n-gram 哈希到固定维度。

    **它不是语义向量**：同义词（「退款」和「退钱」）不会靠近，只有字面相近的
    文本才会。但足够把整条链路（切分 / 存储 / 召回 / 预算）跑通、够写测试、
    也够几千条笔记的粗略召回。要真语义就换 `OpenAIEmbedding` 或
    `SentenceTransformerEmbedding` —— 接口一样，换一行的事。

    稳定性：用 `zlib.crc32` 而不是 `hash()` —— 后者每个进程的盐不一样，
    存进库里的向量下次就失效了。
    '''

    def __init__(self, dimension: int = 256, ngram: int = 2):
        if not isinstance(dimension, int) or dimension < 8:
            raise ValueError(f"dimension 必须是 >= 8 的整数，得到 {dimension!r}")
        if not isinstance(ngram, int) or ngram < 1:
            raise ValueError(f"ngram 必须是正整数，得到 {ngram!r}")
        self.dimension = dimension
        self.ngram = ngram

    def embed(self, text: str) -> List[float]:
        dimension = self.dimension
        if dimension is None:  # 构造时已校验，理论到不了
            raise ValueError("dimension 必须 >= 8 的整数")
        vector = [0.0] * dimension
        normalized = "".join(
            char for char in str(text or "").lower() if not char.isspace()
        )
        if not normalized:
            return vector

        for size in range(1, self.ngram + 1):
            for start in range(len(normalized) - size + 1):
                gram = normalized[start : start + size]
                index = zlib.crc32(gram.encode("utf-8")) % dimension
                vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbedding(EmbeddingClient):
    '''OpenAI（以及任何兼容 `/v1/embeddings` 的端点）的向量。

        OpenAIEmbedding(model="text-embedding-3-small")      # 读 OPENAI_API_KEY
        OpenAIEmbedding(model="bge-m3", base_url="http://localhost:8080/v1",
                        api_key="none")

    `client=` 可以塞一个已经建好的 `openai.OpenAI`（想自己配代理 / 重试时用）。
    '''

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        client: object = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 64,
    ):
        self.model = str(model)
        self.dimensions = dimensions
        self.batch_size = max(1, int(batch_size))
        self.dimension = dimensions or _OPENAI_DIMENSIONS.get(self.model)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:  # pragma: no cover - openai 是必需依赖
                raise ImportError(
                    "OpenAIEmbedding 需要 openai 包：pip install openai"
                ) from None
            options: dict = {}
            if api_key is not None:
                options["api_key"] = api_key
            if base_url is not None:
                options["base_url"] = base_url
            if timeout is not None:
                options["timeout"] = timeout
            client = OpenAI(**options)
        self.client = client

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        pending = ["" if text is None else str(text) for text in texts]
        if not pending:
            return []

        vectors: List[List[float]] = []
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            options: dict = {"model": self.model, "input": batch}
            if self.dimensions:
                options["dimensions"] = self.dimensions
            response = getattr(self.client, "embeddings").create(**options)
            rows = sorted(
                response.data, key=lambda row: getattr(row, "index", 0)
            )
            vectors.extend(
                [float(value) for value in row.embedding] for row in rows
            )
        return vectors


class SentenceTransformerEmbedding(EmbeddingClient):
    '''本地向量模型（sentence-transformers）。

        SentenceTransformerEmbedding("BAAI/bge-small-zh-v1.5")
        SentenceTransformerEmbedding(model=already_loaded_model)

    BGE / m3e / gte 这些中文模型都走这条路。`model=` 也可以直接给一个
    已经加载好的对象（只要它有 `.encode`），方便复用 / 测试。
    '''

    def __init__(
        self,
        model: object = None,
        *,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "SentenceTransformerEmbedding 需要 sentence-transformers：\n"
                    "  pip install sentence-transformers\n"
                    "（也可以把已经加载好的模型用 model= 传进来，"
                    "或者干脆用 HashEmbedding 先跑通链路）"
                ) from None
            model = SentenceTransformer(model_name, device=device)

        encode = getattr(model, "encode", None)
        if not callable(encode):
            raise TypeError(
                f"SentenceTransformerEmbedding 需要带 .encode() 的模型，得到 {model!r}"
            )
        self.model = model
        self.normalize = bool(normalize)
        dimension = getattr(model, "get_sentence_embedding_dimension", None)
        self.dimension = dimension() if callable(dimension) else None

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        prepared = ["" if text is None else str(text) for text in texts]
        if not prepared:
            return []
        encoded = getattr(self.model, "encode")(
            prepared, normalize_embeddings=self.normalize
        )
        return [[float(value) for value in row] for row in encoded]
