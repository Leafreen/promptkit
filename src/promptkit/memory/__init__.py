# memory/__init__.py
'''记忆系统：长期 / 短期记忆 + 读、写、召回三件套。

三层记忆各归各的：

| 层 | 生命周期 | 谁负责 |
| --- | --- | --- |
| 工作记忆 | 单次 `Workflow.run` | `Context`（**别**拿长期记忆替代它） |
| 短期记忆 | 单次会话 | `SessionMemory`（包 `Agent.history`，只读） |
| 长期记忆 | 跨会话 | `KVMemory`（结构化）/ `VectorMemory`（语义召回） |

四条边界（整个设计就是为了守住它们）：

1. **显式优于魔法**：没有「自动记住一切」。读什么、写什么、何时召回，
   都由工作流里的步骤写清楚（`ReadMemory` / `WriteMemory` / `RecallMemory`
   / `MemoryStep`）。
2. **接口薄**：`Memory` 只有 `read` / `write` / `recall`（外加可选的
   `forget`）。不塞更多。
3. **适配器隔离**：向量库 / embedding / rerank 都是接口，实现在
   `memory.adapters` 里，核心零依赖。
4. **可遗忘**：`forget(key)`、按时间过期、按新旧降权。

最短的用法：

    from promptkit.memory import InMemoryMemory
    from promptkit.core import ReadMemory, WriteMemory, Workflow

    mem = InMemoryMemory()
    wf = Workflow(
        WriteMemory(mem, "level", "中级"),
        ReadMemory(mem, "level", output_key="level"),
    )
    wf.run()

持久化 + 语义召回：

    from promptkit.memory import KVMemory, VectorMemory, TokenBudget
    from promptkit.memory.adapters import HashEmbedding, OpenAIEmbedding

    kv = KVMemory("memory.db", namespace="user_1")
    vec = VectorMemory(OpenAIEmbedding(model="text-embedding-3-small"),
                       namespace="user_1", budget=TokenBudget(1500))

存哪儿不想自己拼路径，就用「存储位置」这层（默认落在 OS 数据目录，
一条命令换位置）：

    from promptkit.memory import StorageConfig, create_memory

    memory = create_memory(StorageConfig(backend="sqlite", namespace="user_1"))
    memory = create_memory(StorageConfig(backend="jsonl", namespace="user_1"),
                           embed_client=HashEmbedding())
'''
from promptkit.memory.base import (
    InMemoryMemory,
    Memory,
    MemoryError,
    MemoryItem,
    empty,
    format_memories,
    item_text,
    replace_item_text,
)
from promptkit.memory.budget import TokenBudget, estimate_tokens
from promptkit.memory.config import WRITE_ON_VALUES, MemoryConfig, RecallConfig
from promptkit.memory.kv import KVMemory
from promptkit.memory.folder_store import FolderVectorStore
from promptkit.memory.layered import LayeredMemory, long_term_memory
from promptkit.memory.paths import (
    DATA_DIR_ENV,
    default_data_dir,
    default_kv_path,
    default_vector_dir,
    expand_path,
    memory_dir,
    safe_namespace_name,
)
from promptkit.memory.session import SessionMemory
from promptkit.memory.storage import (
    BACKENDS,
    StorageConfig,
    create_memory,
    register_memory_backend,
)
from promptkit.memory.tool import MEMORY_LOGGER, RecallMemoryTool
from promptkit.memory.vector import (
    EmbeddingClient,
    InMemoryVectorStore,
    Reranker,
    VectorMemory,
    VectorStore,
    chunk_text,
    cosine,
)

__all__ = [
    # 接口与数据
    "Memory",
    "MemoryError",
    "MemoryItem",
    "InMemoryMemory",
    "item_text",
    "replace_item_text",
    "format_memories",
    "empty",
    # 实现
    "KVMemory",
    "SessionMemory",
    "VectorMemory",
    "InMemoryVectorStore",
    "FolderVectorStore",
    "LayeredMemory",
    "long_term_memory",
    # 存哪儿（存储位置抽象）
    "StorageConfig",
    "create_memory",
    "register_memory_backend",
    "BACKENDS",
    "DATA_DIR_ENV",
    "default_data_dir",
    "default_kv_path",
    "default_vector_dir",
    "memory_dir",
    "safe_namespace_name",
    "expand_path",
    # 向量链路的三个接口
    "EmbeddingClient",
    "VectorStore",
    "Reranker",
    "chunk_text",
    "cosine",
    # 预算
    "TokenBudget",
    "estimate_tokens",
    # 配置
    "MemoryConfig",
    "RecallConfig",
    "WRITE_ON_VALUES",
    # 工具
    "RecallMemoryTool",
    "MEMORY_LOGGER",
]
