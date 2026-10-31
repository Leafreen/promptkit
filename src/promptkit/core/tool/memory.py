# core/tool/memory.py
'''`RecallMemoryTool`：**只给读**的记忆工具，给 Agent 自主循环用。

固定上下文用 `MemoryStep` / `RecallMemory` 就够了；只有「模型自己决定要不要
回忆」的场景才需要工具。所以这里刻意**不给写**：写记忆必须由工作流显式做
（`WriteMemory` / `MemoryStep`），不能让模型随手往长期记忆里灌东西。

四道闸门（缺一个都可能把上下文或钱包撑爆）：

1. **调用次数**：一轮最多 `max_calls` 次，超了直接让模型用已有信息回答
2. **top_k 上限**：模型要 50 条也只给 `max_top_k` 条
3. **单条截断**：每条不超过 `max_item_chars` 字
4. **元数据默认不给**：`include_metadata=False`；要给就配 `redact=` 名单

再加一条**审计日志**（`logging.getLogger("promptkit.memory")`），
出问题能回查「它当时到底召回了什么」。
'''
from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, Iterable, Optional, Sequence, Union, cast

from promptkit.memory.base import Memory, item_text
from promptkit.memory.budget import TokenBudget
from promptkit.memory.vector import EmbeddingClient
from promptkit.core.tool.base import Tool, ToolSchema

__all__ = [
    "RecallMemoryTool",
    "WriteMemoryTool",
    "MEMORY_REFERENCE",
    "WRITE_MEMORY",
    "MEMORY_LOGGER",
]

#: 记忆层的 logger（`promptkit.memory`），和 `promptkit` 主 logger 是父子关系
MEMORY_LOGGER = "promptkit.memory"

DEFAULT_DESCRIPTION = (
    "根据查询在长期记忆里做语义召回，返回相关的过往记录。"
    "当你需要用户的偏好、以前聊过的结论、或者笔记里的内容时调用它；"
    "只是聊当前这几句时不要调用 —— 对话内容你已经看得到。"
)

DEFAULT_PARAMETERS: ToolSchema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要在记忆里找什么，用一句话描述（不要照抄用户原话也行）",
        },
        "top_k": {
            "type": "integer",
            "description": "要几条，默认 3；服务端会再封顶",
        },
    },
    "required": ["query"],
}


class RecallMemoryTool(Tool):
    '''把 `memory.recall()` 包成一个 function-calling 工具。

        tool = RecallMemoryTool(vector_memory)
        agent.ask("接着上次的进度讲", tools=[tool])

    限流是**按实例**算的：`reset()` 手动清零；`ChatWorkflow` 每轮会重置
    （挂在 `Tools(...)` 上的话，需要自己每轮调一次 `reset()`）。
    '''

    def __init__(
        self,
        memory: Memory,
        *,
        top_k: int = 3,
        max_top_k: int = 5,
        max_calls: int = 3,
        max_item_chars: int = 500,
        include_metadata: bool = False,
        redact: Iterable[str] = (),
        budget: Optional[TokenBudget] = None,
        name: str = "recall_memory",
        description: Optional[str] = None,
        parameters: Optional[ToolSchema] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if memory is None:
            raise ValueError("RecallMemoryTool 需要 memory（一个 Memory 实例）")
        if int(max_top_k) < 1:
            raise ValueError(f"max_top_k 必须是正整数，得到 {max_top_k!r}")
        if int(max_calls) < 1:
            raise ValueError(f"max_calls 必须是正整数，得到 {max_calls!r}")
        if int(max_item_chars) < 1:
            raise ValueError(f"max_item_chars 必须是正整数，得到 {max_item_chars!r}")

        self.memory = memory
        self.top_k = int(top_k)
        self.max_top_k = int(max_top_k)
        self.max_calls = int(max_calls)
        self.max_item_chars = int(max_item_chars)
        self.include_metadata = bool(include_metadata)
        self.redact = {str(field) for field in redact}
        self.budget = budget
        self.logger = logger or logging.getLogger(MEMORY_LOGGER)
        self._calls = 0

        super().__init__(
            name=name,
            description=DEFAULT_DESCRIPTION if description is None else description,
            parameters=DEFAULT_PARAMETERS if parameters is None else parameters,
            func=self.run,
        )

    # ---------------- 状态 ----------------
    @property
    def calls(self) -> int:
        '''本次「一轮」里已经被调用了几次。'''
        return self._calls

    def reset(self) -> "RecallMemoryTool":
        '''清零调用计数（换一轮对话时调）。'''
        self._calls = 0
        return self

    # ---------------- 工具本体 ----------------
    def run(self, query: str, top_k: Optional[int] = None) -> Dict[str, object]:
        '''工具的实际实现；返回值由 `Tool.call()` 转成 JSON 文本交给模型。'''
        text = "" if query is None else str(query).strip()
        if not text:
            return {"error": "query 不能为空", "results": []}

        if self._calls >= self.max_calls:
            return {
                "error": (
                    f"本次对话已经调用过 {self.max_calls} 次记忆召回，"
                    "请直接用已经拿到的信息回答"
                ),
                "results": [],
            }
        self._calls += 1

        requested = self.top_k if top_k is None else _int_or(top_k, self.top_k)
        capped = max(1, min(requested, self.max_top_k))
        try:
            items: Sequence[object] = self.memory.recall(text, top_k=capped)
        except NotImplementedError as exc:  # 纯 KV 记忆
            return {"error": str(exc), "results": []}

        if self.budget is not None:
            items = self.budget.trim(items)

        results = [self._shape(item) for item in items]
        self.logger.info(
            "recall_memory query=%r top_k=%d hits=%d calls=%d",
            text[:80],
            capped,
            len(results),
            self._calls,
        )
        return {"query": text, "results": results, "count": len(results)}

    # ---------------- 结果整形 ----------------
    def _shape(self, item: object) -> Dict[str, object]:
        '''一条结果：截断正文、四舍五入分数、按需过滤 metadata。'''
        body = item_text(item)
        if len(body) > self.max_item_chars:
            body = body[: max(1, self.max_item_chars - 1)] + "…"

        shaped: Dict[str, object] = {"text": body}
        score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
        if isinstance(score, (int, float)):
            shaped["score"] = round(float(score), 4)

        if self.include_metadata:
            raw = (
                item.get("metadata")
                if isinstance(item, dict)
                else getattr(item, "metadata", None)
            )
            metadata = dict(raw or {})
            shaped["metadata"] = {
                key: value
                for key, value in metadata.items()
                if str(key) not in self.redact
            }
        return shaped

    def __repr__(self) -> str:
        return (
            f"<RecallMemoryTool name={self.name!r} memory={type(self.memory).__name__} "
            f"top_k={self.top_k}/{self.max_top_k} calls={self._calls}/{self.max_calls}>"
        )


class WriteMemoryTool(Tool):
    '''把 `memory.write()` 包成一个 function-calling 工具，让模型**能写**长期记忆。

    和 `RecallMemoryTool`（只读）对称，但**写是显式授权**的：只有把这个工具
    加进 `Tools(...)`，模型才有写权限。默认不随 `MEMORY_REFERENCE` 赠送。

    模型传参：

        write_memory(content="我喜欢用生活例子理解数学符号")     # 自由笔记，自动建键
        write_memory(content="中级", key="level")                # 结构化键值

    省略 `key` 时自动用 `note:{uuid}` 建键 —— 这种笔记会被 `LayeredMemory`
    同时写进向量索引（`mirror=True`），之后能用 `recall_memory` 按意思找到。

    两道闸门（对照 `RecallMemoryTool` 的四道，写更危险所以更严）：

    1. **调用次数**：一轮最多 `max_calls` 次，超了让模型直接回答
    2. **单条截断**：每条不超过 `max_chars` 字，防止一次灌一篇论文

    同样有审计日志，出问题能回查「它当时往记忆里塞了什么」。
    '''

    DEFAULT_DESCRIPTION = (
        "把一条信息写入长期记忆，之后可以被 recall_memory 召回。"
        "只在用户明确要求记住，或出现值得长期保留的偏好、进度、结论时调用；"
        "不要记录密码、密钥等敏感信息，也不要记录普通闲聊。"
    )

    DEFAULT_PARAMETERS: ToolSchema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要记住的内容（一句话或一段笔记）",
            },
            "key": {
                "type": "string",
                "description": (
                    "可选：指定记忆的键名（如 'level'、'preferred_style'）。"
                    "不填则自动生成一个笔记键，之后靠语义召回而不是按键读取。"
                ),
            },
        },
        "required": ["content"],
    }

    def __init__(
        self,
        memory: Memory,
        *,
        name: str = "write_memory",
        description: Optional[str] = None,
        parameters: Optional[ToolSchema] = None,
        max_calls: int = 5,
        max_chars: int = 2000,
        key_prefix: str = "note",
        logger: Optional[logging.Logger] = None,
    ):
        if memory is None:
            raise ValueError("WriteMemoryTool 需要 memory（一个 Memory 实例）")
        if not callable(getattr(memory, "write", None)):
            raise TypeError(
                "WriteMemoryTool 的 memory 必须实现 write()，"
                f"得到 {type(memory).__name__}"
            )
        if int(max_calls) < 1:
            raise ValueError(f"max_calls 必须是正整数，得到 {max_calls!r}")
        if int(max_chars) < 1:
            raise ValueError(f"max_chars 必须是正整数，得到 {max_chars!r}")

        self.memory = memory
        self.max_calls = int(max_calls)
        self.max_chars = int(max_chars)
        self.key_prefix = str(key_prefix)
        self.logger = logger or logging.getLogger(MEMORY_LOGGER)
        self._calls = 0

        super().__init__(
            name=name,
            description=self.DEFAULT_DESCRIPTION if description is None else description,
            parameters=self.DEFAULT_PARAMETERS if parameters is None else parameters,
            func=self.run,
        )

    # ---------------- 状态 ----------------
    @property
    def calls(self) -> int:
        '''本次「一轮」里已经被调用了几次。'''
        return self._calls

    def reset(self) -> "WriteMemoryTool":
        '''清零调用计数（换一轮对话时调）。'''
        self._calls = 0
        return self

    # ---------------- 工具本体 ----------------
    def run(self, content: str, key: Optional[str] = None) -> Dict[str, object]:
        '''工具的实际实现；返回值由 `Tool.call()` 转成 JSON 文本交给模型。'''
        if self._calls >= self.max_calls:
            return {
                "ok": False,
                "error": (
                    f"本次对话已经写入过 {self.max_calls} 次记忆，"
                    "请直接用对话回答，不要继续写记忆"
                ),
            }

        text = "" if content is None else str(content).strip()
        if not text:
            return {"ok": False, "error": "content 不能为空"}

        if len(text) > self.max_chars:
            text = text[: max(1, self.max_chars - 1)] + "…"

        store_key = ""
        if key is not None:
            store_key = str(key).strip()
        if not store_key:
            store_key = f"{self.key_prefix}:{uuid.uuid4().hex}"

        self._calls += 1
        try:
            self.memory.write(store_key, text)
        except Exception as exc:  # noqa: BLE001 - 记忆后端出错不该把整轮对话炸掉
            self._calls -= 1
            self.logger.exception("write_memory 失败 key=%s", store_key)
            return {"ok": False, "error": f"写入失败：{exc}"}

        self.logger.info(
            "write_memory key=%s chars=%d calls=%d",
            store_key,
            len(text),
            self._calls,
        )
        return {"ok": True, "key": store_key, "chars": len(text)}

    def __repr__(self) -> str:
        return (
            f"<WriteMemoryTool name={self.name!r} "
            f"memory={type(self.memory).__name__} "
            f"calls={self._calls}/{self.max_calls} "
            f"max_chars={self.max_chars}>"
        )


def _int_or(value: object, fallback: int) -> int:
    '''模型传的 `top_k` 可能是字符串甚至胡说，别让它炸掉工具。'''
    try:
        return int(cast(Union[str, int, float], value))
    except (TypeError, ValueError):
        return fallback


def MEMORY_REFERENCE(  # noqa: N802 - 跟 DSL 命名（工厂，返回 Tool）
    source: object,
    *,
    name: str = "recall_memory",
    description: Optional[str] = None,
    top_k: int = 3,
    max_top_k: int = 5,
    max_calls: int = 3,
    max_item_chars: int = 500,
    budget: Optional[TokenBudget] = None,
    namespace: Optional[str] = None,
    embed_client: Optional[EmbeddingClient] = None,
    **kwargs,
) -> Tool:
    '''把「命名空间 / 路径 / Memory / 现成 Tool」统一成一个**只读**的回忆工具。

    一行把长期记忆接进聊天，跟 `KNOWLEDGE_REFERENCE` 对称：

        from promptkit.memory import MEMORY_REFERENCE
        from promptkit.core.tool.builtin import WEB_SEARCH, WRITE_FILE

        Tools(WEB_SEARCH, MEMORY_REFERENCE("learner_001"), WRITE_FILE)

    - 传命名空间（`"learner_001"`）→ 用默认数据目录建 KV + 向量记忆
    - 传路径（含分隔符、或以 `.db`/`.sqlite`/`.jsonl` 结尾）→ KV 落在这个文件，
      向量落在 `<路径>.vectors/`
    - 传 `Memory`（有 `recall`）→ 直接 `RecallMemoryTool` 包一层
    - 传 `Tool` → 原样返回

    默认用 `HashEmbedding`（本地词法向量，免密钥、不联网）；要真语义就传
    `embed_client=OpenAIEmbedding(...)`。刻意**只读**：写记忆仍走显式的
    `WriteMemory` / `MemoryStep`，不能让模型随手往长期记忆里灌。
    '''
    # 1) 已经是工具，原样返回
    if isinstance(source, Tool):
        return source

    def wrap(memory: Memory) -> Tool:
        return RecallMemoryTool(
            memory,
            name=name,
            description=description,
            top_k=top_k,
            max_top_k=max_top_k,
            max_calls=max_calls,
            max_item_chars=max_item_chars,
            budget=budget,
        )

    # 2) 现成的记忆对象（有 recall 就行）
    if isinstance(source, Memory):
        return wrap(source)
    if callable(getattr(source, "recall", None)):
        return wrap(cast(Memory, source))

    # 3) 命名空间 或 路径 → 先建一份长期记忆，再包一层
    if isinstance(source, (str, os.PathLike)):
        memory = _memory_from_source(
            source, namespace=namespace, embed_client=embed_client, **kwargs
        )
        return wrap(memory)

    raise TypeError(
        "MEMORY_REFERENCE 需要命名空间字符串、路径、Memory 或 Tool，"
        f"得到 {type(source).__name__}"
    )


def _memory_from_source(
    source: object,
    *,
    namespace: Optional[str] = None,
    embed_client: Optional[EmbeddingClient] = None,
    **kwargs,
) -> Memory:
    '''把「命名空间 / 路径」编译成一个 `LayeredMemory`（KV + 向量）。

    `MEMORY_REFERENCE` 和 `WRITE_MEMORY` 共用这段，保证两个工具拿到的
    记忆对象**配置一致**（同一个路径 → 同一种 KV/向量布局）。规则本身
    放在 `memory.as_memory`，`Compact(...).To(...)` 也走同一套。
    '''
    from promptkit.memory.layered import as_memory

    return as_memory(
        source, namespace=namespace, embed_client=embed_client, **kwargs
    )


def WRITE_MEMORY(  # noqa: N802 - 跟 DSL 命名（工厂，返回 Tool）
    source: object,
    *,
    name: str = "write_memory",
    description: Optional[str] = None,
    max_calls: int = 5,
    max_chars: int = 2000,
    key_prefix: str = "note",
    namespace: Optional[str] = None,
    embed_client: Optional[EmbeddingClient] = None,
    **kwargs,
) -> Tool:
    '''把「命名空间 / 路径 / Memory / 现成 Tool」统一成一个**可写**的记忆工具。

    一行把「写记忆」接进聊天，跟 `MEMORY_REFERENCE`（只读）对称：

        from promptkit.memory import MEMORY_REFERENCE, WRITE_MEMORY
        from promptkit.core.tool.builtin import WEB_SEARCH, WRITE_FILE

        Tools(
            WEB_SEARCH,
            MEMORY_REFERENCE("learner_001"),   # 读：按意思召回
            WRITE_MEMORY("learner_001"),       # 写：模型自己决定记什么
            WRITE_FILE,
        )

    - 传命名空间 / 路径 → 用默认数据目录建 KV + 向量记忆（`mirror=True`，
      写入会同时进向量索引，之后能被 `recall_memory` 召回）
    - 传 `Memory`（有 `write`）→ 直接 `WriteMemoryTool` 包一层
    - 传 `Tool` → 原样返回

    **读和写要共用同一个 memory 对象**：如果传路径，`MEMORY_REFERENCE("x")`
    和 `WRITE_MEMORY("x")` 会各自建一份记忆实例；KV（SQLite）在磁盘上能互通，
    但向量索引是进程内的，写进去的笔记要等 `memory.close()` 落盘后另一个实例
    才能召回。要「写了立刻能召回」，请显式建一份记忆再传给两个工厂：

        memory = long_term_memory("learner_001")
        recall_tool = MEMORY_REFERENCE(memory)
        write_tool = WRITE_MEMORY(memory)

    模型调用：

        write_memory(content="我喜欢用生活例子理解数学符号")   # 自由笔记
        write_memory(content="中级", key="level")              # 结构化键值
    '''
    # 1) 已经是工具，原样返回
    if isinstance(source, Tool):
        return source

    def wrap(memory: Memory) -> Tool:
        return WriteMemoryTool(
            memory,
            name=name,
            description=description,
            max_calls=max_calls,
            max_chars=max_chars,
            key_prefix=key_prefix,
        )

    # 2) 现成的记忆对象（有 write 就行）
    if isinstance(source, Memory):
        return wrap(source)
    if callable(getattr(source, "write", None)):
        return wrap(cast(Memory, source))

    # 3) 命名空间 或 路径 → 先建一份长期记忆，再包一层
    if isinstance(source, (str, os.PathLike)):
        memory = _memory_from_source(
            source, namespace=namespace, embed_client=embed_client, **kwargs
        )
        return wrap(memory)

    raise TypeError(
        "WRITE_MEMORY 需要命名空间字符串、路径、Memory 或 Tool，"
        f"得到 {type(source).__name__}"
    )
