# memory/tool.py
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
from typing import Any, Dict, Iterable, Optional, Sequence

from promptkit.memory.base import Memory, item_text
from promptkit.memory.budget import TokenBudget
from promptkit.tools import Tool

__all__ = ["RecallMemoryTool", "MEMORY_REFERENCE", "MEMORY_LOGGER"]

#: 记忆层的 logger（`promptkit.memory`），和 `promptkit` 主 logger 是父子关系
MEMORY_LOGGER = "promptkit.memory"

DEFAULT_DESCRIPTION = (
    "根据查询在长期记忆里做语义召回，返回相关的过往记录。"
    "当你需要用户的偏好、以前聊过的结论、或者笔记里的内容时调用它；"
    "只是聊当前这几句时不要调用 —— 对话内容你已经看得到。"
)

DEFAULT_PARAMETERS: Dict[str, Any] = {
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
        parameters: Optional[Dict[str, Any]] = None,
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
            parameters=dict(DEFAULT_PARAMETERS if parameters is None else parameters),
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
    def run(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
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
            items: Sequence[Any] = self.memory.recall(text, top_k=capped)
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
    def _shape(self, item: Any) -> Dict[str, Any]:
        '''一条结果：截断正文、四舍五入分数、按需过滤 metadata。'''
        body = item_text(item)
        if len(body) > self.max_item_chars:
            body = body[: max(1, self.max_item_chars - 1)] + "…"

        shaped: Dict[str, Any] = {"text": body}
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


def _int_or(value: Any, fallback: int) -> int:
    '''模型传的 `top_k` 可能是字符串甚至胡说，别让它炸掉工具。'''
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def MEMORY_REFERENCE(  # noqa: N802 - 跟 DSL 命名（工厂，返回 Tool）
    source: Any,
    *,
    name: str = "recall_memory",
    description: Optional[str] = None,
    top_k: int = 3,
    max_top_k: int = 5,
    max_calls: int = 3,
    max_item_chars: int = 500,
    budget: Optional[TokenBudget] = None,
    namespace: Optional[str] = None,
    embed_client: Optional[Any] = None,
    **kwargs: Any,
) -> Tool:
    '''把「命名空间 / 路径 / Memory / 现成 Tool」统一成一个**只读**的回忆工具。

    一行把长期记忆接进聊天，跟 `KNOWLEDGE_REFERENCE` 对称：

        from promptkit.memory import MEMORY_REFERENCE
        from promptkit.default_tools import WEB_SEARCH, WRITE_FILE

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

    tool_kwargs: Dict[str, Any] = dict(
        name=name,
        description=description,
        top_k=top_k,
        max_top_k=max_top_k,
        max_calls=max_calls,
        max_item_chars=max_item_chars,
        budget=budget,
    )

    # 2) 现成的记忆对象（有 recall 就行）
    if isinstance(source, Memory) or callable(getattr(source, "recall", None)):
        return RecallMemoryTool(source, **tool_kwargs)

    # 3) 命名空间 或 路径 → 先建一份长期记忆，再包一层
    if isinstance(source, (str, os.PathLike)):
        from promptkit.memory.layered import long_term_memory

        if embed_client is None:
            from promptkit.memory.adapters import HashEmbedding

            embed_client = HashEmbedding(dimension=512)

        text = os.fspath(source)
        if os.sep in text or text.endswith((".db", ".sqlite", ".db3", ".jsonl")):
            # 路径：KV 一个文件，向量并排一个文件夹
            identity = namespace or (
                os.path.splitext(os.path.basename(text))[0] or "default"
            )
            memory = long_term_memory(
                identity,
                path=text,
                vector_path=text + ".vectors",
                embed_client=embed_client,
                **kwargs,
            )
        else:
            # 命名空间：默认数据目录
            memory = long_term_memory(
                namespace or text,
                embed_client=embed_client,
                **kwargs,
            )
        return RecallMemoryTool(memory, **tool_kwargs)

    raise TypeError(
        "MEMORY_REFERENCE 需要命名空间字符串、路径、Memory 或 Tool，"
        f"得到 {type(source).__name__}"
    )
