# memory/base.py
'''记忆的抽象：`Memory` 接口 + 数据单位 `MemoryItem` + 一个最朴素的实现。

**薄接口**：只有 `read` / `write` / `recall` 三件事。多一点都不塞。

- `read` / `write`：按键存取（KV 语义），适合结构化记忆（画像、偏好、进度）
- `recall`：语义召回，适合非结构化记忆（笔记、历史、文档）

纯 KV 的记忆可以不实现 `recall`（默认抛 `NotImplementedError`）；
反过来，只做向量的记忆可以不实现 `read`（`VectorMemory` 就是这样）。

分层（别混用）
--------------
| 层 | 生命周期 | 谁负责 |
| --- | --- | --- |
| 工作记忆 | 单次 `Workflow.run` | `Context`（已有，**不要**用长期记忆替代它） |
| 短期记忆 | 单次会话 | `SessionMemory`（包 `Agent.history`） |
| 长期记忆 | 跨会话 | `KVMemory` / `VectorMemory` |

**显式优于魔法**：这里没有任何「自动记住一切」的钩子。读什么、写什么、
什么时候召回，都由工作流里的步骤写清楚（`ReadMemory` / `WriteMemory` /
`RecallMemory` / `MemoryStep`）。
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "Memory",
    "MemoryError",
    "MemoryItem",
    "InMemoryMemory",
    "item_text",
    "replace_item_text",
    "format_memories",
    "empty",
]


class MemoryError(RuntimeError):
    '''记忆层出错（只读记忆被写、序列化失败、后端不可用……）。

    单独一个类型，是为了让调用方把「记忆后端的问题」和「工作流写错了」
    分开处理 —— 后者重试多少次都一样。
    '''


def item_text(item: Any) -> str:
    '''从「一条召回结果」里取正文。

    召回结果可能是 `MemoryItem`、`dict`（`recall()` 的公开格式），
    也可能直接是字符串（自己实现的记忆爱返回什么返回什么）。
    '''
    if isinstance(item, MemoryItem):
        return item.text
    if isinstance(item, dict):
        text = item.get("text", item.get("content", ""))
        return "" if text is None else str(text)
    return "" if item is None else str(item)


@dataclass
class MemoryItem:
    '''一条要被存储 / 召回的记录。

    `id` 是存储层的主键（`VectorMemory` 会按 chunk 拆成 `key#0`、`key#1`…）；
    `metadata` 里约定放 `key`（逻辑键，`forget(key)` 靠它）、`ts`（写入时间，
    过期 / 降权靠它）、`source` 之类，其它随便加。
    '''

    id: str
    text: str = ""
    vector: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: 召回时由存储层 / 重排器填上
    score: float = 0.0

    @property
    def key(self) -> str:
        '''逻辑键（`metadata["key"]`）。'''
        return str(self.metadata.get("key", ""))

    def to_dict(
        self, *, include_metadata: bool = True, include_id: bool = True
    ) -> Dict[str, Any]:
        '''转成 `recall()` 的公开格式（普通 dict，能直接 JSON 序列化）。'''
        data: Dict[str, Any] = {"text": self.text, "score": round(self.score, 4)}
        if include_id:
            data["id"] = self.id
        if include_metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def __repr__(self) -> str:
        preview = self.text[:24] + ("…" if len(self.text) > 24 else "")
        return f"MemoryItem(id={self.id!r}, score={self.score:.3f}, text={preview!r})"


class Memory(ABC):
    '''记忆的薄接口。

    自己实现一个记忆只要两个方法（`read` / `write`），想做语义召回再加
    `recall`，想支持删除再加 `forget`。

        class RedisMemory(Memory):
            def read(self, key, default=None): ...
            def write(self, key, value): ...
    '''

    #: 多用户 / 多主题隔离用的命名空间（实现里应该真的用上它）
    namespace: str = "default"

    # ---------------- 必须实现 ----------------
    @abstractmethod
    def read(self, key: str, default: Any = None) -> Any:
        '''按键读，没有就返回 `default`。'''

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        '''按键写（覆盖）。'''

    # ---------------- 可选实现 ----------------
    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''按语义召回，返回 `[{text, score, ...}, ...]`。

        默认抛 `NotImplementedError`：纯 KV 记忆没有「相似度」这回事。
        '''
        raise NotImplementedError(
            f"{type(self).__name__} 不支持语义召回（recall）；"
            "要召回请用 VectorMemory，或者自己实现 recall()"
        )

    def forget(self, key: str) -> bool:
        '''删掉一个键（含它的所有 chunk），返回是否真的删掉了什么。

        「可遗忘」是记忆系统的基本要求，但删除的语义各家不同，
        所以默认抛 `NotImplementedError` 而不是悄悄当成空实现。
        '''
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 forget()；"
            "KVMemory / VectorMemory 实现了，自定义记忆请自己补上"
        )

    # ---------------- 便利（基于上面几个） ----------------
    def has(self, key: str) -> bool:
        '''这个键存在吗（实现方式是「读出来看不看得到默认值哨兵」）。'''
        sentinel = object()
        return self.read(key, sentinel) is not sentinel

    def get(self, key: str, default: Any = None) -> Any:
        '''`read` 的别名，方便当 Mapping 用。'''
        return self.read(key, default)

    def __contains__(self, key: object) -> bool:
        return self.has(str(key))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} namespace={self.namespace!r}>"


class InMemoryMemory(Memory):
    '''一个 dict 记忆：进程内有效，进程退出就没了。

    用途很明确：**测试、原型、以及抄它的骨架去写自己的适配器**。
    它不做任何序列化 —— 存进去是对象，取出来还是同一个对象。
    '''

    def __init__(self, namespace: str = "default", initial: Optional[Dict[str, Any]] = None):
        self.namespace = str(namespace)
        self._data: Dict[str, Any] = dict(initial or {})

    def read(self, key: str, default: Any = None) -> Any:
        return self._data.get(str(key), default)

    def write(self, key: str, value: Any) -> None:
        self._data[str(key)] = value

    def forget(self, key: str) -> bool:
        return self._data.pop(str(key), _MISSING) is not _MISSING

    def keys(self) -> List[str]:
        return list(self._data)

    def items(self) -> Dict[str, Any]:
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


#: `forget` / `has` 用的哨兵（和 `None` 区分开：值真的可以是 None）
_MISSING: Any = object()


def format_memories(
    items: Sequence[Any],
    *,
    numbered: bool = True,
    with_score: bool = False,
    max_chars: int = 0,
) -> str:
    '''把召回结果拼成给人看的（和给模型看的）文本。

    召回结果是结构化的 `list[dict]`，直接塞进 `{{变量}}` 会变成一坨 repr；
    喂给模型之前基本都要过这一道。

        [1] 退款政策：7 天内无理由……
        [2] 退款需要订单号……

    `max_chars > 0` 时每条再截断一次（防止一条超长笔记吃掉整个上下文）。
    '''
    lines: List[str] = []
    for index, item in enumerate(items or (), start=1):
        text = item_text(item).strip()
        if not text:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max(1, max_chars - 1)] + "…"
        prefix = f"[{index}] " if numbered else ""
        suffix = ""
        if with_score:
            score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
            if score is not None:
                suffix = f"（相似度 {float(score):.2f}）"
        lines.append(f"{prefix}{text}{suffix}")
    return "\n".join(lines)


def replace_item_text(item: Any, text: str) -> Any:
    '''截断后「同类型」地换掉正文（`TokenBudget` 用）。'''
    if isinstance(item, MemoryItem):
        return replace(item, text=text)
    if isinstance(item, dict):
        return {**item, "text": text}
    return text


def empty(value: Any) -> bool:
    '''值算不算「空」（写记忆前用它跳过空内容）。'''
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
