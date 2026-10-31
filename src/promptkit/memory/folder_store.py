# memory/folder_store.py
'''一个文件夹 = 一份向量记忆（零依赖的持久化 `VectorStore`）。

目录长这样：

    <path>/
    └── index.jsonl        # 一行一条：id / text / vector / metadata

为什么是 JSONL 而不是 `index.faiss` + `metadata.json` 两个文件：
**索引和原文必须同生共死**。两个文件就得自己保证「索引写成功、元数据没写成功」
这种半截状态，而单文件追加写只有「写了」和「没写」两种结果。
（要 FAISS / Chroma / Qdrant 就换 `store=`，接口是同一套四个方法。）

写入策略：**追加 + 定期压实**。

- `upsert` / `delete` 往文件尾追加一条操作，内存里的 dict 同步更新 —— O(1)，
  不会因为「写一条就重写整个文件」而变成 O(n²)
- 操作数超过 `compact_every`（或 `close()` / `flush()` 时）重写整份快照，
  文件不会无限长
- 读到坏行（手动编辑过、断电写了一半）**跳过并继续**：一份记忆不该因为
  一行脏数据整份打不开

这不是给百万级向量准备的（搜索是线性扫描，所有向量都在内存里）。
几千到几万条完全够，而「指定一个文件夹就能持久化」是它最大的价值。
'''
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from promptkit.memory.base import MemoryError, MemoryItem
from promptkit.memory.vector import VectorStore, cosine

__all__ = ["FolderVectorStore"]

#: 文件夹里的文件名
INDEX_FILE = "index.jsonl"


def _replace_with_retry(source: str, target: str, attempts: int = 5) -> None:
    '''`os.replace`，Windows 上失败就重试几次。

    刚写完关掉的临时文件经常正被杀毒 / 索引服务扫着，那一刻改不了名
    （`PermissionError` / WinError 5）。等几十毫秒通常就好了 ——
    `session.py` 里也有一份同样的重试，原因一模一样。
    '''
    delay = 0.02
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2


class FolderVectorStore(VectorStore):
    '''把向量记忆存进一个文件夹。

        store = FolderVectorStore("~/.promptkit/memory/vector/user_001")
        store = FolderVectorStore(path, compact_every=200)   # 更勤快地压实

    也可以不那么直接：`VectorMemory(embed, path=...)` 内部用的就是它。
    '''

    def __init__(self, path: str, *, compact_every: Optional[int] = None):
        if not path:
            raise ValueError("FolderVectorStore 需要一个文件夹路径")
        from promptkit.memory.paths import expand_path

        self.path = expand_path(path)
        self.index_path = os.path.join(self.path, INDEX_FILE)
        self.compact_every = int(compact_every or 500)
        if self.compact_every < 1:
            raise ValueError(
                f"compact_every 必须是正整数，得到 {compact_every!r}"
            )
        self._rows: Dict[Tuple[str, str], MemoryItem] = {}
        self._ops = 0
        os.makedirs(self.path, exist_ok=True)
        self._load()

    # ---------------- 加载 / 追加 ----------------
    def _load(self) -> None:
        '''回放 index.jsonl（快照 + 追加日志是同一种格式）。'''
        try:
            handle = open(self.index_path, encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            raise MemoryError(
                f"打不开向量记忆 {self.index_path!r}：{exc}"
            ) from None

        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 脏行跳过，不让它毒死整份记忆
                if not isinstance(row, dict):
                    continue
                namespace = str(row.get("namespace", "default"))
                if row.get("op") == "del":
                    for item_id in row.get("ids") or ():
                        self._rows.pop((namespace, str(item_id)), None)
                    continue
                try:
                    self._rows[(namespace, str(row["id"]))] = MemoryItem(
                        id=str(row["id"]),
                        text=str(row.get("text") or ""),
                        vector=row.get("vector") or None,
                        metadata=dict(row.get("metadata") or {}),
                    )
                except (KeyError, TypeError):
                    continue

    def _append(self, payload: Dict[str, object]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            with open(self.index_path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
        except OSError as exc:
            raise MemoryError(
                f"写不进向量记忆 {self.index_path!r}：{exc}"
            ) from None
        self._ops += 1
        if self._ops >= self.compact_every:
            self.flush()

    # ---------------- VectorStore 接口 ----------------
    def upsert(self, namespace: str, item: MemoryItem) -> None:
        namespace = str(namespace)
        stored = replace(item, metadata=dict(item.metadata))
        self._rows[(namespace, stored.id)] = stored
        self._append(
            {
                "op": "put",
                "namespace": namespace,
                "id": stored.id,
                "text": stored.text,
                "vector": list(stored.vector) if stored.vector is not None else None,
                "metadata": dict(stored.metadata),
            }
        )

    def search(
        self, namespace: str, vector: Sequence[float], top_k: int = 5
    ) -> List[MemoryItem]:
        namespace = str(namespace)
        scored: List[MemoryItem] = []
        for (stored_ns, _), item in self._rows.items():
            if stored_ns != namespace:
                continue
            scored.append(replace(item, score=cosine(vector, item.vector or ())))
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
        doomed = [
            item_id
            for (stored_ns, item_id), item in self._rows.items()
            if stored_ns == namespace
            and (ids is None or item_id in {str(value) for value in ids})
            and (key is None or item.key == str(key))
        ]
        for item_id in doomed:
            self._rows.pop((namespace, item_id), None)
        if doomed:
            self._append({"op": "del", "namespace": namespace, "ids": doomed})
        return len(doomed)

    def count(self, namespace: str) -> int:
        namespace = str(namespace)
        return sum(1 for stored_ns, _ in self._rows if stored_ns == namespace)

    def clear(self, namespace: str) -> None:
        self.delete(str(namespace))

    # ---------------- 落盘 / 维护 ----------------
    def flush(self) -> None:
        '''把整份记忆重写成快照（原子替换：先写 .tmp 再改名）。'''
        os.makedirs(self.path, exist_ok=True)
        temporary = self.index_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
                for (namespace, item_id), item in self._rows.items():
                    handle.write(
                        json.dumps(
                            {
                                "op": "put",
                                "namespace": namespace,
                                "id": item_id,
                                "text": item.text,
                                "vector": (
                                    list(item.vector)
                                    if item.vector is not None
                                    else None
                                ),
                                "metadata": dict(item.metadata),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            _replace_with_retry(temporary, self.index_path)
        except OSError as exc:
            raise MemoryError(
                f"写不进向量记忆 {self.index_path!r}：{exc}"
            ) from None
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        self._ops = 0

    def close(self) -> None:
        '''压实并落盘（可以重复调）。'''
        self.flush()

    def __enter__(self) -> "FolderVectorStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._rows)

    def size_bytes(self) -> int:
        '''索引文件多大（运维查「记忆是不是长胖了」用）。'''
        try:
            return os.path.getsize(self.index_path)
        except OSError:
            return 0

    def __repr__(self) -> str:
        return (
            f"<FolderVectorStore {self.path!r} rows={len(self._rows)}>"
        )
