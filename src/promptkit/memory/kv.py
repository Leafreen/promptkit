# memory/kv.py
'''KV 记忆：SQLite，零额外依赖，跨进程 / 跨会话都还在。

存的是 `namespace + key -> JSON 值`。适合**结构化**的长期记忆：

    画像        {"level": "中级", "preferred_style": "例子优先"}
    进度        {"topics_covered": ["抽样", "问卷设计"]}
    配置        任意 JSON

不适合放「一篇长笔记」那种要语义召回的东西 —— 那是 `VectorMemory` 的活。
（塞进来也不会坏，只是只能按键取，召回不了。）

为什么是 SQLite：标准库自带、单文件、支持并发读、Windows 上不用装服务。
`path=":memory:"` 就是纯内存（测试常用）。
'''
from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from promptkit.memory.base import Memory, MemoryError
from promptkit.memory.paths import default_kv_path, expand_path

__all__ = ["KVMemory"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
)
"""


def _json_default(value: Any) -> Any:
    '''JSON 编码不认识的类型时退一步（pydantic / dataclass / set）。'''
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    raise TypeError(f"{type(value).__name__} 不是 JSON 可序列化的类型")


class KVMemory(Memory):
    '''按键存取的长期记忆（SQLite 后端）。

        kv = KVMemory(namespace="learner_001")       # 落到默认数据目录
        kv = KVMemory("~/data/tutor.db", namespace="learner_001")   # 自己指定
        kv.write("level", "中级")
        kv.read("level")                 # '中级'
        kv.read("missing", "默认值")      # '默认值'
        kv.forget("level")                # True
        kv.backup("backup/20261020.db")   # 热备份
        kv.close()

    `path` 省略时落在 `<data_dir>/memory/kv/<namespace>.db`
    （Windows `%APPDATA%`，macOS `Library/Application Support`，
    Linux `$XDG_DATA_HOME`；可用 `PROMPTKIT_DATA_DIR` 整体改掉）。
    父目录会自动创建。

    一个实例持有一个连接（`check_same_thread=False`）；工作流是顺序执行的，
    不需要连接池。用完记得 `close()`，或者用 `with KVMemory(...) as kv:`
    （短脚本不关也行，进程退出会释放）。
    '''

    def __init__(
        self,
        path: Optional[str] = None,
        namespace: str = "default",
        *,
        timeout: float = 5.0,
    ):
        self.namespace = str(namespace)
        self.timeout = float(timeout)
        # `path=None` 落到默认数据目录（<data_dir>/memory/kv/<ns>.db），
        # 而不是当前工作目录 —— `KVMemory("memory.db")` 会把记忆丢进用户的项目里
        raw = str(path) if path is not None else default_kv_path(self.namespace)
        self.path = raw if raw == ":memory:" else expand_path(raw)
        self._conn: Optional[sqlite3.Connection] = None
        # 构造时就把目录建好：`StorageConfig` 报错要早，别等第一次 write
        self._prepare_parent()

    def _prepare_parent(self) -> None:
        '''自动把父目录建出来（用户不用手动 mkdir）。'''
        if self.path == ":memory:":
            return
        folder = os.path.dirname(self.path)
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                raise MemoryError(
                    f"建不了记忆目录 {folder!r}：{exc}"
                ) from None

    # ---------------- 连接 / 建表 ----------------
    @property
    def conn(self) -> sqlite3.Connection:
        '''懒连接：构造 `KVMemory(...)` 不会立刻碰磁盘。'''
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        # 目录在 __init__ 里已经建好；这里再兜一次（运行中被删掉的情况）
        self._prepare_parent()
        try:
            connection = sqlite3.connect(
                self.path, timeout=self.timeout, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise MemoryError(
                f"打不开记忆数据库 {self.path!r}：{exc}"
            ) from None
        connection.execute(_SCHEMA)
        connection.commit()
        return connection

    def close(self) -> None:
        '''关掉连接（可以重复调）。'''
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "KVMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ---------------- 读写 ----------------
    def read(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value FROM memory WHERE namespace = ? AND key = ?",
            (self.namespace, str(key)),
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise MemoryError(
                f"记忆 {self.namespace}/{key} 里存的不是合法 JSON：{exc}"
            ) from None

    def write(self, key: str, value: Any) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False, default=_json_default)
        except (TypeError, ValueError) as exc:
            raise MemoryError(
                f"写不进记忆 {self.namespace}/{key}：值必须能 JSON 序列化（{exc}）。"
                "对象请用 dict / pydantic / dataclass，或者自己先转换"
            ) from None
        self.conn.execute(
            """
            INSERT INTO memory (namespace, key, value, created_at, updated_at)
            VALUES (?, ?, ?, strftime('%s', 'now') * 1.0, strftime('%s', 'now') * 1.0)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (self.namespace, str(key), payload),
        )
        self.conn.commit()

    def forget(self, key: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM memory WHERE namespace = ? AND key = ?",
            (self.namespace, str(key)),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ---------------- 查询 ----------------
    def keys(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT key FROM memory WHERE namespace = ? ORDER BY key",
            (self.namespace,),
        ).fetchall()
        return [row[0] for row in rows]

    def items(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value FROM memory WHERE namespace = ? ORDER BY key",
            (self.namespace,),
        ).fetchall()
        result: Dict[str, Any] = {}
        for key, raw in rows:
            try:
                result[key] = json.loads(raw)
            except json.JSONDecodeError:
                result[key] = raw
        return result

    def count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memory WHERE namespace = ?", (self.namespace,)
        ).fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        '''清空这个 namespace（别的 namespace 不动）。'''
        self.conn.execute(
            "DELETE FROM memory WHERE namespace = ?", (self.namespace,)
        )
        self.conn.commit()

    # ---------------- 生命周期：备份 / 迁移 / 压缩 / 删除 ----------------
    def backup(self, target: str) -> str:
        '''热备份到另一个 `.db` 文件，返回真实路径。

        用 SQLite 的在线备份 API：**边写边备**也不会拿到半截库
        （直接复制文件在写入过程中可能复制到一半事务）。
        '''
        full = expand_path(target)
        folder = os.path.dirname(full)
        if folder:
            os.makedirs(folder, exist_ok=True)
        destination = sqlite3.connect(full)
        try:
            self.conn.backup(destination)
        finally:
            destination.close()
        return full

    def export_json(self, path: str) -> str:
        '''把当前 namespace 导成 JSON（换后端 / 迁移 / 人肉检查用）。'''
        payload = {
            "version": 1,
            "namespace": self.namespace,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": self.items(),
        }
        full = expand_path(path)
        folder = os.path.dirname(full)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return full

    def import_json(self, path: str, *, replace: bool = False) -> int:
        '''从 `export_json` 写的文件里导回来，返回导入了多少条。

        `replace=True` 先清空当前 namespace；默认是合并（同名键覆盖）。
        namespace **以本实例为准** —— 导入不会把数据写到别的 namespace 去。
        '''
        full = expand_path(path)
        with open(full, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
            raise MemoryError(
                f"{full!r} 不是 KVMemory 导出的格式（需要 {{'items': {{...}}}}）"
            )
        items = payload["items"]
        if replace:
            self.clear()
        for key, value in items.items():
            self.write(key, value)
        return len(items)

    def vacuum(self) -> None:
        '''压实数据库文件（删过很多键之后能把空间还回去）。'''
        self.conn.execute("VACUUM")
        self.conn.commit()

    def size_bytes(self) -> int:
        '''数据库文件多大（`:memory:` 返回 0）。'''
        if self.path == ":memory:":
            return 0
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def destroy(self) -> bool:
        '''删库（关连接 + 删文件），返回是否真的删掉了。

        ⚠️ 不可恢复。只影响**这一个文件**，别的 namespace 如果存在同库里，
        会跟着一起没了 —— 想按用户删请一个 namespace 一个文件
        （默认路径就是按 namespace 分文件的）。
        '''
        self.close()
        if self.path == ":memory:":
            return False
        for attempt in range(3):
            try:
                os.remove(self.path)
                return True
            except FileNotFoundError:
                return False
            except PermissionError:
                if attempt == 2:
                    raise MemoryError(
                        f"删不掉 {self.path!r}：文件还被别的进程占着"
                    ) from None
                time.sleep(0.05 * (attempt + 1))
        return False

    def namespaces(self) -> List[str]:
        '''库里有哪几个 namespace（排查「串号」时有用）。'''
        rows = self.conn.execute(
            "SELECT DISTINCT namespace FROM memory ORDER BY namespace"
        ).fetchall()
        return [row[0] for row in rows]

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"<KVMemory {self.path!r} namespace={self.namespace!r} "
            f"keys={self.count()}>"
        )
