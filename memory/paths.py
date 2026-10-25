# memory/paths.py
'''记忆存哪儿：默认数据目录 + 目录命名规则。

**核心库不碰 `os.getcwd()`**，也不把 `"memory.db"` 写死 —— 那会把记忆丢进
用户的项目目录。这里只有两个原则：

1. 默认落在操作系统的「应用数据目录」下（Windows `%APPDATA%`、macOS
   `Library/Application Support`、Linux `$XDG_DATA_HOME`），
   可以用环境变量 `PROMPTKIT_DATA_DIR` 整体改掉（测试 / 便携部署用）。
2. 用 `namespace` 定文件名：一个用户一个文件 / 一个文件夹，天然隔离。

目录结构：

    <data_dir>/                 # 默认 OS 数据目录 / promptkit
    └── memory/
        ├── kv/
        │   └── user_001.db     # KVMemory 的默认位置
        └── vector/
            └── user_001/       # VectorMemory 的默认文件夹
                └── index.jsonl

要换位置就传 `path=`（`KVMemory(path=...)` / `VectorMemory(path=...)` /
`StorageConfig(path=...)`）—— 默认值只是「不用动脑也能用」，不是唯一答案。
'''
from __future__ import annotations

import os
import re
import sys
import zlib

__all__ = [
    "DATA_DIR_ENV",
    "default_data_dir",
    "memory_dir",
    "default_kv_path",
    "default_vector_dir",
    "safe_namespace_name",
    "expand_path",
]

#: 整体改数据目录（测试、便携部署、容器里挂卷都用它）
DATA_DIR_ENV = "PROMPTKIT_DATA_DIR"

#: 在数据目录下用哪个子目录
APP_DIR = "promptkit"

#: 文件名里不允许出现的字符：路径分隔符 + Windows 保留字符 + 控制字符
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def expand_path(path: str) -> str:
    '''`~` / 环境变量展开成绝对路径（不建目录，不碰文件）。'''
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def default_data_dir() -> str:
    '''默认数据目录（`PROMPTKIT_DATA_DIR` 优先）。

        Windows   %APPDATA%\\promptkit
        macOS     ~/Library/Application Support/promptkit
        其它        $XDG_DATA_HOME/promptkit 或 ~/.local/share/promptkit

    每个平台都留了兜底：`APPDATA` / `HOME` 缺失时退回当前目录下的
    `.promptkit`，而不是 `KeyError` 炸掉（容器里很常见）。
    '''
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return expand_path(override)

    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")

    if not base or not os.path.isabs(base):
        base = os.path.join(os.getcwd(), ".promptkit")
    return os.path.join(base, APP_DIR)


def memory_dir() -> str:
    '''`<data_dir>/memory`（KV 和向量都在它下面）。'''
    return os.path.join(default_data_dir(), "memory")


def safe_namespace_name(namespace: str) -> str:
    '''把 namespace 变成一个安全的文件 / 文件夹名。

    - 去掉路径分隔符和 Windows 保留字符（顺便堵住 `../` 这类路径穿越）
    - 中文、空格、点号都保留（`张三` 就是 `张三.db`，好认）
    - 一旦真的被改过，追加 8 位 crc32，避免 `a/b` 和 `a_b` 撞到同一个文件
    '''
    raw = "" if namespace is None else str(namespace)
    name = _BAD_NAME.sub("_", raw).strip().strip(".")
    if not name or name in (".", ".."):
        name = "default"
    if name != raw:
        # 同名不同 namespace 绝不能落到同一个文件：用 crc32 做个稳定区分
        checksum = zlib.crc32(raw.encode("utf-8")) & 0xFFFFFFFF
        name = f"{name[:40]}-{checksum:08x}"
    return name


def default_kv_path(namespace: str = "default") -> str:
    '''`KVMemory` 不给 `path` 时的默认位置：`<data_dir>/memory/kv/<ns>.db`。'''
    return os.path.join(memory_dir(), "kv", f"{safe_namespace_name(namespace)}.db")


def default_vector_dir(namespace: str = "default") -> str:
    '''`VectorMemory` 不给路径时的默认文件夹：`<data_dir>/memory/vector/<ns>/`。'''
    return os.path.join(memory_dir(), "vector", safe_namespace_name(namespace))
