# coding/sandbox.py
'''编码任务的安全边界：所有文件操作都必须过这一层。

模型会生成路径。路径可能是 `../../etc/passwd`、`.env`、`C:\\Windows\\...`，
也可能是 `a.py`（正常）。这一层负责：

| 红线 | 做法 |
| --- | --- |
| 越界 | 只接受相对路径；`realpath` 之后必须还在 `base_dir` 里（符号链接也跑不掉） |
| 绝对路径 | 直接拒绝（`D:/x`、`/etc/x` 都算） |
| 敏感文件 | `.env` / `id_rsa*` / `*.pem` / `.git/config` … 读写都拒绝，除非 `allow=` 显式放行 |
| 写权限 | `allow_write=False` 时一律只读（**工具默认只读**） |
| 命令 | 配了 `allow_commands` 就只放行这些可执行文件，并挡掉 shell 元字符 |
| 失控 | 单文件读取截断、列目录条数上限、搜索命中上限 |
| 审计 | 每次读 / 写 / 执行都打一条 `promptkit.coding` 日志 |

**它不是沙箱级隔离**（没有容器、没有系统调用过滤）。真要跑不受信任的代码，
请用 Docker / 虚拟机 —— 这里的定位是「防止模型手滑，不防止恶意提权」。
'''
from __future__ import annotations

import fnmatch
import logging
import os
import re
import shlex
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from promptkit.coding.errors import CommandError, SandboxError
from promptkit.core.settings import settings

__all__ = [
    "Sandbox",
    "CODING_LOGGER",
    "DEFAULT_SENSITIVE",
    "DEFAULT_SKIP_DIRS",
]

#: 编码 / 执行相关的审计日志
CODING_LOGGER = "promptkit.coding"

#: 默认不碰的文件（按相对路径和文件名匹配，支持 fnmatch 通配）
DEFAULT_SENSITIVE: Sequence[str] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_dsa*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".aws/credentials",
    ".git/config",
)

#: 列目录 / 搜索时跳过的目录（噪音大、也没人想让模型读）
DEFAULT_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "dist", "build", ".idea"}
)


def _decode_bytes(raw: bytes, encoding: Optional[str] = None) -> str:
    '''按 utf-8 -> 系统编码 -> gbk 的顺序解码。

    源码文件可能是 GBK（Windows 上的老项目），写死 utf-8 会直接抛
    `UnicodeDecodeError`，而那会看起来像「文件读不了」。
    '''
    if not raw:
        return ""
    candidates: List[str] = []
    if encoding:
        candidates.append(encoding)
    candidates += ["utf-8", settings.encoding, "gbk", "latin-1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


class Sandbox:
    '''一个目录范围内的读写沙箱。

        sandbox = Sandbox("D:/proj", allow_write=True,
                          allow_commands=["pytest", "python", "ruff"])
        text = sandbox.read_text("src/main.py")
        sandbox.write_text("src/main.py", text)
        sandbox.run(["python", "-m", "pytest"])     # 见 coding.exec

    默认**只读**：要写必须显式 `allow_write=True`。
    '''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        allow_write: bool = False,
        allow: Iterable[str] = (),
        sensitive: Optional[Sequence[str]] = None,
        allow_commands: Optional[Sequence[str]] = None,
        max_read_chars: int = 200_000,
        max_list: int = 500,
        max_search_hits: int = 50,
        max_search_files: int = 500,
        max_per_file: int = 10,
        max_line_chars: int = 200,
        encoding: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base = os.path.realpath(os.path.abspath(os.path.expanduser(str(base_dir))))
        if not os.path.isdir(self.base):
            raise SandboxError(f"base_dir 不存在或不是目录：{self.base!r}")
        self.allow_write = bool(allow_write)
        #: 即使命中敏感规则也放行的路径（比如 `.env.example`）
        self.allow = [str(item) for item in allow]
        self.sensitive = list(DEFAULT_SENSITIVE if sensitive is None else sensitive)
        #: None = 不限制命令；给了就只放行这些可执行文件
        self.allow_commands = (
            None
            if allow_commands is None
            else [str(item).lower() for item in allow_commands]
        )
        self.max_read_chars = int(max_read_chars)
        self.max_list = int(max_list)
        self.max_search_hits = int(max_search_hits)
        self.max_search_files = int(max_search_files)
        self.max_per_file = int(max_per_file)
        self.max_line_chars = int(max_line_chars)
        self.encoding = encoding
        self.logger = logger or logging.getLogger(CODING_LOGGER)

    # ---------------- 路径 ----------------
    def _inside(self, full: str) -> bool:
        base = os.path.normcase(self.base)
        target = os.path.normcase(full)
        try:
            return os.path.commonpath([base, target]) == base
        except ValueError:      # 不同盘符
            return False

    def _check_sensitive(self, relative: str, *, write: bool) -> None:
        name = os.path.basename(relative)
        normalized = relative.replace("\\", "/")
        for pattern in self.allow:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern):
                return
        for pattern in self.sensitive:
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern):
                raise SandboxError(
                    f"{relative!r} 是敏感文件（匹配规则 {pattern!r}），"
                    + ("不允许写" if write else "不允许读")
                    + "；确实需要就把它加进 allow="
                )

    def resolve(self, path: str, *, write: bool = False) -> str:
        '''把相对路径解析成绝对路径，顺便做完全部检查。'''
        if not isinstance(path, str) or not path.strip():
            raise SandboxError("路径不能是空的")
        candidate = path.strip().strip('"').strip("'")
        # Windows 上 `ntpath.isabs("/etc/passwd")` 是 False（缺盘符），
        # 所以还要单独挡一下「以分隔符开头」的 POSIX 绝对路径
        if (
            candidate.startswith(("/", "\\"))
            or os.path.isabs(candidate)
            or re.match(r"^[A-Za-z]:[\\/]", candidate)
        ):
            raise SandboxError(f"只接受相对路径，得到绝对路径：{path!r}")
        normalized = os.path.normpath(candidate)
        if normalized == ".." or normalized.startswith(".." + os.sep):
            raise SandboxError(f"路径越界（不能跑到 {self.base!r} 外面）：{path!r}")
        full = os.path.realpath(os.path.join(self.base, normalized))
        if not self._inside(full):
            raise SandboxError(
                f"路径越界：{path!r} 解析后是 {full!r}，不在 {self.base!r} 里"
            )
        relative = os.path.relpath(full, self.base)
        self._check_sensitive(relative, write=write)
        if write and not self.allow_write:
            raise SandboxError(
                "这个沙箱是只读的（allow_write=False）；"
                "要写就显式建 Sandbox(..., allow_write=True)"
            )
        return full

    def relative(self, path: str) -> str:
        '''绝对路径 -> 相对 base 的、正斜杠分隔的路径（给人看 / 写进 context）。'''
        return os.path.relpath(path, self.base).replace("\\", "/")

    # ---------------- 读 ----------------
    def read_text(self, path: str, *, max_chars: Optional[int] = None) -> str:
        full = self.resolve(path)
        try:
            with open(full, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            raise SandboxError(f"文件不存在：{path!r}") from None
        except IsADirectoryError:
            raise SandboxError(f"{path!r} 是目录，不能当文件读") from None
        except OSError as exc:
            raise SandboxError(f"读不了 {path!r}：{exc}") from None

        text = _decode_bytes(raw, self.encoding)
        limit = self.max_read_chars if max_chars is None else int(max_chars)
        self.logger.info("read %s (%d 字符)", self.relative(full), len(text))
        if limit and len(text) > limit:
            return (
                text[:limit]
                + f"\n…（已截断：原文 {len(text)} 字符，只给了前 {limit} 字符）"
            )
        return text

    def exists(self, path: str) -> bool:
        try:
            return os.path.exists(self.resolve(path))
        except SandboxError:
            return False

    # ---------------- 写 ----------------
    def write_text(self, path: str, text: str) -> str:
        '''原子写入（先写临时文件再改名），返回真实路径。'''
        full = self.resolve(path, write=True)
        folder = os.path.dirname(full)
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                raise SandboxError(f"建不了目录 {folder!r}：{exc}") from None

        payload = text if isinstance(text, str) else str(text)
        temporary = full + ".tmp"
        try:
            with open(temporary, "w", encoding=self.encoding or settings.encoding,
                      newline="\n") as handle:
                handle.write(payload)
            _replace_with_retry(temporary, full)
        except OSError as exc:
            raise SandboxError(f"写不了 {path!r}：{exc}") from None
        self.logger.info("write %s (%d 字符)", self.relative(full), len(payload))
        return full

    def remove(self, path: str) -> bool:
        full = self.resolve(path, write=True)
        try:
            os.remove(full)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise SandboxError(f"删不掉 {path!r}：{exc}") from None
        self.logger.info("delete %s", self.relative(full))
        return True

    # ---------------- 看 ----------------
    def list_dir(
        self, path: str = ".", pattern: Optional[str] = None
    ) -> List[str]:
        '''列目录（相对路径、排序、跳过噪音目录）。'''
        full = self.resolve(path or ".")
        if not os.path.isdir(full):
            raise SandboxError(f"{path!r} 不是目录")
        found: List[str] = []
        for root, dirs, files in os.walk(full):
            dirs[:] = sorted(
                name for name in dirs if name not in DEFAULT_SKIP_DIRS
            )
            for name in sorted(files):
                relative = self.relative(os.path.join(root, name))
                if pattern and not fnmatch.fnmatch(name, pattern):
                    continue
                found.append(relative)
                if len(found) >= self.max_list:
                    self.logger.info("list %s -> %d 条（已截断）", path, len(found))
                    return found
        self.logger.info("list %s -> %d 条", path, len(found))
        return found

    def search(
        self,
        query: str,
        *,
        glob: str = "*.py",
        max_files: Optional[int] = None,
        ignore_case: bool = False,
        context: int = 0,
        max_per_file: Optional[int] = None,
        files_only: bool = False,
        exclude: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        '''在 `glob` 匹配的文件里搜 `query`（正则；写错了就当普通字符串）。

        返回 `[{path, line, text}, ...]`；`context=N` 时每条的 `before` / `after`
        多带 N 行源码（**看缩进和上下文，模型才改得对**）。

        - `files_only=True`：只回「哪个文件命中几处」，用来定位「这东西在哪」
        - `max_per_file`：一个文件最多报几处（默认 10），避免一个文件吃光配额
        - `exclude`：按相对路径排除（`tests/*`、`*.min.js` …）
        - `max_files`：最多扫多少个文件（默认 `self.max_search_files`）
        '''
        if not query or not str(query).strip():
            raise SandboxError("搜索内容不能是空的")
        flags = re.IGNORECASE if ignore_case else 0
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

        depth = max(0, int(context))
        per_file = (
            self.max_per_file if max_per_file is None else max(1, int(max_per_file))
        )
        file_budget = (
            self.max_search_files if max_files is None else max(1, int(max_files))
        )
        excluded = [str(item) for item in exclude]

        hits: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []
        scanned = 0
        stop = False

        for root, dirs, filenames in os.walk(self.base):
            dirs[:] = sorted(name for name in dirs if name not in DEFAULT_SKIP_DIRS)
            for name in sorted(filenames):
                if stop:
                    break
                if not fnmatch.fnmatch(name, glob):
                    continue
                relative = self.relative(os.path.join(root, name))
                if any(fnmatch.fnmatch(relative, item) for item in excluded):
                    continue
                if scanned >= file_budget:
                    stop = True
                    break
                scanned += 1

                full = os.path.join(root, name)
                try:
                    if os.path.getsize(full) > 1_000_000:
                        continue
                    with open(full, "rb") as handle:
                        text = _decode_bytes(handle.read(), self.encoding)
                except OSError:
                    continue

                lines = text.splitlines()
                matched_here = 0
                for number, line in enumerate(lines, start=1):
                    if not pattern.search(line):
                        continue
                    matched_here += 1
                    if matched_here > per_file:
                        break
                    if files_only:
                        continue
                    hit: Dict[str, Any] = {
                        "path": relative,
                        "line": number,
                        "text": line.strip()[: self.max_line_chars],
                    }
                    if depth:
                        start = max(0, number - 1 - depth)
                        hit["before"] = [
                            item.rstrip()[: self.max_line_chars]
                            for item in lines[start : number - 1]
                        ]
                        hit["after"] = [
                            item.rstrip()[: self.max_line_chars]
                            for item in lines[number : number + depth]
                        ]
                    hits.append(hit)
                    if len(hits) >= self.max_search_hits:
                        stop = True
                        break

                if files_only and matched_here:
                    files.append({"path": relative, "matches": matched_here})
                    if len(files) >= self.max_search_hits:
                        stop = True

        result = files if files_only else hits
        self.logger.info(
            "search %r (%s) -> %d 条%s，扫了 %d 个文件",
            query,
            glob,
            len(result),
            "（已截断）" if stop else "",
            scanned,
        )
        return result

    # ---------------- 命令 ----------------
    def check_command(self, command: str) -> None:
        '''校验命令：allowlist（第一段可执行文件）+ 挡 shell 元字符。'''
        if self.allow_commands is None:
            return
        text = (command or "").strip()
        if not text:
            raise CommandError("命令是空的")
        tokens = shlex.split(text, posix=(os.name != "nt"))
        if not tokens:
            raise CommandError("命令是空的")
        executable = os.path.basename(tokens[0]).lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in self.allow_commands:
            raise CommandError(
                f"命令 {executable!r} 不在 allowlist 里；允许的："
                + ", ".join(self.allow_commands)
            )
        # allowlist 之下再挡一层：`pytest; rm -rf /` 这类拼接
        if re.search(r"[|&;<>`$]", text):
            raise CommandError(
                "命令里带了 shell 元字符（| & ; < > ` $），allowlist 模式下不允许；"
                "请拆成多条命令分别执行"
            )

    def __repr__(self) -> str:
        mode = "读写" if self.allow_write else "只读"
        return f"<Sandbox {self.base!r} {mode}>"


def _replace_with_retry(source: str, target: str, attempts: int = 5) -> None:
    '''`os.replace` + Windows 重试（杀软 / 索引服务偶尔占着刚写完的临时文件）。'''
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
