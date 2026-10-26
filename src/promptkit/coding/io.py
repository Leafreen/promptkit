# coding/io.py
'''文件原语：读 / 写 / 应用补丁 / 列目录 / 搜代码。

命名上有一处**刻意避开**：`ReadFile` 这个名字已经被
`promptkit.core.ReadFile`（一个惰性表达式）占了，所以这里的步骤叫
`ReadFileStep` / `WriteFileStep`。`ApplyDiff` / `ListDir` / `SearchCode`
没有冲突，保持短名字。

四个原语的分工：

| 步骤 | 干什么 | 典型场景 |
| --- | --- | --- |
| `ReadFileStep` | 读文件 -> Context | 改代码前先拿上下文 |
| `WriteFileStep` | 整文件写 | 新建文件 / 写 `final.patch` |
| `ApplyDiff` | 应用 unified diff | 改现有文件（省 token） |
| `ApplyChanges` | 按 `{"changes": [...]}` 批量写 | 多文件重构 |
| `ListDir` / `SearchCode` | 找文件 / 搜代码 | 定位要改哪里 |

所有路径都过 `Sandbox`：不允许 `..`、绝对路径、敏感文件；写操作在
步骤里**默认允许**（是你显式写了 `WriteFileStep(...)`），但工具层默认只读。
'''
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from promptkit.coding.diff import apply_hunks, validate_patch
from promptkit.coding.errors import DiffError
from promptkit.coding.formats import validate_changes
from promptkit.coding.sandbox import Sandbox
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context

__all__ = [
    "ReadFileStep",
    "WriteFileStep",
    "ApplyDiff",
    "ApplyChanges",
    "ListDir",
    "SearchCode",
]


def _render(value: Any, context: "Context") -> Any:
    return value.eval(context) if isinstance(value, Expr) else value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


# ============================ 读 ============================
@dataclass
class ReadFileStep(Step):
    '''读一个文件，写进 Context。

        ReadFileStep("src/main.py", output_key="main_source")

    `output_key` 省略时写 `"source"`；读多个文件就分别起名字。
    `max_chars` 可以覆盖沙箱的默认截断长度。
    '''

    path: Any
    output_key: str = "source"
    base_dir: str = "."
    max_chars: Optional[int] = None
    encoding: Optional[str] = None

    name: str = "read_file"

    def execute(self, context: "Context") -> None:
        path = _text(_render(self.path, context))
        sandbox = Sandbox(self.base_dir, encoding=self.encoding)
        text = sandbox.read_text(path, max_chars=self.max_chars)
        context[self.output_key] = text
        context.trace.append(f"{self.name}:{path}")

    def describe(self) -> str:
        return f"ReadFileStep({self.path!r} -> {self.output_key})"


# ============================ 写 ============================
@dataclass
class WriteFileStep(Step):
    '''把内容整块写进文件（自动建目录、原子写、统一 LF）。

        WriteFileStep("outputs/final.patch", content=Var("final_diff"))

    `output_key` 给了就顺手把「写了哪个文件、多少字符」写进 Context，
    方便后面 `Echo` 或审计。
    '''

    path: Any
    content: Any
    output_key: Optional[str] = None
    base_dir: str = "."
    encoding: Optional[str] = None

    name: str = "write_file"

    def execute(self, context: "Context") -> None:
        path = _text(_render(self.path, context))
        content = _text(_render(self.content, context))
        sandbox = Sandbox(self.base_dir, allow_write=True, encoding=self.encoding)
        full = sandbox.write_text(path, content)
        if self.output_key:
            context[self.output_key] = {
                "path": sandbox.relative(full),
                "chars": len(content),
                "written": True,
            }
        context.trace.append(f"{self.name}:{path}")

    def describe(self) -> str:
        return f"WriteFileStep({self.path!r})"


# ============================ 打补丁 ============================
def apply_patch(
    sandbox: Sandbox,
    diff_text: str,
    *,
    path: Optional[str] = None,
    backup: bool = False,
) -> Dict[str, Any]:
    '''把补丁应用到文件上，返回 `{"files", "created", "deleted", "hunks"}`。

    先全部算好再写盘：任何一个 hunk 对不上就整体失败，不会留下改了一半的文件。
    `path` 给了就是单文件模式（补丁头里的路径不参与）；没给就按补丁头多文件应用。
    '''
    if not (diff_text or "").strip():
        raise DiffError("补丁是空的")
    patches = validate_patch(diff_text, check_paths=path is None)

    applied: List[str] = []
    created: List[str] = []
    deleted: List[str] = []
    hunks_total = 0

    if path is not None:
        hunks = [hunk for patch in patches for hunk in patch.hunks]
        hunks_total = len(hunks)
        exists = sandbox.exists(path)
        source = sandbox.read_text(path) if exists else ""
        new_text = apply_hunks(source, hunks, path=path)
        if backup and exists:
            sandbox.write_text(path + ".orig", source)
        sandbox.write_text(path, new_text)
        applied.append(path)
        if not exists:
            created.append(path)
        return {"files": applied, "created": created, "deleted": [], "hunks": hunks_total}

    for patch in patches:
        name = patch.path
        if not name:
            raise DiffError("补丁里没有文件名")
        hunks_total += len(patch.hunks)
        if patch.is_deletion:
            if sandbox.remove(name):
                deleted.append(name)
            continue
        exists = sandbox.exists(name)
        source = sandbox.read_text(name) if exists else ""
        new_text = apply_hunks(source, patch.hunks, path=name)
        if backup and exists:
            sandbox.write_text(name + ".orig", source)
        sandbox.write_text(name, new_text)
        applied.append(name)
        if not exists:
            created.append(name)

    return {
        "files": applied,
        "created": created,
        "deleted": deleted,
        "hunks": hunks_total,
    }


@dataclass
class ApplyDiff(Step):
    '''把一个 unified diff 应用到文件上。

        ApplyDiff("src/main.py", diff=Var("patch"))     # 单文件
        ApplyDiff(diff=Var("patch"))                    # 多文件（按补丁头里的路径）
        ApplyDiff(..., backup=True)                     # 顺手留一份 .orig

    应用是**先全部算好再写盘**：任何一个 hunk 对不上就整体失败，不会留下
    改了一半的文件。`backup=True` 时把原内容写到 `<path>.orig`。
    '''

    path: Optional[Any] = None
    diff: Any = None
    base_dir: str = "."
    backup: bool = False
    output_key: str = "apply_result"

    name: str = "apply_diff"

    def __post_init__(self) -> None:
        # 字段顺序按规格来：`ApplyDiff(path, diff)` 和 `ApplyDiff(diff=...)` 都要能用
        if self.diff is None:
            raise ValueError(
                "ApplyDiff 需要 diff=（补丁文本，通常是 Var('patch')）"
            )

    def execute(self, context: "Context") -> None:
        patch_text = _text(_render(self.diff, context))
        target = _text(_render(self.path, context)) if self.path is not None else None
        result = apply_patch(
            Sandbox(self.base_dir, allow_write=True),
            patch_text,
            path=target,
            backup=self.backup,
        )
        context[self.output_key] = result
        context.trace.append(f"{self.name}:{len(result['files'])}")

    def describe(self) -> str:
        where = f"{self.path!r}" if self.path is not None else "补丁里的路径"
        return f"ApplyDiff({where} -> {self.output_key})"


@dataclass
class ApplyChanges(Step):
    '''按 `{"changes": [...]}` 批量写 / 删文件（`output_format="json"` 的落盘端）。

        CodingStep(..., output_format="json", output_key="changes"),
        ApplyChanges(Var("changes")),
    '''

    changes: Any
    base_dir: str = "."
    output_key: str = "apply_result"

    name: str = "apply_changes"

    def execute(self, context: "Context") -> None:
        raw = _render(self.changes, context)
        if isinstance(raw, dict):
            raw = raw.get("changes")
        items = validate_changes(raw)

        sandbox = Sandbox(self.base_dir, allow_write=True)
        written: List[str] = []
        deleted: List[str] = []
        for item in items:
            if item["action"] == "delete":
                if sandbox.remove(item["path"]):
                    deleted.append(item["path"])
                continue
            sandbox.write_text(item["path"], item["content"])
            written.append(item["path"])

        context[self.output_key] = {
            "files": written,
            "created": [],
            "deleted": deleted,
            "changes": len(items),
        }
        context.trace.append(f"{self.name}:{len(items)}")

    def describe(self) -> str:
        return f"ApplyChanges(-> {self.output_key})"


def format_hits(hits: List[Dict[str, Any]]) -> str:
    '''把搜索结果拼成「grep 风格」的文本（给模型 / 给人看）。

        coding/diff.py:265: def apply_hunks(
          263|     result = list(lines)
          264|     offset = 0

    `before` / `after` 是 `Sandbox.search(context=N)` 带出来的上下文行。
    '''
    if not hits:
        return "（没有匹配）"
    lines: List[str] = []
    for hit in hits:
        if "matches" in hit and "line" not in hit:      # files_only 模式
            lines.append(f"{hit['path']}（{hit['matches']} 处）")
            continue
        line_number = int(hit.get("line", 0))
        before = list(hit.get("before") or ())
        lines.append(f"{hit['path']}:{line_number}: {hit['text']}")
        start = line_number - len(before)
        for offset, text in enumerate(before):
            lines.append(f"  {start + offset:>4}| {text}")
        for offset, text in enumerate(hit.get("after") or (), start=1):
            lines.append(f"  {line_number + offset:>4}| {text}")
    return "\n".join(lines)


# ============================ 看 ============================
@dataclass
class ListDir(Step):
    '''列目录（相对路径、排序、跳过 `.git` / `__pycache__` 这类噪音目录）。

        ListDir("src", pattern="*.py", output_key="py_files")
    '''

    path: Any = "."
    pattern: Optional[str] = None
    output_key: str = "files"
    base_dir: str = "."

    name: str = "list_dir"

    def execute(self, context: "Context") -> None:
        path = _text(_render(self.path, context)) or "."
        sandbox = Sandbox(self.base_dir)
        files = sandbox.list_dir(path, self.pattern)
        context[self.output_key] = files
        context.trace.append(f"{self.name}:{len(files)}")

    def describe(self) -> str:
        pattern = f", pattern={self.pattern!r}" if self.pattern else ""
        return f"ListDir({self.path!r}{pattern} -> {self.output_key})"


@dataclass
class SearchCode(Step):
    '''在源码里搜一段内容（正则；写错了当普通字符串）。

        SearchCode("def calculate", glob="*.py", output_key="hits")
        SearchCode("样本量", context=2, as_text=True, output_key="where")

    - `context=N`：每条命中多带前后 N 行（**改代码前最好看一眼上下文**）
    - `files_only=True`：只回「哪个文件命中几处」，适合先定位再 `ReadFileStep`
    - `exclude=("tests/*",)`：按相对路径排除
    - `as_text=True`：拼成 `路径:行号: 内容` 的文本；默认是 `list[dict]`，
      方便下游用 `Var("hits")[0]["path"]` 取值
    '''

    query: Any
    glob: str = "*.py"
    output_key: str = "hits"
    base_dir: str = "."
    ignore_case: bool = False
    context: int = 0
    max_per_file: int = 10
    files_only: bool = False
    exclude: Tuple[str, ...] = ()
    as_text: bool = False

    name: str = "search_code"

    def execute(self, context: "Context") -> None:
        query = _text(_render(self.query, context))
        sandbox = Sandbox(self.base_dir)
        hits = sandbox.search(
            query,
            glob=self.glob,
            ignore_case=self.ignore_case,
            context=self.context,
            max_per_file=self.max_per_file,
            files_only=self.files_only,
            exclude=self.exclude,
        )
        context[self.output_key] = format_hits(hits) if self.as_text else hits
        context.trace.append(f"{self.name}:{len(hits)}")

    def describe(self) -> str:
        extra = f", context={self.context}" if self.context else ""
        extra += ", files_only" if self.files_only else ""
        return (
            f"SearchCode({self.query!r}, glob={self.glob!r}{extra}"
            f" -> {self.output_key})"
        )
