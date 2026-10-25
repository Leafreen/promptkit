# coding/diff.py
'''统一 diff（unified diff）的解析与应用 —— 不依赖 `git apply`。

为什么自己写而不是 shell 出去跑 `git apply`：

1. **要能给出可读的错误**。「第 2 个 hunk 的上下文找不到」比
   `error: patch failed: src/a.py:10` 更容易塞回提示词让模型改。
2. **要能先干跑**。`CodingStep` 在写文件之前先验证一遍「这个补丁apply得上」，
   失败了就带错误重试 —— 而不是把文件改坏再回滚。
3. **不要求用户装 git**，也能在临时目录里做演示。

支持的东西：多文件补丁、多 hunk、按上下文偏移搜索（模型给的行号经常偏几行）、
新建文件（`--- /dev/null`）、删除文件（`+++ /dev/null`）、`\\ No newline at end of file`
标记、以及常见的「模型把空上下文行写成空行」。
'''
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from promptkit.coding.errors import DiffError

__all__ = [
    "Hunk",
    "FilePatch",
    "parse_patch",
    "validate_patch",
    "apply_hunks",
    "apply_patch_text",
    "strip_code_fence",
]

_HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


@dataclass
class Hunk:
    '''一个 `@@ -a,b +c,d @@` 块。

    `lines` 里每项是 `(标记, 文本)`，标记为 `" "`（上下文）/ `"+"`（新增）/ `"-"`（删除）。
    '''

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[Tuple[str, str]] = field(default_factory=list)

    def old_lines(self) -> List[str]:
        return [text for mark, text in self.lines if mark in " -"]

    def new_lines(self) -> List[str]:
        return [text for mark, text in self.lines if mark in " +"]

    def __repr__(self) -> str:
        return (
            f"Hunk(-{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count}, {len(self.lines)} 行)"
        )


@dataclass
class FilePatch:
    '''一个文件的补丁（`--- 旧` / `+++ 新` 加若干 hunk）。'''

    old_path: Optional[str]
    new_path: Optional[str]
    hunks: List[Hunk] = field(default_factory=list)

    @property
    def path(self) -> Optional[str]:
        '''这个补丁要写到哪个文件（新建看新路径，删除看旧路径）。'''
        return self.new_path or self.old_path

    @property
    def is_creation(self) -> bool:
        return self.old_path is None

    @property
    def is_deletion(self) -> bool:
        return self.new_path is None

    def __repr__(self) -> str:
        return f"FilePatch({self.old_path!r} -> {self.new_path!r}, {len(self.hunks)} hunk)"


# ============================ 解析 ============================
def strip_code_fence(text: str) -> str:
    '''去掉模型最爱加的 ``` 围栏（```diff / ```python …）。'''
    body = (text or "").strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip("\n")


def _is_file_header(lines: Sequence[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].startswith("--- ")
        and lines[index + 1].startswith("+++ ")
    )


def _clean_path(raw: str) -> Optional[str]:
    '''从 `--- a/src/a.py\t2026-10-22` 这种行里取出路径。'''
    text = (raw or "").strip().strip('"')
    if not text:
        return None
    text = text.split("\t")[0].strip()  # git 用 tab 分隔时间戳
    if text == "/dev/null":
        return None
    if text.startswith(("a/", "b/")):
        text = text[2:]                 # git 的 a/ b/ 前缀
    return text or None


def _header_path(line: str, marker: str) -> Optional[str]:
    return _clean_path(line[len(marker) :])


def _parse_hunk(
    lines: Sequence[str], index: int, match: "re.Match[str]"
) -> Tuple[Hunk, int]:
    '''从 hunk 头开始解析一个块，返回 (Hunk, 下一行下标)。

    用**行数**驱动解析，而不是「遇到下一个 @@ 为止」：删除行的内容如果正好
    以 `- ` 开头，文本上会和 `--- ` 撞车；按行数走就不会误判。
    '''
    old_start = int(match.group(1))
    old_count = int(match.group(2) or 1)
    new_start = int(match.group(3))
    new_count = int(match.group(4) or 1)
    index += 1

    body: List[Tuple[str, str]] = []
    old_seen = new_seen = 0
    while index < len(lines) and (old_seen < old_count or new_seen < new_count):
        line = lines[index]
        if line.startswith("\\"):       # \ No newline at end of file
            index += 1
            continue
        mark = line[0] if line else " "
        text = line[1:] if line else ""
        if mark == " ":
            old_seen += 1
            new_seen += 1
        elif mark == "-":
            old_seen += 1
        elif mark == "+":
            new_seen += 1
        else:
            break
        body.append((mark, text))
        index += 1

    if old_seen != old_count or new_seen != new_count:
        raise DiffError(
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@ 的行数对不上："
            f"实际读到 {old_seen} 行旧内容 / {new_seen} 行新内容。"
            "检查一下是不是漏了某一行、或者多写了一行"
        )
    return Hunk(old_start, old_count, new_start, new_count, body), index


def parse_patch(text: str) -> List[FilePatch]:
    '''把补丁文本解析成 `FilePatch` 列表（结构错了抛 `DiffError`）。'''
    lines = strip_code_fence(text or "").splitlines()
    if not lines:
        raise DiffError("补丁是空的")

    index = 0
    # 前面可能有「好的，以下是修改：」这类说明，跳过
    while index < len(lines) and not _is_file_header(lines, index):
        index += 1
    if index >= len(lines):
        raise DiffError(
            "补丁里找不到 `--- 旧文件` / `+++ 新文件` 头。"
            "需要标准 unified diff（不是一整段代码）"
        )

    patches: List[FilePatch] = []
    while index < len(lines):
        if not _is_file_header(lines, index):
            index += 1
            continue

        old_path = _header_path(lines[index], "---")
        new_path = _header_path(lines[index + 1], "+++")
        index += 2

        hunks: List[Hunk] = []
        while index < len(lines) and not _is_file_header(lines, index):
            match = _HUNK_HEADER.match(lines[index])
            if match is None:
                index += 1
                continue
            hunk, index = _parse_hunk(lines, index, match)
            hunks.append(hunk)

        if not hunks:
            raise DiffError(
                f"文件 {new_path or old_path!r} 的补丁里没有任何 @@ hunk，"
                "等于什么都没改"
            )
        patches.append(FilePatch(old_path, new_path, hunks))

    if not patches:
        raise DiffError("补丁里没有任何可应用的改动")
    return patches


def validate_patch(text: str, *, check_paths: bool = True) -> List[FilePatch]:
    '''解析 + 路径安全校验（绝对路径 / `..` 一律拒绝）。

    注意这里**不**检查「补丁和现有文件对不对得上」—— 那需要文件内容，
    由 `CodingStep`（干跑）和 `ApplyDiff`（真应用）各自的沙箱负责。
    '''
    patches = parse_patch(text)
    if not check_paths:
        return patches

    for patch in patches:
        target = patch.path
        if not target:
            raise DiffError("补丁里既没有旧文件也没有新文件，无法判断改哪个文件")
        if (
            target.startswith(("/", "\\"))
            or os.path.isabs(target)
            or re.match(r"^[A-Za-z]:[\\/]", target)
        ):
            raise DiffError(
                f"补丁里的路径是绝对路径：{target!r}；只允许项目内的相对路径"
            )
        parts = re.split(r"[\\/]+", target)
        if ".." in parts:
            raise DiffError(
                f"补丁里的路径包含 `..`：{target!r}；只允许项目内的相对路径"
            )
    return patches


# ============================ 应用 ============================
def _locate(
    lines: Sequence[str], block: Sequence[str], expected: int, window: int
) -> Optional[int]:
    '''在 `expected` 附近找一个能让 `block` 完全对上的位置。'''
    if not block:
        return max(0, min(expected, len(lines)))
    for delta in range(0, window + 1):
        for position in (expected + delta, expected - delta):
            if position < 0 or position + len(block) > len(lines):
                continue
            if list(lines[position : position + len(block)]) == list(block):
                return position
    return None


def apply_hunks(
    source: str, hunks: Sequence[Hunk], *, path: str = "<补丁>", fuzz: int = 200
) -> str:
    '''把一串 hunk 应用到源码文本上，返回新文本（不改文件）。

    - 行尾统一成 `\\n`（这个项目所有写文件的步骤都这么干）
    - 原文末尾有没有换行，结果里保持一样
    - 行号对不上时在 `fuzz` 行范围内前后找上下文；找不到就报错，**绝不硬改**
    '''
    normalized = (source or "").replace("\r\n", "\n").replace("\r", "\n")
    trailing = normalized.endswith("\n")
    lines = normalized.split("\n") if normalized else []
    if trailing:
        lines = lines[:-1]

    result = list(lines)
    offset = 0
    for number, hunk in enumerate(hunks, start=1):
        expected = max(0, hunk.old_start - 1 + offset)
        old_block = hunk.old_lines()
        new_block = hunk.new_lines()
        position = _locate(result, old_block, expected, fuzz)
        if position is None:
            wanted = "\n".join(f"  {line}" for line in old_block[:6]) or "  （空块）"
            raise DiffError(
                f"{path}：第 {number} 个 hunk 的上下文找不到（在 {expected + 1} 行附近找了 "
                f"{fuzz} 行）。补丁可能基于旧版本文件。期望看到：\n{wanted}"
            )
        result[position : position + len(old_block)] = new_block
        offset += len(new_block) - len(old_block)

    text = "\n".join(result)
    if trailing or (not source and result):
        text += "\n"
    return text


def apply_patch_text(
    source: str, patch: str, *, path: str = "<补丁>", check_paths: bool = False
) -> str:
    '''单文件便捷入口：解析补丁 -> 应用 -> 返回新文本。'''
    patches = validate_patch(patch, check_paths=check_paths)
    hunks = [hunk for file_patch in patches for hunk in file_patch.hunks]
    return apply_hunks(source, hunks, path=path)
