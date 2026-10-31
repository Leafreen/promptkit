# core/tool/coding.py
'''给 Agent 自主编码用的工具（`Tool` 包装）。

和 `coding.io` 的步骤是同一批能力，区别在**谁决定调用**：

- 步骤：工作流作者写死顺序（`ReadFileStep` → `CodingStep` → `ApplyDiff` → `RunTests`）
- 工具：模型自己决定读哪个文件、什么时候打补丁、要不要跑测试

安全默认值（比步骤严格，因为这里是模型在决定）：

| 工具 | 默认 |
| --- | --- |
| 读 / 列 / 搜 | 允许（但仍受沙箱：相对路径、敏感文件黑名单、截断） |
| 写 / 打补丁 | **只读**，要显式 `allow_write=True` |
| 跑命令 | 不限制；建议传 `allow_commands=["pytest", "python", …]` |

没有「交互式确认」：工具里没有等人按 y 的通道。确认的语义就是**你有没有把这个
工具交给模型** —— 所以别把 `WriteFileTool(allow_write=True)` 塞进一个没人看着的循环。
'''
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

from promptkit.coding.errors import CodingError
from promptkit.coding.exec import run_command
from promptkit.coding.io import apply_patch, format_hits
from promptkit.coding.sandbox import CODING_LOGGER, Sandbox
from promptkit.core.tool.base import Tool, ToolSchema

__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ApplyDiffTool",
    "ListDirTool",
    "SearchCodeTool",
    "RunCommandTool",
    "RunTestsTool",
    "default_coding_tools",
]


class _CodingTool(Tool):
    '''所有编码工具的公共部分：沙箱 + 把异常变成给模型看的错误文本。'''

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: ToolSchema,
        func: Callable[..., object],
        sandbox: Optional[Sandbox] = None,
        base_dir: str = ".",
        allow_write: bool = False,
        allow_commands: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.sandbox = sandbox or Sandbox(
            base_dir,
            allow_write=allow_write,
            allow_commands=allow_commands,
        )
        self.logger = logger or logging.getLogger(CODING_LOGGER)
        super().__init__(
            name=name, description=description, parameters=parameters, func=func
        )

    # 统一的错误出口：工具不抛异常，把错误文本交给模型自己处理
    def _guard(self, action: str, actor: Callable[..., object], **kwargs: object) -> object:
        try:
            return actor(**kwargs)
        except CodingError as exc:
            self.logger.info("tool %s 失败：%s", self.name, exc)
            return f"错误：{exc}"
        except Exception as exc:  # noqa: BLE001 - 工具的契约是「总能返回」
            self.logger.warning("tool %s 异常：%s: %s", self.name, type(exc).__name__, exc)
            return f"错误：{type(exc).__name__}: {exc}"


# ============================ 读 ============================
class ReadFileTool(_CodingTool):
    '''读一个文件。路径必须是相对 `base_dir` 的。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        sandbox: Optional[Sandbox] = None,
        max_chars: Optional[int] = None,
    ):
        self.max_chars = max_chars
        super().__init__(
            name="read_file",
            description=(
                "读取项目里的一个文件，返回它的内容。"
                "path 必须是相对项目根的路径（比如 src/main.py），"
                "不能用绝对路径或 ..。读不到会返回错误说明。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 src/main.py"}
                },
                "required": ["path"],
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
        )

    def run(self, path: str) -> object:
        return self._guard("read", self._read, path=path)

    def _read(self, path: str) -> str:
        return self.sandbox.read_text(path, max_chars=self.max_chars)


class ListDirTool(_CodingTool):
    '''列目录（跳过 .git / __pycache__ 等）。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        sandbox: Optional[Sandbox] = None,
    ):
        super().__init__(
            name="list_dir",
            description=(
                "列出目录下的文件（相对路径，自动跳过 .git/__pycache__ 等）。"
                "可以用 pattern 过滤，比如 *.py。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录，默认 ."},
                    "pattern": {"type": "string", "description": "文件名通配，比如 *.py"},
                },
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
        )

    def run(self, path: str = ".", pattern: Optional[str] = None) -> object:
        return self._guard("list", self._list, path=path, pattern=pattern)

    def _list(self, path: str = ".", pattern: Optional[str] = None) -> str:
        files = self.sandbox.list_dir(path or ".", pattern)
        return "\n".join(files) if files else "（没有文件）"


class SearchCodeTool(_CodingTool):
    '''在源码里搜内容，返回 `路径:行号: 内容`（可选带上下文行）。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        sandbox: Optional[Sandbox] = None,
    ):
        super().__init__(
            name="search_code",
            description=(
                "在项目源码里搜索一段内容（正则，写错了当普通字符串），"
                "返回「路径:行号: 内容」。用来定位某个函数 / 变量在哪里定义、"
                "谁在调用它。context>0 时会多带前后几行源码。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要找的内容或正则"},
                    "glob": {
                        "type": "string",
                        "description": "文件通配，默认 *.py",
                    },
                    "context": {
                        "type": "integer",
                        "description": "每条命中带几行上下文（默认 0，建议 2）",
                    },
                },
                "required": ["query"],
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
        )

    def run(self, query: str, glob: str = "*.py", context: int = 0) -> object:
        return self._guard(
            "search", self._search, query=query, glob=glob, context=context
        )

    def _search(self, query: str, glob: str = "*.py", context: int = 0) -> str:
        hits = self.sandbox.search(
            query, glob=glob or "*.py", context=max(0, int(context or 0))
        )
        return format_hits(hits)


# ============================ 写 ============================
class WriteFileTool(_CodingTool):
    '''整文件写。默认只读：要 `allow_write=True` 才能写。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        allow_write: bool = False,
        sandbox: Optional[Sandbox] = None,
    ):
        super().__init__(
            name="write_file",
            description=(
                "用给定内容**整体覆盖**一个文件（自动建目录）。"
                "适合新建文件或整文件重写；只想改几行请用 apply_diff。"
                "path 必须是相对路径。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径"},
                    "content": {"type": "string", "description": "完整文件内容"},
                },
                "required": ["path", "content"],
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
            allow_write=allow_write,
        )

    def run(self, path: str, content: str) -> object:
        return self._guard("write", self._write, path=path, content=content)

    def _write(self, path: str, content: str) -> str:
        full = self.sandbox.write_text(path, content)
        return f"已写入 {self.sandbox.relative(full)}（{len(content)} 字符）"


class ApplyDiffTool(_CodingTool):
    '''按 unified diff 改文件。默认只读：要 `allow_write=True`。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        allow_write: bool = False,
        sandbox: Optional[Sandbox] = None,
    ):
        super().__init__(
            name="apply_diff",
            description=(
                "把一个 unified diff 应用到文件上，返回改了哪些文件。"
                "diff 里用 `--- 相对路径` / `+++ 相对路径` 起头；"
                "给了 path 参数就只改那一个文件。"
                "只用于**局部修改**：整文件重写或新建文件请用 write_file。"
                "hunk 头的行数必须和实际内容一致，否则补丁会被拒绝。"
                "补丁对不上（上下文不匹配）会返回错误，不会硬改。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "diff": {"type": "string", "description": "unified diff 文本"},
                    "path": {
                        "type": "string",
                        "description": "只改这个文件（可选；不给就按补丁头里的路径）",
                    },
                },
                "required": ["diff"],
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
            allow_write=allow_write,
        )

    def run(self, diff: str, path: Optional[str] = None) -> object:
        return self._guard("apply", self._apply, diff=diff, path=path)

    def _apply(self, diff: str, path: Optional[str] = None) -> Dict[str, object]:
        return apply_patch(self.sandbox, diff, path=path or None)


# ============================ 跑 ============================
class RunCommandTool(_CodingTool):
    '''跑一条命令（建议配 `allow_commands`）。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        allow_commands: Optional[Sequence[str]] = None,
        timeout: float = 120.0,
        sandbox: Optional[Sandbox] = None,
    ):
        self.timeout = float(timeout)
        super().__init__(
            name="run_command",
            description=(
                "在项目根目录跑一条 shell 命令，返回退出码和输出。"
                "只用它跑测试 / 格式化 / 查看状态，不要跑会卡住的交互式命令。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整命令，一行"},
                    "cwd": {"type": "string", "description": "相对项目根的工作目录"},
                },
                "required": ["command"],
            },
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
            allow_commands=allow_commands,
        )

    def run(self, command: str, cwd: Optional[str] = None) -> object:
        return self._guard("run", self._run, command=command, cwd=cwd)

    def _run(self, command: str, cwd: Optional[str] = None) -> Dict[str, object]:
        return run_command(
            command,
            cwd=cwd,
            timeout=self.timeout,
            sandbox=self.sandbox,
        )


class RunTestsTool(_CodingTool):
    '''跑**固定**的测试命令（不给模型改命令的机会）。'''

    def __init__(
        self,
        base_dir: str = ".",
        *,
        command: str = "python -m pytest -q",
        allow_commands: Optional[Sequence[str]] = None,
        timeout: float = 600.0,
        sandbox: Optional[Sandbox] = None,
    ):
        self.command = str(command)
        self.timeout = float(timeout)
        super().__init__(
            name="run_tests",
            description=(
                f"运行项目的测试命令（固定为 `{self.command}`），"
                "返回是否通过以及输出。改完代码后调用它验证。"
            ),
            parameters={"type": "object", "properties": {}},
            func=self.run,
            sandbox=sandbox,
            base_dir=base_dir,
            allow_commands=allow_commands,
        )

    def run(self) -> object:
        result = self._guard("run_tests", self._run)
        if isinstance(result, dict):
            return (
                ("通过" if result.get("passed") else "未通过")
                + "\n"
                + str(result.get("output", ""))
            )
        return result

    def _run(self) -> Dict[str, object]:
        return run_command(
            self.command, timeout=self.timeout, sandbox=self.sandbox
        )


def default_coding_tools(
    base_dir: str = ".",
    *,
    allow_write: bool = False,
    allow_commands: Optional[Sequence[str]] = None,
    test_command: str = "python -m pytest -q",
) -> List[Tool]:
    '''一组开箱可用的编码工具（默认只读）。

        from promptkit.coding import default_coding_tools
        agent.ask("修掉 tests 里的失败", tools=default_coding_tools("."))
    '''
    sandbox = Sandbox(
        base_dir,
        allow_write=allow_write,
        allow_commands=allow_commands,
    )
    return [
        ReadFileTool(sandbox=sandbox),
        ListDirTool(sandbox=sandbox),
        SearchCodeTool(sandbox=sandbox),
        ApplyDiffTool(sandbox=sandbox),
        RunTestsTool(command=test_command, sandbox=sandbox),
    ]
