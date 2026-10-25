# coding/git.py
'''版本控制原语：`GitDiff` / `GitCommit`。

两个都**不走 shell**（`subprocess.run([...], shell=False)`）：提交信息里有引号、
中文、换行时，拼字符串十有八九会把命令拼坏。

结果统一是 `CommandResult`（和 `RunCommand` 一样），所以条件可以照旧写
`If(Var("commit_result")["passed"] == False)`。
'''
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from promptkit.coding.errors import CommandError
from promptkit.coding.exec import CommandResult, _decode
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context

__all__ = ["git", "GitDiff", "GitCommit"]


def git(
    args: List[str],
    *,
    cwd: str = ".",
    timeout: float = 120.0,
) -> CommandResult:
    '''跑一条 git 子命令（列表形式，不经过 shell）。'''
    command = "git " + " ".join(args)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "returncode": None,
            "passed": False,
            "timed_out": True,
            "duration_s": round(time.perf_counter() - started, 2),
            "output": f"git 超时（>{timeout}s）",
            "stdout": "",
            "stderr": "",
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "passed": False,
            "timed_out": False,
            "duration_s": round(time.perf_counter() - started, 2),
            "output": f"git 起不来：{exc}（装了吗？在 PATH 里吗？）",
            "stdout": "",
            "stderr": "",
        }

    stdout = _decode(completed.stdout)
    stderr = _decode(completed.stderr)
    parts = [part.strip() for part in (stdout, stderr) if part.strip()]
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "timed_out": False,
        "duration_s": round(time.perf_counter() - started, 2),
        "output": "\n".join(parts) or f"（无输出；退出码 {completed.returncode}）",
        "stdout": stdout,
        "stderr": stderr,
    }


@dataclass
class GitDiff(Step):
    '''把 `git diff` 写进 Context（默认只取未暂存的改动）。

        GitDiff(output_key="final_diff")
        GitDiff(staged=True, path="src/", output_key="staged_diff")
    '''

    output_key: str = "git_diff"
    staged: bool = False
    path: Optional[Any] = None
    base_dir: str = "."
    timeout: float = 60.0

    name: str = "git_diff"

    def execute(self, context: "Context") -> None:
        args = ["diff"]
        if self.staged:
            args.append("--cached")
        path = self.path
        if isinstance(path, Expr):
            path = path.eval(context)
        if path:
            args += ["--", str(path)]
        result = git(args, cwd=self.base_dir, timeout=self.timeout)
        context[self.output_key] = result["stdout"] if result["passed"] else ""
        context[f"{self.output_key}_result"] = result
        context.trace.append(
            f"{self.name}:{'pass' if result['passed'] else 'fail'}"
        )

    def describe(self) -> str:
        extra = ", staged" if self.staged else ""
        return f"GitDiff({self.path or '.'}{extra} -> {self.output_key})"


@dataclass
class GitCommit(Step):
    '''先 `git add -A`（可选），再 `git commit -m <message>`。

        GitCommit(message=Var("commit_message"))

    `add_all=False` 时只提交已经 `git add` 过的内容。结果写进 Context
    （`commit_result`），提交失败**不抛异常** —— 让工作流自己决定要不要停。
    '''

    message: Any
    add_all: bool = True
    output_key: str = "commit_result"
    base_dir: str = "."
    timeout: float = 120.0

    name: str = "git_commit"

    def execute(self, context: "Context") -> None:
        message = self.message
        if isinstance(message, Expr):
            message = message.eval(context)
        text = "" if message is None else str(message).strip()
        if not text:
            raise CommandError("GitCommit 的提交信息是空的")

        result: Optional[CommandResult] = None
        if self.add_all:
            result = git(["add", "-A"], cwd=self.base_dir, timeout=self.timeout)
            if not result["passed"]:
                context[self.output_key] = result
                context.trace.append(f"{self.name}:add-failed")
                return

        commit = git(
            ["commit", "-m", text], cwd=self.base_dir, timeout=self.timeout
        )
        if result is not None:
            commit = {
                **commit,
                "output": f"{result['output']}\n{commit['output']}".strip(),
            }
        context[self.output_key] = commit
        context.trace.append(
            f"{self.name}:{'pass' if commit['passed'] else 'fail'}"
        )

    def describe(self) -> str:
        return f"GitCommit({self.output_key})"
