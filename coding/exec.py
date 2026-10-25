# coding/exec.py
'''跑命令：`RunCommand` / `RunTests` / `LintStep`。

写进 Context 的是一份**结构化结果**，不是一个字符串：

    {
      "command": "python -m unittest -q",
      "returncode": 0,
      "passed": True,          # returncode == 0 且没超时
      "timed_out": False,
      "duration_s": 1.23,
      "output": "…stdout + stderr…",
      "stdout": "...",
      "stderr": "..."
    }

这样条件才好写：`If(Var("test_result")["passed"] == False).Then(...)`。
（DSL 里没有 `Var("x").passed` 这种属性访问，请用 `["passed"]` 或 `.attr("passed")`。）

**这不是沙箱**：命令本身能干什么取决于操作系统权限。`allowlist=` 只是挡住
「模型随手拼一条 rm -rf」，真要隔离请用容器。
'''
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from promptkit.coding.errors import CommandError
from promptkit.coding.sandbox import Sandbox
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注
    from promptkit.core.context import Context

__all__ = ["run_command", "CommandResult", "RunCommand", "RunTests", "LintStep"]

#: 命令结果的形状
CommandResult = Dict[str, Any]

#: 默认最多回传多少字符（超了截断，防止测试日志把上下文吃光）
DEFAULT_MAX_OUTPUT = 8000


def _decode(raw: bytes) -> str:
    '''utf-8 -> 本机编码 -> gbk 的兜底解码（Windows 上同一个命令可能两种都吐）。'''
    if not raw:
        return ""
    for encoding in ("utf-8", settings.encoding, "gbk", "latin-1"):
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit] + f"\n…（输出已截断：共 {len(text)} 字符）"
    return text


def run_command(
    command: str,
    *,
    cwd: Optional[str] = None,
    timeout: float = 120.0,
    shell: bool = True,
    sandbox: Optional[Sandbox] = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT,
) -> CommandResult:
    '''跑一条命令，返回结构化结果。**不抛异常**（失败也写进结果里）。

    抛异常的只有「配置层面就不该跑」：allowlist 不放行（`CommandError`）、
    cwd 越界（`SandboxError`）。
    '''
    text = (command or "").strip()
    if not text:
        raise CommandError("命令是空的")

    workdir = None
    if sandbox is not None:
        sandbox.check_command(text)          # allowlist + 元字符
        workdir = sandbox.resolve(cwd or ".") if cwd else sandbox.base
        if not os.path.isdir(workdir):
            raise CommandError(f"cwd 不是目录：{workdir!r}")
    elif cwd:
        workdir = os.path.abspath(os.path.expanduser(str(cwd)))

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            text,
            shell=shell,
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        partial = _decode(exc.stdout or b"") + _decode(exc.stderr or b"")
        return {
            "command": text,
            "returncode": None,
            "passed": False,
            "timed_out": True,
            "duration_s": round(duration, 2),
            "output": _truncate(
                f"命令超时（超过 {timeout}s 仍未结束）\n{partial}".strip(),
                max_output_chars,
            ),
            "stdout": _decode(exc.stdout or b""),
            "stderr": _decode(exc.stderr or b""),
        }
    except OSError as exc:
        return {
            "command": text,
            "returncode": None,
            "passed": False,
            "timed_out": False,
            "duration_s": round(time.perf_counter() - started, 2),
            "output": f"命令起不来：{type(exc).__name__}: {exc}",
            "stdout": "",
            "stderr": "",
        }

    stdout = _decode(completed.stdout)
    stderr = _decode(completed.stderr)
    parts = [part for part in (stdout.strip(), stderr.strip()) if part]
    output = "\n".join(parts)
    if not parts:
        output = f"（命令执行了，但没有任何输出；退出码 {completed.returncode}）"
    elif completed.returncode != 0:
        output += f"\n[退出码 {completed.returncode}]"

    return {
        "command": text,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "timed_out": False,
        "duration_s": round(time.perf_counter() - started, 2),
        "output": _truncate(output, max_output_chars),
        "stdout": stdout,
        "stderr": stderr,
    }


# ============================ 步骤 ============================
@dataclass
class RunCommand(Step):
    '''执行一条命令，结果写进 Context。

        RunCommand("python -m unittest -q", output_key="test_result",
                   allowlist=["python"], timeout=300)

    `allowlist` 省略时**不限制**（自己写的命令自己负责）；给了就只放行
    这些可执行文件，并拒绝带 shell 元字符的命令（`; | & > $` 之类）。
    '''

    command: Any
    output_key: str = "command_result"
    cwd: Optional[Any] = None
    timeout: float = 120.0
    allowlist: Optional[Sequence[str]] = None
    base_dir: str = "."
    shell: bool = True
    max_output_chars: int = DEFAULT_MAX_OUTPUT
    name: str = "run_command"

    def __post_init__(self) -> None:
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError(f"timeout 必须是正数，得到 {self.timeout!r}")
        if not self.output_key:
            raise ValueError("output_key 不能为空")

    def _render(self, value: Any, context: "Context") -> Any:
        return value.eval(context) if isinstance(value, Expr) else value

    def execute(self, context: "Context") -> None:
        command = self._render(self.command, context)
        if command is None:
            raise CommandError("命令是空的")
        cwd = self._render(self.cwd, context) if self.cwd is not None else None
        sandbox = Sandbox(
            self.base_dir,
            allow_write=False,
            allow_commands=self.allowlist,
        )
        result = run_command(
            str(command),
            cwd=str(cwd) if cwd is not None else None,
            timeout=float(self.timeout),
            shell=self.shell,
            sandbox=sandbox,
            max_output_chars=self.max_output_chars,
        )
        context[self.output_key] = result
        context.trace.append(
            f"{self.name}:{'pass' if result['passed'] else 'fail'}"
        )

    def describe(self) -> str:
        command = self.command if isinstance(self.command, str) else repr(self.command)
        return f"{self.name}({command!r} -> {self.output_key})"


@dataclass
class RunTests(RunCommand):
    '''跑测试。和 `RunCommand` 唯一的区别是默认值和 `passed` 语义更明确：

        RunTests("python -m pytest -q")          # 取 output_key="test_result"

    默认超时给得宽（600s）：测试本来就慢，30 秒超时会让人误以为测试挂了。
    '''

    output_key: str = "test_result"
    timeout: float = 600.0
    name: str = "run_tests"


@dataclass
class LintStep(RunCommand):
    '''跑静态检查（ruff / eslint / mypy …），结果写 `lint_result`。

        LintStep("ruff check src/", allowlist=["ruff"])

    注意 `passed` 只看退出码：lint 工具通常「有问题就非 0」，和测试同构。
    '''

    output_key: str = "lint_result"
    timeout: float = 300.0
    name: str = "lint"
