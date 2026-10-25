# coding/__init__.py
'''编码任务：读文件 → 生成补丁 → 应用 → 跑测试 → 修。

对话任务里模型产出的是**文本**；编码任务里模型产出的是**要执行的变更**，
所以这一层的重点不是「提示词」，而是那条闭环和它的校验：

    ReadFileStep ──> CodingStep(校验: 结构 + 干跑) ──> ApplyDiff ──> RunTests
                          ^                                            │
                          └──────── FixLoop(带错误重试) <───────────────┘

三块能力：

- **IO 原语**（`coding.io`）：`ReadFileStep` / `WriteFileStep` / `ApplyDiff` /
  `ApplyChanges` / `ListDir` / `SearchCode`
- **执行原语**（`coding.exec` / `coding.git`）：`RunCommand` / `RunTests` /
  `LintStep` / `GitDiff` / `GitCommit`
- **生成与修复**（`coding.step` / `coding.loop`）：`CodingStep` / `FixLoop`

再加一层**安全边界**（`coding.sandbox.Sandbox`）：相对路径、`base_dir` 之内、
敏感文件黑名单、写权限、命令 allowlist、审计日志。

    from promptkit.coding import (ApplyDiff, CodingStep, FixLoop, ReadFileStep,
                                  RunTests)

    wf = Workflow(
        ReadFileStep("src/calc.py", output_key="source"),
        FixLoop(
            [
                CodingStep(
                    task=Var("task"),
                    context={"src/calc.py": Var("source")},
                    error=Var("test_result"),
                    path="src/calc.py",
                ),
                ApplyDiff("src/calc.py", diff=Var("patch")),
                RunTests("python -m unittest -q", allowlist=["python"]),
            ],
            until=Var("test_result")["passed"],
            max_rounds=3,
        ),
    )

注意：这些步骤**不从 `promptkit.core` 导出**（core 是底层，不能反过来依赖
coding）。显式 `from promptkit.coding import ...` 即可，也避免了和
`core.ReadFile`（惰性表达式）、`default_tools.WriteFile`（工具）撞名。
'''
from promptkit.coding.diff import (
    FilePatch,
    Hunk,
    apply_hunks,
    apply_patch_text,
    parse_patch,
    strip_code_fence,
    validate_patch,
)
from promptkit.coding.errors import (
    CodeFormatError,
    CodingError,
    CommandError,
    DiffError,
    SandboxError,
)
from promptkit.coding.exec import (
    CommandResult,
    LintStep,
    RunCommand,
    RunTests,
    run_command,
)
from promptkit.coding.formats import (
    OUTPUT_FORMATS,
    ChangesOutput,
    CodeOutput,
    DiffOutput,
    FullFileOutput,
    get_output,
    validate_changes,
)
from promptkit.coding.git import GitCommit, GitDiff, git
from promptkit.coding.io import (
    ApplyChanges,
    ApplyDiff,
    ListDir,
    ReadFileStep,
    SearchCode,
    WriteFileStep,
    apply_patch,
    format_hits,
)
from promptkit.coding.loop import FixLoop
from promptkit.coding.sandbox import CODING_LOGGER, Sandbox
from promptkit.coding.step import CodingStep
from promptkit.coding.tools import (
    ApplyDiffTool,
    ListDirTool,
    ReadFileTool,
    RunCommandTool,
    RunTestsTool,
    SearchCodeTool,
    WriteFileTool,
    default_coding_tools,
)

__all__ = [
    # 异常
    "CodingError",
    "CodeFormatError",
    "DiffError",
    "SandboxError",
    "CommandError",
    # 沙箱 / 审计
    "Sandbox",
    "CODING_LOGGER",
    # diff
    "Hunk",
    "FilePatch",
    "parse_patch",
    "validate_patch",
    "apply_hunks",
    "apply_patch_text",
    "strip_code_fence",
    # 输出格式
    "CodeOutput",
    "DiffOutput",
    "FullFileOutput",
    "ChangesOutput",
    "get_output",
    "validate_changes",
    "OUTPUT_FORMATS",
    # 生成 / 修复
    "CodingStep",
    "FixLoop",
    # IO 原语
    "ReadFileStep",
    "WriteFileStep",
    "ApplyDiff",
    "ApplyChanges",
    "ListDir",
    "SearchCode",
    "apply_patch",
    "format_hits",
    # 执行 / 版本控制
    "run_command",
    "CommandResult",
    "RunCommand",
    "RunTests",
    "LintStep",
    "git",
    "GitDiff",
    "GitCommit",
    # 工具
    "ReadFileTool",
    "WriteFileTool",
    "ApplyDiffTool",
    "ListDirTool",
    "SearchCodeTool",
    "RunCommandTool",
    "RunTestsTool",
    "default_coding_tools",
]
