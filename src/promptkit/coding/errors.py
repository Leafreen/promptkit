# coding/errors.py
'''编码任务用到的异常。

分开定义是为了让调用方能区分「模型输出不能执行」和「文件/沙箱不允许」：

    CodingError
    ├── CodeFormatError   模型给的东西解析不了 / 校验不过（值得重试）
    ├── DiffError         补丁结构或上下文对不上（值得带错误重试）
    ├── SandboxError      路径越界、写权限、敏感文件（**不该**重试，是配置问题）
    └── CommandError      命令被拒绝 / 起不来（同上）
'''
from __future__ import annotations

__all__ = ["CodingError", "CodeFormatError", "DiffError", "SandboxError", "CommandError"]


class CodingError(RuntimeError):
    '''编码任务相关错误的基类。'''


class CodeFormatError(CodingError):
    '''模型输出不是可执行的代码变更（解析失败 / 语法错 / 结构不对）。

    这是**值得重试**的错误：把消息原样塞回下一次提示词，模型通常能改对。
    '''


class DiffError(CodingError):
    '''补丁本身或它的应用过程有问题（hunk 行数不对、上下文找不到、路径不合法）。'''


class SandboxError(CodingError, PermissionError):
    '''路径越界 / 没有写权限 / 碰到敏感文件。

    也继承 `PermissionError`：调用方想统一按「没权限」处理时也能捕获到。
    '''

    def __str__(self) -> str:  # PermissionError 的 str 行为和普通异常一致，显式写清
        return Exception.__str__(self)


class CommandError(CodingError):
    '''命令被 allowlist 拒绝，或者根本起不来。'''
