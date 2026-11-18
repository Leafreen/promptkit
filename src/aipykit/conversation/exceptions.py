# conversation/exceptions.py
'''对话状态层的异常类型。

全部继承自 `SessionError`，调用方可以只捕获一个基类；需要区分原因时
再按具体类型分支。
'''
from __future__ import annotations

__all__ = [
    "BranchLimitExceeded",
    "RewindBeyondCompression",
    "RewindBeyondLimit",
    "SessionError",
    "SnapshotReadonly",
]


class SessionError(Exception):
    '''会话 / 快照 / 分支操作失败的基类。'''


class RewindBeyondCompression(SessionError):
    '''回退目标落在压缩区间内，且配置不允许回退进摘要。'''


class RewindBeyondLimit(SessionError):
    '''回退的轮次超过 `RewindConfig.max_rewind_turns` 且策略为 `error`。'''


class SnapshotReadonly(SessionError):
    '''对只读快照执行 Fork（只读快照只能查看，不能派生会话）。'''


class BranchLimitExceeded(SessionError):
    '''分支数量超过 `RewindConfig.max_branches` 上限。'''


# 本文件核心职责：声明对话状态层全部自定义异常，统一挂在 SessionError 之下。
