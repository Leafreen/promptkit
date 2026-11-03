# conversation/semantic.py
'''语义糖 Step：快照、回退、分叉、分支、合并。

这些 Step **只调用 Session 的公开 CRUD 方法**（append / replace_all /
set_compressed_upto / clone ……），不直接读写 `session._messages`。
分支执行只写独立的 `branch_ctx`，结果汇总在 `context[f"_branch_{name}"]`，
父 Context 不会被分支里的步骤污染。
'''
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

from promptkit.conversation.config import RewindConfig
from promptkit.conversation.exceptions import (
    BranchLimitExceeded,
    RewindBeyondCompression,
    RewindBeyondLimit,
    SessionError,
    SnapshotReadonly,
)
from promptkit.conversation.message import Message
from promptkit.conversation.session import Session
from promptkit.conversation.steps import SESSION_KEY, session_from_context
from promptkit.core.context import Context
from promptkit.core.steps.interface import Step, StepLike, coerce_steps, execute_step

__all__ = [
    "BRANCH_COUNT_KEY",
    "FORK_PREFIX",
    "REWIND_CONFIG_KEY",
    "SNAPSHOTS_KEY",
    "Branch",
    "Branches",
    "Fork",
    "Merge",
    "Restore",
    "Snapshot",
    "SnapshotRecord",
    "branch_key",
]

#: Context 里存放 `{name: SnapshotRecord}` 的键
SNAPSHOTS_KEY = "_snapshots"
#: Context 里可以预置一份 RewindConfig
REWIND_CONFIG_KEY = "_rewind_config"
#: Context 里记录本工作流已经创建了多少分支
BRANCH_COUNT_KEY = "_branch_count"
#: Fork 出来的独立会话存放在 `_fork_{name}`
FORK_PREFIX = "_fork_"


def branch_key(name: str) -> str:
    '''分支结果在父 Context 里的键：`_branch_{name}`。'''
    return f"_branch_{name}"


@dataclass(frozen=True)
class SnapshotRecord:
    '''一次快照的不可变记录。'''

    name: str
    messages: tuple[Message, ...]
    position: int
    compressed_upto: int
    readonly: bool = False


class Snapshot(Step):
    '''把当前会话状态存进 `context["_snapshots"][name]`。'''

    name = "snapshot"

    def __init__(self, snapshot_name: str, *, readonly: bool = False) -> None:
        if not snapshot_name:
            raise ValueError("Snapshot 需要非空的快照名")
        self.snapshot_name = snapshot_name
        self.readonly = readonly

    def execute(self, context: Context) -> None:
        '''记录当前会话的消息、位置与压缩边界。'''
        session = session_from_context(context)
        records = _snapshots(context)
        messages = copy.deepcopy(session.messages)
        records[self.snapshot_name] = SnapshotRecord(
            name=self.snapshot_name,
            messages=tuple(messages),
            position=session.length,
            compressed_upto=session.compressed_upto,
            readonly=self.readonly,
        )
        context.trace.append(
            {
                "step": self.name,
                "snapshot": self.snapshot_name,
                "messages": session.length,
                "readonly": self.readonly,
            }
        )

    def describe(self) -> str:
        return f"Snapshot({self.snapshot_name!r})"


class Restore(Step):
    '''把当前会话回退到某个快照（会检查压缩边界与轮次上限）。'''

    name = "restore"

    def __init__(
        self, snapshot_name: str, *, config: RewindConfig | None = None
    ) -> None:
        self.snapshot_name = snapshot_name
        self.config = config

    def execute(self, context: Context) -> None:
        '''用快照内容整体替换当前会话，并恢复压缩边界。'''
        session = session_from_context(context)
        record = _record(context, self.snapshot_name)
        restored, compressed = _resolve_rewind(
            session, record, _config(context, self.config)
        )
        session.replace_all(restored)
        session.set_compressed_upto(compressed)
        context.trace.append(
            {
                "step": self.name,
                "snapshot": self.snapshot_name,
                "messages": session.length,
            }
        )

    def describe(self) -> str:
        return f"Restore({self.snapshot_name!r})"


class Fork(Step):
    '''从快照派生一个独立会话，存进 `context[f"_fork_{name}"]`。

    父会话不受影响；派生会话用 `Session.clone()`（deepcopy）保证完全隔离。
    '''

    name = "fork"

    def __init__(
        self,
        fork_name: str,
        snapshot_name: str,
        *,
        config: RewindConfig | None = None,
    ) -> None:
        self.fork_name = fork_name
        self.snapshot_name = snapshot_name
        self.config = config

    def execute(self, context: Context) -> None:
        '''检查只读 / 压缩边界后，克隆父会话并回退到快照。'''
        record = _record(context, self.snapshot_name)
        if record.readonly:
            raise SnapshotReadonly(
                f"快照 {self.snapshot_name!r} 是只读的，不能 Fork"
            )
        parent = session_from_context(context)
        restored, compressed = _resolve_rewind(
            parent, record, _config(context, self.config)
        )
        child = parent.clone()
        child.replace_all(restored)
        child.set_compressed_upto(compressed)
        context[f"{FORK_PREFIX}{self.fork_name}"] = child
        context.trace.append(
            {
                "step": self.name,
                "fork": self.fork_name,
                "snapshot": self.snapshot_name,
                "messages": child.length,
            }
        )

    def describe(self) -> str:
        return f"Fork({self.fork_name!r}, {self.snapshot_name!r})"


class Branch(Step):
    '''从当前会话（或指定快照）克隆出一个分支并执行一组步骤。

    分支里的所有执行只发生在独立的 `branch_ctx` 上；结束后整个 Context
    放进父 Context 的 `_branch_{name}` 键，父会话 / 父 Context 不被改动。
    '''

    name = "branch"

    def __init__(
        self,
        branch_name: str,
        *steps: StepLike,
        snapshot_name: str | None = None,
        config: RewindConfig | None = None,
    ) -> None:
        if not branch_name:
            raise ValueError("Branch 需要非空的分支名")
        self.branch_name = branch_name
        self.steps = coerce_steps(*steps)
        self.snapshot_name = snapshot_name
        self.config = config

    def execute(self, context: Context) -> None:
        '''克隆会话、跑分支步骤、把结果汇总进父 Context。'''
        cfg = _config(context, self.config)
        used = _branch_count(context)
        if used >= cfg.max_branches:
            raise BranchLimitExceeded(
                f"分支数量超过上限 {cfg.max_branches}（当前已创建 {used} 个）"
            )

        parent = session_from_context(context)
        branch_session = parent.clone()
        if self.snapshot_name is not None:
            record = _record(context, self.snapshot_name)
            restored, compressed = _resolve_rewind(parent, record, cfg)
            branch_session.replace_all(restored)
            branch_session.set_compressed_upto(compressed)

        branch_ctx = _make_branch_context(context, branch_session, cfg)
        for step in self.steps:
            execute_step(step, branch_ctx)

        context[branch_key(self.branch_name)] = branch_ctx
        context[BRANCH_COUNT_KEY] = used + 1
        context.trace.append(
            {
                "step": self.name,
                "branch": self.branch_name,
                "messages": branch_session.length,
                "snapshot": self.snapshot_name,
            }
        )

    def describe(self) -> str:
        if self.snapshot_name is None:
            return f"Branch({self.branch_name!r})"
        return f"Branch({self.branch_name!r}, snapshot={self.snapshot_name!r})"


class Branches(Step):
    '''按顺序执行多个 `Branch`（`max_concurrency` 仅保留接口，这里永远串行）。'''

    name = "branches"

    def __init__(self, *branches: Branch) -> None:
        for item in branches:
            if not isinstance(item, Branch):
                raise TypeError(
                    f"Branches 只接受 Branch，得到 {type(item).__name__}"
                )
        self.branches: list[Branch] = list(branches)

    def execute(self, context: Context) -> None:
        '''逐个执行分支；每个分支自身负责结果汇总与上限检查。'''
        for branch in self.branches:
            execute_step(branch, context)
        context.trace.append(
            {
                "step": self.name,
                "branches": [branch.branch_name for branch in self.branches],
            }
        )

    def describe(self) -> str:
        names = ", ".join(branch.branch_name for branch in self.branches)
        return f"Branches({names})"


class Merge(Step):
    '''把若干分支的公开结果汇总到父 Context 的一个键里。'''

    name = "merge"

    def __init__(self, *branch_names: str, output: str = "merged") -> None:
        self.branch_names = list(branch_names)
        self.output = output

    def execute(self, context: Context) -> None:
        '''按分支名收集非内部键的变量，写入 `context[output]`。'''
        result: dict[str, dict[str, object]] = {}
        for name in self.branch_names:
            value = context.data.get(branch_key(name))
            if not isinstance(value, Context):
                raise SessionError(
                    f"没有找到分支 {name!r} 的结果（先执行 Branch 步骤）"
                )
            result[name] = _public_data(value)
        context[self.output] = result
        context.trace.append(
            {"step": self.name, "branches": list(self.branch_names), "output": self.output}
        )

    def describe(self) -> str:
        names = ", ".join(self.branch_names)
        return f"Merge({names}, output={self.output!r})"


# ---------------- 内部 ----------------
def _snapshots(context: Context) -> dict[str, SnapshotRecord]:
    '''取出（必要时创建）当前 Context 的快照表。'''
    value = context.data.get(SNAPSHOTS_KEY)
    if isinstance(value, dict):
        return cast(dict[str, SnapshotRecord], value)
    created: dict[str, SnapshotRecord] = {}
    context[SNAPSHOTS_KEY] = created
    return created


def _record(context: Context, name: str) -> SnapshotRecord:
    '''按名字取快照；不存在时给出明确的 SessionError。'''
    records = _snapshots(context)
    record = records.get(name)
    if record is None:
        raise SessionError(f"没有名为 {name!r} 的快照（先执行 Snapshot 步骤）")
    return record


def _config(context: Context, explicit: RewindConfig | None) -> RewindConfig:
    '''显式配置 -> Context 预置配置 -> 默认配置。'''
    if explicit is not None:
        return explicit
    stored = context.data.get(REWIND_CONFIG_KEY)
    if isinstance(stored, RewindConfig):
        return stored
    return RewindConfig()


def _branch_count(context: Context) -> int:
    '''当前工作流已创建的分支数量。'''
    value = context.data.get(BRANCH_COUNT_KEY, 0)
    return value if isinstance(value, int) else 0


def _resolve_rewind(
    session: Session, record: SnapshotRecord, config: RewindConfig
) -> tuple[list[Message], int]:
    '''计算「回退到快照」的最终消息列表与压缩边界。

    规则：

    - 快照位置落在压缩区间内、且不允许回退进摘要 -> `RewindBeyondCompression`
    - 被回退的 user 消息轮次超过上限：
      `error` -> `RewindBeyondLimit`；`clamp` -> 保留多出来的最近轮次
    '''
    if (
        record.position < session.compressed_upto
        and not config.allow_rewind_into_summary
    ):
        raise RewindBeyondCompression(
            f"快照 {record.name!r} 位于压缩区间（位置 {record.position}"
            f" < 边界 {session.compressed_upto}），且 allow_rewind_into_summary=False"
        )

    current = session.messages
    removed = current[record.position:]
    turns = sum(1 for message in removed if message.role == "user")
    if turns <= config.max_rewind_turns:
        restored = list(record.messages)
    elif config.on_beyond_limit == "error":
        raise RewindBeyondLimit(
            f"回退 {turns} 轮超过上限 {config.max_rewind_turns}；"
            "把 on_beyond_limit 设为 'clamp' 可以只回退到允许的位置"
        )
    else:
        restored = _clamp_restore(record, current, turns, config.max_rewind_turns)

    return copy.deepcopy(restored), record.compressed_upto


def _clamp_restore(
    record: SnapshotRecord,
    current: list[Message],
    turns: int,
    limit: int,
) -> list[Message]:
    '''`clamp` 策略：从快照位置往后走，直到只剩允许的轮次。'''
    start = record.position
    excess = turns - limit
    while excess > 0 and start < len(current):
        if current[start].role == "user":
            excess -= 1
        start += 1
    return [*record.messages, *current[record.position:start]]


def _make_branch_context(
    parent: Context, session: Session, config: RewindConfig
) -> Context:
    '''构造完全独立的 branch_ctx：复制公开变量，换掉会话。'''
    data: dict[object, object] = {}
    for key, value in parent.data.items():
        if key == "agent":
            data[key] = value  # agent 有意共享：分支复用同一个 client
        elif isinstance(key, str) and key.startswith("_"):
            continue
        else:
            data[key] = copy.deepcopy(value)
    data[SESSION_KEY] = session
    data[REWIND_CONFIG_KEY] = config
    return Context(data=data, trace=[], types=dict(parent.types))


def _public_data(context: Context) -> dict[str, object]:
    '''分支 Context 里可以对外汇总的部分（去掉 `_` 开头的内部键与 agent）。'''
    return {
        str(key): copy.deepcopy(value)
        for key, value in context.data.items()
        if isinstance(key, str) and not key.startswith("_") and key != "agent"
    }


# 本文件核心职责：实现 Snapshot / Restore / Fork / Branch / Branches / Merge 语义糖 Step。
