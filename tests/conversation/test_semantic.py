'''语义 Step 测试：Snapshot / Restore / Fork / Branch / Branches / Merge。'''
import pytest

from promptkit.conversation.config import RewindConfig
from promptkit.conversation.exceptions import (
    BranchLimitExceeded,
    RewindBeyondCompression,
    RewindBeyondLimit,
    SessionError,
    SnapshotReadonly,
)
from promptkit.conversation.semantic import (
    BRANCH_COUNT_KEY,
    Branch,
    Branches,
    Fork,
    Merge,
    Restore,
    Snapshot,
    branch_key,
)
from promptkit.conversation.session import Session
from promptkit.conversation.steps import SESSION_KEY, session_from_context
from promptkit.core.context import Context
from promptkit.core.steps.interface import Step


class MarkStep(Step):
    '''在分支里写一个变量，并可选地往分支会话追加一条消息。'''

    name = "mark"

    def __init__(self, key: str, value: object, *, message: str | None = None) -> None:
        self.key = key
        self.value = value
        self.message = message

    def execute(self, context: Context) -> None:
        if self.message is not None:
            session_from_context(context).append_user(self.message)
        context[self.key] = self.value


class TouchListStep(Step):
    '''改一个父 Context 里的可变对象，用来验证分支没有污染父 Context。'''

    name = "touch"

    def __init__(self, key: str) -> None:
        self.key = key

    def execute(self, context: Context) -> None:
        value = context[self.key]
        value.append("branch")


def make_context(session: Session | None = None, **data: object) -> Context:
    active = session if session is not None else Session()
    return Context(data={SESSION_KEY: active, **data})


def test_snapshot_then_restore():
    session = Session()
    context = make_context(session)
    session.append_user("a")
    Snapshot("base").execute(context)
    session.append_user("b")
    Restore("base").execute(context)

    assert [m.content for m in session.messages] == ["a"]
    assert context.trace[-1]["snapshot"] == "base"


def test_restore_beyond_compression_raises():
    session = Session()
    context = make_context(session)
    Snapshot("base").execute(context)
    session.append_user("a")
    session.set_compressed_upto(1)

    with pytest.raises(RewindBeyondCompression):
        Restore("base").execute(context)


def test_restore_into_summary_allowed_by_config():
    session = Session()
    context = make_context(session)
    Snapshot("base").execute(context)
    session.append_user("a")
    session.set_compressed_upto(1)

    Restore(
        "base", config=RewindConfig(allow_rewind_into_summary=True)
    ).execute(context)
    assert session.compressed_upto == 0


def test_restore_over_turn_limit_raises_by_default():
    session = Session()
    context = make_context(session)
    Snapshot("base").execute(context)
    for index in range(3):
        session.append_user(f"q{index}")
        session.append_assistant(f"a{index}")

    with pytest.raises(RewindBeyondLimit):
        Restore("base", config=RewindConfig(max_rewind_turns=1)).execute(context)


def test_restore_over_turn_limit_can_clamp():
    session = Session()
    context = make_context(session)
    Snapshot("base").execute(context)
    for index in range(3):
        session.append_user(f"q{index}")
        session.append_assistant(f"a{index}")

    config = RewindConfig(max_rewind_turns=1, on_beyond_limit="clamp")
    Restore("base", config=config).execute(context)

    # 只回退到允许的最早位置：保留最后一个被允许拍掉的轮次之前的内容
    assert [m.content for m in session.messages] == ["q0", "a0", "q1"]


def test_fork_requires_non_readonly_snapshot():
    session = Session()
    context = make_context(session)
    Snapshot("base", readonly=True).execute(context)
    with pytest.raises(SnapshotReadonly):
        Fork("child", "base").execute(context)


def test_fork_creates_isolated_session():
    session = Session()
    context = make_context(session)
    session.append_user("a")
    Snapshot("base").execute(context)
    session.append_user("b")

    Fork("child", "base").execute(context)
    child = context["_fork_child"]
    assert isinstance(child, Session)
    assert [m.content for m in child.messages] == ["a"]
    assert child is not session
    assert child.session_id != session.session_id
    assert [m.content for m in session.messages] == ["a", "b"]


def test_fork_checks_compression_boundary():
    session = Session()
    context = make_context(session)
    Snapshot("base").execute(context)
    session.append_user("a")
    session.set_compressed_upto(1)
    with pytest.raises(RewindBeyondCompression):
        Fork("child", "base").execute(context)


def test_branch_isolation_and_summary_key():
    session = Session()
    context = make_context(session, shared=["parent"])
    branch = Branch("b1", MarkStep("answer", "yes", message="分支问题"), TouchListStep("shared"))
    branch.execute(context)

    branch_ctx = context[branch_key("b1")]
    assert isinstance(branch_ctx, Context)
    assert branch_ctx["answer"] == "yes"
    assert context["shared"] == ["parent"]  # 父 Context 没被改
    assert session.length == 0  # 父会话没被改
    assert branch_ctx["_session"].length == 1
    assert branch_ctx["_session"] is not session
    assert context[BRANCH_COUNT_KEY] == 1


def test_branch_from_snapshot():
    session = Session()
    context = make_context(session)
    session.append_user("base")
    Snapshot("base").execute(context)
    session.append_user("later")

    Branch("b1", MarkStep("answer", "ok"), snapshot_name="base").execute(context)
    branch_ctx = context[branch_key("b1")]
    assert [m.content for m in branch_ctx["_session"].messages] == ["base"]
    assert [m.content for m in session.messages] == ["base", "later"]


def test_branch_limit_raises():
    context = make_context()
    config = RewindConfig(max_branches=1)
    context["_rewind_config"] = config
    # 第一个分支成功，第二个命中上限
    with pytest.raises(BranchLimitExceeded):
        Branches(Branch("a"), Branch("b")).execute(context)
    assert context[BRANCH_COUNT_KEY] == 1
    with pytest.raises(BranchLimitExceeded):
        Branch("c").execute(context)


def test_branches_runs_all_and_merge_collects():
    context = make_context()
    Branches(
        Branch("a", MarkStep("answer", "A")),
        Branch("b", MarkStep("answer", "B")),
    ).execute(context)
    Merge("a", "b").execute(context)

    merged = context["merged"]
    assert merged["a"]["answer"] == "A"
    assert merged["b"]["answer"] == "B"
    assert context.trace[-1]["step"] == "merge"


def test_merge_missing_branch_raises():
    context = make_context()
    with pytest.raises(SessionError):
        Merge("missing").execute(context)


def test_missing_snapshot_raises():
    context = make_context()
    with pytest.raises(SessionError):
        Restore("missing").execute(context)


def test_branches_requires_branch_instances():
    with pytest.raises(TypeError):
        Branches(MarkStep("answer", "x"))  # type: ignore[arg-type]
