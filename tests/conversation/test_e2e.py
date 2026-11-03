'''端到端测试：Snapshot -> Branches -> Merge 完整流程（FakeClient，无真实 API）。'''
from .fake_client import make_agent

from promptkit.conversation.semantic import (
    Branch,
    Branches,
    Merge,
    Snapshot,
    branch_key,
)
from promptkit.conversation.steps import session_from_context
from promptkit.core.context import Context
from promptkit.core.display_sys.render import Verbosity
from promptkit.core.steps.interface import Step
from promptkit.core.steps.workflow import Workflow


class BranchAskStep(Step):
    '''在分支会话里追加问题、调用 FakeClient、再把回答写回分支会话。'''

    name = "branch_ask"

    def __init__(self, prompt: str, *, output: str = "answer") -> None:
        self.prompt = prompt
        self.output = output

    def execute(self, context: Context) -> None:
        agent = context.data["agent"]
        session = session_from_context(context)
        session.append_user(self.prompt)
        response = agent.client.chat.completions.create(
            model="fake", messages=[m.to_dict() for m in session.messages]
        )
        reply = response.choices[0].message.content
        session.append_assistant(reply)
        context[self.output] = reply


def test_snapshot_branches_merge_end_to_end():
    agent = make_agent(["简洁回答", "详细回答"])
    workflow = Workflow(
        Snapshot("base"),
        Branches(
            Branch(
                "concise",
                BranchAskStep("请简洁地回答"),
                snapshot_name="base",
            ),
            Branch(
                "detailed",
                BranchAskStep("请详细地回答"),
                snapshot_name="base",
            ),
        ),
        Merge("concise", "detailed"),
    )

    context = workflow.run(
        agent=agent,
        _session=agent.session,
        verbose=Verbosity.SILENT,
        topic="测试",
    )

    merged = context["merged"]
    assert merged["concise"]["answer"] == "简洁回答"
    assert merged["detailed"]["answer"] == "详细回答"

    concise_ctx = context[branch_key("concise")]
    detailed_ctx = context[branch_key("detailed")]
    assert concise_ctx["_session"] is not detailed_ctx["_session"]
    assert [m.content for m in concise_ctx["_session"].messages] == [
        "请简洁地回答",
        "简洁回答",
    ]
    assert [m.content for m in detailed_ctx["_session"].messages] == [
        "请详细地回答",
        "详细回答",
    ]

    # 父会话没被任何分支污染
    assert agent.session.length == 0
    # 父 Context 的用户变量也没被改
    assert context["topic"] == "测试"


def test_workflow_restore_flow():
    agent = make_agent([])
    workflow = Workflow(
        Snapshot("base"),
        Branches(),
        Merge(),
    )

    # 先往父会话里放一条消息，再跑快照 -> 空分支 -> 空合并
    agent.session.append_user("起点")
    context = workflow.run(
        agent=agent,
        _session=agent.session,
        verbose=Verbosity.SILENT,
    )

    assert context["_snapshots"]["base"].position == 1
    assert context["merged"] == {}
    assert [m.content for m in agent.session.messages] == ["起点"]
