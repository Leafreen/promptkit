'''对话状态层 · 语义糖分叉合并示例。

完整走一遍 Snapshot -> Branches -> Merge：

- 先对当前对话拍一个快照；
- 两条 Branch 各自从快照克隆出**独立会话**，互不影响；
- Merge 把两条分支的公开结果汇总到 `context["comparison"]`。

父会话不会被任何分支改动，分支结果只会出现在 `_branch_{name}` 键上。
不调用任何模型 API，`_session` 直接用内存里的 `Session`。

运行：
    python examples/conversation_branching.py
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.conversation import (
    Branch,
    Branches,
    Merge,
    Message,
    Session,
    SessionAppend,
    Snapshot,
)
from promptkit.core import FunctionStep, Verbosity, Workflow


def build_session() -> Session:
    '''准备一段已经很进入状态的对话。'''
    return Session(
        [
            Message(role="system", content="你是机票客服助手。", id="m-sys"),
            Message(role="user", content="我想改签明天的机票", id="m-q1"),
            Message(role="assistant", content="请提供订单号。", id="m-a1"),
        ]
    )


# 快照在分叉前拍下；两条分支都从它克隆会话
branch_workflow = Workflow(
    Snapshot("before_split"),
    Branches(
        Branch(
            "refund",
            SessionAppend(role="user", content="改走退款流程"),
            FunctionStep(lambda _ctx: "退款：已提交申请，3 个工作日到账", output_key="reply"),
            snapshot_name="before_split",
        ),
        Branch(
            "rebook",
            SessionAppend(role="user", content="改走改签流程"),
            FunctionStep(lambda _ctx: "改签：已锁定后天同一时段航班", output_key="reply"),
            snapshot_name="before_split",
        ),
    ),
    Merge("refund", "rebook", output="comparison"),
    name="branch",
)


def branch_session(context, name: str) -> Session:
    '''从 `_branch_{name}` 里取出分支自己的会话。'''
    return context[f"_branch_{name}"]["_session"]


def main() -> int:
    session = build_session()
    print(f"分叉前父会话：{session.length} 条")

    context = branch_workflow.run(verbose=Verbosity.SILENT, _session=session)

    print()
    print("两条分支的结论：")
    for name, result in context["comparison"].items():
        print(f"  {name}: {result['reply']}")

    print()
    print("会话隔离检查：")
    print(f"  父会话            {session.length} 条（未被分支改动）")
    print(f"  分支 refund       {branch_session(context, 'refund').length} 条")
    print(f"  分支 rebook       {branch_session(context, 'rebook').length} 条")
    print(f"  refund 会话 id    {branch_session(context, 'refund').session_id}")
    print(f"  rebook 会话 id    {branch_session(context, 'rebook').session_id}")

    print()
    print("执行轨迹：")
    for item in context.trace:
        print(f"  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
