'''对话状态层 · 基础 CRUD 组合示例。

把 SessionAppend / SessionInsert / SessionUpdate / SessionRemove /
SessionTruncate / SessionClear 组合成一条普通工作流：所有 Step 只认
`context["_session"]`，不直接碰 `Session` 的内部字段。

这个示例不调用任何模型 API：`_session` 直接用内存里的 `Session`。

运行：
    python examples/conversation_crud.py
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.conversation import (
    Message,
    Session,
    SessionAppend,
    SessionClear,
    SessionInsert,
    SessionRemove,
    SessionTruncate,
    SessionUpdate,
)
from promptkit.core import Verbosity, Workflow


def build_session() -> Session:
    '''准备一个带固定 id 的会话，方便后面按 id 更新 / 删除。'''
    return Session(
        [
            Message(role="system", content="你是 promptkit 的演示助手。", id="m-sys"),
            Message(role="user", content="帮我写个 hello 函数", id="m-q1"),
            Message(role="assistant", content="def hello(): return 'hi'", id="m-a1"),
        ]
    )


# 六个 CRUD Step：追加 -> 插入 -> 更新 -> 删除 -> 截断
crud_workflow = Workflow(
    SessionAppend(role="user", content="再加一个类型注解"),
    SessionInsert(index=3, role="assistant", content="（插入的过渡说明）"),
    SessionUpdate(message_id="m-q1", content="帮我写个带类型注解的 hello 函数"),
    SessionRemove(message_id="m-a1"),
    SessionTruncate(length=3),
    name="crud",
)

# 清空单独放在一条工作流里，避免把上面的演示结果一起抹掉
clear_workflow = Workflow(SessionClear(), name="clear")


def print_messages(title: str, session: Session) -> None:
    '''按位置打印会话内容，顺便展示消息 id。'''
    print(f"{title}（{session.length} 条）")
    for index, message in enumerate(session.messages):
        print(f"  [{index}] {message.role:<9} {message.content}  (id={message.id})")


def print_trace(title: str, trace: list) -> None:
    '''打印每个 Step 写进 ctx.trace 的结果。'''
    print(f"{title}（{len(trace)} 条）")
    for item in trace:
        print(f"  {item}")


def main() -> int:
    session = build_session()
    print_messages("初始会话", session)

    context = crud_workflow.run(verbose=Verbosity.SILENT, _session=session)
    print()
    print_messages("CRUD 之后", session)
    print()
    print_trace("CRUD 轨迹", context.trace)

    clear_workflow.run(verbose=Verbosity.SILENT, _session=session)
    print()
    print_messages("Clear 之后", session)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
