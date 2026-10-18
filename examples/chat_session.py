'''退出时保留上下文，下次接着聊。

核心就两件事：**存**（`Session.save`）和**装回来**（`Session.load`）。

用法一：`ChatWorkflow(session=...)` —— 一行搞定
    开跑前 load、结束时 save，连异常退出都会存（聊到一半 Ctrl+C 不会丢）。

用法二：手动管（不想用 ChatWorkflow 时）
    s = Session("sessions/a.json")
    s.load(agent)          # 上次聊到哪，装回来
    ...                    # 正常跟 agent 聊
    s.save(agent)          # 退出前存一次

    # 或者让上下文管理器负责（异常也存）：
    with Session("sessions/a.json").open(agent):
        agent.run()

运行（需要 API Key + 网络）：
    python examples/chat_session.py      # 第一次：从零开始
    python examples/chat_session.py      # 第二次：接着上次
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import ChatWorkflow  # noqa: E402
from promptkit.session import Session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_PATH = os.path.join(HERE, "sessions", "tutor.json")

chat = ChatWorkflow(
    role="市场调研分析大赛的导师",
    want="帮参赛者找到合适的选题方向",
    rules=["像导师一样专业、耐心", "每次回复不超过 300 字"],
    exit_words=["exit", "quit"],
    session=SESSION_PATH,          # ← 就这一行
)


def manual_round(agent: Agent) -> None:
    '''不用 ChatWorkflow 时的等价写法（异常也会存）。'''
    with Session(SESSION_PATH).open(agent):
        agent.run()


if __name__ == "__main__":
    session = Session(SESSION_PATH)
    if session.exists():
        info = session.read()
        print(f"接着上次聊：已有 {info['turns']} 轮，存于 {info['saved_at']}")
    else:
        print("新会话。")

    tutor = Agent(
        start_prompts=[{"role": "system", "content": "你是一位市场调研导师。"}]
    )
    chat.run(agent=tutor)

    print()
    print(f"会话文件 -> {SESSION_PATH}")
    print(f"累计 {Session(SESSION_PATH).turns()} 轮")
    print()
    print("手写 Workflow 也一样，给 run() 传 session 就行：")
    print('  wf.run(agent=agent, session="sessions/tutor.json")')
