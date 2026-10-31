'''聊天机器人：声明式（`ChatWorkflow`）和手写（普通 Step）两种写法。

两者编译出的东西是一样的 —— `ChatWorkflow` 只是语法糖，
`build()` 返回的就是普通 `Workflow`，现有的步骤树 / 事件 / Verbosity 全适用。

运行（需要 API Key + 网络）：
    python examples/chat_bot.py
'''
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    ChatLoop,
    ChatWorkflow,
    Deliverable,
    Echo,
    If,
    Read,
    Role,
    Rules,
    Set,
    SystemPrompt,
    Tools,
    Var,
    Want,
    Workflow,
    ExitWords,
)
from promptkit.default_tools import WEB_SEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

SYSTEM = (
    "你是一位市场调研分析大赛的导师。"
    "回答专业、分步骤、不编造；不确定就直说。"
)


def declarative() -> ChatWorkflow:
    '''写法一：声明式。角色 / 目标 / 规矩 / 交付物分开写。'''
    return ChatWorkflow(
        role="市场调研分析大赛的导师",
        want="帮参赛者找到合适的选题方向，并产出结构化的选题建议",
        rules=[
            "像导师一样专业、耐心",
            "每次回复不超过 300 字",
            "先了解用户背景，再给建议",
        ],
        deliverables=[
            Deliverable(
                "选题建议",
                description="3-5 个可行的选题方向，含背景、方法、预期结论",
                format="markdown",
                save_to=os.path.join(OUT_DIR, "选题建议.md"),
            ),
        ],
        behavior=[
            If(Var("usr_input") == "help").Then(Echo("输入 exit 退出。")),
            # 多行输入：敲 ml 进入多行模式，空行结束（详见 README）
            If(Var("usr_input") == "ml").Then(
                Set(Var("usr_input"), Read("... ").Until("").OnBreak()),
            ),
        ],
        tools=[WEB_SEARCH],
        exit_words=["exit", "quit"],
    )


def hand_written() -> Workflow:
    '''写法二：手写。`ChatWorkflow.build()` 展开后基本就是这个样子。'''
    return Workflow(
        Role("市场调研分析大赛的导师"),
        Want("帮参赛者找到合适的选题方向"),
        Rules("像导师一样专业、耐心", "每次回复不超过 300 字"),
        Tools(WEB_SEARCH),
        ExitWords("exit", "quit"),
        SystemPrompt(),
        ChatLoop(
            If(Var("usr_input") == "help").Then(Echo("输入 exit 退出。")),
            If(Var("usr_input") == "ml").Then(
                Set(Var("usr_input"), Read("... ").Until("").OnBreak()),
            ),
        ),
        name="chat_hand_written",
    )


chat = declarative()
wf = hand_written()


if __name__ == "__main__":
    print("=== 编译出来的步骤树（declare 版）===")
    print(chat.build())
    print()
    print("=== 编译出来的步骤树（手写版）===")
    print(wf)
    print()
    print("=== system prompt ===")
    print(chat.build_system_prompt())
    print()

    tutor = Agent(start_prompts=[{"role": "system", "content": SYSTEM}])
    chat.run(agent=tutor)
    print()
    print(f"交付物 -> {os.path.join(OUT_DIR, '选题建议.md')}")
