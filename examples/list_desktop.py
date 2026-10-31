'''用工具调用列出桌面文件 —— 顺便演示「过程」和「结果」分开重定向。

    python examples/list_desktop.py                # 过程走 stderr，两者混着看
    python examples/list_desktop.py > r.txt 2> t.txt
    python examples/list_desktop.py 1>/dev/null    # 只看过程
'''
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    Echo,
    Function,
    LLMStep,
    NewVar,
    Prompt,
    Return,
    Set,
    Var,
    VarExpr,
    Workflow,
)
from promptkit.core.tool import RUN_COMMAND  # noqa: E402

# RUN_COMMAND 的描述里带了当前系统和 shell 语法，
# 所以模型不会跑去试 ls ~/Desktop
ListDesktop = Function(args=(), name="ListDesktop").Do(
    LLMStep(
        prompt=Prompt(
            role="user",
            content="请列出我桌面（Desktop）上的所有文件。可以用 run_command 工具。",
        ),
        tools=[RUN_COMMAND],
        output_key="files",
    ),
    Return(VarExpr("files")),
)


if __name__ == "__main__":
    # verbose 的思考 / 工具调用会打到 stderr，
    # 下面 Echo 的结果留在 stdout
    agent = Agent(verbose=True)

    wf = Workflow()
    wf.add(
        NewVar("files"),
        Set(Var("files"), ListDesktop().result),
        Echo("桌面上的文件列表：", VarExpr("files")),
    )
    wf.run()
