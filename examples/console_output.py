'''终端显示分层：Markdown 渲染 + verbose 分级 + 日志落盘 + 结果导出。

演示「终端给人看、文件给排查用」这套分工：

    终端（NORMAL）   只有对话，Markdown 排好版，公式转成 Unicode
    终端（VERBOSE）  + 步骤轨迹 / 工具调用 / 思考
    日志文件        全量事件（步骤耗时、工具参数、token 用量）
    输出文件        结果 Markdown + 对话历史 JSON

`rich` 和 `sympy` 都是**可选**的，没装也能跑（退回纯文本 + 内置公式映射）：

    pip install "promptkit[render]"

运行（需要 API Key + 网络）：
    python examples/console_output.py
'''
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import TUTOR  # noqa: E402
from promptkit.core import (  # noqa: E402
    HAVE_RICH,
    HAVE_SYMPY,
    Break,
    Echo,
    If,
    LLMStep,
    NewVar,
    Prompt,
    Read,
    Save,
    SaveHistory,
    Set,
    Str,
    Var,
    Verbosity,
    While,
    Workflow,
    log_display,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
LOG_FILE = os.path.join(HERE, "logs", "run.log")

#: 这一轮的结果存哪儿
ANSWER_MD = os.path.join(OUT_DIR, "回答.md")
INFO_JSON = os.path.join(OUT_DIR, "基本信息.json")
HISTORY_JSON = os.path.join(OUT_DIR, "history.json")
HISTORY_MD = os.path.join(OUT_DIR, "history.md")


def build_workflow() -> Workflow:
    '''工作流本身不 print，也不 open() —— 显示和落盘都声明成 Step。'''
    return Workflow(
        Str("usr_input"),
        NewVar("signal", default="start"),
        NewVar("basic_info", default={}),

        Echo("先了解一下背景。"),
        Echo("你关心哪个方向？"),
        Set(Var("basic_info")["方向"], Read(">>> ")),
        Echo("你的数据基础怎么样？（会 SPSS / 会 Python / 都不会）"),
        Set(Var("basic_info")["数据基础"], Read(">>> ")),

        While(Var("signal") != "end").Do(
            Set(Var("usr_input"), Read(">>> ")),
            NewVar("model_reply"),
            If(Var("usr_input") == "exit").Then(
                Set(Var("signal"), "end"),
                Break(),
            ),
            LLMStep(
                prompt=Prompt(
                    role="user",
                    content=(
                        "背景信息：{{basic_info}}\n\n"
                        "我的问题：{{usr_input}}\n\n"
                        "请用 Markdown 回答（可以有标题、列表、表格、代码块）；"
                        "涉及公式请用 $...$ 或 $$...$$ 写 LaTeX。"
                    ),
                ),
                output_key="model_reply",
                cautions=[TUTOR],
            ),
            # 关键：声明「这段按 Markdown 显示」，而不是自己 print
            Echo(Var("model_reply"), render="markdown", title="选题建议"),
        ),

        # 落盘也是 Step：结果 Markdown 追加写，基本信息存成真 JSON
        Save(Var("model_reply"), file=ANSWER_MD, append=True),
        Save(Var("basic_info"), file=INFO_JSON, format="json"),
        SaveHistory(HISTORY_JSON, include_variables=True),
    )


wf = build_workflow()


if __name__ == "__main__":
    print(f"rich = {HAVE_RICH} | sympy = {HAVE_SYMPY}")
    if not HAVE_RICH:
        print('提示：装了 rich 才有排版效果 -> pip install "promptkit[render]"')

    agent = Agent(
        start_prompts=[
            Prompt(
                role="system",
                content="你是一位市场调研分析大赛的导师，回答专业、分步骤、不编造。",
            ).render()
        ]
    )

    # 终端只留对话；过程全量写进日志文件。两条路互不影响。
    wf.run(
        agent=agent,
        verbose=Verbosity.NORMAL,          # 排查时改成 Verbosity.DEBUG
        on_event=log_display(LOG_FILE),
    )

    # history 属于 Agent，不属于 Workflow
    agent.export_history(HISTORY_JSON.replace(".json", "-agent.json"), fmt="json")
    agent.export_history(HISTORY_MD, fmt="markdown")

    print()
    print(f"日志     -> {LOG_FILE}")
    print(f"回答     -> {ANSWER_MD}")
    print(f"历史     -> {HISTORY_MD}")
