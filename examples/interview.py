'''模拟面试工作流。

所有逻辑都写在下面的工作流声明里；main 只负责 run()。

运行前准备密钥（二选一）：
    set DEEPSEEK_API_KEY=sk-xxxx      # Windows
    export DEEPSEEK_API_KEY=sk-xxxx   # Linux / macOS
或者在项目根目录的 .env 里写一行 DEEPSEEK_API_KEY=sk-xxxx。

运行：
    python examples/interview.py
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import *  # noqa: E402,F403

# ================== 模拟面试工作流 ====================
# 1. 输入面试的岗位
# 2. 附加信息
# 3. 生成面试问题
# 4. 回答，并在最后互补
# =====================================================

Input = Function(args=("job", "extra_info"), name="greeting").Do(
    Prompt(role="user", content="你是一个面试官，准备面试一个{{job}}岗位。下面是附加信息：{{extra_info}}。请先问好，然后开始面试。未接收到指令，不要生成问题")
)

GenerateQuestions = Function(args=("job", "extra_info", "user_answer"), name="question").Do(
    Prompt(role="user", content="候选人刚才的回答是：{{user_answer}}\n请针对{{job}}岗位生成下一个面试问题，只问一个问题，不要给出答案。")
)

SumUp = Function(args=("job", "extra_info"), name="summary").Do(
    Prompt(role="user", content="请回答问题，并在最后总结。并给出评价")
)

wf = Workflow()

# Agent() 会读取 DEEPSEEK_API_KEY（或 .env），并把自己注册成默认 agent，
# 所以最后 wf.run() 不需要传任何参数。
agent = Agent()

wf.add(
    # 1 + 2. Read 是惰性的，真正读输入发生在 wf.run() 的时候
    Set("job", Read("请输入面试的岗位：")),
    Set("extra_info", Read("请输入附加信息：")),

    # 变量先声明，之后才能用 Var(...) 读
    NewVar("state", default="进行中"),
    NewVar("user_answer", default="（面试刚开始，候选人还没有回答问题）"),
    NewVar("turn", default=0),

    # 开场问好
    Input(job=Var("job"), extra_info=Var("extra_info")),

    # 3 + 4. 一轮一轮问下去，直到候选人说「结束」
    While(Condition(Var("state") != "结束"), max_iterations=20).Do(
        # agent_reply()：先执行这一步，再把模型回复写进 question
        Var(
            name="question",
            val=GenerateQuestions(
                job=Var("job"),
                extra_info=Var("extra_info"),
                user_answer=Var("user_answer"),
            ).agent_reply(),
        ),
        Echo("面试官：", Var("question")),
        Set("user_answer", Read("\n你的回答（输入「结束」提前结束面试）：")),
        Set("state", Var("user_answer")),
        Set("turn", Var("turn") + 1),
    ),

    # 面试结束后总结
    SumUp(job=Var("job"), extra_info=Var("extra_info")),

    Echo("=" * 50),
    Echo("面试记录：共 ", Var("turn"), " 轮"),
    Echo("面试官总结：", Var("summary")),
)


if __name__ == "__main__":
    wf.run()
