'''课程学习助手 —— 把 DSL 函数和 Skill 挂到 ChatWorkflow 上。

    python examples/course_assistant.py --show    # 看编译出来的工作流和工具（不调模型）
    python examples/course_assistant.py --fake    # 脚本化 agent 走一轮（不需要 Key）
    python examples/course_assistant.py           # 真模型，交互式对话

要点：

- `Function(args=..., name=..., description=...).Do(...)` 定义的是 **DSL 函数**，
  它可以直接放进 `ChatWorkflow(skills=[...])` —— 编译时会变成一个工具，
  模型自己决定什么时候调。
- 工具的 `description` 是**给模型看的**，比代码注释重要得多。
- `Function` 里 `tools=[WRITE]` 表示「这个函数内部再跑模型时，模型可以用这些工具」。
- `ChatWorkflow` 的 `skills=` 和 `tools=` 是两类东西：
  `tools` 是现成的工具，`skills` 是「带参数的能力」，会被包装成工具。
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions.builtin import Tone  # noqa: E402
from promptkit.core import (  # noqa: E402
    ChatWorkflow,
    Function,
    LLMStep,
    Prompt,
    Verbosity,
)
from promptkit.default_tools import WEB_SEARCH, WriteFile  # noqa: E402
from promptkit.memory import KVMemory  # noqa: E402
from promptkit.skill import SkillContext, SkillRegistry, skill  # noqa: E402
from promptkit.skill.chat import AgentHolder  # noqa: E402

# ================================= 参数区 ================================
SUBJECT = "机器学习"
TEXTBOOK = "《机器学习》（周志华）"
CHECK = "费曼学习法"

BASE_DIR = os.path.join(PROJECT_DIR, ".tmp", "course-assistant")
FILE_OUTPUT_PATH = BASE_DIR          # 想写到桌面就换成别处；这里默认落在 .tmp
SESSION = os.path.join(BASE_DIR, "session.json")

# ================================= 定义区 =================================
os.makedirs(BASE_DIR, exist_ok=True)
WRITE = WriteFile(FILE_OUTPUT_PATH)
#: 长期记忆：一个 SQLite 键值库，文件在 .tmp 下，下次运行还在
memory = KVMemory(os.path.join(BASE_DIR, "memory.db"))


@skill(
    name="remember_topic",
    description="把刚学完的小节记进学习记录（长期记忆），返回累计学了几节",
    tags=["memory"],
)
def remember_topic(topic: str, ctx: SkillContext) -> str:
    '''Skill 里用记忆：ctx.read / ctx.remember 就够，不用自己连数据库。'''
    done = list(ctx.read("topics", []))
    if topic not in done:
        done.append(topic)
    ctx.remember("topics", done)
    return f"已记录 {topic}（累计 {len(done)} 节）"


#: 一个 DSL 函数：参数 ref_path，内部还会调模型（模型可以顺手用 WRITE）
MakeLab = Function(
    args=("ref_path",),
    name="make_lab",
    description="根据当前学习主题生成一份实验任务清单；ref_path 是参考文档地址",
    output_key="lab",
    tools=[WRITE],
    # 继承父级状态：技能里能直接用 {{usr_input}} / {{topics}} 这些变量
    inherit=True,
).Do(
    LLMStep(
        prompt=Prompt(
            role="user",
            content=(
                f"我们在学《{TEXTBOOK}》的 {SUBJECT}。"
                "刚才讨论的是：{{usr_input}}\n"
                "请给出一个实验任务清单（3-5 条，每条一句话）。"
                "参考文档地址：{{ref_path}}"
            ),
        ),
        output_key="lab",
    )
)


def build():
    return ChatWorkflow(
        role=f"{SUBJECT}学习助手",
        want=f"根据具体情况讲解，并用{CHECK}验收",
        tools=[WEB_SEARCH, WRITE],
        skills=[MakeLab, remember_topic],
        rules=[
            "尽量把该推导的公式逐步推出来",
            "除非用户指定，不要生成任何文件",
            "学完一小节就用 remember_topic 记一笔",
        ],
        cautions=[Tone("形象又不失专业性")],
        # 只让技能用记忆：不用配 memory_config（配了就是聊天层也自动读写）
        memory=memory,
        session=SESSION,
    )


# ================================= 脚本化 agent ================================
class ScriptedAgent:
    '''不开 API 也能跑一轮：直接回一句准备好的话。'''

    vendor = "fake"
    model = "scripted"

    def __init__(self, reply="（脚本化回复）我先把这一节的要点讲清楚，再出一道自测题。"):
        self.reply = reply
        self.history = []
        self.start_prompts = [{"role": "system", "content": "脚本化 agent"}]

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        return self.reply

    def reset(self):
        self.history = []


def show() -> int:
    workflow = build().build()
    print(workflow.render())
    print("\n== 技能编译成了哪些工具 ==")
    from promptkit.skill.chat import as_tools

    for tool in as_tools([MakeLab, remember_topic], holder=AgentHolder(memory=memory)):
        print(f"- {tool.name}: {tool.description}")
        print(f"  参数：{tool.parameters}")
    print(f"\n记忆文件：{os.path.join(BASE_DIR, 'memory.db')}")
    print(f"已经记下的小节：{memory.read('topics', [])}")
    return 0


def fake() -> int:
    from unittest import mock

    from promptkit.tools import as_tools

    workflow = build().build()
    with mock.patch("builtins.input", side_effect=["这一节讲什么？", "exit"]):
        context = workflow.run(agent=ScriptedAgent(), verbose=Verbosity.SILENT)
    print("模型回复：", context.get("model_reply"))
    print("工具表：", [tool.name for tool in as_tools(context.get("chat_tools"))])
    print("会话文件：", SESSION)
    return 0


def log() -> int:
    topics = memory.read("topics", [])
    if not topics:
        print("还没有学习记录；用 --remember \"4.3 决策树\" 记一条试试。")
        return 0
    print(f"学习记录（{len(topics)} 节）：")
    for index, topic in enumerate(topics, start=1):
        print(f"  {index}. {topic}")
    return 0


def remember(topics) -> int:
    '''直接调 Skill（不经过模型），演示 ctx.remember / ctx.read。'''
    registry = SkillRegistry([remember_topic])
    context = SkillContext(memory=memory, skills=registry)
    for topic in topics:
        print(registry.execute("remember_topic", {"topic": topic}, ctx=context))
    return 0


def main(argv) -> int:
    if "--show" in argv:
        return show()
    if "--fake" in argv:
        return fake()
    if "--log" in argv:
        return log()
    if "--remember" in argv:
        index = argv.index("--remember") + 1
        topics = argv[index:] or ["（未指定小节）"]
        return remember([item for item in topics if not item.startswith("--")])

    agent = Agent(register_as_default=False)
    print(f"模型：{agent.vendor}/{agent.model}（输入 exit 退出）\n")
    messages = []
    index = 1
    while index < len(argv):
        if argv[index] == "--say" and index + 1 < len(argv):
            messages.append(argv[index + 1])
            index += 2
            continue
        index += 1

    workflow = build().build()
    if messages:
        from unittest import mock

        with mock.patch("builtins.input", side_effect=[*messages, "exit"]):
            context = workflow.run(agent=agent, verbose=Verbosity.SILENT)
        called = [
            message["content"]
            for message in agent.history
            if message.get("role") == "tool"
        ]
        if called:
            print(f"（这一轮模型调了 {len(called)} 次工具）")
        print(context.get("model_reply"))
        return 0
    workflow.run(agent=agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
