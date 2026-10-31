'''手写 Workflow 的完整记忆学习助手（不使用 ChatWorkflow）。

记忆分两层：
- 短期对话：session.json，保存本轮聊天的消息历史，下次启动自动续接。
- 长期记忆：tutor.db + tutor.db.vectors/，保存需要跨会话记住的笔记。

模型可以调用两个记忆工具：
- recall_memory：按意思召回长期记忆（只读）。
- write_memory：把值得长期保留的信息写入长期记忆（显式授权）。

先修改下面的 KNOWLEDGE_PATH，再运行：
    python examples/handwritten_memory_bot.py

可以试试第一轮说「请记住：我喜欢用生活例子理解数学符号」，
退出后再启动，问「我喜欢怎样理解数学符号？」。
'''
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit import features  # noqa: E402

features.enable("knowledge")

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    ChatLoop,
    Echo,
    ExitWords,
    FunctionStep,
    If,
    Role,
    Rules,
    SystemPrompt,
    Tools,
    Var,
    Verbosity,
    Want,
    Workflow,
)
from promptkit.core.tool import WEB_SEARCH, WriteFile  # noqa: E402
from promptkit.knowledge import KNOWLEDGE_REFERENCE  # noqa: E402
from promptkit.memory import (  # noqa: E402
    MEMORY_REFERENCE,
    WRITE_MEMORY,
    TokenBudget,
    long_term_memory,
)
from promptkit.memory.adapters import HashEmbedding  # noqa: E402

# ============================ 配置区 ============================
KNOWLEDGE_PATH = Path(r"D:\学习资源\运筹学")  # 改成你的资料目录
DATA_DIR = PROJECT_DIR / ".tmp" / "handwritten-memory-bot"
MEMORY_PATH = DATA_DIR / "tutor.db"
SESSION_PATH = DATA_DIR / "session.json"
OUTPUT_DIR = DATA_DIR / "output"
NAMESPACE = "learner_001"  # 不同用户使用不同 namespace，隔离记忆


# ============================ 长期记忆 ============================
def open_memory():
    '''打开 KV + 向量记忆；本地 HashEmbedding 不需要额外的模型 Key。'''
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    return long_term_memory(
        NAMESPACE,
        path=str(MEMORY_PATH),
        vector_path=str(MEMORY_PATH) + ".vectors",
        embed_client=HashEmbedding(dimension=512),
        budget=TokenBudget(max_tokens=600),
    )


# ============================ 手写工作流 ============================
def build_workflow(memory, knowledge_path=KNOWLEDGE_PATH) -> Workflow:
    '''显式接好工具和每轮状态；不依赖 ChatWorkflow 的自动编译。

    读和写必须共用同一个 memory 实例，否则写进去的内容无法从另一个实例
    立即召回（KV 走磁盘能互通，但向量索引是进程内的）。
    '''
    recall_tool = MEMORY_REFERENCE(
        memory,
        top_k=3,
        max_calls=3,
        budget=TokenBudget(max_tokens=600),
    )
    # 一行拿到「写记忆」工具，对称于 MEMORY_REFERENCE。
    # 写是显式授权的：只有把 write_tool 加进 Tools(...)，模型才能写记忆。
    write_tool = WRITE_MEMORY(memory, max_calls=5)

    return Workflow(name="handwritten_memory_tutor").add(
        Role("《运筹学》学习助手"),
        Want("形象讲解资料中的知识点，并按需生成 numpy 手写实验任务单"),
        Rules(
            "使用专业、形象的语言，先给直觉解释，再给公式或实验",
            "优先调用 search_knowledge_base 查资料；资料不足时再联网搜索",
            "需要以往偏好、学习进度或历史结论时，调用 recall_memory，不要凭空编造",
            "只有用户明确要求记住，或出现值得长期保留的信息时，才调用 write_memory",
            "只有用户要求生成实验清单或保存成果时才创建文件，其他时间不要创建文件",
        ),
        Tools(
            WEB_SEARCH,
            KNOWLEDGE_REFERENCE(str(knowledge_path)),
            recall_tool,
            write_tool,
            WriteFile(str(OUTPUT_DIR)),
        ),
        ExitWords("exit", "quit"),
        SystemPrompt(),
        ChatLoop(
            behavior=[
                # 两个工具的 max_calls 都按实例计数，所以每轮开始前清零。
                FunctionStep(
                    lambda context: (recall_tool.reset(), write_tool.reset()),
                    name="reset_memory_tools",
                ),
                If(Var("usr_input") == "help").Then(
                    Echo("输入 exit 或 quit 退出；可以直接说「请记住：……」。")
                ),
            ],
        ),
    )


def main() -> None:
    if not KNOWLEDGE_PATH.exists():
        raise SystemExit(f"知识库路径不存在，请修改 KNOWLEDGE_PATH：{KNOWLEDGE_PATH}")

    memory = open_memory()
    try:
        # Agent() 从环境变量或厂商配置读取 Key，不要把 API Key 写进源码。
        agent = Agent()
        workflow = build_workflow(memory)
        print(f"长期记忆：{MEMORY_PATH}")
        print(f"对话会话：{SESSION_PATH}")
        print("输入 help 查看提示，输入 exit 退出。\n")
        workflow.run(
            agent=agent,
            session=str(SESSION_PATH),  # 短期对话历史：自动 load / save
            verbose=Verbosity.NORMAL,
        )
    finally:
        memory.close()  # 关闭 KV，并把向量索引落盘


if __name__ == "__main__":
    main()
