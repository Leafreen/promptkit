'''一个会记住你的答疑教练 —— 跑两遍就能看出区别。

    python examples/memory_tutor.py "样本量到底怎么定？"
    python examples/memory_tutor.py "那分层抽样呢？"        # 第二遍：它记得你
    python examples/memory_tutor.py --show                 # 看记忆里存了什么
    python examples/memory_tutor.py --forget level         # 忘掉一条

每次运行分四步，全部是显式步骤（模型只有「写摘要」这一件事需要产出，
写哪种记忆由工作流决定）：

    读画像（KVMemory） → 召回笔记（VectorMemory） → 回答 → 抽取摘要 → 写回记忆

两份记忆各管一摊，位置都是**显式传进来**的：

    .tmp/memory-tutor/memory.db      KV：画像 / 进度（一个文件）
    .tmp/memory-tutor/vectors/       向量：笔记索引（一个文件夹）

不想传路径的话，`KVMemory(namespace=...)` / `VectorMemory(..., path=...)`
会落到系统数据目录（见 README 第 23 节「存在哪」）；这里放项目内是为了
方便你 `rm -rf .tmp/memory-tutor` 一键重来。
'''
import os
import sys
import time

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
# （包目录 == 项目根目录，所以父目录是 D:\projects）
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import JSON_ONLY  # noqa: E402
from promptkit.core import (  # noqa: E402
    Call,
    LLMStep,
    ReadMemory,
    RecallMemory,
    Set,
    Var,
    Workflow,
    WriteMemory,
)
from promptkit.memory import KVMemory, TokenBudget, VectorMemory  # noqa: E402
from promptkit.memory.adapters import HashEmbedding  # noqa: E402
from promptkit.prompt import Prompt  # noqa: E402

#: 演示用的记忆位置（.tmp/ 在 .gitignore 里）
MEMORY_DIR = os.path.join(PROJECT_DIR, ".tmp", "memory-tutor")
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
VECTOR_DIR = os.path.join(MEMORY_DIR, "vectors")
#: 多用户隔离：一个用户一个 namespace
USER = "learner_001"
#: 笔记在 KV 里的前缀（`--show` / 排查时用）
NOTE_PREFIX = "note:"


# ============================ 结构化摘要 ============================
class Session(BaseModel):
    '''这一轮问答的「收获」。模型只负责产出它，写不写由工作流决定。'''

    digest: str          # 一句话概括（进向量库，下次召回）
    level: str           # 对学员水平的判断：入门 / 中级 / 高级
    weak_points: list[str]  # 这次暴露出来的薄弱点


def join_list(values) -> str:
    '''把列表拼成可读文本（prompt 里插 list 会变成 repr）。'''
    if not values:
        return "（还不了解）"
    return "、".join(str(value) for value in values)


# ============================ 记忆 ============================
def open_memory():
    '''打开两份记忆：KV 存结构化值，向量存能被召回的笔记。

    两者都**显式指定位置**（`DB_PATH` / `VECTOR_DIR`）。不传 path 的话会落到
    系统数据目录 —— 交付给用户的产品通常就该用默认值，让用户不用管路径。

    换真语义只要把 HashEmbedding 换成 OpenAIEmbedding
    （或者 SentenceTransformerEmbedding 跑本地 BGE），接口一样。
    '''
    kv = KVMemory(DB_PATH, namespace=USER)
    vector = VectorMemory(
        HashEmbedding(dimension=512),
        namespace=USER,
        path=VECTOR_DIR,          # 一个文件夹 = 一份向量记忆
        chunk_size=300,
        overlap=40,
        budget=TokenBudget(max_tokens=600),
    )
    return kv, vector


# ============================ 一轮问答 ============================
def build(question: str, kv: KVMemory, vector: VectorMemory) -> Workflow:
    '''编译一轮问答的步骤。'''
    note_key = f"{NOTE_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}"
    answer_prompt = (
        "你是学习教练，正在给一位学员答疑。\n"
        "学员画像：水平={{level}}；已知薄弱点={{weak_text}}；这是第{{visits}}次来。\n"
        "\n"
        "从长期记忆里召回的相关笔记：\n"
        "{{notes}}\n"
        "\n"
        "问题：{{question}}\n"
        "\n"
        "要求：先给结论，再给 2-3 步可执行的做法；不超过 200 字。"
        "如果画像或笔记是空的（第一次来），先简单确认一下他的基础再回答。"
    )
    summary_prompt = (
        "下面是学员的一次提问和你的回答。\n"
        "\n问题：{{question}}\n回答：{{answer}}\n\n"
        "请提炼这次交流的收获，输出 JSON（字段：digest / level / weak_points）。\n"
        "- digest：一句话概括这次讲了什么（≤ 60 字），以后要靠它回忆\n"
        "- level：这个学员的水平，只能是 入门 / 中级 / 高级\n"
        "- weak_points：这次暴露的薄弱点（没有就给空数组）"
    )

    return Workflow(
        # 1) 读画像：第一次来全是默认值
        ReadMemory(kv, "level", output_key="level", default="未知"),
        ReadMemory(kv, "weak_points", output_key="weak_points", default=[]),
        ReadMemory(kv, "visits", output_key="visits", default=0),
        Set("weak_text", Call(join_list, Var("weak_points"))),
        # 2) 按问题召回笔记（第一次跑是空的）
        RecallMemory(
            vector,
            query=Var("question"),
            output_key="notes",
            top_k=2,
            budget=TokenBudget(max_tokens=600),
        ),
        # 3) 回答
        LLMStep(Prompt(role="user", content=answer_prompt), output_key="answer"),
        # 4) 把这一轮提炼成结构化摘要（唯一需要模型产出的「记忆候选」）
        LLMStep(
            Prompt(role="user", content=summary_prompt),
            Session,
            output_key="summary",
            cautions=[JSON_ONLY],
        ),
        # 5) 显式写回：画像进 KV，摘要两边都写（KV 存原文，向量库供召回）
        WriteMemory(kv, "level", Var("summary").attr("level")),
        WriteMemory(kv, "weak_points", Var("summary").attr("weak_points")),
        WriteMemory(kv, "visits", Var("visits") + 1),
        WriteMemory(kv, note_key, Var("summary").attr("digest")),
        WriteMemory(vector, note_key, Var("summary").attr("digest")),
        name="tutor_turn",
    )


# ============================ 命令行 ============================
def show(kv: KVMemory, vector: VectorMemory) -> None:
    print(f"记忆文件：{DB_PATH}")
    print(f"向量文件夹：{VECTOR_DIR}")
    print(f"namespace：{USER}\n")
    if not kv.count():
        print("（空的：还没聊过）")
        return
    for key, value in kv.items().items():
        print(f"  {key:<24} = {value!r}")
    print(f"\n向量索引：{vector.count()} 个 chunk")


def forget(kv: KVMemory, vector: VectorMemory, key: str) -> None:
    removed = kv.forget(key)
    vector.forget(key)
    print(f"忘掉 {key!r}：{'已删除' if removed else '本来就没有'}")


def main(argv) -> int:
    kv, vector = open_memory()
    try:
        if "--show" in argv:
            show(kv, vector)
            return 0
        if "--forget" in argv:
            index = argv.index("--forget")
            key = argv[index + 1] if len(argv) > index + 1 else ""
            if not key:
                print("用法：python examples/memory_tutor.py --forget level")
                return 2
            forget(kv, vector, key)
            return 0

        question = " ".join(argv[1:]).strip() or input("问点什么：").strip()
        if not question:
            print("问题不能为空")
            return 2

        agent = Agent(register_as_default=False)
        ctx = build(question, kv, vector).run(agent=agent, question=question)

        print(f"问题：{question}")
        print(
            f"读到的画像：level={ctx['level']!r}  weak_points={ctx['weak_points']!r}  "
            f"visits={ctx['visits']}"
        )
        print(f"召回笔记：{'（没有，这是第一次）' if not ctx['notes'] else ''}")
        if ctx["notes"]:
            print(ctx["notes"])
        print("\n回答：")
        print(ctx["answer"])
        print(f"\n提炼出的摘要：{ctx['summary'].model_dump()}")
        print(f"写回记忆：level / weak_points / visits={ctx['visits'] + 1} / 一条 note")
        print("（再跑一次同一个问题，看它是不是记得你）")
        return 0
    finally:
        kv.close()
        vector.close()          # 向量库落盘（文件夹里那个 index.jsonl）


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
