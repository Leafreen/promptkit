'''记忆系统演示：结构化 KV + 语义召回 + 预算 + 遗忘。

运行（**不需要 API Key**，向量用本地词法向量，不联网）：

    python examples/memory.py

三段：

1. 长期记忆（结构化）：`KVMemory`（SQLite），按键读写
2. 长期记忆（非结构化）：`VectorMemory`，切块 -> 向量化 -> 召回
3. 放进 `Workflow`：`ReadMemory` / `RecallMemory` / `WriteMemory`

想接进聊天机器人（声明式），见文件末尾注释里的 `ChatWorkflow` 写法。
'''
import os
import sys
import tempfile

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptkit.core import (  # noqa: E402
    Echo,
    ReadMemory,
    RecallMemory,
    Workflow,
    WriteMemory,
)
from promptkit.memory import (  # noqa: E402
    KVMemory,
    TokenBudget,
    VectorMemory,
    format_memories,
)
from promptkit.memory.adapters import HashEmbedding  # noqa: E402

NOTES = (
    "退款政策：签收后 7 天内可无理由退款，需要订单号。\n"
    "发货时间：下单后 48 小时内发出，偏远地区顺延。\n"
    "发票：确认收货后可以在订单页申请电子发票。\n"
)


def main() -> None:
    db_path = os.path.join(tempfile.gettempdir(), "promptkit-memory-demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)                     # 每次从干净的记忆开始

    # ---------------- 1. 结构化长期记忆 ----------------
    memory = KVMemory(db_path, namespace="learner_001")
    memory.write("level", "中级")
    memory.write("preferred_style", "先给结论，再举例子")
    memory.write("topics_covered", ["退款", "发货"])

    print("== 结构化记忆（KVMemory）==")
    print("level            =", memory.read("level"))
    print("preferred_style  =", memory.read("preferred_style"))
    print("topics_covered   =", memory.read("topics_covered"))
    print("读一个没有的键    =", memory.read("signup_date", "（没有）"))
    print("当前键            =", memory.keys())

    # ---------------- 2. 语义召回 ----------------
    # HashEmbedding 是「词法级」向量：不联网、零依赖，够演示和写测试；
    # 要真语义就换成 OpenAIEmbedding / SentenceTransformerEmbedding（接口一样）。
    vector = VectorMemory(
        HashEmbedding(dimension=512),
        namespace="learner_001",
        chunk_size=60,
        overlap=10,
        budget=TokenBudget(max_tokens=120),    # 召回后按 token 裁剪
    )
    vector.write("notes", NOTES)
    print(f"\n== 语义召回（VectorMemory，{vector.count()} 个 chunk）==")
    print("问：退款和发票怎么弄？")
    print(format_memories(vector.recall("退款和发票怎么弄", top_k=2)))

    # ---------------- 3. 放进工作流 ----------------
    print("\n== 工作流里的记忆步骤 ==")
    context = Workflow(
        # 开跑前读结构化记忆
        ReadMemory(memory, "level", output_key="level"),
        # 运行时按当前问题召回
        RecallMemory(
            vector,
            query="退款和发票怎么弄",              # 也可以 query_key="usr_input"
            output_key="related",
            top_k=2,
            budget=TokenBudget(max_tokens=120),
        ),
        Echo("学员水平：{{level}}"),
        Echo("参考资料：\n{{related}}"),
        # 把这一轮的结论写回去（显式写，模型没有写权限）
        WriteMemory(memory, "last_question", "退款和发票怎么弄"),
    ).run()
    print("写回的 last_question =", memory.read("last_question"))
    print("context 里的 related 长度 =", len(context["related"]))

    # ---------------- 4. 遗忘 ----------------
    # 「可遗忘」是记忆的基本要求：写进去的必须能删掉
    memory.forget("last_question")
    vector.forget("notes")
    print("\n== 遗忘之后 ==")
    print("KV 里还剩：", memory.keys())
    print("向量库里还剩：", vector.count(), "个 chunk")

    memory.close()
    os.remove(db_path)

    # ---------------- 5. 存哪：默认目录 + StorageConfig ----------------
    print("\n== 存储位置 ==")
    from unittest import mock

    from promptkit.memory import (
        StorageConfig,
        create_memory,
        default_data_dir,
        default_kv_path,
        default_vector_dir,
    )

    # 不传 path 时落在系统数据目录（这里用临时目录模拟，别真写你的 APPDATA）
    data_dir = tempfile.mkdtemp(prefix="promptkit-demo-")
    with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": data_dir}):
        print("默认数据目录：", default_data_dir())
        print("KV 默认文件  ：", default_kv_path("learner_001"))
        print("向量默认目录 ：", default_vector_dir("learner_001"))

        kv = create_memory(StorageConfig(backend="sqlite", namespace="learner_001"))
        kv.write("level", "中级")
        backup = kv.backup(os.path.join(data_dir, "backup", "learner_001.db"))
        exported = kv.export_json(os.path.join(data_dir, "learner_001.json"))
        print("热备份 / 导出：", os.path.basename(backup), "/", os.path.basename(exported))

        restored = KVMemory(backup, namespace="learner_001")
        print("备份里读回来 ：", restored.read("level"))
        restored.close()

        kv.destroy()                        # 删库（不可恢复）
        print("destroy 之后还存在吗：", os.path.exists(kv.path))


# ---------------- 想接进聊天机器人？ ----------------
# from promptkit.core import ChatWorkflow
# from promptkit.memory import MemoryConfig, RecallConfig, long_term_memory
#
# chat = ChatWorkflow(
#     role="客服",
#     memory=long_term_memory("learner_001",     # 默认位置：KV + 向量都在数据目录
#                             embed_client=HashEmbedding(dimension=512)),
#     memory_config=MemoryConfig(
#         read=["level"],
#         recall=RecallConfig(query_key="usr_input", output_key="related", top_k=3),
#         write=["topics_covered"],
#         budget=TokenBudget(max_tokens=1000),
#     ),
# )
# chat.run(agent=Agent())      # 这一步要 API Key；不需要 Key 的演示就到上面为止

if __name__ == "__main__":
    main()
