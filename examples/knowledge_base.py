'''用 Workflow 搭一个知识库问答（RAG）。

⚠️ 知识库是**实验性功能，默认关闭**，所以下面要先 enable 一下。

分工：
  - `KnowledgeBase`（promptkit/knowledge.py）负责**检索** —— 普通 Python
  - `Workflow` 负责**编排** —— 什么时候检索、检索不到怎么办、资料怎么进提示词

两种做法都演示了：
  ① 显式检索步骤：可控、便宜、可测试（推荐先用这个）
  ② 交给模型自己检索（tools）—— 灵活，但多花 token

运行（需要 API Key + 网络）：
    python examples/knowledge_base.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from promptkit import features  # noqa: E402

features.enable("knowledge")  # ← 实验性功能，默认关闭

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    Call,
    Echo,
    Goto,
    If,
    Label,
    Len,
    LLMStep,
    NewVar,
    Not,
    Prompt,
    Set,
    Var,
    Workflow,
)
from promptkit.cautions import JSON_ONLY  # noqa: E402
from promptkit.knowledge import KnowledgeBase  # noqa: E402

# ====================== 1. 建库 ======================
# 真实项目里换成 KnowledgeBase.from_dir("docs") 或 .add_file(...)
KB = KnowledgeBase()
KB.add(
    '退款政策\n'
    '未发货的订单可以直接申请退款，提交后 3 个工作日原路退回。\n'
    '已发货的订单需要先申请退货，我们收到货后 7 天内处理退款。\n'
    '超过 30 天的订单不支持退款，请联系人工客服。',
    source="policy.md",
)
KB.add(
    '配送范围\n'
    '目前只支持中国大陆地区，港澳台及海外暂不支持配送。\n'
    '普通快递 2-4 天送达，偏远地区 5-7 天。',
    source="shipping.md",
)
KB.add(
    '发票\n'
    '支持开具电子发票，下单时填写抬头和税号即可，发货后自动发到邮箱。\n'
    '纸质发票需要联系客服单独申请。',
    source="invoice.md",
)

#: 低于这个分数就当作「没搜到」。
#: 词法检索里，不相关的查询也会因为一两个单字沾边拿到 0.3~0.5 分，
#: 设个门槛比让模型去读垃圾资料划算。具体数值按你的库调。
MIN_SCORE = 1.0


class Answer(BaseModel):
    answer: str
    used: List[int]  # 用到了第几段资料
    enough: bool  # 资料够不够回答


# ====================== 2. 显式检索的 RAG ======================
rag = Workflow(
    NewVar("question"),
    NewVar("hits"),
    NewVar("context"),
    NewVar("answer"),

    # ① 检索：把普通 Python 函数接进工作流
    Set("hits", Call(KB.search, Var("question"), k=4, min_score=MIN_SCORE)),
    Echo("（检索到 ", Call(len, Var("hits")), " 段资料）"),

    # ② 检索不到就别花 token 了
    If(Len(Var("hits")) == 0).Then(
        Set("answer", "知识库里没有和这个问题相关的内容。"),
        Goto("done"),
    ),

    # ③ 把资料拼成提示词能用的文本
    Set("context", Call(KnowledgeBase.format, Var("hits"), query=Var("question"))),

    # ④ 让模型基于资料回答，并强制标注用到哪几段
    LLMStep[Answer](
        prompt=Prompt(
            role="user",
            content=(
                "请只根据下面的资料回答问题。资料里没有的信息不要编。\n"
                "回答里必须给出你用到资料的编号。\n\n"
                "资料：\n{{context}}\n\n"
                "问题：{{question}}"
            ),
        ),
        output_model=Answer,
        output_key="answer",
        cautions=[JSON_ONLY],
        name="answer_from_kb",
    ),

    # ⑤ 资料不够就承认，而不是硬编
    If(Not(Var("answer.enough"))).Then(
        Echo("⚠️ 资料不足以回答，建议转人工。"),
    ),

    Label("done"),
    Echo("答：", Var("answer.answer")),
    Echo("引用：", Var("answer.used")),
    name="rag",
)


# ====================== 3. 让模型自己检索 ======================
agentic = Workflow(
    LLMStep(
        prompt=Prompt(
            role="user",
            content="回答用户的问题：{{question}}\n不知道就说不知道。",
        ),
        tools=[KB.as_tool()],  # 模型自己决定查什么
        output_key="reply",
        name="agentic_rag",
    ),
    Echo("答：", Var("reply")),
    name="agentic",
)


def show_retrieval() -> None:
    '''不花 token 的部分：看看检索效果。'''
    print("=" * 64)
    print(f"知识库：{KB!r}")
    print("=" * 64)
    for question in ("怎么退款", "发货了还能退吗", "能开发票吗", "港澳台能送吗", "量子纠缠"):
        hits = KB.search(question, k=2, min_score=MIN_SCORE)
        print(f"\n问：{question}")
        if not hits:
            # 词法检索的固有局限：问法和资料用词完全不搭边就搜不到（或者分数太低）。
            # 真项目里把 KnowledgeBase.search 换成向量检索即可，工作流不用动。
            print("  （没搜到 → 工作流会走「资料不足」分支）")
            continue
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank}. [{hit.score:5.2f}] {hit.source}: {hit.text[:34]}…")


def main() -> int:
    show_retrieval()

    print()
    print("=" * 64)
    print("工作流结构")
    print("=" * 64)
    print(rag)

    print()
    print("=" * 64)
    print("跑显式检索版")
    print("=" * 64)
    agent = Agent()
    ctx = rag.run(agent=agent, question="我买的东西已经发货了，还能退款吗？")
    print()
    print("answer =", ctx["answer"])

    print()
    print("=" * 64)
    print("跑模型自主检索版")
    print("=" * 64)
    print(agentic.run(agent=agent, question="你们能开发票吗？")["reply"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
