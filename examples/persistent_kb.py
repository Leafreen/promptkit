'''持久化知识库：**建一次，反复用**。

⚠️ 知识库是**实验性功能，默认关闭**，所以下面要先 enable 一下。

    python examples/persistent_kb.py build      # 扫 docs 目录 → kb.json（只处理新文件）
    python examples/persistent_kb.py search 退款  # 查一下，不花 token
    python examples/persistent_kb.py stats       # 看看库里有什么
    python examples/persistent_kb.py ask 已经发货了还能退吗

`build` 是**增量**的：按「修改时间 + 大小」判断文件变没变，
所以反复跑不会重复灌，改过的文件会自动替换掉旧内容，删掉的文件会被清出去。

运行 ask 需要 API Key + 网络。
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from typing import List  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from promptkit import features  # noqa: E402

features.enable("knowledge")  # ← 实验性功能，默认关闭

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import JSON_ONLY  # noqa: E402
from promptkit.core import (  # noqa: E402
    Call,
    Echo,
    Goto,
    If,
    Label,
    Len,
    LLMStep,
    Not,
    Prompt,
    Set,
    Var,
    Workflow,
)
from promptkit.knowledge import KnowledgeBase  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "kb_docs")
DEFAULT_KB = os.path.join(HERE, "kb.json")

#: 低于这个分数就当作「没搜到」（词法检索里不相关查询也会拿到 0.3~0.5 分）
MIN_SCORE = 1.0


class Answer(BaseModel):
    answer: str
    used: List[int]
    enough: bool


# ====================== 建库 ======================
def build(kb_path: str, docs: str, *, force: bool = False, with_index: bool = False) -> int:
    # 已经有库就加载了接着同步；否则从空库开始
    if os.path.exists(kb_path):
        kb = KnowledgeBase.load(kb_path)
        print(f"已加载 {kb_path}：{len(kb)} 块")
    else:
        kb = KnowledgeBase(max_chars=300, overlap=60)
        print("新建知识库")

    stats = kb.sync_dir(docs, force=force)
    kb.save(kb_path, include_index=with_index)

    print(
        f"同步 {docs} → 新增 {stats['added']}，更新 {stats['updated']}，"
        f"跳过 {stats['skipped']}，移除 {stats['removed']}"
    )
    print(f"保存到 {kb_path}（{os.path.getsize(kb_path) / 1024:.1f} KB）")
    print_kb_stats(kb)
    return 0


def print_kb_stats(kb: KnowledgeBase) -> None:
    for key, value in kb.stats().items():
        print(f"  {key:12} {value}")


# ====================== 用库 ======================
def load(kb_path: str) -> KnowledgeBase:
    if not os.path.exists(kb_path):
        raise SystemExit(f"没有 {kb_path}，先跑：python {__file__} build")
    return KnowledgeBase.load(kb_path)


def search(kb_path: str, question: str, k: int = 3) -> int:
    kb = load(kb_path)
    hits = kb.search(question, k=k, min_score=MIN_SCORE)
    print(f"问：{question}")
    if not hits:
        print("  （没搜到）")
        return 0
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank}. [{hit.score:5.2f}] {hit.source}")
        print(f"     {hit.text[:70]}…")
    return 0


def build_workflow(kb: KnowledgeBase) -> Workflow:
    '''问答工作流。和 non-persistent 版唯一的区别是 KB 来自磁盘。'''
    return Workflow(
        Set("hits", Call(kb.search, Var("question"), k=4, min_score=MIN_SCORE)),
        Echo("（检索到 ", Call(len, Var("hits")), " 段资料）"),

        If(Len(Var("hits")) == 0).Then(
            Set("answer", "知识库里没有和这个问题相关的内容。"),
            Goto("done"),
        ),

        Set("context", Call(KnowledgeBase.format, Var("hits"), query=Var("question"))),
        LLMStep[Answer](
            prompt=Prompt(
                role="user",
                content=(
                    "只根据下面的资料回答，资料里没有的不要编。"
                    "回答里要给出用到资料的编号。\n\n"
                    "资料：\n{{context}}\n\n问题：{{question}}"
                ),
            ),
            output_model=Answer,
            output_key="answer",
            cautions=[JSON_ONLY],
            name="answer_from_kb",
        ),
        If(Not(Var("answer.enough"))).Then(Echo("⚠️ 资料不足，建议转人工。")),

        Label("done"),
        Echo("答：", Var("answer.answer")),
        Echo("引用：", Var("answer.used")),
        name="persistent_rag",
    )


def ask(kb_path: str, question: str) -> int:
    kb = load(kb_path)
    workflow = build_workflow(kb)
    ctx = workflow.run(agent=Agent(), question=question)
    answer = ctx["answer"]
    print()
    print(answer.answer if isinstance(answer, Answer) else answer)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="持久化知识库示例")
    parser.add_argument("--kb", default=DEFAULT_KB, help=f"知识库文件（默认 {DEFAULT_KB}）")
    parser.add_argument("--docs", default=DOCS, help="文档目录")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="扫描文档目录并保存")
    build_parser.add_argument("--force", action="store_true", help="无视台账，全部重灌")
    build_parser.add_argument(
        "--with-index", action="store_true", help="连倒排索引一起存（文件更大，加载更快）"
    )

    search_parser = sub.add_parser("search", help="只检索，不调用模型")
    search_parser.add_argument("question")
    search_parser.add_argument("-k", type=int, default=3)

    sub.add_parser("stats", help="看库里有什么")
    ask_parser = sub.add_parser("ask", help="检索 + 让模型回答（需要 API Key）")
    ask_parser.add_argument("question")

    args = parser.parse_args(argv)

    if args.command == "build":
        return build(args.kb, args.docs, force=args.force, with_index=args.with_index)
    if args.command == "search":
        return search(args.kb, args.question, args.k)
    if args.command == "stats":
        print_kb_stats(load(args.kb))
        return 0
    if args.command == "ask":
        return ask(args.kb, args.question)
    parser.error(f"不认识 {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
