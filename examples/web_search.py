'''联网搜索问答：让模型自己决定搜什么。

    python examples/web_search.py                     # 只演示检索（不花 token）
    python examples/web_search.py "今天的新闻"          # 检索 + 让模型回答

后端按环境变量自动选。配了 Key 就用 Key（国内的两个排前面）：

    BOCHA_API_KEY       博查，国内首选
    ZHIPUAI_API_KEY     智谱
    TAVILY_API_KEY / SERPER_API_KEY / BRAVE_API_KEY

都没配就走降级链：百度 → Bing 中国 → DuckDuckGo。
换顺序：PROMPTKIT_SEARCH_BACKENDS=bing,baidu

⚠️ 搜索结果会被塞进提示词。网页内容**不可信**，可能包含「忽略之前的指令」
这类注入文本 —— 别把搜索结果当指令执行。
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import JSON_ONLY  # noqa: E402
from promptkit.core import Echo, LLMStep, Prompt, Var, Workflow  # noqa: E402
from promptkit.default_tools import WEB_SEARCH  # noqa: E402
from promptkit.default_tools.web_search import SearchError, WebSearch  # noqa: E402

DEFAULT_QUESTION = "Python 3.13 有哪些新特性？"


class Answer(BaseModel):
    answer: str
    sources: List[str]  # 引用到的链接


# ====================== 工作流 ======================
# 显式检索 + 让模型基于结果回答，并强制给出出处
wf = Workflow(
    LLMStep[Answer](
        prompt=Prompt(
            role="user",
            content=(
                "回答下面的问题：{{question}}\n\n"
                "需要最新或你不确定的信息时，先调用 web_search 工具再回答。\n"
                "回答里必须给出你引用过的来源链接。"
            ),
        ),
        output_model=Answer,
        output_key="answer",
        tools=[WEB_SEARCH],
        cautions=[JSON_ONLY],
        name="web_qa",
    ),
    Echo("答：", Var("answer.answer")),
    Echo("来源：", Var("answer.sources")),
    name="web_search_qa",
)


def show_retrieval(question: str) -> None:
    '''不花 token 的部分：看看检索本身能拿到什么。'''
    search = WebSearch()
    print("=" * 64)
    print(f"后端：{search.resolve_backend()}")
    print("=" * 64)
    try:
        results = search.search(question, k=3)
    except SearchError as exc:
        print(f"搜索失败：{exc}")
        print()
        print("提示：免密钥的抓页面后端在国内经常被限流。")
        print("     配一个 BOCHA_API_KEY（博查）会稳很多。")
        return
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}] {item.title}")
        print(f"    {item.url}")
        print(f"    {item.snippet[:100]}")


def main(argv: List[str]) -> int:
    question = " ".join(argv) or DEFAULT_QUESTION

    show_retrieval(question)

    print()
    print("=" * 64)
    print("工作流结构")
    print("=" * 64)
    print(wf)

    print()
    print("=" * 64)
    print("让模型回答（需要 API Key + 网络）")
    print("=" * 64)
    try:
        ctx = wf.run(agent=Agent(), question=question)
    except Exception as exc:  # noqa: BLE001 - 演示脚本，失败就打印
        print(f"跑不起来：{type(exc).__name__}: {exc}")
        return 1

    answer = ctx["answer"]
    if isinstance(answer, Answer):
        print()
        print(answer.answer)
        print()
        for source in answer.sources:
            print(" -", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
