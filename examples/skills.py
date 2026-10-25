'''Skill：把「提示词 + 控制流 + 工具 + 输出契约」封装成可注册、可描述、可执行的单元。

    python examples/skills.py --show     # 只看注册表和 Schema（不调模型）
    python examples/skills.py            # 用脚本化假 agent 跑一遍（不需要 API Key）
    python examples/skills.py --live     # 用真模型跑：Skill 直接调 + Agent 自己挑

这里定义三个 Skill：

    classify_sentiment(text)   结构化输出（Pydantic），靠 @skill(output_model=...)
    write_headline(topic)      纯文本输出
    brief(topic)               组合：先分类再起标题 —— Skill 调 Skill

`@skill` 会从函数签名自动生成 parameters，所以 `registry.to_tools(ctx)` 之后
Agent 就知道「有什么能力、要什么参数、返回什么」。
'''
import json
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

from typing import Literal  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent  # noqa: E402
from promptkit.skill import SkillContext, SkillRegistry, skill  # noqa: E402


# ============================ 输出契约 ============================
class Sentiment(BaseModel):
    '''情感分类的结构化结果。'''

    label: Literal["正面", "负面", "中性"]
    confidence: float


# ============================ 三个 Skill ============================
@skill(
    name="classify_sentiment",
    description="判断一段中文文本的情感倾向，返回 label 和 confidence",
    tags=["nlp", "classification"],
    output_model=Sentiment,
)
def classify_sentiment(text: str, ctx: SkillContext) -> Sentiment:
    return ctx.structured(
        "判断下面这段话的情感，只输出 label（正面/负面/中性）和 confidence（0-1）：\n"
        f"{text}",
        Sentiment,
    )


@skill(
    name="write_headline",
    description="给一个主题写一条 15 字以内的中文标题",
    tags=["writing"],
)
def write_headline(topic: str, tone: str = "专业", ctx: SkillContext = None) -> str:
    return ctx.ask(f"用「{tone}」的语气，给「{topic}」写一条 15 字以内的中文标题，只给标题。")


@skill(
    name="brief",
    description="对一个主题做完整简报：先判断它的情感基调，再起一条标题",
    tags=["composite"],
)
def brief(topic: str, ctx: SkillContext) -> str:
    '''组合 Skill：内部用 ctx.call_skill 调另外两个。'''
    sentiment = ctx.call_skill("classify_sentiment", text=topic)
    headline = ctx.call_skill("write_headline", topic=topic, tone="专业")
    return f"标题：{headline}\n基调：{sentiment.label}（{sentiment.confidence:.2f}）"


# ============================ 脚本化假 agent ============================
class ScriptedAgent:
    '''不开 API 也能跑：看到「情感」就回 JSON，否则回一条标题。'''

    vendor = "fake"
    model = "scripted"

    def __init__(self):
        self.prompts = []

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        text = prompt["content"] if isinstance(prompt, dict) else str(prompt)
        self.prompts.append(text)
        if "情感" in text:
            return json.dumps({"label": "正面", "confidence": 0.92}, ensure_ascii=False)
        return "提示词工程：让模型稳定交付"


# ============================ 演示 ============================
def build_registry() -> SkillRegistry:
    return SkillRegistry([classify_sentiment, write_headline, brief])


def show() -> int:
    registry = build_registry()
    print(f"注册表：{registry}\n")
    for item in registry:
        print(f"- {item.name}（tags={list(item.tags)}）")
        print(f"  描述：{item.description}")
        print(f"  参数：{json.dumps(item.parameters, ensure_ascii=False)}")
        if item.returns:
            print(f"  返回：{json.dumps(item.returns, ensure_ascii=False)[:120]}…")
        print()

    print("导成 OpenAI tools 的样子（第一条）：")
    print(json.dumps(registry.to_openai_tools()[0], ensure_ascii=False, indent=2))
    return 0


def run_fake() -> int:
    registry = build_registry()
    agent = ScriptedAgent()
    ctx = SkillContext(agent=agent, skills=registry)

    print("== 直接调 Skill ==")
    result = registry.execute("classify_sentiment", {"text": "这个功能太好用了"}, ctx=ctx)
    print(f"分类：{result.label}（{result.confidence}）")

    print("\n== Skill 调 Skill（brief 内部调两个）==")
    print(registry.execute("brief", {"topic": "提示词工程"}, ctx=ctx))

    print("\n== 作为 Tool 给 Agent 用 ==")
    tools = registry.to_tools(ctx)
    print("工具：" + ", ".join(tool.name for tool in tools))
    classify = next(tool for tool in tools if tool.name == "classify_sentiment")
    print("调用结果：" + classify.call({"text": "我真服了"}))

    print("\n== 轨迹 ==")
    for entry in ctx.trace:
        print(" ", entry)
    return 0


def run_live() -> int:
    registry = build_registry()
    agent = Agent(register_as_default=False)
    ctx = SkillContext(agent=agent, skills=registry)
    print(f"模型：{agent.vendor}/{agent.model}\n")

    print("== 直接调 brief（内部会调另外两个 Skill）==")
    print(registry.execute("brief", {"topic": "企业为什么要做提示词工程"}, ctx=ctx))

    print("\n== 把 Skill 交给 Agent，让它自己挑 ==")
    reply = agent.ask(
        "请用 write_headline 这个工具，给「提示词工程」写一条标题。",
        tools=registry.to_tools(ctx),
    )
    print(reply)
    return 0


def main(argv) -> int:
    if "--show" in argv:
        return show()
    if "--live" in argv:
        return run_live()
    return run_fake()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
