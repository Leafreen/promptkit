'''Caution（事前约束）+ Formatter（事后解析）+ output_model（类型契约）。

运行（需要 API Key + 网络）：
    python examples/cautions.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import (  # noqa: E402
    JSON_ONLY,
    Language,
    MaxLength,
    NoExtraText,
    OutputFormat,
    Tone,
    json_only,
)
from promptkit.core import (  # noqa: E402
    Echo,
    If,
    LLMStep,
    Prompt,
    Var,
    Workflow,
    clear_default_cautions,
    set_default_cautions,
)
from promptkit.core.model_action.caution_related.formatter import RegexFormatter  # noqa: E402


class Intent(BaseModel):
    label: str  # 退款 / 咨询 / 投诉


class Reply(BaseModel):
    content: str


# ============ 1. 约束是对象：组合、复用、可测 ============
STRICT_JSON = OutputFormat("JSON") + NoExtraText() + Language("中文")

# ============ 2. 整个工作流在模块级声明，一眼看全 ============
wf = Workflow(
    # 用 Caution 事前引导，用 output_model 事后兜底
    LLMStep[Intent](
        prompt=Prompt(
            role="user",
            content="判断下面这句话的意图：{{text}}\n只输出 label 字段。",
        ),
        output_model=Intent,
        output_key="intent",
        cautions=[JSON_ONLY],  # 预置组合
        auto_schema=True,  # 顺手把 JSON Schema 也贴进去
        name="classify_intent",
    ),
    Echo("意图 = ", "{{intent.label}}"),

    # `+` 临时拼一套约束
    If(Var("intent.label") == "退款").Then(
        LLMStep[Reply](
            prompt=Prompt(role="user", content="给用户写一句退款成功通知。"),
            output_model=Reply,
            output_key="reply",
            cautions=STRICT_JSON + Tone("礼貌"),
            name="refund_reply",
        ),
    ).Else(
        LLMStep[Reply](
            prompt=Prompt(role="user", content="给用户写一句通用回复。"),
            output_model=Reply,
            output_key="reply",
            cautions=[json_only(), MaxLength(50)],
            name="generic_reply",
        ),
    ),
    Echo("回复 = ", "{{reply.content}}"),

    # 换 Formatter：不输出 JSON 也能解析
    LLMStep(
        prompt=Prompt(role="user", content="用一个词形容情绪：{{text}}"),
        output_model=Intent,
        output_key="emotion",
        cautions=[NoExtraText(), Language("中文")],
        formatter=RegexFormatter(r"情绪：(?P<label>\w+)"),
        name="emotion_by_regex",
    ),
    Echo("情绪 = ", "{{emotion.label}}"),

    # Workflow 级约束：里面所有 LLMStep 共享
    cautions=[Tone("简洁")],
    name="intent_pipeline",
)


def main() -> int:
    print("=" * 60)
    print("约束是对象：这段文本就是会被追加到提示词末尾的内容")
    print("=" * 60)
    print(JSON_ONLY.to_text())
    print()
    print("--- 上面那个 LLMStep 真正发出去的提示词 ---")
    print(wf.steps[0].build_prompt())

    print()
    print("=" * 60)
    print("步骤树")
    print("=" * 60)
    print(wf)

    print()
    print("=" * 60)
    print("跑起来")
    print("=" * 60)
    # 全局默认约束：所有 LLMStep 自动带上（测试里请用 using_default_cautions 包住）
    set_default_cautions([NoExtraText()])
    try:
        ctx = wf.run(agent=Agent(), text="我要退款，这个订单不想要了")
    finally:
        clear_default_cautions()

    print()
    print("intent  =", ctx["intent"])
    print("reply   =", ctx["reply"])
    print("emotion =", ctx["emotion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
