'''用 If(...).Then(...).Else(...) 声明式地写客服意图路由。

对照 __main__.py 里手写 if/else 的版本，这里是同样的逻辑，但分支结构被声明成了
一棵步骤树，可以整体复用、嵌套和测试。

运行：
    python examples/refund_workflow.py
    python examples/refund_workflow.py "我要投诉"
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    Condition,
    FunctionStep,
    If,
    LLMStep,
    Workflow,
)
from promptkit.prompt import Prompt  # noqa: E402


class Intent(BaseModel):
    label: str  # "退款" / "咨询" / "投诉"


class Order(BaseModel):
    status: str  # "已发货" / "未发货"


# ====================== 条件 ======================
class NeedRefund(Condition):
    name = "need_refund"

    def evaluate(self, context) -> bool:
        return context["intent"].label.strip() == "退款"


class NeedComplaint(Condition):
    name = "need_complaint"

    def evaluate(self, context) -> bool:
        return context["intent"].label.strip() == "投诉"


class Shipped(Condition):
    name = "shipped"

    def evaluate(self, context) -> bool:
        return context["order"].status.strip() == "已发货"


# ====================== 工作流 ======================
workflow = Workflow(
    # 第一步：识别意图，结果写入 context["intent"]
    LLMStep(
        Prompt(
            role="user",
            content=(
                "判断下面这句话的意图：{{text}}\n"
                '只输出 JSON，例如 {"label": "退款"}，'
                "label 只能是 退款、咨询、投诉 之一。"
            ),
        ),
        Intent,
        output_key="intent",
    ),
    If(NeedRefund()).Then(
        # 退款：先查订单状态，再按状态回复
        LLMStep(
            Prompt(
                role="user",
                content=(
                    "根据客户这句话判断订单状态：{{text}}\n"
                    '只输出 JSON，例如 {"status": "已发货"}，'
                    "status 只能是 已发货、未发货 之一。"
                ),
            ),
            Order,
            output_key="order",
        ),
        If(Shipped()).Then(
            FunctionStep(lambda _ctx: "需要先申请退货。", output_key="reply"),
        ).Else(
            FunctionStep(lambda _ctx: "可以退款。", output_key="reply"),
        ),
    ).Else(
        # 非退款：再看是不是投诉
        If(NeedComplaint()).Then(
            FunctionStep(lambda _ctx: "已转投诉专员。", output_key="reply"),
        ).Else(
            FunctionStep(lambda _ctx: "请问有什么可以帮您？", output_key="reply"),
        )
    ),
)


def main(argv) -> int:
    messages = argv[1:] or ["我买的杯子不合适，想退款，订单号 12345"]
    agent = Agent()  # 读取 DEEPSEEK_API_KEY / .env

    for message in messages:
        print(f"用户：{message}")
        context = workflow.run(agent=agent, text=message)
        print(f"客服：{context['reply']}")
        print(f"      执行轨迹：{' -> '.join(context.trace)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
