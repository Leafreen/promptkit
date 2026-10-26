'''让模型自己调用工具（function calling）。

运行：
    python examples/tool_calling.py
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import (  # noqa: E402
    Echo,
    LLMStep,
    Prompt,
    Var,
    Workflow,
)

# ====================== 工具本身 ======================
# 就是普通的 Python 函数：名字、docstring、类型标注会自动变成 JSON Schema
_ORDERS = {
    "12345": {"status": "已发货", "amount": 199.0},
    "67890": {"status": "未发货", "amount": 58.5},
}


def query_order(order_id: str) -> dict:
    """查询订单的状态和金额。order_id 是订单号。"""
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"没有订单 {order_id}"}
    return {"order_id": order_id, **order}


def refund(order_id: str) -> str:
    """对指定订单发起退款。"""
    order = _ORDERS.get(order_id)
    if order is None:
        return f"退款失败：没有订单 {order_id}"
    if order["status"] == "已发货":
        return "退款失败：订单已发货，需要先申请退货"
    return f"订单 {order_id} 已退款 {order['amount']} 元"


TOOLS = [query_order, refund]

# ====================== 工作流 ======================
wf = Workflow()

# Agent() 读取 DEEPSEEK_API_KEY / .env，并注册为默认 agent
agent = Agent()

wf.add(
    # 传了 tools，模型就会自己决定调哪个工具、调几次，直到能回答为止
    LLMStep(
        Prompt(
            role="user",
            content="帮我看看订单 67890 现在什么状态，能直接退款吗？能的话就退了。",
        ),
        tools=TOOLS,
        output_key="answer",
    ),
    Echo("答：", Var("answer")),

    # 同一个 agent，接着问第二轮（模型记得上文）
    LLMStep(
        Prompt(role="user", content="那订单 12345 呢？"),
        tools=TOOLS,
        output_key="answer",
    ),
    Echo("答：", Var("answer")),
)

if __name__ == "__main__":
    print(wf)
    print()
    wf.run()
