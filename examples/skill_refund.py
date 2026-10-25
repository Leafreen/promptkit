'''一个真实点的 Skill 示例：退款处理。

    python examples/skill_refund.py --show    # 看两个 Skill 的描述和参数（不调模型）
    python examples/skill_refund.py           # 用脚本化假 agent 跑三单（不需要 Key）
    python examples/skill_refund.py --live    # 用真模型跑一单

这里有两个 Skill，正好代表两种形态：

    refund_policy   纯规则函数，**不碰模型**（装饰器自动识别出没有 ctx）
    refund_flow     完整工作流：查单 -> 判规则 -> 模型判物流状态 -> 分支话术 -> 组装结果

关键设计：**规则在前，模型在后**。

1. `refund_policy` 先按公司规则判定能不能退 —— 不能退就直接返回，
   一次模型调用都不花（`--fake` 里会看到被拒那单的调用次数是 0）。
2. 能退才进工作流：模型只负责「判断物流状态」和「写给客户看的话术」，
   不做「能不能退」的决定。
3. `action` 由代码根据状态拼出来，不交给模型编。

Skill 内部就是普通 Python：`Workflow`、`If/Else`、`LLMStep`、`Prompt` 都用得上。
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
from promptkit.core import If, LLMStep, Prompt, Var, Workflow  # noqa: E402
from promptkit.skill import SkillContext, SkillRegistry, skill  # noqa: E402

# ============================ 假订单库 ============================
#: 线上当然是查数据库；这里放一个字典，让示例自包含。
ORDERS = {
    "B2002": {
        "id": "B2002",
        "amount": 89,
        "days": 1,
        "note": "客户昨天下单，仓库已打包但还没交给快递，物流单号还没生成。",
    },
    "A1001": {
        "id": "A1001",
        "amount": 299,
        "days": 3,
        "note": "客户 3 天前下单，快递单号 SF1234，今天上午已从杭州发出，在途。",
    },
    "C3003": {
        "id": "C3003",
        "amount": 1299,
        "days": 21,
        "note": "客户 21 天前下单，快递单号 JD8899，已签收。",
    },
}


def lookup_order(order_id: str) -> dict:
    '''按单号查订单（找不到就明确抛错，别让后面拿到 None）。'''
    if order_id not in ORDERS:
        raise KeyError(f"查不到订单 {order_id!r}；演示里可用：{', '.join(ORDERS)}")
    return dict(ORDERS[order_id])


# ============================ 输出契约 ============================
class RefundPolicy(BaseModel):
    allowed: bool
    rule: str
    days_since_purchase: int
    note: str = ""


class OrderStatus(BaseModel):
    status: Literal["未发货", "已发货", "已签收"]
    reason: str = ""


class RefundResult(BaseModel):
    allowed: bool
    action: Literal["直接退款", "退货后退款", "拒绝"]
    status: str
    reply: str
    rule: str


# ============================ Skill 1：纯规则 ============================
@skill(
    name="refund_policy",
    description="按公司退款规则判断一单能不能退（纯规则，不调用模型）",
    tags=["policy"],
    output_model=RefundPolicy,
)
def refund_policy(order: dict, reason: str = "") -> RefundPolicy:
    '''7 天内走「无理由」直接退；超过 7 天转人工审核。

    注意这里**没有 ctx** —— 装饰器会识别出来，调用时不注入上下文。

    规则只用手上真有的字段（天数、金额、原因）。「发没发货」要靠读物流备注，
    那是下一步模型干的事 —— 所以「未发货就直接退」不在这里判，在流程末尾
    按模型给出的状态决定 `action`。
    '''
    days = int(order.get("days") or 0)
    if days <= 7:
        return RefundPolicy(
            allowed=True,
            rule="7 天无理由",
            days_since_purchase=days,
            note=f"原因：{reason}" if reason else "",
        )
    return RefundPolicy(
        allowed=False,
        rule=f"已超过 7 天（{days} 天）",
        days_since_purchase=days,
        note="需人工审核，建议走售后工单",
    )


# ============================ Skill 2：完整工作流 ============================
@skill(
    name="refund_flow",
    description="处理一笔退款：查单、判规则、按物流状态给方案和话术",
    tags=["support", "workflow"],
    output_model=RefundResult,
)
def refund_flow(
    order_id: str, reason: str = "质量问题", ctx: SkillContext = None
) -> RefundResult:
    # 1) 查单：用 ctx 里的工具，而不是自己去读数据库
    order = ctx.tool("lookup_order")(order_id)

    # 2) 规则先判：不能退就直接结束，**一次模型调用都不花**
    policy = ctx.call_skill("refund_policy", order=order, reason=reason)
    if not policy.allowed:
        return RefundResult(
            allowed=False,
            action="拒绝",
            status="未知",
            rule=policy.rule,
            reply=f"很抱歉，这笔订单{policy.rule}，暂时无法直接退款，已为您转售后审核。",
        )

    # 3) 工作流：模型只做「判状态」和「写话术」，不做决定
    workflow = Workflow(
        LLMStep(
            Prompt(
                role="user",
                content=(
                    "下面是这笔订单的物流备注：\n{{note}}\n\n"
                    "判断它现在的物流状态，只输出 status（未发货 / 已发货 / 已签收），"
                    "reason 用一句话说明依据。"
                ),
            ),
            OrderStatus,
            output_key="order_status",
        ),
        If(Var("order_status").attr("status") == "未发货").Then(
            LLMStep(
                Prompt(
                    role="user",
                    content=(
                        "给客户写一条退款确认：订单 {{order_id}}，金额 {{amount}} 元，"
                        "原因「{{reason}}」。两句话以内，直接给客户看，不要客套开场。"
                    ),
                ),
                output_key="reply",
            ),
        ).Else(
            LLMStep(
                Prompt(
                    role="user",
                    content=(
                        "给客户写一条退货指引：订单 {{order_id}}，金额 {{amount}} 元，"
                        "原因「{{reason}}」。说明先寄回、签收后退款，"
                        "两句话以内，直接给客户看。"
                    ),
                ),
                output_key="reply",
            ),
        ),
        name="refund_reply",
    )

    context = ctx.run(
        workflow,
        note=order["note"],
        order_id=order["id"],
        amount=order["amount"],
        reason=reason,
    )

    # 4) 组装：action 由代码按状态决定，不让模型编
    status = context["order_status"].status
    action = "直接退款" if status == "未发货" else "退货后退款"
    return RefundResult(
        allowed=True,
        action=action,
        status=status,
        reply=context["reply"],
        rule=policy.rule,
    )


# ============================ 脚本化假 agent ============================
class ScriptedAgent:
    '''不开 API 也能跑：按顺序回话，并记下被调了几次。'''

    vendor = "fake"
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        self.calls += 1
        if not self.replies:
            raise AssertionError("脚本回复用完了")
        return self.replies.pop(0)


# ============================ 演示 ============================
def build_registry() -> SkillRegistry:
    return SkillRegistry([refund_policy, refund_flow])


def show() -> int:
    registry = build_registry()
    print(f"注册表：{registry}\n")
    for item in registry:
        print(f"- {item.name}")
        print(f"  描述：{item.description}")
        print(f"  参数：{json.dumps(item.parameters, ensure_ascii=False)}")
        print(f"  返回：{json.dumps(item.returns, ensure_ascii=False) if item.returns else '（无）'}")
        print()
    return 0


def run_fake() -> int:
    registry = build_registry()
    agent = ScriptedAgent(
        [
            '{"status": "未发货", "reason": "仓库已打包，还没生成物流单号"}',
            "已为您原路退款 89 元，预计 1-3 个工作日到账。",
            '{"status": "已发货", "reason": "已有快递单号且在途"}',
            "请把商品寄回，签收后我们立即为您退款 299 元，运费由我们承担。",
        ]
    )
    ctx = SkillContext(agent=agent, skills=registry, tools={"lookup_order": lookup_order})

    for order_id in ("B2002", "A1001", "C3003"):
        before = agent.calls
        result = registry.execute(
            "refund_flow", {"order_id": order_id, "reason": "不想要了"}, ctx=ctx
        )
        print(f"== {order_id} ==")
        print(f"  能不能退：{result.allowed}（规则：{result.rule}）")
        print(f"  方案：{result.action}，状态：{result.status}")
        print(f"  话术：{result.reply}")
        print(f"  这次花了 {agent.calls - before} 次模型调用\n")

    print("== 轨迹 ==")
    for entry in ctx.trace:
        print(" ", entry)
    return 0


def run_live() -> int:
    registry = build_registry()
    agent = Agent(register_as_default=False)
    ctx = SkillContext(agent=agent, skills=registry, tools={"lookup_order": lookup_order})
    print(f"模型：{agent.vendor}/{agent.model}\n")

    result = registry.execute(
        "refund_flow", {"order_id": "A1001", "reason": "尺寸不合适"}, ctx=ctx
    )
    print(f"能不能退：{result.allowed}（{result.rule}）")
    print(f"方案：{result.action}（状态：{result.status}）")
    print(f"话术：{result.reply}")
    return 0


def main(argv) -> int:
    if "--show" in argv:
        return show()
    if "--live" in argv:
        return run_live()
    return run_fake()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
