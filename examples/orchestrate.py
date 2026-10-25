'''多代理编排：`Plan` → `Dispatch` → `Aggregate`。

    python examples/orchestrate.py            # 固定模板（可测、成本可预测）
    python examples/orchestrate.py --plan     # 让模型自己拆任务（LLM 规划）
    python examples/orchestrate.py --show     # 只打印工作流结构，不调模型

固定模板跑的是三个子代理 + 一个汇总：

    research (researcher)  ──>  writer (writer)  ──>  reviewer (reviewer)
                                                              │
                                             AggregateStep <──┘

每个任务的 agent 都由 `RoleFactory` **现造一个**（独立 history），任务之间只通过
`{{变量}}` 显式传值；编排器负责依赖顺序、失败降级和 token 预算。

`--plan` 模式下多一步：让一个规划 agent 把需求拆成 JSON 任务列表（最多 4 个），
再用同一个 `ExecutePlan` 跑 —— 这就是「规划与执行分离」：规划器只产出计划，
编排器只管跑。

需要模型：`Agent()`（默认 DeepSeek，读 `.env` / `DEEPSEEK_API_KEY`）。
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import Echo, Var, Workflow  # noqa: E402
from promptkit.orchestrate import (  # noqa: E402
    AggregateStep,
    ExecutePlan,
    OrchestratorConfig,
    Plan,
    PlanStep,
    RoleFactory,
    Task,
    sub_agent,
)

BRIEF = (
    "围绕「为什么企业需要提示词工程」，先给出 3 条一句话理由，"
    "再写一段 150 字左右的推广文案，最后自查一遍。"
)

ROLES = {
    "researcher": "你是市场调研员。只给事实、理由和一句话结论，不要客套和铺垫。",
    "writer": "你是资深中文文案。输出短文案，不超过 200 字，不要小标题和列表。",
    "reviewer": "你是严格的内容审稿人。指出具体问题（引用原句），并给出改后的版本。",
    "general": "你是通用助手，回答简洁、直接，不编造事实。",
}

#: 固定模板：拆任务这件事写死在代码里（推荐先用这一种）
TEMPLATE = Plan.from_template(
    [
        Task(
            "research",
            "给出 3 条「企业需要提示词工程」的理由，每条一句话。",
            role="researcher",
            output_key="research",
        ),
        Task(
            "write",
            "根据这些理由写一段 150 字左右的推广文案：\n{{research}}",
            role="writer",
            depends_on=("research",),
            output_key="draft",
        ),
        Task(
            "review",
            "审一下这段文案，指出一个具体问题并给出改后的版本：\n{{draft}}",
            role="reviewer",
            depends_on=("write",),
            output_key="reviewed",
        ),
    ],
    task=BRIEF,
)


def config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_concurrency=1,          # 串行：依赖链上本来也没有能并行的
        max_total_tokens=20000,     # 超了后面的任务会被标成 skipped，而不是硬跑
        per_task_timeout=180,
    )


def build(use_planner: bool, planner: Agent, aggregator: Agent) -> Workflow:
    factory = RoleFactory(ROLES, default_system=ROLES["general"])

    steps = []
    if use_planner:
        steps.append(
            PlanStep(
                task=BRIEF,
                planner=planner,
                max_tasks=4,
                output_key="plan",
            )
        )
        steps.append(Echo("计划：{{plan}}"))
        steps.append(
            ExecutePlan(
                plan=Var("plan"),
                agent_factory=factory,
                config=config(),
            )
        )
    else:
        steps.append(
            ExecutePlan(plan=TEMPLATE, agent_factory=factory, config=config())
        )

    steps.append(
        AggregateStep(
            results_key="task_results",
            agent=aggregator,
            output_key="final",
            instructions=(
                "把下面几份子任务结果合成一份「可以交给老板」的说明："
                "先给结论，再给文案，最后列出待确认的问题。不要编造没出现过的信息。"
            ),
        )
    )
    return Workflow(*steps, name="multi_agent_demo")


def main(argv) -> int:
    verbose = "--verbose" in argv
    use_planner = "--plan" in argv

    planner = Agent(register_as_default=False)
    aggregator = sub_agent(
        "你是编辑：擅长把多份材料合成一份连贯、可信、不重复的说明。",
        model=planner.model,
        vendor=planner.vendor,
    )
    workflow = build(use_planner, planner, aggregator)

    if "--show" in argv:
        print(workflow.render())
        return 0

    print(f"模型：{planner.vendor}/{planner.model}")
    print(f"模式：{'LLM 规划' if use_planner else '固定模板'}（最多 4 个任务，预算 20000 tokens）\n")
    print(workflow.render())
    print()

    context = workflow.run(agent=planner, verbose=verbose)
    result = context["plan_result"]

    print("== 每个子任务 ==")
    print(result.summary())
    for item in result.results:
        if not item.ok:
            continue
        text = " ".join(str(item.output).split())
        print(f"\n--- {item.name} ---\n{text[:400]}{'…' if len(text) > 400 else ''}")

    print("\n== 汇总 ==")
    print(context["final"])
    print(f"\n（总 tokens：{result.tokens}，用时 {result.elapsed:.1f}s）")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
