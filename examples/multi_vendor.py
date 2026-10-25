'''同一个工作流，换三家厂商跑 —— 换 Agent 就够了，Workflow 一行都不用改。

运行：
    set DEEPSEEK_API_KEY=sk-xxxx       # 有哪家的密钥就跑哪家
    set OPENAI_API_KEY=sk-xxxx
    set ANTHROPIC_API_KEY=sk-ant-xxxx
    python examples/multi_vendor.py                # 有密钥的都跑一遍
    python examples/multi_vendor.py anthropic      # 只跑指定厂商

这个示例想说明的是**分层**：`Workflow` 只认 `agent.ask()`，厂商差异全在
`Agent` 背后的客户端（`client_sys/`）里消化掉了：

    Workflow ──> Agent ──> client_sys.OpenAIClient     （OpenAI / DeepSeek / 兼容端点）
                       └─> client_sys.AnthropicClient  （Claude，协议翻译）
'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptkit.agent import Agent  # noqa: E402
from promptkit.client_sys import ClientError  # noqa: E402
from promptkit.core import Echo, LLMStep, Set, Workflow  # noqa: E402
from promptkit.prompt import Prompt  # noqa: E402


def build_workflow() -> Workflow:
    '''一份和厂商无关的工作流：问一句，打印回答。'''
    return Workflow(
        Set("topic", "为什么天空是蓝色的？"),
        # LLMStep 会把结果写进 context["output"]
        LLMStep(Prompt(role="user", content="用一句话回答：{{topic}}")),
        Echo("{{output}}"),
    )


def main(argv):
    targets = argv[1:] or ["deepseek", "openai", "anthropic"]
    workflow = build_workflow()

    for name in targets:
        try:
            agent = Agent(vendor=name, register_as_default=False)
        except ValueError as exc:  # 没有这家的密钥
            print(f"[{name}] 跳过：{exc}")
            continue

        print(f"\n=== {name} / {agent.model} ===")
        try:
            workflow.run(agent=agent)
        except ClientError as exc:  # 网络 / 鉴权 / 余额不足
            print(f"[{name}] 调用失败：{exc}")


if __name__ == "__main__":
    main(sys.argv)
