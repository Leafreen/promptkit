# examples/multimodal_chat.py
'''多模态对话示例：图片理解 -> 生成报告。

不依赖真实 API —— 用一个 FakeClient 演示完整工作流：

1. `Prompt(content=[TextPart, Var("image")])` 组合文本和图片部件
2. `LLMStep` 执行前做能力协商（`client.supports("image")`），不支持就抛
   `UnsupportedModality`，不静默降级成纯文本
3. `Var("image")` 拿到字符串 "cat.png" 时按扩展名推断成 `ImagePart`

跑法：`python examples/multimodal_chat.py`
换成真实模型：把 `make_fake_agent()` 换成 `Agent(vendor="doubao")`
（需要 `DOUBAO_API_KEY` 环境变量），其余代码一行不动。
'''
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from promptkit.agent import Agent
from promptkit.core.content import TextPart
from promptkit.core.display_sys.render import Verbosity
from promptkit.core.steps.actions.llm import LLMStep
from promptkit.core.steps.logics.var import Var
from promptkit.core.steps.workflow import Workflow
from promptkit.prompt import Prompt


class _Completion:
    '''最小可用的 chat completion 假返回。'''

    def __init__(self, content: str):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {
                "content": content, "role": "assistant", "tool_calls": None,
            })()})()
        ]
        self.usage = None


class FakeChatClient:
    '''假装支持图片理解的客户端：记录收到的消息，按队列回复。'''

    vendor = "fake"

    def __init__(self, replies: Optional[List[str]] = None):
        self.replies = list(replies) if replies else ["好的"]
        self.calls: List[Dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def supports(self, modality: str) -> bool:
        return modality in ("text", "image")

    def create(self, **kwargs: Any) -> _Completion:
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "好的"
        return _Completion(reply)


def make_fake_agent() -> Agent:
    '''真 Agent + 假客户端：走完整的渲染、协商、消息装配流程。'''
    agent = Agent(api_key="fake", model="fake", register_as_default=False)
    agent.client = FakeChatClient(  # type: ignore[assignment]
        replies=[
            "一只橘猫趴在键盘上睡觉",
            "# 图片报告\n\n图中是一只橘猫，建议先把它挪开再工作。",
        ]
    )
    return agent


def main() -> None:
    agent = make_fake_agent()

    workflow = Workflow(
        # 第一步：多模态 —— 文本指令 + 图片变量（按扩展名推断成 ImagePart）
        LLMStep(
            Prompt(
                role="user",
                content=[TextPart(text="用一句话描述这张图片。"), Var("image")],
            ),
            output_key="description",
        ),
        # 第二步：纯文本 —— 拿第一步的描述写报告
        LLMStep(
            Prompt(role="user", content="根据这段描述写一份简短报告：{{description}}"),
            output_key="report",
        ),
    )

    context = workflow.run(agent=agent, verbose=Verbosity.SILENT, image="cat.png")

    print("描述：", context["description"])
    print("报告：", context["report"])

    # 看一眼假客户端实际收到了什么：图片部件原样发出，没被拍平成文本
    sent = agent.client.calls[0]["messages"][-1]["content"]  # type: ignore[attr-defined]
    print("发出的消息部件：", [part["type"] for part in sent])


if __name__ == "__main__":
    main()
