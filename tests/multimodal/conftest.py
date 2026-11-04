'''共享工具：多模态测试用的假客户端（不打网络）。

- `FakeChatClient`：对话客户端（`BaseClient` 的鸭子类型），`supports(modality)`
  可配置，`create()` 记录请求、按队列返回固定回复。
- `FakeMultimodalClient`：多模态客户端，`supports(modality, operation)` 按
  capabilities 声明，`generate()` / `understand()` 返回固定结果。
'''
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from promptkit.agent import Agent
from promptkit.client_sys.multimodal import Media, MultimodalResult


class _Completion:
    '''长得像 openai SDK 返回值的假 Completion。'''

    def __init__(self, content: str) -> None:
        message = type("Message", (), {})()
        message.role = "assistant"
        message.content = content
        message.tool_calls = None
        choice = type("Choice", (), {})()
        choice.message = message
        self.choices = [choice]
        self.usage = None


class FakeChatClient:
    '''假对话客户端：声明支持的模态，把每次 create 调用记下来。'''

    vendor = "fake"

    def __init__(
        self,
        replies: Optional[Iterable[str]] = None,
        modalities: Iterable[str] = ("text",),
    ) -> None:
        self.replies = list(replies) if replies else ["好的"]
        self.modalities = frozenset(modalities)
        self.calls: List[Dict[str, Any]] = []
        # Agent 按 client.chat.completions.create(...) 调，属性指回自己即可
        self.chat = self
        self.completions = self

    def supports(self, modality: str) -> bool:
        return modality in self.modalities

    def create(self, **kwargs: Any) -> _Completion:
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "好的"
        return _Completion(reply)


def make_agent(
    replies: Optional[Iterable[str]] = None,
    modalities: Iterable[str] = ("text",),
) -> Agent:
    '''造一个挂 FakeChatClient 的 Agent（不打网络、不注册为默认实例）。'''
    agent = Agent(api_key="fake", model="fake", register_as_default=False)
    agent.client = FakeChatClient(replies, modalities)  # type: ignore[assignment]
    return agent


def client_of(agent: Agent) -> FakeChatClient:
    '''取回挂在 Agent 上的假客户端，方便断言调用记录。'''
    return agent.client  # type: ignore[return-value]


class FakeMultimodalClient:
    '''假多模态客户端：按 capabilities 声明能力，返回固定结果并记录请求。'''

    vendor = "fake-mm"

    def __init__(
        self,
        capabilities: Iterable[Tuple[str, str]] = (),
        *,
        generate_media: Optional[Media] = None,
        understand_text: str = "这是一段描述",
    ) -> None:
        self.capabilities = frozenset(capabilities)
        self.generate_media = generate_media
        self.understand_text = understand_text
        self.requests: List[Tuple[Any, ...]] = []

    def supports(self, modality: str, operation: str) -> bool:
        return (modality, operation) in self.capabilities

    def generate(self, modality: str, prompt: str, **params: Any) -> MultimodalResult:
        self.requests.append(("generate", modality, prompt, params))
        return MultimodalResult(
            operation="generate",
            modality=modality,
            model=params.get("model"),
            media=self.generate_media
            or Media.from_url(f"https://example.com/generated.{modality}"),
        )

    def understand(
        self,
        modality: str,
        source: Any = None,
        prompt: Optional[str] = None,
        **params: Any,
    ) -> MultimodalResult:
        self.requests.append(("understand", modality, source, prompt, params))
        return MultimodalResult(
            operation="understand",
            modality=modality,
            text=self.understand_text,
        )
