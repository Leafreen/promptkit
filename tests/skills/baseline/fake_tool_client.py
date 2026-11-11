'''基线测试用的假客户端：第一轮返回 tool_calls，第二轮返回纯文本。'''
import json
from types import SimpleNamespace
from typing import Any


class FakeToolCall:
    '''长得像 OpenAI 的 tool_call；Agent 会调 model_dump() 记进历史。'''

    def __init__(self, name: str, arguments: dict, call_id: str = "call_1") -> None:
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeCompletions:
    '''按脚本出牌：先 tool_calls 后文本；脚本用完重复最后一条。'''

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        message = self.script[index]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None
        )


def make_tool_agent(script: list) -> Any:
    '''建一个不联网的 Agent，chat 按 script 顺序返回 SimpleNamespace 消息。'''
    from promptkit.agent import Agent

    agent = Agent(api_key="sk-test", register_as_default=False)
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(script))
    )
    return agent


def tool_call_message(name: str, arguments: dict) -> Any:
    return SimpleNamespace(content="", tool_calls=[FakeToolCall(name, arguments)])


def text_message(text: str) -> Any:
    return SimpleNamespace(content=text, tool_calls=None)
