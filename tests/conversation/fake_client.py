'''测试用 FakeClient：全部本地响应，绝不访问真实 LLM API。'''
from types import SimpleNamespace
from typing import Any


class FakeCompletions:
    '''模仿 `client.chat.completions`，按顺序返回预设回复。'''

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeClient 没有更多预设回复了")
        text = self.replies.pop(0)
        message = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class FakeClient:
    '''长得像厂商客户端，但只认预设回复。'''

    def __init__(self, replies: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(replies))


def make_agent(replies: list[str]) -> Any:
    '''建一个不连网、不注册为默认的 Agent，并挂上 FakeClient。'''
    from promptkit.agent import Agent

    agent = Agent(api_key="sk-test", register_as_default=False)
    agent.client = FakeClient(replies)
    return agent
