'''把「显示」变成事件流：Agent 只发事件，怎么显示由你决定。

运行（需要 API Key + 网络）：
    python examples/streaming_events.py
'''
import logging
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import EventBus, EventType, Workflow  # noqa: E402
from promptkit.core.display_sys.display import console_display, json_display  # noqa: E402
from promptkit.core import Echo, LLMStep, Prompt, Var  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")


def my_handler(event):
    '''自定义 handler：想记日志 / 发 WebSocket 都行。'''
    if event.type == EventType.TOOL_CALL_START:
        logger.info("tool=%s", event.data.get("name"))
    if event.type == EventType.THINKING_DELTA:
        logger.info("thinking: %s", event.data.get("text", "")[:40])


def main() -> int:
    question = "用一句话说说 Python 的 GIL 是什么。"

    print("=" * 60)
    print("1) 默认静默 —— 什么都不打印，只要结果")
    print("=" * 60)
    agent = Agent()
    answer = agent.ask(question)
    print("答案：", answer[:60], "…")

    print()
    print("=" * 60)
    print("2) 挂 console_display —— 看思考和回复")
    print("=" * 60)
    bus = EventBus()
    bus.on(console_display(show_thinking=True, show_tools=True))
    agent = Agent(bus=bus)
    agent.ask(question)
    print()

    print()
    print("=" * 60)
    print("3) 流式 —— 逐 token 拿事件")
    print("=" * 60)
    agent = Agent()
    text = ""
    for event in agent.ask_stream(question):
        if event.type == EventType.THINKING_DELTA:
            sys.stdout.write(f"\033[90m{event.data['text']}\033[0m")
        elif event.type == EventType.MESSAGE_DELTA:
            sys.stdout.write(event.data["text"])
            text += event.data["text"]
        elif event.type == EventType.MESSAGE_END:
            text = event.data["text"]
    print("\n（共 %d 字）" % len(text))

    print()
    print("=" * 60)
    print("4) 转发成 JSON 行（日志 / WebSocket / SSE 都能吃）")
    print("=" * 60)
    agent = Agent()
    for event in agent.ask_stream("用两个字回答：你好"):
        json_display()(event)

    print()
    print("=" * 60)
    print("5) 工作流级别 —— 每一步都看得见")
    print("=" * 60)
    wf = Workflow(
        LLMStep(Prompt(role="user", content=question), output_key="a"),
        Echo("答：", Var("a")),
    )
    wf.run(agent=Agent(bus=bus), verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
