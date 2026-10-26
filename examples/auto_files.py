'''让**模型自己决定**什么时候把成果写进文件。

对比两种「生成文件」：

| 方式 | 谁决定 | 什么时候用 |
| --- | --- | --- |
| `Save` / `Deliverable(save_to=...)` | **你**（写工作流时） | 你明确知道第几步该落盘 |
| `WriteFile` 工具 | **模型**（跑的时候） | 用户临时要一份他能带走的成果 |

后者才是「在合适的地方自动生成文件」：模型判断「用户要一份选题建议」，
就调 `write_file`，框架负责安全地写下去。

运行（需要 API Key + 网络）：
    python examples/auto_files.py
'''
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import ChatWorkflow  # noqa: E402
from promptkit.default_tools import WEB_SEARCH, WriteFile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

#: 沙箱根目录：模型只能写这下面（扩展名也限定）
writer = WriteFile(OUT_DIR, allow=("md", "json", "csv", "txt"))

chat = ChatWorkflow(
    role="市场调研分析大赛的导师",
    want="帮参赛者找选题；用户要一份能带走的成果时，用 write_file 存成文件",
    rules=[
        "像导师一样专业、耐心",
        "用户说「给我一份 / 导出 / 保存」时，调用 write_file 落地成文件",
        "写文件时文件名用中文、带合适扩展名（.md / .json / .csv）",
    ],
    tools=[writer, WEB_SEARCH],
    exit_words=["exit", "quit"],
)


if __name__ == "__main__":
    print("沙箱目录:", OUT_DIR)
    print("模型能看到的工具:", ["write_file", "web_search"])
    print()

    agent = Agent(
        start_prompts=[
            {
                "role": "system",
                "content": "你是一位市场调研分析大赛的导师。",
            }
        ]
    )
    chat.run(agent=agent)

    print()
    print("模型自己写了这些文件：")
    for item in writer.written or ["（这轮没写文件）"]:
        print("  •", item)
    print()
    print("清单文件:", os.path.join(OUT_DIR, "_index.json"))
