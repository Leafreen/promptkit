'''多轮对话里，把一个应用从「需求 + 验收测试」做到「测试全绿」。

    python examples/coding_chat.py                       # 交互式聊
    python examples/coding_chat.py --fresh               # 清空工作区和会话，从零开始
    python examples/coding_chat.py --say "..." --say "..."   # 脚本化多轮（演示 / CI）
    python examples/coding_chat.py --show                # 只看工作区现状 + 跑一遍测试

第一次运行会在 `.tmp/coding-chat/workspace/` 里放好三样东西：

    README.md             需求（Todo 类要提供什么）
    tests/test_todo.py    验收测试（4 条）
    todo.py               空壳（`raise NotImplementedError`）

然后进入 `ChatWorkflow`：模型自己决定**读哪个文件、怎么改、什么时候跑测试** ——
读/搜/写/打补丁/跑测试都是工具（`default_coding_tools`），不是写死的顺序。

会话存在 `.tmp/coding-chat/session.json`：退出再进来还能接着聊（`--fresh` 才清空）。

注意：这是**真模型 + 真文件 + 真测试**，所以慢（一轮十几秒到一两分钟都正常）。
想先看「不用对话的一次性闭环」，看 `examples/coding_agent.py`。
'''
import os
import shutil
import subprocess
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.coding import RunTests, default_coding_tools  # noqa: E402
from promptkit.core import ChatWorkflow, Workflow, settings  # noqa: E402

# 编码任务的工具调用天然多（读文件 ×N → 改 → 跑测试 → 修 → 再跑），
# 默认的 8 轮很容易不够。这里给这只「编码助手」放宽到 20 轮。
settings.max_tool_rounds = 20

BASE_DIR = os.path.join(PROJECT_DIR, ".tmp", "coding-chat")
WORKSPACE = os.path.join(BASE_DIR, "workspace")
SESSION = os.path.join(BASE_DIR, "session.json")
TEST_COMMAND = "python -m unittest discover -s . -q"

README = '''# todo-cli

一个极简的命令行待办工具。请实现 `todo.py` 里的 `Todo` 类：

| 方法 | 行为 |
| --- | --- |
| `Todo(path)` | `path` 是 JSON 数据文件；文件不存在就当空列表，父目录自动创建 |
| `add(text)` | 追加一条 `{"text": text, "done": false}`，并**立刻落盘** |
| `list()` | 返回全部条目（`list[dict]`） |
| `done(index)` | 把第 index 条标记为 `done=True` 并落盘；下标越界抛 `IndexError` |
| `clear()` | 清空所有条目并落盘 |

要求：

- 只用标准库；数据是 UTF-8 JSON 数组
- 重新 `Todo(path)` 打开要能读到上次的数据
- 验收标准就是 `tests/test_todo.py`（`python -m unittest discover -s . -q` 必须通过）
'''

TESTS = '''import json
import os
import tempfile
import unittest

from todo import Todo


class TestTodo(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(prefix="todo-"), "data.json")
        self.todo = Todo(self.path)

    def test_add_and_list(self):
        self.todo.add("写代码")
        self.assertEqual([item["text"] for item in self.todo.list()], ["写代码"])

    def test_persists_across_instances(self):
        self.todo.add("a")
        again = Todo(self.path)
        self.assertEqual(len(again.list()), 1)

    def test_done_marks_item(self):
        self.todo.add("a")
        self.todo.done(0)
        self.assertTrue(self.todo.list()[0]["done"])

    def test_done_out_of_range(self):
        with self.assertRaises(IndexError):
            self.todo.done(3)

    def test_clear(self):
        self.todo.add("a")
        self.todo.clear()
        self.assertEqual(self.todo.list(), [])


if __name__ == "__main__":
    unittest.main()
'''

STUB = '''"""todo-cli：待办工具（还没实现）。"""


class Todo:
    def __init__(self, path):
        raise NotImplementedError("请实现 Todo")


def main(argv=None):
    raise NotImplementedError("命令行入口还没实现")
'''


# ============================ 工作区 ============================
def prepare(*, fresh: bool) -> None:
    '''准备演示项目：已存在就复用（方便多轮续聊），--fresh 才覆盖。'''
    if fresh and os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(os.path.join(WORKSPACE, "tests"), exist_ok=True)

    seed = {
        "README.md": README,
        os.path.join("tests", "__init__.py"): "",   # 让 discover 能递归进 tests/
        os.path.join("tests", "test_todo.py"): TESTS,
        "todo.py": STUB,
    }
    for relative, content in seed.items():
        target = os.path.join(WORKSPACE, relative)
        if os.path.exists(target):
            continue
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)


def run_tests(command: str = TEST_COMMAND) -> dict:
    '''直接跑一遍测试（不走步骤，用来显示「现在到底过没过」）。'''
    result = subprocess.run(
        command, shell=True, cwd=WORKSPACE, capture_output=True
    )
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    return {"passed": result.returncode == 0, "output": output, "code": result.returncode}


def show() -> int:
    print(f"工作区：{WORKSPACE}")
    print("文件：")
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in sorted(files):
            relative = os.path.relpath(os.path.join(root, name), WORKSPACE)
            print("  " + relative.replace("\\", "/"))

    result = run_tests()
    print(f"\n测试（{'通过' if result['passed'] else '未通过'}）：")
    print("\n".join(result["output"].strip().splitlines()[-12:]))
    return 0 if result["passed"] else 1


# ============================ 对话 ============================
def build_chat() -> ChatWorkflow:
    '''一个会自己读文件、改代码、跑测试的编码助手。'''
    tools = default_coding_tools(
        WORKSPACE,
        allow_write=True,                 # 工具默认只读，这里显式开写
        allow_commands=["python"],        # 只允许跑 python
        test_command=TEST_COMMAND,
    )
    return ChatWorkflow(
        role="资深 Python 工程师",
        want="按 README 的需求把 todo.py 实现到验收测试全绿",
        rules=[
            "先读 README.md 和 tests/ 里的测试，再动手改代码",
            f"只改 {WORKSPACE} 里的文件，测试命令是 `{TEST_COMMAND}`",
            "每次改完代码都要跑一遍测试，用测试输出决定下一步",
            "不要修改 tests/ 里的验收测试（除非我明确要求加测试）",
            "改现有文件用 apply_diff，新建文件用 write_file",
            "测试全绿后再回复我「完成」，并说明改了哪些文件",
        ],
        tools=tools,
        read_prompt="你：",
        exit_words=["exit", "quit", "结束"],
        max_turns=12,
        session=SESSION,
        name="coding_chat",
    )


def feed(messages):
    '''脚本化喂对话：ChatLoop 是读 stdin 的，这里临时把 input() 换成队列。'''
    from unittest import mock

    return mock.patch(
        "builtins.input", side_effect=[*messages, "exit"]
    )


def chat(messages=None, *, verbose: bool = False) -> int:
    agent = Agent(register_as_default=False)
    workflow = build_chat()
    print(f"模型：{agent.vendor}/{agent.model}")
    print(f"工作区：{WORKSPACE}\n")

    if messages:
        with feed(messages):
            context = workflow.run(agent=agent, verbose=verbose)
    else:
        context = workflow.run(agent=agent, verbose=verbose)

    print("\n== 这一轮之后的测试 ==")
    final = Workflow(
        RunTests(TEST_COMMAND, base_dir=WORKSPACE, allowlist=["python"])
    ).run()
    result = final["test_result"]
    print(f"通过：{result['passed']}")
    print("\n".join(result["output"].strip().splitlines()[-10:]))
    print(f"\n记忆/会话文件：{SESSION}（下次直接跑就能接着聊）")
    return 0 if result["passed"] else 1


# ============================ 入口 ============================
def main(argv) -> int:
    fresh = "--fresh" in argv
    prepare(fresh=fresh)

    if "--show" in argv:
        return show()

    messages = []
    index = 1
    while index < len(argv):
        if argv[index] == "--say" and index + 1 < len(argv):
            messages.append(argv[index + 1])
            index += 2
            continue
        index += 1
    return chat(messages or None, verbose="--verbose" in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
