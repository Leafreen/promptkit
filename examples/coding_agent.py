'''一个真的会「改代码 + 跑测试 + 失败再改」的编码工作流。

    python examples/coding_agent.py

它会：

1. 在 `.tmp/coding-demo/` 里生成一个**有小 bug** 的项目（`add` 写成了减法）
2. 先跑一遍测试，看到它失败
3. 让模型生成 unified diff -> 干跑校验 -> 应用 -> 跑测试
4. 失败就带着测试输出再改一轮（最多 3 轮）
5. 打印最终代码、轮次和测试输出

需要模型：`Agent()`（默认 DeepSeek，读 `.env` / `DEEPSEEK_API_KEY`）。
**不需要**装 pytest：测试用标准库 `unittest`，命令是 `python -m unittest`。
'''
import os
import shutil
import subprocess
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

from promptkit.agent import Agent  # noqa: E402
from promptkit.cautions import Custom  # noqa: E402
from promptkit.coding import (  # noqa: E402
    ApplyDiff,
    CodingStep,
    FixLoop,
    ReadFileStep,
    RunTests,
)
from promptkit.core import Var, Workflow, Verbosity  # noqa: E402


DEMO_DIR = os.path.join(PROJECT_DIR, ".tmp", "coding-demo")
TEST_COMMAND = "python -m unittest discover -s . -q"

BUGGY = '''def add(a, b):
    """两个数相加。"""
    return a - b


def greet(name):
    return f"你好，{name}！"
'''

TESTS = '''import unittest

from calc import add, greet


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_greet(self):
        self.assertEqual(greet("小明"), "你好，小明！")


if __name__ == "__main__":
    unittest.main()
'''


def prepare() -> None:
    '''每次从干净的项目开始。'''
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)
    with open(os.path.join(DEMO_DIR, "calc.py"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(BUGGY)
    with open(
        os.path.join(DEMO_DIR, "test_calc.py"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(TESTS)


def run_tests_directly() -> None:
    '''先跑一次测试，让输出里看得见「它真的失败了」。'''
    result = subprocess.run(
        TEST_COMMAND, shell=True, cwd=DEMO_DIR, capture_output=True
    )
    text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    print(f"== 基线测试（退出码 {result.returncode}）==")
    print(_tail(text, 12))


def build() -> Workflow:
    return Workflow(
        # 把要改的文件和测试都读进来：模型需要知道测试期望什么
        ReadFileStep("calc.py", output_key="source", base_dir=DEMO_DIR),
        ReadFileStep("test_calc.py", output_key="tests", base_dir=DEMO_DIR),
        # 先跑一次，拿到失败信息（FixLoop 的 until 也用它判断）
        RunTests(TEST_COMMAND, base_dir=DEMO_DIR, allowlist=["python"]),
        FixLoop(
            (
                CodingStep(
                    task="修好 calc.py 里 add 的 bug：它应该返回 a + b，现在返回 a - b",
                    context={
                        "calc.py": Var("source"),
                        "test_calc.py": Var("tests"),
                    },
                    error=Var("test_result"),
                    path="calc.py",          # 给了 path：补丁会在写盘前干跑校验
                    base_dir=DEMO_DIR,
                    cautions=[
                        Custom("只改 calc.py 里出错的那一行，其他代码保持不动"),
                        Custom("不要修改测试文件"),
                    ],
                ),
                ApplyDiff("calc.py", diff=Var("patch"), base_dir=DEMO_DIR),
                RunTests(TEST_COMMAND, base_dir=DEMO_DIR, allowlist=["python"]),
        ),
            until=Var("test_result")["passed"],
            max_rounds=3,
        ),
        name="coding_demo",
    )


def main() -> int:
    prepare()
    run_tests_directly()

    agent = Agent(register_as_default=False)
    print(f"\n== 开始修复（模型 {agent.vendor}/{agent.model}）==")
    workflow = build()
    print(workflow.render())
    context = workflow.run(agent=agent, verbose=Verbosity.SILENT)

    print("\n== 结果 ==")
    print("用了这么多轮修复：", context.get("fix_round"))
    print("是否用尽轮次：", context.get("fix_exhausted"))
    print("补丁前 3 行：")
    print("\n".join(str(context.get("patch", "")).splitlines()[:3]))
    print("应用结果：", context.get("apply_result"))
    print("测试通过：", context["test_result"]["passed"])
    print("\n== 最终 calc.py ==")
    with open(os.path.join(DEMO_DIR, "calc.py"), encoding="utf-8") as handle:
        print(handle.read())
    print("== 测试输出（尾部） ==")
    print(_tail(context["test_result"]["output"], 8))
    return 0 if context["test_result"]["passed"] else 1


def _tail(text: str, lines: int) -> str:
    content = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(content[-lines:]) if content else "（没有输出）"


if __name__ == "__main__":
    raise SystemExit(main())
