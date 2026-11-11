'''Skill 里带脚本：SKILL.md 写「什么时候用、怎么汇报」，scripts/ 放干活的。

    python examples/skill_with_script.py          # 演示（不需要 API Key）
    python examples/skill_with_script.py --live   # 真模型：自己调工具再汇报

目录结构（常见约定）：

    skills_md/word-count/
    ├── SKILL.md               # 说明：模型读完知道「该调 count_words、怎么汇报」
    └── scripts/
        └── count_words.py     # 脚本：独立可跑，不依赖 promptkit

模型**不会**直接执行脚本 —— 脚本先包成一个普通 Tool（这里用 subprocess
调），SKILL.md 的正文告诉模型「先调工具、再按格式汇报」。
'''
import os
import subprocess
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from promptkit.core.tool import Tool  # noqa: E402
from promptkit.skill import as_skill  # noqa: E402

SKILL_DIR = os.path.join(HERE, "skills_md", "word-count")
SCRIPT = os.path.join(SKILL_DIR, "scripts", "count_words.py")


def count_words(path: str) -> str:
    '''统计一个文本文件的字数、行数和估算阅读时长（调用 word-count 技能的脚本）。

    Args:
        path: 要统计的文本文件路径
    '''
    proc = subprocess.run(
        [sys.executable, SCRIPT, path],
        capture_output=True, text=True, timeout=30,
    )
    return proc.stdout.strip() or f"脚本执行失败：{proc.stderr.strip()}"


COUNT_WORDS = Tool.from_function(count_words)


def main(argv) -> int:
    # 目录模式：自动找里面的 SKILL.md
    skill = as_skill(SKILL_DIR)
    print(f"技能：{skill.name} — {skill.description}")

    demo_file = os.path.join(SKILL_DIR, "SKILL.md")  # 拿技能文件自己当输入

    print("\n== 1. 工具跑脚本（模型拿不到说明时，这就是全部能力）==")
    print(COUNT_WORDS.call({"path": demo_file}))

    print("\n== 2. 技能正文渲染（模型读完知道怎么汇报）==")
    print(skill.call({"path": demo_file}))

    if "--live" not in argv:
        print("\n（加 --live 用真模型跑完整闭环：调 count_words -> 按 SKILL.md 汇报）")
        print("挂进聊天：ChatWorkflow(skills=[skill], tools=[COUNT_WORDS])")
        return 0

    from promptkit.agent import Agent

    agent = Agent()
    reply = agent.ask(
        f"请按 word_count_report 技能的要求，统计并汇报这个文件：{demo_file}",
        tools=[COUNT_WORDS, skill],
    )
    print("\n== 模型回答 ==\n" + reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
