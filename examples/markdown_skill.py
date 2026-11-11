'''引入外部 Skill：从 `.md` 文件加载技能，不改代码就能扩展能力。

    python examples/markdown_skill.py          # 假 agent 演示（不需要 API Key）
    python examples/markdown_skill.py --live   # 真模型：Agent 自己挑技能来调用

一个 `.md` 技能 = 一段说明 + 一份参数表（frontmatter 可选）。加载后它就是
一个普通 Tool：模型调用它时拿到**整理好的操作说明**，然后按说明继续干活。

    from promptkit.skill import as_skill, as_skills

    as_skill("skills_md/weekly-report.md")   # 单个文件
    as_skills("skills_md/")                  # 目录下所有 .md
'''
import json
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from promptkit.skill import as_skill, as_skills  # noqa: E402


def show(skill_tool) -> None:
    '''打印技能被加载成了什么样的工具。'''
    print(f"名字：{skill_tool.name}")
    print(f"描述：{skill_tool.description}")
    print(f"参数 Schema：{json.dumps(skill_tool.parameters, ensure_ascii=False)}")
    print(f"来源：{skill_tool.source.name}")


def main(argv) -> int:
    # 1) 加载：一个文件 / 整个目录
    weekly = as_skill(os.path.join(HERE, "skills_md", "weekly-report.md"))
    show(weekly)

    print("\n== 直接调用：参数填进 {{占位符}}，返回整理好的说明 ==")
    instructions = weekly.call({"period": "2026-W45", "style": "活泼一点"})
    print(instructions[:120] + "…")

    print("\n== 整个目录一次性加载 ==")
    for item in as_skills(os.path.join(HERE, "skills_md")):
        print(f"- {item.name}: {item.description}")

    if "--live" not in argv:
        print("\n（加 --live 用真模型跑：Agent 收到 tools 后自己决定调用 weekly_report）")
        print("提示：加载结果可直接挂进 ChatWorkflow(skills=[weekly]) 或 Tools(weekly)")
        return 0

    from promptkit.agent import Agent

    agent = Agent()
    reply = agent.ask(
        "我这一周修好了登录页的样式 bug、把周报模板做完了 80%，"
        "下周打算写周报的自动生成脚本。请用 weekly_report 工具帮我整理。",
        tools=[weekly],
    )
    print("\n== 模型回答 ==\n" + reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
