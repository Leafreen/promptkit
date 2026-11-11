'''目录式技能包：SKILL.md + references/ + scripts/ 一次加载。

    python examples/skill_package.py          # 演示（不需要 API Key）
    python examples/skill_package.py --live   # 真模型：自己读参考、调脚本

和 `skill_with_script.py` 的区别：脚本参数从 argparse 自动推断成工具
Schema，参考资料由 `read_skill_resource` 按需读取，不用手工包 Tool。

    report-kit/
    ├── SKILL.md
    ├── references/style.md
    └── scripts/make_report.py
'''
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from promptkit.skill import as_skill_package  # noqa: E402

SKILL_DIR = os.path.join(HERE, "skills_md", "report-kit")


def main(argv) -> int:
    package = as_skill_package(SKILL_DIR)
    print(package.describe())

    print("\n== 0. 始终可见的触发块（放进 ChatWorkflow 的 rules）==")
    print(package.routing_prompt())

    print("\n== 1. 工具列表 ==")
    for tool in package.tools():
        print(f"- {tool.name}: {tool.description}")

    print("\n== 2. 资源清单 ==")
    print(package.list_resources())

    print("\n== 3. 按需读参考 ==")
    print(package.read_resource("references/style.md"))

    print("\n== 4. 脚本（argparse 自动变成工具参数）==")
    print(
        package.run_script(
            "report_kit_make_report",
            topic="本周进展",
            points=["完成技能包加载", "补齐资源读取", "脚本参数自动推断"],
            out="out/report.md",
            dry_run=True,
        )
    )

    if "--live" not in argv:
        print("\n（加 --live 让模型自己决定读哪份参考、怎么调脚本）")
        return 0

    from promptkit.agent import Agent

    agent = Agent()
    reply = agent.ask(
        package.routing_prompt()
        + "\n\n请用 report_kit 技能，把「支持目录式技能包」整理成一份简短报告，"
        "主题是「promptkit 技能包」，要点包括：资源按需加载、脚本参数自动推断、"
        "依赖状态可见。",
        tools=package.tools(),
    )
    print("\n== 模型回答 ==\n" + reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
