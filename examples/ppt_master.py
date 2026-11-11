'''接入 ppt-master 技能包：SKILL.md 当说明书，脚本用命令行调。

    python examples/ppt_master.py                  # 只检查加载结果（不联网）
    python examples/ppt_master.py --install-deps   # 先装技能自己的依赖
    python examples/ppt_master.py --run            # 让模型真正做一份 PPT

为什么不把脚本逐个挂成工具：ppt-master 有 352 个脚本，355 个工具会把每次
请求的 tools 列表撑到几百 KB，模型反而挑不出来。它的 SKILL.md 本来就是按
「用命令行跑脚本」写的，所以这里只挂 3 个技能工具 + 1 个命令工具。

技能目录默认读环境变量 PPT_MASTER_DIR，没有就用 DEFAULT_SKILL_DIR；
Python 解释器由 resolve_python 自动探测，环境变量 PROMPTKIT_PYTHON 可覆盖。
'''
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.agent import Agent  # noqa: E402
from promptkit.core import LLMStep, Prompt, Verbosity, Workflow  # noqa: E402
from promptkit.core.settings import using_settings  # noqa: E402
from promptkit.skill import as_skill_package, resolve_python  # noqa: E402

DEFAULT_SKILL_DIR = str(PROJECT_DIR / "examples" / "skills" / "ppt-master")
COMMAND_TIMEOUT = 600          # 组版脚本可能跑几分钟，默认 30 秒不够
MAX_TOOL_ROUNDS = 40           # 读文档 → 规划 → 写脚本 → 组版 → 校验，链路很长
DEFAULT_TASK = "请制作一份关于决策树的讲解 PPT，10 页左右，中文"
DEFAULT_OUT_DIR = str(PROJECT_DIR / "pptwork")   # 产物目录，与只读技能目录解耦


def install_dependencies(python: str, skill_dir: Path) -> None:
    '''装技能自己的 requirements.txt。放在脚本里做，别让模型现场 pip install。'''
    requirements = skill_dir / "requirements.txt"
    if not requirements.is_file():
        print(f"（{skill_dir} 里没有 requirements.txt，跳过）")
        return
    print(f"安装技能依赖：{requirements}")
    subprocess.run(
        [python, "-m", "pip", "install", "-r", str(requirements)],
        check=False,
    )


def build_task_prompt(
    task: str, skill_dir: Path, python: str, out_dir: Path
) -> str:
    '''把环境事实一次性告诉模型，省掉它自己探测的十几轮。'''
    return (
        f"{task}\n\n"
        "环境事实（已确认，不要再花轮次探测）：\n"
        f"- 技能目录（只读，放说明书和脚本）：{skill_dir}\n"
        f"- 产物目录（可写，所有文件都放这里）：{out_dir}\n"
        f"- 可用 Python：{python}（你直接写 python 也会被换成它）\n"
        "- 技能依赖已装好；脚本都在技能目录的 scripts/ 下\n\n"
        "做事要求：\n"
        "- 先读技能说明，按它给出的流程和阶段执行，不要自创流程\n"
        "- 参考文档用 read_skill_resource 按需读，只读当前步骤需要的那几份\n"
        f"- 命令的工作目录已经是 {out_dir}，写文件也用相对路径\n"
        "- 写任何文件（SVG、脚本、配置）都用 write_file，"
        "不要在 run_command 里用 python -c 拼文件内容\n"
        "- 不要把产物写到技能目录里，技能目录只读\n"
        "- 最后把生成文件的完整路径告诉用户\n"
        "- 某个脚本报错时先看报错再改，不要反复重跑同一条命令\n"
    )


def count_scripts(skill_dir: Path) -> int:
    '''数一下 scripts/ 下能跑的入口脚本（下划线目录是库模块，不算），仅用于打印。'''
    root = skill_dir / "scripts"
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*.py")
        if path.is_file()
        and path.name != "__init__.py"
        and not path.name.startswith("_")
        and not any(
            part.startswith("_") for part in path.relative_to(root).parts[:-1]
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="接入 ppt-master 技能包")
    parser.add_argument(
        "--skill-dir",
        default=os.environ.get("PPT_MASTER_DIR", DEFAULT_SKILL_DIR),
        help="ppt-master 技能目录（含 SKILL.md 的那一层）",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="要做的事")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="产物目录（可写）；技能目录只读时靠它把产物分出去",
    )
    parser.add_argument(
        "--install-deps", action="store_true", help="先装技能的 requirements.txt"
    )
    parser.add_argument("--run", action="store_true", help="真模型跑一遍")
    parser.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS)
    args = parser.parse_args(argv)

    skill_dir = Path(args.skill_dir).expanduser().resolve()
    if not skill_dir.is_dir():
        raise SystemExit(f"技能目录不存在：{skill_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve()

    python = resolve_python(root=skill_dir)
    if args.install_deps:
        install_dependencies(python, skill_dir)

    # scripts=[] 表示不逐个挂脚本；SKILL.md 里的命令行用法照常可用。
    # root=out_dir 让命令的工作目录和写文件的落点都指向产物目录，
    # 技能目录只读也能正常出图。
    package = as_skill_package(skill_dir, scripts=[])
    run_tool = package.command_tool(
        python=python, timeout=COMMAND_TIMEOUT, root=out_dir
    )
    # tools() 已内置 write_file（写到 cwd=out_dir），无需再挂 write_tool
    tools = [*package.tools(cwd=out_dir), run_tool]

    print(package.describe())
    print(
        f"技能自带 {count_scripts(skill_dir)} 个脚本，"
        "全部通过 run_command 调用，不逐个挂成工具"
    )
    print(f"\nPython：{python}")
    print(f"产物目录：{out_dir}")
    print("挂给模型的工具：" + "、".join(tool.name for tool in tools))

    if not args.run:
        print("\n（这只是加载检查；加 --run 让模型真正开工）")
        return 0

    workflow = Workflow(
        LLMStep(
            prompt=Prompt(build_task_prompt(args.task, skill_dir, python, out_dir)),
            tools=tools,
        )
    )
    with using_settings(max_tool_rounds=args.max_rounds):
        context = workflow.run(agent=Agent(), verbose=Verbosity.VERBOSE)

    print("\n== 模型最终回复 ==\n" + str(context["output"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
