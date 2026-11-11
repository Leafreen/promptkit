'''阶段 0 基线：示例包能加载、工具清单正确、最小工作流真能调到工具。

跑在现有实现（as_skill_package）之上；阶段 1 落地 SkillPackage.load()
后切换门面 API，本文件作为回归基线保留。
'''
import sys
import tempfile
from pathlib import Path

from promptkit.core import LLMStep, Prompt, Workflow
from promptkit.skill import as_skill_package

from tests.skills.baseline.fake_tool_client import (
    make_tool_agent,
    text_message,
    tool_call_message,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "skills"
HELLO_DIR = EXAMPLES / "hello-skill"
PPT_DIR = EXAMPLES / "ppt-master"


def test_load_hello_skill():
    package = as_skill_package(HELLO_DIR)

    assert package.skill.name == "hello-skill"
    assert "打招呼" in package.skill.description

    names = [tool.name for tool in package.tools()]
    assert names == [
        "read_skill",
        "list_resources",
        "read_resource",
        "write_file",
        "list_scripts",
        "run_script",
    ]


def test_hello_skill_read_resource():
    '''resources/ 目录（新约定的主目录）里的文件必须能读。'''
    package = as_skill_package(HELLO_DIR)
    text = package.read_resource("resources/note.md")
    assert "示例资源" in text


def test_load_ppt_master_example():
    '''精简 ppt-master：references/ 兼容目录要索引到，脚本要发现。'''
    package = as_skill_package(PPT_DIR)

    assert package.skill.name == "ppt-master"
    assert "PPT" in package.skill.description

    references = [path.relative_to(PPT_DIR).as_posix() for path in package.references]
    assert references == [
        "references/image-palettes/duotone.md",
        "references/shared-standards.md",
    ]

    script_names = [spec.name for spec in package.scripts]
    assert script_names == ["build_deck.py", "validate_deck.py"]

    text = package.read_resource("references/shared-standards.md")
    assert text.strip(), "references/ 里的文件读不到"


def test_baseline_workflow():
    '''最小工作流 + 假客户端：模型第一轮调 run_script，第二轮收尾。'''
    workdir = Path(tempfile.mkdtemp(prefix="skill_baseline_"))
    package = as_skill_package(HELLO_DIR)
    tools = package.tools(cwd=workdir)

    agent = make_tool_agent(
        [
            tool_call_message("run_script", {"name": "hello.py", "args": "--name 基线"}),
            text_message("已完成"),
        ]
    )
    workflow = Workflow(LLMStep(prompt=Prompt("打个招呼"), tools=tools))
    context = workflow.run(agent=agent)

    assert context["output"] == "已完成"
    greeting = workdir / "greeting.txt"
    assert greeting.is_file(), f"脚本工具没被执行（{workdir} 里没有 greeting.txt）"
    assert "基线" in greeting.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
