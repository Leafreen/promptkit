'''目录式技能包：资源读取、脚本执行、依赖检查、allowed-tools。'''
import json
import sys
import tempfile
from pathlib import Path

import pytest

from promptkit.skill import (
    PathEscapeError,
    ScriptNotFoundError,
    SkillDependencyError,
    SkillValidationError,
    as_skill_package,
)


@pytest.fixture
def workdir():
    '''用 mkdtemp 而不是 tmp_path：conftest 会在每个测试结束后立刻删掉，
    避免临时目录留在项目根目录里影响 TestPackaging 的扫描。'''
    return Path(tempfile.mkdtemp(prefix="skill_package_"))


SCRIPT = '''"""生成报告。"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="生成报告")
    parser.add_argument("--topic", required=True, help="报告主题")
    parser.add_argument("--points", nargs="+", help="报告要点")
    parser.add_argument("--out", default="out.txt", help="输出文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印")
    args = parser.parse_args()
    text = args.topic + "|" + ",".join(args.points or [])
    if args.dry_run:
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
'''

GENERIC_SCRIPT = '''import sys
print("|".join(sys.argv[1:]))
'''


def make_skill(workdir, *, frontmatter="", scripts=None, body="# 演示\n"):
    root = workdir / "demo-skill"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\n"
        "name: demo_skill\n"
        "description: 演示技能包\n"
        f"{frontmatter}"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    (root / "references").mkdir()
    (root / "references" / "style.md").write_text("写作规范", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (root / "templates").mkdir()
    (root / "templates" / "base.txt").write_text("模板", encoding="utf-8")

    if scripts is not None:
        (root / "scripts").mkdir()
        for name, source in scripts.items():
            (root / "scripts" / name).write_text(source, encoding="utf-8")
    return root


def test_loads_resources_and_appends_index(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    assert package.skill.name == "demo_skill"
    assert [path.name for path in package.references] == ["style.md"]
    assert [path.name for path in package.assets] == ["logo.png"]
    assert [path.name for path in package.templates] == ["base.txt"]

    body = package.skill.instructions
    assert "references/style.md" in body
    assert "assets/logo.png" in body
    assert "make_report.py" in body
    assert str(root) in body


def test_tools_include_resources_and_scripts(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    names = [tool.name for tool in package.tools()]
    assert names == [
        "read_skill",
        "list_resources",
        "read_resource",
        "write_file",
        "list_scripts",
        "run_script",
    ]


def test_tools_without_scripts_skip_script_tools(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)

    names = [tool.name for tool in package.tools()]
    assert "list_scripts" not in names
    assert "run_script" not in names
    assert "read_skill" in names


def test_tools_namespace_prefixes_tool_names(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    names = [tool.name for tool in package.tools(namespace=True)]
    assert names == [
        "demo_skill__read_skill",
        "demo_skill__list_resources",
        "demo_skill__read_resource",
        "demo_skill__write_file",
        "demo_skill__list_scripts",
        "demo_skill__run_script",
    ]


def test_tools_enable_disable_filters(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    names = [tool.name for tool in package.tools(enable={"read_skill"})]
    assert names == ["read_skill"]

    names = [tool.name for tool in package.tools(disable={"write_file"})]
    assert "write_file" not in names
    assert "run_script" in names


def test_read_text_and_binary(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)

    assert package.read_resource("references/style.md") == "写作规范"
    binary = json.loads(package.read_resource("assets/logo.png"))
    assert binary["binary"] is True
    assert binary["path"] == "assets/logo.png"


@pytest.mark.parametrize("path", ["../secret.txt", "C:/Windows/win.ini", ".env"])
def test_read_resource_rejects_escape(workdir, path):
    root = make_skill(workdir)
    package = as_skill_package(root)
    assert package.read_resource(path).startswith("错误：")


def test_list_resources_filters_by_kind(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    items = json.loads(package.list_resources("reference"))
    assert [item["path"] for item in items] == ["references/style.md"]

    scripts = json.loads(package.list_resources("script"))
    assert scripts[0]["path"] == "scripts/make_report.py"
    assert package.list_resources("nope").startswith("错误：")


def test_list_scripts_returns_relative_names(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    scripts = json.loads(package._list_scripts())
    assert scripts == [
        {"path": "make_report.py", "description": "生成报告。"}
    ]


def test_run_script_executes_and_writes(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)
    work = workdir / "work"

    result = json.loads(
        package.run_script(
            "make_report.py",
            "--topic 周报 --points 完成A 完成B --dry-run",
            cwd=work,
        )
    )
    assert result["returncode"] == 0
    assert "周报|完成A,完成B" in result["stdout"]

    written = json.loads(
        package.run_script(
            "make_report.py",
            "--topic 周报 --points 完成A --out report.txt",
            cwd=work,
        )
    )
    assert written["returncode"] == 0
    assert (work / "report.txt").read_text(encoding="utf-8").startswith("周报|")


def test_run_script_reports_nonzero_exit(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    result = json.loads(package.run_script("make_report.py", ""))
    assert result["returncode"] != 0
    assert "topic" in result["stderr"]


def test_run_script_rejects_unknown_and_escape(workdir):
    root = make_skill(workdir, scripts={"make_report.py": SCRIPT})
    package = as_skill_package(root)

    with pytest.raises(ScriptNotFoundError):
        package.run_script("missing_script.py")
    with pytest.raises(PathEscapeError):
        package.run_script("../secret.py")


def test_run_script_passes_args_without_shell(workdir):
    root = make_skill(workdir, scripts={"echo_args.py": GENERIC_SCRIPT})
    package = as_skill_package(root)

    result = json.loads(package.run_script("echo_args.py", "a;b c"))
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "a;b|c"


def test_allowed_tools_restricts_resource_calls(workdir):
    root = make_skill(
        workdir,
        frontmatter="allowed-tools: [read_resource]\n",
        scripts={"make_report.py": SCRIPT},
    )
    package = as_skill_package(root)

    # read_resource 放行了，正常返回
    assert package.read_resource("references/style.md") == "写作规范"
    # list_resources 没放行，调用时返回错误
    assert package.list_resources("reference").startswith("错误：")


def test_anthropic_tool_names_are_mapped(workdir):
    root = make_skill(
        workdir,
        frontmatter="allowed-tools: [Read, Glob]\n",
        scripts={"make_report.py": SCRIPT},
    )
    package = as_skill_package(root)

    assert package.allowed_tools == ("read_resource", "list_resources")


def test_unknown_allowed_tools_blocks_resources(workdir):
    root = make_skill(workdir, frontmatter="allowed-tools: [web_search]\n")
    package = as_skill_package(root)

    assert package.allowed_tools == ("__none__",)
    assert package.read_resource("references/style.md").startswith("错误：")
    assert package.list_resources("reference").startswith("错误：")


def test_dependencies_are_reported_and_can_be_strict(workdir):
    frontmatter = "dependencies: [definitely_missing_package_xyz]\n"
    root = make_skill(workdir, frontmatter=frontmatter)

    package = as_skill_package(root)
    assert package.missing_dependencies == ("definitely_missing_package_xyz",)
    assert "缺失" in package.skill.instructions

    with pytest.raises(SkillDependencyError):
        as_skill_package(root, strict_dependencies=True)


def test_missing_entry_is_rejected(workdir):
    empty = workdir / "empty"
    empty.mkdir()
    with pytest.raises(SkillValidationError):
        as_skill_package(empty)


def test_scripts_frontmatter_is_a_whitelist(workdir):
    root = make_skill(
        workdir,
        frontmatter=(
            "scripts:\n"
            "  only_this:\n"
            "    path: scripts/make_report.py\n"
            "    description: 只暴露这个脚本\n"
        ),
        scripts={"make_report.py": SCRIPT, "extra.py": GENERIC_SCRIPT},
    )
    package = as_skill_package(root)

    assert [spec.name for spec in package.scripts] == ["make_report.py"]
    assert package.scripts[0].description == "只暴露这个脚本"
    assert "extra" not in [spec.name for spec in package.scripts]


def test_agent_md_entry_and_prompt_routes(workdir):
    root = workdir / "agent-skill"
    root.mkdir()
    (root / "AGENT.md").write_text(
        "---\n"
        "name: agent_skill\n"
        "description: 用 AGENT.md 当入口\n"
        "---\n"
        "# 入口\n",
        encoding="utf-8",
    )

    package = as_skill_package(root)
    assert package.skill.name == "agent_skill"
    assert package.instructions.startswith("# 入口")
    assert "技能包资源" in package.instructions

    routing = package.routing_prompt()
    assert "agent_skill" in routing
    assert "read_skill" in routing
    assert "用 AGENT.md 当入口" in routing


def test_skill_md_wins_over_agent_md(workdir):
    root = workdir / "both"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: from_skill\ndescription: skill\n---\n# S\n",
        encoding="utf-8",
    )
    (root / "AGENT.md").write_text(
        "---\nname: from_agent\ndescription: agent\n---\n# A\n",
        encoding="utf-8",
    )

    assert as_skill_package(root).skill.name == "from_skill"


def test_frontmatter_block_scalars(workdir):
    '''ppt-master 等技能用 `description: >` 多行描述，不能误报缩进错误。'''
    from promptkit.skill import as_skill

    root = workdir / "folded"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\n"
        "name: folded_skill\n"
        "description: >\n"
        "  AI-driven presentation workflow for generating editable PPTX decks\n"
        "  and slides, reconstructing page visuals, creating reusable\n"
        "  Brand/Style/Layout/Deck workspaces.\n"
        "metadata:\n"
        '  version: "6.9.0"\n'
        "  sponsors:\n"
        '    - "SPONSORS.md"\n'
        '    - "SPONSORS_CN.md"\n'
        "---\n"
        "# 正文\n",
        encoding="utf-8",
    )

    skill = as_skill(root)
    # 原始 frontmatter 保留 clip 语义的尾换行；description 展示时被 strip
    assert skill.frontmatter["description"].endswith("workspaces.\n")
    assert skill.description.strip() == (
        "AI-driven presentation workflow for generating editable PPTX decks "
        "and slides, reconstructing page visuals, creating reusable "
        "Brand/Style/Layout/Deck workspaces."
    )
    assert skill.frontmatter["metadata"]["version"] == "6.9.0"
    assert skill.frontmatter["metadata"]["sponsors"] == ["SPONSORS.md", "SPONSORS_CN.md"]


def test_frontmatter_literal_and_strip_chomping(workdir):
    from promptkit.skill import as_skill

    root = workdir / "literal"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\n"
        "name: literal_skill\n"
        "description: x\n"
        "script_header: |\n"
        "  line one\n"
        "  line two\n"
        "short: |-\n"
        "  no trailing\n"
        "---\n"
        "# 正文\n",
        encoding="utf-8",
    )

    skill = as_skill(root)
    assert skill.frontmatter["script_header"] == "line one\nline two\n"
    assert skill.frontmatter["short"] == "no trailing"


def test_command_tool_enters_workdir_and_rewrites_python(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)
    tool = package.command_tool(python=sys.executable)

    assert tool.name == "run_command"
    assert list(tool.parameters["properties"]) == ["command"]

    # 同时验证两件事：cwd 切到了技能目录，python 前缀被换成指定的解释器
    output = tool.func('python -c "import os; print(os.getcwd())"')
    assert str(root).lower() in output.lower()


def test_command_tool_leaves_other_commands_untouched(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)
    tool = package.command_tool(timeout=30)

    script = "scripts/echo_argv.py"
    (root / "scripts").mkdir(exist_ok=True)
    (root / script).write_text(
        "import sys\nprint('|'.join(sys.argv[1:]))\n", encoding="utf-8"
    )
    output = tool.func(f'"{sys.executable}" {script} alpha beta')
    assert "alpha|beta" in output


def test_command_tool_description_mentions_workdir(workdir):
    package = as_skill_package(make_skill(workdir))
    tool = package.command_tool(python=sys.executable, timeout=120)

    assert str(package.workdir) in tool.description
    assert "120 秒" in tool.description
    assert sys.executable in tool.description


def test_command_tool_accepts_custom_name(workdir):
    package = as_skill_package(make_skill(workdir))
    tool = package.command_tool(name="run_ppt", description="自定义")
    assert tool.name == "run_ppt"
    assert tool.description == "自定义"


def test_library_files_are_not_scripts(workdir):
    '''tests/、下划线目录、test_*.py 是库代码，不是命令行入口。'''
    root = make_skill(workdir)
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "entry.py").write_text("print(1)\n", encoding="utf-8")
    tests_dir = scripts / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_x.py").write_text("print(2)\n", encoding="utf-8")
    private_dir = scripts / "__lib"
    private_dir.mkdir()
    (private_dir / "helper.py").write_text("print(3)\n", encoding="utf-8")
    (scripts / "test_top.py").write_text("print(4)\n", encoding="utf-8")

    package = as_skill_package(root)
    names = [spec.name for spec in package.scripts]

    assert names == ["entry.py"]


def test_write_tool_writes_under_workdir(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)
    tool = package.write_tool()

    tool.func(path="out/big.svg", content="<svg>大文件内容</svg>")
    assert (root / "out" / "big.svg").read_text(encoding="utf-8") == (
        "<svg>大文件内容</svg>"
    )
    assert "write_file" in package.command_tool().description


def test_write_tool_root_moves_output_outside_skill(workdir):
    '''技能目录只读时，产物应能落到另一个目录。'''
    root = make_skill(workdir)
    package = as_skill_package(root)
    output = workdir / "elsewhere" / "deck"

    tool = package.write_tool(root=output)
    tool.func(path="slides/s1.svg", content="<svg/>")

    assert (output / "slides" / "s1.svg").read_text(encoding="utf-8") == "<svg/>"
    assert not (root / "slides").exists()
    assert str(output) in tool.description


def test_write_tool_root_still_blocks_escape(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)
    output = workdir / "out_only"

    tool = package.write_tool(root=output)
    result = tool.func(path="../../escaped.txt", content="x")

    assert "越界" in result
    assert not (workdir / "escaped.txt").exists()


def test_command_tool_root_runs_elsewhere(workdir):
    root = make_skill(workdir)
    package = as_skill_package(root)
    output = workdir / "cmd_root"

    tool = package.command_tool(python=sys.executable, root=output)
    result = tool.func('python -c "import os; print(os.getcwd())"')

    assert str(output).lower() in result.lower()
    assert output.is_dir()
