'''阶段 4：frontmatter 字段补齐（entrypoint / tools 白名单 / 依赖并集）。'''
import logging
import tempfile
from pathlib import Path

import pytest

from promptkit.skill import SkillPackage, as_skill_package


@pytest.fixture
def workdir():
    return Path(tempfile.mkdtemp(prefix="skill_stage4_"))


def write_skill(
    root: Path,
    name: str = "demo-skill",
    *,
    frontmatter: str = "",
    body: str = "# 正文\n",
    scripts: bool = False,
    requirements: str = "",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: 演示技能\n{frontmatter}---\n{body}",
        encoding="utf-8",
    )
    if scripts:
        (root / "scripts").mkdir()
        (root / "scripts" / "main.py").write_text("print(1)\n", encoding="utf-8")
    if requirements:
        (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    return root


# ---------------------------- entrypoint ----------------------------


def test_entrypoint_is_recorded(workdir):
    root = write_skill(workdir / "demo", scripts=True, frontmatter="entrypoint: scripts/main.py\n")

    pkg = SkillPackage.load(root)

    assert pkg.entrypoint == "scripts/main.py"
    assert "入口脚本：scripts/main.py" in pkg.system_prompt()


def test_entrypoint_missing_file_only_warns(workdir, caplog):
    caplog.set_level(logging.WARNING, logger="promptkit.skill")
    root = write_skill(workdir / "demo", frontmatter="entrypoint: scripts/nope.py\n")

    pkg = as_skill_package(root)

    assert pkg.entrypoint == ""
    assert "entrypoint" in caplog.text


def test_entrypoint_rejects_absolute_and_escape(workdir, caplog):
    caplog.set_level(logging.WARNING, logger="promptkit.skill")
    absolute = write_skill(workdir / "abs", frontmatter="entrypoint: C:/Windows/win.ini\n")
    escape = write_skill(workdir / "esc", frontmatter="entrypoint: ../outside.py\n")

    assert as_skill_package(absolute).entrypoint == ""
    assert as_skill_package(escape).entrypoint == ""
    assert caplog.text.count("entrypoint") >= 2


def test_no_entrypoint_by_default(workdir):
    assert as_skill_package(write_skill(workdir / "demo")).entrypoint == ""


# ---------------------------- tools 白名单 ----------------------------


def test_tools_whitelist_limits_generated_tools(workdir):
    root = write_skill(
        workdir / "demo",
        scripts=True,
        frontmatter="tools: [read_skill, list_scripts, run_script]\n",
    )

    pkg = SkillPackage.load(root)

    assert [tool.name for tool in pkg.tools()] == [
        "read_skill",
        "list_scripts",
        "run_script",
    ]
    assert pkg.tool_whitelist == ("read_skill", "list_scripts", "run_script")


def test_tools_whitelist_can_disable_write_file(workdir):
    root = write_skill(workdir / "demo", frontmatter="tools: [read_skill]\n")

    names = [tool.name for tool in as_skill_package(root).tools()]

    assert names == ["read_skill"]


def test_tools_whitelist_filters_system_prompt(workdir):
    root = write_skill(
        workdir / "demo",
        scripts=True,
        frontmatter="tools: [read_skill, run_script]\n",
    )

    prompt = SkillPackage.load(root).system_prompt()

    assert "可用工具：read_skill、run_script" in prompt


def test_tools_whitelist_accepts_anthropic_names(workdir):
    root = write_skill(
        workdir / "demo",
        scripts=True,
        frontmatter="tools: [Read, Bash]\n",
    )

    assert as_skill_package(root).tool_whitelist == ("read_resource", "run_script")


def test_tools_whitelist_unknown_names_block_everything(workdir):
    root = write_skill(workdir / "demo", frontmatter="tools: [web_search]\n")
    pkg = as_skill_package(root)

    assert pkg.tool_whitelist == ("__none__",)
    assert pkg.tools() == []


def test_tools_whitelist_combines_with_enable(workdir):
    root = write_skill(
        workdir / "demo",
        scripts=True,
        frontmatter="tools: [read_skill, list_scripts, run_script]\n",
    )
    pkg = as_skill_package(root)

    names = [tool.name for tool in pkg.tools(enable={"run_script"})]

    assert names == ["run_script"]


def test_no_tools_field_keeps_defaults(workdir):
    root = write_skill(workdir / "demo", scripts=True)

    pkg = as_skill_package(root)

    assert pkg.tool_whitelist == ()
    assert [tool.name for tool in pkg.tools()] == [
        "read_skill",
        "write_file",
        "list_scripts",
        "run_script",
    ]


# ---------------------------- 依赖并集 ----------------------------


def test_requirements_txt_joins_dependencies(workdir):
    root = write_skill(
        workdir / "demo",
        frontmatter="dependencies: [definitely_missing_a]\n",
        requirements="# 注释行\n-r other.txt\ndefinitely_missing_b>=1.0  # 行尾注释\n\ndefinitely_missing_a\n",
    )

    pkg = as_skill_package(root)

    assert pkg.dependencies == ("definitely_missing_a", "definitely_missing_b")
    assert pkg.missing_dependencies == ("definitely_missing_a", "definitely_missing_b")


def test_requirements_txt_only(workdir):
    root = write_skill(workdir / "demo", requirements="definitely_missing_c\n")

    assert as_skill_package(root).dependencies == ("definitely_missing_c",)


def test_installed_package_from_requirements_is_not_missing(workdir):
    root = write_skill(workdir / "demo", requirements="json\n")

    pkg = as_skill_package(root)

    assert pkg.dependencies == ("json",)
    assert pkg.missing_dependencies == ()


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v"]))
