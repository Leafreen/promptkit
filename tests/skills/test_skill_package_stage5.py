'''阶段 5：run_command 默认关闭 + allow_commands 白名单。'''
import sys
import tempfile
from pathlib import Path

import pytest

from promptkit.skill import as_skill_package


@pytest.fixture
def workdir():
    return Path(tempfile.mkdtemp(prefix="skill_stage5_"))


def write_skill(root: Path, *, frontmatter: str = "", scripts: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: 演示技能\n"
        f"{frontmatter}---\n# 正文\n",
        encoding="utf-8",
    )
    if scripts:
        (root / "scripts").mkdir()
        (root / "scripts" / "main.py").write_text("print(1)\n", encoding="utf-8")
    return root


def command_tool(package, **kwargs):
    return next(
        tool for tool in package.tools(**kwargs) if tool.name.endswith("run_command")
    )


def test_run_command_is_disabled_by_default(workdir):
    package = as_skill_package(write_skill(workdir / "demo", scripts=True))

    names = [tool.name for tool in package.tools()]

    assert "run_command" not in names


def test_run_command_requires_allow_commands(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(package, enable={"run_command"})

    result = tool.call({"command": f'"{sys.executable}" -c "print(1)"'})

    assert result.startswith("错误：")
    assert "allow_commands" in result


def test_run_command_runs_whitelisted_command_in_cwd(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    out = workdir / "out"
    tool = command_tool(
        package, enable={"run_command"}, allow_commands=["python"], cwd=out
    )

    result = tool.call(
        {"command": f'"{sys.executable}" -c "import os; print(os.getcwd())"'}
    )

    assert str(out.resolve()).lower() in result.lower()


def test_run_command_rejects_unlisted_command(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(package, enable={"run_command"}, allow_commands=["python"])

    result = tool.call({"command": "git status"})

    assert "不在 allow_commands 白名单" in result
    assert "python" in result


def test_run_command_accepts_bare_python_after_rewrite(workdir):
    '''白名单写 python；命令开头的 python 会被换成配置的解释器。'''
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(
        package,
        enable={"run_command"},
        allow_commands=["python"],
        python=sys.executable,
        cwd=workdir / "out",
    )

    result = tool.call({"command": 'python -c "print(42)"'})

    assert "42" in result


def test_run_command_disable_wins_over_enable(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))

    names = [
        tool.name
        for tool in package.tools(
            enable={"run_command"}, disable={"run_command"}, allow_commands=["python"]
        )
    ]

    assert "run_command" not in names


def test_allow_commands_alone_enables_run_command(workdir):
    '''显式给白名单就够，不必再写 enable={"run_command"}。'''
    package = as_skill_package(write_skill(workdir / "demo"))

    names = [tool.name for tool in package.tools(allow_commands=["python"])]

    assert names == ["read_skill", "write_file", "run_command"]


def test_run_command_follows_tool_whitelist(workdir):
    root = write_skill(workdir / "demo", frontmatter="tools: [read_skill]\n")
    package = as_skill_package(root)

    names = [tool.name for tool in package.tools(allow_commands=["python"])]

    assert names == ["read_skill"]


def test_run_command_respects_namespace(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(
        package, enable={"run_command"}, allow_commands=["python"], namespace=True
    )

    assert tool.name == "demo_skill__run_command"


def test_run_command_description_mentions_env_and_allowlist(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(
        package,
        enable={"run_command"},
        allow_commands=["python", "git"],
        python=sys.executable,
        cwd=workdir / "out",
    )

    assert str((workdir / "out").resolve()) in tool.description
    assert sys.executable in tool.description
    assert "python、git" in tool.description


def test_run_command_rejects_empty_command(workdir):
    package = as_skill_package(write_skill(workdir / "demo"))
    tool = command_tool(package, enable={"run_command"}, allow_commands=["python"])

    assert tool.call({"command": "  "}).startswith("错误：")


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
