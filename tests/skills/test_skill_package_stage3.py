'''阶段 3：system_prompt 环境事实、批量加载、多包合并去重。'''
import sys
import tempfile
from pathlib import Path

import pytest

from promptkit.skill import (
    DuplicateToolError,
    SkillPackage,
    load_skills_from_dir,
    merge_tools,
)


@pytest.fixture
def workdir():
    return Path(tempfile.mkdtemp(prefix="skill_stage3_"))


def write_skill(
    root: Path, name: str, *, body: str = "# 正文\n", scripts: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} 的说明\n---\n{body}",
        encoding="utf-8",
    )
    if scripts:
        (root / "scripts").mkdir()
        (root / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")
    return root


# ---------------------------- system_prompt ----------------------------


def test_system_prompt_includes_env_facts(workdir):
    root = write_skill(workdir / "hello-skill", "hello-skill", scripts=True)
    pkg = SkillPackage.load(root)

    prompt = pkg.system_prompt(python=sys.executable, cwd=workdir / "out")

    assert f"Python 解释器：{sys.executable}" in prompt
    assert f"工作目录：{(workdir / 'out').resolve()}" in prompt
    assert "可用工具：read_skill、write_file、list_scripts、run_script" in prompt


def test_system_prompt_does_not_create_workdir(workdir):
    '''system_prompt 只展示环境事实，不应在包里凭空创建 output/。'''
    root = write_skill(workdir / "hello-skill", "hello-skill")
    pkg = SkillPackage.load(root)

    pkg.system_prompt()

    assert not (root / "output").exists()


def test_system_prompt_defaults_to_output_dir(workdir):
    root = write_skill(workdir / "hello-skill", "hello-skill")
    pkg = SkillPackage.load(root)

    prompt = pkg.system_prompt()

    assert sys.executable in prompt
    assert f"工作目录：{(root / 'output').resolve()}" in prompt


def test_system_prompt_truncates_long_body(workdir):
    body = "\n".join(f"第 {index} 行" for index in range(60))
    root = write_skill(workdir / "long-skill", "long-skill", body=body)
    pkg = SkillPackage.load(root)

    prompt = pkg.system_prompt(max_lines=10)

    assert "第 9 行" in prompt
    assert "第 10 行" not in prompt
    assert "正文共 60 行" in prompt


# ---------------------------- 批量加载 ----------------------------


def test_load_skills_from_dir_finds_packages(workdir):
    write_skill(workdir / "alpha-skill", "alpha-skill")
    write_skill(workdir / "beta-skill", "beta-skill", scripts=True)
    (workdir / "not-a-skill").mkdir()          # 没有入口文件，跳过

    packages = load_skills_from_dir(workdir)

    assert [pkg.name for pkg in packages] == ["alpha-skill", "beta-skill"]


def test_load_skills_from_dir_skips_invalid_package(workdir):
    write_skill(workdir / "good-skill", "good-skill")
    bad = workdir / "bad-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\ndescription: 没有 name\n---\n正文\n", encoding="utf-8")

    packages = load_skills_from_dir(workdir)

    assert [pkg.name for pkg in packages] == ["good-skill"]


def test_load_skills_from_dir_recursive(workdir):
    write_skill(workdir / "group" / "inner-skill", "inner-skill")

    assert load_skills_from_dir(workdir) == []
    names = [pkg.name for pkg in load_skills_from_dir(workdir, recursive=True)]
    assert names == ["inner-skill"]


def test_load_skills_from_dir_respects_pattern(workdir):
    write_skill(workdir / "alpha-skill", "alpha-skill")
    write_skill(workdir / "beta-skill", "beta-skill")

    names = [pkg.name for pkg in load_skills_from_dir(workdir, pattern="alpha-*")]

    assert names == ["alpha-skill"]


# ---------------------------- 多包合并 ----------------------------


def test_merge_tools_namespaces_by_default(workdir):
    alpha = SkillPackage.load(write_skill(workdir / "alpha-skill", "alpha-skill", scripts=True))
    beta = SkillPackage.load(write_skill(workdir / "beta-skill", "beta-skill"))

    tools = merge_tools([alpha, beta])
    names = [tool.name for tool in tools]

    assert "alpha_skill__run_script" in names
    assert "beta_skill__run_script" not in names       # beta 没有 scripts
    assert "beta_skill__read_skill" in names
    assert len(names) == len(set(names))


def test_merge_tools_detects_duplicate_names(workdir):
    alpha = SkillPackage.load(write_skill(workdir / "alpha-skill", "alpha-skill"))
    beta = SkillPackage.load(write_skill(workdir / "beta-skill", "beta-skill"))

    with pytest.raises(DuplicateToolError) as exc:
        merge_tools([alpha, beta], namespace=False)

    assert "read_skill" in exc.value.message
    assert "alpha-skill" in exc.value.message
    assert "beta-skill" in exc.value.message
    assert exc.value.code == "SKILL_DUP_TOOL"
    assert exc.value.hint


def test_merge_tools_passes_interpreter_and_cwd(workdir):
    alpha = SkillPackage.load(write_skill(workdir / "alpha-skill", "alpha-skill", scripts=True))
    out = workdir / "shared-out"

    tools = merge_tools(
        [alpha], python=sys.executable, cwd=out, namespace=False
    )
    run_script = next(tool for tool in tools if tool.name == "run_script")

    assert sys.executable in run_script.description
    assert str(out.resolve()) in run_script.description
    assert out.is_dir()


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v"]))
