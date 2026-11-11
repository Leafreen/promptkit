'''阶段 1：SkillPackage.load() 门面 + 严格 frontmatter 校验 + 异常体系。'''
import tempfile
from pathlib import Path

import pytest

import promptkit.skill.package as package_mod
from promptkit.skill import (
    DependencyInstallError,
    InvalidSkillPackageError,
    SkillPackage,
    as_skill_package,
)


@pytest.fixture
def workdir():
    return Path(tempfile.mkdtemp(prefix="skill_load_"))


def write_skill(root: Path, frontmatter: str, body: str = "# 正文\n") -> Path:
    root.mkdir(exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\n" + frontmatter + "---\n" + body, encoding="utf-8"
    )
    return root


def test_load_exposes_facade_fields(workdir):
    root = write_skill(workdir / "hello-skill", "name: hello-skill\ndescription: 打招呼\n")
    (root / "resources").mkdir()
    (root / "resources" / "note.md").write_text("hi", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "hello.py").write_text("print(1)\n", encoding="utf-8")

    pkg = SkillPackage.load(root)

    assert pkg.name == "hello-skill"
    assert pkg.description == "打招呼"
    assert pkg.skill_md.strip() == "# 正文"
    assert pkg.frontmatter["name"] == "hello-skill"
    assert pkg.has_scripts is True
    assert pkg.has_resources is True


def test_load_rejects_missing_name(workdir):
    root = write_skill(workdir / "x", "description: 有描述\n")
    with pytest.raises(InvalidSkillPackageError) as exc:
        SkillPackage.load(root)
    assert exc.value.code == "SKILL_INVALID"
    assert "name" in exc.value.message
    assert exc.value.hint
    assert exc.value.docs_url.endswith("skill-invalid-package.md")


def test_load_rejects_non_kebab_name(workdir):
    root = write_skill(workdir / "demo", "name: Demo_Skill\ndescription: 描述\n")
    with pytest.raises(InvalidSkillPackageError) as exc:
        SkillPackage.load(root)
    assert "kebab-case" in exc.value.message


def test_load_rejects_missing_description(workdir):
    root = write_skill(workdir / "x-skill", "name: x-skill\n")
    with pytest.raises(InvalidSkillPackageError) as exc:
        SkillPackage.load(root)
    assert "description" in exc.value.message


def test_load_rejects_missing_entry(workdir):
    empty = workdir / "empty"
    empty.mkdir()
    with pytest.raises(InvalidSkillPackageError) as exc:
        SkillPackage.load(empty)
    assert exc.value.code == "SKILL_INVALID"


def test_load_rejects_unsatisfied_python_version(workdir):
    root = write_skill(
        workdir / "x-skill",
        'name: x-skill\ndescription: 描述\npython_version: ">=99.0"\n',
    )
    with pytest.raises(InvalidSkillPackageError) as exc:
        SkillPackage.load(root)
    assert "python_version" in exc.value.message


def test_load_accepts_satisfied_python_version(workdir):
    root = write_skill(
        workdir / "x-skill",
        'name: x-skill\ndescription: 描述\npython_version: ">=3.8"\n',
    )
    assert SkillPackage.load(root).name == "x-skill"


def test_as_skill_package_stays_lenient(workdir):
    '''旧门面不强制 kebab-case；load() 才是严格入口。'''
    root = write_skill(workdir / "demo", "name: demo_skill\ndescription: 描述\n")
    assert as_skill_package(root).name == "demo_skill"


def test_system_prompt_includes_env_facts(workdir):
    root = write_skill(workdir / "hello-skill", "name: hello-skill\ndescription: 打招呼\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "hello.py").write_text("print(1)\n", encoding="utf-8")

    pkg = SkillPackage.load(root)
    prompt = pkg.system_prompt(max_lines=10)

    assert "hello-skill" in prompt
    assert "打招呼" in prompt
    assert str(root) in prompt
    assert "# 正文" in prompt
    assert "run_script" in prompt
    assert "hello.py" in prompt


def test_install_deps_noop_without_requirements(workdir):
    root = write_skill(workdir / "hello-skill", "name: hello-skill\ndescription: 打招呼\n")
    assert SkillPackage.load(root).install_deps() is None


def test_install_deps_raises_on_failure(workdir, monkeypatch):
    root = write_skill(workdir / "hello-skill", "name: hello-skill\ndescription: 打招呼\n")
    (root / "requirements.txt").write_text("definitely_missing_pkg\n", encoding="utf-8")
    pkg = SkillPackage.load(root)

    class FakeResult:
        returncode = 1

    monkeypatch.setattr(package_mod.subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(DependencyInstallError) as exc:
        pkg.install_deps()
    assert exc.value.code == "SKILL_DEP_INSTALL"


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v"]))
