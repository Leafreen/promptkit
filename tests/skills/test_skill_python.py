'''解释器解析：优先级、错误处理与技能包虚拟环境探测。'''
import os
import sys
from pathlib import Path

import pytest

from promptkit.skill import (
    PythonInterpreterError,
    as_skill_package,
    resolve_python,
)
from promptkit.skill.python import ENV_VAR


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    '''默认清掉环境变量，避免开发者本机配置影响断言。'''
    monkeypatch.delenv(ENV_VAR, raising=False)


def test_explicit_interpreter_wins(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "definitely-not-a-python")
    assert Path(resolve_python(sys.executable)) == Path(sys.executable)


def test_explicit_broken_interpreter_raises(tmp_path):
    with pytest.raises(PythonInterpreterError) as excinfo:
        resolve_python(tmp_path / "missing-python")
    assert excinfo.value.code == "SKILL_PYTHON"


def test_env_var_used_without_explicit():
    os.environ[ENV_VAR] = sys.executable
    try:
        assert Path(resolve_python()) == Path(sys.executable)
    finally:
        os.environ.pop(ENV_VAR, None)


def test_env_var_broken_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing-python"))
    with pytest.raises(PythonInterpreterError):
        resolve_python()


def test_missing_venv_falls_back_to_current(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    assert Path(resolve_python(root=tmp_path)) == Path(sys.executable)


def test_venv_is_preferred_when_available(tmp_path, monkeypatch):
    from promptkit.skill import python as resolver

    venv_python = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv_python = venv_python / ("python.exe" if os.name == "nt" else "python")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        resolver, "_probe", lambda path: Path(path) in {venv_python, Path(sys.executable)}
    )

    assert Path(resolve_python(root=tmp_path)) == venv_python


def test_non_executable_file_rejected(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("not a real interpreter", encoding="utf-8")

    with pytest.raises(PythonInterpreterError):
        resolve_python(fake)


def test_package_runs_script_without_python_argument(tmp_path):
    root = tmp_path / "demo"
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: 演示技能\n---\n# 正文\n", encoding="utf-8"
    )
    (root / "scripts" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    package = as_skill_package(root)

    result = package.run_script("main.py")

    assert '"returncode": 0' in result
    assert "ok" in result
