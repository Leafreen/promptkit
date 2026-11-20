"""根命名空间导出契约测试：`from aipykit import *` 必须覆盖全部公开功能。

这些用例守住三件事：

1. ``aipykit.__all__`` 无重复，且每个名字都能真正取到值；
2. 各子包 ``__all__`` 里声明过的名字一个都不能在根命名空间漏掉；
3. 老的深层导入路径（``aipykit.core.Workflow`` 等）保持可用。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest

import aipykit


def _exported() -> set[str]:
    return set(aipykit.__all__)


def test_all_has_no_duplicates() -> None:
    names = list(aipykit.__all__)
    assert len(names) == len(set(names)), "aipykit.__all__ 里有重复名字"


def test_all_names_resolve() -> None:
    """每个公开名字都能从根命名空间取到；可选依赖缺失时给出明确提示。"""
    for name in aipykit.__all__:
        try:
            getattr(aipykit, name)
        except AttributeError as exc:  # pragma: no cover - 仅在缺可选依赖时触发
            assert "可选依赖" in str(exc), f"{name} 解析失败：{exc}"


def test_subpackage_all_is_subset_of_root() -> None:
    """各子包 ``__all__`` 声明的名字，根命名空间必须全部覆盖。"""
    exported = _exported()
    missing: dict[str, list[str]] = {}
    for module_name in dict(aipykit._EXPORT_GROUPS):
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            if name not in exported:
                missing.setdefault(module_name, []).append(name)
    assert not missing, f"以下子包符号没有聚合到根命名空间：{missing}"


def test_star_import_exposes_every_public_name() -> None:
    namespace: dict[str, object] = {}
    exec("from aipykit import *", namespace)  # noqa: S102 - 就是要验证 import *
    declared = {name for name in namespace if not name.startswith("__")}
    assert declared == _exported()
    for name in aipykit.__all__:
        assert name in namespace


def test_root_import_is_lazy() -> None:
    """`import aipykit` 不应把核心子包一起拖进来。"""
    code = (
        "import sys, aipykit; "
        "loaded = [m for m in ('aipykit.core', 'aipykit.agent', 'pydantic') "
        "if m in sys.modules]; "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"import aipykit 时提前加载了：{result.stdout}"


def test_key_api_is_directly_usable() -> None:
    """常用 API 从根命名空间拿到后即可直接用，不需要额外导入子模块。"""
    assert aipykit.Prompt("你好，{{name}}").render(name="小明") == {
        "role": "user",
        "content": "你好，小明",
    }
    assert str(aipykit.Var("score") + 1)  # 表达式系统可用
    assert aipykit.Workflow is not None
    assert aipykit.Tool.from_function(lambda city: city, name="echo_city") is not None
    assert aipykit.Echo("hi") is not None
    assert aipykit.KnowledgeBase.from_texts(["退款流程"]).search("退款")
    assert aipykit.InMemoryMemory().read("missing", "默认值") == "默认值"


def test_submodules_are_lazily_addressable() -> None:
    """`aipykit.core` / `aipykit.config` 这类属性访问不需要显式 import。"""
    assert isinstance(aipykit.config, types.ModuleType)
    assert isinstance(aipykit.core, types.ModuleType)
    assert aipykit.core.Workflow is aipykit.Workflow


def test_features_is_module() -> None:
    """`from aipykit import features` 拿到模块，而不是同名函数。"""
    assert isinstance(aipykit.features, types.ModuleType)
    assert callable(aipykit.features.enable)
    assert aipykit.features is importlib.import_module("aipykit.features")


def test_skill_module_and_decorator_are_reachable() -> None:
    """根命名空间的 `skill` 是模块；装饰器走模块属性，深层导入不被打断。"""
    import aipykit.skill.package as package_mod

    assert isinstance(aipykit.skill, types.ModuleType)
    assert callable(aipykit.skill.skill)
    assert aipykit.skill.SkillRegistry is aipykit.SkillRegistry
    assert package_mod is importlib.import_module("aipykit.skill.package")


def test_backward_compatible_deep_imports() -> None:
    """老路径全部保持可用，关键对象与根命名空间指向同一个东西。"""
    from aipykit.core import Workflow
    from aipykit.core.steps import LLMStep
    from aipykit.mcp import MCPError
    from aipykit.memory import KVMemory
    from aipykit.orchestrate import Orchestrator
    from aipykit.session import Session
    from aipykit.tui import TuiSession

    assert Workflow is aipykit.Workflow
    assert LLMStep is aipykit.LLMStep
    assert KVMemory is aipykit.KVMemory
    assert Orchestrator is aipykit.Orchestrator
    assert Session is aipykit.Session
    assert TuiSession is aipykit.TuiSession
    assert issubclass(MCPError, Exception)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        aipykit.DoesNotExist  # noqa: B018 - 就是在取一个不存在的属性
