# tests/test_settings_stage1.py
'''阶段 1：Settings 不可变 + Workflow 级覆盖的验收测试。'''
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from promptkit.core import (
    Const,
    Context,
    LoopLimitExceeded,
    While,
    Workflow,
)
from promptkit.core.settings import (
    Settings,
    reset_settings,
    settings,
    using_settings,
    validate_settings,
)


@pytest.fixture(autouse=True)
def _clean_settings():
    '''每个用例前后都把全局覆盖清干净，避免相互污染。'''
    reset_settings()
    yield
    reset_settings()


def test_settings_snapshot_is_frozen() -> None:
    '''`Settings` 是快照，不能再当可变全局用。'''
    snapshot = Settings()
    with pytest.raises(FrozenInstanceError):
        snapshot.max_iterations = 1  # type: ignore[misc]
    assert snapshot.max_iterations == Settings().max_iterations


def test_direct_assignment_warns_but_still_works() -> None:
    '''老写法 `settings.xxx = ...` 保留可用，但会发 DeprecationWarning。'''
    with pytest.warns(DeprecationWarning, match="阶段 1"):
        settings.max_retries = 9
    assert settings.max_retries == 9
    assert settings.to_dict()["max_retries"] == 9


def test_using_settings_restores() -> None:
    '''using_settings 退出后，全局默认回到原值。'''
    assert settings.max_iterations == Settings().max_iterations
    with using_settings(max_iterations=3):
        assert settings.max_iterations == 3
    assert settings.max_iterations == Settings().max_iterations


def test_workflow_construction_override() -> None:
    '''`Workflow(max_iterations=...)` 对条件循环生效，且不用改全局。'''
    wf = Workflow(While(Const(True)), max_iterations=2)
    with pytest.raises(LoopLimitExceeded, match="2 次"):
        wf.execute(Context())

    # 全局默认没被污染
    assert settings.max_iterations == Settings().max_iterations


def test_run_override_beats_construction_override() -> None:
    '''`wf.run(max_iterations=...)` 可以再覆盖一次。'''
    wf = Workflow(While(Const(True)), max_iterations=10)
    with pytest.raises(LoopLimitExceeded, match="3 次"):
        wf.run(max_iterations=3)


def test_invalid_override_is_rejected_early() -> None:
    '''未知设置项 / 非法值在构造时就报错，而不是跑起来才炸。'''
    with pytest.raises(TypeError, match="没有这个设置项"):
        Workflow(overrides={"max_iteration": 1})  # 少个 s
    with pytest.raises(ValueError, match="正整数"):
        validate_settings({"max_iterations": 0})
