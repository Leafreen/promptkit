'''功能开关。

默认只开**核心能力**：工作流 / 提示词 / Agent / 工具 / 约束 / 事件 / 结构化输出。
实验性功能默认**关闭**，而且关得很明确 —— import 的时候直接报错并告诉你怎么打开，
而不是留一个半边能用、半边不能用、没人说得清状态的模块。

    from promptkit import features

    features.enable("knowledge")        # 或者设环境变量 PROMPTKIT_ENABLE_KNOWLEDGE=1

    from promptkit.knowledge import KnowledgeBase

    features.disable("knowledge")       # 关回去
    features.status()                   # 看看现在都开了什么

测试里想临时开关，用 `features.using(...)`，退出会自动还原。
'''
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List

__all__ = [
    "FeatureDisabled",
    "FEATURE_ENV",
    "features",
    "enabled",
    "enable",
    "disable",
    "require",
    "using",
    "status",
]


class FeatureDisabled(RuntimeError):
    '''用到了没打开的功能。'''


#: 功能名 -> 对应的环境变量名
FEATURE_ENV: Dict[str, str] = {
    "knowledge": "PROMPTKIT_ENABLE_KNOWLEDGE",
}

#: 「打开」的各种写法
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}

#: 运行时开关。默认全关 —— 要显式打开才算数。
_enabled: Dict[str, bool] = {name: False for name in FEATURE_ENV}


def features() -> List[str]:
    '''所有可开关的功能名。'''
    return sorted(FEATURE_ENV)


def _check(name: str) -> str:
    if name not in FEATURE_ENV:
        raise KeyError(
            f"没有这个功能开关：{name!r}；可用的有：{', '.join(features()) or '（空）'}"
        )
    return FEATURE_ENV[name]


def enabled(name: str) -> bool:
    '''功能开没开：运行时开关优先，其次是环境变量。'''
    env = _check(name)
    if _enabled.get(name):
        return True
    return os.environ.get(env, "").strip().lower() in _TRUTHY


def enable(name: str, flag: bool = True) -> None:
    '''打开（或关掉）一个功能。'''
    _check(name)
    _enabled[name] = bool(flag)


def disable(name: str) -> None:
    '''关掉一个功能（环境变量仍然可以把它打开）。'''
    enable(name, False)


def require(name: str) -> None:
    '''要求某个功能是开的，否则抛 FeatureDisabled。

    给实验性模块在 import 时做守卫用。
    '''
    if enabled(name):
        return
    env = _check(name)
    raise FeatureDisabled(
        f"功能 {name!r} 默认关闭。启用方式（任选其一）：\n"
        f"  1) from promptkit import features; features.enable({name!r})\n"
        f"  2) 设置环境变量 {env}=1\n"
        f"（必须在 import 用到它的模块**之前**做）"
    )


@contextmanager
def using(name: str, flag: bool = True) -> Iterator[bool]:
    '''临时打开/关掉一个功能，退出时还原（测试用）。'''
    _check(name)
    saved = _enabled.get(name, False)
    _enabled[name] = bool(flag)
    try:
        yield bool(flag)
    finally:
        _enabled[name] = saved


def status() -> Dict[str, str]:
    '''每个功能现在的状态。'''
    return {
        name: (
            "on (runtime)" if _enabled.get(name) else "on (env)" if enabled(name) else "off"
        )
        for name in features()
    }
