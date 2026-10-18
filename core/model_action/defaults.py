'''全局默认约束。

不想每个 LLMStep 都写一遍 `cautions=[...]`，就设一次全局默认：

    set_default_cautions([NoExtraText(), Language("中文")])

    # 之后所有 LLMStep 自动带上（Step 自己的追加在后面，就近优先）
    LLMStep(prompt=..., output_model=Intent, output_key="intent")

⚠️ 这是**进程级可变状态**，和 `Agent.default()` 一样：
测试里用完记得 `clear_default_cautions()`，或者用 `using_default_cautions(...)`
这个上下文管理器，它会自动还原。
'''
from contextlib import contextmanager
from typing import Any, Iterator, List, Sequence

from promptkit.core.model_action.caution_related.caution import Caution, flatten_cautions

_default_cautions: List[Caution] = []


def set_default_cautions(cautions: Any) -> None:
    '''替换全局默认约束。'''
    global _default_cautions
    _default_cautions = list(flatten_cautions(cautions))


def add_default_caution(*cautions: Any) -> None:
    '''往全局默认里追加（不动已有的）。'''
    global _default_cautions
    _default_cautions = [*_default_cautions, *flatten_cautions(cautions)]


def get_default_cautions() -> List[Caution]:
    '''返回全局默认的**副本**，外面改它不会影响全局。'''
    return list(_default_cautions)


def clear_default_cautions() -> None:
    '''清空全局默认。'''
    global _default_cautions
    _default_cautions = []


@contextmanager
def using_default_cautions(*cautions: Any) -> Iterator[List[Caution]]:
    '''临时替换全局默认约束，退出时自动还原（测试用）。

        with using_default_cautions(NoExtraText()):
            ...
    '''
    global _default_cautions
    saved = _default_cautions
    _default_cautions = list(flatten_cautions(cautions))
    try:
        yield list(_default_cautions)
    finally:
        _default_cautions = saved


__all__ = [
    "set_default_cautions",
    "add_default_caution",
    "get_default_cautions",
    "clear_default_cautions",
    "using_default_cautions",
]
