'''约束（Caution）：一段追加到提示词末尾的指令。

约束是**对象**，不是硬编码字符串 —— 所以可以组合（`+`）、复用、单独测试：

    cautions=[OutputFormat("JSON"), NoExtraText()]
    cautions=OutputFormat("JSON") + NoExtraText() + Language("中文")
    cautions=JSON_ONLY

Prompt 只写任务，约束单独管理，渲染时才拼到 user 内容末尾。
'''
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

# Workflow 级约束挂在 Context 上的键名
WORKFLOW_CAUTIONS_KEY = "_cautions"


class Caution(ABC):
    '''约束基类。子类实现 `to_text()`。'''

    @abstractmethod
    def to_text(self) -> str:
        '''这个约束的文本表示。空字符串表示「什么都没说」，会被跳过。'''

    # ---------- 组合 ----------
    def __add__(self, other: Any) -> "CautionGroup":
        return CautionGroup([self, other])

    def __radd__(self, other: Any) -> "Caution":
        # sum([c1, c2]) 从 0 开始，所以 0 + c1 要返回 c1 本身
        if other == 0:
            return self
        return CautionGroup([other, self])

    # ---------- 展示 / 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        '''给日志 / trace 用的可序列化形式。

        注意这是**有损**的：只记类型名和文本。想还原请直接用代码构造，
        或者用 `Custom(text)` 装回一段纯文本。
        '''
        return {"type": type(self).__name__, "text": self.to_text()}

    def __bool__(self) -> bool:
        return bool(self.to_text())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_text()!r})"


class CautionGroup(Caution):
    '''多个约束的组合，按顺序拼成多行。'''

    def __init__(self, cautions: Iterable[Any] = ()):
        self.cautions: List[Caution] = list(flatten_cautions(cautions))

    def to_text(self) -> str:
        return "\n".join(c.to_text() for c in self.cautions if c.to_text())

    def __add__(self, other: Any) -> "CautionGroup":
        return CautionGroup([*self.cautions, *flatten_cautions([other])])

    def __iter__(self) -> Iterator[Caution]:
        return iter(self.cautions)

    def __len__(self) -> int:
        return len(self.cautions)

    def __bool__(self) -> bool:
        return any(bool(c) for c in self.cautions)

    def __repr__(self) -> str:
        return f"CautionGroup({[c.to_text() for c in self.cautions]!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CautionGroup", "cautions": [c.to_dict() for c in self.cautions]}


class Custom(Caution):
    '''一段纯文本约束（想快速塞一句就用它）。'''

    def __init__(self, text: str):
        self.text = text

    def to_text(self) -> str:
        return self.text


def as_caution(item: Any) -> Caution:
    '''把「看起来像约束」的东西变成 Caution。'''
    if isinstance(item, Caution):
        return item
    if isinstance(item, str):
        return Custom(item)
    raise TypeError(
        f"约束必须是 Caution（或字符串），得到 {type(item).__name__}：{item!r}"
    )


def flatten_cautions(items: Any) -> Iterator[Caution]:
    '''展开嵌套的列表 / 元组 / CautionGroup，顺便把字符串包成 Custom。'''
    if items is None:
        return
    if isinstance(items, CautionGroup):
        for item in items.cautions:
            yield from flatten_cautions(item)
        return
    if isinstance(items, Caution):
        yield items
        return
    if isinstance(items, str):
        yield Custom(items)
        return
    if isinstance(items, (list, tuple, set)):
        for item in items:
            yield from flatten_cautions(item)
        return
    raise TypeError(
        f"约束必须是 Caution、字符串或它们的列表，得到 {type(items).__name__}：{items!r}"
    )


def join_cautions(cautions: Iterable[Caution]) -> str:
    '''把若干约束拼成一段文本（空的行会被丢掉）。'''
    return "\n".join(c.to_text() for c in cautions if c.to_text())


def resolve_cautions(
    step: Any = (),
    workflow: Any = (),
    *,
    use_defaults: bool = True,
    dedup: bool = True,
) -> List[Caution]:
    '''按**优先级从低到高**合并三层约束：全局 -> Workflow -> Step。

    后面的更「就近」，所以排在文本更靠后的位置（离模型最近的指令）。
    `dedup=True` 会去掉文本完全相同的重复项（全局 + 预置组合很容易撞）。
    '''
    merged: List[Caution] = []

    if use_defaults:
        # 延迟导入：避免 defaults <-> caution 的循环
        from promptkit.core.model_action.defaults import get_default_cautions

        merged.extend(flatten_cautions(get_default_cautions()))

    merged.extend(flatten_cautions(workflow))
    merged.extend(flatten_cautions(step))

    if not dedup:
        return [c for c in merged if c.to_text()]

    seen = set()
    result: List[Caution] = []
    for caution in merged:
        text = caution.to_text()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(caution)
    return result


__all__ = [
    "Caution",
    "CautionGroup",
    "Custom",
    "WORKFLOW_CAUTIONS_KEY",
    "as_caution",
    "flatten_cautions",
    "join_cautions",
    "resolve_cautions",
]
