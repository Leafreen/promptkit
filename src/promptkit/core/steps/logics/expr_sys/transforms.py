# core/steps/logics/expr_sys/transforms.py
'''可注册、可序列化的数据转换：`Transform("join", Var("lines"), sep="\\n")`。

为什么要有它
------------
`Call` 能直接调任意 Python 函数，这带来两个问题：

1. **不可序列化**：lambda / 闭包 / 动态表达式没法落盘、没法从配置重建；
2. **不可审计**：看一眼 DSL 不知道它到底干了什么。

`Transform` 用「名字 → 函数」的注册表把这件事收紧：DSL 里只出现注册过的
名字，参数仍是表达式（`Var(...)` / 常量），因此可以完整地 `to_dict()` /
`from_dict()`。

内置转换
--------
`join` / `split` / `upper` / `lower` / `strip` / `len` / `append` /
`extend` / `list` / `dict` / `json_dumps` / `json_loads` / `format`

扩展方式
--------

    from promptkit.core import transform

    @transform("shout")
    def _shout(text: str) -> str:
        return text.upper() + "!"

    Transform("shout", Var("text"))

未注册的名字会抛 `UnknownTransformError`（而不是在运行时才发现）。
'''
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from promptkit.core.errors import UnknownTransformError

from .expr import Const, Expr, VarExpr, as_expr

__all__ = [
    "Transform",
    "get_transform",
    "register_transform",
    "registered_transforms",
    "transform",
    "transform_name",
]

TransformFn = Callable[..., Any]

#: 名字 -> 函数
_REGISTRY: Dict[str, TransformFn] = {}
#: 函数 -> 名字（用于把旧的 `Call(fn, ...)` 翻译成 `Transform("name", ...)`）
_BY_FUNC: Dict[TransformFn, str] = {}


def register_transform(name: str, fn: TransformFn) -> TransformFn:
    '''注册一个具名转换；同名覆盖，返回原函数方便当装饰器用。'''
    if not isinstance(name, str) or not name:
        raise TypeError(f"转换名必须是非空字符串，得到 {name!r}")
    if not callable(fn):
        raise TypeError(f"转换实现必须可调用，得到 {fn!r}")
    _REGISTRY[name] = fn
    _BY_FUNC.setdefault(fn, name)
    return fn


def transform(name: str) -> Callable[[TransformFn], TransformFn]:
    '''装饰器写法：`@transform("join")`。'''

    def decorator(fn: TransformFn) -> TransformFn:
        return register_transform(name, fn)

    return decorator


def get_transform(name: str) -> TransformFn:
    '''按名字取转换函数；未注册时抛 `UnknownTransformError`。'''
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "（无）"
        raise UnknownTransformError(
            f"没有注册名为 {name!r} 的转换；可用：{available}"
        ) from None


def transform_name(fn: TransformFn) -> str | None:
    '''反查函数注册名；没注册过返回 None。'''
    return _BY_FUNC.get(fn)


def registered_transforms() -> List[str]:
    '''已注册的转换名（排序后）。'''
    return sorted(_REGISTRY)


class Transform(Expr):
    '''具名数据转换表达式。

        Transform("join", Var("lines"), sep="\\n")
        Transform("list")
        Transform("append", Var("lines"), Var("line"))
    '''

    def __init__(self, name: str, *args: object, **kwargs: object):
        if not isinstance(name, str) or not name:
            raise TypeError(f"转换名必须是非空字符串，得到 {name!r}")
        # 构造时就校验，避免写错名字要等到运行时才炸
        get_transform(name)
        self.name = name
        self.args = [as_expr(arg) for arg in args]
        self.kwargs = {key: as_expr(value) for key, value in kwargs.items()}

    def eval(self, context: object) -> object:
        '''从注册表取函数，求值参数后调用。'''
        from promptkit.core.context import Context

        if not isinstance(context, Context):
            raise TypeError(f"Transform 需要 Context，得到 {type(context).__name__}")
        fn = get_transform(self.name)
        return fn(
            *(arg.eval(context) for arg in self.args),
            **{key: value.eval(context) for key, value in self.kwargs.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        '''序列化成 `{"type": "transform", "name": ..., "args": [...]}`。'''
        return {
            "type": "transform",
            "name": self.name,
            "args": [_expr_to_dict(arg) for arg in self.args],
            "kwargs": {key: _expr_to_dict(value) for key, value in self.kwargs.items()},
        }

    @classmethod
    def from_dict(cls, data: object) -> Transform:
        '''从 `to_dict()` 的结果重建；坏数据抛 TypeError。'''
        if not isinstance(data, dict):
            raise TypeError(f"Transform 数据必须是 dict，得到 {type(data).__name__}")
        if data.get("type") != "transform":
            raise TypeError(f"不是 transform 数据：{data.get('type')!r}")
        name = data.get("name")
        if not isinstance(name, str):
            raise TypeError(f"name 必须是字符串，得到 {name!r}")
        raw_args = data.get("args", [])
        raw_kwargs = data.get("kwargs", {})
        if not isinstance(raw_args, list):
            raise TypeError("args 必须是列表")
        if not isinstance(raw_kwargs, dict):
            raise TypeError("kwargs 必须是字典")
        return cls(
            name,
            *[_expr_from_dict(item) for item in raw_args],
            **{key: _expr_from_dict(value) for key, value in raw_kwargs.items()},
        )

    def __repr__(self) -> str:
        parts = [repr(self.name)] + [repr(arg) for arg in self.args]
        parts += [f"{key}={value!r}" for key, value in self.kwargs.items()]
        return f"Transform({', '.join(parts)})"


def _expr_to_dict(value: Expr) -> dict[str, Any]:
    '''只序列化 Var / 常量 / 嵌套 Transform；别的一律拒绝。'''
    if isinstance(value, Transform):
        return value.to_dict()
    if isinstance(value, VarExpr):
        return {"type": "var", "name": value.var}
    if isinstance(value, Const):
        return {"type": "const", "value": value.value}
    raise TypeError(
        f"Transform 只能序列化 Var / 常量 / 嵌套 Transform，"
        f"得到 {type(value).__name__}"
    )


def _expr_from_dict(data: object) -> Expr:
    '''`_expr_to_dict` 的逆操作。'''
    if not isinstance(data, dict):
        raise TypeError(f"表达式数据必须是 dict，得到 {type(data).__name__}")
    kind = data.get("type")
    if kind == "transform":
        return Transform.from_dict(data)
    if kind == "var":
        name = data.get("name")
        if not isinstance(name, str):
            raise TypeError(f"var 的 name 必须是字符串，得到 {name!r}")
        return VarExpr(name)
    if kind == "const":
        return Const(data.get("value"))
    raise TypeError(f"未知的表达式类型：{kind!r}")


# ---------------------- 内置转换 ----------------------

@transform("join")
def _join(items: List[object], sep: str = "\n") -> str:
    '''把序列拼成字符串，默认换行分隔。'''
    return sep.join(str(item) for item in items)


@transform("split")
def _split(text: str, sep: str | None = None) -> List[str]:
    '''按分隔符切分字符串；sep=None 表示按空白切。'''
    return text.split(sep)


@transform("upper")
def _upper(text: str) -> str:
    '''转大写。'''
    return text.upper()


@transform("lower")
def _lower(text: str) -> str:
    '''转小写。'''
    return text.lower()


@transform("strip")
def _strip(text: str, chars: str | None = None) -> str:
    '''去掉首尾空白（或指定字符）。'''
    return text.strip(chars)


@transform("len")
def _len(value: object) -> int:
    '''长度。'''
    return len(value)  # type: ignore[arg-type]


@transform("append")
def _append(items: List[object], item: object) -> List[object]:
    '''返回「追加了一项」的新列表；不修改入参。'''
    return [*items, item]


@transform("extend")
def _extend(items: List[object], extra: List[object]) -> List[object]:
    '''返回「拼接后」的新列表；不修改入参。'''
    return [*items, *extra]


@transform("list")
def _to_list(value: object = ()) -> List[object]:
    '''转成列表。'''
    return list(value)  # type: ignore[arg-type]


@transform("dict")
def _to_dict(source: object = (), **kwargs: object) -> Dict[str, object]:
    '''转成字典；支持 dict() / dict(mapping, **kwargs) 两种用法。'''
    return dict(source, **kwargs)  # type: ignore[call-overload]


@transform("json_dumps")
def _json_dumps(value: object, ensure_ascii: bool = False, indent: int | None = None) -> str:
    '''对象转 JSON 字符串。'''
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


@transform("json_loads")
def _json_loads(text: str) -> object:
    '''JSON 字符串转对象。'''
    return json.loads(text)


@transform("format")
def _format(template: str, **values: object) -> str:
    '''`str.format` 的表达式版本：`Transform("format", "{a}-{b}", a=..., b=...)`。'''
    return template.format(**values)


# 本文件核心职责：Transform 注册表 + 内置转换 + 可序列化的 Transform 表达式。
