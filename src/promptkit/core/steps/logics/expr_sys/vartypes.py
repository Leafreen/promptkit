# core/steps/logics/expr_sys/vartypes.py
'''变量类型：Number / Str / File / PyObj / Bool / Time。

类型只做三件事：
    1. 声明变量时给出默认值
    2. 每次给这个变量赋值时，把值转换成对应的 Python 类型
    3. 提供该类型的基础操作，用 `File.read(Var("f"))` 这种函数式写法接进 DSL

所以 Read("请输入 a：") 读回来的 "12" 在赋值给 Number 变量时会变成 int 12，
`+` 才是加法而不是字符串拼接。

类型操作是**普通 Python 函数**，通过 `VarType.op` / `File.read` 包成
`CallExpr`，运行时才求值。这样 `File.exists(Var("f"))` 不会在传参时
去碰文件，而是等整个工作流跑到那一步。
'''
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Tuple, Union

from promptkit.core._missing import MISSING
from promptkit.core.errors import VarTypeError

if TYPE_CHECKING:  # 只用于类型标注，避免与 var.py 循环导入
    from promptkit.core.steps.logics.expr_sys.expr import CallExpr
    from promptkit.core.steps.logics.var import VarDecl


@dataclass(frozen=True)
class FileRef:
    '''File 变量的值：一个本地路径的轻量句柄。

    它不在赋值时校验文件存不存在（生成步骤常常先给路径、后写文件），
    只在真正读 / 写 / 查大小时才碰磁盘。
    '''

    path: str

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path
    
    def __bool__(self) -> bool:
        return self.exists()

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def suffix(self) -> str:
        return os.path.splitext(self.path)[1]

    @property
    def parent(self) -> "FileRef":
        return FileRef(os.path.dirname(self.path))

    @property
    def absolute(self) -> "FileRef":
        return FileRef(os.path.abspath(os.path.expanduser(self.path)))

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def is_file(self) -> bool:
        return os.path.isfile(self.path)

    def is_dir(self) -> bool:
        return os.path.isdir(self.path)

    def size(self) -> int:
        return os.path.getsize(self.path) if self.exists() else 0

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        with open(self.path, "r", encoding=encoding, errors=errors) as handle:
            return handle.read()

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as handle:
            return handle.read()

    def write_text(
        self, text: str, encoding: str = "utf-8", errors: str = "strict"
    ) -> "FileRef":
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.path, "w", encoding=encoding, errors=errors) as handle:
            handle.write(text)
        return self

    def write_bytes(self, data: bytes) -> "FileRef":
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.path, "wb") as handle:
            handle.write(data)
        return self

    def __repr__(self) -> str:
        return f"FileRef({self.path!r})"


class VarType:
    '''变量类型。实例本身可以直接当声明用：

        Number("a")                  # 声明一个 Number 变量，默认 0
        Number("a", default=1.5)
        File("src")                  # 声明一个 File 变量
        NewVar("a", type=Number)     # 等价写法

    类型的基础操作挂在实例上，例如：

        File.read(Var("src"))        # -> CallExpr，运行时读文件
        Time.format(Var("t"), "%Y")  # -> CallExpr
        Bool.not_(Var("ok"))
    '''

    def __init__(
        self,
        name: str,
        coerce: Callable[[object, str], object],
        default: object = MISSING,
        *,
        ops: Optional[Mapping[str, Callable[..., object]]] = None,
        default_factory: Optional[Callable[[], object]] = None,
    ):
        self.name = name
        self._coerce = coerce
        self.default = default
        self._ops: Dict[str, Callable[..., object]] = dict(ops or {})
        self._default_factory = default_factory

    def coerce(self, value: object, var_name: str) -> object:
        '''把 value 转成该类型；转不了就抛 VarTypeError。'''

        return self._coerce(value, var_name)

    def make_default(self) -> object:
        '''没有显式给 default 时用：优先工厂，其次静态默认值。'''
        if self._default_factory is not None:
            return self._default_factory()
        return self.default

    def operations(self) -> Tuple[str, ...]:
        '''这个类型暴露的基础操作名。'''
        return tuple(self._ops)

    def op(
        self, name: str, *args: object, **kwargs: object
    ) -> "CallExpr":
        '''把基础操作包成惰性表达式：`File.op("read", Var("f"))`。'''
        handler = self._ops.get(name)
        if handler is None:
            raise AttributeError(
                f"{self.name} 没有操作 {name!r}；可用：{', '.join(self.operations())}"
            )
        from promptkit.core.steps.logics.expr_sys.expr import CallExpr

        return CallExpr(handler, *args, **kwargs)

    def __getattr__(self, name: str) -> Callable[..., object]:
        '''让类型操作读起来像函数：`File.read(...)` / `Time.format(...)`。'''
        if name.startswith("_"):
            raise AttributeError(name)
        ops = self.__dict__.get("_ops") or {}
        if name not in ops:
            raise AttributeError(
                f"{self.__dict__.get('name', 'VarType')} 没有操作 {name!r}；"
                f"可用：{', '.join(ops)}"
            )

        def operation(*args: object, **kwargs: object) -> "CallExpr":
            return self.op(name, *args, **kwargs)

        # repr(CallExpr) 会读 __name__，这里保留类型操作的名字
        operation.__name__ = name  # type: ignore[attr-defined]
        return operation

    def __call__(self, name: str, default: object = MISSING) -> "VarDecl":
        '''声明一个该类型的变量，等价于 NewVar(name, type=self)。'''
        # 延迟导入，避免 core.vartypes <-> core.steps.var 的循环依赖
        from promptkit.core.steps.logics.var import NewVar

        return NewVar(name, type=self, default=default)

    def __repr__(self) -> str:
        return self.name


def _to_number(value: object, var_name: str) -> Union[int, float]:
    # bool 是 int 的子类，这里不特殊处理，方便当 0/1 用
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    pass
    raise VarTypeError(
        f"变量 {var_name!r} 是 Number，无法把 {value!r} 转成数字"
    )


def _to_str(value: object, var_name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return str(value)
    raise VarTypeError(
        f"变量 {var_name!r} 是 Str，无法把 {type(value).__name__} 转成字符串"
    )


def _to_file(value: object, var_name: str) -> FileRef:
    if isinstance(value, FileRef):
        return value
    if isinstance(value, os.PathLike):
        return FileRef(os.fspath(value))
    if isinstance(value, str):
        text = value.strip()
        if text:
            return FileRef(os.path.expanduser(text))
        raise VarTypeError(f"变量 {var_name!r} 是 File，路径不能是空字符串")
    # Media 这类「有 path 属性」的对象也可以直接赋给 File
    path = getattr(value, "path", None)
    if isinstance(path, str) and path:
        return FileRef(os.path.expanduser(path))
    raise VarTypeError(
        f"变量 {var_name!r} 是 File，无法把 {type(value).__name__} 转成文件路径"
    )


def _to_pyobj(value: object, var_name: str) -> object:
    '''PyObj 接受任意 Python 对象，不做转换。'''
    return value


_BOOL_TRUE = {"true", "yes", "on", "1", "是", "真"}
_BOOL_FALSE = {"false", "no", "off", "0", "否", "假"}


def _to_bool(value: object, var_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
    raise VarTypeError(
        f"变量 {var_name!r} 是 Bool，无法把 {value!r} 转成布尔值"
        "（可接受 true/false、yes/no、on/off、1/0、是/否）"
    )


_TIME_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y%m%d",
)


def _to_time(value: object, var_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            pass
    if isinstance(value, str):
        text = value.strip()
        if text:
            # Python 3.10 的 fromisoformat 不认结尾的 Z
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                return datetime.fromisoformat(iso)
            except ValueError:
                pass
            for fmt in _TIME_FORMATS:
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue
    raise VarTypeError(
        f"变量 {var_name!r} 是 Time，无法把 {value!r} 转成时间"
        "（可接受 datetime / date / 秒级时间戳 / ISO 8601 字符串）"
    )


# ---------------- File 的基础操作 ----------------
def _as_file(value: object) -> FileRef:
    if isinstance(value, FileRef):
        return value
    if isinstance(value, str):
        return FileRef(value)
    if isinstance(value, os.PathLike):
        return FileRef(os.fsdecode(value))
    raise TypeError(f"期望文件路径或 FileRef，得到 {type(value).__name__}")


def _file_read(value: object, encoding: str = "utf-8", errors: str = "strict") -> str:
    return _as_file(value).read_text(encoding=encoding, errors=errors)


def _file_read_bytes(value: object) -> bytes:
    return _as_file(value).read_bytes()


def _file_write(value: object, content: str, encoding: str = "utf-8") -> FileRef:
    return _as_file(value).write_text(str(content), encoding=encoding)


def _file_exists(value: object) -> bool:
    return _as_file(value).exists()


def _file_is_file(value: object) -> bool:
    return _as_file(value).is_file()


def _file_is_dir(value: object) -> bool:
    return _as_file(value).is_dir()


def _file_filename(value: object) -> str:
    return _as_file(value).name


def _file_suffix(value: object) -> str:
    return _as_file(value).suffix


def _file_size(value: object) -> int:
    return _as_file(value).size()


def _file_absolute(value: object) -> FileRef:
    return _as_file(value).absolute


# ---------------- PyObj 的基础操作 ----------------
def _pyobj_attr(value: object, name: str) -> object:
    return getattr(value, name)


def _pyobj_get(value: object, key: object, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return value[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return getattr(value, str(key), default)


def _pyobj_call(value: object, *args: object, **kwargs: object) -> object:
    if not callable(value):
        raise TypeError(f"{value!r} 不是可调用对象")
    return value(*args, **kwargs)


def _pyobj_keys(value: object) -> Tuple[object, ...]:
    keys = getattr(value, "keys", None)
    if not callable(keys):
        raise TypeError(f"{type(value).__name__} 没有 keys()")
    return tuple(keys())


def _pyobj_type_name(value: object) -> str:
    return type(value).__name__


# ---------------- Bool 的基础操作 ----------------
def _bool_not(value: object) -> bool:
    return not bool(value)


def _bool_and(left: object, right: object) -> bool:
    return bool(left) and bool(right)


def _bool_or(left: object, right: object) -> bool:
    return bool(left) or bool(right)


def _bool_int(value: object) -> int:
    return int(bool(value))


# ---------------- Time 的基础操作 ----------------
def _time_now() -> datetime:
    return datetime.now()


def _time_format(value: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return value.strftime(fmt)


def _time_iso(value: datetime) -> str:
    return value.isoformat()


def _time_add(
    value: datetime,
    seconds: float = 0,
    *,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    weeks: float = 0,
) -> datetime:
    return value + timedelta(
        seconds=seconds,
        days=days,
        hours=hours,
        minutes=minutes,
        weeks=weeks,
    )


def _time_diff(value: datetime, other: datetime) -> float:
    '''两个时间相差多少秒（value - other）。'''
    return (value - other).total_seconds()


def _time_year(value: datetime) -> int:
    return value.year


def _time_month(value: datetime) -> int:
    return value.month


def _time_day(value: datetime) -> int:
    return value.day


def _time_timestamp(value: datetime) -> float:
    return value.timestamp()


# 只需要这些类型
Number = VarType("Number", _to_number, 0)
Str = VarType("Str", _to_str, "")
File = VarType(
    "File",
    _to_file,
    FileRef(""),
    ops={
        "read": _file_read,
        "read_bytes": _file_read_bytes,
        "write": _file_write,
        "exists": _file_exists,
        "is_file": _file_is_file,
        "is_dir": _file_is_dir,
        "filename": _file_filename,
        "suffix": _file_suffix,
        "size": _file_size,
        "absolute": _file_absolute,
    },
)
PyObj = VarType(
    "PyObj",
    _to_pyobj,
    None,
    ops={
        "attr": _pyobj_attr,
        "get": _pyobj_get,
        "call": _pyobj_call,
        "keys": _pyobj_keys,
        "type_name": _pyobj_type_name,
    },
)
Bool = VarType(
    "Bool",
    _to_bool,
    False,
    ops={
        "not_": _bool_not,
        "and_": _bool_and,
        "or_": _bool_or,
        "int_": _bool_int,
    },
)
Time = VarType(
    "Time",
    _to_time,
    default_factory=datetime.now,
    ops={
        "now": _time_now,
        "format": _time_format,
        "iso": _time_iso,
        "add": _time_add,
        "diff": _time_diff,
        "year": _time_year,
        "month": _time_month,
        "day": _time_day,
        "timestamp": _time_timestamp,
    },
)


def var_type_name(var_type: object) -> str:
    '''给错误信息和 repr 用。'''
    return var_type.name if isinstance(var_type, VarType) else str(var_type)


__all__ = [
    "VarType",
    "FileRef",
    "Number",
    "Str",
    "File",
    "PyObj",
    "Bool",
    "Time",
    "var_type_name",
]
