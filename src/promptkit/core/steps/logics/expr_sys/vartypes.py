# core/vartypes.py
'''变量类型：Number 和 Str。

类型只做两件事：
    1. 声明变量时给出默认值
    2. 每次给这个变量赋值时，把值转换成对应的 Python 类型

所以 Read("请输入 a：") 读回来的 "12" 在赋值给 Number 变量时会变成 int 12，
`+` 才是加法而不是字符串拼接。
'''
from typing import Any, Callable

from promptkit.core._missing import MISSING
from promptkit.core.errors import VarTypeError


class VarType:
    '''变量类型。实例本身可以直接当声明用：

        Number("a")                  # 声明一个 Number 变量，默认 0
        Number("a", default=1.5)
        NewVar("a", type=Number)     # 等价写法
    '''

    def __init__(self, name: str, coerce: Callable[[Any, str], Any], default: Any):
        self.name = name
        self._coerce = coerce
        self.default = default

    def coerce(self, value: Any, var_name: str) -> Any:
        '''把 value 转成该类型；转不了就抛 VarTypeError。'''
        
        return self._coerce(value, var_name)

    def __call__(self, name: str, default: Any = MISSING):
        '''声明一个该类型的变量，等价于 NewVar(name, type=self)。'''
        # 延迟导入，避免 core.vartypes <-> core.steps.var 的循环依赖
        from promptkit.core.steps.logics.var import NewVar

        return NewVar(name, type=self, default=default)

    def __repr__(self) -> str:
        return self.name


def _to_number(value: Any, var_name: str) -> Any:
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


def _to_str(value: Any, var_name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return str(value)
    raise VarTypeError(
        f"变量 {var_name!r} 是 Str，无法把 {type(value).__name__} 转成字符串"
    )


# 只需要这两种类型
Number = VarType("Number", _to_number, 0)
Str = VarType("Str", _to_str, "")


def var_type_name(var_type: Any) -> str:
    '''给错误信息和 repr 用。'''
    return var_type.name if isinstance(var_type, VarType) else str(var_type)
