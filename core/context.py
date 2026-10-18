# core/context.py
'''所有步骤共享的状态容器。'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # 只用于类型标注：Context 本身是叶子模块，不依赖 steps
    from promptkit.core.steps.logics.expr_sys.vartypes import VarType


@dataclass
class Context:
    '''所有步骤共享的状态容器。

    `types` 记录每个变量声明的类型（Number / Str）。因为赋值统一走
    `__setitem__`，所以类型转换只需要在这一个地方做，任何步骤写进来
    的值都会被转换成声明的类型。
    '''

    data: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)  # 跟踪步骤的执行过程
    types: Dict[str, VarType] = field(default_factory=dict)  # 变量名 -> 类型

    # ---------------- 类型转换 ----------------
    def _coerce(self, key: str, value: Any) -> Any:
        var_type = self.types.get(key)
        return value if var_type is None else var_type.coerce(value, key)

    def declare(self, key: str, var_type: VarType, default: Any = None) -> Any:
        '''声明变量类型。

        变量还不存在时用 default 初始化；已经存在时（例如通过
        `Workflow.run(n="12")` 传进来的）也会按类型转换一次。
        '''
        self.types[key] = var_type
        if key in self.data:
            self.data[key] = var_type.coerce(self.data[key], key)
            return self.data[key]
        return self.setdefault(key, default)

    def type_of(self, key: str):
        '''返回变量声明的类型，没声明则返回 None。'''
        return self.types.get(key)

    # ---------------- 映射接口 ----------------
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = self._coerce(key, value)

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setdefault(self, key, default=None):
        if key in self.data:
            return self.data[key]
        self.data[key] = self._coerce(key, default)
        return self.data[key]

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def __repr__(self) -> str:
        keys = ", ".join(map(str, self.data))
        return f"Context(keys=[{keys}], trace={self.trace!r})"
