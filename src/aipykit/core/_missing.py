# core/_missing.py
'''共享的「没有传这个参数」哨兵。

必须全局只有一个实例：如果在两个模块里各写一个 `object()`，
`x is MISSING` 就会在某些调用路径上意外为 False。
'''


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()
