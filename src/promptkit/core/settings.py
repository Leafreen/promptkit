# core/settings.py
'''DSL 的默认值：`Settings` 是不可变快照，全局可变状态被收窄成「显式覆盖」。

目标形态（阶段 1 起）：

    wf = Workflow(..., max_iterations=1000)   # 构造时覆盖
    wf.run(max_iterations=50)                 # 运行时再覆盖一次
    with using_settings(max_iterations=5):    # 临时改全局默认（测试 / 调试）
        wf.run()

老写法仍然可用但已弃用：

    settings.max_iterations = 5   # DeprecationWarning，等价于改全局默认

`Settings` 本身是 `@dataclass(frozen=True)`：它是一个**快照**，不是可变全局。
真正生效的全局默认由模块级 `settings` 代理对象读写，覆盖值存在模块内，
不进 `Settings` 实例 —— 这样 `Settings()` 永远拿到出厂配置。

**在「用的时候」读，不是构造的时候读**：循环、函数递归、工作流都在执行时
调用 `current(context, ...)`，所以 `Workflow(max_iterations=...)` /
`wf.run(max_iterations=...)` 对已经建好的步骤树也生效。

⚠️ `settings.xxx = ...` 是进程级可变状态；测试里请用 `using_settings(...)`
或 `reset_settings()` 收尾。
'''
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional

if TYPE_CHECKING:  # 只用于类型标注，避免 settings -> context 的循环依赖
    from promptkit.core.context import Context

__all__ = [
    "Settings",
    "settings",
    "using_settings",
    "reset_settings",
    "settings_names",
    "current",
    "scoped",
    "validate_settings",
    "SETTINGS_OVERRIDES_KEY",
    "SETTINGS_RUN_KEY",
]

#: Context 里存放「本次运行覆盖值」的键
SETTINGS_OVERRIDES_KEY = "_settings_overrides"
#: Context 里存放 `wf.run(...)` 级覆盖的键（优先级高于构造覆盖）
SETTINGS_RUN_KEY = "_settings_run_overrides"


@dataclass(frozen=True)
class Settings:
    '''所有可调的默认值（不可变快照）。'''

    # ---- 循环保护 ----
    #: While / DoWhile 的迭代上限。条件循环更容易真的失控（比如模型一直没说"结束"），
    #: 所以默认给小一点，早点报错比跑一万轮便宜。
    max_iterations: int = 100
    #: For / ForEach 的迭代上限。遍历通常由数据量决定，给宽一点。
    max_for_iterations: int = 10000
    #: 函数递归上限。递归最容易缺少终止条件，早点报错比撞 Python 栈上限清晰。
    max_recursion_depth: int = 100

    # ---- 工作流保护 ----
    #: 一个 Workflow 最多几个步骤（构造时检查）
    max_steps: int = 50
    #: Goto 连续跳转上限
    max_jumps: int = 10000

    # ---- 模型调用 ----
    #: LLMStep 结构化输出解析失败后的重试次数
    max_retries: int = 2
    #: Agent 工具调用循环的上限
    max_tool_rounds: int = 8
    #: LLMStep 是否默认把 output_model 的 JSON Schema 注入提示词
    auto_schema: bool = False

    # ---- 文件读写 ----
    #: Save / ReadFile / SaveHistory 用的编码
    encoding: str = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"Settings({body})"


#: 出厂默认（reset_settings 会还原到这里）
_FACTORY = Settings()

#: 当前生效的全局覆盖值（只有显式改过的键才在这里）
_OVERRIDES: Dict[str, Any] = {}

#: 必须是正整数的字段
_POSITIVE_INTS = (
    "max_iterations",
    "max_for_iterations",
    "max_recursion_depth",
    "max_steps",
    "max_jumps",
    "max_tool_rounds",
)

_FIELD_NAMES = {field.name for field in fields(Settings)}


class _SettingsProxy:
    '''模块级 `settings`：读属性拿到「覆盖值 / 出厂值」，写属性发弃用警告。'''

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name in _FIELD_NAMES:
            return _OVERRIDES.get(name, getattr(_FACTORY, name))
        raise AttributeError(f"Settings 没有 {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        validate_settings({name: value})
        warnings.warn(
            f"settings.{name} = ... 已弃用（阶段 1）："
            "请改在 Workflow(...) / wf.run(...) 里显式覆盖，"
            "或临时用 using_settings(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        _OVERRIDES[name] = value

    def to_dict(self) -> Dict[str, Any]:
        '''合并后的当前值（覆盖值优先）。'''
        return {name: getattr(self, name) for name in sorted(_FIELD_NAMES)}

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"settings({body})"


#: 当前生效的设置视图。读属性 = 当前值；直接赋值 = 已弃用的全局修改。
settings = _SettingsProxy()


def settings_names() -> List[str]:
    '''所有可调的设置名。'''
    return sorted(_FIELD_NAMES)


def validate_settings(overrides: Mapping[str, Any]) -> None:
    '''校验一批覆盖值；非法键 / 非法值立刻报错。'''
    for key, value in overrides.items():
        if key not in _FIELD_NAMES:
            raise TypeError(
                f"没有这个设置项 {key!r}；可用：{', '.join(settings_names())}"
            )
        if key in _POSITIVE_INTS and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} 必须是正整数，得到 {value!r}")
        if key == "max_retries" and (not isinstance(value, int) or value < 0):
            raise ValueError(f"max_retries 必须是非负整数，得到 {value!r}")
        if key == "encoding" and not isinstance(value, str):
            raise TypeError(f"encoding 必须是字符串，得到 {value!r}")


def reset_settings() -> None:
    '''还原成出厂默认（清掉所有全局覆盖）。'''
    _OVERRIDES.clear()


@contextmanager
def _apply(overrides: Mapping[str, Any]) -> Iterator[_SettingsProxy]:
    saved = {key: _OVERRIDES.get(key, _MISSING) for key in overrides}
    _OVERRIDES.update(overrides)
    try:
        yield settings
    finally:
        for key, value in saved.items():
            if value is _MISSING:
                _OVERRIDES.pop(key, None)
            else:
                _OVERRIDES[key] = value


_MISSING = object()


def using_settings(**overrides: Any) -> Any:
    '''临时改一批设置，退出时还原（测试 / 调试用）。

        with using_settings(max_iterations=3, auto_schema=True):
            wf.run()

    参数在**调用时**就校验，写错了立刻报错，而不是等进 with 才炸。
    '''
    validate_settings(overrides)
    return _apply(overrides)


def current(context: "object", name: str) -> Any:
    '''取某个设置在**当前 Context 下**的实际生效值。

    优先级：`Workflow(...)` 构造覆盖 < `wf.run(...)` 运行覆盖 < 全局默认。
    没有 Context（或 Context 没这个键）时退回全局。
    '''
    data = getattr(context, "data", None)
    if isinstance(data, dict):
        run_overrides = data.get(SETTINGS_RUN_KEY)
        if isinstance(run_overrides, dict) and name in run_overrides:
            return run_overrides[name]
        overrides = data.get(SETTINGS_OVERRIDES_KEY)
        if isinstance(overrides, dict) and name in overrides:
            return overrides[name]
    return getattr(settings, name)


@contextmanager
def scoped(
    context: "Context", overrides: Optional[Mapping[str, Any]]
) -> Iterator[None]:
    '''在某个 Context 上临时合并一批覆盖值（嵌套 Workflow 安全）。

    没有覆盖时几乎零开销；退出时恢复原来那份，不污染外层。
    '''
    if not overrides:
        yield
        return
    data = getattr(context, "data", None)
    if not isinstance(data, dict):
        yield
        return
    saved = data.get(SETTINGS_OVERRIDES_KEY)
    merged: Dict[str, Any] = {}
    if isinstance(saved, dict):
        merged.update(saved)
    merged.update(overrides)
    data[SETTINGS_OVERRIDES_KEY] = merged
    try:
        yield
    finally:
        if saved is None:
            data.pop(SETTINGS_OVERRIDES_KEY, None)
        else:
            data[SETTINGS_OVERRIDES_KEY] = saved


# 本文件核心职责：不可变 Settings 快照 + 兼容的全局代理 + Context 级覆盖解析。
