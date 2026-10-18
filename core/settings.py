'''DSL 的全局默认值。

这些数字原先散在每个工厂函数的签名里，想统一调一下要翻好几个文件，
而且互相不一致（`While` 默认 100，`ForEach` 默认 10000，纯属历史巧合）。

    from promptkit.core import settings

    settings.max_retries = 3            # 全局改，之后构造的都按这个来
    settings.encoding = "utf-8"

    with using_settings(max_iterations=5):   # 临时改，退出自动还原
        wf.run()

**在「用的时候」读，不是构造的时候读** —— 所以 `settings.max_iterations = 5`
对**已经构造好**的循环也生效，`using_settings` 包住 `wf.run()` 也有意义。

单次调用永远可以显式覆盖：`While(cond, max_iterations=3)` 优先于全局值。

⚠️ 这是**进程级可变状态**（和 `set_default_cautions` / `Agent.default()` 一样）。
测试里用 `using_settings(...)` 或 `reset_settings()` 收尾。
'''
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterator, List

__all__ = [
    "Settings",
    "settings",
    "using_settings",
    "reset_settings",
    "settings_names",
]


@dataclass
class Settings:
    '''所有可调的全局默认值。'''

    # ---- 循环保护 ----
    #: While / DoWhile 的迭代上限。条件循环更容易真的失控（比如模型一直没说"结束"），
    #: 所以默认给小一点，早点报错比跑一万轮便宜。
    max_iterations: int = 100
    #: For / ForEach 的迭代上限。遍历通常由数据量决定，给宽一点。
    max_for_iterations: int = 10000

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

#: 当前生效的设置。直接改属性就是全局修改。
settings = Settings()

#: 必须是正整数的字段
_POSITIVE_INTS = (
    "max_iterations",
    "max_for_iterations",
    "max_steps",
    "max_jumps",
    "max_tool_rounds",
)

_FIELD_NAMES = {field.name for field in fields(Settings)}


def settings_names() -> List[str]:
    '''所有可调的设置名。'''
    return sorted(_FIELD_NAMES)


def _validate(overrides: Dict[str, Any]) -> None:
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
    '''还原成出厂默认。'''
    for field in fields(Settings):
        setattr(settings, field.name, getattr(_FACTORY, field.name))


@contextmanager
def _apply(overrides: Dict[str, Any]) -> Iterator[Settings]:
    saved = {key: getattr(settings, key) for key in overrides}
    for key, value in overrides.items():
        setattr(settings, key, value)
    try:
        yield settings
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)


def using_settings(**overrides: Any) -> Any:
    '''临时改一批设置，退出时还原（测试 / 调试用）。

        with using_settings(max_iterations=3, auto_schema=True):
            wf.run()

    参数在**调用时**就校验，写错了立刻报错，而不是等进 with 才炸。
    '''
    _validate(overrides)
    return _apply(overrides)
