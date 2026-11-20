# client_sys/reasoning.py
'''推理等级：一份写法，各厂商各自翻译。

控制「想多深」的参数，每家名字和形状都不一样：

    ==========  ==========================================================
    厂商        实际发出去的参数
    ==========  ==========================================================
    DeepSeek    ``reasoning_effort`` + ``extra_body.thinking``
    OpenAI      ``reasoning_effort``
    Anthropic   ``thinking.budget_tokens``（只支持不带工具的单次调用）
    豆包/Ark    ``extra_body.thinking``（只有开关，没有等级）
    ==========  ==========================================================

所以调用方只写 ``Agent(reasoning="low")``（或
``Agent(params={"reasoning": "low"})``），由 `ClientSpec.reasoning` 声明、
这里统一翻成厂商参数。厂商原生参数仍然可以照常从 ``params=`` 传：
显式写了统一等级时以等级为准，没写就原样透传。
'''
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Mapping, Optional

if TYPE_CHECKING:  # 只用于类型标注，避免和 client.py 互相 import
    from aipykit.client_sys.client import ClientSpec

__all__ = [
    "OFF_ALIASES",
    "ReasoningSpec",
    "reasoning_params",
    "expand_reasoning",
]

#: 表示「关掉思考」的等级写法
OFF_ALIASES: FrozenSet[str] = frozenset({"off", "none", "disabled", "false", "no"})


@dataclass(frozen=True)
class ReasoningSpec:
    '''一家厂商的推理等级 -> 请求参数片段。

    ``levels`` / ``off`` 里放的是**最终要发出去的形状**，所以嵌套参数
    （``extra_body``、``thinking``）直接写完整，不做二次猜测。
    '''

    #: 等级名（小写）-> 参数片段。按从低到高声明，最后一个视为最高档
    levels: Mapping[str, Mapping[str, Any]]
    #: 关闭思考时发的参数片段
    off: Mapping[str, Any] = field(default_factory=dict)
    #: 没显式指定等级时用哪个；None 表示默认一个都不加
    default: Optional[str] = None
    #: 给人看的补充说明（报错 / 文档用）
    note: str = ""

    def keys(self) -> FrozenSet[str]:
        '''这个规格会碰到的顶层参数名。

        展开显式等级时用它清掉旧值，避免「新等级」和厂商默认的
        「旧等级」一起发出去。
        '''
        names = set(self.off)
        for fragment in self.levels.values():
            names.update(fragment)
        return frozenset(names)


def reasoning_params(spec: "ClientSpec", level: Any) -> Dict[str, Any]:
    '''把统一等级翻成某家厂商的请求参数。

    ``level`` 大小写不敏感；``off`` / ``none`` / ``disabled`` / ``false`` / ``no``
    都表示关闭。厂商没声明推理参数、或者等级不认识时抛 ``ValueError``
    —— 等级写错了应该在构造 Agent 时就报，而不是等第一次请求。
    '''
    config = spec.reasoning
    if config is None:
        raise ValueError(
            f"{spec.label} 没有声明统一的推理等级；"
            "请用 params= 直接传这家厂商的原生参数"
        )

    key = str(level).strip().lower()
    if key in OFF_ALIASES:
        fragment = config.off
    elif key in config.levels:
        fragment = config.levels[key]
    else:
        choices = ", ".join([*config.levels, "off"])
        raise ValueError(f"{spec.label} 不支持推理等级 {level!r}；可选：{choices}")
    return copy.deepcopy(dict(fragment))


def expand_reasoning(spec: "ClientSpec", params: Mapping[str, Any]) -> Dict[str, Any]:
    '''把 ``reasoning``（旧写法 ``reasoning_effort``）展开成厂商参数。

    - 显式写了等级：先清掉同一规格会碰到的旧参数，再放翻译结果，
      保证「低」不会和厂商默认的「高」一起发出去
    - 没写等级：用 ``spec.reasoning.default`` 补齐，但不覆盖 params 里
      用户已经写好的厂商原生参数
    - 厂商没声明推理规格：``reasoning`` 报错（这个键厂商不认识），
      ``reasoning_effort`` 原样透传（保持旧行为）
    '''
    merged = dict(params)
    has_reasoning = "reasoning" in merged
    has_legacy = "reasoning_effort" in merged
    level = merged.pop("reasoning", None)
    legacy = merged.pop("reasoning_effort", None)

    config = spec.reasoning
    if config is None:
        if has_reasoning:
            raise ValueError(
                f"{spec.label} 没有声明统一的推理等级；"
                "请用 params= 直接传这家厂商的原生参数"
            )
        if has_legacy:
            merged["reasoning_effort"] = legacy
        return merged

    if level is None:
        level = legacy
    if level is None:
        if config.default is None:
            return merged
        return {**reasoning_params(spec, config.default), **merged}

    fragment = reasoning_params(spec, level)
    if has_reasoning or has_legacy:
        reserved = config.keys()
        merged = {key: value for key, value in merged.items() if key not in reserved}
    return {**fragment, **merged}
