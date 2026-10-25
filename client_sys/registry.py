# client_sys/registry.py
'''厂商注册表：名字 -> `ClientSpec`，外加「用哪家厂商」「密钥从哪来」「造哪个客户端」。

三件事都在这里，因为它们互相依赖：

- `get_spec("claude")`   —— 名字（含别称）-> 厂商信息
- `detect_vendor(...)`   —— 什么都没说时，按 base_url / 模型名 / 环境变量猜
- `resolve_target(...)`  —— 猜完再把 model / base_url 补齐并校验
- `make_client(...)`     —— 按 `kind` 造具体客户端，顺手解决密钥

想加一家新厂商（比如公司内网的网关），不用改任何已有代码：

    from promptkit.client_sys import ClientSpec, register_vendor

    register_vendor(
        ClientSpec(
            name="my-gateway",
            label="公司网关",
            kind="openai",                       # 它说 Chat Completions 协议
            env_vars=("MY_GATEWAY_KEY",),
            default_model="gpt-4o-mini",
            default_base_url="https://llm.corp/v1",
        )
    )
    Agent(vendor="my-gateway")
'''
from __future__ import annotations

import copy
import os
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from promptkit import config as config_module
from promptkit.client_sys.client import BaseClient, ClientError, ClientSpec

__all__ = [
    "PROVIDERS",
    "DEFAULT_VENDOR",
    "FALLBACK_ENV_VAR",
    "get_spec",
    "vendor_names",
    "register_vendor",
    "detect_vendor",
    "resolve_api_key",
    "api_key_source",
    "resolve_target",
    "make_client",
    "with_params",
]

#: 什么都不指定时用哪家（保持 promptkit 原来的 DeepSeek 默认）
DEFAULT_VENDOR = "deepseek"

#: 谁都认的「通用密钥」：不想为每家复制一遍密钥时用它
FALLBACK_ENV_VAR = "PROMPTKIT_API_KEY"

#: 依次尝试的厂商（`detect_vendor` 扫环境变量时按这个顺序）
_SNIFF_ORDER: Tuple[str, ...] = ("deepseek", "anthropic", "openai")

#: 模型名前缀 -> 厂商（越长的前缀越先匹配）
_MODEL_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("deepseek/", "deepseek"),
    ("claude", "anthropic"),
    ("deepseek", "deepseek"),
    ("chatgpt", "openai"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("davinci", "openai"),
    ("text-", "openai"),
)

#: base_url 里出现这些就够了
_HOST_HINTS: Tuple[Tuple[str, str], ...] = (
    ("anthropic", "anthropic"),
    ("deepseek", "deepseek"),
    ("openai.com", "openai"),
)


DEEPSEEK = ClientSpec(
    name="deepseek",
    label="DeepSeek",
    kind="openai",
    env_vars=("DEEPSEEK_API_KEY",),
    default_model=config_module.DEFAULT_MODEL,
    default_base_url=config_module.DEFAULT_BASE_URL,
    # 原来的 Agent 写死的东西，现在归厂商规格管：
    # 换厂商时不会再把 DeepSeek 的思考参数发给人家
    default_params={
        "reasoning_effort": config_module.DEFAULT_REASONING_EFFORT,
        "extra_body": dict(config_module.DEFAULT_EXTRA_BODY),
    },
    aliases=("ds",),
)

OPENAI = ClientSpec(
    name="openai",
    label="OpenAI",
    kind="openai",
    env_vars=("OPENAI_API_KEY",),
    default_model="gpt-4o-mini",
    default_base_url="https://api.openai.com/v1",
    aliases=("gpt",),
)

ANTHROPIC = ClientSpec(
    name="anthropic",
    label="Anthropic Claude",
    kind="anthropic",
    env_vars=("ANTHROPIC_API_KEY",),
    default_model="claude-sonnet-4-5",
    default_base_url="https://api.anthropic.com",
    # Messages API 硬性要求 max_tokens
    default_params={"max_tokens": 4096},
    aliases=("claude",),
)

#: 只说「OpenAI 兼容」、不替用户猜端点的那种（Ollama / vLLM / 中转站 / 通义…）
OPENAI_COMPATIBLE = ClientSpec(
    name="openai-compatible",
    label="OpenAI 兼容端点",
    kind="openai",
    env_vars=("OPENAI_API_KEY",),
    default_model=None,
    default_base_url=None,
    aliases=("openai_compatible", "compatible", "custom", "generic"),
    note="需要显式传 base_url 和 model",
)

#: 内置厂商（按展示顺序）
PROVIDERS: Dict[str, ClientSpec] = {
    spec.name: spec
    for spec in (DEEPSEEK, OPENAI, ANTHROPIC, OPENAI_COMPATIBLE)
}

#: 名字 / 别称 -> 规格
_ALIASES: Dict[str, ClientSpec] = {}
for _spec in PROVIDERS.values():
    for _name in _spec.names():
        _ALIASES[_name.lower()] = _spec

#: 内置实现认的 `kind`（延迟 import，避免 import promptkit 时就拖进 openai）
_KINDS: Tuple[str, ...] = ("openai", "anthropic")

#: 自定义厂商的工厂（`register_vendor(spec, factory=...)` 塞进来）
_FACTORIES: Dict[str, Callable[..., BaseClient]] = {}


# ============================ 厂商 ============================
def vendor_names() -> List[str]:
    '''已注册的厂商名（不含别称；`register_vendor` 加的也算）。'''
    return list(PROVIDERS)


def get_spec(vendor: Union[str, ClientSpec]) -> ClientSpec:
    '''名字 / 别称 / 规格对象 -> 规格对象。'''
    if isinstance(vendor, ClientSpec):
        return vendor
    key = str(vendor or "").strip().lower().replace("_", "-")
    spec = _ALIASES.get(key)
    if spec is None:
        raise ValueError(
            f"不认识的厂商 {vendor!r}；可用：{', '.join(vendor_names())}"
            "（别称也算，比如 'claude' 就是 anthropic）"
        )
    return spec


def register_vendor(
    spec: ClientSpec, factory: Optional[Callable[..., BaseClient]] = None
) -> ClientSpec:
    '''注册一家新厂商，返回规格（方便链式注册）。

    `factory` 省略时按 `spec.kind` 选内置实现；要接一种完全不同的协议，
    就自己写个 `BaseClient` 子类把 `factory` 传进来。
    '''
    if spec.kind not in _KINDS and factory is None:
        raise ValueError(
            f"厂商 {spec.name!r} 的 kind={spec.kind!r} 没有内置实现；"
            f"可用：{', '.join(_KINDS)}，或者自己传一个 factory="
        )
    PROVIDERS[spec.name] = spec
    for name in spec.names():
        _ALIASES[name.lower()] = spec
    if factory is not None:
        _FACTORIES[spec.name] = factory
    return spec


def _host_of(base_url: str) -> str:
    parsed = urllib.parse.urlparse(str(base_url))
    return (parsed.hostname or parsed.path or "").lower()


def detect_vendor(
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    '''什么都没显式说时，猜一个厂商。

    优先级：`base_url` 的主机名 > 模型名前缀 > 环境变量里现成的密钥 >
    `DEFAULT_VENDOR`（DeepSeek）。

    `api_key` 是「显式传进来的密钥」——传了就说明用户心里已经有个默认厂商了，
    不再去环境变量里猜（否则 `Agent(api_key="sk-test")` 的行为会随开发机上的
    环境变量漂移，测试也会飘）。
    '''
    if base_url:
        host = _host_of(base_url)
        for needle, name in _HOST_HINTS:
            if needle in host:
                return name
        return "openai-compatible"

    if model:
        lowered = str(model).strip().lower()
        for prefix, name in _MODEL_PREFIXES:
            if lowered.startswith(prefix):
                return name

    if api_key is None:
        # `.env` 也是配置来源，不读它的话「只写 .env」的用户永远猜不到厂商
        config_module.load_env_file()
        for name in _SNIFF_ORDER:
            spec = PROVIDERS[name]
            if any(os.environ.get(var) for var in spec.env_vars):
                return name

    return DEFAULT_VENDOR


# ============================ 密钥 ============================
def resolve_api_key(
    vendor: Union[str, ClientSpec], api_key: Optional[str] = None
) -> str:
    '''按 `参数 -> 厂商环境变量 -> PROMPTKIT_API_KEY -> .env` 的顺序找密钥。'''
    spec = get_spec(vendor)
    # 和原来的行为一致：.env 里的键会被读进环境变量（已有的环境变量优先）
    config_module.load_env_file()

    if api_key:
        return api_key

    for name in (*spec.env_vars, FALLBACK_ENV_VAR):
        value = os.environ.get(name)
        if value:
            return value

    want = " 或 ".join((*spec.env_vars, FALLBACK_ENV_VAR))
    hint = (
        "。如果这是本地端点（Ollama / vLLM），它通常不校验密钥，"
        "随便给一个就行：api_key='ollama'"
        if spec.name == "openai-compatible"
        else ""
    )
    raise ValueError(
        f"没有找到 {spec.label} 的 API Key：请设置环境变量 {want}，"
        f"或在 {config_module.ENV_FILE} 里写一行 {spec.env_vars[0]}=sk-xxxx{hint}"
    )


def api_key_source(
    vendor: Union[str, ClientSpec], api_key: Optional[str] = None
) -> str:
    '''密钥是从哪来的（排查问题时有用）。'''
    spec = get_spec(vendor)
    if api_key:
        return "参数"
    for name in (*spec.env_vars, FALLBACK_ENV_VAR):
        if os.environ.get(name):
            return f"环境变量 {name}"
    return f"{config_module.ENV_FILE}"


# ============================ 组装 ============================
def resolve_target(
    *,
    vendor: Optional[Union[str, ClientSpec]] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[ClientSpec, str, str]:
    '''定下 (厂商, 模型, 端点) 三件套，缺的用厂商默认值补，补不上就报错。'''
    spec = get_spec(vendor) if vendor is not None else get_spec(
        detect_vendor(model=model, base_url=base_url, api_key=api_key)
    )

    resolved_model = model or spec.default_model
    if not resolved_model:
        raise ValueError(
            f"{spec.label} 没有默认模型，请显式传 model=；"
            "比如 Agent(vendor='openai-compatible', base_url='http://localhost:11434/v1', "
            "model='qwen2.5')"
        )

    resolved_base = base_url or spec.default_base_url
    if not resolved_base:
        raise ValueError(
            f"{spec.label} 没有默认端点，请显式传 base_url=；"
            "比如 Agent(base_url='http://localhost:11434/v1', model='qwen2.5')"
        )

    return spec, str(resolved_model), str(resolved_base)


def make_client(
    vendor: Union[str, ClientSpec],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> BaseClient:
    '''按厂商造一个客户端（顺手把密钥解决了）。'''
    spec = get_spec(vendor)
    target = base_url or spec.default_base_url
    if not target:
        raise ClientError(
            f"{spec.label} 没有默认端点，请显式传 base_url="
            f"（{spec.note or '例如 http://localhost:11434/v1'}）"
        )
    key = resolve_api_key(spec, api_key)

    factory = _FACTORIES.get(spec.name)
    if factory is None:
        if spec.kind == "anthropic":
            from promptkit.client_sys.anthropic_client import AnthropicClient

            factory = AnthropicClient
        elif spec.kind == "openai":
            from promptkit.client_sys.openai_client import OpenAIClient

            factory = OpenAIClient
        else:
            raise ClientError(
                f"厂商 {spec.name!r} 的 kind={spec.kind!r} 没有对应实现；"
                f"可用：{', '.join(_KINDS)}，或者自己传一个 factory="
            )

    return factory(
        api_key=key,
        base_url=target,
        spec=spec,
        timeout=timeout,
        **kwargs,
    )


def with_params(spec: ClientSpec, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    '''厂商默认参数 + 用户参数（浅合并，用户优先）。

    浅合并是有意的：`extra_body` 这种嵌套结构整块替换比猜「用户想改哪个
    子字段」好懂。开头 `deepcopy` 是为了让每个实例拿到独立的副本 ——
    不然改自己的 `agent.params["extra_body"]` 会把厂商规格也改了。
    '''
    merged: Dict[str, Any] = copy.deepcopy(dict(spec.default_params))
    if params:
        merged.update(params)
    return merged
