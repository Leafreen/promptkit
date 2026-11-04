# client_sys/multimodal.py
'''多模态的统一接口：任意模态、两种操作、一个请求对象。

和 `client_sys` 的对话客户端是同一套分层思路：

    multimodal.py           Modality / Media / MultimodalRequest / BaseMultimodalClient
                            + 厂商注册表（默认通用网关）
    generic_multimodal.py   通用网关实现：所有模态走同一个 JSON 接口

调用方只认「模态 + 操作 + 输入」这一个口径：

    client = make_multimodal_client()             # 默认通用网关
    client.generate("image", "一只猫在月球上")
    client.understand("video", "demo.mp4", prompt="描述这段视频")

模态名是普通字符串：文本、图片、视频、音频、富文本，甚至未来新增的
任意类型，都可以直接传；客户端只做统一的参数校验与格式标准化，
不再为每种模态写分支。厂商可以在 `capabilities` 里声明支持的组合，
但默认网关接受任意组合。
'''
from __future__ import annotations

import base64
import mimetypes
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from promptkit import config as config_module
from promptkit.client_sys.client import ClientError

__all__ = [
    "Modality",
    "Operation",
    "Media",
    "MultimodalRequest",
    "MultimodalResult",
    "MultimodalError",
    "MultimodalSpec",
    "BaseMultimodalClient",
    "DEFAULT_MULTIMODAL_VENDOR",
    "GENERIC",
    "DOUBAO",
    "MULTIMODAL_PROVIDERS",
    "normalize_modality",
    "normalize_operation",
    "normalize_format",
    "get_multimodal_spec",
    "register_multimodal_provider",
    "resolve_multimodal_api_key",
    "resolve_multimodal_target",
    "make_multimodal_client",
    "multimodal_vendor_names",
]


class MultimodalError(ClientError):
    '''多模态接口出错（协议 / 鉴权 / 轮询超时 / 模态不支持）。'''


class Modality:
    """已知模态的常量。

    故意**不用 Enum**：`Modality` 的值只是普通字符串，未来新增模态时
    直接传字符串即可，不会因为未知值而构造失败。
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    RICH_TEXT = "rich_text"

    @classmethod
    def known(cls) -> Tuple[str, ...]:
        return (cls.TEXT, cls.IMAGE, cls.VIDEO, cls.AUDIO, cls.RICH_TEXT)


class Operation:
    '''多模态的两类操作：生成 / 理解。'''

    GENERATE = "generate"
    UNDERSTAND = "understand"

    @classmethod
    def known(cls) -> Tuple[str, ...]:
        return (cls.GENERATE, cls.UNDERSTAND)


def normalize_modality(value: object) -> str:
    '''把模态名规整成小写字符串；空值拒绝，未知值放行（为扩展留口）。'''
    name = str(value or "").strip().lower()
    if not name:
        raise ValueError("modality 不能为空（text / image / video / audio …，或自定义模态名）")
    return name


def normalize_operation(value: object) -> str:
    '''把操作名规整成小写字符串。'''
    name = str(value or "").strip().lower()
    if not name:
        raise ValueError("operation 不能为空（generate / understand）")
    return name


def normalize_format(value: object) -> Optional[str]:
    '''把输出格式规整成小写字符串；空值表示交给网关决定。'''
    if value is None:
        return None
    name = str(value).strip().lower()
    return name or None


def _guess_mime(path: Optional[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if not path:
        return None
    guessed, _ = mimetypes.guess_type(path)
    return guessed


@dataclass(frozen=True)
class Media:
    '''一份媒体输入 / 输出。

    URL、本地路径、内存字节三选一即可（生成结果通常是前两者）。
    `path` 不要求文件已经存在 —— 它也可以表示「准备写到这个位置」。
    '''

    url: Optional[str] = None
    path: Optional[str] = None
    data: Optional[bytes] = None
    mime: Optional[str] = None

    def __post_init__(self) -> None:
        if not (self.url or self.path or self.data):
            raise ValueError("Media 需要 url / path / data 之一")

    # ---------------- 构造 ----------------
    @classmethod
    def from_path(
        cls, path: Union[str, os.PathLike[str]], mime: Optional[str] = None
    ) -> "Media":
        return cls(path=os.fspath(path), mime=mime)

    @classmethod
    def from_url(cls, url: str, mime: Optional[str] = None) -> "Media":
        return cls(url=url, mime=mime)

    @classmethod
    def from_bytes(cls, data: bytes, mime: Optional[str] = None) -> "Media":
        return cls(data=data, mime=mime)

    @classmethod
    def from_text(cls, text: str, mime: str = "text/plain") -> "Media":
        '''把一段文本 / HTML 富文本包装成媒体输入。'''
        return cls(data=text.encode("utf-8"), mime=mime)

    @classmethod
    def from_any(cls, value: object) -> "Media":
        '''把路径 / URL / bytes / FileRef / 已有 Media 统一成 Media。

        `FileRef` 那种「有 path 属性的对象」也能被接住，所以变量类型系统
        和这里的媒体对象不需要互相 import。
        '''
        if isinstance(value, Media):
            return value
        if isinstance(value, (bytes, bytearray)):
            return cls(data=bytes(value))
        if isinstance(value, os.PathLike):
            return cls(path=os.fspath(value))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("媒体路径 / URL 不能是空字符串")
            if text.startswith(("http://", "https://")):
                return cls(url=text)
            return cls(path=text)

        path = getattr(value, "path", None)
        if isinstance(path, str) and path:
            return cls(path=path)
        url = getattr(value, "url", None)
        if isinstance(url, str) and url:
            return cls(url=url)
        data = getattr(value, "data", None)
        if isinstance(data, (bytes, bytearray)):
            return cls(data=bytes(data))
        raise TypeError(
            f"无法把 {type(value).__name__} 当成媒体："
            "需要路径 / URL / bytes / Media / 带 path 属性的对象"
        )

    # ---------------- 基础操作 ----------------
    @property
    def content_type(self) -> Optional[str]:
        return _guess_mime(self.path, self.mime)

    def exists(self) -> bool:
        path = self.path
        if not path:
            return False
        return os.path.exists(path)

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path:
            with open(self.path, "rb") as handle:
                return handle.read()
        raise MultimodalError("这是一份远程 URL 媒体，没有本地字节可读")

    def to_data_url(self) -> str:
        '''给需要 base64 的接口用（本地文件 / 内存字节）。远程 URL 原样返回。'''
        if self.url and not self.path and self.data is None:
            return self.url
        payload = self.read_bytes()
        mime = self.content_type or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    def save(self, path: Union[str, os.PathLike[str]]) -> "Media":
        '''把字节落盘（远程 URL 会走一次 HTTP 下载）。'''
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))
        folder = os.path.dirname(target)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if self.data is not None:
            payload = self.data
        elif self.path:
            if os.path.abspath(self.path) == target:
                return self
            with open(self.path, "rb") as handle:
                payload = handle.read()
        elif self.url:
            with urllib.request.urlopen(self.url) as response:  # noqa: S310 - 显式 URL
                payload = response.read()
        else:
            raise MultimodalError("Media 没有可保存的内容")

        with open(target, "wb") as handle:
            handle.write(payload)
        return Media(path=target, mime=_guess_mime(target, self.mime))

    def __fspath__(self) -> str:
        if self.path:
            return self.path
        raise TypeError("远程 URL 媒体没有本地路径")

    def __str__(self) -> str:
        return self.path or self.url or f"<{len(self.data or b'')} bytes>"

    def __repr__(self) -> str:
        source = self.path or self.url
        if source:
            return f"Media({source!r})"
        return f"Media(bytes={len(self.data or b'')})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "path": self.path,
            "mime": self.content_type,
            "bytes": len(self.data or b""),
        }


@dataclass
class MultimodalRequest:
    '''所有模态共用的请求。'''

    operation: str
    modality: str
    prompt: Optional[str] = None
    inputs: Tuple[Media, ...] = ()
    model: Optional[str] = None
    output_path: Optional[str] = None
    output_modality: Optional[str] = None
    output_format: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        self.operation = normalize_operation(self.operation)
        self.modality = normalize_modality(self.modality)
        if self.output_modality is not None:
            self.output_modality = normalize_modality(self.output_modality)
        self.output_format = normalize_format(self.output_format)
        self.inputs = tuple(
            item if isinstance(item, Media) else Media.from_any(item)
            for item in self.inputs
        )
        self.params = dict(self.params)
        if self.prompt is not None and not isinstance(self.prompt, str):
            self.prompt = str(self.prompt)

    @property
    def supports_inputs(self) -> bool:
        return bool(self.inputs)

    @property
    def desired_output_modality(self) -> str:
        '''生成默认输出输入模态；理解默认输出文本。显式指定优先。'''
        if self.output_modality:
            return self.output_modality
        if self.operation == Operation.UNDERSTAND:
            return Modality.TEXT
        return self.modality


@dataclass
class MultimodalResult:
    '''统一结果：文本（理解）或媒体（生成）。'''

    operation: str
    modality: str
    model: Optional[str] = None
    text: Optional[str] = None
    media: Optional[Media] = None
    usage: Optional[Mapping[str, Any]] = None
    raw: Any = None
    task_id: Optional[str] = None

    @property
    def content(self) -> object:
        '''一步取值：有媒体给媒体，有文本给文本，否则原样透传网关响应。'''
        if self.media is not None:
            return self.media
        if self.text is not None:
            return self.text
        return self.raw


@dataclass(frozen=True)
class MultimodalSpec:
    '''一家多模态厂商的静态信息。'''

    name: str
    label: str
    kind: str
    env_vars: Tuple[str, ...] = ()
    default_base_url: Optional[str] = None
    #: 模态 -> 默认模型（text / image / video，或未来新增的模态）
    models: Mapping[str, str] = field(default_factory=dict)
    #: 支持的能力组合，例如 {("image", "generate"), ("video", "understand")}
    capabilities: FrozenSet[Tuple[str, str]] = frozenset()
    aliases: Tuple[str, ...] = ()
    note: str = ""

    def names(self) -> Tuple[str, ...]:
        return (self.name, *self.aliases)

    def model_for(self, modality: str) -> Optional[str]:
        return self.models.get(normalize_modality(modality))

    def supports(self, modality: str, operation: str) -> bool:
        return (
            normalize_modality(modality),
            normalize_operation(operation),
        ) in self.capabilities


class BaseMultimodalClient(ABC):
    '''多模态客户端接口。

    子类只需要实现 `run()`；`generate()` / `understand()` 是统一入口的糖。
    '''

    vendor: str = "base"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[MultimodalSpec] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.spec = spec
        self.timeout = timeout

    @property
    def capabilities(self) -> FrozenSet[Tuple[str, str]]:
        return frozenset() if self.spec is None else self.spec.capabilities

    def supports(self, modality: str, operation: str) -> bool:
        return (
            normalize_modality(modality),
            normalize_operation(operation),
        ) in self.capabilities

    def require(self, modality: str, operation: str) -> None:
        '''不支持的能力在发请求前就报错，别让它变成「静默返回空」。'''
        if not self.supports(modality, operation):
            raise MultimodalError(
                f"{self.vendor} 不支持 {modality}:{operation}；"
                f"可用能力：{', '.join(sorted(f'{m}:{o}' for m, o in self.capabilities))}"
            )

    @abstractmethod
    def run(self, request: MultimodalRequest) -> MultimodalResult:
        '''执行一次多模态请求。'''

    def generate(
        self,
        modality: str,
        prompt: str,
        inputs: Union[object, Sequence[object], None] = None,
        *,
        model: Optional[str] = None,
        save_to: Optional[str] = None,
        output_modality: Optional[str] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        **params: Any,
    ) -> MultimodalResult:
        '''生成：任意模态，输出类型 / 格式可由调用方指定。'''
        return self.run(
            MultimodalRequest(
                operation=Operation.GENERATE,
                modality=modality,
                prompt=prompt,
                inputs=_as_inputs(inputs),
                model=model,
                output_path=save_to,
                output_modality=output_modality,
                output_format=output_format,
                params=params,
                timeout=timeout,
            )
        )

    def understand(
        self,
        modality: str,
        source: Union[object, Sequence[object], None] = None,
        prompt: Optional[str] = None,
        *,
        model: Optional[str] = None,
        save_to: Optional[str] = None,
        output_modality: Optional[str] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        **params: Any,
    ) -> MultimodalResult:
        '''理解：任意模态输入，默认输出文本。'''
        return self.run(
            MultimodalRequest(
                operation=Operation.UNDERSTAND,
                modality=modality,
                prompt=prompt,
                inputs=_as_inputs(source),
                model=model,
                output_path=save_to,
                output_modality=output_modality,
                output_format=output_format,
                params=params,
                timeout=timeout,
            )
        )

    def close(self) -> None:
        '''释放底层连接。默认什么也不做。'''

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} vendor={self.vendor!r} "
            f"base_url={self.base_url!r}>"
        )


def _as_inputs(value: Union[object, Sequence[object], None]) -> Tuple[Media, ...]:
    '''单个输入 / 输入列表 / None -> Media 元组。'''
    if value is None:
        return ()
    if isinstance(value, Media):
        return (value,)
    if isinstance(value, (str, bytes, os.PathLike)) or hasattr(value, "path"):
        return (Media.from_any(value),)
    if isinstance(value, Iterable):
        return tuple(Media.from_any(item) for item in value)
    return (Media.from_any(value),)


# ============================ 厂商注册表 ============================
#: 默认多模态实现：通用网关（所有模态一个端点）
DEFAULT_MULTIMODAL_VENDOR = "generic"

GENERIC = MultimodalSpec(
    name="generic",
    label="通用多模态网关",
    kind="generic",
    env_vars=("PROMPTKIT_MULTIMODAL_API_KEY", "MULTIMODAL_API_KEY"),
    default_base_url=config_module.MULTIMODAL_BASE_URL,
    models={},
    capabilities=frozenset(
        (modality, operation)
        for modality in Modality.known()
        for operation in Operation.known()
    ),
    aliases=("gateway", "universal", "doubao", "seedance", "ark", "volcengine"),
    note="所有模态统一 POST 到同一个 JSON 端点；模型与输出格式由请求参数指定",
)

#: 兼容旧名字：现在指的就是通用网关
DOUBAO = GENERIC

#: 内置多模态厂商（按展示顺序）
MULTIMODAL_PROVIDERS: Dict[str, MultimodalSpec] = {GENERIC.name: GENERIC}

_ALIASES: Dict[str, MultimodalSpec] = {}
for _spec in MULTIMODAL_PROVIDERS.values():
    for _name in _spec.names():
        _ALIASES[_name.lower()] = _spec

_FACTORIES: Dict[str, Callable[..., BaseMultimodalClient]] = {}

#: 内置实现认的 `kind`
_KINDS: Tuple[str, ...] = ("generic",)


def multimodal_vendor_names() -> list[str]:
    return list(MULTIMODAL_PROVIDERS)


def get_multimodal_spec(vendor: Union[str, MultimodalSpec, None]) -> MultimodalSpec:
    if isinstance(vendor, MultimodalSpec):
        return vendor
    key = str(vendor or DEFAULT_MULTIMODAL_VENDOR).strip().lower().replace("_", "-")
    spec = _ALIASES.get(key)
    if spec is None:
        raise ValueError(
            f"不认识的多模态厂商 {vendor!r}；可用：{', '.join(multimodal_vendor_names())}"
        )
    return spec


def register_multimodal_provider(
    spec: MultimodalSpec,
    factory: Optional[Callable[..., BaseMultimodalClient]] = None,
) -> MultimodalSpec:
    '''注册一家多模态厂商，返回规格（方便链式注册）。'''
    if spec.kind not in _KINDS and factory is None:
        raise ValueError(
            f"厂商 {spec.name!r} 的 kind={spec.kind!r} 没有内置实现；"
            f"可用：{', '.join(_KINDS)}，或者自己传一个 factory="
        )
    MULTIMODAL_PROVIDERS[spec.name] = spec
    for name in spec.names():
        _ALIASES[name.lower()] = spec
    if factory is not None:
        _FACTORIES[spec.name] = factory
    return spec


def resolve_multimodal_api_key(
    vendor: Union[str, MultimodalSpec], api_key: Optional[str] = None
) -> str:
    '''按 `参数 -> 厂商环境变量 -> PROMPTKIT_API_KEY -> .env` 找密钥。'''
    spec = get_multimodal_spec(vendor)
    config_module.load_env_file()
    if api_key:
        return api_key
    for name in (*spec.env_vars, "PROMPTKIT_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    want = " 或 ".join((*spec.env_vars, "PROMPTKIT_API_KEY"))
    raise ValueError(
        f"没有找到 {spec.label} 的多模态 API Key：请设置环境变量 {want}，"
        f"或在 {config_module.ENV_FILE} 里写一行 {spec.env_vars[0]}=..."
    )


def resolve_multimodal_target(
    vendor: Optional[Union[str, MultimodalSpec]] = None,
    *,
    base_url: Optional[str] = None,
) -> Tuple[MultimodalSpec, Optional[str]]:
    spec = get_multimodal_spec(vendor or DEFAULT_MULTIMODAL_VENDOR)
    return spec, base_url or spec.default_base_url


def make_multimodal_client(
    vendor: Optional[Union[str, MultimodalSpec]] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> BaseMultimodalClient:
    '''按厂商造一个多模态客户端，默认通用网关。'''
    spec, target = resolve_multimodal_target(vendor, base_url=base_url)
    key = resolve_multimodal_api_key(spec, api_key)

    factory = _FACTORIES.get(spec.name)
    if factory is None:
        if spec.kind == "generic":
            from promptkit.client_sys.generic_multimodal import GenericMultimodalClient

            factory = GenericMultimodalClient
        else:
            raise MultimodalError(
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
