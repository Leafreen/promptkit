# client_sys/multimodal.py
'''多模态的统一接口：三类模态、两种操作、一个请求对象。

和 `client_sys` 的对话客户端是同一套分层思路：

    multimodal.py           Modality / Media / MultimodalRequest / BaseMultimodalClient
                            + 厂商注册表（默认豆包 / Seedance）
    doubao_multimodal.py   豆包方舟（Ark）实现：文本、图片、视频

调用方只认「模态 + 操作 + 输入」这一个口径：

    client = make_multimodal_client()             # 默认豆包
    client.generate("image", "一只猫在月球上")
    client.understand("video", "demo.mp4", prompt="描述这段视频")

想加一种新模态（音频 / 3D……）不需要改这里：模态名是普通字符串，
厂商在 `capabilities` 里声明自己支持哪些 `(modality, operation)` 组合即可。
'''
from __future__ import annotations

import base64
import mimetypes
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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
    "DOUBAO",
    "MULTIMODAL_PROVIDERS",
    "normalize_modality",
    "normalize_operation",
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
    '''已知模态的常量。

    故意**不用 Enum**：`Modality` 的值只是普通字符串，未来新增 "audio"
    这类模态时，providers 只要在自己的 `capabilities` 里声明即可，
    不需要改这张表，也不会因为传了未知字符串就构造失败。
    '''

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"

    @classmethod
    def known(cls) -> Tuple[str, ...]:
        return (cls.TEXT, cls.IMAGE, cls.VIDEO)


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
        raise ValueError("modality 不能为空（text / image / video，或自定义模态名）")
    return name


def normalize_operation(value: object) -> str:
    '''把操作名规整成小写字符串。'''
    name = str(value or "").strip().lower()
    if not name:
        raise ValueError("operation 不能为空（generate / understand）")
    return name


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
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        self.operation = normalize_operation(self.operation)
        self.modality = normalize_modality(self.modality)
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
        '''一步取值：有媒体给媒体，否则给文本。'''
        return self.media if self.media is not None else self.text


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
        timeout: Optional[float] = None,
        **params: Any,
    ) -> MultimodalResult:
        '''生成：文本 / 图片 / 视频。'''
        return self.run(
            MultimodalRequest(
                operation=Operation.GENERATE,
                modality=modality,
                prompt=prompt,
                inputs=_as_inputs(inputs),
                model=model,
                output_path=save_to,
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
        timeout: Optional[float] = None,
        **params: Any,
    ) -> MultimodalResult:
        '''理解：图片 / 视频（文本也能走，但对齐的是同一个入口）。'''
        return self.run(
            MultimodalRequest(
                operation=Operation.UNDERSTAND,
                modality=modality,
                prompt=prompt,
                inputs=_as_inputs(source),
                model=model,
                output_path=save_to,
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
#: 默认多模态实现：豆包方舟（Seedance 负责视频）
DEFAULT_MULTIMODAL_VENDOR = "doubao"

DOUBAO = MultimodalSpec(
    name="doubao",
    label="豆包方舟（Seedance）",
    kind="doubao",
    env_vars=("ARK_API_KEY", "DOUBAO_API_KEY", "VOLCENGINE_API_KEY"),
    default_base_url=config_module.DOUBAO_BASE_URL,
    models={
        Modality.TEXT: config_module.DOUBAO_TEXT_MODEL,
        Modality.IMAGE: config_module.DOUBAO_IMAGE_MODEL,
        Modality.VIDEO: config_module.DOUBAO_VIDEO_MODEL,
    },
    capabilities=frozenset(
        {
            (Modality.TEXT, Operation.GENERATE),
            (Modality.TEXT, Operation.UNDERSTAND),
            (Modality.IMAGE, Operation.GENERATE),
            (Modality.IMAGE, Operation.UNDERSTAND),
            (Modality.VIDEO, Operation.GENERATE),
            (Modality.VIDEO, Operation.UNDERSTAND),
        }
    ),
    aliases=("seedance", "ark", "volcengine"),
    note="文本 / 图片理解走方舟 Chat Completions，图片生成走 Seedream，"
    "视频生成走 Seedance 任务接口",
)

#: 内置多模态厂商（按展示顺序）
MULTIMODAL_PROVIDERS: Dict[str, MultimodalSpec] = {DOUBAO.name: DOUBAO}

_ALIASES: Dict[str, MultimodalSpec] = {}
for _spec in MULTIMODAL_PROVIDERS.values():
    for _name in _spec.names():
        _ALIASES[_name.lower()] = _spec

_FACTORIES: Dict[str, Callable[..., BaseMultimodalClient]] = {}

#: 内置实现认的 `kind`
_KINDS: Tuple[str, ...] = ("doubao",)


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
    '''按厂商造一个多模态客户端，默认豆包（Seedance）。'''
    spec, target = resolve_multimodal_target(vendor, base_url=base_url)
    key = resolve_multimodal_api_key(spec, api_key)

    factory = _FACTORIES.get(spec.name)
    if factory is None:
        if spec.kind == "doubao":
            from promptkit.client_sys.doubao_multimodal import DoubaoMultimodalClient

            factory = DoubaoMultimodalClient
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


def clone_with_model(request: MultimodalRequest, model: str) -> MultimodalRequest:
    '''复制请求并替换模型（provider 实现里解析默认模型时用）。'''
    return replace(request, model=model)
