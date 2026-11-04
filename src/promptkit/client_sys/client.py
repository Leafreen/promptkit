# client_sys/client.py
'''客户端接口：一个厂商该长什么样。

只有两件事：

1. `ClientSpec` —— 厂商的「静态信息」：叫什么、密钥从哪个环境变量读、
   默认模型 / 端点是什么、有哪些默认请求参数。
2. `BaseClient` —— 厂商的「行为」：`create()` 收一批 Chat Completions 风格的
   参数，回一个 `Completion`（或流式 `Chunk` 迭代器）。

为什么接口长这样
----------------
`Agent` 原本直接调 `openai.OpenAI().chat.completions.create(...)`。既然所有
调用都从这一个洞进出，那就把洞口标准化：**promptkit 内部的协议就是
Chat Completions**，新厂商只要在这个洞口上做翻译。

所以 `BaseClient` 顺手把 `chat` / `completions` 两个属性指向自己 ——
`client.chat.completions.create(...)` 和 `client.create(...)` 是同一个东西。
好处是老的 `agent.client.chat.completions.create(...)` 写法、以及测试里
塞 `SimpleNamespace(chat=SimpleNamespace(completions=...))` 的假客户端
都还照常能用；想直接调 `client.create(...)` 也可以。
'''
from __future__ import annotations

import base64
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "ClientError",
    "UnsupportedModality",
    "ClientSpec",
    "BaseClient",
    "ChatClient",
    "part_data_url",
]


class ClientError(RuntimeError):
    '''厂商客户端出错（网络 / 鉴权 / 协议错误）。

    单独一个类型，是为了让调用方能把「模型接口挂了」和「工作流 DSL 写错了」
    分开处理 —— 前者值得重试，后者重试多少次都一样。
    '''


class UnsupportedModality(ClientError):
    '''厂商不支持某个模态（图片 / 音频 / 视频 / 文件）。

    不静默降级：模型吃不了这个模态就立刻抛错，而不是偷偷转成文本。
    '''

    def __init__(self, client: str, modality: str):
        self.client = client
        self.modality = modality
        super().__init__(
            f"{client} 不支持 {modality} 模态；"
            "请换支持它的厂商，或把内容换成文本"
        )


def part_data_url(part: Mapping[str, Any]) -> str:
    '''把标准内容部件字典（含 source / source_type / media_type）转成 URL。

    - ``url``：原样返回
    - ``base64``：拼成 data URL
    - ``file``：读取本地文件编码成 data URL
    '''
    source = str(part.get("source") or "")
    source_type = part.get("source_type")
    media_type = part.get("media_type")
    if source_type == "url":
        return source
    if source_type == "base64":
        return f"data:{media_type or 'application/octet-stream'};base64,{source}"
    if source_type == "file":
        if not media_type:
            media_type = mimetypes.guess_type(source)[0] or "application/octet-stream"
        with open(source, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("ascii")
        return f"data:{media_type};base64,{payload}"
    raise ClientError(f"未知的内容来源类型：{source_type!r}")


@dataclass(frozen=True)
class ClientSpec:
    '''一家厂商的静态信息。

    `default_params` 是**发请求时自动带上的参数**（思考模式之类）。
    它不是共享配置，`Agent` 会 `dict(...)` 一份再合并用户参数，
    所以这里放心写字面量。
    '''

    #: 内部名字，也是 `vendor=` 接受的值
    name: str
    #: 给人看的名字（报错信息里用）
    label: str
    #: 用哪个客户端实现：`"openai"`（Chat Completions 兼容）或 `"anthropic"`
    kind: str
    #: 依次尝试的环境变量名
    env_vars: Tuple[str, ...] = ()
    #: `model=None` 时用它；None 表示必须显式指定
    default_model: Optional[str] = None
    #: `base_url=None` 时用它；None 表示必须显式指定
    default_base_url: Optional[str] = None
    #: 默认请求参数
    default_params: Mapping[str, Any] = field(default_factory=dict)
    #: 别称（`vendor="claude"` 也行）
    aliases: Tuple[str, ...] = ()
    #: 补充说明（`vendor_names()` 的展示、报错提示用）
    note: str = ""

    def names(self) -> Tuple[str, ...]:
        return (self.name, *self.aliases)


class BaseClient(ABC):
    '''一家厂商的客户端。

    子类只要实现 `create()`；其余都是给调用方用的便利。
    '''

    #: 厂商名（和 `ClientSpec.name` 一致）
    vendor: str = "base"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[ClientSpec] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.spec = spec
        self.timeout = timeout

    # ---------------- OpenAI 风格的调用面 ----------------
    # Agent 是按 `client.chat.completions.create(...)` 调的。让这两个属性
    # 指回自己，等于免费提供一层兼容外壳，也不用多造一个类。
    @property
    def chat(self) -> "BaseClient":
        return self

    @property
    def completions(self) -> "BaseClient":
        return self

    # ---------------- 真正要实现的 ----------------
    @abstractmethod
    def create(
        self,
        *,
        model: str,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **params: Any,
    ) -> Any:
        '''发一次请求。

        - 参数和返回值都按 OpenAI Chat Completions 的语义：
          `messages` 是 OpenAI 形状的消息列表，`tools` 是
          `{"type": "function", "function": {...}}` 列表
        - `stream=False` 返回 `Completion`（或 SDK 原始对象）
        - `stream=True` 返回**迭代器**，逐条产出 `Chunk`
        - `params` 是这次调用的额外参数（温度、max_tokens、思考模式……）
        '''

    # ---------------- 多模态能力 ----------------
    def supports(self, modality: str) -> bool:
        '''这个客户端支持哪种模态：text / image / audio / video / file。

        默认只支持文本；各适配器按厂商能力覆写。
        '''
        return modality == "text"

    def _translate_message(self, message: Any) -> Any:
        '''把消息里 list 形态的 content 翻成厂商格式。

        标准部件字典（`Prompt.render()` 的输出）带 ``source`` 键，
        认出来就交给 `_to_vendor_part`；其余形状（纯文本、厂商原生
        block、非 dict 消息）原样透传 —— 老行为一行不变。
        '''
        if not isinstance(message, dict):
            return message
        content = message.get("content")
        if not isinstance(content, list):
            return message
        if not any(isinstance(p, dict) and "source" in p for p in content):
            return message
        converted = [
            self._to_vendor_part(p)
            if isinstance(p, dict) and "source" in p
            else p
            for p in content
        ]
        return {**message, "content": converted}

    def _to_vendor_part(self, part: Mapping[str, Any]) -> Dict[str, Any]:
        '''把一个标准内容部件字典翻成厂商格式。默认谁都不支持。'''
        raise UnsupportedModality(self.vendor, str(part.get("type")))

    # ---------------- 杂项 ----------------
    def close(self) -> None:
        '''释放底层连接。默认什么也不做。'''

    def __enter__(self) -> "BaseClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} vendor={self.vendor!r} base_url={self.base_url!r}>"


#: 类型别名：凡是有 `create()` 的都算客户端
ChatClient = BaseClient
