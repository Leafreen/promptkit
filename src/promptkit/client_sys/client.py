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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

__all__ = [
    "ClientError",
    "ClientSpec",
    "BaseClient",
    "ChatClient",
]


class ClientError(RuntimeError):
    '''厂商客户端出错（网络 / 鉴权 / 协议错误）。

    单独一个类型，是为了让调用方能把「模型接口挂了」和「工作流 DSL 写错了」
    分开处理 —— 前者值得重试，后者重试多少次都一样。
    '''


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
