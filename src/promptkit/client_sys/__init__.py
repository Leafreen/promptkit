# client_sys/__init__.py
'''多厂商模型客户端。

一句话：**promptkit 内部只说 OpenAI Chat Completions 这一种方言**，
这个包负责把别家厂商翻译进来、翻译回去。

    from promptkit.client_sys import make_client

    client = make_client("anthropic", api_key="sk-ant-xxx")
    reply = client.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(reply.text)

大多数时候不用直接碰这里 —— `Agent(vendor="anthropic")` 就够了，
`agent.client` 就是这里造出来的客户端。

分层（依赖从上往下）：

    types.py            Completion / Chunk / Message …… 纯数据，不依赖任何东西
    client.py           BaseClient（接口）+ ClientSpec（厂商信息）+ ClientError
    openai_client.py    OpenAI / DeepSeek / 任何 OpenAI 兼容端点（透传 SDK）
    anthropic_client.py Messages API <- 翻译 -> Chat Completions
    registry.py         厂商表 + 识别 + 密钥 + 工厂
    multimodal.py       多模态统一接口 + 厂商表（默认豆包 / Seedance）
    doubao_multimodal.py 豆包方舟：文本 / 图片 / 视频

加新厂商见 `registry.register_vendor`；加新模态厂商见
`multimodal.register_multimodal_provider`。
'''
from promptkit.client_sys.types import (
    Choice,
    Chunk,
    Completion,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from promptkit.client_sys.client import (
    BaseClient,
    ChatClient,
    ClientError,
    ClientSpec,
)
from promptkit.client_sys.openai_client import OpenAIClient
from promptkit.client_sys.anthropic_client import (
    DEFAULT_API_VERSION,
    AnthropicClient,
    iter_sse,
)
from promptkit.client_sys.registry import (
    DEFAULT_VENDOR,
    FALLBACK_ENV_VAR,
    PROVIDERS,
    api_key_source,
    detect_vendor,
    get_spec,
    make_client,
    register_vendor,
    resolve_api_key,
    resolve_target,
    vendor_names,
    with_params,
)
from promptkit.client_sys.multimodal import (
    DOUBAO,
    DEFAULT_MULTIMODAL_VENDOR,
    MULTIMODAL_PROVIDERS,
    BaseMultimodalClient,
    Media,
    Modality,
    MultimodalError,
    MultimodalRequest,
    MultimodalResult,
    MultimodalSpec,
    Operation,
    get_multimodal_spec,
    make_multimodal_client,
    multimodal_vendor_names,
    normalize_modality,
    normalize_operation,
    register_multimodal_provider,
    resolve_multimodal_api_key,
    resolve_multimodal_target,
)
from promptkit.client_sys.doubao_multimodal import DoubaoMultimodalClient

__all__ = [
    # 数据
    "Choice",
    "Chunk",
    "Completion",
    "FunctionCall",
    "Message",
    "ToolCall",
    "Usage",
    # 接口
    "BaseClient",
    "ChatClient",
    "ClientError",
    "ClientSpec",
    # 实现
    "OpenAIClient",
    "AnthropicClient",
    "iter_sse",
    "DEFAULT_API_VERSION",
    # 注册表
    "PROVIDERS",
    "DEFAULT_VENDOR",
    "FALLBACK_ENV_VAR",
    "api_key_source",
    "detect_vendor",
    "get_spec",
    "make_client",
    "register_vendor",
    "resolve_api_key",
    "resolve_target",
    "vendor_names",
    "with_params",
    # 多模态
    "Modality",
    "Operation",
    "Media",
    "MultimodalRequest",
    "MultimodalResult",
    "MultimodalSpec",
    "MultimodalError",
    "BaseMultimodalClient",
    "DoubaoMultimodalClient",
    "DOUBAO",
    "MULTIMODAL_PROVIDERS",
    "DEFAULT_MULTIMODAL_VENDOR",
    "normalize_modality",
    "normalize_operation",
    "get_multimodal_spec",
    "register_multimodal_provider",
    "resolve_multimodal_api_key",
    "resolve_multimodal_target",
    "make_multimodal_client",
    "multimodal_vendor_names",
]
