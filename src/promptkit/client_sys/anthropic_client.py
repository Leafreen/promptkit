# client_sys/anthropic_client.py
'''Anthropic Claude（Messages API）客户端：两边协议它来翻译。

OpenAI 的 Chat Completions 和 Anthropic 的 Messages 只差四件事：

================  ==================================  ==========================================
差异              OpenAI / DeepSeek                   Anthropic
================  ==================================  ==========================================
系统提示          `messages` 里 role="system"         顶层 `system` 字段
工具调用          消息上的 `tool_calls` + role="tool"   assistant 的 `tool_use` 块 + user 的 `tool_result` 块
参数              `max_tokens` 可选、`stop`            `max_tokens` **必填**、`stop_sequences`
流式              `choices[0].delta` / SSE 数据块       `content_block_delta` 事件
================  ==================================  ==========================================

这个模块把四件事全吃掉：`create()` 收 OpenAI 形状、发 Anthropic 请求、再把
响应翻译回 `types.py` 的 `Completion` / `Chunk`。`Agent` 那边一行都不用改。

刻意**不用 anthropic SDK**：只有一个端点、一种 SSE 格式，urllib 几十行就够了，
还省掉一个依赖（和 `default_tools/web_search.py` 一个路子）。

已知限制
--------
Anthropic 的 extended thinking 没接：开启之后多轮工具调用必须把带 `signature`
的 thinking 块原样回传，而 promptkit 的通用历史里没有地方存它。真传了
`thinking=` 会直接报一个能看懂的错，而不是让 API 甩 400 出来。
'''
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from promptkit.client_sys.client import BaseClient, ClientError, ClientSpec
from promptkit.client_sys.types import (
    Choice,
    Chunk,
    Completion,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)

__all__ = ["AnthropicClient", "iter_sse", "DEFAULT_API_VERSION"]

#: 官方端点（`base_url` 省略时用）
DEFAULT_BASE_URL = "https://api.anthropic.com"
#: 这个头和密钥一样必须带，否则 API 直接拒绝
DEFAULT_API_VERSION = "2023-06-01"
#: Messages API 要求显式给 max_tokens
DEFAULT_MAX_TOKENS = 4096
#: 没设 timeout 时给个大方的值：长回答 + 思考可以跑很久
DEFAULT_TIMEOUT = 600.0

#: Anthropic 的 stop_reason -> OpenAI 的 finish_reason
_FINISH_REASONS: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "stop",
}

#: 参数改名表：OpenAI 名字 -> Anthropic 名字
_PARAM_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("top_k", "top_k"),
    ("stop", "stop_sequences"),
    ("stop_sequences", "stop_sequences"),
    ("system", "system"),
    ("metadata", "metadata"),
)

#: 能直接透传的参数名（报错时列给用户看）
SUPPORTED_PARAMS: Tuple[str, ...] = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "stop_sequences",
    "system",
    "metadata",
    "tool_choice",
)

#: OpenAI 的 tool_choice 字符串 -> Anthropic 的对象
_TOOL_CHOICE = {
    "auto": {"type": "auto"},
    "none": {"type": "none"},
    "required": {"type": "any"},
    "any": {"type": "any"},
}


def _text_of(content: Any) -> str:
    '''把消息的 content 拍平成文本。

    OpenAI 的 content 可能是字符串、也可能是 `[{"type": "text", ...}]`
    这样的块列表；promptkit 自己只发字符串，但历史里可能有别的来源。
    '''
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "image_url":
                    parts.append("[图片]")
                elif part.get("text"):
                    parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _json_object(raw: Any) -> Dict[str, Any]:
    '''工具参数 -> dict。

    模型给的参数是 JSON 字符串，正常情况一次 `json.loads` 就完了。
    解析不了就塞进 `{"_raw": ...}` 而不是丢掉 —— 丢掉等于告诉模型
    「你上次这么调用时什么都没传」，它会更糊涂。
    '''
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw[:4000]}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _tool_choice(value: Any) -> Dict[str, Any]:
    '''OpenAI 的 tool_choice -> Anthropic 的 tool_choice。'''
    if isinstance(value, str):
        return dict(_TOOL_CHOICE.get(value, {"type": "auto"}))
    if isinstance(value, dict):
        if value.get("type") == "function":
            name = (value.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": name}
            return {"type": "auto"}
        if "type" in value:
            return dict(value)
    return {"type": "auto"}


# ============================ SSE ============================
def _decode_line(line: Any) -> str:
    '''一行 SSE 可能是 bytes 也可能是 str。'''
    if isinstance(line, (bytes, bytearray)):
        return bytes(line).decode("utf-8", errors="replace").rstrip("\r\n")
    return str(line).rstrip("\r\n")


def _sse_payload(name: str, buffer: Sequence[str]) -> Optional[Tuple[str, Any]]:
    '''把一组的 `data:` 行拼成 (事件名, 解析后的 JSON)。脏数据返回 None。'''
    raw = "\n".join(buffer)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    event = name or str(payload.get("type") or "")
    return event, payload


def iter_sse(lines: Iterable[Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    '''把 SSE 的行迭代成 `(事件名, data)`。

    只关心 `event:` / `data:` 两种字段：注释行（以 `:` 开头，Anthropic 用来
    做心跳）和未知字段直接跳过。一组 `data:` 行以空行结束，多行会拼起来 ——
    这是 SSE 规范，虽然 Anthropic 目前每个事件只发一行。
    '''
    buffer: List[str] = []
    name = ""
    for line in lines:
        text = _decode_line(line)
        if not text:
            if buffer:
                event = _sse_payload(name, buffer)
                buffer = []
                name = ""
                if event is not None:
                    yield event
            else:
                name = ""
            continue
        if text.startswith(":"):  # 注释 / 心跳
            continue
        field, _, value = text.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            name = value
        elif field == "data":
            buffer.append(value)

    if buffer:  # 最后没有空行收尾的流
        event = _sse_payload(name, buffer)
        if event is not None:
            yield event


# ============================ HTTP ============================
def _iter_response_lines(response: Any) -> Iterator[str]:
    '''逐行读 HTTP 响应，读完关掉。'''
    try:
        for line in response:
            yield _decode_line(line)
    finally:
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()


def _http_error_message(exc: Exception, url: str) -> str:
    '''把 HTTPError 变成一句人话（优先用厂商给的 error.message）。'''
    detail = ""
    try:
        detail = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 读不到响应体就算了
        detail = ""
    message = ""
    try:
        parsed = json.loads(detail)
        message = str((parsed.get("error") or {}).get("message") or "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        message = ""
    text = message or detail.strip()[:400] or str(getattr(exc, "reason", ""))
    return f"Anthropic HTTP {getattr(exc, 'code', '?')}：{text}（{url}）"


# ============================ 客户端 ============================
class AnthropicClient(BaseClient):
    '''Anthropic Messages API 客户端，对外说 Chat Completions。

        client = AnthropicClient(api_key="sk-ant-xxx")
        reply = client.create(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "你好"}],
        )
        print(reply.text)

    `transport=` 用来换掉真正的 HTTP（测试、公司内网代理、或者你想自己
    加一层重试）：

        def transport(url, body, stream):
            return requests.post(url, json=body).json()   # stream=False
    '''

    vendor = "anthropic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spec: Optional[ClientSpec] = None,
        timeout: Optional[float] = None,
        transport: Optional[Any] = None,
        api_version: str = DEFAULT_API_VERSION,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            spec=spec,
            timeout=timeout,
        )
        self.api_version = api_version
        #: 默认 max_tokens（`params` 里带了就以参数为准）
        configured = DEFAULT_MAX_TOKENS
        if spec is not None and spec.default_params.get("max_tokens"):
            configured = int(spec.default_params["max_tokens"])
        self.default_max_tokens = configured
        #: 真正发请求的地方：`(url, body, stream) -> dict | Iterator[str]`
        self.transport = transport or self._http_transport

    # ---------------- 地址 ----------------
    @property
    def endpoint(self) -> str:
        '''完整的 messages 端点（`base_url` 带不带 `/v1` 都行）。'''
        base = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    # ---------------- 入口 ----------------
    def create(
        self,
        *,
        model: str,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **params: Any,
    ) -> Any:
        '''发一次请求。`stream=False` 回 `Completion`，`True` 回 `Chunk` 迭代器。'''
        if not model:
            raise ClientError(
                "Anthropic 需要 model：Claude 没有默认模型能蒙，"
                "请显式传 model=（例如 model='claude-sonnet-4-5'）"
            )
        body = self._build_body(
            model=model, messages=messages, tools=tools, stream=stream, params=params
        )
        if stream:
            return self._stream(body)
        data = self.transport(self.endpoint, body, False)
        if not isinstance(data, dict):
            raise ClientError(
                f"Anthropic 返回了 {type(data).__name__}，期望 dict；"
                "自定义 transport 记得把响应解析成 JSON"
            )
        return self._to_completion(data)

    # ---------------- 请求翻译 ----------------
    def _build_body(
        self,
        *,
        model: str,
        messages: Any,
        tools: Any,
        stream: bool,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        options = dict(params)
        # 思考模式：明确拒绝，而不是让 API 甩一个看不懂的 400
        if "thinking" in options or "extra_body" in options:
            raise ClientError(
                "Anthropic 的 extended thinking 还没接：开启后多轮工具调用必须"
                "回传带 signature 的 thinking 块，promptkit 的通用历史里没地方存。"
                "要用思考模式请继续用 DeepSeek，或者把 tools 去掉再开。"
            )

        system, translated = self._translate_messages(messages)
        body: Dict[str, Any] = {
            "model": model,
            "messages": translated,
            "max_tokens": int(options.pop("max_tokens", self.default_max_tokens)),
            "stream": bool(stream),
        }

        if system:
            body["system"] = system
        if tools:
            body["tools"] = [self._tool(item) for item in tools]

        for source, target in _PARAM_ALIASES:
            value = options.pop(source, None)
            if value is not None:
                body[target] = value

        if "tool_choice" in options:
            body["tool_choice"] = _tool_choice(options.pop("tool_choice"))

        if options:
            raise ClientError(
                f"Anthropic 不认识这些参数：{', '.join(sorted(options))}。"
                f"能用的有：{', '.join(SUPPORTED_PARAMS)}；"
                "`reasoning_effort` / `extra_body` / `response_format` 这类是"
                "OpenAI / DeepSeek 专有的。"
            )
        return body

    @classmethod
    def _translate_messages(cls, messages: Any) -> Tuple[str, List[Dict[str, Any]]]:
        '''OpenAI 消息列表 -> (system, Anthropic messages)。

        顺手做三件 Anthropic 要求的事：system 提到顶层、工具结果变成 user 的
        `tool_result` 块、相邻同角色消息合并成一 turn。
        '''
        system_parts: List[str] = []
        turns: List[Dict[str, Any]] = []

        for raw in messages or ():
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "user")
            text = _text_of(raw.get("content"))

            if role == "system":
                if text.strip():
                    system_parts.append(text)
                continue

            if role in ("tool", "function"):
                # 工具结果在 Anthropic 里是 user 消息里的一个块
                call_id = raw.get("tool_call_id") or raw.get("name") or ""
                cls._append(
                    turns,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(call_id),
                            "content": text or "（工具没有返回内容）",
                        }
                    ],
                )
                continue

            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if text:
                    blocks.append({"type": "text", "text": text})
                for call in raw.get("tool_calls") or ():
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "input": _json_object(function.get("arguments")),
                        }
                    )
                # 空的 assistant 消息会让 API 报错（text 块不能为空），直接丢
                if blocks:
                    cls._append(turns, "assistant", blocks)
                continue

            if text:
                cls._append(turns, "user", [{"type": "text", "text": text}])

        return "\n\n".join(system_parts), turns

    @staticmethod
    def _append(
        turns: List[Dict[str, Any]], role: str, blocks: List[Dict[str, Any]]
    ) -> None:
        '''追加一个 turn；和上一个同角色就合并（Anthropic 要求角色交替）。'''
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
            return
        turns.append({"role": role, "content": blocks})

    @staticmethod
    def _tool(tool: Any) -> Dict[str, Any]:
        '''OpenAI 的 function 工具 -> Anthropic 的 input_schema 工具。'''
        if not isinstance(tool, dict):
            tool = getattr(tool, "schema", lambda: {})()
        function = tool.get("function", tool)
        definition: Dict[str, Any] = {
            "name": function.get("name", ""),
            "input_schema": function.get("parameters")
            or {"type": "object", "properties": {}},
        }
        description = function.get("description")
        if description:
            definition["description"] = description
        return definition

    # ---------------- 响应翻译 ----------------
    @staticmethod
    def _to_completion(data: Dict[str, Any]) -> Completion:
        '''Anthropic 的 message -> `Completion`。'''
        texts: List[str] = []
        thinking: List[str] = []
        calls: List[ToolCall] = []

        for block in data.get("content") or ():
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text") or "")
            elif kind == "thinking":
                thinking.append(block.get("thinking") or "")
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        function=FunctionCall(
                            name=str(block.get("name") or ""),
                            arguments=json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        ),
                    )
                )

        message = Message(
            role="assistant",
            content="".join(texts),
            tool_calls=calls or None,
            reasoning_content="".join(thinking) or None,
        )
        finish = _FINISH_REASONS.get(str(data.get("stop_reason") or ""), "stop")
        usage = data.get("usage")
        return Completion(
            choices=[Choice(index=0, message=message, finish_reason=finish)],
            usage=Usage.from_anthropic(usage) if isinstance(usage, dict) else None,
            model=data.get("model"),
            id=data.get("id"),
            raw=data,
        )

    # ---------------- 流式 ----------------
    def _stream(self, body: Dict[str, Any]) -> Iterator[Chunk]:
        for name, payload in iter_sse(self.transport(self.endpoint, body, True)):
            kind = payload.get("type") or name

            if kind == "content_block_start":
                block = payload.get("content_block") or {}
                if block.get("type") == "tool_use":
                    yield Chunk(
                        choices=[
                            Choice(
                                delta=Message(
                                    tool_calls=[
                                        ToolCall(
                                            index=_index_of(payload),
                                            id=str(block.get("id") or ""),
                                            function=FunctionCall(
                                                name=str(block.get("name") or ""),
                                                arguments="",
                                            ),
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                continue

            if kind == "content_block_delta":
                delta = payload.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    yield Chunk(
                        choices=[
                            Choice(delta=Message(content=delta.get("text") or ""))
                        ]
                    )
                elif dtype == "thinking_delta":
                    yield Chunk(
                        choices=[
                            Choice(
                                delta=Message(
                                    reasoning_content=delta.get("thinking") or ""
                                )
                            )
                        ]
                    )
                elif dtype == "input_json_delta":
                    yield Chunk(
                        choices=[
                            Choice(
                                delta=Message(
                                    tool_calls=[
                                        ToolCall(
                                            index=_index_of(payload),
                                            function=FunctionCall(
                                                arguments=delta.get("partial_json") or ""
                                            ),
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                continue

            if kind == "message_start":
                usage = ((payload.get("message") or {}).get("usage")) or {}
                chunk = _usage_chunk(usage)
                if chunk is not None:
                    yield chunk
                continue

            if kind == "message_delta":
                delta = payload.get("delta") or {}
                chunk = _usage_chunk(payload.get("usage") or {})
                if chunk is not None:
                    yield chunk
                finish = _FINISH_REASONS.get(str(delta.get("stop_reason") or ""))
                if finish:
                    yield Chunk(choices=[Choice(finish_reason=finish)])
                continue

            if kind == "error":
                error = payload.get("error") or {}
                raise ClientError(
                    "Anthropic 流式中途报错："
                    f"{error.get('type') or 'unknown'} {error.get('message') or ''}".strip()
                )

    # ---------------- 真正的 HTTP ----------------
    def _http_transport(
        self, url: str, body: Dict[str, Any], stream: bool
    ) -> Any:
        '''用 urllib 发请求。非流式返回 dict，流式返回逐行迭代器。'''
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": self.api_version,
                "content-type": "application/json",
                "accept": "text/event-stream" if stream else "application/json",
            },
        )
        timeout = self.timeout or DEFAULT_TIMEOUT
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise ClientError(_http_error_message(exc, url)) from None
        except urllib.error.URLError as exc:
            raise ClientError(f"连接 Anthropic 失败：{exc.reason}（{url}）") from None
        except TimeoutError:
            raise ClientError(f"请求 Anthropic 超时（>{timeout}s）：{url}") from None
        except OSError as exc:
            raise ClientError(f"网络错误：{exc}（{url}）") from None

        if stream:
            # 不能在这里 read()，也不能用 with：生成器活到迭代结束才算完
            return _iter_response_lines(response)

        try:
            raw = response.read()
        finally:
            closer = getattr(response, "close", None)
            if callable(closer):
                closer()
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ClientError(f"Anthropic 响应不是合法 JSON：{exc}") from None


def _index_of(payload: Dict[str, Any]) -> int:
    '''内容块的下标（流式拼工具调用要用）。'''
    try:
        return int(payload.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _usage_chunk(usage: Dict[str, Any]) -> Optional[Chunk]:
    '''用量单独一个分片（`Agent` 会忽略，日志 handler 可以记）。'''
    if not isinstance(usage, dict):
        return None
    parsed = Usage.from_anthropic(usage)
    if not parsed.total_tokens:
        return None
    return Chunk(choices=[], usage=parsed)
