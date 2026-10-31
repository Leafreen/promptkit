'''Agent：模型对话客户端。

配置（密钥从哪来、用哪个模型）都在 promptkit/config.py 和
promptkit/client_sys/registry.py；Agent 只负责「怎么干活」：发请求、跑工具循环、
**发事件**。

厂商是可换的（`Agent(vendor="openai")` / `Agent(vendor="anthropic")`），
但 Agent 的代码里**没有一处按厂商分支** —— 客户端层（`client_sys/`）把每家
厂商的协议都翻译成 Chat Completions 这一种方言，Agent 只认这一个。

Agent 自己不 print。想看过程就挂 handler：

    bus = EventBus()
    bus.on(console_display())
    agent = Agent(bus=bus)
    agent.ask("你好")
'''
import json
import os
import sys
from contextlib import contextmanager
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel

from promptkit.client_sys import (
    BaseClient,
    ClientError,  # 再导出：ask() 抛的网络 / 鉴权错误就是它
    make_client,
    resolve_api_key as resolve_vendor_api_key,
    resolve_target,
    with_params,
)
from promptkit.config import (
    DEFAULT_BASE_URL,  # 再导出，保持老代码可 import
    DEFAULT_EXTRA_BODY,  # 同上（现在归 DeepSeek 厂商规格管）
    DEFAULT_MODEL,  # 同上
    DEFAULT_REASONING_EFFORT,  # 同上
    DEFAULT_START_PROMPTS,
    ENV_FILE,
    ENV_VAR,
    load_env_file,  # 再导出，保持 `from promptkit.agent import load_env_file` 可用
    resolve_api_key,  # 再导出：老代码仍可用（只认 DeepSeek 的密钥）
)
from promptkit.client_sys.types import Chunk, Completion, Message as ClientMessage
from promptkit.core.display_sys.display import console_display
from promptkit.core.event_sys import (
    Event,
    EventBus,
    EventHandler,
    EventType,
    extract_reasoning,
)
from promptkit.core.display_sys.render import Verbosity, as_verbosity, render_markdown
from promptkit.core.settings import settings
from promptkit.executor import Executor
from promptkit.prompt import Prompt
from promptkit.core.tool import Tool, as_tools
from promptkit.types import (
    JSONValue,
    Message,
    MessageContent,
    MessageRole,
    PromptInput,
    ToolCall,
    ToolCallFunction,
    ToolLike,
)


__all__ = [
    "Agent",
    "Message",
    "ToolLoopExceeded",
    "run_prompt",
    "load_env_file",
    "normalize_history",
    "ENV_VAR",
    "ENV_FILE",
]


class ToolLoopExceeded(Exception):
    '''模型反复要求调用工具，超过了 max_tool_rounds。'''


#: 导出 Markdown 历史时的角色名
_ROLE_LABELS = {"user": "你", "assistant": "助手", "system": "系统", "tool": "工具"}

#: 历史里允许出现的角色。`system` 不算历史 —— 它属于 `start_prompts`
_HISTORY_ROLES = frozenset({"user", "assistant", "tool", "function"})


def normalize_history(records: object) -> List[Message]:
    '''把「各种来源的历史」统一成干净的 `[{role, content}, ...]`。

    接受的东西：

    - 文件路径（`.json` / `.jsonl`）—— 自己读
    - 消息列表
    - `Session` 存出来的整包 `{"messages": [...], ...}`
    - `export_history` 写的裸数组

    会**丢掉**不认识的条目，而不是报错 —— 会话文件是「尽力恢复」的东西，
    为了一条脏数据让整个对话开不起来不值得。`system` 也丢掉：
    它属于 `start_prompts`，混进历史里会被当成"上一轮我说过"。
    '''
    raw = _load_history_source(records)
    if isinstance(raw, dict):
        raw = raw.get("messages") or raw.get("history") or []
    if not isinstance(raw, (list, tuple)):
        return []

    cleaned: List[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role not in _HISTORY_ROLES:
            continue
        content: MessageContent = item.get("content")
        message: Message = {"role": cast(MessageRole, role), "content": content}
        # 工具调用记录要原样带着，否则续聊时模型会发现「工具结果没有对应的调用」
        if item.get("tool_calls"):
            message["tool_calls"] = item["tool_calls"]
        if item.get("tool_call_id"):
            message["tool_call_id"] = item["tool_call_id"]
        if item.get("name"):
            message["name"] = item["name"]
        cleaned.append(message)
    return cleaned


def _load_history_source(records: object) -> object:
    '''路径就开文件读，其它原样返回。'''
    if not isinstance(records, (str, os.PathLike)):
        return records
    path = os.fspath(records)
    if not os.path.exists(path):
        raise FileNotFoundError(f"读不到会话文件：{path}")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if path.lower().endswith(".jsonl"):
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    return json.loads(text)


def _history_as_markdown(history: Sequence[Message]) -> str:
    '''把对话历史写成 Markdown（给人看的）。'''
    lines: List[str] = ["# 对话历史", ""]
    for message in history:
        role = str(message.get("role", "?"))
        label = _ROLE_LABELS.get(role, role)
        lines.append(f"## {label}")
        lines.append("")
        content = message.get("content")
        lines.append("" if content is None else str(content))
        lines.append("")
        if message.get("tool_calls"):
            import json

            lines.append("```json")
            lines.append(json.dumps(message["tool_calls"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


class _SimpleCall:
    '''把流式拼出来的工具调用包装成和非流式一样的形状。'''

    def __init__(self, payload: ToolCall):
        self.id = payload["id"]
        self.function = _SimpleFunction(payload["function"])


class _SimpleFunction:
    def __init__(self, payload: ToolCallFunction):
        self.name = payload["name"]
        self.arguments = payload["arguments"]


class Agent:
    '''模型对话客户端。`Agent()` 不带参数即可用：

    - 默认还是 DeepSeek：密钥按 `api_key 参数 -> 环境变量 DEEPSEEK_API_KEY ->
      项目根目录 .env` 的顺序查找，模型 `deepseek-flash`，开了思考模式。
    - 换厂商只要说一句，密钥 / 端点 / 模型都会跟着换：

        Agent(vendor="openai")                 # OPENAI_API_KEY，gpt-4o-mini
        Agent(vendor="openai", model="gpt-4o")
        Agent(vendor="anthropic")              # ANTHROPIC_API_KEY，claude-sonnet-4-5
        Agent(model="claude-sonnet-4-5")       # 按模型名自动认出是 Anthropic
        Agent(base_url="http://localhost:11434/v1",
              model="qwen2.5", api_key="ollama")   # 本地 / 中转 / 任何兼容端点

    - 实例化时会自动注册为「默认 agent」，所以 `Workflow.run()` 不用再传 agent。
    - 事件：传 `bus=` 或 `on_event=` 就能看到思考过程 / 工具调用 / 回复。
      `verbose=True` 等价于挂一个 console_display()。

    `agent.client` 是厂商客户端（`client_sys.BaseClient`）：它长得像 openai SDK
    （`client.chat.completions.create(...)` 照样能用），要拿原生 SDK 就看
    `agent.client.sdk`（Anthropic 那边没有 SDK，就没这个属性）。
    '''

    _default: Optional["Agent"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        start_prompts: Optional[Sequence[Dict[str, str]]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        bus: Optional[EventBus] = None,
        on_event: Optional[EventHandler] = None,
        verbose: bool = False,
        register_as_default: bool = True,
        vendor: Optional[str] = None,
        params: Optional[Dict[str, JSONValue]] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        # 厂商 / 模型 / 端点是一组，缺的用厂商默认值补（补不上会在这一步报错）
        self.spec, self.model, self.base_url = resolve_target(
            vendor=vendor, model=model, base_url=base_url, api_key=api_key
        )
        self.vendor = self.spec.name
        self.api_key = resolve_vendor_api_key(self.spec, api_key)

        # 厂商默认参数（DeepSeek 的思考模式就住这儿）+ 用户参数。用户优先。
        self.params: Dict[str, JSONValue] = with_params(self.spec, params)
        if max_tokens is not None:
            self.params["max_tokens"] = max_tokens
        self.timeout = timeout

        # 不要用可变对象做默认参数，否则所有实例会共享同一份列表
        start = DEFAULT_START_PROMPTS if start_prompts is None else start_prompts
        self.start_prompts: List[Message] = [cast(Message, item) for item in start]

        self.bus = bus if bus is not None else EventBus()
        if on_event is not None:
            self.bus.on(on_event)
        if verbose:
            # verbose 只输出「过程」（思考 / 工具调用 / 步骤），写到 stderr；
            # 最终回答是**结果**，留在 stdout —— 两者天然分开。
            self.bus.on(
                console_display(show_message=False, stream=sys.stderr)
            )

        self.client: BaseClient = make_client(
            self.spec,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        self.history: List[Message] = []  # 上下文
        self.messages: List[Message] = self._build_messages()

        if register_as_default:
            Agent._default = self

    # ---------------- 默认 agent ----------------
    @classmethod
    def default(cls) -> Optional["Agent"]:
        '''返回最近一次创建的 Agent（`Agent()` 会自动注册）。'''
        return cls._default

    @classmethod
    def set_default(cls, agent: Optional["Agent"]) -> None:
        '''手动指定默认 agent。'''
        cls._default = agent

    def __repr__(self) -> str:
        return f"<Agent {self.vendor} model={self.model!r}>"

    def on(self, handler: EventHandler) -> EventHandler:
        '''挂一个事件 handler，返回它（可以当装饰器）。'''
        return self.bus.on(handler)

    def _build_messages(self) -> List[Message]:
        '''把系统提示和上下文拼成要发送给模型的消息列表。'''
        return [*self.start_prompts, *self.history]

    # ---------------- 请求 ----------------
    def _create(
        self, tools: Optional[List[Dict[str, JSONValue]]] = None, stream: bool = False
    ) -> object:
        '''发一次请求。只有传了工具才带 tools 参数。

        `self.params` 是厂商默认参数（DeepSeek 的思考模式）和用户参数的合并结果；
        Anthropic 那种不认这些参数的厂商，由客户端自己忽略 / 翻译，
        Agent 不需要知道。
        '''
        # 显式的 model / messages / stream / tools 优先于 params 里的同名项
        extra = {
            key: value
            for key, value in self.params.items()
            if key not in ("model", "messages", "stream", "tools")
        }
        # `chat.completions` 只是兼容外壳，等价于 client.create(...)
        if tools:
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=stream,
                tools=tools,
                **extra,
            )
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=stream,
            **extra,
        )

    def _append_message(self, message: Message) -> None:
        self.history.append(message)
        self.messages = self._build_messages()

    def _append_assistant(self, assistant: object) -> Message:
        '''把助手的回复记进上下文。

        关键是**保留 reasoning_content**：请求里带了 tools 时，DeepSeek 要求
        把之前所有轮次的 reasoning_content 原样传回去，否则会返回 400。
        '''
        message: Message = {
            "role": "assistant",
            "content": getattr(assistant, "content", None) or "",
        }
        tool_calls = getattr(assistant, "tool_calls", None)
        if tool_calls:
            message["tool_calls"] = [
                cast(
                    ToolCall,
                    call.model_dump() if hasattr(call, "model_dump") else dict(call),
                )
                for call in tool_calls
            ]
        reasoning = extract_reasoning(assistant)
        if reasoning:
            message["reasoning_content"] = reasoning

        self._append_message(message)
        return message

    def _run_tool(self, tools_by_name: Dict[str, Tool], call: object) -> Message:
        '''执行一次工具调用，把结果包成 role="tool" 的消息。

        工具报错不会中断整个对话，而是把错误文本交给模型，让它自己决定怎么办。
        '''
        function = getattr(call, "function", None)
        name = getattr(function, "name", "") or ""
        raw = getattr(function, "arguments", "") or "{}"
        call_id = cast(str, getattr(call, "id", None) or name)
        reply: Message = {"role": "tool", "content": "", "tool_call_id": call_id}

        tool = tools_by_name.get(name)
        if tool is None:
            reply["content"] = f"错误：没有名为 {name!r} 的工具"
            return reply

        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            reply["content"] = f"错误：工具参数不是合法 JSON（{exc}）：{raw}"
            return reply

        try:
            reply["content"] = tool.call(arguments)
        except Exception as exc:  # noqa: BLE001 - 交给模型处理
            reply["content"] = f"错误：工具执行失败 {type(exc).__name__}: {exc}"
        return reply

    @contextmanager
    def _extra_handlers(
        self, bus: EventBus, on_event: Optional[EventHandler], verbose: bool
    ) -> Iterator[EventBus]:
        '''给「这一次调用」临时挂 handler，结束就摘掉。'''
        attached: List[EventHandler] = []
        if on_event is not None:
            attached.append(bus.on(on_event))
        if verbose:
            attached.append(
                bus.on(console_display(show_message=False, stream=sys.stderr))
            )
        try:
            yield bus
        finally:
            for handler in attached:
                bus.off(handler)

    # ---------------- 非流式 ----------------
    def ask(
        self,
        prompt: PromptInput,
        tools: Optional[Sequence[ToolLike]] = None,
        max_tool_rounds: Optional[int] = None,
        bus: Optional[EventBus] = None,
        on_event: Optional[EventHandler] = None,
        verbose: bool = False,
        stream: bool = False,
    ) -> str:
        '''向该助手发送一个提示，返回助手的回复文本。

        - `prompt`：纯文本（按 user 角色发送），或 {"role": ..., "content": ...}
        - `tools`：Tool / 普通函数 / API 原生 dict 都行。传了就会自动跑
          「模型要工具 -> 执行 -> 结果喂回模型」的循环
        - `max_tool_rounds`：工具循环上限；省略时用 `settings.max_tool_rounds`
        - `stream=True`：内部走流式，但仍返回完整字符串，同时把事件发给 bus
        - `verbose=True`：这一次调用临时打印思考 / 工具 / 回复
        '''
        target = bus if bus is not None else self.bus
        rounds = settings.max_tool_rounds if max_tool_rounds is None else max_tool_rounds

        with self._extra_handlers(target, on_event, verbose):
            if stream:
                text = ""
                for event in self._stream_events(prompt, tools, rounds):
                    target.emit(event)
                    if event.type == EventType.MESSAGE_END:
                        text = event.data.get("text", text)
                return text
            return self._ask_once(prompt, tools, target, rounds)

    def _ask_once(
        self,
        prompt: PromptInput,
        tools: Optional[Sequence[ToolLike]],
        bus: EventBus,
        max_tool_rounds: int,
    ) -> str:
        self._append_message(self._normalise(prompt))

        tool_list = as_tools(tools)
        schemas = [item.schema() for item in tool_list] or None
        by_name = {item.name: item for item in tool_list}

        for _ in range(max_tool_rounds + 1):
            response = cast(Completion, self._create(schemas))
            self._emit_usage(bus, response)
            assistant = response.choices[0].message

            reasoning = extract_reasoning(assistant)
            if reasoning:
                bus.emit(Event(EventType.THINKING_START, {}))
                bus.emit(Event(EventType.THINKING_DELTA, {"text": reasoning}))
                bus.emit(Event(EventType.THINKING_END, {"text": reasoning}))

            self._append_assistant(assistant)
            content = getattr(assistant, "content", None) or ""
            calls = getattr(assistant, "tool_calls", None)

            if not calls:
                bus.emit(Event(EventType.MESSAGE_END, {"text": content}))
                return content

            # 工具里可能又调 agent.ask（比如 Skill 内部要问一次模型）。那是**另一次
            # 对话**，不能写进当前这段「assistant(tool_calls) -> tool」之间，否则
            # 消息序列就坏了（DeepSeek/OpenAI 会直接 400）。
            # 做法：工具执行期间把 history 换成「这条 tool_calls 之前」的副本，
            # 跑完丢掉副本、只把 tool 结果写回真正的历史。
            saved_history, saved_messages = self.history, self.messages
            clean = list(saved_history)
            if clean and clean[-1].get("tool_calls"):
                clean = clean[:-1]
            self.history = clean
            self.messages = self._build_messages()
            replies: List[Message] = []
            try:
                for call in calls:
                    function = getattr(call, "function", None)
                    name = getattr(function, "name", "") or ""
                    arguments = getattr(function, "arguments", "") or ""
                    bus.emit(
                        Event(
                            EventType.TOOL_CALL_START,
                            {"name": name, "arguments": arguments},
                        )
                    )
                    bus.emit(
                        Event(
                            EventType.TOOL_CALL_END,
                            {"name": name, "arguments": arguments},
                        )
                    )
                    reply = self._run_tool(by_name, call)
                    bus.emit(
                        Event(
                            EventType.TOOL_RESULT,
                            {
                                "name": name,
                                "arguments": arguments,
                                "result": reply["content"],
                                "tool_call_id": reply["tool_call_id"],
                            },
                        )
                    )
                    replies.append(reply)
            finally:
                self.history = saved_history
                self.messages = saved_messages

            for reply in replies:
                self._append_message(reply)

        raise ToolLoopExceeded(
            f"连续 {max_tool_rounds} 轮工具调用后模型还在要工具，已中止；"
            "检查一下工具是不是一直失败或者互相循环调用"
        )

    @staticmethod
    def _emit_usage(bus: EventBus, response: object) -> None:
        '''把这次调用的 token 用量发出去（服务端没给就跳过）。

        终端默认不显示，`log_display(...)` 会写进日志文件。
        '''
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        data = {
            key: getattr(usage, key, None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        if all(value is None for value in data.values()):
            return
        bus.emit(Event(EventType.USAGE, data))

    # ---------------- 流式 ----------------
    def ask_stream(
        self,
        prompt: PromptInput,
        tools: Optional[Sequence[ToolLike]] = None,
        max_tool_rounds: Optional[int] = None,
        on_event: Optional[EventHandler] = None,
        verbose: bool = False,
    ) -> Iterator[Event]:
        '''流式版：**逐个事件**返回，边到边处理。

            for event in agent.ask_stream("你好"):
                print(event)

        最后一条是 MESSAGE_END，`event.data["text"]` 是完整回复。

        注意：这个方法**只把事件交给你**，不会再发给 `agent.bus`——
        否则自己迭代 + 总线显示就会看到两遍。想让总线处理就用
        `agent.ask(..., stream=True)` 或 `agent.ask(..., verbose=True)`。
        '''
        handlers: List[EventHandler] = []
        if on_event is not None:
            handlers.append(on_event)
        if verbose:
            handlers.append(console_display(show_message=False, stream=sys.stderr))

        rounds = settings.max_tool_rounds if max_tool_rounds is None else max_tool_rounds
        stream = self._stream_events(prompt, tools, rounds)
        if not handlers:
            yield from stream
            return

        # 临时挂上，只为这一次迭代服务
        for handler in handlers:
            self.bus.on(handler)
        try:
            yield from stream
        finally:
            for handler in handlers:
                self.bus.off(handler)

    def _stream_events(
        self,
        prompt: PromptInput,
        tools: Optional[Sequence[ToolLike]],
        max_tool_rounds: int,
    ) -> Iterator[Event]:
        self._append_message(self._normalise(prompt))

        tool_list = as_tools(tools)
        schemas = [item.schema() for item in tool_list] or None
        by_name = {item.name: item for item in tool_list}

        for _ in range(max_tool_rounds + 1):
            content = ""
            reasoning = ""
            pending: Dict[int, Dict[str, str]] = {}
            reasoning_open = False

            for chunk in cast(Iterator[Chunk], self._create(schemas, stream=True)):
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                piece = extract_reasoning(delta)
                if piece:
                    if not reasoning_open:
                        reasoning_open = True
                        yield Event(EventType.THINKING_START, {})
                    reasoning += piece
                    yield Event(EventType.THINKING_DELTA, {"text": piece})

                text = getattr(delta, "content", None)
                if text:
                    # 开始说正文了，思考就结束了（不能等到整个流跑完才收尾，
                    # 否则 THINKING_END 会排在正文后面）
                    if reasoning_open:
                        reasoning_open = False
                        yield Event(EventType.THINKING_END, {"text": reasoning})
                    content += text
                    yield Event(EventType.MESSAGE_DELTA, {"text": text})

                for call in getattr(delta, "tool_calls", None) or []:
                    if reasoning_open:
                        reasoning_open = False
                        yield Event(EventType.THINKING_END, {"text": reasoning})
                    index = getattr(call, "index", 0) or 0
                    slot = pending.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    call_id = getattr(call, "id", None)
                    if call_id:
                        slot["id"] = call_id

                    function = getattr(call, "function", None)
                    name = getattr(function, "name", None)
                    if name and not slot["name"]:
                        slot["name"] = name
                        yield Event(EventType.TOOL_CALL_START, {"name": name})

                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        slot["arguments"] += arguments
                        yield Event(
                            EventType.TOOL_CALL_DELTA,
                            {"name": slot["name"], "text": arguments},
                        )

            if reasoning_open:
                reasoning_open = False
                yield Event(EventType.THINKING_END, {"text": reasoning})

            calls: List[ToolCall] = [
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": slot["name"],
                        "arguments": slot["arguments"],
                    },
                }
                for index, slot in sorted(pending.items())
            ]

            assistant: Message = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = calls
            if reasoning:
                assistant["reasoning_content"] = reasoning
            self._append_message(assistant)

            if not calls:
                yield Event(EventType.MESSAGE_END, {"text": content})
                return

            for call in calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                yield Event(
                    EventType.TOOL_CALL_END, {"name": name, "arguments": arguments}
                )

                reply = self._run_tool(by_name, _SimpleCall(call))
                yield Event(
                    EventType.TOOL_RESULT,
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": reply["content"],
                        "tool_call_id": reply["tool_call_id"],
                    },
                )
                self._append_message(reply)

        raise ToolLoopExceeded(
            f"连续 {max_tool_rounds} 轮工具调用后模型还在要工具，已中止；"
            "检查一下工具是不是一直失败或者互相循环调用"
        )

    # ---------------- 其它 ----------------
    @staticmethod
    def _normalise(prompt: PromptInput) -> Message:
        if isinstance(prompt, dict):
            return {
                "role": cast(MessageRole, prompt.get("role", "user")),
                "content": cast(MessageContent, prompt.get("content", "")),
            }
        return {"role": "user", "content": str(prompt)}

    def rebuild(self) -> None:
        '''按当前的 `start_prompts` + `history` 重建要发给模型的消息列表。

        **不**清空历史。改了 `start_prompts` 之后必须调它 ——
        `messages` 是算好的缓存，只改 `start_prompts` 不会自动生效。
        （`reset()` = 清空历史 + 重建，两者别搞混。）
        '''
        self.messages = self._build_messages()

    def reset(self) -> None:
        '''重置上下文：清空历史并重建消息列表。'''
        self.history = []
        self.rebuild()

    def load_history(self, records: object, *, replace: bool = True) -> int:
        '''把对话历史装回来，返回恢复了几条。

        这是 `export_history` 的反向操作，用来实现「这次退出、下次接着聊」：

            agent.export_history("sessions/a.json")      # 上次
            agent.load_history("sessions/a.json")        # 这次

        接受三种输入：文件路径（`.json` / `.jsonl`）、消息列表、
        或者 `Session` 存出来的整包 dict（取其中的 `messages`）。

        `replace=True` 覆盖现有历史；`False` 追加在后面。
        '''
        messages = normalize_history(records)
        if replace:
            self.history = messages
        else:
            self.history = [*self.history, *messages]
        self.messages = self._build_messages()
        return len(messages)

    def get_history(self) -> List[Message]:
        '''返回上下文（副本，避免外部意外修改）。'''
        return list(self.history)

    def export_history(self, path: str, fmt: str = "json") -> str:
        '''把对话历史导出到文件，返回写入的绝对路径。

        三种格式：

        - `"json"`：一个数组，方便程序读回来
        - `"jsonl"`：一行一条，方便 `grep` / 追加
        - `"markdown"`：给人看的，按角色分节

            agent.export_history("outputs/history.json")
            agent.export_history("outputs/history.md", fmt="markdown")
        '''
        import json
        import os

        if fmt not in ("json", "jsonl", "markdown", "md"):
            raise ValueError(
                f"fmt 只能是 'json' / 'jsonl' / 'markdown'，得到 {fmt!r}"
            )

        full = os.path.abspath(os.path.expanduser(str(path)))
        folder = os.path.dirname(full)
        if folder:
            os.makedirs(folder, exist_ok=True)

        history = self.get_history()
        # newline="\n"：别让 Windows 把 \n 翻成 \r\n，导出的文件要能跨机器比对
        with open(full, "w", encoding="utf-8", newline="\n") as handle:
            if fmt == "json":
                json.dump(history, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            elif fmt == "jsonl":
                for message in history:
                    handle.write(json.dumps(message, ensure_ascii=False) + "\n")
            else:
                handle.write(_history_as_markdown(history))
        return full

    # ====================== 运行模型 ========================
    def run(
        self, tools: Optional[Sequence[ToolLike]] = None, verbose: bool = True
    ) -> None:
        '''命令行对话，输入空行或 Ctrl+C / Ctrl+D 结束。'''
        print("输入空行结束对话。")
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt.strip():
                break

            reply = self.ask(prompt, tools=tools, verbose=verbose)
            # 回答由调用方打印：verbose 只管过程
            print(f"Assistant: {reply}")


def run_prompt(
    agent: Agent,
    prompt: Prompt,
    output_model: Type[BaseModel],
    reset: bool = True,
    tools: Optional[Sequence[ToolLike]] = None,
    **kwargs,
) -> BaseModel:
    '''运行一次「提示词 -> 结构化输出」，返回 output_model 的实例。'''
    executor = Executor(agent=agent)
    if reset:
        agent.reset()
    return executor.run(prompt, output_model, tools=tools, **kwargs)

