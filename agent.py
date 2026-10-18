'''Agent：DeepSeek 客户端封装。

配置（密钥从哪来、用哪个模型）都在 promptkit/config.py；
Agent 只负责「怎么干活」：发请求、跑工具循环、**发事件**。

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
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from openai import OpenAI
from pydantic import BaseModel

from promptkit.config import (
    DEFAULT_BASE_URL,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_START_PROMPTS,
    ENV_FILE,
    ENV_VAR,
    load_env_file,  # 再导出，保持 `from promptkit.agent import load_env_file` 可用
    resolve_api_key,
)
from promptkit.core.display_sys.display import console_display
from promptkit.core.event_sys import Event, EventBus, EventType, extract_reasoning
from promptkit.core.display_sys.render import Verbosity, as_verbosity, render_markdown
from promptkit.core.settings import settings
from promptkit.executor import Executor
from promptkit.prompt import Prompt
from promptkit.tools import Tool, as_tools

# ask() 接受的消息类型：纯文本，或 {"role": ..., "content": ...}
Message = Union[str, Dict[str, str]]

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


def normalize_history(records: Any) -> List[Dict[str, Any]]:
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

    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role not in _HISTORY_ROLES:
            continue
        message: Dict[str, Any] = {"role": role, "content": item.get("content")}
        # 工具调用记录要原样带着，否则续聊时模型会发现「工具结果没有对应的调用」
        if item.get("tool_calls"):
            message["tool_calls"] = item["tool_calls"]
        if item.get("tool_call_id"):
            message["tool_call_id"] = item["tool_call_id"]
        if item.get("name"):
            message["name"] = item["name"]
        cleaned.append(message)
    return cleaned


def _load_history_source(records: Any) -> Any:
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


def _history_as_markdown(history: Sequence[dict]) -> str:
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

    def __init__(self, payload: Dict[str, Any]):
        self.id = payload["id"]
        self.function = _SimpleFunction(payload["function"])


class _SimpleFunction:
    def __init__(self, payload: Dict[str, Any]):
        self.name = payload["name"]
        self.arguments = payload["arguments"]


class Agent:
    '''DeepSeek 客户端。`Agent()` 不带参数即可用：

    密钥按 `api_key 参数 -> 环境变量 DEEPSEEK_API_KEY -> 项目根目录 .env` 的顺序查找。
    实例化时会自动注册为「默认 agent」，所以 `Workflow.run()` 不用再传 agent。

    事件：传 `bus=` 或 `on_event=` 就能看到思考过程 / 工具调用 / 回复。
    `verbose=True` 等价于挂一个 console_display()。
    '''

    _default: Optional["Agent"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        start_prompts: Optional[List[Dict[str, str]]] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        bus: Optional[EventBus] = None,
        on_event: Optional[Any] = None,
        verbose: bool = False,
        register_as_default: bool = True,
    ):
        self.api_key = resolve_api_key(api_key)  # API key

        # 不要用可变对象做默认参数，否则所有实例会共享同一份列表
        self.start_prompts = list(
            DEFAULT_START_PROMPTS if start_prompts is None else start_prompts
        )
        self.model = model
        self.base_url = base_url

        self.bus = bus if bus is not None else EventBus()
        if on_event is not None:
            self.bus.on(on_event)
        if verbose:
            # verbose 只输出「过程」（思考 / 工具调用 / 步骤），写到 stderr；
            # 最终回答是**结果**，留在 stdout —— 两者天然分开。
            self.bus.on(
                console_display(show_message=False, stream=sys.stderr)
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)  # 客户端

        self.history: List[Dict[str, Any]] = []  # 上下文
        self.messages: List[Dict[str, Any]] = self._build_messages()

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
        return f"<Agent model={self.model!r}>"

    def on(self, handler: Any) -> Any:
        '''挂一个事件 handler，返回它（可以当装饰器）。'''
        return self.bus.on(handler)

    def _build_messages(self) -> List[Dict[str, Any]]:
        '''把系统提示和上下文拼成要发送给模型的消息列表。'''
        return [*self.start_prompts, *self.history]

    # ---------------- 请求 ----------------
    def _create(
        self, tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False
    ):
        '''发一次请求。只有传了工具才带 tools 参数。'''
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "extra_body": dict(DEFAULT_EXTRA_BODY),
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)

    def _append_message(self, message: Dict[str, Any]) -> None:
        self.history.append(message)
        self.messages = self._build_messages()

    def _append_assistant(self, assistant: Any) -> Dict[str, Any]:
        '''把助手的回复记进上下文。

        关键是**保留 reasoning_content**：请求里带了 tools 时，DeepSeek 要求
        把之前所有轮次的 reasoning_content 原样传回去，否则会返回 400。
        '''
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(assistant, "content", None) or "",
        }
        tool_calls = getattr(assistant, "tool_calls", None)
        if tool_calls:
            message["tool_calls"] = [
                call.model_dump() if hasattr(call, "model_dump") else dict(call)
                for call in tool_calls
            ]
        reasoning = extract_reasoning(assistant)
        if reasoning:
            message["reasoning_content"] = reasoning

        self._append_message(message)
        return message

    def _run_tool(self, tools_by_name: Dict[str, Tool], call: Any) -> Dict[str, Any]:
        '''执行一次工具调用，把结果包成 role="tool" 的消息。

        工具报错不会中断整个对话，而是把错误文本交给模型，让它自己决定怎么办。
        '''
        function = getattr(call, "function", None)
        name = getattr(function, "name", "") or ""
        raw = getattr(function, "arguments", "") or "{}"
        call_id = getattr(call, "id", None) or name
        reply: Dict[str, Any] = {"role": "tool", "tool_call_id": call_id}

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
        self, bus: EventBus, on_event: Optional[Any], verbose: bool
    ) -> Iterator[EventBus]:
        '''给「这一次调用」临时挂 handler，结束就摘掉。'''
        attached = []
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
        prompt: Message,
        tools: Optional[Sequence[Any]] = None,
        max_tool_rounds: Optional[int] = None,
        bus: Optional[EventBus] = None,
        on_event: Optional[Any] = None,
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
        prompt: Message,
        tools: Optional[Sequence[Any]],
        bus: EventBus,
        max_tool_rounds: int,
    ) -> str:
        self._append_message(self._normalise(prompt))

        tool_list = as_tools(tools)
        schemas = [item.schema() for item in tool_list] or None
        by_name = {item.name: item for item in tool_list}

        for _ in range(max_tool_rounds + 1):
            response = self._create(schemas)
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
                self._append_message(reply)

        raise ToolLoopExceeded(
            f"连续 {max_tool_rounds} 轮工具调用后模型还在要工具，已中止；"
            "检查一下工具是不是一直失败或者互相循环调用"
        )

    @staticmethod
    def _emit_usage(bus: EventBus, response: Any) -> None:
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
        prompt: Message,
        tools: Optional[Sequence[Any]] = None,
        max_tool_rounds: Optional[int] = None,
        on_event: Optional[Any] = None,
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
        handlers = []
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
        prompt: Message,
        tools: Optional[Sequence[Any]],
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

            for chunk in self._create(schemas, stream=True):
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

            calls = [
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

            assistant: Dict[str, Any] = {"role": "assistant", "content": content}
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
    def _normalise(prompt: Message) -> Dict[str, str]:
        if isinstance(prompt, dict):
            return {
                "role": prompt.get("role", "user"),
                "content": prompt.get("content", ""),
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

    def load_history(self, records: Any, *, replace: bool = True) -> int:
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

    def get_history(self) -> List[dict]:
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
    def run(self, tools: Optional[Sequence[Any]] = None, verbose: bool = True) -> None:
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
    tools: Optional[Sequence[Any]] = None,
    **kwargs,
) -> BaseModel:
    '''运行一次「提示词 -> 结构化输出」，返回 output_model 的实例。'''
    executor = Executor(agent=agent)
    if reset:
        agent.reset()
    return executor.run(prompt, output_model, tools=tools, **kwargs)

