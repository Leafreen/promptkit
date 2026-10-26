'''把一次对话的上下文存到磁盘，下次接着聊。

`Agent.history` 只在内存里 —— 进程一退就没了。想「这次退出、下次无缝衔接」，
需要两件事：**存**（`save`）和**装回来**（`load`）。这个模块就是它俩。

    from promptkit.session import Session

    session = Session("sessions/tutor.json")
    session.load(agent)          # 上次聊到哪，装回来（文件不存在就什么也不做）
    ...                          # 正常跟 agent 聊
    session.save(agent)          # 退出前存一次

更省事的写法（异常退出也会存）：

    with Session("sessions/tutor.json").open(agent) as session:
        agent.run()

或者直接用 `ChatWorkflow`：

    ChatWorkflow(role="导师", session="sessions/tutor.json").run(agent=agent)

存的是什么
----------
一个带版本号的 JSON 对象（`messages` / `system_prompt` / `saved_at` / `turns`）。
读的时候**很宽松**：`export_history` 写的裸数组也认，`Session` 自己的整包也认。
写的时候**原子替换**（先写临时文件再改名），中途崩了不会留下半截文件。
'''
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from promptkit.agent import Agent

__all__ = [
    "Session",
    "SESSION_VERSION",
    "as_session",
    "run_with_session",
    "set_system_prompt_reset",
]

#: 会话文件格式版本
SESSION_VERSION = 1


def _jsonable_variables(variables: Dict[str, Any]) -> Dict[str, Any]:
    '''把要存下来的变量转成能 `json.dump` 的东西。

    转不了（自定义对象之类）就用 `repr` 兜底 —— 会话文件是「尽力保存」，
    为了一个怪对象导致整份会话存不下来不值得。
    '''
    jsonable: Dict[str, Any] = {}
    for name, value in variables.items():
        try:
            json.dumps(value, ensure_ascii=False)
            jsonable[name] = value
        except (TypeError, ValueError):
            jsonable[name] = repr(value)
    return jsonable


def _replace_with_retry(source: str, target: str, attempts: int = 5) -> None:
    '''`os.replace`，Windows 上失败就重试几次。

    Windows 上刚写完关掉的临时文件经常正被杀毒 / 索引服务扫，那一刻改不了名
    （`PermissionError` / WinError 5）。等几十毫秒通常就好了。
    '''
    delay = 0.02
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2


class Session:
    '''一个会话文件：`save` / `load` / `exists` / `clear`。

    同一个 `Session` 对象可以反复用；`save` 会原子替换整个文件。

    `variables` 是「除了对话历史之外还想记住的 Context 变量名」。
    聊天机器人开头常有一段问卷（`Read` 填 `basic_info`），那些答案不在
    `messages` 里，光存历史的话每次启动都会重新问一遍。把它列进来就可以了：

        session = Session("sessions/a.json", variables=["basic_info"])
    '''

    def __init__(
        self,
        path: str,
        *,
        encoding: Optional[str] = None,
        indent: int = 2,
        variables: Sequence[str] = (),
    ):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.encoding = encoding
        self.indent = indent
        #: 要跟着会话一起存的 Context 变量名
        self.variables: Tuple[str, ...] = tuple(str(name) for name in variables)

    # ---------------- 查询 ----------------
    def exists(self) -> bool:
        '''会话文件在不在了。'''
        return os.path.exists(self.path)

    def read(self) -> Dict[str, Any]:
        '''把会话文件读成 dict（不会有副作用）。文件不在就返回空结构。'''
        if not self.exists():
            return {"version": SESSION_VERSION, "messages": [], "turns": 0}
        payload = self._read_raw()
        if isinstance(payload, list):          # export_history 写的裸数组
            return {
                "version": SESSION_VERSION,
                "messages": payload,
                "turns": len(payload) // 2,
            }
        if not isinstance(payload, dict):
            return {"version": SESSION_VERSION, "messages": [], "turns": 0}
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = payload.get("history") or []
        return {**payload, "messages": messages}

    def turns(self) -> int:
        '''会话里大概有几轮对话（一问一答算一轮）。'''
        return int(self.read().get("turns") or 0)

    def load_variables(self) -> Dict[str, Any]:
        '''读回上次存下的 Context 变量（没有就返回空 dict）。'''
        stored = self.read().get("variables")
        return dict(stored) if isinstance(stored, dict) else {}

    # ---------------- 保存 / 恢复 ----------------
    def save(
        self,
        agent: "Agent",
        *,
        variables: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> str:
        '''把 `agent` 的上下文写进文件，返回路径。

        `variables` 是「顺便记下来的 Context 变量」（`basic_info` 这种），
        用 `load_variables()` 读回来。`metadata` 只是方便事后查，
        `load` 时不会用到。
        '''
        from promptkit.agent import normalize_history

        messages = normalize_history(agent.get_history())
        system = ""
        for message in getattr(agent, "start_prompts", []) or []:
            if isinstance(message, dict) and message.get("role") == "system":
                system = message.get("content") or ""
                break

        payload: Dict[str, Any] = {
            "version": SESSION_VERSION,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turns": len(messages) // 2,
            "system_prompt": system,
            "messages": messages,
            **metadata,
        }
        if variables:
            payload["variables"] = _jsonable_variables(variables)

        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # 原子写：先写临时文件再改名。中途崩了不会留下半截 JSON。
        handle, temp = tempfile.mkstemp(
            dir=folder or ".", prefix=".session-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding=self.encoding or "utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=self.indent)
                stream.write("\n")
            _replace_with_retry(temp, self.path)
        except BaseException:
            if os.path.exists(temp):
                os.remove(temp)
            raise
        return self.path

    def load(self, agent: "Agent", *, replace: bool = True) -> int:
        '''把会话里的上下文装回 `agent`，返回恢复了几条。

        文件不存在时**什么也不做**（返回 0，不动 `agent`）——
        第一次跑就是这种情形。而且「没找到东西」不该顺手把 agent 里
        已有的历史清掉：想从头开始请显式 `agent.reset()` 或 `clear()`。

        装完之后 `start_prompts` 要不要跟着改由调用方决定 ——
        `ChatWorkflow` 会用自己的配置重新生成 system prompt。
        '''
        payload = self.read()
        messages = payload.get("messages") or []
        if not messages:
            return 0
        return agent.load_history(messages, replace=replace)

    def clear(self) -> None:
        '''删掉会话文件（想重新开始就把旧会话删了）。'''
        if self.exists():
            os.remove(self.path)

    # ---------------- 便利 ----------------
    @contextmanager
    def open(self, agent: "Agent", **metadata: Any) -> Iterator["Session"]:
        '''先 `load`，退出时 `save`（**异常退出也会存**）。

            with Session("sessions/a.json").open(agent):
                agent.run()
        '''
        self.load(agent)
        try:
            yield self
        finally:
            self.save(agent, **metadata)

    def _read_raw(self) -> Any:
        with open(self.path, encoding=self.encoding or "utf-8") as handle:
            return json.load(handle)

    def __repr__(self) -> str:
        state = "有" if self.exists() else "无"
        return f"<Session {self.path!r} {state}>"


def as_session(value: Union[str, "Session", None]) -> Optional["Session"]:
    '''把 `str` / `Session` / `None` 统一成 `Session`（或 `None`）。'''
    if value is None:
        return None
    if isinstance(value, Session):
        return value
    if isinstance(value, (str, os.PathLike)):
        return Session(os.fspath(value))
    raise TypeError(f"session 只能是路径或 Session，得到 {type(value).__name__}")


def set_system_prompt_reset(workflow: Any, reset: bool) -> None:
    '''把 `reset` 选项写进 workflow 里所有 `SystemPrompt` 步骤。

    续接会话时必须让 `SystemPrompt` **不要**清空历史 —— 否则刚 `load`
    回来的上下文会被它一句 `reset()` 抹掉。
    '''
    from promptkit.core.steps.actions.chat_util import SystemPrompt

    def walk(steps: Any) -> None:
        for step in steps:
            if isinstance(step, SystemPrompt):
                step.reset = reset
            children = getattr(step, "children", None)
            if callable(children):
                walk(children())

    walk(getattr(workflow, "steps", []))


def run_with_session(
    workflow: Any,
    agent: Any = None,
    session: Union[str, "Session", None] = None,
    *,
    reset: Optional[bool] = None,
    verbose: Any = None,
    variables: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    agent_required: str = "run(session=...)",
    **kwargs: Any,
) -> Any:
    '''`load -> 跑 -> 存`（异常也存）。`Workflow.run(session=...)` 走的就是这里。

    顺序很重要：**先决定要不要保留历史，再装历史**。
    反过来的话，`SystemPrompt(reset=True)` 会把刚装回来的清掉。

    `variables` 是要**跟着会话一起存**的 Context 变量名（比如 `["basic_info"]`）。
    存了之后，下次 `workflow.run()` 会以关键字参数把它们塞回 Context ——
    这样开头那段问卷就会读到上次填好的值（当然，问卷本身得用
    `If(Var("basic_info") == {})` 之类的条件跳过）。
    '''
    active = as_session(session)
    if active is None:
        return workflow.run(agent=agent, verbose=verbose, **kwargs)

    if agent is None:
        from promptkit.agent import Agent

        agent = Agent.default()
        if agent is None:
            raise ValueError(
                f"{agent_required} 需要 agent：要么先 `Agent()` 建一个"
                "（会自动注册为默认），要么显式传 `agent=...`。"
            )

    names = tuple(variables) if variables is not None else active.variables

    # 要记的变量：把上次存的值当 kwargs 传给 workflow.run()。
    # `NewVar(...)` 用的是 setdefault，所以传进来的值不会被默认值盖掉。
    seeded: Dict[str, Any] = {}
    if names:
        stored = active.load_variables()
        seeded = {name: stored[name] for name in names if name in stored}

    keep = (active.exists()) if reset is None else (not reset)
    restored = active.load(agent)

    set_system_prompt_reset(workflow, not keep)
    context: Any = None
    try:
        context = workflow.run(
            agent=agent, verbose=verbose, **{**seeded, **kwargs}
        )
        return context
    finally:
        # 异常退出也要存 —— 聊到一半 Ctrl+C 不该把整段对话丢掉
        snapshot: Optional[Dict[str, Any]] = None
        if names and context is not None:
            snapshot = {name: context.get(name) for name in names}
        active.save(
            agent,
            variables=snapshot,
            restored=restored,
            **(metadata or {}),
        )
