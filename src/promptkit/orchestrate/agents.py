# orchestrate/agents.py
'''子代理的造法与「结构化回答」的小工具。

隔离原则在这里落地：**每个子任务一个 Agent 实例**。`Agent` 的 `history`
是实例属性，所以「新造一个」就是最干净的隔离；不想造新的就用
`OrchestratorConfig(isolate=True)`（默认），每个任务前 `agent.reset()`。
'''
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Type


__all__ = ["sub_agent", "RoleFactory", "structured_answer", "as_text"]


def sub_agent(
    system: str,
    *,
    model: Optional[str] = None,
    vendor: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    '''按一段 system 说明造一个**独立**的子代理。

    它有自己的 `history`，不继承父 Agent 的对话；`register_as_default=False`
    也不会抢走默认 agent。

        researcher = sub_agent("你是市场调研专家，只给事实和数据来源。")
    '''
    if not str(system or "").strip():
        raise ValueError("sub_agent 需要一段非空的 system 说明，否则它不知道自己要干嘛")
    from promptkit.agent import Agent

    return Agent(
        api_key=api_key,
        start_prompts=[{"role": "system", "content": str(system).strip()}],
        model=model,
        vendor=vendor,
        base_url=base_url,
        params=params,
        timeout=timeout,
        max_tokens=max_tokens,
        register_as_default=False,
    )


class RoleFactory:
    '''按任务的角色名造子代理：`ExecutePlan(agent_factory=RoleFactory({...}))`。

        factory = RoleFactory({
            "researcher": "你是市场调研专家……",
            "writer": "你是资深报告撰写者……",
        })
        Task("research", "……", role="researcher")

    `roles` 里没有的角色：有 `default_system` 就用它，否则抛 `KeyError`
    （显式失败，别让任务悄悄用错角色）。
    '''

    def __init__(
        self,
        roles: Mapping[str, str],
        *,
        default_system: Optional[str] = None,
        **agent_kwargs: Any,
    ):
        self.roles = {str(key): str(value) for key, value in dict(roles).items()}
        self.default_system = default_system
        self.agent_kwargs = dict(agent_kwargs)

    def __call__(self, task: Any) -> Any:
        role = str(getattr(task, "role", "") or "").strip()
        system = self.roles.get(role) or self.default_system
        if system is None:
            known = ", ".join(sorted(self.roles)) or "（空）"
            raise KeyError(
                f"任务 {getattr(task, 'name', '?')!r} 要的角色 {role!r} 没有定义；"
                f"已定义：{known}"
            )
        return sub_agent(system, **self.agent_kwargs)


def structured_answer(
    agent: Any,
    text: str,
    model: Type[Any],
    *,
    tools: Optional[Sequence[Any]] = None,
    bus: Any = None,
    max_retries: Optional[int] = None,
) -> Any:
    '''把 `text` 发给 agent，要求它输出 `model` 的结构化结果。

    为什么不用 `Executor`：`Executor` 会先 `Prompt.render()` 一遍，而这里
    的 `text` 里可能带着上一个任务的输出（里面有 `{{...}}`），会被当成模板
    变量去解析、抛 `KeyError`。编排层的输入是**数据**，不该二次渲染。

    实现委托给 `promptkit.executor.run_structured`（技能层用的是同一个）。
    '''
    from promptkit.executor import run_structured

    return run_structured(
        agent, text, model, tools=tools, bus=bus, max_retries=max_retries
    )


def as_text(value: Any) -> str:
    '''把上下文里的值变成本文（dict / list 用 JSON，一眼能看出结构）。'''
    if isinstance(value, str):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            value = dump()
        except Exception:      # noqa: BLE001
            return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)
