"""SummonSubAgentTool 单元测试：护栏检查、深度恢复、token 累加、trace 记录。"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aipykit.agent import Agent
from aipykit.agent_factory import SubAgentFactory
from aipykit.core.tool import Tool, as_tools
from aipykit.core.tool.base import _func_has_ctx_param
from aipykit.core.tool.registry import ToolRegistry
from aipykit.core.tool.subagent import SummonSubAgentTool


# ---------------------------------------------------------------------------
# 共享夹具
# ---------------------------------------------------------------------------

@pytest.fixture()
def factory() -> SubAgentFactory:
    return SubAgentFactory(
        api_key="sk-test",
        roles={
            "researcher": "你是市场调研专家。",
            "analyst": "你是数据分析师。",
        },
    )


@pytest.fixture()
def summon(factory: SubAgentFactory) -> SummonSubAgentTool:
    return SummonSubAgentTool(
        factory,
        max_depth=2,
        max_summon=5,
        max_total_tokens=50000,
        timeout=2.0,  # 测试里短一点
    )


@pytest.fixture()
def ctx() -> dict[str, Any]:
    """每次测试拿一份新的干净 ctx。"""
    return {}


def _fake_agent(reply: str) -> Agent:
    """构造一个不连网的 Agent（FakeClient 给固定回复）。"""
    from tests.conversation.fake_client import make_agent

    agent = make_agent([reply])
    return agent


# ---------------------------------------------------------------------------
# Tool 协议 + _ctx 注入
# ---------------------------------------------------------------------------

def test_tool_subclass_has_correct_schema(summon: SummonSubAgentTool) -> None:
    """继承 Tool 后 name / description / parameters 必须被 schema() 正确导出。"""
    assert summon.name == "summon_subagent"
    assert summon.description
    assert summon.parameters["required"] == ["role", "task"]
    schema = summon.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "summon_subagent"


def test_func_accepts_ctx_detected(summon: SummonSubAgentTool) -> None:
    """SummonSubAgentTool.run 必须被检测为需要 _ctx。"""
    # 先清掉可能残留的缓存（不同测试间可能被污染）
    if hasattr(summon.func, "__promptkit_has_ctx__"):
        delattr(summon.func, "__promptkit_has_ctx__")
    assert _func_has_ctx_param(summon.func) is True


def test_plain_tool_does_not_accept_ctx() -> None:
    """普通 lambda / Tool.func 不会被误认为需要 _ctx。"""
    t = Tool(
        name="plain",
        description="plain",
        parameters={"type": "object", "properties": {}},
        func=lambda x: x,
    )
    if hasattr(t.func, "__promptkit_has_ctx__"):
        delattr(t.func, "__promptkit_has_ctx__")
    assert _func_has_ctx_param(t.func) is False


def test_tool_call_ctx_ignored_for_plain_tool() -> None:
    """给不需要 _ctx 的 Tool 传 ctx 必须零副作用。"""
    captured: dict[str, Any] = {}

    def my_tool(x: str) -> str:
        captured["x"] = x
        return "ok"

    t = Tool(
        name="my",
        description="",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        func=my_tool,
    )
    # 即使 ctx 里塞了 _ctx 键，也不会误传给 my_tool
    result = t.call({"x": "hello"}, ctx={"_ctx": "should_not_leak"})
    assert result == "ok"
    assert captured == {"x": "hello"}


def test_tool_registry_execute_injects_ctx(summon: SummonSubAgentTool) -> None:
    """ToolRegistry.execute 对声明了 _ctx 的 Tool 会注入运行时上下文。"""
    # 用一个 mock func 记录被调用时的 kwargs
    captured: dict[str, Any] = {}

    def recorder(role: str, task: str, _ctx: dict | None = None) -> str:
        captured["role"] = role
        captured["task"] = task
        captured["_ctx"] = _ctx
        return "ok"

    t = Tool(
        name="recorder",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "task": {"type": "string"},
            },
            "required": ["role", "task"],
        },
        func=recorder,
    )
    # 清缓存
    if hasattr(recorder, "__promptkit_has_ctx__"):
        delattr(recorder, "__promptkit_has_ctx__")

    registry = ToolRegistry([t])
    fake_ctx = {"_depth": 1, "_summon_count": 3}
    result = registry.execute("recorder", {"role": "a", "task": "b"}, ctx=fake_ctx)

    assert result == "ok"
    assert captured["role"] == "a"
    assert captured["task"] == "b"
    assert captured["_ctx"] is fake_ctx


def test_tool_registry_execute_no_ctx_for_plain_tool() -> None:
    """registry.execute 对不需要 _ctx 的 Tool 不注入。"""
    captured: dict[str, Any] = {}

    def plain(x: str) -> str:
        captured["kwargs"] = {k: v for k, v in locals().items() if k != "self"}
        return x

    t = Tool(
        name="plain",
        description="",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        func=plain,
    )
    if hasattr(plain, "__promptkit_has_ctx__"):
        delattr(plain, "__promptkit_has_ctx__")

    registry = ToolRegistry([t])
    result = registry.execute("plain", {"x": "hi"}, ctx={"_depth": 5})
    assert result == "hi"


# ---------------------------------------------------------------------------
# 护栏检查
# ---------------------------------------------------------------------------

def test_depth_exceeded_returns_error_string(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """_depth 已经到 max_depth 时返回错误字符串，不抛异常。"""
    ctx["_depth"] = summon.max_depth  # == 2

    # 用一个 mock factory.create 确保不会被调用
    with patch.object(summon.factory, "create") as mock_create:
        result = summon.run("researcher", "帮我调研", _ctx=ctx)
        # 必须是错误字符串，不能抛异常
        assert isinstance(result, str)
        assert "深度" in result
        assert mock_create.called is False


def test_summon_count_exceeded_returns_error_string(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """_summon_count 已达 max_summon 时返回错误字符串。"""
    ctx["_summon_count"] = summon.max_summon  # == 5

    with patch.object(summon.factory, "create") as mock_create:
        result = summon.run("researcher", "调研", _ctx=ctx)
        assert isinstance(result, str)
        assert "次数" in result
        assert mock_create.called is False


def test_budget_exceeded_returns_error_string(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """_total_tokens 已达预算上限时返回错误字符串。"""
    ctx["_total_tokens"] = summon.max_total_tokens + 100

    with patch.object(summon.factory, "create") as mock_create:
        result = summon.run("researcher", "调研", _ctx=ctx)
        assert isinstance(result, str)
        assert "预算" in result
        assert mock_create.called is False


def test_unknown_role_returns_error_string_not_raised(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """未知角色返回错误字符串，而不是抛 UnknownRoleError。"""
    # 必须不抛异常
    result = summon.run("ghost", "调研", _ctx=ctx)
    assert isinstance(result, str)
    assert "子 Agent 执行失败" in result
    assert "可用角色" in result


def test_no_ctx_passes_without_guards(
    summon: SummonSubAgentTool, factory: SubAgentFactory
) -> None:
    """_ctx=None 时护栏检查跳过（ctx.get 返回默认值），直接执行。"""
    # 用 mock factory.run 直接返回结果
    with patch.object(factory, "run", return_value="直接执行") as mock_run:
        result = summon.run("researcher", "调研", _ctx=None)
        assert result == "直接执行"
        # 因为 _ctx=None，护栏没检查
        # 但也意味着 _summon_count 不会累加（这是有意的：裸用不计数）


# ---------------------------------------------------------------------------
# 深度恢复（关键约束：finally 必须执行）
# ---------------------------------------------------------------------------

def test_depth_restored_on_success(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """成功执行后 _depth 恢复到原值。"""
    ctx["_depth"] = 0
    ctx["_summon_count"] = 0

    child = _fake_agent("child result")
    # 让 factory.create 返回这个 child
    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(summon.factory, "run", return_value="child result"):
            summon.run("researcher", "调研", _ctx=ctx)

    assert ctx["_depth"] == 0, "深度必须恢复"
    assert ctx["_summon_count"] == 1


def test_depth_restored_on_failure(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """子 Agent 抛异常后 _depth 必须恢复（finally 关键路径）。"""
    ctx["_depth"] = 1
    ctx["_summon_count"] = 2

    child = _fake_agent("will_fail")
    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(
            summon.factory, "run", side_effect=RuntimeError("boom")
        ):
            result = summon.run("researcher", "调研", _ctx=ctx)

    assert "子 Agent 执行失败" in result
    # 深度必须恢复到调用前的 1，不能变成 2（泄漏）
    assert ctx["_depth"] == 1, (
        f"深度泄漏：调用前 1，调用后变成 {ctx['_depth']}"
    )
    # 次数仍然累加（失败也算一次召唤）
    assert ctx["_summon_count"] == 3


def test_depth_restored_on_timeout(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """超时场景下深度也必须恢复。"""
    ctx["_depth"] = 0

    child = _fake_agent("slow")

    def _slow_run(*args, **kwargs):
        time.sleep(0.5)
        return "late"

    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(summon.factory, "run", side_effect=_slow_run):
            summon.run("researcher", "调研", _ctx=ctx)

    # 超时或不超时，深度都必须恢复
    assert ctx["_depth"] == 0


# ---------------------------------------------------------------------------
# token 累加
# ---------------------------------------------------------------------------

def test_tokens_accumulate_via_usage_event(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """子 Agent 发出 USAGE 事件后，token 累加到 ctx["_total_tokens"]。"""
    ctx["_depth"] = 0
    ctx["_summon_count"] = 0
    ctx["_total_tokens"] = 100

    from promptkit.core.event_sys import Event, EventType

    child = _fake_agent("ok")

    # 在 ask 执行时模拟发 USAGE 事件
    original_ask = child.ask

    def patched_ask(*args, **kwargs):
        child.bus.emit(Event(EventType.USAGE, {"total_tokens": 500}))
        return original_ask(*args, **kwargs)

    with patch.object(child, "ask", side_effect=patched_ask):
        with patch.object(summon.factory, "create", return_value=child):
            summon.run("researcher", "调研", _ctx=ctx)

    assert ctx["_total_tokens"] == 600  # 100 + 500


# ---------------------------------------------------------------------------
# summon_count 累加
# ---------------------------------------------------------------------------

def test_summon_count_increments_on_every_call(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """每次调用 _summon_count 都 +1，不管成功失败。"""
    ctx["_depth"] = 0
    ctx["_summon_count"] = 0

    child = _fake_agent("ok")
    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(summon.factory, "run", return_value="ok"):
            summon.run("researcher", "a", _ctx=ctx)
            summon.run("analyst", "b", _ctx=ctx)

    assert ctx["_summon_count"] == 2


# ---------------------------------------------------------------------------
# trace 记录
# ---------------------------------------------------------------------------

def test_trace_recorded_on_success(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """成功召唤时 trace 里有一条 success 记录。"""
    child = _fake_agent("ok")
    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(summon.factory, "run", return_value="ok"):
            summon.run("researcher", "调研任务", _ctx=ctx)

    traces = ctx.get("_trace", [])
    assert len(traces) == 1
    entry = traces[0]
    assert entry["type"] == "summon_subagent"
    assert entry["role"] == "researcher"
    assert entry["status"] == "success"
    assert entry["task_preview"] == "调研任务"[:100]
    assert entry["tokens"] == 0  # FakeClient 没有发 USAGE
    assert entry["duration"] >= 0
    assert "reason" not in entry


def test_trace_recorded_on_failure(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """失败召唤时 trace 里有一条 failed 记录。"""
    child = _fake_agent("fail")
    with patch.object(summon.factory, "create", return_value=child):
        with patch.object(summon.factory, "run", side_effect=ValueError("bad")):
            summon.run("researcher", "调研", _ctx=ctx)

    traces = ctx["_trace"]
    assert len(traces) == 1
    entry = traces[0]
    assert entry["status"] == "failed"
    assert "reason" in entry
    assert "ValueError" in entry["reason"]


def test_trace_recorded_on_denied(
    summon: SummonSubAgentTool, ctx: dict[str, Any]
) -> None:
    """被护栏拒绝时 trace 里有一条 denied 记录。"""
    ctx["_depth"] = summon.max_depth

    summon.run("researcher", "调研", _ctx=ctx)

    traces = ctx["_trace"]
    assert len(traces) == 1
    assert traces[0]["status"] == "denied"
    assert "reason" in traces[0]


# ---------------------------------------------------------------------------
# 构造参数校验
# ---------------------------------------------------------------------------

def test_constructor_rejects_bad_args(factory: SubAgentFactory) -> None:
    with pytest.raises(ValueError):
        SummonSubAgentTool(factory, max_depth=-1)
    with pytest.raises(ValueError):
        SummonSubAgentTool(factory, max_summon=0)
    with pytest.raises(ValueError):
        SummonSubAgentTool(factory, timeout=0)


def test_constructor_rejects_non_factory() -> None:
    with pytest.raises(TypeError):
        SummonSubAgentTool("not a factory")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 兜底：Agent._run_tool 集成 ctx 注入
# ---------------------------------------------------------------------------

def test_agent_run_tool_passes_ctx_to_aware_tool() -> None:
    """真实 Agent._run_tool（非 ToolRegistry.execute）也能向声明了 _ctx 的
    Tool 注入 Agent._ctx —— 这是 Agent 直接调 tool.call() 的路径。"""
    captured: dict[str, Any] = {}

    def aware_func(x: str, _ctx: dict | None = None) -> str:
        captured["x"] = x
        captured["_ctx"] = _ctx
        return "ok"

    tool = Tool(
        name="aware",
        description="",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        func=aware_func,
    )
    # 清缓存
    if hasattr(aware_func, "__promptkit_has_ctx__"):
        delattr(aware_func, "__promptkit_has_ctx__")

    agent = Agent(api_key="sk-test", register_as_default=False)
    fake_ctx = {"_depth": 2, "_summon_count": 3}
    agent._ctx = fake_ctx

    # 直接调 _run_tool —— 模拟工具循环里 Agent 的调用路径
    fake_call = SimpleNamespace(
        function=SimpleNamespace(name="aware", arguments='{"x": "hello"}'),
        id="call_1",
    )
    tools_by_name = {"aware": tool}
    reply = agent._run_tool(tools_by_name, fake_call)

    assert reply["content"] == "ok"
    assert captured["x"] == "hello"
    assert captured["_ctx"] is fake_ctx


def test_agent_run_tool_still_works_for_plain_tool() -> None:
    """Agent._run_tool 给普通 Tool 传 ctx=None 时零副作用。"""
    plain = Tool(
        name="plain",
        description="",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        func=lambda x: x.upper(),
    )

    agent = Agent(api_key="sk-test", register_as_default=False)
    agent._ctx = {"_depth": 5}  # 有 ctx 但 func 不接收

    fake_call = SimpleNamespace(
        function=SimpleNamespace(name="plain", arguments='{"x": "hi"}'),
        id="call_1",
    )
    reply = agent._run_tool({"plain": plain}, fake_call)
    assert reply["content"] == "HI"  # 工具正常执行
