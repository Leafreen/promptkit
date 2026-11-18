"""SubAgentFactory 单元测试：未知角色、隔离 Session、工具优先级。"""
from __future__ import annotations

import pytest

from aipykit.agent_factory import (
    SubAgentError,
    SubAgentFactory,
    UnknownRoleError,
)
from aipykit.agent import Agent
from aipykit.core.tool import Tool


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
        model="deepseek-chat",
    )


@pytest.fixture()
def factory_with_tools() -> SubAgentFactory:
    """带默认工具 + 角色专属工具的工厂。"""
    default_tool = Tool(
        name="default_tool",
        description="默认工具",
        parameters={"type": "object", "properties": {}},
        func=lambda: "default",
    )
    role_tool = Tool(
        name="role_tool",
        description="角色专属",
        parameters={"type": "object", "properties": {}},
        func=lambda: "role",
    )
    return SubAgentFactory(
        api_key="sk-test",
        roles={"researcher": "调研专家。", "coder": "工程师。"},
        default_tools=[default_tool],
        role_tools={"coder": [role_tool]},
    )


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------

def test_available_roles_sorted(factory: SubAgentFactory) -> None:
    """available_roles 返回字母排序后的角色名列表。"""
    assert factory.available_roles() == ["analyst", "researcher"]


def test_create_returns_isolated_session(factory: SubAgentFactory) -> None:
    """子 Agent 的 Session 是全新空 Session，不继承主 Agent。"""
    main_agent = Agent(api_key="sk-test", register_as_default=False)
    main_agent.history.append({"role": "user", "content": "主对话"})

    child = factory.create("researcher")
    # 子 Agent 不能是同一个实例
    assert child is not main_agent
    # 子 Agent 的 Session 不能跟主 Agent 共享
    assert child.session is not main_agent.session
    # 子 Agent history 必须是空（只有 system prompt 不算 history）
    assert len(child.history) == 0
    # 子 Agent 的 start_prompts 只有一条 system
    assert len(child.start_prompts) == 1
    assert child.start_prompts[0]["role"] == "system"
    assert child.start_prompts[0]["content"] == "你是市场调研专家。"


def test_create_unknown_role_raises_with_hint(factory: SubAgentFactory) -> None:
    """未知角色抛 UnknownRoleError，hint 列出可用角色。"""
    with pytest.raises(UnknownRoleError) as exc_info:
        factory.create("nonexistent")
    err = exc_info.value
    assert err.role == "nonexistent"
    assert set(err.available) == {"analyst", "researcher"}
    # 提示字符串里必须带可用角色（用户能直接看懂）
    msg = str(err)
    assert "analyst" in msg
    assert "researcher" in msg
    # UnknownRoleError 必须是 SubAgentError 的子类（方便整体捕获）
    assert isinstance(err, SubAgentError)


def test_factory_rejects_empty_roles() -> None:
    """初始化时 roles 为空直接抛 ValueError，而不是运行时 UnknownRoleError。"""
    with pytest.raises(ValueError):
        SubAgentFactory(api_key="sk-test", roles={})


# ---------------------------------------------------------------------------
# 工具优先级：显式 > 角色专属 > 默认
# ---------------------------------------------------------------------------

def test_create_role_tools_override_defaults(
    factory_with_tools: SubAgentFactory,
) -> None:
    """role_tools[role] 覆盖 default_tools。"""
    child = factory_with_tools.create("coder")
    tool_names = [t.name for t in child._summon_tools]
    assert "role_tool" in tool_names
    assert "default_tool" not in tool_names


def test_create_role_falls_back_to_defaults(
    factory_with_tools: SubAgentFactory,
) -> None:
    """角色不在 role_tools 里时用 default_tools。"""
    child = factory_with_tools.create("researcher")
    tool_names = [t.name for t in child._summon_tools]
    assert "default_tool" in tool_names


def test_create_explicit_tools_override_all(
    factory_with_tools: SubAgentFactory,
) -> None:
    """create(tools=...) 显式传入的优先级最高。"""
    explicit = Tool(
        name="explicit",
        description="显式覆盖",
        parameters={"type": "object", "properties": {}},
        func=lambda: "explicit",
    )
    child = factory_with_tools.create("coder", tools=[explicit])
    tool_names = [t.name for t in child._summon_tools]
    assert tool_names == ["explicit"]


# ---------------------------------------------------------------------------
# 子 Agent 配置独立
# ---------------------------------------------------------------------------

def test_child_has_custom_max_tool_rounds(factory: SubAgentFactory) -> None:
    """子 Agent 的 max_tool_rounds 和主 Agent 无关。"""
    child = factory.create("researcher", max_tool_calls=3, max_turns=5)
    assert child._max_tool_rounds == 3
    assert child._max_turns == 5
    assert child._role_name == "researcher"
