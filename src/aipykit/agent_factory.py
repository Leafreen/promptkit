"""SubAgentFactory：按角色创建独立子 Agent。

和主 Agent 完全隔离：新建空 Session、独立 max_turns / max_tool_rounds、
可选角色专属工具。主 Agent 不会把自己的 history 或 system prompt 泄漏
给子 Agent，子 Agent 也不会反向写主 Agent 的上下文。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, cast

from aipykit.agent import Agent
from aipykit.core.tool import Tool, as_tools
from aipykit.types import ToolLike


class SubAgentError(Exception):
    """子 Agent 工厂 / 召唤相关的所有错误的基类。"""


class UnknownRoleError(SubAgentError, KeyError):
    """请求了一个不在 roles 里的角色。"""

    def __init__(self, role: str, available: Sequence[str]) -> None:
        joined = ", ".join(sorted(available)) or "(无)"
        super().__init__(
            f"未知角色 {role!r}；可用角色：{joined}"
        )
        self.role = role
        self.available = list(available)


class SubAgentFactory:
    """按角色模板创建独立子 Agent。

    典型用法::

        factory = SubAgentFactory(
            api_key="sk-xxx",
            roles={
                "researcher": "你是市场调研专家，擅长搜索和整理资料。",
                "coder":      "你是 Python 工程师，擅长阅读和修改代码。",
            },
            default_tools=[WEB_SEARCH],
            role_tools={"coder": [READ_FILE, WRITE_FILE]},
        )
        child = factory.create("researcher")
        child.ask("帮我查一下 2026 年 AI Agent 市场规模")

    子 Agent 的 Session 是全新的空 Session，不继承主 Agent 的 history。
    """

    def __init__(
        self,
        api_key: str,
        roles: Dict[str, str],
        *,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        default_tools: Optional[Sequence[ToolLike]] = None,
        role_tools: Optional[Dict[str, Sequence[ToolLike]]] = None,
    ) -> None:
        """初始化子 Agent 工厂。

        Args:
            api_key: 调用模型的 API 密钥。
            roles: 角色名到 system prompt 的映射；key 必须非空。
            base_url: 可选的厂商端点；不传则由 Agent 自己按 vendor 推断。
            model: 子 Agent 统一使用的模型名，默认 ``deepseek-chat``。
            default_tools: 所有子 Agent 默认挂载的工具。
            role_tools: 按角色覆盖的工具列表；优先级高于 default_tools，
                但 ``create(tools=...)`` 显式传入的优先级最高。
        """
        if not roles:
            raise ValueError("SubAgentFactory 至少要声明一个角色")
        for name, prompt in roles.items():
            if not name or not isinstance(name, str):
                raise ValueError(f"角色名必须是非空字符串，得到 {name!r}")
            if not prompt or not isinstance(prompt, str):
                raise ValueError(f"角色 {name!r} 的 system prompt 必须是非空字符串")

        self.api_key = api_key
        self.roles: Dict[str, str] = dict(roles)
        self.base_url = base_url
        self.model = model
        self.default_tools: List[Tool] = as_tools(default_tools)
        self.role_tools: Dict[str, List[Tool]] = {
            role: as_tools(tools) for role, tools in (role_tools or {}).items()
        }

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #

    def available_roles(self) -> List[str]:
        """返回所有已注册的角色名（按字母排序）。"""
        return sorted(self.roles)

    def create(
        self,
        role: str,
        *,
        tools: Optional[Sequence[ToolLike]] = None,
        max_turns: int = 10,
        max_tool_calls: int = 20,
    ) -> Agent:
        """创建一个子 Agent。

        子 Agent 有独立的空 Session，不继承主 Agent 的 history；
        不会注册为默认 Agent（``register_as_default=False``），避免把主 Agent 覆盖掉。

        工具优先级从高到低：
            ``create(tools=...)`` > ``role_tools[role]`` > ``default_tools``

        Args:
            role: 目标角色名，必须在 ``roles`` 里。
            tools: 显式覆盖的工具列表；传了就完全替换掉角色 / 默认工具。
            max_turns: 子 Agent 单次对话最多回答轮数（仅用于语义提示，
                不强制中断；由调用方自行约束）。
            max_tool_calls: 子 Agent 单次对话最多工具调用轮数（对应 Agent
                的 ``max_tool_rounds``）。

        Returns:
            一个全新的、独立的 Agent 实例。

        Raises:
            UnknownRoleError: ``role`` 不在已注册角色里。
        """
        if role not in self.roles:
            raise UnknownRoleError(role, self.available_roles())

        # ---- 决定工具集：显式传入 > 角色专属 > 默认 ----
        if tools is not None:
            resolved_tools = as_tools(tools)
        elif role in self.role_tools:
            resolved_tools = list(self.role_tools[role])
        else:
            resolved_tools = list(self.default_tools)

        # ---- 构造 start_prompts：只有一条 system prompt ----
        start_prompts = [
            {"role": "system", "content": self.roles[role]},
        ]

        child = Agent(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            start_prompts=start_prompts,
            register_as_default=False,
        )
        # 子 Agent 不能再召唤子 Agent（否则递归深度无限）。我们在 SummonSubAgentTool
        # 内部做了深度护栏，但这里额外保证"不显式传工具时不会意外继承主 Agent 的
        # SummonSubAgentTool"。如果调用方确实想让子 Agent 也能召唤，得显式传进来。
        child._summon_tools = resolved_tools
        child._max_tool_rounds = max_tool_calls
        child._role_name = role
        child._max_turns = max_turns
        return child

    def run(
        self,
        agent: Agent,
        task: str,
    ) -> str:
        """让子 Agent 执行一个任务，返回最终回复。

        这是 ``SummonSubAgentTool`` 内部会调用的便捷方法，
        也可以直接在测试里用。
        """
        tools = getattr(agent, "_summon_tools", None)
        max_rounds = getattr(agent, "_max_tool_rounds", None)
        return agent.ask(
            task,
            tools=tools,
            max_tool_rounds=max_rounds,
        )


# 本文件职责：SubAgentFactory 按角色创建独立子 Agent，角色不存在时抛 UnknownRoleError
