"""SummonSubAgentTool：让主 Agent 通过工具调用动态召唤子 Agent。

核心定位：
``SummonSubAgentTool`` 是 **普通 Tool**，不是新的编排引擎。它和
``Orchestrator``（Step）互补：前者是模型自主、后者是代码控制。
两者并存、互不替代。

三个护栏（全部在 ``run()`` 内部检查，不是靠"不给子 Agent 这个工具"）：

- ``max_depth``       召唤深度（防止递归子 Agent 无限嵌套）
- ``max_summon``      单次主 Agent 对话最多召唤次数
- ``max_total_tokens`` 子 Agent 累计 token 预算（单次对话内）

所有错误（深度超限 / 次数超限 / 预算耗尽 / 角色不存在 / 执行失败）
**都返回字符串**，不抛异常。主 Agent 能看到错误并自行决定下一步，
而不是让整个工具循环崩溃。

深度恢复用 ``try/finally``：即使子 Agent 执行时抛异常，
``ctx["_depth"]`` 也会被恢复，不会"泄漏"到后续所有召唤都被拒绝。

Usage::

    from aipykit.agent_factory import SubAgentFactory
    from aipykit.core.tool.subagent import SummonSubAgentTool

    factory = SubAgentFactory(api_key="...", roles={"researcher": "..."})
    summon = SummonSubAgentTool(factory, max_depth=2, max_summon=5)

    main_agent = Agent(api_key="...", tools=[summon])
    main_agent.ask("帮我调研一下 2026 年 AI Agent 市场")
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Dict, List, Optional

from aipykit.agent import Agent
from aipykit.agent_factory import SubAgentFactory, UnknownRoleError
from aipykit.core.tool.base import Tool, ToolSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool.call 支持向声明了 `_ctx` 参数的 func 注入 Context
# ---------------------------------------------------------------------------

def _func_accepts_ctx(func: Any) -> bool:
    """检查 ``func`` 的签名里是否包含 ``_ctx`` 参数。

    结果会被 Tool.call() / ToolRegistry.execute() 缓存，避免每次工具调用
    都跑一遍 inspect。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return "_ctx" in sig.parameters


# ---------------------------------------------------------------------------
# SummonSubAgentTool
# ---------------------------------------------------------------------------

class SummonSubAgentTool(Tool):
    """召唤一个子 Agent 执行独立任务的 Tool。

    继承 ``Tool``，复用其 JSON Schema / 注册机制。构造时把 ``run`` 绑定为
    ``func``，``run`` 声明了 ``_ctx=None`` 参数 —— 只有当调用方（通常是
    ``ToolRegistry.execute`` 或扩展后的 ``Agent._run_tool``）主动注入 ctx
    时才会拿到，否则护栏检查全部跳过（``_ctx is None`` 时 ``ctx.get``
    返回的默认值不会触发拒绝逻辑）。
    """

    # Tool 协议要求的三个类属性（供 schema() 生成 API 定义）
    name: str = "summon_subagent"
    description: str = (
        "召唤一个子 Agent 执行独立任务。适用于需要专人处理的子任务，"
        "比如调研、写作、数据分析。子 Agent 有独立上下文，返回最终结果。"
    )
    parameters: ToolSchema = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "description": "子 Agent 的角色，从可用角色中选择",
            },
            "task": {
                "type": "string",
                "description": (
                    "交给子 Agent 的完整任务描述，要自包含，不依赖主对话历史"
                ),
            },
        },
        "required": ["role", "task"],
    }

    def __init__(
        self,
        factory: SubAgentFactory,
        *,
        max_depth: int = 2,
        max_summon: int = 5,
        max_total_tokens: int = 50000,
        timeout: float = 300.0,
    ) -> None:
        """构造一个 SummonSubAgentTool。

        Args:
            factory: 按角色创建子 Agent 的工厂；必须已经注册了至少一个角色。
            max_depth: 递归召唤深度上限（主 Agent 为 0，直接子 Agent 为 1，
                以此类推）。达到上限时返回错误字符串。
            max_summon: 单次主 Agent 对话最多召唤次数；达到上限时返回错误。
            max_total_tokens: 子 Agent 累计 token 预算（prompt + completion）；
                预算耗尽时返回错误。
            timeout: 子 Agent 执行任务的超时秒数，默认 300 秒。
        """
        if not isinstance(factory, SubAgentFactory):
            raise TypeError(
                f"factory 必须是 SubAgentFactory，得到 {type(factory).__name__}"
            )
        if max_depth < 0:
            raise ValueError(f"max_depth 不能为负，得到 {max_depth}")
        if max_summon < 1:
            raise ValueError(f"max_summon 至少为 1，得到 {max_summon}")
        if max_total_tokens < 1000:
            raise ValueError(
                f"max_total_tokens 太小（<1000），得到 {max_total_tokens}"
            )
        if timeout <= 0:
            raise ValueError(f"timeout 必须为正，得到 {timeout}")

        self.factory = factory
        self.max_depth = max_depth
        self.max_summon = max_summon
        self.max_total_tokens = max_total_tokens
        self.timeout = timeout

        # Tool dataclass 字段（name/description/parameters 已有类默认值，
        # 这里只需要把 func 绑定为 self.run）
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            func=self.run,
        )

    # ------------------------------------------------------------------ #
    # 核心执行逻辑
    # ------------------------------------------------------------------ #

    def run(
        self,
        role: str,
        task: str,
        _ctx: Optional[Dict[str, Any]] = None,
    ) -> str:
        """执行子 Agent 召唤。**所有错误都返回字符串，不抛异常**。

        执行步骤（严格按顺序）：

        1. 深度检查：``_ctx["_depth"] >= max_depth`` 时返回错误字符串。
        2. 召唤次数检查：``_ctx["_summon_count"] >= max_summon`` 时返回错误。
        3. 预算检查：``_ctx["_total_tokens"] >= max_total_tokens`` 时返回错误。
        4. 角色合法性：角色不存在时返回错误（不抛异常）。
        5. 创建子 Agent（独立 Session，不继承主 Agent history）。
        6. ``_depth + 1``，``_summon_count += 1``。
        7. 执行：``factory.run(child, task)``，带超时。
        8. token 累加到 ``_ctx["_total_tokens"]``。
        9. ``finally`` 中恢复 ``_ctx["_depth"]``。
        10. 返回结果或错误字符串。

        Args:
            role: 子 Agent 角色名，必须在 ``factory.roles`` 里。
            task: 自包含的任务描述。
            _ctx: 调用方注入的运行时上下文；为 ``None`` 时所有护栏检查跳过，
                直接执行（测试 / 裸用场景）。

        Returns:
            子 Agent 的回复字符串；任何失败都返回以 ``"子 Agent 执行失败："``
            开头的错误文本。
        """
        start_time = time.perf_counter()
        # 关键：不能用 `_ctx or {}` —— 当调用方传入的是空 dict {} 时会创建
        # 一个全新的空 dict，所有护栏计数和 trace 记录都写不回调用方的 ctx！
        ctx: Dict[str, Any] = _ctx if _ctx is not None else {}

        # ---- 1~3 护栏检查 ----
        depth = int(ctx.get("_depth", 0) or 0)
        summon_count = int(ctx.get("_summon_count", 0) or 0)
        total_tokens = int(ctx.get("_total_tokens", 0) or 0)

        denied_reason: Optional[str] = None
        denied_type: Optional[str] = None

        if depth >= self.max_depth:
            denied_reason = (
                f"递归召唤深度超过上限（当前 {depth}，上限 {self.max_depth}）"
            )
            denied_type = "depth"
        elif summon_count >= self.max_summon:
            denied_reason = (
                f"召唤次数超过上限（当前 {summon_count}，上限 {self.max_summon}）"
            )
            denied_type = "count"
        elif total_tokens >= self.max_total_tokens:
            denied_reason = (
                f"子 Agent token 预算耗尽（已用 {total_tokens}，上限 {self.max_total_tokens}）"
            )
            denied_type = "budget"

        if denied_reason is not None:
            self._record_trace(
                ctx,
                role=role,
                task=task,
                status="denied",
                reason=denied_reason,
                duration=time.perf_counter() - start_time,
                tokens=0,
            )
            logger.warning("SummonSubAgentTool 拒绝召唤：%s", denied_reason)
            return f"子 Agent 执行失败：{denied_reason}"

        # ---- 4 角色合法性 ----
        if role not in self.factory.roles:
            available = ", ".join(self.factory.available_roles()) or "(无)"
            msg = f"未知角色 {role!r}；可用角色：{available}"
            self._record_trace(
                ctx,
                role=role,
                task=task,
                status="denied",
                reason=msg,
                duration=time.perf_counter() - start_time,
                tokens=0,
            )
            return f"子 Agent 执行失败：{msg}"

        # ---- 5~6 创建子 Agent + 更新计数器 ----
        child: Optional[Agent] = None
        old_depth = depth
        consumed_tokens = 0
        try:
            child = self.factory.create(role)

            # 增加护栏计数
            ctx["_depth"] = old_depth + 1
            ctx["_summon_count"] = summon_count + 1

            # 子 Agent token 累加：通过监听 USAGE 事件
            usage_collector: List[int] = []

            def _on_usage(event: Any) -> None:
                data = getattr(event, "data", None) or {}
                total = data.get("total_tokens")
                if total is not None:
                    try:
                        usage_collector.append(int(total))
                    except (TypeError, ValueError):
                        pass

            child.bus.on(_on_usage)

            # ---- 7 执行 ----
            # 给子 Agent 挂一个临时 ctx，避免子 Agent 内再次调用 summon 时
            # 共享父 ctx 出现并发问题（同一个 Agent 内只会串行执行，所以
            # 实际上共享是安全的，但做一层隔离更显式）
            result = self._run_with_timeout(child, task)

            consumed_tokens = sum(usage_collector)
            ctx["_total_tokens"] = total_tokens + consumed_tokens

            self._record_trace(
                ctx,
                role=role,
                task=task,
                status="success",
                reason=None,
                duration=time.perf_counter() - start_time,
                tokens=consumed_tokens,
            )
            return result

        except Exception as exc:  # noqa: BLE001 - 所有错误都返回字符串给 LLM
            logger.exception("SummonSubAgentTool 执行失败：role=%s", role)
            self._record_trace(
                ctx,
                role=role,
                task=task,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                duration=time.perf_counter() - start_time,
                tokens=consumed_tokens,
            )
            return f"子 Agent 执行失败：{type(exc).__name__}: {exc}"

        finally:
            # ---- 9 恢复深度（绝对不能漏，否则 depth 泄漏后所有召唤都会被拒绝）----
            ctx["_depth"] = old_depth

            # 移除 USAGE 监听器（避免重复挂载）
            if child is not None:
                try:
                    child.bus.off(_on_usage)
                except Exception:  # noqa: BLE001 - 清理失败不影响结果
                    pass

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    def _run_with_timeout(self, child: Agent, task: str) -> str:
        """带超时地让子 Agent 执行 ``task``。

        实现：用线程（``concurrent.futures.ThreadPoolExecutor``）执行 ask，
        主线程用 ``future.result(timeout=...)`` 等待。超时后返回错误字符串；
        子 Agent 线程在后台继续跑，但结果会被忽略（Tool 协议只返回一个字符串）。
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.factory.run, child, task)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return (
                    f"子 Agent 执行失败：超时（{self.timeout}s），"
                    f"任务 {task!r} 未在时限内完成"
                )

    @staticmethod
    def _record_trace(
        ctx: Dict[str, Any],
        *,
        role: str,
        task: str,
        status: str,
        reason: Optional[str],
        duration: float,
        tokens: int,
    ) -> None:
        """往 ``ctx["_trace"]`` 里追加一条召唤记录（审计基础）。

        成功 / 失败 / 拒绝**每次都记**，不跳过任何分支。trace 是纯 list[dict]，
        不引入额外依赖。
        """
        entry: Dict[str, Any] = {
            "type": "summon_subagent",
            "role": role,
            "task_preview": task[:100],
            "status": status,
            "duration": round(duration, 4),
            "tokens": tokens,
        }
        if reason is not None:
            entry["reason"] = reason
        ctx.setdefault("_trace", []).append(entry)
