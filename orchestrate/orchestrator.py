# orchestrate/orchestrator.py
'''`Orchestrator`：按依赖顺序跑任务、汇总结果、处理失败、守住预算。

它**只执行**：任务从哪来（模板 / LLM 规划）不归它管，怎么规划在
`orchestrate.planner`。
'''
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from promptkit.core.event_sys import EventBus, EventType
from promptkit.orchestrate.agents import as_text, structured_answer
from promptkit.orchestrate.errors import PlanError, TaskError
from promptkit.orchestrate.task import Plan, Task, TaskResult
from promptkit.prompt import render_text

__all__ = ["OrchestratorConfig", "Orchestrator", "PlanResult"]


@dataclass
class OrchestratorConfig:
    '''编排器的护栏。默认值都很保守：串行、10 个任务、不限预算。'''

    #: 同时最多跑几个任务（同一层里才有意义）
    max_concurrency: int = 1
    #: 整个计划最多花多少 token（None = 不限）。
    #: 并发时可能超出最多 `max_concurrency - 1` 个任务的用量（它们同时开跑）。
    max_total_tokens: Optional[int] = None
    #: 最多几个任务（防止 LLM 一次规划出 50 个）
    max_tasks: int = 10
    #: 单个任务的默认超时（秒）；任务自己的 `timeout` 优先
    per_task_timeout: Optional[float] = None
    #: True = 任何一个任务失败就终止整个计划（比任务级 abort 更严）
    stop_on_error: bool = False
    #: True = 每个任务开始前把 agent 的历史清掉（上下文隔离的兜底）
    isolate: bool = True

    def __post_init__(self) -> None:
        self.max_concurrency = int(self.max_concurrency)
        self.max_tasks = int(self.max_tasks)
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency 至少是 1")
        if self.max_tasks < 1:
            raise ValueError("max_tasks 至少是 1")
        if self.max_total_tokens is not None and int(self.max_total_tokens) <= 0:
            raise ValueError("max_total_tokens 要么不设，要么大于 0")
        if self.per_task_timeout is not None and float(self.per_task_timeout) <= 0:
            raise ValueError("per_task_timeout 要么不设，要么大于 0")


@dataclass
class PlanResult:
    '''整个计划跑完之后的结果。'''

    task: str = ""
    results: List[TaskResult] = field(default_factory=list)
    elapsed: float = 0.0

    # ---------------- 聚合视图 ----------------
    @property
    def ok(self) -> bool:
        '''全部成功才算成功（跳过也算没成功）。'''
        return bool(self.results) and all(item.ok for item in self.results)

    @property
    def succeeded(self) -> List[TaskResult]:
        return [item for item in self.results if item.ok]

    @property
    def failed(self) -> List[TaskResult]:
        return [item for item in self.results if item.status == "failed"]

    @property
    def skipped(self) -> List[TaskResult]:
        return [item for item in self.results if item.status == "skipped"]

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.results)

    def by_name(self, name: str) -> Optional[TaskResult]:
        for item in self.results:
            if item.name == name:
                return item
        return None

    def to_context(self) -> Dict[str, Any]:
        '''成功任务的结果，按 `output_key`（默认任务名）摊平成一个 dict。'''
        return {
            (item.output_key or item.name): item.output
            for item in self.results
            if item.ok
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "ok": self.ok,
            "elapsed": round(self.elapsed, 3),
            "tokens": self.tokens,
            "results": [item.to_dict() for item in self.results],
        }

    def summary(self) -> str:
        '''一段人能看、模型也能看的汇总。'''
        lines = [
            f"共 {len(self.results)} 个任务：成功 {len(self.succeeded)} / "
            f"失败 {len(self.failed)} / 跳过 {len(self.skipped)}，"
            f"用时 {self.elapsed:.1f}s，tokens {self.tokens}"
        ]
        for item in self.results:
            mark = {"success": "✓", "failed": "✗", "skipped": "·"}.get(item.status, "?")
            detail = f"{item.latency:.1f}s"
            if item.tokens:
                detail += f", {item.tokens} tokens"
            if item.attempts > 1:
                detail += f", 试了 {item.attempts} 次"
            line = f"  {mark} {item.name} ({detail})"
            if item.error:
                line += f"：{_short(item.error)}"
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<PlanResult ok={self.ok} 成功 {len(self.succeeded)}/{len(self.results)} "
            f"{self.elapsed:.1f}s>"
        )


class Orchestrator:
    '''执行一个 `Plan`。

        orchestrator = Orchestrator(
            OrchestratorConfig(max_concurrency=2, max_total_tokens=30000),
            agent_factory=RoleFactory({"researcher": "……"}),
        )
        result = orchestrator.run(plan, context={"usr_input": "……"})
    '''

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        *,
        agent: Any = None,
        agent_factory: Optional[Callable[[Task], Any]] = None,
        on_result: Optional[Callable[[Task, TaskResult], None]] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.agent = agent
        self.agent_factory = agent_factory
        self.on_result = on_result

    # ---------------- 入口 ----------------
    def run(self, plan: Any, context: Optional[Mapping[str, Any]] = None) -> PlanResult:
        plan = self._coerce(plan)
        config = self.config
        if len(plan) > config.max_tasks:
            raise PlanError(
                f"计划有 {len(plan)} 个任务，超过 max_tasks={config.max_tasks}；"
                "要么拆小一点，要么调大上限"
            )

        available: Dict[str, Any] = dict(context or {})
        by_name: Dict[str, TaskResult] = {}
        # 自己攒 token：任务结果里带着用量，一层跑完再加总
        self._spent = 0
        started = time.perf_counter()
        aborted: Optional[Task] = None

        for layer in plan.waves():
            runnable: List[Task] = []
            for task in layer:
                reason = self._skip_reason(task, by_name)
                if reason:
                    by_name[task.name] = self._skipped(task, reason)
                else:
                    runnable.append(task)

            if runnable:
                pairs: List[tuple] = []
                if config.max_concurrency <= 1:
                    # 串行：一个失败立刻能决定要不要中止，不用等整层跑完
                    for task in runnable:
                        result = self._run_task(task, available)
                        pairs.append((task, result))
                        if result.status == "failed" and (
                            task.on_failure == "abort" or config.stop_on_error
                        ):
                            aborted = task
                            break
                else:
                    pairs = list(zip(runnable, self._execute_layer(runnable, available)))
                    for task, result in pairs:
                        if result.status == "failed" and (
                            task.on_failure == "abort" or config.stop_on_error
                        ):
                            aborted = task
                            break

                for task, result in pairs:
                    by_name[task.name] = result
                    if result.ok:
                        # 按 output_key 和任务名各放一份：下游用 {{名字}} 或 {{key}} 都行
                        available[task.key] = result.output
                        available[task.name] = result.output
                    if self.on_result is not None:
                        self.on_result(task, result)
            if aborted is not None:
                break

        # 没轮到（中止 / 预算用完）的任务也要有结果，别让它们凭空消失
        for task in plan.tasks:
            if task.name not in by_name:
                reason = (
                    f"计划已中止（{aborted.name} 失败）"
                    if aborted is not None
                    else "没有执行"
                )
                by_name[task.name] = self._skipped(task, reason)

        result = PlanResult(
            task=plan.task,
            results=[by_name[task.name] for task in plan.tasks],
            elapsed=time.perf_counter() - started,
        )
        if aborted is not None:
            failed = by_name[aborted.name]
            raise TaskError(
                f"任务 {aborted.name!r} 失败，计划已中止：{failed.error}",
                task=aborted.name,
                result=result,
            )
        return result

    # ---------------- 一层 ----------------
    def _execute_layer(
        self, tasks: Sequence[Task], available: Mapping[str, Any]
    ) -> List[TaskResult]:
        if len(tasks) == 1 or self.config.max_concurrency <= 1:
            return [self._run_task(task, available) for task in tasks]

        workers = min(self.config.max_concurrency, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._run_task, task, available) for task in tasks]
            results: List[TaskResult] = []
            for task, future in zip(tasks, futures):
                try:
                    results.append(future.result())
                except Exception as exc:      # noqa: BLE001 - 一个任务崩了不该带走别人
                    results.append(
                        TaskResult(
                            name=task.name,
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            output_key=task.key,
                        )
                    )
            return results

    # ---------------- 单个任务 ----------------
    def _run_task(self, task: Task, available: Mapping[str, Any]) -> TaskResult:
        started = time.perf_counter()
        # 默认按「失败」起步：只有真拿到结果才改成 success，避免「异常但算成功」
        result = TaskResult(name=task.name, status="failed", output_key=task.key)

        # 同一层并发时，大家可能都是「预算还够」才开跑的 —— 这里再挡一次，
        # 免得一层里的第 N 个任务明明超预算还照跑
        limit = self.config.max_total_tokens
        limit = self.config.max_total_tokens
        if limit is not None and getattr(self, "_spent", 0) >= int(limit):
            return self._finish(
                self._skipped(
                    task, f"超出 token 预算（已用 {self._spent}/{limit}）"
                )
            )

        agent = self._agent_for(task)
        if agent is None:
            result.status = "failed"
            result.error = (
                "没有可用的 agent：给 Task(agent=...)、ExecutePlan(agent=...)、"
                "ExecutePlan(agent_factory=...) 或设一个默认 Agent"
            )
            result.latency = time.perf_counter() - started
            return self._finish(result)

        try:
            prompt = self._prompt_for(task, available)
        except KeyError as exc:
            result.status = "failed"
            result.error = f"提示词里的变量不存在：{exc}"
            result.latency = time.perf_counter() - started
            return self._finish(result)

        last_error: Optional[Exception] = None
        attempts = 0
        # skip / abort 不重试：它们的语义是「别在这上面耗」
        max_tries = (
            1 if task.on_failure in ("skip", "abort") else task.max_retries + 1
        )
        for attempt in range(max_tries):
            attempts += 1
            try:
                value, tokens = self._invoke(agent, task, prompt, attempt, last_error)
                result.status = "success"
                result.output = value
                result.tokens += tokens
                break
            except Exception as exc:      # noqa: BLE001 - 失败要按策略处理
                last_error = exc
                result.error = f"{type(exc).__name__}: {exc}"

        if not result.ok and task.on_failure == "skip":
            result.status = "skipped"
        elif not result.ok and task.on_failure == "fallback":
            attempts += 1
            try:
                value, tokens = self._invoke(
                    task.fallback_agent, task, prompt, 0, last_error
                )
                result.status = "success"
                result.output = value
                result.tokens += tokens
                result.error = f"主 agent 失败（{last_error}），已由 fallback 完成"
            except Exception as exc:      # noqa: BLE001
                result.error = f"{result.error}；fallback 也失败：{exc}"

        result.attempts = attempts
        result.latency = time.perf_counter() - started
        return self._finish(result)

    def _finish(self, result: TaskResult) -> TaskResult:
        '''把这一任务的 token 记到总额里（同一层后面的任务就能看见预算变化）。'''
        self._spent = getattr(self, "_spent", 0) + int(result.tokens or 0)
        return result

    def _invoke(
        self,
        agent: Any,
        task: Task,
        prompt: str,
        attempt: int,
        last_error: Optional[Exception],
    ) -> Any:
        '''跑一次模型调用，返回 (结果, 这次花掉的 token)。'''
        bus = EventBus()
        counter = _TokenCounter()
        bus.on(counter.handler)
        if self.config.isolate:
            reset = getattr(agent, "reset", None)
            if callable(reset):
                reset()

        text = prompt
        if attempt > 0 and last_error is not None:
            text = f"{prompt}\n\n（上一次执行出错了：{last_error}\n请重试。）"

        def call() -> Any:
            if task.output_model is not None:
                return structured_answer(
                    agent, text, task.output_model, tools=task.tools, bus=bus
                )
            return agent.ask(
                {"role": "user", "content": text}, tools=task.tools, bus=bus
            )

        value = self._with_timeout(task, call)
        return value, counter.total

    def _with_timeout(self, task: Task, call: Callable[[], Any]) -> Any:
        timeout = task.timeout if task.timeout is not None else self.config.per_task_timeout
        if not timeout:
            return call()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(call)
            try:
                return future.result(timeout=float(timeout))
            except FuturesTimeout:
                future.cancel()
                raise TaskError(
                    f"任务 {task.name!r} 超时（{timeout}s）", task=task.name
                ) from None
        finally:
            # wait=False：同步 Agent 没有取消机制，这里只是不再等它
            pool.shutdown(wait=False)

    # ---------------- 杂项 ----------------
    def _coerce(self, plan: Any) -> Plan:
        if isinstance(plan, Plan):
            return plan
        if isinstance(plan, (list, tuple)):
            return Plan.from_template(plan)
        raise PlanError(f"看不懂的计划：{type(plan).__name__}")

    def _agent_for(self, task: Task) -> Any:
        if task.agent is not None:
            return task.agent
        if self.agent_factory is not None:
            agent = self.agent_factory(task)
            if agent is not None:
                return agent
        if self.agent is not None:
            return self.agent
        from promptkit.agent import Agent

        return Agent.default()

    def _prompt_for(self, task: Task, available: Mapping[str, Any]) -> str:
        values = {key: as_text(value) for key, value in available.items()}
        if task.input_keys:
            missing = [key for key in task.input_keys if key not in values]
            if missing:
                raise KeyError(
                    f"任务 {task.name!r} 需要 {missing}，但上下文里没有；"
                    f"现有：{', '.join(sorted(values)) or '（空）'}"
                )
            values = {key: values[key] for key in task.input_keys}
        prompt = render_text(task.description, values, allow_dollar=False)
        return prompt + self._upstream_block(task, available)

    @staticmethod
    def _upstream_block(task: Task, available: Mapping[str, Any]) -> str:
        '''把依赖任务的输出附在提示词后面（已经在 `{{名字}}` 里引用的不重复附）。

        没有这一步，「依赖 A，但描述里没写 {{A}}」的任务拿不到 A 的结果 ——
        人写模板时记得引用，模型规划时经常不记得。
        '''
        blocks: List[str] = []
        for name in task.depends_on:
            if "{{" + name + "}}" in task.description:
                continue
            output = available.get(name)
            if output is None:
                continue
            blocks.append(f"## {name} 的结果\n{as_text(output)}")
        if not blocks:
            return ""
        return "\n\n上游任务的结果（请基于它们完成上面的任务）：\n" + "\n\n".join(
            blocks
        )

    def _skip_reason(
        self, task: Task, by_name: Mapping[str, TaskResult]
    ) -> Optional[str]:
        limit = self.config.max_total_tokens
        if limit is not None and self._spent >= int(limit):
            return f"超出 token 预算（已用 {self._spent}/{limit}）"
        bad = [
            name
            for name in task.depends_on
            if name not in by_name or not by_name[name].ok
        ]
        if bad:
            return "依赖没有成功：" + ", ".join(bad)
        return None

    @staticmethod
    def _skipped(task: Task, reason: str) -> TaskResult:
        return TaskResult(
            name=task.name,
            status="skipped",
            error=reason,
            output_key=task.key,
        )


class _TokenCounter:
    '''挂在一个临时 EventBus 上，把 USAGE 事件里的 total_tokens 加起来。'''

    def __init__(self) -> None:
        self.total = 0

    def handler(self, event: Any) -> None:
        if event.type != EventType.USAGE:
            return
        try:
            self.total += int(event.data.get("total_tokens") or 0)
        except (TypeError, ValueError):
            return


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
