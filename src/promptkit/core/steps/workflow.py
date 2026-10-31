# core/workflow.py
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, TextIO, Union

from promptkit.core.model_action.caution_related.caution import WORKFLOW_CAUTIONS_KEY, flatten_cautions
from promptkit.core.context import Context
from promptkit.core.display_sys.display import console_display
from promptkit.core.errors import (
    BreakSignal,
    ContinueSignal,
    GotoSignal,
    LoopLimitExceeded,
    ReturnSignal,
    UnknownLabel,
    WorkflowError,
)
from promptkit.core.event_sys import EVENT_BUS_KEY, EventBus, EventHandler
from promptkit.core.display_sys.render import Verbosity, as_verbosity
from promptkit.core.settings import settings
from promptkit.core.steps.actions.interactive import OUTPUT_STREAM_KEY, VERBOSITY_KEY
from promptkit.core.steps.logics.goto import collect_labels
from promptkit.core.steps.interface import (
    HOOKS_KEY,
    Step,
    StepLike,
    coerce_steps,
    execute_step,
)
from promptkit.types import AgentLike, JSONValue

if TYPE_CHECKING:
    from promptkit.session import Session


class Workflow(Step):
    '''按顺序执行一组 Step，所有 Step 共享同一个 Context。

    Workflow 本身也是一个 Step，所以可以嵌套进 If/Then/Else
    或者另一个 Workflow：

        Workflow(
            If(c1).Then(Workflow(step_a, step_b)).Else(step_c),
        )

    顶层支持 `Label("name")` / `Goto("name")` 跳转（见 core/steps/goto.py）。
    '''

    name = "workflow"

    def __init__(
        self,
        *steps: StepLike,
        max_steps: Optional[int] = None,
        max_jumps: Optional[int] = None,
        name: str = "workflow",
        cautions: object = None,
    ):
        self.steps: List[Step] = coerce_steps(*steps)
        # `None` = 用全局设置；存原值，用到的时候再解析，
        # 这样改了 settings.max_steps 对已经建好的工作流也生效
        self.max_steps = max_steps
        self.max_jumps = max_jumps
        self.name = name
        # Workflow 级约束：挂在 context 上，里面的 LLMStep 共享
        self.cautions = list(flatten_cautions(cautions))

        limit = self.step_limit()
        if len(self.steps) > limit:
            raise ValueError(f"步骤数超过上限 {limit}")

    def step_limit(self) -> int:
        '''实际生效的步骤数上限（显式值优先，否则取全局设置）。'''
        value = settings.max_steps if self.max_steps is None else self.max_steps
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"max_steps 必须是正整数，得到 {value!r}")
        return value

    def jump_limit(self) -> int:
        '''实际生效的 Goto 跳转上限。'''
        value = settings.max_jumps if self.max_jumps is None else self.max_jumps
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"max_jumps 必须是正整数，得到 {value!r}")
        return value

    def add(self, *steps: StepLike) -> "Workflow":
        '''追加步骤，返回自身以便链式调用。'''
        new_steps = coerce_steps(*steps)
        limit = self.step_limit()
        if len(self.steps) + len(new_steps) > limit:
            raise ValueError(f"步骤数超过上限 {limit}")
        self.steps.extend(new_steps)
        return self

    def labels(self) -> dict:
        '''顶层步骤里的标签 -> 下标。'''
        return collect_labels(self.steps)

    def execute(self, context: Context) -> None:
        '''在已有的 Context 上执行所有步骤（用于嵌套）。

        Label / Goto 按「词法作用域」解析：本工作流认识这个标签就跳，
        不认识就继续往外抛，让外层工作流处理。
        '''
        context.setdefault("messages", [])
        labels = self.labels()

        index = 0
        jumps = 0
        while index < len(self.steps):
            step = self.steps[index]
            index += 1
            try:
                execute_step(step, context)
            except GotoSignal as signal:
                if signal.target not in labels:
                    raise  # 交给外层工作流
                jumps += 1
                limit = self.jump_limit()
                if jumps > limit:
                    raise LoopLimitExceeded(
                        f"Goto 连续跳转超过 {limit} 次仍未结束，"
                        f"检查一下 {signal.target!r} 附近是不是死循环"
                    ) from None
                index = labels[signal.target] + 1

    # ---------------- 展示成步骤树 ----------------
    def describe(self) -> str:
        if self.name == "workflow":
            return "Workflow"
        return f"Workflow({self.name!r})"

    def children(self) -> List[Step]:
        return list(self.steps)

    def __repr__(self) -> str:
        '''print(wf) 直接打印整棵步骤树。'''
        return self.render()

    def run(
        self,
        agent: Optional[AgentLike] = None,
        bus: Optional[EventBus] = None,
        verbose: Union[Verbosity, int, bool] = Verbosity.NORMAL,
        on_event: Optional[EventHandler] = None,
        out: Optional[TextIO] = None,
        session: Optional[Union[str, "Session"]] = None,
        reset: Optional[bool] = None,
        variables: Optional[Sequence[str]] = None,
        hooks: object = None,
        judge: object = None,
        metadata: Optional[Dict[str, JSONValue]] = None,
        **kwargs,
    ) -> Context:
        '''新建一个 Context 并执行，返回它。

        agent 省略时用 `Agent()` 自动注册的默认实例，所以常见写法就是：

            wf = Workflow()
            agent = Agent()
            wf.run()

        关于「过程」和「结果」的分流，三个参数各管一段：

        - `out=`：**结果**（`Echo` 的输出）写到哪里，默认 `sys.stdout`
        - `verbose=`：**过程**（思考 / 工具调用 / 步骤轨迹）写到 `sys.stderr`
        - `bus=` / `on_event=`：自己接管过程事件，想发哪儿发哪儿 —
          日志文件就是这么挂上去的：`on_event=log_display("logs/run.log")`

        `verbose` 可以是 `Verbosity`（推荐）、`int`，或者老的 `True / False`：

            wf.run(verbose=Verbosity.SILENT)    # 连 Echo 都不打，结果从 ctx 取
            wf.run(verbose=Verbosity.NORMAL)    # 默认：只有对话
            wf.run(verbose=Verbosity.VERBOSE)   # + 步骤轨迹 / 工具调用 / 思考
            wf.run(verbose=Verbosity.DEBUG)     # + 原始事件
            wf.run(verbose=True, out=open("result.txt", "w"))   # 老写法仍可用

        `session=`：会话文件路径（或 `Session`）。给了就
        **开跑前 load、结束时 save（异常也 save）**，下次同一个路径接着聊。
        手写 Workflow 想保留上下文就靠它：

            wf.run(agent=agent, session="sessions/tutor.json")
            wf.run(agent=agent, session=Session(p).Variables("basic_info"))

        `reset=`：要不要清空历史。`None`（默认）= 有会话文件就保留、
        没有就清空；显式传就按你传的来。

        `variables=`：要跟着会话一起存的 Context 变量名。给了就覆盖
        `Session.variables`（`Session(path).Variables(...)` 的链式写法）。

        `metadata=`：额外写进会话文件的元信息（`ChatWorkflow` 用它记 role），
        读的时候不会用到。

        `hooks=`：`promptkit.api.Hooks` 实例，给每一步挂上
        前置拦截 / 后置处理 / 异常捕获钩子（详见 `promptkit.api`）。

        `judge=`：语义判定器（如 `LLMJudge(agent)`）。注入一次，工作流里的
        `Intent` / `Similar` / `Classify` 就不用再逐个 `.Using(judge)`；
        单个条件上的 `.Using()` 优先于这里的注入。
        '''
        if session is not None:
            return self._run_with_session(
                session,
                agent=agent,
                bus=bus,
                verbose=verbose,
                on_event=on_event,
                out=out,
                reset=reset,
                variables=variables,
                hooks=hooks,
                judge=judge,
                metadata=metadata,
                **kwargs,
            )

        context = Context(data=dict(kwargs))
        if agent is not None:
            context["agent"] = agent
        if judge is not None:
            from promptkit.core.steps.logics.semantic import SEMANTIC_JUDGE_KEY

            context[SEMANTIC_JUDGE_KEY] = judge
        if out is not None:
            context[OUTPUT_STREAM_KEY] = out
        if hooks is not None:
            context[HOOKS_KEY] = hooks
        if self.cautions:
            context[WORKFLOW_CAUTIONS_KEY] = list(self.cautions)

        # 显示分级：接受 bool（老写法）/ int / Verbosity，`Echo` 也会看这个
        level = as_verbosity(verbose)
        context[VERBOSITY_KEY] = int(level)

        if bus is None and agent is not None:
            # 没显式给 bus，就借用 agent 的那条总线
            bus = getattr(agent, "bus", None)
        if bus is None:
            bus = EventBus()

        handlers = []
        if on_event is not None:
            handlers.append(bus.on(on_event))
        if level >= Verbosity.VERBOSE:
            # 过程信息走 stderr，且**不重复输出回答**——回答是结果，由 Echo 负责
            handlers.append(
                bus.on(
                    console_display(
                        show_thinking=True,
                        show_tools=True,
                        show_steps=True,
                        show_message=False,
                        stream=sys.stderr,
                    )
                )
            )
        if handlers:
            context[EVENT_BUS_KEY] = bus
        elif bus.has_handlers:
            context[EVENT_BUS_KEY] = bus
        # 总线一个 handler 都没有时不挂上去，省掉所有事件开销

        try:
            self.execute(context)
        except GotoSignal as signal:
            known = ", ".join(sorted(self.labels())) or "（一个都没有）"
            raise UnknownLabel(
                f"没有找到标签 {signal.target!r}。这条工作流顶层的标签有：{known}。"
                "Goto 只能跳到工作流的顶层 Label —— 跳到 While/If 内部是不行的"
                "（循环体内的跳转请用 Continue / Break）"
            ) from None
        except ReturnSignal:
            raise WorkflowError(
                "Return() 逃到了工作流外面：它只能用在 "
                "Function(...).Do(...) 的函数体里"
            ) from None
        except (BreakSignal, ContinueSignal) as signal:
            raise WorkflowError(
                f"{type(signal).__name__} 逃到了工作流外面：Break() / Continue() "
                "只能用在循环里（While / DoWhile / For / ForEach）；"
                "Switch 的 case 不需要 break，也不会拦住它"
            ) from None
        finally:
            for handler in handlers:
                bus.off(handler)

        return context

    def _run_with_session(
        self,
        session: Union[str, "Session"],
        *,
        agent: Optional[AgentLike],
        bus: Optional[EventBus],
        verbose: Union[Verbosity, int, bool],
        on_event: Optional[EventHandler],
        out: Optional[TextIO],
        reset: Optional[bool],
        variables: Optional[Sequence[str]],
        hooks: object,
        judge: object,
        metadata: Optional[Dict[str, JSONValue]],
        **kwargs,
    ) -> Context:
        '''`run(session=...)` 的实现：load -> 跑 -> 存（异常也存）。

        顺序很重要：**先决定要不要保留历史，再装历史**。反过来的话，
        `SystemPrompt(reset=True)` 会把刚装回来的清掉。

        `variables` 是要跟着会话一起存的 Context 变量名（比如
        `["basic_info"]`）。存了之后，下次 `run()` 会把它们当关键字参数
        塞回 Context —— 开头那段问卷就能读到上次填好的值。

        会话是「Agent 层」的东西，`core` 不该硬依赖它，所以延迟导入。
        '''
        from promptkit.session import as_session, set_system_prompt_reset

        active = as_session(session)
        if active is None:
            raise ValueError("run(session=...) 需要路径或 Session 实例，不能是 None")
        if agent is None:
            from promptkit.agent import Agent

            agent = Agent.default()
            if agent is None:
                raise ValueError(
                    "run(session=...) 需要 agent：要么先 `Agent()` 建一个"
                    "（会自动注册为默认），要么显式传 `agent=...`。"
                )

        names = tuple(variables) if variables is not None else active.variables

        # 要记的变量：把上次存的值当 kwargs 传给内层 run()。
        # `NewVar(...)` 用的是 setdefault，所以传进来的值不会被默认值盖掉。
        seeded: Dict[str, JSONValue] = {}
        if names:
            stored = active.load_variables()
            seeded = {name: stored[name] for name in names if name in stored}

        keep = active.exists() if reset is None else (not reset)
        restored = active.load(agent)

        set_system_prompt_reset(self, not keep)
        context: Optional[Context] = None
        try:
            context = self.run(
                agent=agent,
                bus=bus,
                verbose=verbose,
                on_event=on_event,
                out=out,
                hooks=hooks,
                judge=judge,
                **{**seeded, **kwargs},
            )
            return context
        finally:
            # 异常退出也要存 —— 聊到一半 Ctrl+C 不该把整段对话丢掉
            snapshot: Optional[Dict[str, object]] = None
            if names and context is not None:
                snapshot = {name: context.get(name) for name in names}
            extra: Dict[str, JSONValue] = dict(metadata or {})
            extra["restored"] = restored
            active._save(agent, variables=snapshot, metadata=extra)
