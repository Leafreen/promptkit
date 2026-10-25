'''promptkit 的冒烟测试 —— 不需要网络和 API Key，直接运行：

    python tests/test_promptkit.py
'''
import json
import os
import platform
import sys
import unittest
import urllib.error
from types import SimpleNamespace
from typing import Optional
import io
import tempfile
from unittest import mock

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai.types.chat import ChatCompletionMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from promptkit.agent import Agent, ToolLoopExceeded, load_env_file  # noqa: E402
from promptkit.executor import (  # noqa: E402
    Executor,
    MaxRetriesExceeded,
    OutputValidationError,
)
from promptkit.prompt import Prompt, render_messages, render_text  # noqa: E402
from promptkit.core.model_action.caution_related.caution import (  # noqa: E402
    join_cautions,
)
from promptkit.core.event_sys import (  # noqa: E402
    Event,
    EventBus,
    EventType,
    emit,
    extract_reasoning,
)
from promptkit.core.display_sys.display import (  # noqa: E402
    collect_events,
    console_display,
    filter_events,
    json_display,
)
from promptkit.default_tools import RUN_COMMAND  # noqa: E402
from promptkit.default_tools import (  # noqa: E402
    DEFAULT_ALLOWED_EXTENSIONS,
    WRITE_FILE,
    WriteFile,
)
from promptkit.default_tools.run_command import _decode, run_command  # noqa: E402
from promptkit.default_tools.web_search import (  # noqa: E402
    BaiduBackend,
    BingCnBackend,
    BochaBackend,
    BraveBackend,
    DuckDuckGoBackend,
    FallbackBackend,
    SearchError,
    SearchResult,
    SerperBackend,
    TavilyBackend,
    WebSearch,
    ZhipuBackend,
    clean_text,
    clamp_k,
    dedupe,
    dig,
    extract_baidu_target,
    parse_baidu,
    parse_bing,
    parse_ddg_lite,
    pick_backend,
    pick_field,
    resolve_baidu_target,
    unwrap_url,
)
from promptkit.cautions import (  # noqa: E402
    ANALYST,
    CODE_ONLY,
    GROUNDED,
    JSON_ONLY,
    JSON_STRICT,
    PLAIN_TEXT,
    SOCRATIC,
    TUTOR,
    AskFirst,
    Citations,
    Concise,
    Custom,
    Grounded,
    Language,
    MaxLength,
    NoExtraText,
    NoFabrication,
    NoMarkdown,
    OutputFormat,
    StepByStep,
    Tone,
    analyst,
    concise_text,
    grounded,
    json_strict,
    json_with_schema,
    socratic,
    translator,
    tutor,
)
from promptkit.core.model_action.caution_related.caution import (  # noqa: E402
    WORKFLOW_CAUTIONS_KEY,
    CautionGroup,
    resolve_cautions,
)
from promptkit.core.model_action.defaults import (  # noqa: E402
    add_default_caution,
    clear_default_cautions,
    get_default_cautions,
    set_default_cautions,
    using_default_cautions,
)
from promptkit.core.settings import (  # noqa: E402
    Settings,
    reset_settings,
    settings,
    settings_names,
    using_settings,
)
from promptkit.core.display_sys.render import (  # noqa: E402
    HAVE_RICH,
    HAVE_SYMPY,
    Verbosity,
    as_verbosity,
    install_hint,
    latex_to_unicode,
    looks_like_math,
    needs_art,
    normalize_latex,
    plain_text,
    render_markdown,
    render_math,
    render_plain,
    strip_markdown,
    get_console,
)
from promptkit.core.display_sys.runlog import (  # noqa: E402
    close_logging,
    get_logger,
    log_display,
    setup_logging,
)
from promptkit.client_sys import (  # noqa: E402
    AnthropicClient,
    BaseClient,
    ClientError,
    ClientSpec,
    Completion,
    OpenAIClient,
    detect_vendor,
    get_spec,
    iter_sse,
    make_client,
    register_vendor,
    resolve_api_key,
    resolve_target,
    vendor_names,
)
from promptkit.core.steps.actions.interactive import VERBOSITY_KEY  # noqa: E402
from promptkit.core.steps.actions.chat_util import (  # noqa: E402
    DEFAULT_EXIT_WORDS,
    Behavior,
    ChatLoop,
    Deliverables,
    ExitWords,
    Role,
    Rules,
    SaveDeliverables,
    SystemPrompt,
    Tools,
    Want,
    build_system_prompt,
    chat_context_vars,
    extension_for,
    flatten_single,
    MakeFile,
)
from promptkit.core.steps.chat import (  # noqa: E402
    ChatWorkflow,
    Deliverable,
    chat_workflow,
)
from promptkit.session import (  # noqa: E402
    SESSION_VERSION,
    Session,
    as_session,
)
from promptkit.memory import (  # noqa: E402
    BACKENDS,
    FolderVectorStore,
    InMemoryMemory,
    InMemoryVectorStore,
    KVMemory,
    LayeredMemory,
    Memory,
    MemoryConfig,
    MemoryError,
    MemoryItem,
    RecallConfig,
    RecallMemoryTool,
    Reranker,
    SessionMemory,
    StorageConfig,
    TokenBudget,
    VectorMemory,
    chunk_text as memory_chunk_text,
    cosine,
    create_memory,
    default_data_dir,
    default_kv_path,
    default_vector_dir,
    estimate_tokens,
    format_memories,
    long_term_memory,
    register_memory_backend,
    safe_namespace_name,
)
from promptkit.memory.adapters import (  # noqa: E402
    ChromaVectorStore,
    CohereReranker,
    CrossEncoderReranker,
    HashEmbedding,
    LlmReranker,
    NoopReranker,
    OpenAIEmbedding,
    QdrantVectorStore,
    SentenceTransformerEmbedding,
)
from promptkit.core import (  # noqa: E402
    ForgetMemory,
    MemoryStep,
    ReadMemory,
    RecallMemory,
    WriteMemory,
    format_recalled,
)
from promptkit.agent import normalize_history  # noqa: E402
from promptkit.core.model_action.caution_related.formatter import (  # noqa: E402
    FormatterError,
    JsonFormatter,
    RegexFormatter,
    XmlFormatter,
    YamlFormatter,
    get_formatter,
)
from promptkit import features  # noqa: E402

# 知识库是实验性功能，默认关闭（见 promptkit/features.py）。
# 测试要在 import 之前打开，否则连模块都进不来。
features.enable("knowledge")

from promptkit.knowledge import (  # noqa: E402
    FORMAT_VERSION,
    KnowledgeBase,
    chunk_text,
    token_weight,
    tokenize,
)
from promptkit.tools import (  # noqa: E402
    Tool,
    as_tool,
    schema_from_signature,
    tool,
)
from promptkit.core import (  # noqa: E402
    And,
    ArgumentError,
    BinaryExpr,
    Break,
    Call,
    Condition,
    Context,
    Continue,
    Dec,
    DoWhile,
    Echo,
    Expr,
    For,
    ForEach,
    Function,
    FunctionStep,
    Goto,
    If,
    Inc,
    Label,
    Len,
    LLMStep,
    LoopLimitExceeded,
    NewVar,
    Not,
    Number,
    Or,
    Read,
    Return,
    Set,
    SetItemStep,
    SetStep,
    SkipStep,
    Step,
    Read,
    ReadFile,
    Save,
    SaveHistory,
    SaveJSON,
    SaveHistoryStep,
    SaveStep,
    Str,
    Switch,
    UndefinedVariable,
    UnknownLabel,
    Var,
    VarTypeError,
    When,
    While,
    Workflow,
    WorkflowError,
)
from promptkit.core.steps.actions.message import MessageStep  # noqa: E402


class Intent(BaseModel):
    label: str


class Order(BaseModel):
    status: str


def printed(fake):
    '''取出被 mock 的 print 打出的第一段文本（忽略 file= 之类的关键字）。'''
    return [call.args[0] if call.args else "" for call in fake.call_args_list]


def prompt_text(recorded):
    '''Agent 收到的提示：可能是字符串，也可能是 {"role", "content"} 消息。'''
    if isinstance(recorded, dict):
        return str(recorded.get("content", ""))
    return str(recorded)


class FakeAgent:
    '''按顺序返回预设回复，并记录收到的提示。'''

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.tools_seen = []
        self.resets = 0

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        self.prompts.append(prompt)
        self.tools_seen.append(tools)
        if not self.replies:
            raise AssertionError("FakeAgent 的回复用完了")
        return self.replies.pop(0)

    def reset(self):
        self.resets += 1


class PromptkitTestCase(unittest.TestCase):
    '''隔离全局状态。

    - `Agent.default()`：否则一个测试创建的 Agent 会被别的测试用上
    - 全局默认约束（set_default_cautions）：否则会悄悄漏进别的测试
    - 全局设置（settings）：改了 max_iterations 之类的会漏给后面所有测试
    - `Workflow.run()` 没传 out 时用的是 sys.stdout，别被换掉
    '''

    def setUp(self):
        self._saved_default_agent = Agent.default()
        Agent.set_default(None)
        self._saved_cautions = get_default_cautions()
        clear_default_cautions()
        self._saved_settings = settings.to_dict()

    def tearDown(self):
        Agent.set_default(self._saved_default_agent)
        set_default_cautions(self._saved_cautions)
        reset_settings()
        for key, value in self._saved_settings.items():
            setattr(settings, key, value)


class TestPrompt(PromptkitTestCase):
    def test_double_brace(self):
        self.assertEqual(
            Prompt(role="user", content="你好，{{name}}").render(name="小明"),
            {"role": "user", "content": "你好，小明"},
        )

    def test_dollar_brace_and_plain_dollar(self):
        prompt = Prompt(role="user", content="${a} 和 $b")
        self.assertEqual(prompt.render(a=1, b=2)["content"], "1 和 2")

    def test_system_role(self):
        rendered = Prompt(role="system", content="{{x}}").render(x="y")
        self.assertEqual(rendered, {"role": "system", "content": "y"})

    def test_json_literal_is_not_a_variable(self):
        # 单个花括号是普通文本，不应该被当成变量
        prompt = Prompt(role="user", content='只输出 {"label": "退款"} 与 {{text}}')
        self.assertEqual(
            prompt.render(text="hi")["content"], '只输出 {"label": "退款"} 与 hi'
        )

    def test_missing_variable_raises(self):
        with self.assertRaises(KeyError):
            Prompt(role="user", content="{{nope}}").render()

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            Prompt(role="tool", content="x").render()

    def test_render_messages(self):
        msgs = render_messages(
            [Prompt(role="system", content="s"), Prompt(role="user", content="{{v}}")],
            v="hello",
        )
        self.assertEqual(
            msgs,
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}],
        )


class TestExtractJson(PromptkitTestCase):
    def test_plain(self):
        self.assertEqual(Executor._extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(
            Executor._extract_json('```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_fenced_without_language(self):
        self.assertEqual(Executor._extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_surrounded_by_prose(self):
        self.assertEqual(
            Executor._extract_json('好的，结果是：{"a": 1} 希望有帮助'), {"a": 1}
        )

    def test_array(self):
        self.assertEqual(Executor._extract_json("结果 [1, 2]"), [1, 2])

    def test_empty_raises(self):
        with self.assertRaises(OutputValidationError):
            Executor._extract_json("")

    def test_garbage_raises(self):
        with self.assertRaises(OutputValidationError):
            Executor._extract_json("我无法回答这个问题")


class TestExecutor(PromptkitTestCase):
    def test_happy_path_sends_rendered_message(self):
        agent = FakeAgent(['{"label": "退款"}'])
        result = Executor(agent).run(
            Prompt(role="user", content="判断 {{text}}"), Intent, text="我要退款"
        )
        self.assertEqual(result.label, "退款")
        self.assertEqual(
            agent.prompts, [{"role": "user", "content": "判断 我要退款"}]
        )

    def test_retries_then_succeeds(self):
        agent = FakeAgent(["不是 JSON", '{"label": "咨询"}'])
        result = Executor(agent).run(
            Prompt(role="user", content="{{text}}"), Intent, text="hi"
        )
        self.assertEqual(result.label, "咨询")
        # 第一次是原始提示（保留 role），第二次是纠错提示
        self.assertEqual(agent.prompts[0], {"role": "user", "content": "hi"})
        self.assertIn("JSON", agent.prompts[1])
        self.assertEqual(len(agent.prompts), 2)

    def test_validation_error_is_retried(self):
        # JSON 合法但缺少 label 字段 -> Pydantic 校验失败 -> 重试
        agent = FakeAgent(['{"wrong": 1}', '{"label": "投诉"}'])
        result = Executor(agent).run(
            Prompt(role="user", content="{{text}}"), Intent, text="hi"
        )
        self.assertEqual(result.label, "投诉")

    def test_max_retries_exceeded(self):
        agent = FakeAgent(["bad", "bad", "bad"])
        with self.assertRaises(MaxRetriesExceeded):
            Executor(agent, max_retries=2).run(
                Prompt(role="user", content="{{text}}"), Intent, text="hi"
            )
        self.assertEqual(len(agent.prompts), 3)

    def test_message_role_is_preserved(self):
        agent = FakeAgent(['{"label": "咨询"}'])
        prompt = Prompt(role="system", content="{{text}}")
        Executor(agent).run(prompt, Intent, text="sys")
        self.assertEqual(agent.prompts[0], {"role": "system", "content": "sys"})


class UpperStep(Step):
    name = "upper"

    def execute(self, context: Context) -> None:
        context["text"] = context["text"].upper()
        context.trace.append(self.name)


class TestWorkflow(PromptkitTestCase):
    def test_message_steps_share_context(self):
        ctx = Workflow(
            MessageStep(role="system", content="你是助手"),
            MessageStep(role="user", content="你好"),
        ).run()
        self.assertEqual(
            ctx["messages"],
            [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "你好"}],
        )
        self.assertEqual(ctx.trace, ["message", "message"])

    def test_kwargs_and_custom_step(self):
        ctx = Workflow(UpperStep()).run(text="abc")
        self.assertEqual(ctx["text"], "ABC")
        self.assertEqual(ctx["messages"], [])

    def test_max_steps(self):
        with self.assertRaises(ValueError):
            Workflow(MessageStep(role="user", content="a"), max_steps=0)

    def test_add_is_chainable(self):
        wf = Workflow().add(MessageStep(role="user", content="a")).add(
            MessageStep(role="user", content="b")
        )
        self.assertEqual(len(wf.run()["messages"]), 2)


def const(value, key="r"):
    '''返回一个把常量写进 key 的步骤。'''
    return FunctionStep(lambda _ctx: value, output_key=key)


class TestIfDsl(PromptkitTestCase):
    def test_then_branch(self):
        ctx = Workflow(If(True).Then(const("then"))).run()
        self.assertEqual(ctx["r"], "then")

    def test_else_branch(self):
        ctx = Workflow(If(False).Then(const("then")).Else(const("else"))).run()
        self.assertEqual(ctx["r"], "else")

    def test_no_else_does_nothing(self):
        ctx = Workflow(If(False).Then(const("then"))).run()
        self.assertNotIn("r", ctx)

    def test_only_else(self):
        ctx = Workflow(If(False).Else(const("else"))).run()
        self.assertEqual(ctx["r"], "else")

    def test_callable_condition_reads_context(self):
        wf = Workflow(If(lambda ctx: ctx["n"] > 1).Then(const("big")).Else(const("small")))
        self.assertEqual(wf.run(n=5)["r"], "big")
        self.assertEqual(wf.run(n=0)["r"], "small")

    def test_custom_condition_class(self):
        class IsBig(Condition):
            name = "is_big"

            def evaluate(self, context):
                return context["n"] > 1

        ctx = Workflow(If(IsBig()).Then(const("big")).Else(const("small"))).run(n=5)
        self.assertEqual(ctx["r"], "big")

    def test_and_or_not(self):
        big = lambda ctx: ctx["n"] > 1  # noqa: E731
        even = lambda ctx: ctx["n"] % 2 == 0  # noqa: E731
        self.assertEqual(Workflow(If(And(big, even)).Then(const("y"))).run(n=4)["r"], "y")
        self.assertNotIn("r", Workflow(If(And(big, even)).Then(const("y"))).run(n=3))
        self.assertEqual(Workflow(If(Or(big, even)).Then(const("y"))).run(n=3)["r"], "y")
        self.assertEqual(Workflow(If(Not(big)).Then(const("y"))).run(n=0)["r"], "y")

    def test_nested_else_if(self):
        # If(c1).Then(...).Else(If(c2).Then(...))
        wf = Workflow(
            If(lambda ctx: ctx["n"] == 1)
            .Then(const("one"))
            .Else(If(lambda ctx: ctx["n"] == 2).Then(const("two")))
        )
        self.assertEqual(wf.run(n=1)["r"], "one")
        self.assertEqual(wf.run(n=2)["r"], "two")
        self.assertNotIn("r", wf.run(n=3))

    def test_then_accepts_multiple_and_lists(self):
        wf = Workflow(If(True).Then(const("a", "x"), [const("b", "y"), const("c", "z")]))
        ctx = wf.run()
        self.assertEqual((ctx["x"], ctx["y"], ctx["z"]), ("a", "b", "c"))

    def test_workflow_can_be_nested_in_branch(self):
        inner = Workflow(const("inner"))
        ctx = Workflow(If(True).Then(inner)).run()
        self.assertEqual(ctx["r"], "inner")

    def test_then_does_not_mutate_original(self):
        base = If(True)
        with_then = base.Then(const("then"))
        with_else = base.Else(const("else"))
        # 原对象没有被任何一个分支修改
        self.assertEqual(len(base.branches), 1)
        self.assertEqual(base.branches[0][1], ())
        self.assertEqual(base.else_steps, ())
        # Then 只改了副本，Else 也一样
        self.assertEqual(len(with_then.branches[0][1]), 1)
        self.assertEqual(with_then.else_steps, ())
        self.assertEqual(with_else.branches[0][1], ())
        self.assertEqual(len(with_else.else_steps), 1)

    def test_trace_records_branch(self):
        ctx = Workflow(
            If(True).Then(const("a")).Else(const("b")),
            If(False).Then(const("c")).Else(const("d")),
        ).run()
        self.assertEqual(
            ctx.trace, ["if:then", "function", "if:else", "function"]
        )

    def test_non_step_raises(self):
        with self.assertRaises(TypeError):
            If(True).Then("不是步骤")

    def test_bad_condition_raises(self):
        with self.assertRaises(TypeError):
            If(123)


class TestLLMStep(PromptkitTestCase):
    def test_structured_output(self):
        agent = FakeAgent(['{"label": "退款"}'])
        ctx = Workflow(
            LLMStep(
                Prompt(role="user", content="意图：{{text}}"),
                Intent,
                output_key="intent",
            )
        ).run(agent=agent, text="我要退款")
        self.assertEqual(ctx["intent"].label, "退款")
        self.assertEqual(agent.prompts[0], {"role": "user", "content": "意图：我要退款"})

    def test_plain_text_uses_default_key(self):
        agent = FakeAgent(["你好呀"])
        ctx = Workflow(LLMStep(Prompt(role="user", content="{{text}}"))).run(
            agent=agent, text="hi"
        )
        self.assertEqual(ctx["output"], "你好呀")

    def test_agent_from_constructor(self):
        agent = FakeAgent(["x"])
        ctx = Workflow(
            LLMStep(Prompt(role="user", content="a"), agent=agent)
        ).run()
        self.assertEqual(ctx["output"], "x")

    def test_missing_agent_raises(self):
        with self.assertRaises(ValueError) as cm:
            Workflow(LLMStep(Prompt(role="user", content="a"))).run()
        self.assertIn("agent", str(cm.exception))

    def test_reset_flag(self):
        agent = FakeAgent(["x"])
        Workflow(
            LLMStep(Prompt(role="user", content="a"), reset=True)
        ).run(agent=agent)
        self.assertEqual(agent.resets, 1)

    def test_trace(self):
        agent = FakeAgent(["x"])
        ctx = Workflow(LLMStep(Prompt(role="user", content="a"))).run(agent=agent)
        self.assertEqual(ctx.trace, ["llm"])


class TestFunctionStep(PromptkitTestCase):
    def test_side_effect_only(self):
        seen = []
        ctx = Workflow(FunctionStep(lambda c: seen.append(c["text"]))).run(text="hi")
        self.assertEqual(seen, ["hi"])
        self.assertNotIn("output", ctx)

    def test_output_key(self):
        ctx = Workflow(FunctionStep(lambda c: c["text"] * 2, output_key="double")).run(
            text="ab"
        )
        self.assertEqual(ctx["double"], "abab")


class TestRefundWorkflowDsl(PromptkitTestCase):
    '''用假 agent 跑通 If/Then/Else 版的客服意图路由。'''

    class NeedRefund(Condition):
        name = "need_refund"

        def evaluate(self, context):
            return context["intent"].label == "退款"

    class NeedComplaint(Condition):
        name = "need_complaint"

        def evaluate(self, context):
            return context["intent"].label == "投诉"

    class Shipped(Condition):
        name = "shipped"

        def evaluate(self, context):
            return context["order"].status == "已发货"

    def build(self):
        return Workflow(
            LLMStep(
                Prompt(role="user", content="意图：{{text}}"),
                Intent,
                output_key="intent",
            ),
            If(self.NeedRefund()).Then(
                LLMStep(
                    Prompt(role="user", content="订单：{{text}}"),
                    Order,
                    output_key="order",
                ),
                If(self.Shipped()).Then(
                    FunctionStep(lambda _c: "需要先申请退货。", output_key="reply")
                ).Else(
                    FunctionStep(lambda _c: "可以退款。", output_key="reply")
                ),
            ).Else(
                If(self.NeedComplaint()).Then(
                    FunctionStep(lambda _c: "已转投诉专员。", output_key="reply")
                ).Else(
                    FunctionStep(lambda _c: "请问有什么可以帮您？", output_key="reply")
                )
            ),
        )

    def test_refund_and_shipped(self):
        agent = FakeAgent(['{"label": "退款"}', '{"status": "已发货"}'])
        ctx = self.build().run(agent=agent, text="我要退款")
        self.assertEqual(ctx["reply"], "需要先申请退货。")
        self.assertEqual(ctx.trace, ["llm", "if:then", "llm", "if:then", "function"])

    def test_refund_and_not_shipped(self):
        agent = FakeAgent(['{"label": "退款"}', '{"status": "未发货"}'])
        ctx = self.build().run(agent=agent, text="我要退款")
        self.assertEqual(ctx["reply"], "可以退款。")
        self.assertEqual(ctx.trace, ["llm", "if:then", "llm", "if:else", "function"])

    def test_complaint(self):
        agent = FakeAgent(['{"label": "投诉"}'])
        ctx = self.build().run(agent=agent, text="我要投诉")
        self.assertEqual(ctx["reply"], "已转投诉专员。")
        self.assertEqual(ctx.trace, ["llm", "if:else", "if:then", "function"])

    def test_other(self):
        agent = FakeAgent(['{"label": "咨询"}'])
        ctx = self.build().run(agent=agent, text="怎么开发票")
        self.assertEqual(ctx["reply"], "请问有什么可以帮您？")
        self.assertEqual(ctx.trace, ["llm", "if:else", "if:else", "function"])


class TestExpr(PromptkitTestCase):
    def test_comparison_is_lazy(self):
        expr = Var("n") > 1
        self.assertIsInstance(expr, Expr)
        self.assertTrue(expr.eval(Context(data={"n": 5})))
        self.assertFalse(expr.eval(Context(data={"n": 0})))

    def test_undefined_variable_raises(self):
        with self.assertRaises(UndefinedVariable):
            Var("nope").eval(Context())

    def test_undefined_variable_is_key_error(self):
        with self.assertRaises(KeyError):
            Var("nope").eval(Context())

    def test_logic_operators(self):
        # 注意：& | 比比较运算符优先级高，所以要加括号
        ctx = Context(data={"n": 4})
        self.assertTrue(((Var("n") > 1) & (Var("n") < 5)).eval(ctx))
        self.assertFalse(((Var("n") > 1) & (Var("n") < 2)).eval(ctx))
        self.assertTrue(((Var("n") > 10) | (Var("n") < 5)).eval(ctx))
        self.assertTrue((~(Var("n") > 10)).eval(ctx))

    def test_arithmetic(self):
        ctx = Context(data={"n": 2})
        self.assertEqual((Var("n") * 3 + 1).eval(ctx), 7)
        self.assertEqual((Var("n") % 2).eval(ctx), 0)

    def test_bool_of_expr_raises(self):
        # 防止有人写 `if Var("n"):` 这种静默错误的代码
        with self.assertRaises(TypeError):
            bool(Var("n"))

    def test_repr(self):
        self.assertEqual(repr(Var("n") != 1), "(Var('n') != 1)")


class TestVars(PromptkitTestCase):
    def test_newvar_declares_with_default(self):
        ctx = Workflow(NewVar("state", default="进行中")).run()
        self.assertEqual(ctx["state"], "进行中")

    def test_newvar_does_not_overwrite(self):
        ctx = Workflow(
            NewVar("n", default=0), Set("n", 5), NewVar("n", default=99)
        ).run()
        self.assertEqual(ctx["n"], 5)

    def test_set_with_expression(self):
        ctx = Workflow(NewVar("n", default=0), Set("n", Var("n") + 1)).run()
        self.assertEqual(ctx["n"], 1)

    def test_newvar_handle_is_an_expression(self):
        state = NewVar("state", default="进行中")
        self.assertTrue((state == "进行中").eval(Context(data={"state": "进行中"})))

    def test_var_val_form(self):
        ctx = Workflow(NewVar("n"), Var(name="n", val=7)).run()
        self.assertEqual(ctx["n"], 7)

    def test_var_accepts_handle(self):
        n = NewVar("n", default=1)
        ctx = Workflow(n, Var(name=n, val=2)).run()
        self.assertEqual(ctx["n"], 2)

    def test_reading_undeclared_raises(self):
        with self.assertRaises(UndefinedVariable):
            Workflow(Set("x", Var("nope") + 1)).run()

    def test_trace(self):
        ctx = Workflow(NewVar("n", default=0), Set("n", 1)).run()
        self.assertEqual(ctx.trace, ["new_var:n", "set:n"])


class TestFunctionFactory(PromptkitTestCase):
    def test_bind_and_run_prompt(self):
        agent = FakeAgent(["你好"])
        Ask = Function(args=("job",), name="ask").Do(
            Prompt(role="user", content="面试一个{{job}}岗位。")
        )
        ctx = Workflow(Ask(job="后端")).run(agent=agent)
        self.assertEqual(
            agent.prompts[0], {"role": "user", "content": "面试一个后端岗位。"}
        )
        self.assertEqual(ctx["ask"], "你好")

    def test_unknown_argument_raises(self):
        Ask = Function(args=("job",), name="ask").Do(Prompt(role="user", content="x"))
        with self.assertRaises(ArgumentError):
            Ask(joob="后端")

    def test_missing_argument_raises(self):
        Ask = Function(args=("job",), name="ask").Do(Prompt(role="user", content="x"))
        with self.assertRaises(ArgumentError):
            Ask()

    def test_duplicate_args_raise(self):
        with self.assertRaises(ArgumentError):
            Function(args=("a", "a"))

    def test_args_are_scoped_to_the_step(self):
        agent = FakeAgent(["ok"])
        Ask = Function(args=("job",), name="ask").Do(
            Prompt(role="user", content="{{job}}")
        )
        ctx = Workflow(NewVar("job", default="外层"), Ask(job="内层")).run(agent=agent)
        self.assertEqual(agent.prompts[0]["content"], "内层")
        # 执行完之后，参数不再可见，外层变量没有被污染
        self.assertEqual(ctx["job"], "外层")

    def test_do_requires_body(self):
        with self.assertRaises(ArgumentError):
            Function(name="x").Do()

    def test_output_model(self):
        agent = FakeAgent(['{"label": "退款"}'])
        Classify = Function(args=(), name="classify", output_model=Intent).Do(
            Prompt(role="user", content="判断意图")
        )
        ctx = Workflow(Classify()).run(agent=agent)
        self.assertEqual(ctx["classify"].label, "退款")

    def test_do_accepts_plain_steps(self):
        Mark = Function(args=(), name="mark").Do(
            FunctionStep(lambda _c: "done", output_key="out")
        )
        ctx = Workflow(Mark()).run()
        self.assertEqual(ctx["out"], "done")

    def test_expression_argument_is_evaluated(self):
        # 传 Var("n") 时应该注入 n 的当前值，而不是 VarExpr 对象
        agent = FakeAgent(["ok"])
        Ask = Function(args=("n",), name="ask").Do(
            Prompt(role="user", content="n={{n}}")
        )
        Workflow(NewVar("n", default=1), Set("n", 2), Ask(n=Var("n"))).run(agent=agent)
        self.assertEqual(agent.prompts[0]["content"], "n=2")

    def test_expression_argument_refreshed_each_iteration(self):
        agent = FakeAgent(["ok", "ok", "ok"])
        Ask = Function(args=("n",), name="ask").Do(
            Prompt(role="user", content="n={{n}}")
        )
        Workflow(
            NewVar("n", default=0),
            While(Var("n") < 3, max_iterations=10).Do(
                Ask(n=Var("n")),
                Set("n", Var("n") + 1),
            ),
        ).run(agent=agent)
        self.assertEqual(
            [p["content"] for p in agent.prompts], ["n=0", "n=1", "n=2"]
        )

    def test_steps_inside_definition_are_reusable(self):
        agent = FakeAgent(["a", "b"])
        Ask = Function(args=(), name="ask").Do(Prompt(role="user", content="问"))
        ctx = Workflow(Ask(), Ask()).run(agent=agent)
        self.assertEqual(ctx["ask"], "b")
        self.assertEqual(len(agent.prompts), 2)


class TestWhile(PromptkitTestCase):
    def test_loop_runs_until_condition_false(self):
        ctx = Workflow(
            NewVar("n", default=0),
            While(Var("n") < 3, max_iterations=10).Do(Set("n", Var("n") + 1)),
        ).run()
        self.assertEqual(ctx["n"], 3)

    def test_condition_never_true_skips_body(self):
        ctx = Workflow(NewVar("n", default=0), While(False).Do(Set("n", 1))).run()
        self.assertEqual(ctx["n"], 0)

    def test_loop_limit_raises_instead_of_hanging(self):
        # 循环体从不更新条件变量 -> 必须报错，而不是死循环
        with self.assertRaises(LoopLimitExceeded):
            Workflow(
                NewVar("n", default=0),
                While(Var("n") < 100, max_iterations=5).Do(Set("other", 1)),
            ).run()

    def test_condition_object(self):
        ctx = Workflow(
            NewVar("state", default="进行中"),
            While(Condition(Var("state") != "结束"), max_iterations=3).Do(
                Set("state", "结束")
            ),
        ).run()
        self.assertEqual(ctx["state"], "结束")

    def test_max_iterations_must_be_positive(self):
        with self.assertRaises(ValueError):
            While(True, max_iterations=0)

    def test_trace(self):
        ctx = Workflow(
            NewVar("n", default=0),
            While(Var("n") < 2, max_iterations=5).Do(Set("n", Var("n") + 1)),
        ).run()
        self.assertEqual(
            ctx.trace,
            ["new_var:n", "while:1", "set:n", "while:2", "set:n", "while:done"],
        )


class TestAgentReply(PromptkitTestCase):
    def test_runs_step_and_returns_reply(self):
        agent = FakeAgent(["问题一"])
        Ask = Function(args=(), name="ask").Do(Prompt(role="user", content="问一个问题"))
        ctx = Workflow(NewVar("q"), Set("q", Ask().agent_reply())).run(agent=agent)
        self.assertEqual(ctx["q"], "问题一")
        self.assertEqual(ctx["ask"], "问题一")

    def test_reevaluated_every_iteration(self):
        agent = FakeAgent(["一", "二", "三"])
        Ask = Function(args=(), name="ask").Do(Prompt(role="user", content="问"))
        ctx = Workflow(
            NewVar("n", default=0),
            NewVar("last"),
            While(Var("n") < 3, max_iterations=10).Do(
                Set("last", Ask().agent_reply()),
                Set("n", Var("n") + 1),
            ),
        ).run(agent=agent)
        self.assertEqual(ctx["last"], "三")
        self.assertEqual(len(agent.prompts), 3)

    def test_structured_model_reply(self):
        agent = FakeAgent(['{"label": "投诉"}'])
        Classify = Function(args=(), name="cls", output_model=Intent).Do(
            Prompt(role="user", content="判断意图")
        )
        ctx = Workflow(Set("r", Classify().agent_reply())).run(agent=agent)
        self.assertEqual(ctx["r"].label, "投诉")


class TestCoreExports(PromptkitTestCase):
    def test_all_names_resolve(self):
        from promptkit import core

        missing = [n for n in core.__all__ if not hasattr(core, n)]
        self.assertEqual(missing, [])

    def test_star_import_gives_the_whole_dsl(self):
        namespace = {}
        exec("from promptkit.core import *", namespace)
        for name in (
            "Workflow",
            "Function",
            "Prompt",
            "While",
            "Var",
            "NewVar",
            "Set",
            "Condition",
            "If",
            "LLMStep",
            "FunctionStep",
        ):
            self.assertIn(name, namespace)


class TestInteractiveSteps(PromptkitTestCase):
    def test_read_is_lazy(self):
        # 构造工作流时不应该去读 stdin
        step = Set("job", Read("岗位："))
        self.assertIsInstance(step, SetStep)

    def test_read_gets_value_at_run_time(self):
        with mock.patch("builtins.input", return_value="  后端工程师  ") as fake:
            ctx = Workflow(Set("job", Read("岗位："))).run()
        fake.assert_called_once_with("岗位：")
        self.assertEqual(ctx["job"], "后端工程师")  # 默认 strip

    def test_read_keeps_whitespace_when_asked(self):
        with mock.patch("builtins.input", return_value="  x  "):
            ctx = Workflow(Set("v", Read(strip=False))).run()
        self.assertEqual(ctx["v"], "  x  ")

    def test_echo_prints(self):
        with mock.patch("builtins.print") as fake:
            ctx = Workflow(
                NewVar("name", default="小明"), Echo("你好 ", Var("name"))
            ).run()
        self.assertEqual(printed(fake), ["你好 小明"])
        self.assertEqual(ctx.trace, ["new_var:name", "echo"])

    def test_read_inside_loop(self):
        with mock.patch("builtins.input", side_effect=["a", "b", "结束"]) as fake:
            ctx = Workflow(
                NewVar("state", default="进行中"),
                NewVar("last"),
                While(Var("state") != "结束", max_iterations=5).Do(
                    Set("last", Read()),
                    Set("state", Var("last")),
                ),
            ).run()
        self.assertEqual(ctx["last"], "结束")
        self.assertEqual(fake.call_count, 3)


class TestDefaultAgent(PromptkitTestCase):
    def test_agent_registers_itself(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            agent = Agent()
        self.assertIs(Agent.default(), agent)

    def test_run_without_agent_uses_default(self):
        # Agent() 注册成默认之后，wf.run() 不需要传参
        saved = Agent.default()
        try:
            agent = Agent(api_key="sk-fake")
            ctx = Workflow().run()
            self.assertIs(Agent.default(), agent)
            self.assertEqual(ctx["messages"], [])
        finally:
            Agent.set_default(saved)

    def test_missing_key_raises_with_hint(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            # .env 读取现在归 config 管，要 patch 那边才有效
            with mock.patch("promptkit.config.load_env_file"):
                with self.assertRaises(ValueError) as cm:
                    Agent()
        self.assertIn("DEEPSEEK_API_KEY", str(cm.exception))

    def test_explicit_key_wins(self):
        agent = Agent(api_key="sk-explicit", register_as_default=False)
        self.assertEqual(agent.api_key, "sk-explicit")

    def test_env_file_loader_does_not_override_env(self):
        import tempfile

        saved = os.environ.get("PROMPTKIT_TEST_KEY")
        try:
            os.environ["PROMPTKIT_TEST_KEY"] = "from-env"
            with tempfile.NamedTemporaryFile(
                "w", suffix=".env", delete=False, encoding="utf-8"
            ) as handle:
                handle.write("PROMPTKIT_TEST_KEY=from-file\n")
                handle.write("# comment\n\n")
                handle.write("PROMPTKIT_OTHER=hello\n")
                path = handle.name
            load_env_file(path)
            self.assertEqual(os.environ["PROMPTKIT_TEST_KEY"], "from-env")
            self.assertEqual(os.environ["PROMPTKIT_OTHER"], "hello")
        finally:
            os.unlink(path)
            os.environ.pop("PROMPTKIT_OTHER", None)
            if saved is None:
                os.environ.pop("PROMPTKIT_TEST_KEY", None)
            else:
                os.environ["PROMPTKIT_TEST_KEY"] = saved


class TestVarTypes(PromptkitTestCase):
    def test_number_declaration_default(self):
        ctx = Workflow(Number("n")).run()
        self.assertEqual(ctx["n"], 0)

    def test_str_declaration_default(self):
        ctx = Workflow(Str("s")).run()
        self.assertEqual(ctx["s"], "")

    def test_number_declaration_with_default(self):
        ctx = Workflow(Number("n", default=1.5)).run()
        self.assertEqual(ctx["n"], 1.5)

    def test_read_string_is_coerced_to_number(self):
        with mock.patch("builtins.input", return_value="12"):
            ctx = Workflow(Number("a"), Set("a", Read())).run()
        self.assertEqual(ctx["a"], 12)
        self.assertIsInstance(ctx["a"], int)

    def test_decimal_string_becomes_float(self):
        ctx = Workflow(Number("a"), Set("a", "1.5")).run()
        self.assertEqual(ctx["a"], 1.5)
        self.assertIsInstance(ctx["a"], float)

    def test_addition_is_arithmetic_not_concatenation(self):
        with mock.patch("builtins.input", side_effect=["12", "34"]):
            ctx = Workflow(
                Number("a"),
                Number("b"),
                Set("a", Read()),
                Set("b", Read()),
            ).run()
        self.assertEqual(Var("a").eval(ctx) + Var("b").eval(ctx), 46)
        self.assertNotEqual(Var("a").eval(ctx) + Var("b").eval(ctx), "1234")

    def test_bad_number_raises(self):
        with self.assertRaises(VarTypeError) as cm:
            Workflow(Number("a"), Set("a", "abc")).run()
        self.assertIn("'a'", str(cm.exception))

    def test_str_coerces_numbers(self):
        ctx = Workflow(Str("s"), Set("s", 42)).run()
        self.assertEqual(ctx["s"], "42")
        self.assertIsInstance(ctx["s"], str)

    def test_str_rejects_containers(self):
        with self.assertRaises(VarTypeError):
            Workflow(Str("s"), Set("s", [1, 2])).run()

    def test_newvar_type_keyword_is_equivalent(self):
        ctx = Workflow(NewVar("n", type=Number), Set("n", "7")).run()
        self.assertEqual(ctx["n"], 7)

    def test_newvar_rejects_unknown_type(self):
        with self.assertRaises(TypeError):
            NewVar("n", type="Number")

    def test_untyped_variable_is_not_coerced(self):
        # 向后兼容：不声明类型就原样保存
        ctx = Workflow(NewVar("x"), Set("x", "12")).run()
        self.assertEqual(ctx["x"], "12")
        self.assertIsInstance(ctx["x"], str)

    def test_run_kwargs_are_coerced_on_declare(self):
        ctx = Workflow(Number("n"), Set("n", Var("n") + 1)).run(n="41")
        self.assertEqual(ctx["n"], 42)

    def test_runtime_type_error_on_wrong_kwarg(self):
        with self.assertRaises(VarTypeError):
            Workflow(Number("n")).run(n="abc")

    def test_call_invokes_python_function(self):
        def add(a, b):
            return a + b

        ctx = Workflow(
            Number("a"),
            Number("b"),
            Set("a", "12"),
            Set("b", "34"),
            Set("sum", Call(add, Var("a"), Var("b"))),
        ).run()
        self.assertEqual(ctx["sum"], 46)

    def test_call_in_echo(self):
        def double(x):
            return x * 2

        with mock.patch("builtins.print") as fake:
            Workflow(
                Number("n"), Set("n", "5"), Echo("结果=", Call(double, Var("n")))
            ).run()
        self.assertEqual(printed(fake), ["结果=10"])

    def test_call_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            Call(42, Var("n"))

    def test_call_supports_kwargs(self):
        ctx = Workflow(
            Number("b"),
            Set("b", "2"),
            Set("r", Call(round, 3.14159, ndigits=Var("b"))),
        ).run()
        self.assertEqual(ctx["r"], 3.14)

    def test_numbers_work_in_conditions(self):
        ctx = Workflow(
            Number("n"),
            Set("n", "5"),
            If(Var("n") > 3).Then(Set("big", "yes")),
        ).run()
        self.assertEqual(ctx["big"], "yes")

    def test_step_cannot_be_used_as_a_value(self):
        # 这是很容易犯的错：把步骤当成值塞进 Set/Echo
        with self.assertRaises(TypeError) as cm:
            Set("x", If(True).Then(Set("y", 1)))
        self.assertIn("步骤", str(cm.exception))


class TestSkipStep(PromptkitTestCase):
    def test_skip_does_nothing(self):
        ctx = Workflow(NewVar("x", default=1), If(True).Then(SkipStep())).run()
        self.assertEqual(ctx["x"], 1)

    def test_skip_makes_an_explicit_empty_branch(self):
        ctx = Workflow(
            NewVar("out", default="untouched"),
            If(False).Then(Set("a", 1)),
            If(Var("out") == "untouched").Then(SkipStep()),
            If(Var("out") == "untouched").Then(Set("b", 2)),
        ).run()
        self.assertNotIn("a", ctx)
        self.assertEqual(ctx["b"], 2)

    def _fizzbuzz(self, n):
        '''like.py 里那种写法：空 Then 分支 + 第二个 If 补 Buzz。'''
        wf = Workflow(
            Number("n"),
            Set("n", n),
            If(Not(BinaryExpr("%", Var("n"), 3) == 0)).Then(
                SkipStep()
            ).Else(
                If(BinaryExpr("%", Var("n"), 5) == 0).Then(
                    Echo("FizzBuzz")
                ).Else(
                    Echo("Fizz")
                )
            ),
            If(
                And(
                    Not(BinaryExpr("%", Var("n"), 3) == 0),
                    BinaryExpr("%", Var("n"), 5) == 0,
                )
            ).Then(Echo("Buzz")),
        )
        with mock.patch("builtins.print") as fake:
            wf.run()
        return "".join(str(call.args[0]) for call in fake.call_args_list)

    def test_fizzbuzz_pattern(self):
        self.assertEqual(self._fizzbuzz(3), "Fizz")
        self.assertEqual(self._fizzbuzz(5), "Buzz")
        self.assertEqual(self._fizzbuzz(15), "FizzBuzz")
        self.assertEqual(self._fizzbuzz(7), "")


class TestElseIf(PromptkitTestCase):
    def test_picks_first_matching_branch(self):
        wf = Workflow(
            NewVar("n", default=0),
            If(Var("n") == 1).Then(const("one"))
            .ElseIf(Var("n") == 2).Then(const("two"))
            .ElseIf(Var("n") == 3).Then(const("three"))
            .Else(const("other")),
        )
        for n, expected in [(1, "one"), (2, "two"), (3, "three"), (9, "other")]:
            self.assertEqual(wf.run(n=n)["r"], expected)

    def test_without_else_nothing_happens(self):
        wf = Workflow(If(False).Then(const("a")).ElseIf(False).Then(const("b")))
        self.assertNotIn("r", wf.run())

    def test_then_applies_to_the_last_branch(self):
        wf = Workflow(
            NewVar("n", default=2),
            If(Var("n") == 1).Then(const("wrong")).ElseIf(Var("n") == 2).Then(
                const("right")
            ),
        )
        self.assertEqual(wf.run()["r"], "right")

    def test_trace_labels_each_branch(self):
        wf = Workflow(
            NewVar("n", default=0),
            If(Var("n") == 1).Then(const("a"))
            .ElseIf(Var("n") == 2).Then(const("b"))
            .Else(const("c")),
        )
        self.assertEqual(wf.run(n=1).trace, ["new_var:n", "if:then", "function"])
        self.assertEqual(wf.run(n=2).trace, ["new_var:n", "if:elif1", "function"])
        self.assertEqual(wf.run(n=9).trace, ["new_var:n", "if:else", "function"])

    def test_elseif_does_not_mutate_original(self):
        base = If(True)
        extended = base.ElseIf(False)
        self.assertEqual(len(base.branches), 1)
        self.assertEqual(len(extended.branches), 2)


class TestForEach(PromptkitTestCase):
    def test_iterates_a_range(self):
        seen = []
        Workflow(
            ForEach("i", range(1, 4)).Do(FunctionStep(lambda c: seen.append(c["i"])))
        ).run()
        self.assertEqual(seen, [1, 2, 3])

    def test_iterates_an_expression(self):
        seen = []
        Workflow(
            NewVar("items", default=["a", "b"]),
            ForEach("x", Var("items")).Do(FunctionStep(lambda c: seen.append(c["x"]))),
        ).run()
        self.assertEqual(seen, ["a", "b"])

    def test_loop_variable_is_type_coerced(self):
        seen = []
        Workflow(
            Number("i"),
            ForEach("i", ["1", "2"]).Do(FunctionStep(lambda c: seen.append(c["i"]))),
        ).run()
        self.assertEqual(seen, [1, 2])

    def test_empty_iterable_does_nothing(self):
        seen = []
        Workflow(ForEach("i", []).Do(FunctionStep(lambda c: seen.append(c["i"])))).run()
        self.assertEqual(seen, [])

    def test_non_iterable_raises(self):
        with self.assertRaises(TypeError) as cm:
            Workflow(ForEach("i", 42).Do(SkipStep())).run()
        self.assertIn("可迭代", str(cm.exception))

    def test_max_iterations_guard(self):
        with self.assertRaises(LoopLimitExceeded):
            Workflow(
                ForEach("i", range(100), max_iterations=3).Do(SkipStep())
            ).run()

    def test_trace(self):
        ctx = Workflow(ForEach("i", range(2)).Do(SkipStep())).run()
        self.assertEqual(ctx.trace, ["foreach:1", "foreach:2", "foreach:done"])


class TestBreakContinue(PromptkitTestCase):
    def test_break_stops_the_loop(self):
        seen = []
        Workflow(
            ForEach("i", range(1, 100)).Do(
                If(Var("i") > 3).Then(Break()),
                FunctionStep(lambda c: seen.append(c["i"])),
            )
        ).run()
        self.assertEqual(seen, [1, 2, 3])

    def test_continue_skips_the_rest(self):
        seen = []
        Workflow(
            ForEach("i", range(1, 7)).Do(
                If(Var("i") % 2 == 0).Then(Continue()),
                FunctionStep(lambda c: seen.append(c["i"])),
            )
        ).run()
        self.assertEqual(seen, [1, 3, 5])

    def test_break_works_in_while(self):
        seen = []
        Workflow(
            NewVar("n", default=0),
            While(True, max_iterations=50).Do(
                Set("n", Var("n") + 1),
                If(Var("n") >= 3).Then(Break()),
                FunctionStep(lambda c: seen.append(c["n"])),
            ),
        ).run()
        self.assertEqual(seen, [1, 2])

    def test_break_only_exits_the_innermost_loop(self):
        pairs = []
        Workflow(
            ForEach("i", range(1, 3)).Do(
                ForEach("j", range(1, 100)).Do(
                    If(Var("j") > 2).Then(Break()),
                    FunctionStep(lambda c: pairs.append((c["i"], c["j"]))),
                )
            )
        ).run()
        self.assertEqual(pairs, [(1, 1), (1, 2), (2, 1), (2, 2)])

    def test_break_outside_a_loop_is_reported(self):
        with self.assertRaises(WorkflowError) as cm:
            Workflow(Break()).run()
        self.assertIn("循环", str(cm.exception))

    def test_continue_outside_a_loop_is_reported(self):
        with self.assertRaises(WorkflowError):
            Workflow(Continue()).run()


class Person:
    '''用来验证「一般 Python 对象」的普通类。'''

    def __init__(self, name, age, tags=()):
        self.name = name
        self.age = age
        self.tags = list(tags)

    def greet(self, who):
        return f"{self.name} -> {who}"

    def __repr__(self):
        return f"Person({self.name!r})"


class TestPythonObjects(PromptkitTestCase):
    def test_store_and_retrieve_by_identity(self):
        ann = Person("Ann", 30)
        ctx = Workflow(NewVar("p", default=ann)).run()
        self.assertIs(ctx["p"], ann)  # 存的是同一个对象，不是拷贝

    def test_run_kwargs_accept_objects(self):
        ann = Person("Ann", 30)
        ctx = Workflow(NewVar("p")).run(p=ann)
        self.assertIs(ctx["p"], ann)

    def test_step_output_can_be_an_object(self):
        ctx = Workflow(
            FunctionStep(lambda _c: Person("Cid", 9), output_key="p")
        ).run()
        self.assertEqual(ctx["p"].name, "Cid")

    def test_indexing(self):
        ctx = Workflow(
            NewVar("xs", default=[10, 20, 30]),
            Set("first", Var("xs")[0]),
            Set("last", Var("xs")[-1]),
        ).run()
        self.assertEqual(ctx["first"], 10)
        self.assertEqual(ctx["last"], 30)

    def test_slicing(self):
        ctx = Workflow(
            NewVar("xs", default=[10, 20, 30, 40]),
            Set("mid", Var("xs")[1:3]),
        ).run()
        self.assertEqual(ctx["mid"], [20, 30])

    def test_dict_key_access(self):
        ctx = Workflow(
            NewVar("d", default={"a": 1, "b": 2}),
            Set("v", Var("d")["b"]),
        ).run()
        self.assertEqual(ctx["v"], 2)

    def test_dynamic_key_is_an_expression(self):
        ctx = Workflow(
            NewVar("d", default={"a": 1}),
            NewVar("k", default="a"),
            Set("v", Var("d")[Var("k")]),
        ).run()
        self.assertEqual(ctx["v"], 1)

    def test_attribute_access(self):
        ctx = Workflow(
            NewVar("p", default=Person("Ann", 30)),
            Set("name", Var("p").attr("name")),
            Set("adult", Var("p").attr("age") >= 18),
        ).run()
        self.assertEqual(ctx["name"], "Ann")
        self.assertTrue(Var("adult").eval(ctx))

    def test_nested_attribute_and_index(self):
        ctx = Workflow(
            NewVar("p", default=Person("Ann", 30, ["dev", "py"])),
            Set("first_tag", Var("p").attr("tags")[0]),
        ).run()
        self.assertEqual(ctx["first_tag"], "dev")

    def test_method_call(self):
        ctx = Workflow(
            NewVar("p", default=Person("Ann", 30)),
            NewVar("s", default="Hello World"),
            Set("greeting", Var("p").method("greet", "Bob")),
            Set("shout", Var("s").method("upper")),
            Set("swapped", Var("s").method("replace", "World", "DSL")),
        ).run()
        self.assertEqual(ctx["greeting"], "Ann -> Bob")
        self.assertEqual(ctx["shout"], "HELLO WORLD")
        self.assertEqual(ctx["swapped"], "Hello DSL")

    def test_len(self):
        ctx = Workflow(
            NewVar("xs", default=[1, 2, 3]), Set("n", Len(Var("xs")))
        ).run()
        self.assertEqual(ctx["n"], 3)

    def test_call_with_a_callable_from_a_variable(self):
        ctx = Workflow(
            NewVar("f", default=lambda x: x * 2),
            NewVar("n", default=21),
            Set("r", Call(Var("f"), Var("n"))),
        ).run()
        self.assertEqual(ctx["r"], 42)

    def test_for_over_an_attribute(self):
        seen = []
        Workflow(
            NewVar("p", default=Person("Ann", 30, ["dev", "py"])),
            ForEach("tag", Var("p").attr("tags")).Do(
                FunctionStep(lambda c: seen.append(c["tag"]))
            ),
        ).run()
        self.assertEqual(seen, ["dev", "py"])

    def test_objects_compare_in_conditions(self):
        ctx = Workflow(
            NewVar("p", default=Person("Ann", 30)),
            NewVar("q", default=Person("Ann", 30)),
            If(Var("p").attr("age") == Var("q").attr("age")).Then(Set("same", True)),
        ).run()
        self.assertTrue(ctx["same"])

    def test_expr_is_not_iterable(self):
        # 有了 __getitem__ 之后必须堵住老的迭代协议，否则会死循环
        with self.assertRaises(TypeError):
            list(Var("xs"))

    def test_expr_is_not_a_sequence(self):
        # 所以 coerce_steps 不会把表达式误当成步骤列表
        from collections.abc import Sequence

        self.assertNotIsInstance(Var("xs"), Sequence)

    def test_passing_an_expr_as_a_step_raises(self):
        with self.assertRaises(TypeError):
            Workflow().add(Var("x"))
        with self.assertRaises(TypeError):
            If(True).Then(Var("x"))

    def test_typed_variables_still_reject_incompatible_objects(self):
        with self.assertRaises(VarTypeError):
            Workflow(Str("s"), Set("s", [1, 2])).run()
        with self.assertRaises(VarTypeError):
            Workflow(Number("n"), Set("n", Person("Ann", 30))).run()


class TestStepTree(PromptkitTestCase):
    def test_workflow_repr_is_the_tree(self):
        wf = Workflow(
            Number("input"),
            Set("input", Read("")),
            If(Var("input") % 15 == 0).Then(Echo("FizzBuzz")).Else(Echo("other")),
        )
        self.assertEqual(
            repr(wf).splitlines(),
            [
                "Workflow",
                "├─ Number('input')",
                "├─ Set('input', Read(''))",
                "└─ If",
                "   ├─ if ((Var('input') % 15) == 0)",
                "   │  └─ Echo('FizzBuzz')",
                "   └─ else",
                "      └─ Echo('other')",
            ],
        )

    def test_render_matches_repr(self):
        wf = Workflow(Echo("a"))
        self.assertEqual(wf.render(), repr(wf))

    def test_condition_text_is_readable(self):
        # 条件归一化之后仍然保留可读文本，而不是 <function <lambda>>
        self.assertIn("if (Var('n') > 1)", Workflow(If(Var("n") > 1).Then(SkipStep())).render())

    def test_nested_if_elseif_branches(self):
        wf = Workflow(
            If(Var("a") == 1).Then(SkipStep())
            .ElseIf(Var("a") == 2).Then(SkipStep())
            .Else(SkipStep())
        )
        text = wf.render()
        self.assertIn("├─ if (Var('a') == 1)", text)
        self.assertIn("├─ elif (Var('a') == 2)", text)
        self.assertIn("└─ else", text)

    def test_while_and_for_bodies_are_nested(self):
        wf = Workflow(
            While(Var("n") < 3, max_iterations=5).Do(
                ForEach("i", range(2)).Do(Break()),
            )
        )
        self.assertEqual(
            wf.render().splitlines(),
            [
                "Workflow",
                "└─ While (Var('n') < 3)",
                "   └─ ForEach 'i' in range(0, 2)",
                "      └─ Break",
            ],
        )

    def test_bound_step_shows_args_and_body(self):
        Ask = Function(args=("job",), name="ask").Do(
            Prompt(role="user", content="面试{{job}}")
        )
        text = Workflow(Ask(job="后端")).render()
        self.assertIn("ask(job='后端')", text)
        self.assertIn("LLM(面试{{job}})", text)

    def test_leaf_steps_are_recognisable(self):
        wf = Workflow(
            MessageStep(role="system", content="你是助手"),
            FunctionStep(lambda _c: None, output_key="out"),
            Echo("hi"),
            Continue(),
        )
        text = wf.render()
        self.assertIn("Message('system', 你是助手)", text)
        self.assertIn("FunctionStep(<lambda>) -> out", text)
        self.assertIn("Echo('hi')", text)
        self.assertIn("Continue", text)

    def test_tree_does_not_execute_anything(self):
        # 渲染是纯展示，不能触发副作用或求值
        with mock.patch("builtins.input") as fake:
            Workflow(Set("x", Read("输入："))).render()
        fake.assert_not_called()
        with mock.patch("builtins.print") as fake:
            Workflow(Echo("hi")).render()
        fake.assert_not_called()

    def test_long_content_is_shortened_to_one_line(self):
        node_text = Workflow(
            MessageStep(role="user", content="x" * 200)
        ).render().splitlines()[1]
        self.assertLess(len(node_text), 80)
        self.assertIn("…", node_text)

    def test_workflow_with_custom_name(self):
        self.assertEqual(Workflow(name="main").render(), "Workflow('main')")


class TestEchoTemplating(PromptkitTestCase):
    def test_braces_are_interpolated(self):
        with mock.patch("builtins.print") as fake:
            Workflow(NewVar("score", default=0.9), Echo("情感分数：{{score}}")).run()
        self.assertEqual(printed(fake), ["情感分数：0.9"])

    def test_equivalent_to_explicit_var(self):
        with mock.patch("builtins.print") as fake:
            Workflow(NewVar("n", default=3), Echo("n={{n}}")).run()
        with mock.patch("builtins.print") as fake2:
            Workflow(NewVar("n", default=3), Echo("n=", Var("n"))).run()
        self.assertEqual(fake.call_args, fake2.call_args)

    def test_missing_variable_raises_with_hint(self):
        with self.assertRaises(KeyError) as cm:
            Workflow(NewVar("a", default=1), Echo("{{nope}}")).run()
        message = str(cm.exception)
        self.assertIn("nope", message)
        self.assertIn("a", message)  # 提示里列出可用变量

    def test_dollar_text_is_left_alone(self):
        # Echo 是普通文本输出，$100 不该被当成变量
        with mock.patch("builtins.print") as fake:
            Workflow(Echo("价格是 $100，折扣 ${50}")).run()
        self.assertEqual(printed(fake), ["价格是 $100，折扣 ${50}"])

    def test_single_braces_are_literal(self):
        with mock.patch("builtins.print") as fake:
            Workflow(Echo('{"label": "退款"}')).run()
        self.assertEqual(printed(fake), ['{"label": "退款"}'])

    def test_non_string_parts_still_work(self):
        with mock.patch("builtins.print") as fake:
            Workflow(NewVar("n", default=7), Echo("n=", Var("n"), " 结束")).run()
        self.assertEqual(printed(fake), ["n=7 结束"])

    def test_render_text_allow_dollar_flag(self):
        self.assertEqual(render_text("$a and {{b}}", {"a": 1, "b": 2}), "1 and 2")
        self.assertEqual(
            render_text("$a and {{b}}", {"b": 2}, allow_dollar=False), "$a and 2"
        )

    def test_prompt_still_supports_dollar_form(self):
        self.assertEqual(
            Prompt(role="user", content="$a/{{b}}").render(a=1, b=2)["content"],
            "1/2",
        )


class TestDollarVariableVsMath(PromptkitTestCase):
    '''回归：`$name` 变量语法和 LaTeX 的 `$...$` 冲突。

    用户写 `Custom("公式请使用$或$$包裹")`，结果 `$或$` 被当成变量 `或`，
    直接 `KeyError` 把整个工作流打挂。

    定下的规则：**双花括号严格，美元符号宽松**。
    `{{name}}` 语法无歧义，写错就报错；`$name` 和公式 / 价格天生冲突，
    找不到变量就原样留着。
    '''

    def test_the_exact_user_bug(self):
        '''`Custom("...$或$$...")` 不该抛 KeyError。'''
        text = join_cautions([Custom("公式请使用$或$$包裹")])
        rendered = Prompt(role="user", content=text).render(files="x")
        self.assertEqual(rendered["content"], "公式请使用$或$$包裹")

    def test_caution_with_dollar_survives_an_llm_step(self):
        step = LLMStep(
            prompt=Prompt(role="user", content="请证明"),
            output_key="out",
            cautions=[Custom("公式请使用$或$$包裹")],
        )
        self.assertIn("公式请使用$或$$包裹", step.build_prompt().content)

    def test_currency_is_left_alone(self):
        for text in ("价格是 $100", "金额 $1.5 元", "总共 $12000"):
            self.assertEqual(render_text(text, {}), text)

    def test_currency_between_two_dollars_is_left_alone(self):
        text = "价格是 $100，折扣 $50"
        self.assertEqual(render_text(text, {}), text)

    def test_latex_is_left_alone(self):
        for text in ("公式 $x_1$ 是价格", "块级 $$y = ax + b$$", "用 $ 或 $$ 包裹"):
            self.assertEqual(render_text(text, {}), text)

    def test_prose_dollar_is_left_alone(self):
        self.assertEqual(render_text("请用 $ 包裹", {}), "请用 $ 包裹")

    def test_missing_dollar_variable_is_kept(self):
        self.assertEqual(render_text("$nope 在这里", {}), "$nope 在这里")

    def test_existing_dollar_variable_still_works(self):
        self.assertEqual(render_text("你好 $name", {"name": "小明"}), "你好 小明")
        self.assertEqual(render_text("你好 ${name}", {"name": "小明"}), "你好 小明")

    def test_dollar_before_dollar_is_never_a_variable(self):
        '''`$x_1$` 即使真有个变量叫 x_1 也不该被替换 —— 那是公式。'''
        self.assertEqual(render_text("$x_1$", {"x_1": 999}), "$x_1$")
        self.assertEqual(render_text("$$y$$", {"y": 999}), "$$y$$")

    def test_braces_are_still_strict(self):
        '''`{{}}` 无歧义，写错必须立刻报错（这是它的价值）。'''
        with self.assertRaises(KeyError) as cm:
            render_text("{{nope}}", {"name": "小明"})
        self.assertIn("nope", str(cm.exception))
        self.assertIn("name", str(cm.exception))  # 列出可用的

    def test_braces_still_take_dot_paths(self):
        self.assertEqual(
            render_text("{{a.b}}", {"a": {"b": 1}}), "1"
        )

    def test_dollar_does_not_take_dot_paths(self):
        # $ 后面跟数字很常见（$1.5），所以放宽会误伤；这是早就有的约定
        self.assertEqual(render_text("$a.b", {"a": 1}), "1.b")

    def test_allow_dollar_false_ignores_all_dollar_forms(self):
        self.assertEqual(
            render_text("$a ${b} {{c}}", {"a": 1, "b": 2, "c": 3}, allow_dollar=False),
            "$a ${b} 3",
        )


class TestCStyleFor(PromptkitTestCase):
    def test_counts_like_c(self):
        # for (i = 0; i < 5; i++) { print(i); }
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 5, Inc("i")).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            )
        ).run()
        self.assertEqual(seen, [0, 1, 2, 3, 4])

    def test_init_runs_once(self):
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 3, Inc("i")).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            )
        ).run()
        self.assertEqual(seen, [0, 1, 2])

    def test_can_count_down(self):
        seen = []
        Workflow(
            For(Set("i", 3), Var("i") > 0, Dec("i")).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            )
        ).run()
        self.assertEqual(seen, [3, 2, 1])

    def test_step_by_two(self):
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 7, Inc("i", 2)).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            )
        ).run()
        self.assertEqual(seen, [0, 2, 4, 6])

    def test_condition_false_initially_runs_once_for_init_only(self):
        seen = []
        Workflow(
            For(Set("i", 99), Var("i") < 5, Inc("i")).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            )
        ).run()
        self.assertEqual(seen, [])
        self.assertEqual(Workflow(For(Set("i", 99), Var("i") < 5, Inc("i"))).run()["i"], 99)

    def test_continue_still_runs_the_step(self):
        # C 里 continue 会跳到 increment，所以不会死循环
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 4, Inc("i")).Do(
                If(Var("i") % 2 == 0).Then(Continue()),
                FunctionStep(lambda c: seen.append(c["i"])),
            )
        ).run()
        self.assertEqual(seen, [1, 3])

    def test_break_exits(self):
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 100, Inc("i")).Do(
                If(Var("i") >= 3).Then(Break()),
                FunctionStep(lambda c: seen.append(c["i"])),
            )
        ).run()
        self.assertEqual(seen, [0, 1, 2])

    def test_multi_step_init_and_step(self):
        seen = []
        Workflow(
            For([Set("i", 0), Set("total", 0)], Var("i") < 3, [Inc("i"), Inc("total")]).Do(
                FunctionStep(lambda c: seen.append((c["i"], c["total"])))
            )
        ).run()
        self.assertEqual(seen, [(0, 0), (1, 1), (2, 2)])

    def test_infinite_for_is_capped(self):
        with self.assertRaises(LoopLimitExceeded):
            Workflow(For(max_iterations=5).Do(SkipStep())).run()

    def test_runaway_for_is_capped(self):
        # step 没有推进条件变量
        with self.assertRaises(LoopLimitExceeded):
            Workflow(
                For(Set("i", 0), Var("i") < 10, SkipStep(), max_iterations=3).Do(
                    SkipStep()
                )
            ).run()

    def test_max_iterations_must_be_positive(self):
        with self.assertRaises(ValueError):
            For(Set("i", 0), True, SkipStep(), max_iterations=0)

    def test_trace(self):
        ctx = Workflow(For(Set("i", 0), Var("i") < 2, Inc("i")).Do(SkipStep())).run()
        self.assertEqual(
            ctx["i"], 2
        )
        self.assertIn("for:1", ctx.trace)
        self.assertIn("for:done", ctx.trace)

    def test_tree_shows_the_three_clauses(self):
        text = Workflow(
            For(Set("i", 0), Var("i") < 5, Inc("i")).Do(Echo(Var("i")))
        ).render()
        self.assertIn("For(Set('i', 0); (Var('i') < 5); i += 1)", text)
        self.assertIn("└─ Echo(Var('i'))", text)

    def test_typed_loop_variable(self):
        # Number 声明 + Inc 走类型转换
        ctx = Workflow(
            Number("i"),
            For(Set("i", 0), Var("i") < 3, Inc("i")).Do(SkipStep()),
        ).run()
        self.assertEqual(ctx["i"], 3)


class TestDoWhile(PromptkitTestCase):
    def test_body_runs_at_least_once(self):
        seen = []
        Workflow(
            Set("n", 0),
            DoWhile(False).Do(FunctionStep(lambda c: seen.append(c["n"]))),
        ).run()
        self.assertEqual(seen, [0])  # 条件一开始就是假的，但循环体仍执行了一次

    def test_loops_until_condition_false(self):
        seen = []
        Workflow(
            Set("n", 0),
            DoWhile(Var("n") < 3).Do(
                Inc("n"),
                FunctionStep(lambda c: seen.append(c["n"])),
            ),
        ).run()
        self.assertEqual(seen, [1, 2, 3])

    def test_break_works(self):
        seen = []
        Workflow(
            Set("n", 0),
            DoWhile(True, max_iterations=50).Do(
                Inc("n"),
                If(Var("n") >= 2).Then(Break()),
                FunctionStep(lambda c: seen.append(c["n"])),
            ),
        ).run()
        self.assertEqual(seen, [1])

    def test_exhausting_runs_once(self):
        # 和 While 的关键区别
        seen = []
        Workflow(
            Set("n", 5),
            DoWhile(Var("n") < 0).Do(FunctionStep(lambda c: seen.append(c["n"]))),
        ).run()
        self.assertEqual(seen, [5])


class TestIncDec(PromptkitTestCase):
    def test_inc(self):
        ctx = Workflow(Set("n", 1), Inc("n")).run()
        self.assertEqual(ctx["n"], 2)

    def test_inc_by(self):
        ctx = Workflow(Set("n", 1), Inc("n", 5)).run()
        self.assertEqual(ctx["n"], 6)

    def test_dec(self):
        ctx = Workflow(Set("n", 5), Dec("n")).run()
        self.assertEqual(ctx["n"], 4)

    def test_inc_accepts_expression(self):
        ctx = Workflow(Set("n", 1), Set("m", 3), Inc("n", Var("m"))).run()
        self.assertEqual(ctx["n"], 4)

    def test_inc_on_undeclared_raises(self):
        with self.assertRaises(UndefinedVariable):
            Workflow(Inc("nope")).run()

    def test_inc_keeps_declared_type(self):
        ctx = Workflow(
            Str("s"), Set("s", "a"), Inc("s", "b")
        ).run()
        self.assertEqual(ctx["s"], "ab")

    def test_describe(self):
        self.assertEqual(Inc("i").describe(), "i += 1")
        self.assertEqual(Dec("i", 2).describe(), "i -= 2")


class TestWhen(PromptkitTestCase):
    def test_picks_branch(self):
        wf = Workflow(NewVar("n", default=0), Set("r", When(Var("n") % 2 == 0, "偶", "奇")))
        self.assertEqual(wf.run(n=4)["r"], "偶")
        self.assertEqual(wf.run(n=3)["r"], "奇")

    def test_only_chosen_side_is_evaluated(self):
        # 和 C 的 ?: 一样是短路的
        calls = []

        def boom():
            calls.append(1)
            raise AssertionError("不该被求值")

        ctx = Workflow(
            Set("r", When(True, "ok", Call(boom))),
        ).run()
        self.assertEqual(ctx["r"], "ok")
        self.assertEqual(calls, [])

    def test_empty_list_does_not_crash_when_len_used(self):
        # 侧边可以是任意表达式，包括 Len()
        ctx = Workflow(
            NewVar("xs", default=[]),
            Set("r", When(Len(Var("xs")) == 0, "空", "有")),
        ).run()
        self.assertEqual(ctx["r"], "空")

    def test_repr_is_readable(self):
        self.assertIn("When(", repr(When(True, "a", "b")))


class TestSwitch(PromptkitTestCase):
    def test_matches_the_first_hit(self):
        wf = Workflow(
            Set("n", 2),
            Switch(Var("n"))
            .Case(1).Then(Set("hit", "one"))
            .Case(2).Then(Set("hit", "two"))
            .Default(Set("hit", "other")),
        )
        self.assertEqual(wf.run()["hit"], "two")

    def test_multiple_values_in_one_case(self):
        # 注意不要在前面 Set("n", ...)，否则会把 run(n=...) 传进来的值覆盖掉
        wf = Workflow(
            Switch(Var("n"))
            .Case(1, 2, 3).Then(Set("hit", "small"))
            .Default(Set("hit", "big")),
        )
        for n, expected in [(1, "small"), (3, "small"), (9, "big")]:
            self.assertEqual(wf.run(n=n)["hit"], expected)

    def test_default_when_nothing_matches(self):
        ctx = Workflow(
            Set("n", 9),
            Switch(Var("n")).Case(1).Then(Set("hit", "one")).Default(Set("hit", "other")),
        ).run()
        self.assertEqual(ctx["hit"], "other")

    def test_no_match_and_no_default_does_nothing(self):
        ctx = Workflow(
            Set("n", 9), Switch(Var("n")).Case(1).Then(Set("hit", "one"))
        ).run()
        self.assertNotIn("hit", ctx)

    def test_no_fallthrough_by_default(self):
        # C 里这里会同时打印 one 和 two，本 DSL 不会
        seen = []
        Workflow(
            Set("n", 1),
            Switch(Var("n"))
            .Case(1).Then(FunctionStep(lambda c: seen.append("one")))
            .Case(2).Then(FunctionStep(lambda c: seen.append("two"))),
        ).run()
        self.assertEqual(seen, ["one"])

    def test_explicit_fallthrough(self):
        seen = []
        Workflow(
            Set("n", 1),
            Switch(Var("n"))
            .Case(1, fallthrough=True).Then(FunctionStep(lambda c: seen.append("one")))
            .Case(2).Then(FunctionStep(lambda c: seen.append("two")))
            .Case(3).Then(FunctionStep(lambda c: seen.append("three"))),
        ).run()
        self.assertEqual(seen, ["one", "two"])

    def test_fallthrough_reaches_default(self):
        seen = []
        Workflow(
            Set("n", 1),
            Switch(Var("n"))
            .Case(1, fallthrough=True).Then(FunctionStep(lambda c: seen.append("one")))
            .Default(FunctionStep(lambda c: seen.append("default"))),
        ).run()
        self.assertEqual(seen, ["one", "default"])

    def test_subject_is_evaluated_once(self):
        calls = []

        def subject():
            calls.append(1)
            return 2

        seen = []
        Workflow(
            Switch(Call(subject))
            .Case(1).Then(FunctionStep(lambda c: seen.append("one")))
            .Case(3).Then(FunctionStep(lambda c: seen.append("three")))
            .Case(2).Then(FunctionStep(lambda c: seen.append("two"))),
        ).run()
        self.assertEqual(len(calls), 1)
        self.assertEqual(seen, ["two"])

    def test_case_value_can_be_an_expression(self):
        ctx = Workflow(
            Set("target", 7),
            Set("n", 7),
            Switch(Var("n"))
            .Case(Var("target")).Then(Set("hit", True))
            .Default(Set("hit", False)),
        ).run()
        self.assertTrue(ctx["hit"])

    def test_works_with_strings(self):
        wf = Workflow(
            Switch(Var("grade"))
            .Case("A").Then(Set("level", "优秀"))
            .Case("B", "C").Then(Set("level", "及格"))
            .Default(Set("level", "未知")),
        )
        self.assertEqual(wf.run(grade="A")["level"], "优秀")
        self.assertEqual(wf.run(grade="C")["level"], "及格")
        self.assertEqual(wf.run(grade="F")["level"], "未知")

    def test_break_targets_the_enclosing_loop_not_the_switch(self):
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 5, Inc("i")).Do(
                Switch(Var("i"))
                .Case(2).Then(Break())
                .Default(FunctionStep(lambda c: seen.append(c["i"]))),
            )
        ).run()
        self.assertEqual(seen, [0, 1])

    def test_nested_switch(self):
        wf = Workflow(
            Set("a", 1),
            Set("b", 2),
            Switch(Var("a"))
            .Case(1).Then(
                Switch(Var("b")).Case(2).Then(Set("hit", "1-2")).Default(Set("hit", "1-x"))
            )
            .Default(Set("hit", "x")),
        )
        self.assertEqual(wf.run()["hit"], "1-2")

    def test_trace(self):
        ctx = Workflow(
            Set("n", 2),
            Switch(Var("n")).Case(1).Then(SkipStep()).Case(2).Then(SkipStep()),
        ).run()
        self.assertEqual(ctx.trace, ["set:n", "switch:case2"])

    def test_trace_when_no_match(self):
        ctx = Workflow(Set("n", 9), Switch(Var("n")).Case(1).Then(SkipStep())).run()
        self.assertEqual(ctx.trace, ["set:n", "switch:default"])

    def test_tree_rendering(self):
        wf = Workflow(
            Switch(Var("grade"))
            .Case("A").Then(Echo("优秀"))
            .Case("B", "C").Then(Echo("及格"))
            .Default(Echo("未知"))
        )
        self.assertEqual(
            wf.render().splitlines(),
            [
                "Workflow",
                "└─ Switch(Var('grade'))",
                "   ├─ case 'A'",
                "   │  └─ Echo('优秀')",
                "   ├─ case 'B', 'C'",
                "   │  └─ Echo('及格')",
                "   └─ default",
                "      └─ Echo('未知')",
            ],
        )

    def test_tree_marks_fallthrough(self):
        wf = Workflow(Switch(Var("n")).Case(1, fallthrough=True).Then(SkipStep()))
        self.assertIn("(fallthrough)", wf.render())

    def test_parsing_errors_are_clear(self):
        with self.assertRaises(TypeError):
            Switch(Var("n")).Case()
        with self.assertRaises(ValueError):
            Switch(Var("n")).Then(SkipStep())

    def test_no_fallthrough_when_body_empty(self):
        # 空 case 不会意外掉进下一个 case
        seen = []
        Workflow(
            Set("n", 1),
            Switch(Var("n"))
            .Case(1)
            .Case(2).Then(FunctionStep(lambda c: seen.append("two"))),
        ).run()
        self.assertEqual(seen, [])


class TestPackaging(PromptkitTestCase):
    def test_importable_as_a_package(self):
        # 关键回归：以前 core 内部用 `from prompt import ...`，
        # 只有把项目目录本身放进 sys.path 才能用；
        # 现在统一成 promptkit.xxx，从父目录导入也可以。
        import importlib

        module = importlib.import_module("promptkit.core")
        self.assertTrue(hasattr(module, "Workflow"))

    def test_no_bare_top_level_imports_remain(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^\s*(?:from|import)\s+(prompt|executor|agent|core)\b", re.MULTILINE
        )
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if ".tmp" in path.parts:
                continue
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                offenders.append(f"{path.name}: {match.group(0).strip()}")
        self.assertEqual(offenders, [])

    def test_every_internal_import_resolves(self):
        '''目录一重构，旧路径的 import 就会变成"运行到才炸"的定时炸弹。

        这条测试静态检查全项目里所有 `promptkit.*` 的 import 目标是否**真的存在**，
        不需要先 import（所以不会因为父包先炸而误报）。
        '''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        package_parent = root.parent
        pattern = re.compile(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
        )

        def exists(module: str) -> bool:
            parts = module.split(".")
            path = package_parent.joinpath(*parts)
            return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()

        offenders = []
        for path in sorted(root.rglob("*.py")):
            if ".tmp" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                module = match.group(1) or match.group(2)
                if module.startswith("promptkit") and not exists(module):
                    line = text[: match.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line}: {module}")
        self.assertEqual(offenders, [])

    def test_every_subpackage_has_an_init(self):
        '''没有 __init__.py 的目录只是命名空间包：能 import，
        但 `find_packages()` 收不到，装成 wheel 之后就丢了。'''
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        # examples/ 和 tests/ 是脚本目录，不是 promptkit 的子包，不要求 __init__.py
        script_dirs = {"examples", "tests"}
        offenders = []
        for folder in sorted(root.rglob("*")):
            if not folder.is_dir():
                continue
            if any(part in (".tmp", "__pycache__", ".git") for part in folder.parts):
                continue
            if folder.name in script_dirs:
                continue
            if not any(folder.glob("*.py")):
                continue  # 纯数据目录（examples/kb_docs 之类）
            if not (folder / "__init__.py").is_file():
                offenders.append(str(folder.relative_to(root)))
        self.assertEqual(offenders, [])

    def test_reexports_point_at_the_new_locations(self):
        '''重构后新旧路径必须是同一个模块对象，不能重复加载出两份。'''
        import importlib

        import promptkit.core as core_module

        pairs = [
            ("promptkit.core.context", "Context"),
            ("promptkit.core.steps.workflow", "Workflow"),
            ("promptkit.core.model_action.defaults", "get_default_cautions"),
            ("promptkit.core.model_action.caution_related.caution", "Caution"),
            ("promptkit.core.model_action.caution_related.formatter", "JsonFormatter"),
            ("promptkit.core.steps.actions.llm", "LLMStep"),
            ("promptkit.core.steps.actions.interactive", "Echo"),
            ("promptkit.core.steps.actions.message", "MessageStep"),
            ("promptkit.core.steps.logics.if_step", "If"),
            ("promptkit.core.steps.logics.loop", "ForEach"),
            ("promptkit.core.steps.logics.switch", "Switch"),
            ("promptkit.core.steps.logics.goto", "Goto"),
            ("promptkit.core.steps.logics.function", "Function"),
            ("promptkit.core.steps.logics.while_step", "While"),
            ("promptkit.core.steps.logics.expr_sys.expr", "VarExpr"),
            ("promptkit.core.steps.logics.var", "NewVar"),
            ("promptkit.core.steps.logics.expr_sys.vartypes", "Number"),
            ("promptkit.core.steps.logics.conditions", "And"),
        ]
        for module_name, attribute in pairs:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attribute), f"{module_name} 里没有 {attribute}"
            )
            # 必须和 core 顶层再导出的是同一个对象
            self.assertIs(
                getattr(module, attribute),
                getattr(core_module, attribute),
                f"{module_name}.{attribute} 和 core.{attribute} 不是同一个对象",
            )

    def test_short_paths_are_available_at_every_level(self):
        '''每一层 __init__ 都再导出自己的子树，所以短路径和全路径拿到同一个对象。

        规则：`__init__` 导出「自己这个目录 + 子目录」的全部内容。
        '''
        import importlib

        import promptkit.core as core_module

        cases = [
            # steps 的两个分支 + 容器
            ("promptkit.core.steps.actions", "LLMStep"),
            ("promptkit.core.steps.actions", "Echo"),
            ("promptkit.core.steps.actions", "Read"),
            ("promptkit.core.steps.actions", "MessageStep"),
            ("promptkit.core.steps", "Workflow"),
            ("promptkit.core.steps.workflow", "Workflow"),
            ("promptkit.core.steps.logics", "If"),
            ("promptkit.core.steps.logics", "NewVar"),
            ("promptkit.core.steps.logics", "Var"),
            ("promptkit.core.steps.logics", "Function"),
            ("promptkit.core.steps.logics", "Condition"),
            ("promptkit.core.steps.logics", "Expr"),
            ("promptkit.core.steps.logics", "Number"),
            ("promptkit.core.steps.logics.expr_sys", "Expr"),
            ("promptkit.core.steps.logics.expr_sys", "VarExpr"),
            ("promptkit.core.steps.logics.expr_sys", "Number"),
            ("promptkit.core.steps.logics.expr_sys", "Str"),
            # model_action 及其子包
            ("promptkit.core.context", "Context"),
            ("promptkit.core.model_action", "Caution"),
            ("promptkit.core.model_action", "JsonFormatter"),
            ("promptkit.core.model_action.caution_related", "Caution"),
            ("promptkit.core.model_action.caution_related", "RegexFormatter"),
        ]
        for module_name, attribute in cases:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, attribute), f"{module_name} 里没有 {attribute}"
            )
            self.assertIs(
                getattr(module, attribute),
                getattr(core_module, attribute),
                f"{module_name}.{attribute} 和 core.{attribute} 不是同一个对象",
            )

    def test_subpackage_all_matches_its_attributes(self):
        '''每个子包的 __all__ 必须真的能取到（防止写错名字）。'''
        import importlib

        for module_name in (
            "promptkit.core.steps",
            "promptkit.core.steps.actions",
            "promptkit.core.steps.logics",
            "promptkit.core.steps.logics.expr_sys",
            "promptkit.core.model_action",
            "promptkit.core.model_action.caution_related",
        ):
            module = importlib.import_module(module_name)
            missing = [n for n in module.__all__ if not hasattr(module, n)]
            self.assertEqual(missing, [], f"{module_name} 的 __all__ 里有取不到的名字")

    def test_steps_layer_does_not_import_model_action_at_runtime(self):
        '''分层方向：steps 是底层，不该在运行时反向依赖 model_action。

        （`llm.py` 需要 `Caution` 是唯一例外 —— 它是个纯策略对象，不是运行时。）
        '''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^(?:from\s+(promptkit[\w.]*)\s+import|import\s+(promptkit[\w.]*))",
            re.MULTILINE,
        )
        allowed = {"promptkit.core.model_action.caution_related.caution"}
        offenders = []
        for path in sorted((root / "core" / "steps").rglob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                dep = match.group(1) or match.group(2)
                if "model_action" in dep and dep not in allowed:
                    rel = str(path.relative_to(root))
                    offenders.append(f"{rel}: {dep}")
        self.assertEqual(offenders, [])

    def test_pyproject_package_list_covers_every_subpackage(self):
        '''打包配置最容易出的错：加了新子包但忘了写进 pyproject.toml，
        结果打出来的 wheel 缺模块 —— 而且本地跑测试完全发现不了。

        这条测试把 pyproject 里的 packages 和磁盘上的实际目录结构对一遍。
        '''
        import pathlib

        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - 3.10 没有 tomllib
            self.skipTest("需要 Python 3.11+ 的 tomllib")

        root = pathlib.Path(__file__).resolve().parent.parent
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(config["tool"]["setuptools"]["packages"])

        # 磁盘上真实的子包：含 __init__.py 的目录。
        # examples/tests 是脚本目录；.tmp / build / dist 是构建产物，都不算。
        skip = {"examples", "tests", ".tmp", "build", "dist", "__pycache__"}
        actual = {"promptkit"}
        for path in sorted(root.rglob("__init__.py")):
            if any(part in skip or part.endswith(".egg-info") for part in path.parts):
                continue
            package = path.parent
            if package == root:
                continue
            actual.add("promptkit." + str(package.relative_to(root)).replace(os.sep, "."))

        self.assertEqual(
            declared - actual, set(), "pyproject 里列了不存在的包"
        )
        self.assertEqual(
            actual - declared, set(), "这些子包没写进 pyproject.toml，打 wheel 会丢"
        )

    def test_pyproject_maps_the_root_to_the_package_name(self):
        '''这个项目是「包目录 == 项目根目录」的平铺结构，必须显式映射，
        否则 setuptools 会把 core / cautions / default_tools 当成顶层包。'''
        import pathlib

        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("需要 Python 3.11+ 的 tomllib")

        root = pathlib.Path(__file__).resolve().parent.parent
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            config["tool"]["setuptools"]["package-dir"], {"promptkit": "."}
        )
        project = config["project"]
        self.assertEqual(project["name"], "promptkit")
        self.assertIn("openai", " ".join(project["dependencies"]))
        self.assertIn("pydantic", " ".join(project["dependencies"]))

    def test_no_stray_top_level_package_dirs(self):
        '''core / cautions / default_tools 只能是 promptkit 的子包，
        不能自己带 __init__.py 变成顶层包（否则 wheel 里多出三个垃圾顶层包）。'''
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for name in ("core", "cautions", "default_tools", "promptkit"):
            candidate = root.parent / name / "__init__.py"
            if name == "promptkit":
                continue
            self.assertFalse(
                candidate.exists(),
                f"{candidate} 会让 {name} 变成顶层包，应该不存在",
            )

    def test_relative_and_absolute_paths_agree(self):
        # promptkit.core 和它内部的模块必须是同一个对象，不能重复加载
        import promptkit.core.steps.workflow as via_package
        import sys

        self.assertIn("promptkit.core.steps.workflow", sys.modules)
        self.assertNotIn("core.workflow", sys.modules)
        self.assertIs(via_package.Workflow, Workflow)


class TestStringConcatenation(PromptkitTestCase):
    def test_reflected_add_drops_the_old_construction_error(self):
        # 以前 `"前缀：" + Var("x")` 会在**构造**时报
        # 「can only concatenate str (not "VarExpr") to str」；
        # 有了 __radd__ 之后能正常构造出表达式。
        expr = "i is now: " + Var("i")
        self.assertIsInstance(expr, Expr)

    def test_reflected_add_with_a_string_variable(self):
        ctx = Workflow(
            NewVar("i", default="3"), Set("msg", "i is now: " + Var("i"))
        ).run()
        self.assertEqual(ctx["msg"], "i is now: 3")

    def test_reflected_sub_mul_div_mod(self):
        ctx = Workflow(
            NewVar("n", default=10),
            Set("a", 100 - Var("n")),
            Set("b", 2 * Var("n")),
            Set("c", 100 / Var("n")),
            Set("d", 33 % Var("n")),
        ).run()
        self.assertEqual((ctx["a"], ctx["b"], ctx["c"], ctx["d"]), (90, 20, 10.0, 3))

    def test_adding_a_number_to_text_is_still_an_error(self):
        # 故意不自动转字符串：那样会掩盖 str/int 混用的 bug
        with self.assertRaises(TypeError) as cm:
            Workflow(NewVar("i", default=1), Set("m", "n: " + Var("i"))).run()
        self.assertIn("{{", str(cm.exception))

    def test_echo_template_is_the_way_to_interpolate(self):
        with mock.patch("builtins.print") as fake:
            Workflow(NewVar("i", default=3), Echo("i is now: {{i}}")).run()
        self.assertEqual(printed(fake), ["i is now: 3"])

    def test_echo_parts_are_the_alternative(self):
        with mock.patch("builtins.print") as fake:
            Workflow(NewVar("i", default=3), Echo("i is now: ", Var("i"))).run()
        self.assertEqual(printed(fake), ["i is now: 3"])


class TestForEachAcceptsHandle(PromptkitTestCase):
    def test_foreach_accepts_a_var_handle(self):
        seen = []
        Workflow(
            ForEach(Var("i"), range(3)).Do(FunctionStep(lambda c: seen.append(c["i"])))
        ).run()
        self.assertEqual(seen, [0, 1, 2])

    def test_foreach_accepts_a_newvar_handle(self):
        seen = []
        Workflow(
            Number("i"),
            ForEach(NewVar("i"), ["1", "2"]).Do(
                FunctionStep(lambda c: seen.append(c["i"]))
            ),
        ).run()
        self.assertEqual(seen, [1, 2])  # Number 类型转换仍然生效

    def test_bad_loop_variable_raises(self):
        with self.assertRaises(TypeError):
            ForEach(123, range(3))


class TestGoto(PromptkitTestCase):
    def test_backward_goto_is_a_retry_loop(self):
        attempts = []
        Workflow(
            NewVar("n", default=0),
            Label("retry"),
            Inc("n"),
            FunctionStep(lambda c: attempts.append(c["n"])),
            If(Var("n") < 3).Then(Goto("retry")),
        ).run()
        self.assertEqual(attempts, [1, 2, 3])

    def test_forward_goto_skips_steps(self):
        seen = []
        Workflow(
            FunctionStep(lambda c: seen.append("before")),
            Goto("skip"),
            FunctionStep(lambda c: seen.append("skipped")),
            Label("skip"),
            FunctionStep(lambda c: seen.append("after")),
        ).run()
        self.assertEqual(seen, ["before", "after"])

    def test_goto_escapes_nested_loops(self):
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 5, Inc("i")).Do(
                ForEach("j", range(5)).Do(
                    If(Var("i") == 1).Then(Goto("escape")),
                    FunctionStep(lambda c: seen.append((c["i"], c["j"]))),
                ),
            ),
            Label("escape"),
            FunctionStep(lambda c: seen.append("escaped")),
        ).run()
        self.assertEqual(seen[-1], "escaped")
        self.assertEqual(len(seen) - 1, 5)  # i=0 跑了 5 次，i=1 立刻跳出

    def test_goto_works_from_inside_an_if_branch(self):
        seen = []
        Workflow(
            If(True).Then(Goto("end")),
            FunctionStep(lambda c: seen.append("never")),
            Label("end"),
            FunctionStep(lambda c: seen.append("done")),
        ).run()
        self.assertEqual(seen, ["done"])

    def test_unknown_label_reports_known_ones(self):
        with self.assertRaises(UnknownLabel) as cm:
            Workflow(Label("real"), Goto("typo")).run()
        message = str(cm.exception)
        self.assertIn("typo", message)
        self.assertIn("real", message)

    def test_cannot_jump_into_a_loop(self):
        with self.assertRaises(UnknownLabel) as cm:
            Workflow(Goto("inner"), While(False).Do(Label("inner"))).run()
        self.assertIn("顶层", str(cm.exception))

    def test_runaway_goto_is_capped(self):
        with self.assertRaises(LoopLimitExceeded):
            Workflow(Label("x"), Goto("x"), max_jumps=20).run()

    def test_max_jumps_is_configurable(self):
        ctx = Workflow(
            NewVar("n", default=0),
            Label("again"),
            Inc("n"),
            If(Var("n") < 200).Then(Goto("again")),
            max_jumps=1000,
        ).run()
        self.assertEqual(ctx["n"], 200)

    def test_trace_records_label_and_goto(self):
        ctx = Workflow(Label("a"), Goto("b"), Label("b"), SkipStep()).run()
        self.assertEqual(ctx.trace, ["label:a", "goto:b"])

    def test_jump_to_a_label_at_the_very_end(self):
        ctx = Workflow(
            NewVar("n", default=0),
            Inc("n"),
            Goto("end"),
            Inc("n"),
            Label("end"),
        ).run()
        self.assertEqual(ctx["n"], 1)

    def test_label_scope_is_per_workflow(self):
        inner = Workflow(
            Label("in"),
            Inc("k"),
            If(Var("k") < 2).Then(Goto("in")),
        )
        ctx = Workflow(NewVar("k", default=0), inner).run()
        self.assertEqual(ctx["k"], 2)

    def test_labels_helper(self):
        wf = Workflow(Label("a"), SkipStep(), Label("b"))
        self.assertEqual(wf.labels(), {"a": 0, "b": 2})

    def test_bad_names_are_rejected(self):
        with self.assertRaises(TypeError):
            Label("")
        with self.assertRaises(TypeError):
            Goto(123)

    def test_tree_rendering(self):
        self.assertEqual(
            Workflow(Label("retry"), Goto("retry")).render().splitlines(),
            ["Workflow", "├─ Label('retry')", "└─ Goto('retry')"],
        )

    def test_break_still_targets_loops_not_goto(self):
        # Goto 和 Break 互不干扰
        seen = []
        Workflow(
            For(Set("i", 0), Var("i") < 5, Inc("i")).Do(
                If(Var("i") == 2).Then(Break()),
                FunctionStep(lambda c: seen.append(c["i"])),
            ),
            Label("after"),
            FunctionStep(lambda c: seen.append("after")),
        ).run()
        self.assertEqual(seen, [0, 1, "after"])


class TestFunctionReturn(PromptkitTestCase):
    def test_return_value(self):
        Add = Function(args=("a", "b"), name="add").Do(Return(Var("a") + Var("b")))
        ctx = Workflow(Set("s", Add(a=1, b=2).result)).run()
        self.assertEqual(ctx["s"], 3)

    def test_return_value_is_also_written_to_output_key(self):
        Add = Function(args=("a", "b"), name="add").Do(Return(Var("a") + Var("b")))
        ctx = Workflow(Add(a=1, b=2)).run()
        self.assertEqual(ctx["add"], 3)

    def test_returns_arbitrary_python_objects(self):
        payload = {"name": "Ann"}
        Make = Function(args=(), name="make").Do(Return(payload))
        ctx = Workflow(Make()).run()
        self.assertIs(ctx["make"], payload)

    def test_return_none_when_no_value_given(self):
        F = Function(args=(), name="f").Do(Return())
        ctx = Workflow(F()).run()
        self.assertIsNone(ctx["f"])

    def test_early_return_from_inside_a_loop(self):
        FirstEven = Function(args=("xs",), name="first_even").Do(
            ForEach("x", Var("xs")).Do(
                If(Var("x") % 2 == 0).Then(Return(Var("x"))),
            ),
            Return(None),
        )
        self.assertEqual(
            Workflow(Set("r", FirstEven(xs=[1, 3, 4, 5]).result)).run()["r"], 4
        )
        self.assertIsNone(
            Workflow(Set("r", FirstEven(xs=[1, 3, 5]).result)).run()["r"]
        )

    def test_return_from_inside_an_if_branch(self):
        Sign = Function(args=("n",), name="sign").Do(
            If(Var("n") > 0).Then(Return("正"))
            .ElseIf(Var("n") < 0).Then(Return("负"))
            .Else(Return("零")),
        )
        wf = Workflow(Set("r", Sign(n=Var("v")).result))
        self.assertEqual(wf.run(v=5)["r"], "正")
        self.assertEqual(wf.run(v=-5)["r"], "负")
        self.assertEqual(wf.run(v=0)["r"], "零")

    def test_function_can_call_another_function(self):
        Inner = Function(args=("x",), name="inner").Do(Return(Var("x") * 2))
        Outer = Function(args=("x",), name="outer").Do(
            Inner(x=Var("x")),
            Return(Var("inner") + 1),
        )
        ctx = Workflow(Set("r", Outer(x=5).result)).run()
        self.assertEqual(ctx["r"], 11)

    def test_return_re_evaluates_on_each_use(self):
        seen = []
        F = Function(args=("n",), name="f").Do(
            FunctionStep(lambda c: seen.append(c["n"])),
            Return(Var("n")),
        )
        ctx = Workflow(
            For(Set("i", 0), Var("i") < 3, Inc("i")).Do(
                Set("last", F(n=Var("i")).result),
            )
        ).run()
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(ctx["last"], 2)

    def test_missing_return_is_reported_clearly(self):
        NoReturn = Function(args=(), name="nothing").Do(FunctionStep(lambda c: None))
        with self.assertRaises(UndefinedVariable) as cm:
            Workflow(Set("r", NoReturn().result)).run()
        self.assertIn("Return", str(cm.exception))

    def test_return_outside_a_function_is_reported(self):
        with self.assertRaises(WorkflowError) as cm:
            Workflow(Return(1)).run()
        self.assertIn("Function", str(cm.exception))

    def test_agent_reply_alias_still_works(self):
        F = Function(args=(), name="f").Do(Return(7))
        ctx = Workflow(Set("r", F().agent_reply())).run()
        self.assertEqual(ctx["r"], 7)

    def test_prompt_body_needs_no_return(self):
        Ask = Function(args=("text",), name="ask").Do(
            Prompt(role="user", content="Q:{{text}}")
        )
        self.assertEqual(Ask.output_key, "ask")
        agent = FakeAgent(["答复"])
        ctx = Workflow(Set("r", Ask(text="hi").result)).run(agent=agent)
        self.assertEqual(ctx["r"], "答复")

    def test_tree_rendering(self):
        Add = Function(args=("a", "b"), name="add").Do(Return(Var("a") + Var("b")))
        self.assertEqual(
            Add(a=1, b=2).render().splitlines(),
            ["add(a=1, b=2)", "└─ Return((Var('a') + Var('b')))"],
        )

    def test_return_value_is_type_coerced(self):
        F = Function(args=(), name="n", output_key="n").Do(Return("42"))
        ctx = Workflow(Number("n"), F()).run()
        self.assertEqual(ctx["n"], 42)


def fake_message(**kwargs):
    '''造一个假的 ChatCompletion 响应。'''
    message = ChatCompletionMessage.model_validate(kwargs)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("假客户端没有更多回复了")
        return self.responses.pop(0)


def fake_agent(responses):
    agent = Agent(api_key="sk-test", register_as_default=False)
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(responses))
    )
    return agent


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def stream_chunks(contents, reasoning=None, tool_call_pieces=None):
    '''造一串假的流式 chunk。

    contents: 依次输出的正文片段
    reasoning: 依次输出的思考片段
    tool_call_pieces: [(index, id, name, arguments片段), ...]
    '''
    chunks = []
    for piece in reasoning or []:
        chunks.append(_chunk(reasoning=piece))
    for piece in contents:
        chunks.append(_chunk(content=piece))
    for index, call_id, name, arguments in tool_call_pieces or []:
        chunks.append(_chunk(tool_call=(index, call_id, name, arguments)))
    return chunks


def _chunk(content=None, reasoning=None, tool_call=None):
    calls = None
    if tool_call is not None:
        index, call_id, name, arguments = tool_call
        calls = [
            SimpleNamespace(
                index=index,
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ]
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=calls
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class TestTools(PromptkitTestCase):
    def test_schema_from_function(self):
        def get_weather(city: str, days: int = 1, tags: list = None) -> str:
            """查询某个城市的天气。

            第二段描述不算进 description。
            """
            return city

        schema = schema_from_signature(get_weather)
        self.assertEqual(
            schema["properties"],
            {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "tags": {"type": "array"},
            },
        )
        self.assertEqual(schema["required"], ["city"])

        tool_obj = as_tool(get_weather)
        self.assertEqual(tool_obj.name, "get_weather")
        self.assertEqual(tool_obj.description, "查询某个城市的天气。")
        self.assertEqual(tool_obj.schema()["type"], "function")
        self.assertEqual(tool_obj.schema()["function"]["name"], "get_weather")

    def test_type_mapping(self):
        def f(a: int, b: float, c: bool, d: dict, e: list, g, h: str = "x") -> str:
            """示例。"""
            return ""

        props = schema_from_signature(f)["properties"]
        self.assertEqual(props["a"]["type"], "integer")
        self.assertEqual(props["b"]["type"], "number")
        self.assertEqual(props["c"]["type"], "boolean")
        self.assertEqual(props["d"]["type"], "object")
        self.assertEqual(props["e"]["type"], "array")
        self.assertEqual(props["g"]["type"], "string")  # 没标注就按 string
        self.assertEqual(props["h"]["type"], "string")

    def test_optional_unwraps(self):
        def f(a: Optional[int] = None) -> str:
            """示例。"""
            return ""

        self.assertEqual(schema_from_signature(f)["properties"]["a"]["type"], "integer")

    def test_full_tool_loop(self):
        seen = []

        def get_weather(city: str) -> str:
            """查天气。"""
            seen.append(city)
            return f"{city}：晴，25℃"

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    reasoning_content="先查天气",
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(role="assistant", content="北京晴，25℃。"),
            ]
        )
        answer = agent.ask("北京天气怎么样？", tools=[get_weather])

        self.assertEqual(answer, "北京晴，25℃。")
        self.assertEqual(seen, ["北京"])
        calls = agent.client.chat.completions.calls
        self.assertEqual(len(calls), 2)
        self.assertIn("tools", calls[0])
        self.assertEqual(
            calls[0]["tools"][0]["function"]["name"], "get_weather"
        )

    def test_message_order_in_history(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(role="assistant", content="好了"),
            ]
        )
        agent.ask("天气？", tools=[get_weather])
        self.assertEqual(
            [m["role"] for m in agent.history],
            ["user", "assistant", "tool", "assistant"],
        )
        tool_message = agent.history[2]
        self.assertEqual(tool_message["tool_call_id"], "call_1")
        self.assertEqual(tool_message["content"], "晴")

    def test_reasoning_content_is_passed_back(self):
        # 带 tools 时 DeepSeek 要求把 reasoning_content 原样传回，否则 400
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    reasoning_content="我要查一下",
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(role="assistant", content="好了"),
            ]
        )
        agent.ask("天气？", tools=[get_weather])

        second = agent.client.chat.completions.calls[1]["messages"]
        assistants = [m for m in second if m.get("role") == "assistant"]
        self.assertEqual(assistants[0].get("reasoning_content"), "我要查一下")

    def test_reasoning_content_is_kept_even_without_tools(self):
        # 这样后续某一轮再加 tools 时，历史的 assistant 消息也带着它
        agent = fake_agent(
            [fake_message(role="assistant", content="hi", reasoning_content="想想")]
        )
        agent.ask("你好")
        self.assertEqual(agent.history[1]["reasoning_content"], "想想")

    def test_unknown_tool_is_reported_to_the_model(self):
        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("不存在", "{}")],
                ),
                fake_message(role="assistant", content="好的"),
            ]
        )
        agent.ask("hi", tools=[lambda: "x"])
        self.assertIn("没有名为", agent.history[2]["content"])

    def test_tool_exception_is_reported_to_the_model(self):
        def boom(x: str) -> str:
            """总是失败。"""
            raise RuntimeError("机器坏了")

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("boom", '{"x": "1"}')],
                ),
                fake_message(role="assistant", content="知道了"),
            ]
        )
        agent.ask("试试", tools=[boom])
        self.assertIn("机器坏了", agent.history[2]["content"])

    def test_bad_json_arguments_are_reported(self):
        def boom(x: str) -> str:
            """示例。"""
            return x

        agent = fake_agent(
            [
                fake_message(
                    role="assistant", content=None, tool_calls=[tool_call("boom", "{坏")]
                ),
                fake_message(role="assistant", content="知道了"),
            ]
        )
        agent.ask("试试", tools=[boom])
        self.assertIn("不是合法 JSON", agent.history[2]["content"])

    def test_tool_loop_is_capped(self):
        def echo(x: str) -> str:
            """回声。"""
            return x

        endless = [
            fake_message(
                role="assistant",
                content=None,
                tool_calls=[tool_call("echo", '{"x": "1"}', f"c{i}")],
            )
            for i in range(10)
        ]
        with self.assertRaises(ToolLoopExceeded):
            fake_agent(endless).ask("hi", tools=[echo], max_tool_rounds=3)

    def test_tools_kwarg_is_absent_when_unused(self):
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        agent.ask("你好")
        self.assertNotIn("tools", agent.client.chat.completions.calls[0])

    def test_tool_result_types_are_serialised(self):
        def give_dict() -> dict:
            """返回字典。"""
            return {"a": 1}

        def give_list() -> list:
            """返回列表。"""
            return [1, 2]

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        tool_call("give_dict", "{}", "c1"),
                        tool_call("give_list", "{}", "c2"),
                    ],
                ),
                fake_message(role="assistant", content="ok"),
            ]
        )
        agent.ask("hi", tools=[give_dict, give_list])
        contents = [m["content"] for m in agent.history if m["role"] == "tool"]
        self.assertEqual(contents, ['{"a": 1}', "[1, 2]"])

    def test_plain_functions_and_dicts_are_accepted(self):
        self.assertIsInstance(as_tool(lambda: 1), Tool)
        self.assertIsInstance(
            as_tool({"type": "function", "function": {"name": "x"}}), Tool
        )
        with self.assertRaises(TypeError):
            as_tool(123)

    def test_dict_tool_without_implementation_reports_clearly(self):
        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("remote", "{}")],
                ),
                fake_message(role="assistant", content="ok"),
            ]
        )
        agent.ask("hi", tools=[{"type": "function", "function": {"name": "remote"}}])
        self.assertIn("没有 Python 实现", agent.history[2]["content"])

    def test_llmstep_passes_tools_and_shows_them_in_the_tree(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        step = LLMStep(
            Prompt(role="user", content="天气？"), tools=[get_weather], output_key="a"
        )
        self.assertIn("tools: get_weather", step.render())

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(role="assistant", content="北京晴"),
            ]
        )
        ctx = Workflow(step).run(agent=agent)
        self.assertEqual(ctx["a"], "北京晴")

    def test_structured_output_with_tools(self):
        class Answer(BaseModel):
            text: str

        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(role="assistant", content='{"text": "北京晴"}'),
            ]
        )
        result = Executor(agent).run(
            Prompt(role="user", content="天气？"), Answer, tools=[get_weather]
        )
        self.assertEqual(result.text, "北京晴")
        self.assertIn("tools", agent.client.chat.completions.calls[0])

    def test_function_factory_accepts_tools(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        Ask = Function(args=("city",), name="ask", tools=[get_weather]).Do(
            Prompt(role="user", content="{{city}} 天气？")
        )
        agent = fake_agent([fake_message(role="assistant", content="晴")])
        Workflow(Ask(city="北京")).run(agent=agent)
        self.assertIn("tools", agent.client.chat.completions.calls[0])

    def test_tool_decorator(self):
        @tool(name="renamed", description="自定义描述")
        def original(x: str) -> str:
            """原描述。"""
            return x

        self.assertEqual(original.name, "renamed")
        self.assertEqual(original.description, "自定义描述")
        self.assertEqual(original.call({"x": "y"}), "y")


class TestEvents(PromptkitTestCase):
    def test_event_type_is_a_str_enum(self):
        # 关键：两种比较都要成立，否则 handler 会静默不匹配
        self.assertEqual(EventType.TOOL_RESULT, "tool_result")
        self.assertTrue(EventType.TOOL_RESULT == "tool_result")
        self.assertEqual(EventType("message_end"), EventType.MESSAGE_END)

    def test_event_accepts_a_plain_string_type(self):
        event = Event("tool_result", {"name": "x"})
        self.assertIs(event.type, EventType.TOOL_RESULT)
        with self.assertRaises(ValueError) as cm:
            Event("没这个类型")
        self.assertIn("未知的事件类型", str(cm.exception))

    def test_event_data_defaults_to_empty(self):
        self.assertEqual(Event(EventType.ERROR).data, {})

    def test_bus_on_off_clear(self):
        bus = EventBus()
        seen = []
        handler = bus.on(seen.append)
        bus.emit(Event(EventType.ERROR, {"x": 1}))
        self.assertEqual(len(seen), 1)
        bus.off(handler)
        bus.emit(Event(EventType.ERROR))
        self.assertEqual(len(seen), 1)
        bus.on(seen.append)
        bus.clear()
        self.assertFalse(bus.has_handlers)

    def test_bus_on_can_be_a_decorator(self):
        bus = EventBus()
        seen = []

        @bus.on
        def handler(event):
            seen.append(event.type)

        bus.emit(Event(EventType.STEP_START))
        self.assertEqual(seen, [EventType.STEP_START])

    def test_bus_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            EventBus().on(123)

    def test_handler_can_add_handlers_during_emit(self):
        bus = EventBus()
        seen = []

        def first(event):
            bus.on(lambda e: seen.append("second"))

        bus.on(first)
        bus.emit(Event(EventType.ERROR))  # 遍历的是副本：不该炸，也不该递归
        self.assertEqual(seen, [])
        bus.emit(Event(EventType.ERROR))  # 下一条事件才会轮到新 handler
        self.assertEqual(seen, ["second"])

    def test_extract_reasoning_handles_several_fields(self):
        self.assertEqual(
            extract_reasoning(SimpleNamespace(reasoning_content="a")), "a"
        )
        self.assertEqual(extract_reasoning(SimpleNamespace(reasoning="b")), "b")
        self.assertEqual(extract_reasoning(SimpleNamespace(thinking="c")), "c")
        self.assertIsNone(extract_reasoning(SimpleNamespace(content="d")))
        self.assertIsNone(extract_reasoning(SimpleNamespace(reasoning_content="")))

    def test_emit_needs_a_bus_in_context(self):
        handler, events = collect_events()
        ctx = Context()
        self.assertFalse(emit(ctx, EventType.ERROR, x=1))  # 没挂总线 -> no-op
        self.assertEqual(events, [])
        ctx["_event_bus"] = EventBus(handler)
        self.assertTrue(emit(ctx, EventType.ERROR, x=1))
        self.assertEqual(events[0].data, {"x": 1})

    def test_console_display_writes_to_given_stream(self):
        import io

        out = io.StringIO()
        handler = console_display(stream=out)
        handler(Event(EventType.MESSAGE_END, {"text": "你好"}))
        self.assertIn("你好", out.getvalue())

    def test_console_display_streaming_prints_once(self):
        import io

        out = io.StringIO()
        handler = console_display(stream=out)
        for piece in "abc":
            handler(Event(EventType.MESSAGE_DELTA, {"text": piece}))
        handler(Event(EventType.MESSAGE_END, {"text": "abc"}))
        # 流式已经把 abc 打过了，MESSAGE_END 不该再打一遍
        self.assertEqual(out.getvalue().strip(), "abc")

    def test_json_display(self):
        import io
        import json as jsonlib

        out = io.StringIO()
        json_display(stream=out)(Event(EventType.TOOL_RESULT, {"name": "x"}))
        payload = jsonlib.loads(out.getvalue())
        self.assertEqual(payload, {"type": "tool_result", "name": "x"})

    def test_filter_events(self):
        keep = filter_events(EventType.ERROR, "tool_result")
        self.assertTrue(keep(Event(EventType.ERROR)))
        self.assertTrue(keep(Event("tool_result")))
        self.assertFalse(keep(Event(EventType.MESSAGE_END)))


class TestAgentEvents(PromptkitTestCase):
    def test_thinking_tool_and_message_events(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    reasoning_content="先查一下",
                    tool_calls=[tool_call("get_weather", '{"city": "北京"}')],
                ),
                fake_message(
                    role="assistant", content="北京晴", reasoning_content="拿到了"
                ),
            ]
        )
        handler, events = collect_events()
        agent.bus.on(handler)
        agent.ask("天气？", tools=[get_weather])

        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                EventType.THINKING_START,
                EventType.THINKING_DELTA,
                EventType.THINKING_END,
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_END,
                EventType.TOOL_RESULT,
                EventType.THINKING_START,
                EventType.THINKING_DELTA,
                EventType.THINKING_END,
                EventType.MESSAGE_END,
            ],
        )
        results = [e.data for e in events if e.type == EventType.TOOL_RESULT]
        self.assertEqual(results[0]["name"], "get_weather")
        self.assertEqual(results[0]["result"], "晴")
        self.assertEqual(events[-1].data["text"], "北京晴")

    def test_no_bus_means_no_events_and_no_output(self):
        # 默认静默：Agent 自己不 print
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        with mock.patch("builtins.print") as fake:
            self.assertEqual(agent.ask("你好"), "hi")
        fake.assert_not_called()
        self.assertFalse(agent.bus.has_handlers)

    def test_verbose_attaches_a_temporary_handler(self):
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        before = len(agent.bus)
        agent.ask("你好", verbose=True)
        self.assertEqual(len(agent.bus), before)  # 用完就摘掉

    def test_on_event_is_per_call(self):
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        seen = []
        agent.ask("你好", on_event=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(agent.bus), 0)

    def test_stream_true_emits_same_events_and_returns_text(self):
        agent = fake_agent(
            [
                stream_chunks(["你", "好"]),
            ]
        )
        handler, events = collect_events()
        text = agent.ask("hi", stream=True, bus=EventBus(handler))
        self.assertEqual(text, "你好")
        self.assertEqual(
            [e.type for e in events],
            [
                EventType.MESSAGE_DELTA,
                EventType.MESSAGE_DELTA,
                EventType.MESSAGE_END,
            ],
        )


class TestAgentStreaming(PromptkitTestCase):
    def test_yields_deltas_then_message_end(self):
        agent = fake_agent([stream_chunks(["北", "京", "晴"], reasoning=["嗯", "…"])])
        events = list(agent.ask_stream("天气？"))
        self.assertEqual(
            [e.type for e in events],
            [
                EventType.THINKING_START,
                EventType.THINKING_DELTA,
                EventType.THINKING_DELTA,
                EventType.THINKING_END,
                EventType.MESSAGE_DELTA,
                EventType.MESSAGE_DELTA,
                EventType.MESSAGE_DELTA,
                EventType.MESSAGE_END,
            ],
        )
        self.assertEqual(events[-1].data["text"], "北京晴")
        self.assertEqual(
            "".join(e.data["text"] for e in events if e.type == EventType.MESSAGE_DELTA),
            "北京晴",
        )

    def test_stream_does_not_double_emit_to_the_bus(self):
        agent = fake_agent([stream_chunks(["hi"])])
        handler, events = collect_events()
        agent.bus.on(handler)
        list(agent.ask_stream("你好"))
        # 自己迭代时不再往总线发，避免显示两遍
        self.assertEqual(events, [])

    def test_streaming_tool_calls_are_reassembled(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return f"{city} 晴"

        agent = fake_agent(
            [
                stream_chunks(
                    [],
                    tool_call_pieces=[
                        (0, "call_1", "get_weather", '{"city"'),
                        (0, None, None, ': "北京"}'),
                    ],
                ),
                stream_chunks(["北京", "晴"]),
            ]
        )
        events = list(agent.ask_stream("天气？", tools=[get_weather]))
        types = [e.type for e in events]
        self.assertIn(EventType.TOOL_CALL_START, types)
        self.assertIn(EventType.TOOL_CALL_DELTA, types)
        self.assertIn(EventType.TOOL_CALL_END, types)
        self.assertIn(EventType.TOOL_RESULT, types)

        result = [e for e in events if e.type == EventType.TOOL_RESULT][0]
        self.assertEqual(result.data["result"], "北京 晴")
        end = [e for e in events if e.type == EventType.TOOL_CALL_END][0]
        self.assertEqual(end.data["arguments"], '{"city": "北京"}')
        self.assertEqual(events[-1].data["text"], "北京晴")

    def test_streaming_keeps_reasoning_in_history(self):
        # 带工具时 reasoning_content 必须回传，流式也一样
        def get_weather(city: str) -> str:
            """查天气。"""
            return "晴"

        agent = fake_agent(
            [
                stream_chunks(
                    [],
                    reasoning=["我", "想想"],
                    tool_call_pieces=[(0, "c1", "get_weather", "{}")],
                ),
                stream_chunks(["好了"]),
            ]
        )
        list(agent.ask_stream("天气？", tools=[get_weather]))
        assistants = [m for m in agent.history if m["role"] == "assistant"]
        self.assertEqual(assistants[0]["reasoning_content"], "我想想")
        second = agent.client.chat.completions.calls[1]["messages"]
        again = [m for m in second if m.get("role") == "assistant"]
        self.assertEqual(again[0]["reasoning_content"], "我想想")

    def test_streaming_tool_loop_is_capped(self):
        def echo(x: str) -> str:
            """回声。"""
            return x

        responses = [
            stream_chunks([], tool_call_pieces=[(0, f"c{i}", "echo", "{}")])
            for i in range(10)
        ]
        with self.assertRaises(ToolLoopExceeded):
            list(fake_agent(responses).ask_stream("hi", tools=[echo], max_tool_rounds=3))


class TestWorkflowEvents(PromptkitTestCase):
    def test_step_events(self):
        handler, events = collect_events()
        ctx = Workflow(
            NewVar("n", default=0), Set("n", 1), Echo("x")
        ).run(on_event=handler)
        steps = [e.data["step"] for e in events if e.type == EventType.STEP_START]
        self.assertEqual(steps, ["NewVar('n')", "Set('n', 1)", "Echo('x')"])
        self.assertEqual(
            len([e for e in events if e.type == EventType.STEP_END]), 3
        )

    def test_verbose_prints_steps_and_messages(self):
        import io

        out = io.StringIO()
        bus = EventBus(console_display(show_steps=True, stream=out))
        Workflow(Echo("hi")).run(bus=bus)
        self.assertIn("▶ Echo('hi')", out.getvalue())

    def test_no_events_when_no_bus(self):
        # 没挂总线时 execute_step 走零开销路径
        handler, events = collect_events()
        Workflow(Set("n", 1)).run()
        self.assertEqual(events, [])

    def test_error_event_is_emitted(self):
        handler, events = collect_events()

        def boom(_ctx):
            raise RuntimeError("炸了")

        with self.assertRaises(RuntimeError):
            Workflow(FunctionStep(boom)).run(on_event=handler)
        errors = [e for e in events if e.type == EventType.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertIn("炸了", errors[0].data["error"])

    def test_break_is_not_an_error_event(self):
        handler, events = collect_events()
        Workflow(
            ForEach("i", range(3)).Do(If(Var("i") == 1).Then(Break()))
        ).run(on_event=handler)
        self.assertEqual([e for e in events if e.type == EventType.ERROR], [])

    def test_workflow_bus_reaches_the_agent(self):
        # LLMStep 会把工作流的总线交给 Agent，所以能看到模型层事件
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        handler, events = collect_events()
        Workflow(
            LLMStep(Prompt(role="user", content="你好"), output_key="a")
        ).run(agent=agent, on_event=handler)
        self.assertIn(EventType.MESSAGE_END, [e.type for e in events])

    def test_workflow_reuses_the_agents_bus(self):
        agent = fake_agent([fake_message(role="assistant", content="hi")])
        handler, events = collect_events()
        agent.bus.on(handler)
        Workflow(
            LLMStep(Prompt(role="user", content="你好"), output_key="a")
        ).run(agent=agent)
        self.assertIn(EventType.MESSAGE_END, [e.type for e in events])

    def test_run_bus_handlers_are_removed_afterwards(self):
        bus = EventBus()
        Workflow(Set("n", 1)).run(bus=bus, verbose=True)
        self.assertEqual(len(bus), 0)


class TestRunCommand(PromptkitTestCase):
    def test_runs_and_returns_output(self):
        self.assertIn("hello", run_command("echo hello"))

    def test_empty_output_is_explained(self):
        # 不能说「没输出」就返回空串：模型会以为工具坏了然后一直重试
        text = run_command("cd /definitely-not-here-xyz 2>&1 || true")
        self.assertTrue(text.strip())

    def test_timeout_is_reported(self):
        text = run_command("python -c \"import time; time.sleep(5)\"", timeout=1)
        self.assertIn("超时", text)

    def test_decode_falls_back_between_encodings(self):
        # cmd 有时吐 UTF-8、有时吐 GBK，两种都要能解出来
        self.assertEqual(_decode("新加卷".encode("utf-8")), "新加卷")
        self.assertEqual(_decode("新加卷".encode("gbk")), "新加卷")
        self.assertEqual(_decode(b""), "")
        # 解不出来的字节不能炸，只能是替换字符
        self.assertIsInstance(_decode(b"\xff\xfe\xab"), str)

    def test_schema_only_exposes_command(self):
        # timeout 是 Python 参数，不该出现在给模型的 schema 里
        props = RUN_COMMAND.schema()["function"]["parameters"]["properties"]
        self.assertEqual(list(props), ["command"])
        self.assertEqual(
            RUN_COMMAND.schema()["function"]["parameters"]["required"], ["command"]
        )

    def test_description_tells_the_model_which_shell(self):
        # 不写系统信息的话模型会一直试 ls ~/Desktop
        description = RUN_COMMAND.description
        self.assertIn(platform.system(), description)
        if platform.system() == "Windows":
            self.assertIn("cmd", description)


class TestOutputSeparation(PromptkitTestCase):
    '''verbose（过程）走 stderr，结果走 stdout。'''

    def test_agent_verbose_goes_to_stderr(self):
        import io

        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch("sys.stdout", out):
            agent = Agent(api_key="sk-test", verbose=True, register_as_default=False)
            agent.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=FakeCompletions(
                        [
                            fake_message(
                                role="assistant",
                                content="最终答案",
                                reasoning_content="我先想想",
                            )
                        ]
                    )
                )
            )
            agent.ask("问题")
        self.assertIn("我先想想", err.getvalue())  # 思考 -> stderr
        self.assertEqual(out.getvalue(), "")  # stdout 保持干净
        self.assertNotIn("最终答案", err.getvalue())  # 回答不算「过程」

    def test_workflow_verbose_keeps_results_on_stdout(self):
        import io

        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            Workflow(Set("n", 1), Echo("结果")).run(verbose=True)
        # stdout 上干干净净只有结果
        self.assertEqual(out.getvalue().strip(), "结果")
        # 过程（步骤轨迹）在 stderr
        self.assertIn("Set('n', 1)", err.getvalue())
        self.assertIn("▶", err.getvalue())

    def test_results_and_trace_can_be_redirected_separately(self):
        import io

        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            Workflow(Echo("只有这一行是结果")).run(verbose=True)
        self.assertEqual(out.getvalue().strip(), "只有这一行是结果")

    def test_run_out_redirects_echo_without_shell_redirection(self):
        import io

        buf = io.StringIO()
        Workflow(Echo("结果"), Echo("第二行")).run(out=buf)
        self.assertEqual(buf.getvalue().splitlines(), ["结果", "第二行"])

    def test_run_out_does_not_leak_to_stdout(self):
        import io

        buf = io.StringIO()
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            Workflow(Echo("结果")).run(out=buf)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(buf.getvalue().strip(), "结果")

    def test_out_and_verbose_are_independent(self):
        import io

        buf, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stderr", err):
            Workflow(Set("n", 1), Echo("结果")).run(verbose=True, out=buf)
        self.assertEqual(buf.getvalue().strip(), "结果")
        self.assertIn("Set('n', 1)", err.getvalue())
        self.assertNotIn("结果", err.getvalue().split("▶")[0])

    def test_workflow_verbose_does_not_repeat_the_model_answer(self):
        # 回答是「结果」，verbose 只输出过程，不能重复一遍
        import io

        agent = fake_agent([fake_message(role="assistant", content="模型回答")])
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            Workflow(
                LLMStep(Prompt(role="user", content="问"), output_key="a")
            ).run(agent=agent, verbose=True)
        self.assertNotIn("模型回答", err.getvalue())

    def test_echo_stream_is_read_from_context(self):
        import io

        buf = io.StringIO()
        ctx = Workflow(Echo("hi")).run(out=buf)
        self.assertIn("echo", ctx.trace)
        self.assertEqual(buf.getvalue().strip(), "hi")


class TestCaution(PromptkitTestCase):
    def test_builtin_text(self):
        self.assertEqual(OutputFormat("JSON").to_text(), "请以 JSON 格式输出。")
        self.assertEqual(Language("英文").to_text(), "请使用英文回答。")
        self.assertEqual(MaxLength(50).to_text(), "回答不超过 50 字。")
        self.assertEqual(Tone("礼貌").to_text(), "语气要求：礼貌。")
        self.assertIn("不要生成", NoExtraText().to_text())

    def test_output_format_extra(self):
        self.assertEqual(
            OutputFormat("JSON", extra="字段名用英文").to_text(),
            "请以 JSON 格式输出，字段名用英文。",
        )

    def test_add_builds_a_group(self):
        group = OutputFormat("JSON") + NoExtraText()
        self.assertIsInstance(group, CautionGroup)
        self.assertEqual(len(group), 2)
        text = group.to_text()
        self.assertIn("JSON", text)
        self.assertIn("不要生成", text)
        self.assertEqual(len(text.splitlines()), 2)

    def test_chained_add_keeps_order(self):
        group = OutputFormat("JSON") + NoExtraText() + Language("中文")
        self.assertEqual(len(group), 3)
        self.assertTrue(group.to_text().startswith("请以 JSON 格式输出。"))
        self.assertTrue(group.to_text().endswith("请使用中文回答。"))

    def test_sum_works(self):
        total = sum([OutputFormat("JSON"), NoExtraText()])
        self.assertIsInstance(total, CautionGroup)
        self.assertEqual(len(total), 2)

    def test_group_is_itself_a_caution(self):
        inner = OutputFormat("JSON") + NoExtraText()
        outer = inner + Tone("礼貌")
        self.assertEqual(len(outer), 3)

    def test_strings_are_wrapped_as_custom(self):
        group = CautionGroup(["第一句", "第二句"])
        self.assertEqual(group.to_text(), "第一句\n第二句")

    def test_flatten_handles_nesting(self):
        group = CautionGroup([[OutputFormat("JSON")], (NoExtraText(),), None])
        self.assertEqual(len(group), 2)

    def test_bad_caution_type_is_rejected(self):
        with self.assertRaises(TypeError) as cm:
            CautionGroup([123])
        self.assertIn("约束必须是", str(cm.exception))

    def test_empty_caution_is_skipped(self):
        group = CautionGroup([Custom(""), OutputFormat("JSON"), Custom("")])
        self.assertEqual(group.to_text(), "请以 JSON 格式输出。")
        self.assertEqual(len(resolve_cautions(group)), 1)

    def test_to_dict_is_serializable(self):
        payload = (OutputFormat("JSON") + NoExtraText()).to_dict()
        self.assertEqual(payload["type"], "CautionGroup")
        self.assertEqual(len(payload["cautions"]), 2)
        json.dumps(payload)  # 必须能过 JSON

    def test_parse_presets(self):
        self.assertIn("JSON", JSON_ONLY.to_text())
        self.assertIn("200", PLAIN_TEXT.to_text())
        self.assertIn("代码", CODE_ONLY.to_text())

    def test_json_with_schema(self):
        group = json_with_schema({"label": "退款/咨询"})
        self.assertIn("输出字段说明", group.to_text())
        self.assertIn("退款", group.to_text())

    def test_resolve_priority_is_global_then_workflow_then_step(self):
        with using_default_cautions(Custom("全局")):
            resolved = resolve_cautions([Custom("Step")], [Custom("Workflow")])
        self.assertEqual([c.to_text() for c in resolved], ["全局", "Workflow", "Step"])

    def test_resolve_dedups_identical_text(self):
        resolved = resolve_cautions([Custom("一样"), Custom("一样"), Custom("不一样")])
        self.assertEqual([c.to_text() for c in resolved], ["一样", "不一样"])

    def test_resolve_can_skip_defaults(self):
        with using_default_cautions(Custom("全局")):
            resolved = resolve_cautions([Custom("Step")], use_defaults=False)
        self.assertEqual([c.to_text() for c in resolved], ["Step"])

    def test_defaults_are_isolated_from_outside_mutation(self):
        set_default_cautions([Custom("全局")])
        get_default_cautions().append(Custom("偷偷加的"))
        self.assertEqual(len(get_default_cautions()), 1)

    def test_using_default_cautions_restores(self):
        set_default_cautions([Custom("原来")])
        with using_default_cautions(Custom("临时")):
            self.assertEqual(get_default_cautions()[0].to_text(), "临时")
        self.assertEqual(get_default_cautions()[0].to_text(), "原来")
        clear_default_cautions()

    def test_add_default_caution_appends(self):
        clear_default_cautions()
        set_default_cautions([Custom("一")])
        add_default_caution(Custom("二"))
        self.assertEqual([c.to_text() for c in get_default_cautions()], ["一", "二"])
        clear_default_cautions()


class TestFormatterLayer(PromptkitTestCase):
    def test_json_plain(self):
        self.assertEqual(JsonFormatter().parse('{"a": 1}'), {"a": 1})

    def test_json_fenced(self):
        self.assertEqual(JsonFormatter().parse('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_with_prose_around_it(self):
        self.assertEqual(JsonFormatter().parse('好的，结果是 {"a": 1} 请查收'), {"a": 1})

    def test_json_array(self):
        self.assertEqual(JsonFormatter().parse("[1, 2]"), [1, 2])

    def test_json_error(self):
        with self.assertRaises(FormatterError) as cm:
            JsonFormatter().parse("完全不是 JSON")
        self.assertIn("无法从模型输出中解析出 JSON", str(cm.exception))

    def test_json_empty(self):
        with self.assertRaises(FormatterError):
            JsonFormatter().parse("   ")

    def test_xml(self):
        parsed = XmlFormatter().parse("<output><a>1</a><b>2</b></output>")
        self.assertEqual(parsed, {"a": "1", "b": "2"})

    def test_xml_root_tag_is_checked(self):
        with self.assertRaises(FormatterError):
            XmlFormatter(root_tag="result").parse("<output><a>1</a></output>")

    def test_xml_bad(self):
        with self.assertRaises(FormatterError):
            XmlFormatter().parse("<output><a>1</a>")

    def test_regex(self):
        parsed = RegexFormatter(r"意图：(?P<label>\w+)").parse("好的，意图：Refund 谢谢")
        self.assertEqual(parsed, {"label": "Refund"})

    def test_regex_needs_named_groups(self):
        with self.assertRaises(FormatterError) as cm:
            RegexFormatter(r"(\w+)").parse("abc")
        self.assertIn("命名捕获组", str(cm.exception))

    def test_regex_no_match(self):
        with self.assertRaises(FormatterError):
            RegexFormatter(r"意图：(?P<label>\w+)").parse("没有")

    def test_get_formatter_by_name(self):
        self.assertIsInstance(get_formatter(None), JsonFormatter)
        self.assertIsInstance(get_formatter("json"), JsonFormatter)
        self.assertIsInstance(get_formatter("xml"), XmlFormatter)
        self.assertIsInstance(get_formatter("yaml"), YamlFormatter)
        self.assertEqual(get_formatter("regex:(?P<a>\\d)").parse("7").get("a"), "7")

    def test_get_formatter_passthrough(self):
        mine = JsonFormatter()
        self.assertIs(get_formatter(mine), mine)

    def test_get_formatter_rejects_unknown(self):
        with self.assertRaises(FormatterError):
            get_formatter("csv")
        with self.assertRaises(TypeError):
            get_formatter(42)

    def test_executor_uses_the_given_formatter(self):
        agent = FakeAgent(["意图：Refund"])
        result = Executor(agent=agent).run(
            Prompt(role="user", content="hi"),
            Intent,
            formatter=RegexFormatter(r"意图：(?P<label>\w+)"),
        )
        self.assertEqual(result.label, "Refund")

    def test_executor_retries_with_a_better_hint(self):
        agent = FakeAgent(["不是 XML", "<output><label>ok</label></output>"])
        result = Executor(agent=agent, max_retries=1).run(
            Prompt(role="user", content="hi"), Intent, formatter=XmlFormatter()
        )
        self.assertEqual(result.label, "ok")
        self.assertIn("xml", prompt_text(agent.prompts[1]))

    def test_extract_json_alias_still_raises_the_old_error(self):
        with self.assertRaises(OutputValidationError):
            Executor._extract_json("nope")

    def test_output_validation_error_is_a_formatter_error(self):
        self.assertTrue(issubclass(OutputValidationError, FormatterError))


class TestLLMStepCautions(PromptkitTestCase):
    def setUp(self):
        super().setUp()
        clear_default_cautions()

    def tearDown(self):
        clear_default_cautions()
        super().tearDown()

    def build(self, **kwargs):
        defaults = dict(
            prompt=Prompt(role="user", content="判断意图：{{text}}"),
            output_model=Intent,
            output_key="intent",
        )
        defaults.update(kwargs)
        return LLMStep(**defaults)

    def test_no_cautions_returns_the_original_prompt(self):
        step = self.build()
        self.assertIs(step.build_prompt(), step.prompt)

    def test_cautions_are_appended_to_the_end(self):
        step = self.build(cautions=[OutputFormat("JSON"), NoExtraText()])
        content = step.build_prompt().content
        self.assertTrue(content.startswith("判断意图：{{text}}"))
        self.assertIn("请以 JSON 格式输出。", content)
        self.assertIn("不要生成", content)

    def test_role_is_preserved(self):
        step = LLMStep(
            prompt=Prompt(role="system", content="你是助手"),
            cautions=[NoExtraText()],
        )
        self.assertEqual(step.build_prompt().role, "system")

    def test_group_and_plus_both_work(self):
        by_list = self.build(cautions=[OutputFormat("JSON") + NoExtraText()])
        by_plus = self.build(cautions=OutputFormat("JSON") + NoExtraText())
        self.assertEqual(by_list.build_prompt(), by_plus.build_prompt())

    def test_preset_can_be_used_directly(self):
        step = self.build(cautions=[JSON_ONLY])
        self.assertIn("请以 JSON 格式输出。", step.build_prompt().content)

    def test_plain_string_caution(self):
        step = self.build(cautions=["只输出一个词。"])
        self.assertIn("只输出一个词。", step.build_prompt().content)

    def test_global_defaults_apply(self):
        set_default_cautions([NoExtraText()])
        self.assertIn("不要生成", self.build().build_prompt().content)

    def test_step_cautions_come_after_global(self):
        set_default_cautions([Custom("全局")])
        content = self.build(cautions=[Custom("Step")]).build_prompt().content
        self.assertLess(content.index("全局"), content.index("Step"))

    def test_use_default_cautions_false_opts_out(self):
        set_default_cautions([Custom("全局")])
        step = self.build(use_default_cautions=False)
        self.assertNotIn("全局", step.build_prompt().content)

    def test_workflow_cautions_apply(self):
        step = self.build()
        ctx = Context()
        ctx[WORKFLOW_CAUTIONS_KEY] = [Custom("Workflow")]
        self.assertIn("Workflow", step.build_prompt(ctx).content)

    def test_workflow_cautions_come_before_step_cautions(self):
        step = self.build(cautions=[Custom("Step")])
        ctx = Context()
        ctx[WORKFLOW_CAUTIONS_KEY] = [Custom("Workflow")]
        content = step.build_prompt(ctx).content
        self.assertLess(content.index("Workflow"), content.index("Step"))

    def test_duplicates_are_deduped_across_levels(self):
        set_default_cautions([NoExtraText()])
        step = self.build(cautions=[JSON_ONLY])  # JSON_ONLY 里也有 NoExtraText
        content = step.build_prompt().content
        self.assertEqual(content.count("不要生成"), 1)

    def test_old_underscore_name_still_works(self):
        step = self.build(cautions=[Custom("X")])
        self.assertIn("X", step._build_prompt().content)

    def test_auto_schema_adds_a_schema_hint(self):
        step = self.build(auto_schema=True)
        content = step.build_prompt().content
        self.assertIn("输出字段说明", content)
        self.assertIn("label", content)

    def test_auto_schema_off_by_default(self):
        self.assertNotIn("输出字段说明", self.build().build_prompt().content)

    def test_auto_schema_skipped_without_output_model(self):
        step = LLMStep(prompt=Prompt(role="user", content="hi"), auto_schema=True)
        self.assertEqual(step.build_prompt().content, "hi")

    def test_cautions_are_not_rendered_into_the_original_prompt(self):
        step = self.build(cautions=[Custom("约束")])
        step.build_prompt()
        self.assertNotIn("约束", step.prompt.content)  # 没被就地改掉

    def test_execute_actually_sends_the_cautions(self):
        agent = FakeAgent(['{"label": "ok"}'])
        ctx = Context(data={"text": "我要退款"})
        self.build(agent=agent, cautions=[OutputFormat("JSON")]).execute(ctx)
        self.assertIn("请以 JSON 格式输出。", prompt_text(agent.prompts[0]))
        self.assertEqual(ctx["intent"].label, "ok")

    def test_workflow_level_cautions_reach_the_agent(self):
        agent = FakeAgent(['{"label": "ok"}'])
        wf = Workflow(
            LLMStep(
                prompt=Prompt(role="user", content="hi"),
                output_model=Intent,
                output_key="intent",
            ),
            cautions=[Custom("工作流级约束")],
        )
        wf.run(agent=agent)
        self.assertIn("工作流级约束", prompt_text(agent.prompts[0]))

    def test_workflow_cautions_default_to_empty(self):
        self.assertEqual(Workflow().cautions, [])

    def test_workflow_cautions_accept_a_string(self):
        self.assertEqual(Workflow(cautions="一句话").cautions[0].to_text(), "一句话")

    def test_describe_shows_the_caution_count(self):
        step = self.build(cautions=[OutputFormat("JSON"), NoExtraText()])
        self.assertIn("2 cautions", step.describe())

    def test_generic_subscript_produces_a_usable_step(self):
        step = LLMStep[Intent](
            prompt=Prompt(role="user", content="hi"),
            output_model=Intent,
            output_key="m",
        )
        self.assertEqual(step.output_key, "m")
        self.assertIn("Intent", step.describe())

    def test_resolved_cautions_helper(self):
        set_default_cautions([Custom("全局")])
        step = self.build(cautions=[Custom("Step")])
        texts = [c.to_text() for c in step.resolved_cautions()]
        self.assertEqual(texts, ["全局", "Step"])


class TestCautionsEndToEnd(PromptkitTestCase):
    def test_json_only_preset_renders_as_designed(self):
        step = LLMStep[Intent](
            prompt=Prompt(role="user", content="判断用户意图：{{text}}"),
            output_model=Intent,
            output_key="intent",
            cautions=[JSON_ONLY],
            auto_schema=True,
        )
        content = step.build_prompt().content
        self.assertTrue(content.startswith("判断用户意图：{{text}}\n\n"))
        for expected in ("请以 JSON 格式输出。", "请使用中文回答。", "输出字段说明"):
            self.assertIn(expected, content)

    def test_formatter_and_caution_work_together(self):
        # 事前引导（Caution）+ 事后解析（Formatter）互不干扰
        agent = FakeAgent(["<output><label>ok</label></output>"])
        ctx = Context()
        LLMStep(
            prompt=Prompt(role="user", content="hi"),
            output_model=Intent,
            output_key="intent",
            agent=agent,
            cautions=[JSON_ONLY],
            formatter=XmlFormatter(),
        ).execute(ctx)
        self.assertEqual(ctx["intent"].label, "ok")
        self.assertIn("请以 JSON 格式输出。", prompt_text(agent.prompts[0]))


class TestVariablePaths(PromptkitTestCase):
    r'''{{a.b}} 这种点路径 —— 之前 \w+ 匹配不到，会**静默原样输出**。'''

    def render(self, text, **variables):
        return render_text(text, variables)

    def test_plain_key_still_works(self):
        self.assertEqual(self.render("你好 {{name}}", name="小明"), "你好 小明")

    def test_attribute_on_an_object(self):
        self.assertEqual(self.render("{{intent.label}}", intent=Intent(label="退款")), "退款")

    def test_dict_key(self):
        self.assertEqual(self.render("{{a.b}}", a={"b": "值"}), "值")

    def test_list_index(self):
        self.assertEqual(self.render("{{hits.0.text}}", hits=[{"text": "第一段"}]), "第一段")

    def test_negative_index(self):
        # 模板正则不含 '-'，负下标走 Var()
        ctx = Context(data={"xs": ["a", "b"]})
        self.assertEqual(Var("xs.-1").eval(ctx), "b")

    def test_nested_path(self):
        payload = {"a": {"b": {"c": "很深"}}}
        self.assertEqual(self.render("{{a.b.c}}", **payload), "很深")

    def test_exact_key_wins_over_path(self):
        # 真的有个 key 叫 "a.b" 时不该被拆开
        self.assertEqual(self.render("{{a.b}}", **{"a.b": "整块"}), "整块")

    def test_missing_field_raises_loudly(self):
        with self.assertRaises(KeyError) as cm:
            self.render("{{intent.nope}}", intent=Intent(label="x"))
        self.assertIn("nope", str(cm.exception))

    def test_missing_root_still_raises(self):
        with self.assertRaises(KeyError) as cm:
            self.render("{{nothing.here}}")
        self.assertIn("不存在", str(cm.exception))

    def test_private_attributes_are_blocked(self):
        with self.assertRaises(KeyError):
            self.render("{{intent.__class__}}", intent=Intent(label="x"))

    def test_dollar_form_does_not_take_dots(self):
        # $1.5 这类文本不能被当成路径
        self.assertEqual(self.render("价格 $1.5", **{"1": "x"}), "价格 x.5")

    def test_prompt_render_uses_paths(self):
        message = Prompt(role="user", content="意图：{{intent.label}}").render(
            intent=Intent(label="咨询")
        )
        self.assertEqual(message["content"], "意图：咨询")

    def test_var_expr_takes_paths(self):
        ctx = Context(data={"intent": Intent(label="投诉")})
        self.assertEqual(Var("intent.label").eval(ctx), "投诉")

    def test_var_expr_path_through_a_list(self):
        ctx = Context(data={"hits": [{"text": "x"}, {"text": "y"}]})
        self.assertEqual(Var("hits.1.text").eval(ctx), "y")

    def test_var_expr_missing_path_raises_undefined(self):
        ctx = Context(data={"intent": Intent(label="x")})
        with self.assertRaises(UndefinedVariable) as cm:
            Var("intent.nope").eval(ctx)
        self.assertIn("NewVar", str(cm.exception))

    def test_echo_interpolates_paths(self):
        with mock.patch("builtins.print") as fake:
            Workflow(Echo("{{intent.label}}")).run(intent=Intent(label="退款"))
        self.assertEqual(printed(fake), ["退款"])

    def test_if_condition_can_use_a_path(self):
        ctx = Workflow(
            If(Var("intent.label") == "退款").Then(Set("hit", 1)).Else(Set("hit", 0))
        ).run(intent=Intent(label="退款"))
        self.assertEqual(ctx["hit"], 1)

    def test_condition_reprs_are_readable(self):
        self.assertIn("Var('a')", repr(Not(Var("a") == 1)))
        self.assertIn("Var('a')", repr(And(Var("a") == 1, Var("b") == 2)))
        self.assertIn("Var('b')", repr(Or(Var("a") == 1, Var("b") == 2)))


class TestKnowledgeBase(PromptkitTestCase):
    def kb(self):
        return KnowledgeBase.from_texts(
            [
                "退款政策：未发货的订单可以直接申请退款，3 个工作日到账。",
                "已发货的订单需要先申请退货，收到货后 7 天内处理退款。",
                "配送范围：目前只支持中国大陆地区，港澳台暂不支持。",
                "发票：可以开具电子发票，下单时填写抬头即可。",
            ]
        )

    def test_tokenize_handles_chinese_and_english(self):
        tokens = tokenize("如何退款 refund")
        self.assertIn("退款", tokens)  # 双字（主力）
        self.assertIn("如何", tokens)
        self.assertIn("退", tokens)  # 单字（兜底召回）
        self.assertIn("refund", tokens)  # 英文单词

    def test_unigrams_are_weighted_down(self):
        # 单字只是「沾边」，不该和双字一样重
        self.assertEqual(token_weight("退款"), 1.0)
        self.assertLess(token_weight("退"), 1.0)
        self.assertEqual(token_weight("refund"), 1.0)

    def test_tokenize_keeps_a_lone_character(self):
        self.assertEqual(tokenize("票"), ["票"])

    def test_tokenize_empty(self):
        self.assertEqual(tokenize(""), [])

    def test_chunk_text_splits_paragraphs(self):
        self.assertEqual(len(chunk_text("a\n\nb\n\nc")), 3)

    def test_chunk_text_respects_max_chars(self):
        chunks = chunk_text("x" * 1000, max_chars=100, overlap=0)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_chunk_text_overlap_keeps_context(self):
        chunks = chunk_text("abcdefghij" * 30, max_chars=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # 有重叠时总长度会超过原文
        self.assertGreater(sum(map(len, chunks)), 300)

    def test_chunk_text_rejects_bad_max_chars(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", max_chars=0)

    def test_search_finds_the_right_chunk(self):
        top = self.kb().search("怎么退款", k=1)[0]
        self.assertIn("退款", top.text)

    def test_search_ranks_the_more_relevant_chunk_first(self):
        top = self.kb().search("已经发货了还能退吗", k=1)[0]
        self.assertIn("已发货", top.text)

    def test_search_by_keyword(self):
        top = self.kb().search("发票", k=1)[0]
        self.assertIn("电子发票", top.text)

    def test_unrelated_query_scores_low(self):
        # 词法检索里不相关查询也会因为一两个单字沾边拿到低分，
        # 但不是 0 —— 所以要用 min_score 卡门槛（见下一个测试）
        hits = self.kb().search("量子纠缠")
        self.assertTrue(all(hit.score < 1.0 for hit in hits))

    def test_min_score_filters_weak_matches(self):
        self.assertEqual(self.kb().search("量子纠缠", min_score=1.0), [])

    def test_min_score_keeps_strong_matches(self):
        self.assertTrue(self.kb().search("怎么退款", min_score=1.0))

    def test_search_respects_k(self):
        self.assertLessEqual(len(self.kb().search("退款 发货 发票 配送", k=2)), 2)

    def test_search_on_empty_kb(self):
        self.assertEqual(KnowledgeBase().search("x"), [])

    def test_search_empty_query(self):
        self.assertEqual(self.kb().search(""), [])

    def test_hit_carries_source_and_score(self):
        kb = KnowledgeBase()
        kb.add("退款说明", source="policy.md")
        hit = kb.search("退款")[0]
        self.assertEqual(hit.source, "policy.md")
        self.assertGreater(hit.score, 0)
        self.assertEqual(hit.to_dict()["source"], "policy.md")

    def test_format_includes_numbering_and_source(self):
        text = KnowledgeBase.format(self.kb().search("退款", k=2))
        self.assertIn("[1]", text)
        self.assertIn("[2]", text)

    def test_format_says_so_when_nothing_found(self):
        text = KnowledgeBase.format([], query="火星")
        self.assertIn("没有", text)
        self.assertIn("火星", text)

    def test_search_text_is_prompt_ready(self):
        self.assertIn("退款", self.kb().search_text("退款", k=1))

    def test_add_chunks_long_text(self):
        kb = KnowledgeBase(max_chars=50, overlap=0)
        kb.add("x" * 200)
        self.assertGreater(len(kb), 1)

    def test_len_and_iter(self):
        kb = self.kb()
        self.assertEqual(len(kb), 4)
        self.assertEqual(len(list(kb)), 4)

    def test_roundtrip_through_json(self):
        import tempfile

        kb = self.kb()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "kb.json")
            kb.save(path)
            loaded = KnowledgeBase.load(path)
        self.assertEqual(len(loaded), len(kb))
        self.assertEqual(loaded.search("退款", k=1)[0].text, kb.search("退款", k=1)[0].text)

    def test_from_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "a.md"), "w", encoding="utf-8") as handle:
                handle.write("退款政策在这里")
            with open(os.path.join(folder, "skip.csv"), "w", encoding="utf-8") as handle:
                handle.write("不该被读进来")
            kb = KnowledgeBase.from_dir(folder)
        self.assertEqual(len(kb), 1)
        self.assertIn("退款", kb.chunks[0])

    def test_as_tool_is_callable_by_the_model(self):
        tool = self.kb().as_tool()
        self.assertEqual(tool.name, "search_knowledge_base")
        self.assertIn("query", tool.schema()["function"]["parameters"]["properties"])
        self.assertIn("退款", tool.call({"query": "退款"}))

    def test_as_tool_k_is_clamped(self):
        tool = self.kb().as_tool(k=2)
        # top_k 超过上限不该炸
        self.assertTrue(tool.call({"query": "退款", "top_k": 999}))


class TestKnowledgeBaseWorkflow(PromptkitTestCase):
    '''把检索接进 wf：检索是普通 Python，编排在 wf 里。'''

    def setUp(self):
        super().setUp()
        self.kb = KnowledgeBase()
        self.kb.add("退款政策：未发货可直接退款。", source="policy.md")
        self.kb.add("配送范围：只支持中国大陆。", source="shipping.md")

    def test_call_bridges_to_python(self):
        ctx = Workflow(
            Set("question", "怎么退款"),
            Set("hits", Call(self.kb.search, Var("question"), k=2)),
        ).run()
        self.assertTrue(ctx["hits"])
        self.assertIn("退款", ctx["hits"][0].text)

    def test_format_step_produces_prompt_ready_text(self):
        ctx = Workflow(
            Set("question", "怎么退款"),
            Set("hits", Call(self.kb.search, Var("question"), k=2)),
            Set("context", Call(KnowledgeBase.format, Var("hits"))),
        ).run()
        self.assertIn("[1]", ctx["context"])

    def test_empty_retrieval_can_be_branched_on(self):
        ctx = Workflow(
            Set("question", "量子纠缠"),
            Set("hits", Call(self.kb.search, Var("question"))),
            If(Len(Var("hits")) == 0).Then(
                Set("answer", "不知道"),
                Goto("done"),
            ),
            Set("answer", "有资料"),
            Label("done"),
        ).run()
        self.assertEqual(ctx["answer"], "不知道")

    def test_hits_can_be_embedded_in_a_prompt(self):
        agent = FakeAgent(["答"])
        wf = Workflow(
            Set("question", "怎么退款"),
            Set("hits", Call(self.kb.search, Var("question"), k=1)),
            Set("context", Call(KnowledgeBase.format, Var("hits"))),
            LLMStep(
                prompt=Prompt(role="user", content="资料：{{context}}\n问：{{question}}"),
                output_key="answer",
            ),
        )
        wf.run(agent=agent)
        sent = prompt_text(agent.prompts[0])
        self.assertIn("退款政策", sent)
        self.assertIn("怎么退款", sent)


class TestKnowledgeBasePersistence(PromptkitTestCase):
    '''持久化：建一次反复用，重跑不重复灌。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.docs = os.path.join(self._tmp.name, "docs")
        os.makedirs(self.docs)
        self.kb_path = os.path.join(self._tmp.name, "kb.json")
        self.write("refund.md", "退款政策：未发货可以直接退款。")

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def write(self, name, text):
        with open(os.path.join(self.docs, name), "w", encoding="utf-8") as handle:
            handle.write(text)

    def build(self, **kwargs):
        kb = KnowledgeBase(max_chars=300, overlap=0)
        kb.sync_dir(self.docs, **kwargs)
        kb.save(self.kb_path)
        return kb

    def test_save_and_load_roundtrip(self):
        kb = self.build()
        loaded = KnowledgeBase.load(self.kb_path)
        self.assertEqual(len(loaded), len(kb))
        self.assertEqual(loaded.search("退款")[0].text, kb.search("退款")[0].text)

    def test_save_creates_missing_directories(self):
        nested = os.path.join(self._tmp.name, "a", "b", "kb.json")
        KnowledgeBase.from_texts(["你好"]).save(nested)
        self.assertTrue(os.path.exists(nested))

    def test_save_is_atomic(self):
        # 目录里不该留下临时文件
        kb = self.build()
        kb.save(self.kb_path)
        leftovers = [n for n in os.listdir(self._tmp.name) if n.startswith(".kb-")]
        self.assertEqual(leftovers, [])

    def test_save_does_not_corrupt_on_failure(self):
        from unittest import mock as _mock

        kb = self.build()
        before = open(self.kb_path, encoding="utf-8").read()
        with _mock.patch("json.dump", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                kb.save(self.kb_path)
        # 原文件必须还在，且没有临时文件残留
        self.assertEqual(open(self.kb_path, encoding="utf-8").read(), before)
        self.assertEqual([n for n in os.listdir(self._tmp.name) if n.startswith(".kb-")], [])

    def test_file_carries_format_and_version(self):
        self.build()
        payload = json.load(open(self.kb_path, encoding="utf-8"))
        self.assertEqual(payload["format"], "promptkit-knowledge")
        self.assertEqual(payload["version"], FORMAT_VERSION)

    def test_config_survives_the_roundtrip(self):
        kb = KnowledgeBase(max_chars=123, overlap=7)
        kb.add("你好")
        kb.save(self.kb_path)
        loaded = KnowledgeBase.load(self.kb_path)
        self.assertEqual(loaded.max_chars, 123)
        self.assertEqual(loaded.overlap, 7)

    def test_newer_version_is_rejected(self):
        self.build()
        payload = json.load(open(self.kb_path, encoding="utf-8"))
        payload["version"] = FORMAT_VERSION + 1
        with open(self.kb_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with self.assertRaises(ValueError) as cm:
            KnowledgeBase.load(self.kb_path)
        self.assertIn("只认到", str(cm.exception))

    def test_old_format_without_version_still_loads(self):
        with open(self.kb_path, "w", encoding="utf-8") as handle:
            json.dump({"chunks": ["旧格式"], "sources": ["old.md"]}, handle)
        kb = KnowledgeBase.load(self.kb_path)
        self.assertEqual(len(kb), 1)
        self.assertEqual(kb.search("旧格式")[0].source, "old.md")

    def test_index_can_be_persisted(self):
        kb = self.build()
        kb.save(self.kb_path, include_index=True)
        payload = json.load(open(self.kb_path, encoding="utf-8"))
        self.assertIn("index", payload)
        loaded = KnowledgeBase.load(self.kb_path)
        self.assertFalse(loaded._dirty)  # 不用重新分词
        self.assertEqual(loaded.search("退款")[0].text, kb.search("退款")[0].text)

    def test_index_is_not_saved_by_default(self):
        self.build()
        payload = json.load(open(self.kb_path, encoding="utf-8"))
        self.assertNotIn("index", payload)

    def test_sync_is_idempotent(self):
        kb = self.build()
        stats = kb.sync_dir(self.docs)
        self.assertEqual(stats, {"added": 0, "updated": 0, "skipped": 1, "removed": 0})
        self.assertEqual(len(kb), 1)

    def test_sync_picks_up_a_new_file(self):
        kb = self.build()
        self.write("shipping.md", "配送范围：只支持大陆。")
        stats = kb.sync_dir(self.docs)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(len(kb), 2)

    def test_sync_replaces_a_changed_file(self):
        kb = self.build()
        self.write("refund.md", "退款政策改版：现在随时可以退。")
        # 保证 mtime 真的变了
        os.utime(os.path.join(self.docs, "refund.md"), (0, 0))
        stats = kb.sync_dir(self.docs)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(len(kb), 1)  # 旧的被替换掉，不是叠加
        self.assertIn("改版", kb.chunks[0])

    def test_sync_prunes_deleted_files(self):
        self.build()
        kb = KnowledgeBase.load(self.kb_path)
        os.remove(os.path.join(self.docs, "refund.md"))
        stats = kb.sync_dir(self.docs)
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(len(kb), 0)

    def test_sync_prune_can_be_disabled(self):
        kb = self.build()
        os.remove(os.path.join(self.docs, "refund.md"))
        stats = kb.sync_dir(self.docs, prune=False)
        self.assertEqual(stats["removed"], 0)
        self.assertEqual(len(kb), 1)

    def test_sync_force_reingests(self):
        kb = self.build()
        stats = kb.sync_dir(self.docs, force=True)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(len(kb), 1)  # 仍然不重复

    def test_manifest_survives_the_roundtrip(self):
        self.build()
        loaded = KnowledgeBase.load(self.kb_path)
        self.assertIn("refund.md", loaded.ingested)
        self.assertEqual(loaded.sync_dir(self.docs)["skipped"], 1)

    def test_remove_source(self):
        kb = KnowledgeBase.from_texts(["a", "b"])
        kb.add("c", source="x.md")
        self.assertEqual(kb.remove_source("x.md"), 1)
        self.assertEqual(len(kb), 2)

    def test_remove_source_keeps_the_index_consistent(self):
        kb = KnowledgeBase.from_texts(["退款说明"])
        kb.add("配送说明", source="s.md")
        kb.remove_source("s.md")
        hits = kb.search("配送")
        self.assertEqual(hits, [])

    def test_stats(self):
        kb = self.build()
        stats = kb.stats()
        self.assertEqual(stats["chunks"], 1)
        self.assertEqual(stats["sources"], 1)
        self.assertEqual(stats["ingested"], 1)
        self.assertGreater(stats["tokens"], 0)

    def test_dotted_source_is_not_confused_with_a_path(self):
        # 来源名和「点路径」没关系，只是普通字符串
        kb = KnowledgeBase()
        kb.add("内容", source="a.b.md")
        self.assertEqual(kb.search("内容")[0].source, "a.b.md")


class TestPersistentKbExample(PromptkitTestCase):
    '''直接跑 examples/persistent_kb.py 的命令行入口（不联网的那几条）。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.docs = os.path.join(self._tmp.name, "docs")
        os.makedirs(self.docs)
        self.kb_path = os.path.join(self._tmp.name, "kb.json")

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def example(self):
        import runpy

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return runpy.run_path(
            os.path.join(root, "examples", "persistent_kb.py"), run_name="notmain"
        )

    def run_main(self, *argv):
        return self.example()["main"](list(argv))

    def test_the_bundled_docs_exist_and_are_ingestible(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs = os.path.join(root, "examples", "kb_docs")
        self.assertTrue(os.path.isdir(docs))
        self.assertTrue(os.listdir(docs))

    def test_build_then_stats_then_search(self):
        self.run_main("--kb", self.kb_path, "--docs", self.docs, "build")
        self.assertTrue(os.path.exists(self.kb_path))

        self.write("refund.md", "退款政策：未发货可以直接退款。")
        self.run_main("--kb", self.kb_path, "--docs", self.docs, "build")
        self.assertEqual(self.run_main("--kb", self.kb_path, "stats"), 0)
        self.assertEqual(self.run_main("--kb", self.kb_path, "search", "退款"), 0)

    def test_build_is_idempotent_via_the_cli(self):
        self.write("refund.md", "退款政策：未发货可以直接退款。")
        self.run_main("--kb", self.kb_path, "--docs", self.docs, "build")
        first = KnowledgeBase.load(self.kb_path)
        self.run_main("--kb", self.kb_path, "--docs", self.docs, "build")
        second = KnowledgeBase.load(self.kb_path)
        self.assertEqual(len(first), len(second))

    def test_search_on_a_missing_kb_tells_you_to_build(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main("--kb", self.kb_path, "search", "x")
        self.assertIn("build", str(cm.exception))

    def test_build_workflow_is_constructible_from_a_loaded_kb(self):
        kb = KnowledgeBase.from_texts(["退款政策：未发货可以直接退款。"])
        workflow = self.example()["build_workflow"](kb)
        self.assertIn("persistent_rag", repr(workflow))

    def write(self, name, text):
        with open(os.path.join(self.docs, name), "w", encoding="utf-8") as handle:
            handle.write(text)


class TestFeatureFlags(PromptkitTestCase):
    '''实验性功能默认关闭 —— 这是「撤回」的落地方式：代码留着，默认不生效。'''

    def test_knowledge_is_a_known_feature(self):
        self.assertIn("knowledge", features.features())

    def test_default_is_off(self):
        # 测试进程里被显式打开了，所以先关掉再看默认
        with features.using("knowledge", False):
            self.assertFalse(features.enabled("knowledge"))

    def test_enable_and_disable(self):
        with features.using("knowledge", False):
            features.enable("knowledge")
            self.assertTrue(features.enabled("knowledge"))
            features.disable("knowledge")
            self.assertFalse(features.enabled("knowledge"))

    def test_env_var_enables(self):
        with features.using("knowledge", False):
            with mock.patch.dict(
                os.environ, {"PROMPTKIT_ENABLE_KNOWLEDGE": "1"}, clear=False
            ):
                self.assertTrue(features.enabled("knowledge"))

    def test_env_var_accepts_common_truthy_spellings(self):
        for value in ("1", "true", "YES", "On"):
            with features.using("knowledge", False):
                with mock.patch.dict(
                    os.environ, {"PROMPTKIT_ENABLE_KNOWLEDGE": value}, clear=False
                ):
                    self.assertTrue(features.enabled("knowledge"), value)

    def test_env_var_zero_is_not_enabled(self):
        with features.using("knowledge", False):
            with mock.patch.dict(
                os.environ, {"PROMPTKIT_ENABLE_KNOWLEDGE": "0"}, clear=False
            ):
                self.assertFalse(features.enabled("knowledge"))

    def test_require_raises_a_helpful_error(self):
        with features.using("knowledge", False):
            with self.assertRaises(features.FeatureDisabled) as cm:
                features.require("knowledge")
        message = str(cm.exception)
        self.assertIn("默认关闭", message)
        self.assertIn("PROMPTKIT_ENABLE_KNOWLEDGE", message)
        self.assertIn("features.enable('knowledge')", message)

    def test_require_passes_when_enabled(self):
        with features.using("knowledge", True):
            features.require("knowledge")  # 不该抛

    def test_using_restores_the_previous_state(self):
        features.enable("knowledge")
        with features.using("knowledge", False):
            self.assertFalse(features.enabled("knowledge"))
        self.assertTrue(features.enabled("knowledge"))

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(KeyError) as cm:
            features.enable("time-travel")
        self.assertIn("没有这个功能开关", str(cm.exception))

    def test_status(self):
        with features.using("knowledge", False):
            self.assertEqual(features.status()["knowledge"], "off")
        with features.using("knowledge", True):
            self.assertEqual(features.status()["knowledge"], "on (runtime)")

    def test_disabled_by_default_in_a_fresh_process(self):
        '''真正证明「默认关闭」：起一个干净的子进程 import 它。'''
        import subprocess

        import promptkit

        # 包目录的**父目录**才在 sys.path 上
        parent = os.path.dirname(os.path.dirname(os.path.abspath(promptkit.__file__)))
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "PROMPTKIT_ENABLE_KNOWLEDGE"
        }
        env["PYTHONPATH"] = parent
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "-c", "import promptkit.knowledge"],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("默认关闭", result.stderr)

        # 环境变量一开就能用
        env["PROMPTKIT_ENABLE_KNOWLEDGE"] = "1"
        ok = subprocess.run(
            [sys.executable, "-c", "import promptkit.knowledge; print('ok')"],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("ok", ok.stdout)

    def test_other_features_still_import_with_it_off(self):
        '''关掉知识库不该影响核心能力。'''
        with features.using("knowledge", False):
            import importlib

            module = importlib.import_module("promptkit.core")
            self.assertTrue(hasattr(module, "Workflow"))
            module = importlib.import_module("promptkit.agent")
            self.assertTrue(hasattr(module, "Agent"))


class TestWebSearchHelpers(PromptkitTestCase):
    '''搜索工具里能离线测的部分：URL 还原、文本清理、解析。'''

    DDG_HTML = """
    <html><body><table>
      <tr><td class="result-link">
        <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&amp;rut=abc">
          Welcome to Python.org
        </a>
      </td></tr>
      <tr><td class="result-snippet">The official home of the <b>Python</b> language.</td></tr>
      <tr><td class="link-text">www.python.org</td></tr>
      <tr><td class="result-link">
        <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">Python 3 Docs</a>
      </td></tr>
      <tr><td class="result-snippet">Docs for Python 3.</td></tr>
      <tr><td class="result-link"><a href="https://plain.example.com/x">No uddg</a></td></tr>
    </table></body></html>
    """

    # ---------------- URL ----------------
    def test_unwrap_ddg_redirect(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1&rut=x"
        self.assertEqual(unwrap_url(href), "https://example.com/a?b=1")

    def test_unwrap_leaves_plain_urls_alone(self):
        self.assertEqual(unwrap_url("https://example.com/x"), "https://example.com/x")

    def test_unwrap_adds_scheme_to_protocol_relative(self):
        self.assertEqual(unwrap_url("//cdn.example.com/y"), "https://cdn.example.com/y")

    def test_unwrap_root_relative_uses_base(self):
        self.assertEqual(
            unwrap_url("/path", base="https://example.com"),
            "https://example.com/path",
        )

    def test_unwrap_empty(self):
        self.assertEqual(unwrap_url(""), "")

    def test_unwrap_decodes_entities_in_the_href(self):
        # 页面里 &amp; 要当成 &
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2F&amp;rut=1"
        self.assertEqual(unwrap_url(href), "https://a.com/")

    # ---------------- 文本 ----------------
    def test_clean_text_strips_tags_entities_and_whitespace(self):
        self.assertEqual(clean_text("  A &amp; B <b>粗</b>\n  C  "), "A & B 粗 C")

    def test_clean_text_handles_none(self):
        self.assertEqual(clean_text(None), "")

    # ---------------- k ----------------
    def test_clamp_k(self):
        self.assertEqual(clamp_k(3), 3)
        self.assertEqual(clamp_k(999), 20)
        self.assertEqual(clamp_k(0), 1)
        self.assertEqual(clamp_k(-5), 1)
        self.assertEqual(clamp_k("7"), 7)
        self.assertEqual(clamp_k("abc"), 5)
        self.assertEqual(clamp_k(None, default=2), 2)

    # ---------------- 去重 ----------------
    def test_dedupe_ignores_trailing_slash(self):
        results = [
            SearchResult(url="https://a.com/", title="A"),
            SearchResult(url="https://a.com", title="dup"),
            SearchResult(url="https://b.com/", title="B"),
        ]
        self.assertEqual([r.title for r in dedupe(results)], ["A", "B"])

    def test_dedupe_drops_empty_urls(self):
        self.assertEqual(dedupe([SearchResult(title="x")]), [])

    # ---------------- 解析 ----------------
    def test_parse_ddg_lite(self):
        results = parse_ddg_lite(self.DDG_HTML)
        self.assertEqual(len(results), 2)  # 第三条没有 uddg，不算结果
        self.assertEqual(results[0].title, "Welcome to Python.org")
        self.assertEqual(results[0].url, "https://www.python.org/")
        self.assertIn("official home", results[0].snippet)
        self.assertEqual(results[1].url, "https://docs.python.org/3/")
        self.assertEqual(results[0].source, "duckduckgo")

    def test_parse_ddg_lite_respects_limit(self):
        self.assertEqual(len(parse_ddg_lite(self.DDG_HTML, limit=1)), 1)

    def test_parse_ddg_lite_is_tolerant(self):
        # 对方改版 / 空页面 -> 空列表，而不是抛异常或者吐垃圾
        for junk in ("", "<html></html>", "完全不是 HTML", "<table><tr><td>x</td></tr></table>"):
            self.assertEqual(parse_ddg_lite(junk), [])

    # ---------------- 格式化 ----------------
    def test_format_has_numbering_url_and_snippet(self):
        text = WebSearch.format(parse_ddg_lite(self.DDG_HTML))
        self.assertIn("[1] Welcome to Python.org", text)
        self.assertIn("https://www.python.org/", text)
        self.assertIn("[2] Python 3 Docs", text)

    def test_format_says_so_when_nothing_found(self):
        self.assertIn("没有搜到", WebSearch.format([]))
        self.assertIn("量子纠缠", WebSearch.format([], query="量子纠缠"))


class TestWebSearchBackends(PromptkitTestCase):
    '''带 Key 的后端：把 HTTP 层打桩，验证请求参数和响应解析。'''

    TAVILY_RESPONSE = {
        "results": [
            {"title": "T1", "url": "https://t1.com", "content": "内容一"},
            {"title": "T2", "url": "https://t2.com", "content": "内容二"},
        ]
    }
    SERPER_RESPONSE = {
        "organic": [{"title": "S1", "link": "https://s1.com", "snippet": "摘要一"}]
    }
    BRAVE_RESPONSE = {
        "web": {
            "results": [
                {"title": "B1", "url": "https://b1.com", "description": "描述一"}
            ]
        }
    }

    def test_tavily_parses_and_sends_the_key(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_post_json",
            return_value=self.TAVILY_RESPONSE,
        ) as fake:
            results = TavilyBackend(api_key="k").search("q", k=2)
        self.assertEqual([r.url for r in results], ["https://t1.com", "https://t2.com"])
        self.assertEqual(results[0].snippet, "内容一")
        self.assertEqual(results[0].source, "tavily")
        url, payload = fake.call_args.args[0], fake.call_args.args[1]
        self.assertEqual(url, TavilyBackend.ENDPOINT)
        self.assertEqual(payload["query"], "q")
        self.assertEqual(payload["max_results"], 2)
        self.assertIn("Bearer k", fake.call_args.kwargs["headers"]["Authorization"])

    def test_serper_parses_and_sends_the_key(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_post_json",
            return_value=self.SERPER_RESPONSE,
        ) as fake:
            results = SerperBackend(api_key="k").search("q", k=3)
        self.assertEqual(results[0].url, "https://s1.com")
        self.assertEqual(results[0].snippet, "摘要一")
        self.assertEqual(results[0].source, "serper")
        self.assertEqual(fake.call_args.args[1], {"q": "q", "num": 3})
        self.assertEqual(fake.call_args.kwargs["headers"]["X-API-KEY"], "k")

    def test_brave_parses(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_get",
            return_value=json.dumps(self.BRAVE_RESPONSE),
        ):
            results = BraveBackend(api_key="k").search("q", k=1)
        self.assertEqual(results[0].url, "https://b1.com")
        self.assertEqual(results[0].snippet, "描述一")
        self.assertEqual(results[0].source, "brave")

    def test_brave_rejects_non_json(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_get", return_value="<html>nope"
        ):
            with self.assertRaises(SearchError):
                BraveBackend(api_key="k").search("q")

    def test_backends_require_a_key(self):
        for cls, env in (
            (TavilyBackend, "TAVILY_API_KEY"),
            (SerperBackend, "SERPER_API_KEY"),
            (BraveBackend, "BRAVE_API_KEY"),
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SearchError) as cm:
                    cls()
            self.assertIn(env, str(cm.exception))

    def test_pick_backend_falls_back_to_a_keyless_chain(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = pick_backend()
        self.assertIsInstance(backend, FallbackBackend)
        # 国内优先：百度 -> Bing 中国 -> DuckDuckGo
        self.assertEqual([b.name for b in backend.backends], ["baidu", "bing", "duckduckgo"])

    def test_pick_backend_prefers_tavily(self):
        with mock.patch.dict(
            os.environ,
            {"TAVILY_API_KEY": "t", "SERPER_API_KEY": "s", "BRAVE_API_KEY": "b"},
            clear=True,
        ):
            self.assertIsInstance(pick_backend(), TavilyBackend)

    def test_pick_backend_prefers_domestic_apis(self):
        # 博查 / 智谱 排在英文后端前面
        with mock.patch.dict(
            os.environ, {"BOCHA_API_KEY": "b", "TAVILY_API_KEY": "t"}, clear=True
        ):
            self.assertIsInstance(pick_backend(), BochaBackend)
        with mock.patch.dict(
            os.environ, {"ZHIPUAI_API_KEY": "z", "TAVILY_API_KEY": "t"}, clear=True
        ):
            self.assertIsInstance(pick_backend(), ZhipuBackend)

    def test_pick_backend_uses_serper_without_tavily(self):
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "s"}, clear=True):
            self.assertIsInstance(pick_backend(), SerperBackend)

    def test_chain_can_be_overridden(self):
        with mock.patch.dict(
            os.environ, {"PROMPTKIT_SEARCH_BACKENDS": "bing"}, clear=True
        ):
            self.assertIsInstance(pick_backend(), BingCnBackend)

    def test_chain_order_can_be_swapped(self):
        with mock.patch.dict(
            os.environ, {"PROMPTKIT_SEARCH_BACKENDS": "bing, baidu"}, clear=True
        ):
            backend = pick_backend()
        self.assertEqual([b.name for b in backend.backends], ["bing", "baidu"])

    def test_unknown_chain_name_is_reported(self):
        with mock.patch.dict(
            os.environ, {"PROMPTKIT_SEARCH_BACKENDS": "google"}, clear=True
        ):
            with self.assertRaises(SearchError) as cm:
                pick_backend()
        self.assertIn("google", str(cm.exception))
        self.assertIn("baidu", str(cm.exception))

    def test_duckduckgo_complains_when_it_parses_nothing(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_get", return_value="<html></html>"
        ):
            with self.assertRaises(SearchError) as cm:
                DuckDuckGoBackend().search("q")
        message = str(cm.exception)
        self.assertIn("没有解析出任何结果", message)
        self.assertIn("PROMPTKIT_SEARCH_BACKENDS", message)  # 告诉你怎么换个后端

    # ---------------- 百度 ----------------
    BAIDU_HTML = """
    <html><body><div id="content_left">
      <div class="result c-container" id="1">
        <h3 class="t"><a href="http://www.baidu.com/link?url=AbC123">Python 官方网站</a></h3>
        <div class="c-abstract">Python 是一门解释型、面向对象的高级编程语言。</div>
      </div>
      <div class="result c-container" id="2">
        <h3 class="t"><a href="http://www.baidu.com/link?url=XyZ789">Python 教程</a></h3>
        <div class="c-abstract">本教程适合想从零开始学习 Python 的读者。</div>
      </div>
    </div>
    <div class="nav"><a href="https://www.baidu.com/more/">百度导航</a></div>
    <script>var x = 1;</script>
    </body></html>
    """

    def test_parse_baidu(self):
        results = parse_baidu(self.BAIDU_HTML)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Python 官方网站")
        self.assertIn("/link?url=AbC123", results[0].url)
        self.assertIn("解释型", results[0].snippet)
        self.assertEqual(results[0].source, "baidu")

    def test_parse_baidu_does_not_attach_the_footer_to_the_last_result(self):
        '''回归：页脚 / 导航的文本不能粘到最后一条结果的摘要上。'''
        last = parse_baidu(self.BAIDU_HTML)[-1]
        self.assertNotIn("百度导航", last.snippet)
        self.assertIn("本教程", last.snippet)

    def test_parse_baidu_ignores_nav_and_script(self):
        titles = [r.title for r in parse_baidu(self.BAIDU_HTML)]
        self.assertNotIn("百度导航", titles)

    def test_parse_baidu_is_tolerant(self):
        for junk in ("", "<html></html>", "不是 HTML", "<html>安全验证</html>"):
            self.assertEqual(parse_baidu(junk), [])

    def test_extract_baidu_target_from_meta_refresh(self):
        html = '<meta http-equiv="refresh" content="0;url=https://www.python.org/">'
        self.assertEqual(extract_baidu_target(html), "https://www.python.org/")

    def test_extract_baidu_target_from_js(self):
        html = '<script>window.location.replace("https://docs.python.org/3/");</script>'
        self.assertEqual(extract_baidu_target(html), "https://docs.python.org/3/")

    def test_extract_baidu_target_gives_up_cleanly(self):
        self.assertEqual(extract_baidu_target("<html>没有跳转</html>"), "")
        self.assertEqual(extract_baidu_target(""), "")

    def test_baidu_can_resolve_redirects(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_get",
            side_effect=[self.BAIDU_HTML, self.BAIDU_HTML],
        ), mock.patch(
            "promptkit.default_tools.web_search.resolve_baidu_target",
            side_effect=["https://real1.com/", "https://real2.com/"],
        ):
            results = BaiduBackend(resolve_urls=True).search("q", k=2)
        self.assertEqual(
            [r.url for r in results], ["https://real1.com/", "https://real2.com/"]
        )

    def test_baidu_sends_browser_headers(self):
        '''裸请求一定会被百度拦（返回 1.4KB 的安全验证页），必须带浏览器头。'''
        with mock.patch(
            "promptkit.default_tools.web_search.http_get", return_value=self.BAIDU_HTML
        ) as fake:
            BaiduBackend().search("q")
        headers = fake.call_args.kwargs["headers"]
        self.assertIn("Referer", headers)
        self.assertIn("Accept", headers)

    def test_resolve_baidu_target_uses_the_final_url_on_http_redirect(self):
        response = mock.MagicMock()
        response.url = "https://real.com/page"
        response.read.return_value = b"<html>whatever</html>"
        response.__enter__ = lambda self: response
        response.__exit__ = lambda *args: False
        with mock.patch(
            "promptkit.default_tools.web_search.urllib.request.urlopen",
            return_value=response,
        ):
            self.assertEqual(resolve_baidu_target("http://baidu.com/link?url=x"), "https://real.com/page")

    def test_resolve_baidu_target_falls_back_to_the_meta_refresh_page(self):
        response = mock.MagicMock()
        response.url = "http://www.baidu.com/link?url=x"
        response.read.return_value = (
            b'<meta http-equiv="refresh" content="0;url=https://from-meta.com/">'
        )
        response.__enter__ = lambda self: response
        response.__exit__ = lambda *args: False
        with mock.patch(
            "promptkit.default_tools.web_search.urllib.request.urlopen",
            return_value=response,
        ):
            self.assertEqual(
                resolve_baidu_target("http://baidu.com/link?url=x"),
                "https://from-meta.com/",
            )

    def test_resolve_baidu_target_gives_up_quietly(self):
        with mock.patch(
            "promptkit.default_tools.web_search.urllib.request.urlopen",
            side_effect=OSError("boom"),
        ):
            self.assertEqual(resolve_baidu_target("http://baidu.com/link?url=x"), "")

    def test_baidu_complains_when_it_parses_nothing(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_get", return_value="<html></html>"
        ):
            with self.assertRaises(SearchError) as cm:
                BaiduBackend().search("q")
        message = str(cm.exception)
        self.assertIn("百度", message)
        self.assertIn("PROMPTKIT_SEARCH_BACKENDS", message)

    # ---------------- Bing 中国 ----------------
    BING_HTML = """
    <html><body><ol id="b_results">
      <li class="b_algo">
        <div class="b_tpcn"><a class="tilk" href="https://www.python.org/">
          <div class="tptt">python.org</div>
          <div class="tpmeta"><div class="b_attribution">www.python.org</div></div>
        </a></div>
        <h2><a target="_blank" href="https://www.python.org/"><strong>Welcome</strong> to Python.org</a></h2>
        <div class="b_caption"><p class="b_lineclamp2">The official home of the Python language.</p></div>
      </li>
      <li class="b_algo">
        <div class="b_tpcn"><a class="tilk" href="https://docs.python.org/3/">python.org</a></div>
        <h2><a href="https://docs.python.org/3/">Python 3 文档</a></h2>
        <div class="b_caption"><p>Python 3.13 及更早版本的文档。</p></div>
      </li>
      <li class="b_ans">不是搜索结果</li>
    </ol><script>ignored()</script></body></html>
    """

    def test_parse_bing(self):
        results = parse_bing(self.BING_HTML)
        self.assertEqual(len(results), 2)  # b_ans 不算
        self.assertEqual(results[0].url, "https://www.python.org/")
        self.assertIn("official home", results[0].snippet)
        self.assertEqual(results[0].source, "bing")

    def test_parse_bing_takes_the_title_from_h2_not_the_attribution(self):
        '''回归：真实页面里 `b_tpcn > a.tilk`（图标 + 域名 + 面包屑）排在 h2 前面。

        照「块里第一个链接」取，标题会变成 "python.orghttps://docs.python.org › …"。
        '''
        title = parse_bing(self.BING_HTML)[0].title
        self.assertEqual(title, "Welcome to Python.org")
        self.assertNotIn("python.org", title.split("Welcome")[0])  # 前面没有面包屑
        self.assertNotIn("›", title)

    def test_parse_bing_falls_back_to_the_attribution_link(self):
        '''万一没有 h2，至少还能拿到归属链接的地址。'''
        html = """
        <ol><li class="b_algo">
          <div class="b_tpcn"><a href="https://fallback.com/">site</a></div>
          <div class="b_caption"><p>摘要</p></div>
        </li></ol>
        """
        results = parse_bing(html)
        self.assertEqual(results[0].url, "https://fallback.com/")
        self.assertEqual(results[0].snippet, "摘要")

    def test_parse_bing_handles_chinese(self):
        self.assertEqual(parse_bing(self.BING_HTML)[1].title, "Python 3 文档")

    def test_parse_bing_ignores_scripts(self):
        self.assertNotIn("ignored", " ".join(r.snippet for r in parse_bing(self.BING_HTML)))

    def test_parse_bing_is_tolerant(self):
        for junk in ("", "<html></html>", "不是 HTML"):
            self.assertEqual(parse_bing(junk), [])

    # ---------------- 降级链 ----------------
    def test_fallback_uses_the_first_success(self):
        dead = mock.Mock()
        dead.name = "dead"
        dead.search.side_effect = SearchError("被限流")
        alive = mock.Mock()
        alive.name = "alive"
        alive.search.return_value = [SearchResult(title="OK", url="https://ok.com")]

        results = FallbackBackend([dead, alive]).search("q")
        self.assertEqual(results[0].title, "OK")
        dead.search.assert_called_once()
        alive.search.assert_called_once()

    def test_fallback_skips_a_backend_that_returns_nothing(self):
        empty = mock.Mock()
        empty.name = "empty"
        empty.search.return_value = []
        alive = mock.Mock()
        alive.name = "alive"
        alive.search.return_value = [SearchResult(url="https://ok.com")]
        self.assertEqual(len(FallbackBackend([empty, alive]).search("q")), 1)

    def test_fallback_reports_every_failure(self):
        bad = mock.Mock()
        bad.name = "bad"
        bad.search.side_effect = SearchError("挂了")
        with self.assertRaises(SearchError) as cm:
            FallbackBackend([bad, bad]).search("q")
        self.assertIn("所有后端都失败了", str(cm.exception))
        self.assertIn("bad", str(cm.exception))

    def test_fallback_needs_at_least_one_backend(self):
        with self.assertRaises(ValueError):
            FallbackBackend([])

    # ---------------- 博查 / 智谱 ----------------
    def test_bocha_parses(self):
        response = {
            "data": {
                "webPages": {
                    "value": [
                        {"name": "博查结果", "url": "https://b.com/1", "summary": "摘要"}
                    ]
                }
            }
        }
        with mock.patch(
            "promptkit.default_tools.web_search.http_post_json", return_value=response
        ) as fake:
            results = BochaBackend(api_key="k").search("q", k=2)
        self.assertEqual(results[0].title, "博查结果")
        self.assertEqual(results[0].url, "https://b.com/1")
        self.assertEqual(results[0].snippet, "摘要")
        self.assertEqual(results[0].source, "bocha")
        self.assertIn("Bearer k", fake.call_args.kwargs["headers"]["Authorization"])
        self.assertEqual(fake.call_args.args[1]["count"], 2)

    def test_zhipu_parses(self):
        response = {
            "search_result": [
                {"title": "智谱结果", "link": "https://z.com/1", "content": "内容"}
            ]
        }
        with mock.patch(
            "promptkit.default_tools.web_search.http_post_json", return_value=response
        ):
            results = ZhipuBackend(api_key="k").search("q")
        self.assertEqual(results[0].url, "https://z.com/1")
        self.assertEqual(results[0].source, "zhipu")

    def test_domestic_backends_require_a_key(self):
        for cls, env in ((BochaBackend, "BOCHA_API_KEY"), (ZhipuBackend, "ZHIPUAI_API_KEY")):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SearchError) as cm:
                    cls()
            self.assertIn(env, str(cm.exception))

    def test_changed_response_shape_is_diagnosable(self):
        with mock.patch(
            "promptkit.default_tools.web_search.http_post_json",
            return_value={"unexpected": "shape"},
        ):
            with self.assertRaises(SearchError) as cm:
                BochaBackend(api_key="k").search("q")
        message = str(cm.exception)
        self.assertIn("没返回结果", message)
        self.assertIn("unexpected", message)  # 把实际字段名报出来，方便排查

    def test_dig_and_pick_field_are_tolerant(self):
        data = {"a": {"b": [{"c": 1}]}}
        self.assertEqual(dig(data, "a.b.0.c"), 1)
        self.assertEqual(dig(data, "nope", "a.b.0.c"), 1)  # 会试下一个路径
        self.assertIsNone(dig(data, "nope"))
        self.assertEqual(pick_field({"name": "N", "title": "T"}, "title", "name"), "T")
        self.assertEqual(pick_field({"name": "N"}, "title", "name"), "N")
        self.assertEqual(pick_field({"title": "  "}, "title"), "")


class TestWebSearchTool(PromptkitTestCase):
    '''当工具给模型用时的行为。'''

    def test_tool_shape(self):
        tool = WebSearch().as_tool()
        self.assertEqual(tool.name, "web_search")
        properties = tool.schema()["function"]["parameters"]["properties"]
        self.assertEqual(sorted(properties), ["query", "top_k"])
        self.assertIn("互联网", tool.description)

    def test_tool_name_is_configurable(self):
        self.assertEqual(WebSearch().as_tool(name="search").name, "search")

    def test_tool_returns_formatted_results(self):
        stub = mock.Mock()
        stub.search.return_value = [
            SearchResult(title="T", url="https://x.com", snippet="S")
        ]
        tool = WebSearch(backend=stub).as_tool()
        out = tool.call({"query": "q", "top_k": 3})
        self.assertIn("[1] T", out)
        self.assertIn("https://x.com", out)
        stub.search.assert_called_once_with("q", k=3)

    def test_tool_returns_an_error_string_instead_of_raising(self):
        '''网络失败要变成工具结果，让模型自己决定怎么办，而不是炸掉整个对话。'''
        stub = mock.Mock()
        stub.search.side_effect = SearchError("连接失败：timed out")
        tool = WebSearch(backend=stub).as_tool()
        out = tool.call({"query": "q"})
        self.assertTrue(out.startswith("搜索失败："))
        self.assertIn("timed out", out)

    def test_tool_clamps_top_k(self):
        stub = mock.Mock()
        stub.search.return_value = []
        WebSearch(backend=stub).as_tool(k=4).call({"query": "q", "top_k": 999})
        self.assertEqual(stub.search.call_args.kwargs["k"], 20)

    def test_empty_query_does_not_hit_the_backend(self):
        stub = mock.Mock()
        self.assertEqual(WebSearch(backend=stub).search("   "), [])
        stub.search.assert_not_called()

    def test_search_text_is_prompt_ready(self):
        stub = mock.Mock()
        stub.search.return_value = [SearchResult(title="T", url="https://x.com")]
        self.assertIn("[1] T", WebSearch(backend=stub).search_text("q"))

    def test_backend_is_chosen_lazily(self):
        '''import 之后再设环境变量也要生效。'''
        search = WebSearch()
        self.assertIsNone(search._resolved)
        with mock.patch.dict(
            os.environ, {"PROMPTKIT_SEARCH_BACKENDS": "bing"}, clear=True
        ):
            self.assertIsInstance(search.resolve_backend(), BingCnBackend)

    def test_module_level_tool_exists(self):
        from promptkit.default_tools import WEB_SEARCH

        self.assertEqual(WEB_SEARCH.name, "web_search")


class TestItemAssignment(PromptkitTestCase):
    '''`Set(Var("d")["k"], v)` —— 读用下标表达式，写也要能写。'''

    def test_dict_item(self):
        ctx = Workflow(
            NewVar("info", default={}),
            Set(Var("info")["姓名"], "小明"),
        ).run()
        self.assertEqual(ctx["info"], {"姓名": "小明"})

    def test_list_index(self):
        ctx = Workflow(
            NewVar("xs", default=[0, 0, 0]),
            Set(Var("xs")[1], 42),
        ).run()
        self.assertEqual(ctx["xs"], [0, 42, 0])

    def test_nested_dict(self):
        ctx = Workflow(
            NewVar("d", default={"a": {}}),
            Set(Var("d")["a"]["b"], 1),
        ).run()
        self.assertEqual(ctx["d"], {"a": {"b": 1}})

    def test_key_can_be_an_expression(self):
        ctx = Workflow(
            NewVar("d", default={}),
            NewVar("k", default="动态键"),
            Set(Var("d")[Var("k")], "值"),
        ).run()
        self.assertEqual(ctx["d"], {"动态键": "值"})

    def test_value_can_be_an_expression(self):
        ctx = Workflow(
            NewVar("d", default={}),
            NewVar("n", default=1),
            Set(Var("d")["n"], Var("n") + 1),
        ).run()
        self.assertEqual(ctx["d"], {"n": 2})

    def test_read_back_with_the_item_expression(self):
        ctx = Workflow(
            NewVar("d", default={}),
            Set(Var("d")["k"], "v"),
            Set("got", Var("d")["k"]),
        ).run()
        self.assertEqual(ctx["got"], "v")

    def test_set_returns_a_step_not_an_expression(self):
        step = Set(Var("d")["k"], 1)
        self.assertIsInstance(step, Step)
        self.assertIsInstance(step, SetItemStep)
        self.assertEqual(step.name, "set_item")

    def test_plain_set_still_returns_set_step(self):
        self.assertIsInstance(Set("x", 1), SetStep)
        self.assertNotIsInstance(Set("x", 1), SetItemStep)

    def test_writing_into_a_non_container_is_reported_clearly(self):
        with self.assertRaises(TypeError) as cm:
            Workflow(NewVar("n", default=5), Set(Var("n")["k"], 1)).run()
        message = str(cm.exception)
        self.assertIn("不能给 int 赋下标值", message)
        self.assertIn("Var('n')", message)

    def test_missing_intermediate_key_is_a_key_error(self):
        # 不自动造中间容器 —— 读到不存在的键就是 KeyError，语义保持一致
        with self.assertRaises(KeyError):
            Workflow(
                NewVar("d", default={}), Set(Var("d")["nope"]["x"], 1)
            ).run()

    def test_step_tree_shows_the_assignment(self):
        step = Set(Var("info")["姓名"], "小明")
        self.assertIn("info", step.describe())
        self.assertIn("小明", step.describe())
        self.assertIn("SetItem", repr(step))

    def test_appears_in_the_workflow_tree(self):
        tree = repr(Workflow(NewVar("d", default={}), Set(Var("d")["k"], 1)))
        self.assertIn("['k'] = 1", tree)

    def test_bare_call_of_getitem_is_an_expression_not_a_step(self):
        '''回归：`Call(getitem(...), ...)` 是表达式，直接放进 Workflow 会报错。

        想写容器要用 Set(Var("d")["k"], v)。
        '''
        from operator import getitem

        expr = Call(getitem(Var("d"), "k"), 1)
        self.assertIsInstance(expr, Expr)
        self.assertNotIsInstance(expr, Step)
        with self.assertRaises(TypeError) as cm:
            Workflow(NewVar("d", default={}), expr)
        self.assertIn("期望 Step", str(cm.exception))

    def test_inc_still_works_on_a_plain_variable(self):
        ctx = Workflow(NewVar("n", default=1), Inc("n")).run()
        self.assertEqual(ctx["n"], 2)


class TestFileIO(PromptkitTestCase):
    '''文件 IO 三件套：Save / ReadFile / SaveHistory。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmp, name)

    def read(self, path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    # ---------------- Save ----------------
    def test_save_writes_one_value_per_line(self):
        target = self.path("a.txt")
        Workflow(Save("一", "二", "三", file=target)).run()
        self.assertEqual(self.read(target), "一\n二\n三\n")

    def test_save_coerces_non_strings(self):
        target = self.path("a.txt")
        Workflow(NewVar("d", default={"a": 1}), Save(42, Var("d"), file=target)).run()
        self.assertEqual(self.read(target), "42\n{'a': 1}\n")

    def test_save_accepts_bare_literals(self):
        '''回归：以前忘了 as_expr，字面量会 AttributeError。'''
        target = self.path("a.txt")
        Workflow(Save("字面量", file=target)).run()
        self.assertEqual(self.read(target), "字面量\n")

    def test_save_is_utf8(self):
        target = self.path("a.txt")
        Workflow(Save("中文 ✅ emoji", file=target)).run()
        with open(target, "rb") as handle:
            raw = handle.read()
        self.assertEqual(raw.decode("utf-8"), "中文 ✅ emoji\n")

    def test_save_writes_lf_not_crlf(self):
        '''回归：Windows 上默认会把 \\n 翻成 \\r\\n，同一份工作流在不同
        平台上写出的字节就不一样了。统一写 LF。'''
        target = self.path("a.txt")
        Workflow(Save("a", "b", file=target)).run()
        with open(target, "rb") as handle:
            raw = handle.read()
        self.assertEqual(raw, b"a\nb\n")
        self.assertNotIn(b"\r", raw)

    def test_save_creates_missing_directories(self):
        target = self.path(os.path.join("sub", "deep", "a.txt"))
        Workflow(Save("x", file=target)).run()
        self.assertTrue(os.path.exists(target))

    def test_save_append_mode(self):
        target = self.path("a.txt")
        Workflow(Save("第一行", file=target)).run()
        Workflow(Save("第二行", file=target, append=True)).run()
        self.assertEqual(self.read(target), "第一行\n第二行\n")

    def test_save_overwrites_by_default(self):
        target = self.path("a.txt")
        Workflow(Save("旧", file=target)).run()
        Workflow(Save("新", file=target)).run()
        self.assertEqual(self.read(target), "新\n")

    def test_save_with_no_values_writes_an_empty_file(self):
        target = self.path("a.txt")
        Workflow(Save(file=target)).run()
        self.assertEqual(self.read(target), "")

    def test_save_is_a_step(self):
        self.assertIsInstance(Save("x"), Step)
        self.assertIsInstance(Save("x"), SaveStep)
        self.assertEqual(Save("x").name, "save")

    def test_save_shows_up_in_the_tree(self):
        node = repr(Workflow(Save("a", file="o.txt"), Save("b", file="o.txt", append=True)))
        self.assertIn("Save('a')", node)
        self.assertIn(">>", node)

    def test_save_trace_records_the_file(self):
        target = self.path("a.txt")
        ctx = Workflow(Save("x", file=target)).run()
        self.assertIn(target, " ".join(ctx.trace))

    # ---------------- ReadFile ----------------
    def test_read_file_strips_by_default(self):
        target = self.path("a.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("  内容 ✅  \n")
        self.assertEqual(Workflow(Set("x", ReadFile(target))).run()["x"], "内容 ✅")

    def test_read_file_can_keep_whitespace(self):
        target = self.path("a.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("  内容  \n")
        ctx = Workflow(Set("x", ReadFile(target, strip=False))).run()
        self.assertEqual(ctx["x"], "  内容  \n")

    def test_read_file_from_file_flag_is_equivalent(self):
        target = self.path("a.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("一样")
        ctx = Workflow(
            Set("a", ReadFile(target)), Set("b", Read(target, from_file=True))
        ).run()
        self.assertEqual(ctx["a"], ctx["b"])

    def test_plain_read_still_reads_stdin(self):
        with mock.patch("builtins.input", return_value="  输入  "):
            self.assertEqual(Workflow(Set("x", Read("提示："))).run()["x"], "输入")

    def test_read_file_missing_is_a_loud_error(self):
        with self.assertRaises(FileNotFoundError) as cm:
            Workflow(Set("x", ReadFile(self.path("nope.txt")))).run()
        self.assertIn("读不到文件", str(cm.exception))

    def test_read_file_is_utf8_by_default(self):
        target = self.path("a.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("中文")
        self.assertEqual(Workflow(Set("x", ReadFile(target))).run()["x"], "中文")

    def test_read_file_is_lazy(self):
        # 构造工作流时不该碰文件
        expr = ReadFile(self.path("never-exists.txt"))
        self.assertIsInstance(expr, Expr)
        self.assertNotIsInstance(expr, Step)
        self.assertEqual(repr(expr), f"ReadFile({self.path('never-exists.txt')!r})")

    def test_read_expr_repr_still_says_read(self):
        self.assertEqual(repr(Read("岗位：")), "Read('岗位：')")

    # ---------------- SaveHistory ----------------
    def test_save_history_writes_json(self):
        target = self.path("h.json")
        Workflow(
            MessageStep(role="user", content="你好"),
            MessageStep(role="assistant", content="你好呀"),
            NewVar("名字", default="小明"),
            SaveHistory(target, include_variables=True),
        ).run()

        payload = json.loads(self.read(target))
        self.assertEqual(sorted(payload), ["messages", "trace", "variables"])
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好呀"},
            ],
        )
        self.assertIn("message", payload["trace"])
        self.assertEqual(payload["variables"]["名字"], "小明")

    def test_save_history_omits_variables_by_default(self):
        target = self.path("h.json")
        Workflow(NewVar("x", default=1), SaveHistory(target)).run()
        self.assertNotIn("variables", json.loads(self.read(target)))

    def test_save_history_skips_internal_runtime_state(self):
        target = self.path("h.json")
        with mock.patch(
            "promptkit.core.steps.actions.interactive._INTERNAL_KEYS",
            frozenset({"agent", "_output_stream", "_event_bus"}),
        ):
            Workflow(SaveHistory(target, include_variables=True)).run(agent=FakeAgent([]))
        variables = json.loads(self.read(target))["variables"]
        self.assertNotIn("agent", variables)
        self.assertNotIn("_output_stream", variables)

    def test_save_history_falls_back_to_repr(self):
        class Weird:
            def __repr__(self):
                return "<奇怪对象>"

        target = self.path("h.json")
        Workflow(
            NewVar("obj", default=Weird()), SaveHistory(target, include_variables=True)
        ).run()
        self.assertEqual(
            json.loads(self.read(target))["variables"]["obj"], "<奇怪对象>"
        )

    def test_save_history_serialises_pydantic_models(self):
        target = self.path("h.json")
        Workflow(
            NewVar("intent", default=Intent(label="退款")),
            SaveHistory(target, include_variables=True),
        ).run()
        self.assertEqual(
            json.loads(self.read(target))["variables"]["intent"], {"label": "退款"}
        )

    def test_save_history_handles_deep_structures(self):
        target = self.path("h.json")
        deep = {"a": {"b": {"c": [1, 2, {"d": (3, 4)}]}}}  # 元组也要能转
        Workflow(NewVar("d", default=deep), SaveHistory(target, include_variables=True)).run()
        variables = json.loads(self.read(target))["variables"]
        self.assertEqual(variables["d"]["a"]["b"]["c"][2]["d"], [3, 4])

    def test_save_history_is_utf8_and_pretty(self):
        target = self.path("h.json")
        Workflow(NewVar("名字", default="小明"), SaveHistory(target, include_variables=True)).run()
        raw = self.read(target)
        self.assertIn("小明", raw)  # ensure_ascii=False
        self.assertIn("\n  ", raw)  # indent=2

    def test_save_history_is_a_step(self):
        self.assertIsInstance(SaveHistory(), Step)
        self.assertIsInstance(SaveHistory(), SaveHistoryStep)
        self.assertEqual(SaveHistory().name, "save_history")

    def test_save_history_tree_shows_the_flag(self):
        self.assertIn("+ 变量", repr(Workflow(SaveHistory("h.json", include_variables=True))))

    # ---------------- 打包：导出 ----------------
    def test_exported_from_every_level(self):
        import importlib

        import promptkit.core as core_module

        for module_name in (
            "promptkit.core",
            "promptkit.core.steps",
            "promptkit.core.steps.actions",
        ):
            module = importlib.import_module(module_name)
            for name in (
                "Save",
                "SaveStep",
                "SaveHistory",
                "SaveHistoryStep",
                "ReadFile",
                "ReadFileExpr",
            ):
                self.assertIn(name, module.__all__, f"{module_name} 的 __all__ 缺 {name}")
                self.assertTrue(hasattr(module, name), f"{module_name} 没有 {name}")
                self.assertIs(
                    getattr(module, name),
                    getattr(core_module, name),
                    f"{module_name}.{name} 和 core.{name} 不是同一个对象",
                )

    def test_star_import_brings_them_in(self):
        namespace = {}
        exec("from promptkit.core import *", namespace)  # noqa: S102 - 就是要测 star import
        for name in ("Save", "SaveHistory", "ReadFile", "ReadFileExpr"):
            self.assertIn(name, namespace)


class TestGlobalSettings(PromptkitTestCase):
    '''DSL 的全局默认值：改一次，所有地方生效；单次调用仍可覆盖。'''

    # ---------------- 基本行为 ----------------
    def test_defaults(self):
        self.assertEqual(settings.max_iterations, 100)
        self.assertEqual(settings.max_for_iterations, 10000)
        self.assertEqual(settings.max_steps, 50)
        self.assertEqual(settings.max_jumps, 10000)
        self.assertEqual(settings.max_retries, 2)
        self.assertEqual(settings.max_tool_rounds, 8)
        self.assertFalse(settings.auto_schema)
        self.assertEqual(settings.encoding, "utf-8")

    def test_settings_names_and_to_dict(self):
        names = settings_names()
        self.assertIn("max_iterations", names)
        self.assertIn("encoding", names)
        self.assertEqual(sorted(settings.to_dict()), names)

    def test_using_settings_restores(self):
        with using_settings(max_iterations=7):
            self.assertEqual(settings.max_iterations, 7)
        self.assertEqual(settings.max_iterations, 100)

    def test_using_settings_restores_on_exception(self):
        with self.assertRaises(RuntimeError):
            with using_settings(max_iterations=7):
                raise RuntimeError("boom")
        self.assertEqual(settings.max_iterations, 100)

    def test_direct_assignment_is_global(self):
        settings.max_retries = 9
        self.assertEqual(settings.max_retries, 9)
        reset_settings()
        self.assertEqual(settings.max_retries, 2)

    def test_reset_settings(self):
        settings.max_iterations = 1
        settings.encoding = "gbk"
        reset_settings()
        self.assertEqual(settings.max_iterations, 100)
        self.assertEqual(settings.encoding, "utf-8")

    # ---------------- 校验 ----------------
    def test_unknown_setting_is_rejected(self):
        with self.assertRaises(TypeError) as cm:
            using_settings(nope=1)
        self.assertIn("没有这个设置项", str(cm.exception))
        self.assertIn("max_iterations", str(cm.exception))  # 列出可用的

    def test_non_positive_limit_is_rejected(self):
        for name in ("max_iterations", "max_for_iterations", "max_steps", "max_jumps"):
            with self.assertRaises(ValueError):
                using_settings(**{name: 0})
            with self.assertRaises(ValueError):
                using_settings(**{name: "10"})

    def test_negative_retries_is_rejected(self):
        with self.assertRaises(ValueError):
            using_settings(max_retries=-1)

    def test_zero_retries_is_allowed(self):
        with using_settings(max_retries=0):
            self.assertEqual(settings.max_retries, 0)

    def test_non_string_encoding_is_rejected(self):
        with self.assertRaises(TypeError):
            using_settings(encoding=123)

    # ---------------- 生效：循环 ----------------
    def test_max_iterations_applies_to_while(self):
        wf = Workflow(While(Var("n") < 999).Do(Set("n", Var("n") + 1)))
        with using_settings(max_iterations=5):
            with self.assertRaises(LoopLimitExceeded) as cm:
                wf.run(n=0)
        self.assertIn("5", str(cm.exception))

    def test_max_iterations_applies_to_a_prebuilt_workflow(self):
        '''重点：设置是在 execute 时读的，所以对已经建好的工作流也生效。'''
        wf = Workflow(While(Var("n") < 999).Do(Set("n", Var("n") + 1)))
        with using_settings(max_iterations=3):
            with self.assertRaises(LoopLimitExceeded):
                wf.run(n=0)

    def test_max_iterations_applies_to_do_while(self):
        wf = Workflow(DoWhile(Var("n") < 999).Do(Set("n", Var("n") + 1)))
        with using_settings(max_iterations=4):
            with self.assertRaises(LoopLimitExceeded) as cm:
                wf.run(n=0)
        self.assertIn("4", str(cm.exception))

    def test_max_for_iterations_applies_to_foreach(self):
        wf = Workflow(ForEach("i", range(100)).Do(SkipStep()))
        with using_settings(max_for_iterations=3):
            with self.assertRaises(LoopLimitExceeded) as cm:
                wf.run()
        self.assertIn("3", str(cm.exception))

    def test_max_for_iterations_applies_to_for(self):
        wf = Workflow(
            For(Set("i", 0), Var("i") < 100, Inc("i")).Do(SkipStep())
        )
        with using_settings(max_for_iterations=3):
            with self.assertRaises(LoopLimitExceeded) as cm:
                wf.run()
        self.assertIn("3", str(cm.exception))

    def test_explicit_limit_beats_the_global(self):
        with using_settings(max_iterations=2):
            ctx = Workflow(
                While(Var("n") < 4, max_iterations=10).Do(Set("n", Var("n") + 1))
            ).run(n=0)
        self.assertEqual(ctx["n"], 4)

    def test_while_and_foreach_have_separate_knobs(self):
        # 改 while 的上限不该影响 ForEach
        with using_settings(max_iterations=2):
            ctx = Workflow(ForEach("i", range(50)).Do(SkipStep())).run()
        self.assertIn("foreach:done", ctx.trace)

    def test_explicit_zero_is_still_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            While(True, max_iterations=0)
        with self.assertRaises(ValueError):
            ForEach("i", [1], max_iterations=0)

    # ---------------- 生效：工作流 ----------------
    def test_max_steps_applies_to_workflow(self):
        with using_settings(max_steps=2):
            with self.assertRaises(ValueError) as cm:
                Workflow(SkipStep(), SkipStep(), SkipStep())
        self.assertIn("步骤数超过上限 2", str(cm.exception))

    def test_max_steps_applies_to_add(self):
        wf = Workflow()
        with using_settings(max_steps=1):
            wf.add(SkipStep())
            with self.assertRaises(ValueError):
                wf.add(SkipStep())

    def test_max_jumps_applies_to_goto(self):
        wf = Workflow(
            Label("again"),
            Inc("n"),
            Goto("again"),
        )
        with using_settings(max_jumps=5):
            with self.assertRaises(LoopLimitExceeded) as cm:
                wf.run(n=0)
        self.assertIn("5", str(cm.exception))

    # ---------------- 生效：模型 ----------------
    def test_max_retries_applies_to_llm_step(self):
        step = LLMStep(prompt=Prompt(role="user", content="hi"), output_model=Intent)
        self.assertEqual(step.retry_count(), 2)
        with using_settings(max_retries=7):
            self.assertEqual(step.retry_count(), 7)

    def test_explicit_max_retries_beats_the_global(self):
        with using_settings(max_retries=7):
            step = LLMStep(
                prompt=Prompt(role="user", content="hi"),
                output_model=Intent,
                max_retries=1,
            )
        self.assertEqual(step.retry_count(), 1)

    def test_auto_schema_applies_to_llm_step(self):
        step = LLMStep(
            prompt=Prompt(role="user", content="hi"), output_model=Intent
        )
        self.assertNotIn("输出字段说明", step.build_prompt().content)
        with using_settings(auto_schema=True):
            self.assertIn("输出字段说明", step.build_prompt().content)

    def test_explicit_auto_schema_beats_the_global(self):
        step = LLMStep(
            prompt=Prompt(role="user", content="hi"),
            output_model=Intent,
            auto_schema=False,
        )
        with using_settings(auto_schema=True):
            self.assertNotIn("输出字段说明", step.build_prompt().content)

    def test_max_tool_rounds_applies_to_agent(self):
        def echo(x: str) -> str:
            """回声。"""
            return x

        responses = [
            fake_message(
                role="assistant",
                content=None,
                tool_calls=[tool_call("echo", "{}", f"c{i}")],
            )
            for i in range(10)
        ]
        agent = fake_agent(responses)
        with using_settings(max_tool_rounds=2):
            with self.assertRaises(ToolLoopExceeded) as cm:
                agent.ask("hi", tools=[echo])
        self.assertIn("2", str(cm.exception))

    def test_explicit_max_tool_rounds_beats_the_global(self):
        def echo(x: str) -> str:
            """回声。"""
            return x

        responses = [
            fake_message(
                role="assistant",
                content=None,
                tool_calls=[tool_call("echo", "{}", f"c{i}")],
            )
            for i in range(10)
        ]
        with using_settings(max_tool_rounds=99):
            with self.assertRaises(ToolLoopExceeded) as cm:
                fake_agent(responses).ask("hi", tools=[echo], max_tool_rounds=1)
        self.assertIn("1", str(cm.exception))

    # ---------------- 生效：文件 IO ----------------
    def test_encoding_applies_to_save_and_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "a.txt")
            with using_settings(encoding="utf-16"):
                Workflow(Save("中文", file=target)).run()
            # utf-16 的文件头是 BOM
            with open(target, "rb") as handle:
                self.assertTrue(handle.read(2) in (b"\xff\xfe", b"\xfe\xff"))
            with using_settings(encoding="utf-16"):
                self.assertEqual(Workflow(Set("x", ReadFile(target))).run()["x"], "中文")

    def test_explicit_encoding_beats_the_global(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "a.txt")
            with using_settings(encoding="utf-16"):
                Workflow(Save("中文", file=target, encoding="utf-8")).run()
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "中文\n")

    def test_current_encoding_helper(self):
        from promptkit.core.steps.actions.interactive import current_encoding

        self.assertEqual(current_encoding("gbk"), "gbk")
        with using_settings(encoding="gbk"):
            self.assertEqual(current_encoding(), "gbk")
            self.assertEqual(current_encoding("utf-8"), "utf-8")


class TestCautionPresets(PromptkitTestCase):
    '''新增的原子约束与预置约束组。'''

    # ---------------- 原子约束 ----------------
    def test_new_primitives_have_text(self):
        self.assertIn("Markdown", NoMarkdown().to_text())
        self.assertIn("不要编造", NoFabrication().to_text())
        self.assertIn("资料", Grounded().to_text())
        self.assertIn("简短", Concise().to_text())
        self.assertIn("分步骤", StepByStep().to_text())

    def test_citations_takes_a_style(self):
        self.assertIn("[编号]", Citations().to_text())
        self.assertIn("[来源: 标题]", Citations("[来源: 标题]").to_text())

    def test_ask_first_takes_a_count(self):
        self.assertIn("最多 3 个", AskFirst().to_text())
        self.assertIn("最多 5 个", AskFirst(5).to_text())

    # ---------------- 预置组 ----------------
    def test_json_strict_adds_markdown_and_fabrication_guards(self):
        text = JSON_STRICT.to_text()
        self.assertIn("JSON", text)
        self.assertIn("Markdown", text)       # 比 JSON_ONLY 多出来的
        self.assertIn("不要编造", text)
        # 严格版一定比宽松版长
        self.assertGreater(len(text), len(JSON_ONLY.to_text()))

    def test_json_strict_accepts_a_language(self):
        self.assertIn("请使用英文回答", json_strict("英文").to_text())

    def test_grounded_is_for_rag(self):
        text = GROUNDED.to_text()
        self.assertIn("只根据", text)
        self.assertIn("出处", text)
        self.assertIn("不要编造", text)

    def test_tutor_takes_a_persona(self):
        text = tutor("市场调研分析大赛的导师").to_text()
        self.assertIn("市场调研分析大赛的导师", text)
        self.assertIn("分步骤", text)
        self.assertIn("不要编造", text)

    def test_socratic_asks_before_answering(self):
        text = SOCRATIC.to_text()
        self.assertIn("先问我", text)
        self.assertIn("引导", text)

    def test_analyst(self):
        text = ANALYST.to_text()
        self.assertIn("数据分析师", text)
        self.assertIn("简短", text)

    def test_translator_outputs_only_the_translation(self):
        text = translator("日文").to_text()
        self.assertIn("译文", text)
        self.assertIn("日文", text)

    def test_concise_text(self):
        text = concise_text(50).to_text()
        self.assertIn("Markdown", text)
        self.assertIn("50", text)

    def test_groups_are_composable(self):
        group = GROUNDED + Tone("耐心")
        self.assertIsInstance(group, CautionGroup)
        self.assertIn("耐心", group.to_text())
        self.assertEqual(len(group), len(GROUNDED) + 1)

    def test_presets_work_in_an_llm_step(self):
        step = LLMStep(
            prompt=Prompt(role="user", content="根据资料回答"),
            output_model=Intent,
            cautions=[GROUNDED, JSON_STRICT],
        )
        text = step.build_prompt().content
        self.assertIn("只根据", text)
        self.assertIn("请以 JSON 格式输出", text)

    def test_preset_instances_are_independent(self):
        # 常量是共享实例，别被就地改坏
        before = GROUNDED.to_text()
        _ = grounded("英文")
        self.assertEqual(GROUNDED.to_text(), before)

    # ---------------- 导出 ----------------
    def test_new_names_are_exported(self):
        import promptkit.cautions as cautions

        for name in (
            "NoMarkdown",
            "NoFabrication",
            "Citations",
            "Grounded",
            "Concise",
            "StepByStep",
            "AskFirst",
            "json_strict",
            "concise_text",
            "grounded",
            "tutor",
            "socratic",
            "analyst",
            "translator",
            "JSON_STRICT",
            "GROUNDED",
            "TUTOR",
            "SOCRATIC",
            "ANALYST",
        ):
            self.assertIn(name, cautions.__all__, f"{name} 没导出")
            self.assertTrue(hasattr(cautions, name), f"{name} 取不到")

    def test_all_names_resolve(self):
        import promptkit.cautions as cautions

        missing = [n for n in cautions.__all__ if not hasattr(cautions, n)]
        self.assertEqual(missing, [])


class TestMathRendering(PromptkitTestCase):
    '''LaTeX -> Unicode。**零依赖**，所以任何环境都能测。'''

    def test_greek_letters(self):
        self.assertEqual(latex_to_unicode(r"\alpha\beta\gamma"), "αβγ")
        self.assertEqual(latex_to_unicode(r"\Gamma\Delta\Omega"), "ΓΔΩ")

    def test_operators(self):
        self.assertEqual(latex_to_unicode(r"a \times b"), "a × b")
        self.assertEqual(latex_to_unicode(r"a \cdot b"), "a · b")
        self.assertEqual(latex_to_unicode(r"a \le b \ge c"), "a ≤ b ≥ c")
        self.assertEqual(latex_to_unicode(r"a \ne b"), "a ≠ b")

    def test_superscripts_and_subscripts(self):
        self.assertEqual(latex_to_unicode("x^2"), "x²")
        self.assertEqual(latex_to_unicode("x^{12}"), "x¹²")
        self.assertEqual(latex_to_unicode("x_1"), "x₁")
        self.assertEqual(latex_to_unicode("x_{ij}"), "xᵢⱼ")
        self.assertEqual(latex_to_unicode(r"a_i^2"), "aᵢ²")

    def test_unmappable_script_keeps_the_caret(self):
        '''回归：以前把转不了的上下标连 `^` 一起吃掉，`\\theta^T` 变成 `\\thetaT`。'''
        self.assertEqual(latex_to_unicode(r"\theta^T"), "θ^T")
        self.assertIn("^", latex_to_unicode("x^y"))

    def test_fraction(self):
        self.assertEqual(latex_to_unicode(r"\frac{a}{b}"), "(a)/(b)")
        self.assertEqual(latex_to_unicode(r"\frac{a+b}{2}"), "(a+b)/(2)")

    def test_sqrt(self):
        self.assertEqual(latex_to_unicode(r"\sqrt{x}"), "√(x)")

    def test_binomial(self):
        '''`\binom` 是组合数，市场调研算样本量常用。'''
        self.assertEqual(latex_to_unicode(r"\binom{2n}{n}"), "C(2n, n)")
        self.assertIn("C(10, 3)", latex_to_unicode(r"\binom{10}{3}"))

    def test_delimiters_and_functions(self):
        self.assertEqual(latex_to_unicode(r"\lfloor x \rfloor"), "⌊ x ⌋")
        self.assertEqual(latex_to_unicode(r"\langle a, b \rangle"), "⟨ a, b ⟩")
        self.assertEqual(latex_to_unicode(r"\therefore"), "∴")
        self.assertEqual(latex_to_unicode(r"\ln x"), "ln x")
        # 下标里含空格 / 箭头时映射不了，原样保留（宁可难看也别改错意思）
        self.assertIn("lim", latex_to_unicode(r"\lim_{x \to 0}"))
        self.assertIn("→", latex_to_unicode(r"\lim_{x \to 0}"))

    def test_sums_and_integrals(self):
        self.assertEqual(latex_to_unicode(r"\sum_{i=1}^{n} i"), "∑ᵢ₌₁ⁿ i")
        self.assertEqual(latex_to_unicode(r"\int_0^1 x"), "∫₀¹ x")

    def test_accents(self):
        # NFC 归一化，所以是预组合的 U+0177，而不是 y + U+0302
        self.assertEqual(latex_to_unicode(r"\hat{y}"), "\u0177")
        self.assertEqual(latex_to_unicode(r"\hat{a}"), "\u00e2")
        self.assertEqual(latex_to_unicode(r"\bar{x}"), "x\u0304")

    def test_left_right_and_spacing_are_dropped(self):
        self.assertEqual(latex_to_unicode(r"\left( x \right)"), "( x )")
        self.assertEqual(latex_to_unicode(r"a \, b"), "a b")

    def test_unknown_command_drops_the_backslash(self):
        '''认不出的命令去掉反斜杠留名字 —— 比漏出 `\\weirdcmd` 有用得多，
        也比「被当成变量名」安全。内容不能丢。'''
        text = latex_to_unicode(r"\weirdcmd{x}")
        self.assertNotIn("\\", text)
        self.assertIn("weirdcmd", text)
        self.assertIn("x", text)

    def test_empty(self):
        self.assertEqual(latex_to_unicode(""), "")

    # ---------------- 「像公式吗」 ----------------
    def test_currency_is_not_math(self):
        '''回归：`$100，折扣 $50` 被当成公式，钱数被搅碎。'''
        self.assertFalse(looks_like_math("100，折扣 "))
        self.assertFalse(looks_like_math("100"))     # 数字开头 -> 是钱不是公式
        self.assertFalse(looks_like_math("1.5"))
        self.assertFalse(looks_like_math(""))

    def test_bare_symbol_counts_as_math(self):
        '''`$n$` / `$y$` 这种单个记号要认 —— 数学散文里太常见了。'''
        for body in ("n", "y", "abc", "x_1", r"\beta_0"):
            self.assertTrue(looks_like_math(body), body)

    def test_real_math_is_math(self):
        self.assertTrue(looks_like_math("x_1"))
        self.assertTrue(looks_like_math(r"\beta_0"))
        self.assertTrue(looks_like_math("y = mx + b"))
        self.assertTrue(looks_like_math("a^2"))

    def test_currency_survives_rendering(self):
        text = plain_text("价格是 $100，折扣 $50")
        self.assertIn("$100", text)
        self.assertIn("$50", text)

    def test_inline_math_is_converted(self):
        self.assertIn("β₀", plain_text(r"其中 $\beta_0$ 是截距"))

    # ---------------- 要不要请 sympy ----------------
    def test_needs_art(self):
        self.assertTrue(needs_art(r"\frac{a}{b}"))
        self.assertTrue(needs_art(r"\int_0^1 x dx"))     # 回归：\b 匹配不到
        self.assertTrue(needs_art(r"\sum_{i=1}^{n}"))
        self.assertTrue(needs_art(r"\sqrt{x}"))
        self.assertFalse(needs_art(r"\beta_0 + \beta_1 x_1"))
        self.assertFalse(needs_art("x^2 + 1"))

    def test_render_math_caches(self):
        render_math.cache_clear() if hasattr(render_math, "cache_clear") else None
        first = render_math(r"\alpha")
        self.assertEqual(first, render_math(r"\alpha"))

    def test_display_math_in_markdown(self):
        text = plain_text("公式：\n$$\ny = \\beta_0 + \\beta_1 x_1\n$$\n完")
        self.assertIn("β₀", text)
        self.assertIn("完", text)


class TestRenderFallback(PromptkitTestCase):
    '''没装 rich 时的兜底路径。'''

    def test_strip_markdown_removes_noise(self):
        text = strip_markdown("# 标题\n\n**粗体** 和 `代码`\n\n- 一项\n")
        self.assertNotIn("#", text)
        self.assertNotIn("**", text)
        self.assertNotIn("`", text)
        self.assertIn("标题", text)
        self.assertIn("粗体", text)

    def test_render_plain_goes_through_print(self):
        with mock.patch("builtins.print") as fake:
            render_plain("你好")
        self.assertEqual(printed(fake), ["你好"])

    def test_render_plain_converts_math(self):
        with mock.patch("builtins.print") as fake:
            render_plain(r"$x_1$ 是价格")
        self.assertIn("x₁", printed(fake)[0])

    def test_render_plain_with_title(self):
        with mock.patch("builtins.print") as fake:
            render_plain("正文", title="标题")
        lines = printed(fake)
        self.assertIn("标题", lines[0])
        self.assertEqual(lines[-1], "正文")

    def test_render_markdown_returns_the_original(self):
        captured = io.StringIO()
        result = render_markdown("# 标题", stream=captured)
        self.assertEqual(result, "# 标题")

    def test_render_markdown_falls_back_to_text(self):
        '''**强制**走兜底路径，而不是「恰好没装 rich 就跳过」。

        靠环境缺失来覆盖降级分支很脆：装了 rich 的机器上这条就永远不跑，
        真出问题时才发现不了。直接把开关按住。
        '''
        captured = io.StringIO()
        with mock.patch("promptkit.core.display_sys.render.HAVE_RICH", False):
            render_markdown("**粗体** 和 `代码`", stream=captured)
        self.assertEqual(captured.getvalue().strip(), "粗体 和 代码")

    def test_render_markdown_fallback_keeps_math(self):
        captured = io.StringIO()
        with mock.patch("promptkit.core.display_sys.render.HAVE_RICH", False):
            render_markdown(r"价格 $x_1$", stream=captured)
        self.assertIn("x₁", captured.getvalue())

    def test_render_markdown_fallback_writes_title(self):
        captured = io.StringIO()
        with mock.patch("promptkit.core.display_sys.render.HAVE_RICH", False):
            render_markdown("正文", stream=captured, title="标题")
        text = captured.getvalue()
        self.assertIn("标题", text)
        self.assertIn("正文", text)
        self.assertIn("─", text)

    def test_get_console_is_none_without_rich(self):
        with mock.patch("promptkit.core.display_sys.render.HAVE_RICH", False):
            self.assertIsNone(get_console())

    def test_rich_path_is_exercised_when_available(self):
        '''装了 rich 就顺便验一下富文本路径真的跑得通。'''
        if not HAVE_RICH:  # pragma: no cover - 没装就跳过
            self.skipTest("没装 rich")
        captured = io.StringIO()
        render_markdown("# 标题\n\n**粗体**", stream=captured)
        self.assertIn("粗体", captured.getvalue())

    def test_install_hint(self):
        self.assertIn("pip install", install_hint("render"))

    def test_rich_availability_flags_are_booleans(self):
        self.assertIsInstance(HAVE_RICH, bool)
        self.assertIsInstance(HAVE_SYMPY, bool)


class TestVerbosity(PromptkitTestCase):
    '''显示分级。'''

    def test_enum_values(self):
        self.assertEqual(Verbosity.SILENT, 0)
        self.assertEqual(Verbosity.NORMAL, 1)
        self.assertEqual(Verbosity.VERBOSE, 2)
        self.assertEqual(Verbosity.DEBUG, 3)

    def test_as_verbosity_accepts_everything(self):
        self.assertIs(as_verbosity(None), Verbosity.NORMAL)
        self.assertIs(as_verbosity(False), Verbosity.NORMAL)
        self.assertIs(as_verbosity(True), Verbosity.VERBOSE)
        self.assertIs(as_verbosity(0), Verbosity.SILENT)
        self.assertIs(as_verbosity(3), Verbosity.DEBUG)
        self.assertIs(as_verbosity(Verbosity.VERBOSE), Verbosity.VERBOSE)

    def test_as_verbosity_rejects_junk(self):
        with self.assertRaises(ValueError):
            as_verbosity(9)
        with self.assertRaises(TypeError):
            as_verbosity("verbose")

    def test_ordering(self):
        self.assertLess(Verbosity.SILENT, Verbosity.NORMAL)
        self.assertLess(Verbosity.NORMAL, Verbosity.VERBOSE)
        self.assertLess(Verbosity.VERBOSE, Verbosity.DEBUG)

    def test_silent_suppresses_echo(self):
        captured = io.StringIO()
        Workflow(Echo("不该出现")).run(verbose=Verbosity.SILENT, out=captured)
        self.assertEqual(captured.getvalue(), "")

    def test_normal_prints_echo(self):
        captured = io.StringIO()
        Workflow(Echo("该出现")).run(verbose=Verbosity.NORMAL, out=captured)
        self.assertEqual(captured.getvalue().strip(), "该出现")

    def test_old_true_false_still_works(self):
        captured = io.StringIO()
        Workflow(Echo("还在")).run(verbose=False, out=captured)
        self.assertEqual(captured.getvalue().strip(), "还在")

    def test_verbosity_is_visible_to_steps(self):
        ctx = Workflow().run(verbose=Verbosity.DEBUG)
        self.assertEqual(ctx[VERBOSITY_KEY], int(Verbosity.DEBUG))

    def test_verbose_goes_to_stderr_normal_to_stdout(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stderr", err):
            Workflow(Echo("结果")).run(verbose=Verbosity.VERBOSE, out=out)
        self.assertEqual(out.getvalue().strip(), "结果")      # 结果 -> stdout
        self.assertIn("Echo(", err.getvalue())                 # 过程 -> stderr
        self.assertNotIn("Echo(", out.getvalue())              # 过程不混进结果

    def test_normal_shows_no_process_at_all(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stderr", err):
            Workflow(Echo("结果")).run(verbose=Verbosity.NORMAL, out=out)
        self.assertEqual(out.getvalue().strip(), "结果")
        self.assertEqual(err.getvalue(), "")

    def test_verbose_int_is_accepted(self):
        out = io.StringIO()
        Workflow(Echo("x")).run(verbose=1, out=out)
        self.assertEqual(out.getvalue().strip(), "x")


class TestEchoRender(PromptkitTestCase):
    '''`Echo(render=...)`。'''

    def test_render_defaults_to_text(self):
        self.assertEqual(Echo("x").render_method, "text")

    def test_render_is_kept(self):
        self.assertEqual(Echo("x", render="markdown").render_method, "markdown")

    def test_title_is_kept(self):
        self.assertEqual(Echo("x", title="标题").title, "标题")

    def test_tree_shows_the_renderer(self):
        self.assertIn("render='markdown'", Echo("x", render="markdown").describe())
        self.assertNotIn("render=", Echo("x").describe())

    def test_markdown_render_does_not_crash_without_rich(self):
        captured = io.StringIO()
        Workflow(Echo("**粗体**", render="markdown")).run(out=captured)
        self.assertIn("粗体", captured.getvalue())

    def test_markdown_render_converts_math(self):
        captured = io.StringIO()
        Workflow(Echo(r"$x_1$", render="markdown")).run(out=captured)
        self.assertIn("x₁", captured.getvalue())

    def test_render_still_interpolates(self):
        captured = io.StringIO()
        Workflow(NewVar("n", default=7), Echo("{{n}}", render="markdown")).run(
            out=captured
        )
        self.assertIn("7", captured.getvalue())

    def test_plain_is_an_alias_of_text(self):
        captured = io.StringIO()
        Workflow(Echo("x", render="plain")).run(out=captured)
        self.assertEqual(captured.getvalue().strip(), "x")


class TestLogDisplay(PromptkitTestCase):
    '''把过程写进日志文件。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.log_path = os.path.join(self.tmp, "logs", "run.log")

    def tearDown(self):
        close_logging()
        self._tmp.cleanup()
        super().tearDown()

    def read_log(self):
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read()

    def test_setup_logging_creates_parent_dirs(self):
        setup_logging(self.log_path)
        get_logger().info("你好")
        self.assertTrue(os.path.exists(self.log_path))
        self.assertIn("你好", self.read_log())

    def test_setup_logging_replaces_instead_of_stacking(self):
        setup_logging(self.log_path)
        setup_logging(self.log_path)
        self.assertEqual(len(get_logger().handlers), 1)

    def test_step_events_are_logged(self):
        Workflow(Echo("x")).run(
            on_event=log_display(self.log_path), verbose=Verbosity.SILENT
        )
        text = self.read_log()
        self.assertIn("step_start", text)
        self.assertIn("echo", text)
        self.assertIn("step_end", text)

    def test_step_end_records_duration(self):
        Workflow(Echo("x")).run(
            on_event=log_display(self.log_path), verbose=Verbosity.SILENT
        )
        self.assertIn("duration_ms=", self.read_log())

    def test_tool_and_message_events_are_logged(self):
        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("echo", '{"x": "hi"}', "c1")],
                ),
                fake_message(role="assistant", content="好了"),
            ]
        )

        def echo(x: str) -> str:
            """回声。"""
            return x

        agent.ask("hi", tools=[echo], on_event=log_display(self.log_path))
        text = self.read_log()
        self.assertIn("tool_call name=echo", text)
        self.assertIn("tool_result name=echo", text)
        self.assertIn("message_end chars=2", text)

    def test_usage_is_logged(self):
        bus = EventBus()
        bus.on(log_display(self.log_path))
        bus.emit(
            Event(
                EventType.USAGE,
                {"prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550},
            )
        )
        text = self.read_log()
        self.assertIn("in=1200", text)
        self.assertIn("out=350", text)

    def test_agent_emits_usage_when_the_server_reports_it(self):
        message = fake_message(role="assistant", content="好")
        response = mock.Mock()
        response.choices = [mock.Mock(message=message)]
        response.usage = mock.Mock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        agent = Agent(api_key="k")
        agent.client = mock.Mock()
        agent.client.chat.completions.create.return_value = response

        events = []
        agent.ask("hi", on_event=events.append)
        usage = [e for e in events if e.type == EventType.USAGE]
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].data["total_tokens"], 15)

    def test_no_usage_object_is_fine(self):
        message = fake_message(role="assistant", content="好")
        response = mock.Mock(spec=["choices"])   # 没有 usage 属性
        response.choices = [mock.Mock(message=message)]
        agent = Agent(api_key="k")
        agent.client = mock.Mock()
        agent.client.chat.completions.create.return_value = response

        events = []
        agent.ask("hi", on_event=events.append)   # 不该炸
        self.assertFalse([e for e in events if e.type == EventType.USAGE])

    def test_thinking_deltas_are_skipped_by_default(self):
        bus = EventBus()
        bus.on(log_display(self.log_path))
        for etype in (EventType.THINKING_DELTA, EventType.MESSAGE_DELTA):
            bus.emit(Event(etype, {"text": "一堆字"}))
        self.assertNotIn("一堆字", self.read_log())

    def test_skip_argument(self):
        bus = EventBus()
        bus.on(log_display(self.log_path, skip=[EventType.STEP_START]))
        bus.emit(Event(EventType.STEP_START, {"name": "x"}))
        bus.emit(Event(EventType.STEP_END, {"name": "y"}))
        text = self.read_log()
        self.assertNotIn("name=x", text)
        self.assertIn("name=y", text)

    def test_broken_stream_does_not_break_the_run(self):
        '''终端和日志是两条独立的路：日志文件写不进去也不该弄挂工作流。'''
        target = os.path.join(self.tmp, "a-directory")
        os.makedirs(target)          # 拿目录当文件用
        with self.assertRaises(OSError):
            setup_logging(target)

    def test_terminal_and_file_are_independent(self):
        out = io.StringIO()
        Workflow(Echo("结果")).run(
            on_event=log_display(self.log_path),
            verbose=Verbosity.SILENT,
            out=out,
        )
        self.assertEqual(out.getvalue(), "")          # SILENT：终端没结果
        self.assertIn("step_start", self.read_log())  # 但日志是全的


class TestExportHistory(PromptkitTestCase):
    '''`Agent.export_history`。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def make_agent(self):
        agent = Agent(api_key="k")
        agent.history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
        ]
        return agent

    def test_json(self):
        path = os.path.join(self.tmp, "deep", "h.json")
        returned = self.make_agent().export_history(path)
        self.assertEqual(returned, os.path.abspath(path))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["content"], "你好")

    def test_jsonl(self):
        path = os.path.join(self.tmp, "h.jsonl")
        self.make_agent().export_history(path, fmt="jsonl")
        with open(path, encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["content"], "你好呀")

    def test_markdown(self):
        path = os.path.join(self.tmp, "h.md")
        self.make_agent().export_history(path, fmt="markdown")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("# 对话历史", text)
        self.assertIn("## 你", text)
        self.assertIn("## 助手", text)
        self.assertIn("你好呀", text)

    def test_md_is_an_alias(self):
        path = os.path.join(self.tmp, "h.md")
        self.make_agent().export_history(path, fmt="md")
        self.assertTrue(os.path.exists(path))

    def test_writes_lf_not_crlf(self):
        path = os.path.join(self.tmp, "h.jsonl")
        self.make_agent().export_history(path, fmt="jsonl")
        with open(path, "rb") as handle:
            self.assertNotIn(b"\r", handle.read())

    def test_bad_format_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            self.make_agent().export_history(os.path.join(self.tmp, "x"), fmt="xml")
        self.assertIn("jsonl", str(cm.exception))

    def test_empty_history(self):
        agent = Agent(api_key="k")
        agent.history = []
        path = os.path.join(self.tmp, "h.json")
        agent.export_history(path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), [])

    def test_tool_calls_show_up_in_markdown(self):
        agent = Agent(api_key="k")
        agent.history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "web_search"}}],
            }
        ]
        path = os.path.join(self.tmp, "h.md")
        agent.export_history(path, fmt="markdown")
        with open(path, encoding="utf-8") as handle:
            self.assertIn("web_search", handle.read())


class TestSaveJSON(PromptkitTestCase):
    '''`Save(..., format="json")` / `SaveJSON`。'''

    def setUp(self):
        super().setUp()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmp, name)

    def read(self, path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_dict_becomes_real_json(self):
        target = self.path("info.json")
        Workflow(
            NewVar("info", default={"b": 2, "a": 1}),
            Save(Var("info"), file=target, format="json"),
        ).run()
        self.assertEqual(json.loads(self.read(target)), {"b": 2, "a": 1})

    def test_text_format_still_uses_repr(self):
        target = self.path("info.txt")
        Workflow(
            NewVar("info", default={"a": 1}), Save(Var("info"), file=target)
        ).run()
        self.assertEqual(self.read(target), "{'a': 1}\n")

    def test_save_json_helper(self):
        target = self.path("info.json")
        Workflow(
            NewVar("info", default={"a": 1}), SaveJSON(Var("info"), target)
        ).run()
        self.assertEqual(json.loads(self.read(target)), {"a": 1})

    def test_pydantic_is_serialised(self):
        target = self.path("intent.json")
        Workflow(
            NewVar("intent", default=Intent(label="退款")),
            SaveJSON(Var("intent"), target),
        ).run()
        self.assertEqual(json.loads(self.read(target)), {"label": "退款"})

    def test_unicode_is_not_escaped(self):
        target = self.path("a.json")
        Workflow(SaveJSON("中文", target)).run()
        self.assertIn("中文", self.read(target))

    def test_json_rejects_multiple_values(self):
        with self.assertRaises(ValueError) as cm:
            Save("a", "b", format="json")
        self.assertIn("只能存一个值", str(cm.exception))

    def test_bad_format_is_rejected(self):
        with self.assertRaises(ValueError):
            Save("a", format="yaml")

    def test_writes_lf(self):
        target = self.path("a.json")
        Workflow(SaveJSON({"a": 1}, target)).run()
        with open(target, "rb") as handle:
            self.assertNotIn(b"\r", handle.read())


class TestRenderExports(PromptkitTestCase):
    '''渲染 / 日志这层也要能从短路径拿到。'''

    def test_core_exports(self):
        import promptkit.core as core

        for name in (
            "Verbosity",
            "as_verbosity",
            "render_markdown",
            "render_math",
            "render_plain",
            "strip_markdown",
            "latex_to_unicode",
            "HAVE_RICH",
            "HAVE_SYMPY",
            "install_hint",
            "setup_logging",
            "close_logging",
            "get_logger",
            "log_display",
        ):
            self.assertIn(name, core.__all__, f"{name} 没导出")
            self.assertTrue(hasattr(core, name), f"{name} 取不到")

    def test_all_names_resolve(self):
        import promptkit.core as core

        self.assertEqual([n for n in core.__all__ if not hasattr(core, n)], [])


class TestDisplaySysLayering(PromptkitTestCase):
    '''`core/display_sys/` 是叶子层：只管「怎么显示」，不认识 DSL。'''

    def test_subpackage_exports_everything(self):
        import promptkit.core.display_sys as display_sys

        missing = [n for n in display_sys.__all__ if not hasattr(display_sys, n)]
        self.assertEqual(missing, [])

    def test_four_modules_are_in_place(self):
        import promptkit.core.display_sys as display_sys

        for module in ("display", "render", "runlog", "tree"):
            self.assertTrue(
                hasattr(display_sys, module), f"display_sys.{module} 丢了"
            )

    def test_old_paths_are_gone(self):
        '''旧路径不该再能导入 —— 否则就是移动没做干净。'''
        import importlib

        for name in ("display", "render", "runlog", "tree"):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(f"promptkit.core.{name}")

    def test_display_sys_does_not_depend_on_steps(self):
        '''层级约束：显示层不该反过来依赖 DSL。

        这条要是挂了，说明有人往 display_sys 里 import 了 steps ——
        那样 `import promptkit.core.display_sys` 会把整个 DSL 拖进来，
        而且迟早绕成循环依赖。
        '''
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        folder = root / "core" / "display_sys"
        offenders = []
        for path in sorted(folder.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("import ", "from ")):
                    continue
                if "promptkit.core.steps" in stripped or "promptkit.core.model_action" in stripped:
                    offenders.append(f"{path.name}: {stripped}")
        self.assertEqual(offenders, [], "display_sys 不该依赖 steps / model_action")

    def test_display_sys_only_depends_on_event_sys(self):
        '''它唯一该依赖的兄弟是 event_sys（要认识事件类型）。

        `__init__.py` 里 `from promptkit.core.display_sys.xxx import ...` 是
        自己引用自己，不算依赖，要排除掉。
        '''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        folder = root / "core" / "display_sys"
        allowed = {"event_sys", "display_sys"}
        pattern = re.compile(r"from promptkit\.core\.(\w+)")
        offenders = []
        for path in sorted(folder.glob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                if match.group(1) not in allowed:
                    offenders.append(f"{path.name} -> {match.group(1)}")
        self.assertEqual(offenders, [])

    def test_old_core_names_still_work(self):
        '''搬家的只是模块路径，`from promptkit.core import ...` 一个都不能少。'''
        import promptkit.core as core

        for name in (
            "console_display",
            "json_display",
            "collect_events",
            "Verbosity",
            "render_markdown",
            "latex_to_unicode",
            "log_display",
            "setup_logging",
        ):
            self.assertTrue(hasattr(core, name), f"core.{name} 没了")
            self.assertIn(name, core.__all__)

    def test_core_and_display_sys_agree(self):
        import promptkit.core as core
        import promptkit.core.display_sys as display_sys

        for name in display_sys.__all__:
            self.assertIs(getattr(core, name), getattr(display_sys, name))


    def test_replace_retries_on_windows_permission_error(self):
        '''回归：Windows 上杀毒 / 索引服务会短暂占住刚写完的临时文件，
        `os.replace` 抛 WinError 5。实测偶发失败过，所以加了重试。'''
        from promptkit.knowledge import _replace_with_retry

        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "a.tmp")
            target = os.path.join(folder, "b.txt")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("内容")

            calls = {"n": 0}
            real_replace = os.replace

            def flaky(src, dst):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise PermissionError(5, "拒绝访问")
                return real_replace(src, dst)

            with mock.patch("promptkit.knowledge.os.replace", side_effect=flaky):
                _replace_with_retry(source, target)
            self.assertEqual(calls["n"], 3)          # 重试到成功为止
            self.assertTrue(os.path.exists(target))

    def test_replace_gives_up_eventually(self):
        from promptkit.knowledge import _replace_with_retry

        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "a.tmp")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("x")
            with mock.patch(
                "promptkit.knowledge.os.replace",
                side_effect=PermissionError(5, "拒绝访问"),
            ):
                with self.assertRaises(PermissionError):
                    _replace_with_retry(source, os.path.join(folder, "b.txt"))


class TestLatexDelimiters(PromptkitTestCase):
    '''四种公式定界符都要认。

    回归背景：模型（尤其 DeepSeek）最常写的是 `\\(...\\)` 和 `\\[...\\]`，
    而这两者**会被 Markdown 当转义序列吃掉** —— rich 输出成裸的 `(` / `[`，
    定界符消失，公式全成原文。所以必须在交给 rich 之前先转。
    '''

    def test_backslash_inline(self):
        text = plain_text(r"长度为 \(2n\) 的序列")
        self.assertIn("2n", text)
        self.assertNotIn("\\(", text)
        self.assertNotIn("\\)", text)
        # 定界符不能变成裸括号漏出来
        self.assertNotIn("(2n)", text)
        self.assertIn("的序列", text)

    def test_backslash_display(self):
        text = plain_text("前缀条件\n\\[ #P\\ge #Q. \\]\n完")
        self.assertIn("≥", text)          # \ge 转了
        self.assertNotIn("\\[", text)
        self.assertIn("完", text)

    def test_backslash_inline_converts_greek(self):
        self.assertIn("β₀", plain_text(r"其中 \(\beta_0\) 是截距"))

    def test_dollar_inline(self):
        self.assertIn("x₁", plain_text(r"其中 $x_1$ 是价格"))

    def test_dollar_display(self):
        self.assertIn("β₀", plain_text("公式\n$$\ny = \\beta_0\n$$\n完"))

    def test_all_four_delimiters_agree(self):
        body = r"\beta_0 + \beta_1"
        for template in (
            "$" + body + "$",
            "$$" + body + "$$",
            r"\(" + body + r"\)",
            r"\[" + body + r"\]",
        ):
            rendered = plain_text(template)
            self.assertIn("β₀", rendered, template)
            self.assertIn("β₁", rendered, template)

    def test_bare_brackets_are_not_math(self):
        '''`[链接](url)` / `(括号)` 这种普通文本不能被当成公式。'''
        for text in ("[链接](http://x)", "看这个（括号）", "[1, 2, 3]"):
            self.assertEqual(plain_text(text), text)

    def test_backslash_delimiters_bypass_the_math_heuristic(self):
        '''`\\(...\\)` 无歧义，不需要过 looks_like_math。

        即使内容不像公式（比如纯中文），也不能把定界符漏出去。
        '''
        text = plain_text(r"\(中文字\)")
        self.assertNotIn("\\(", text)
        self.assertNotIn("\\)", text)

    def test_markdown_render_does_not_leak_backslash_delimiters(self):
        '''关键回归：交给 rich 之后 `\\(` 不能变成裸的 `(`。'''
        captured = io.StringIO()
        render_markdown(r"用 \(P\) 表示入栈", stream=captured)
        out = captured.getvalue()
        self.assertNotIn("\\(", out)
        self.assertIn("P", out)

    def test_two_n_is_math(self):
        '''`$2n$` / `\\(2n\\)` 是公式，`$100` 是钱。'''
        self.assertTrue(looks_like_math("2n"))
        self.assertTrue(looks_like_math("3x"))
        self.assertFalse(looks_like_math("100"))
        self.assertFalse(looks_like_math("1.5"))

    def test_operators_mixed_between_digits_and_letters(self):
        for body in ("n+1", "2n-1", "n/2", "a*b", "x=1"):
            self.assertTrue(looks_like_math(body), body)

    def test_boxed_is_stripped_not_leaked(self):
        '''`\\boxed{x}` 只保留内容。

        回归：sympy 不认识 `\\boxed`，会把它当成**变量名**，输出 `boxed⋅x`。
        所以必须在解析前就拆掉，两条路径行为一致。
        '''
        text = latex_to_unicode(r"\boxed{\displaystyle C_n=\frac{1}{n+1}\binom{2n}{n}}")
        self.assertNotIn("\\boxed", text)
        self.assertNotIn("boxed", text)
        self.assertIn("Cₙ", text)
        self.assertIn("C(2n, n)", text)

    def test_nested_braces_are_handled(self):
        '''回归：以前用 `\\{([^{}]*)\\}`，遇到嵌套括号整条失效。'''
        self.assertEqual(
            latex_to_unicode(r"\boxed{\frac{\binom{2n}{n}}{n+1}}"),
            "(C(2n, n))/(n+1)",
        )
        self.assertEqual(latex_to_unicode(r"\frac{\frac{a}{b}}{c}"), "((a)/(b))/(c)")

    def test_display_variants_of_frac_and_binom(self):
        for body in (r"\dfrac{a}{b}", r"\tfrac{a}{b}"):
            self.assertEqual(latex_to_unicode(body), "(a)/(b)")
        for body in (r"\dbinom{2n}{n}", r"\tbinom{2n}{n}"):
            self.assertEqual(latex_to_unicode(body), "C(2n, n)")

    def test_proof_symbols(self):
        self.assertEqual(latex_to_unicode(r"\blacksquare"), "■")
        self.assertEqual(latex_to_unicode(r"\implies"), "⟹")
        self.assertEqual(latex_to_unicode(r"\therefore"), "∴")

    def test_cases_environment(self):
        text = latex_to_unicode(
            r"\begin{cases} u+d=2n \\ u-d=2 \end{cases}"
        )
        self.assertNotIn("\\begin", text)
        self.assertNotIn("cases", text)
        self.assertIn("u+d=2n", text)
        self.assertIn("u-d=2", text)
        self.assertIn("\n", text)   # 两行

    def test_sympy_boolean_results_are_rejected(self):
        '''回归：恒等式会被 sympy 算成 True / False，屏幕上只剩一个 False。'''
        if not HAVE_SYMPY:  # pragma: no cover
            self.skipTest("没装 sympy")
        self.assertEqual(render_math("x = x", display=True), "x = x")
        self.assertNotIn("True", render_math("x = x", display=True))

    def test_sympy_leaked_command_names_are_rejected(self):
        '''回归：`\\boxed` / `\\blacksquare` 会被 sympy 当变量名。'''
        if not HAVE_SYMPY:  # pragma: no cover
            self.skipTest("没装 sympy")
        rendered = render_math(r"\blacksquare \frac{a}{b}", display=True)
        self.assertNotIn("blacksquare", rendered)
        self.assertIn("■", rendered)

    def test_full_catalan_answer(self):
        '''用户那条回答的完整端到端片段。

        只断言「不再漏 LaTeX」这类**跟环境无关**的不变量 ——
        装了 sympy 会输出多行数学排版，没装是 `C(2n, n)` 这种平铺写法，
        两边都对，不该让测试依赖装没装 sympy。
        '''
        source = (
            r"因此合法出栈顺序总数为" "\n"
            r"\[ \binom{2n}{n}-\binom{2n}{n-1}. \]" "\n"
            r"即" "\n"
            r"\[ \boxed{\displaystyle \frac{1}{n+1}\binom{2n}{n}} \]"
        )
        captured = io.StringIO()
        render_markdown(source, stream=captured)
        out = captured.getvalue()
        # 定界符和命令都不能漏出去
        for leaked in ("\\[", "\\]", "\\binom", "\\boxed", "\\displaystyle", "\\frac"):
            self.assertNotIn(leaked, out, f"{leaked} 漏出来了")
        # 内容还在（sympy 会写成 2⋅n，降级路径是 2n，两边都算对）
        squashed = out.replace("⋅", "").replace("*", "").replace(" ", "")
        self.assertIn("2n", squashed)
        self.assertIn("n-1", squashed)
        self.assertIn("n+1", squashed)

    def test_standalone_proof_markers(self):
        '''`\\blacksquare` 常被模型写在公式**外面**当证明结束符。'''
        self.assertEqual(plain_text("\\blacksquare"), "■")
        self.assertEqual(plain_text("证毕 \\qed"), "证毕 ∎")
        self.assertEqual(plain_text("对 \\checkmark"), "对 ✓")
        # 不能误伤同前缀的命令
        self.assertIn("\\blacksquarefoo", plain_text("\\blacksquarefoo"))

    def test_displaystyle_does_not_break_sympy(self):
        '''回归：`\\displaystyle` 会让 sympy 的 parse_latex 直接失败，
        于是同一个文档里有的公式走了 sympy、有的走了降级，前后不一致。'''
        self.assertEqual(normalize_latex(r"\displaystyle x"), "x")
        self.assertNotIn("displaystyle", render_math(r"\displaystyle \frac{a}{b}", display=True))


class TestLatexRenderer(PromptkitTestCase):
    '''tokenizer 版渲染器。

    回归背景：以前是一堆积木式正则（`\\frac{a}{b}`、`\\boxed{x}` 各来一条），
    只能一个个补丁。TeX 的参数本来就**可以不写花括号**（`\\frac15`）、可以
    **任意嵌套**，正则根本覆盖不住 —— 用户的积分推导里 `\\frac15ln|x+1|`
    整条漏成原文、`{}` 和 `[2mm]` 也漏出来。
    '''

    # ---------------- TeX 简写参数 ----------------
    def test_frac_with_single_char_args(self):
        '''`\\frac15` 就是 `\\frac{1}{5}`，TeX 允许省略花括号。'''
        self.assertEqual(latex_to_unicode(r"\frac15"), "(1)/(5)")
        self.assertEqual(latex_to_unicode(r"\frac12 x"), "(1)/(2) x")
        self.assertEqual(latex_to_unicode(r"\frac ab"), "(a)/(b)")

    def test_superscript_single_char(self):
        self.assertEqual(latex_to_unicode("x^2"), "x²")
        self.assertEqual(latex_to_unicode(r"\alpha^2"), "α²")

    def test_sqrt_optional_index(self):
        self.assertEqual(latex_to_unicode(r"\sqrt[3]{x}"), "³√(x)")
        self.assertEqual(latex_to_unicode(r"\sqrt[n]{x}"), "ⁿ√(x)")

    # ---------------- 结构 ----------------
    def test_empty_group_is_invisible(self):
        '''回归：`={}` 里的 `{}` 之前会原样漏出来。'''
        self.assertEqual(latex_to_unicode("={} x"), "= x")
        self.assertEqual(latex_to_unicode("{}"), "")

    def test_line_break_with_spacing_is_dropped(self):
        '''回归：`\\\\[2mm]` 之前会漏成 `[2mm]`。'''
        self.assertEqual(latex_to_unicode(r"a \\[2mm] b"), "a\nb")
        self.assertEqual(latex_to_unicode(r"a \\ b"), "a\nb")
        self.assertNotIn("[2mm]", latex_to_unicode(r"a \\[2mm] b"))

    def test_left_right_are_transparent(self):
        self.assertEqual(latex_to_unicode(r"\left( x \right)"), "( x )")
        self.assertEqual(latex_to_unicode(r"\left[ x \right]"), "[ x ]")

    def test_displaystyle_is_dropped(self):
        self.assertEqual(latex_to_unicode(r"\displaystyle x"), "x")

    def test_deeply_nested(self):
        self.assertEqual(
            latex_to_unicode(r"\boxed{\frac{\binom{2n}{n}}{n+1}}"),
            "(C(2n, n))/(n+1)",
        )
        self.assertEqual(
            latex_to_unicode(r"\frac{\frac{a}{b}}{c}"), "((a)/(b))/(c)"
        )
        self.assertEqual(latex_to_unicode(r"\sqrt{\sqrt{x}}"), "√(√(x))")

    # ---------------- 保真：不能改模型的写法 ----------------
    def test_ln_is_not_rewritten_to_log(self):
        '''sympy 会把 `\\ln` 渲染成 `log(x, E)`，那不是模型写的东西。'''
        self.assertEqual(latex_to_unicode(r"\ln|x+1|"), "ln|x+1|")
        self.assertNotIn("log", latex_to_unicode(r"\ln x"))

    def test_arctan_is_not_rewritten_to_atan(self):
        self.assertEqual(latex_to_unicode(r"\arctan x"), "arctan x")
        self.assertNotIn("atan", latex_to_unicode(r"\arctan x"))

    def test_functions_keep_their_names(self):
        for name in ("ln", "log", "exp", "sin", "cos", "tan", "arctan", "lim"):
            self.assertEqual(latex_to_unicode("\\" + name), name)

    # ---------------- 环境 ----------------
    def test_cases_environment_becomes_lines(self):
        text = latex_to_unicode(r"\begin{cases} u+d=2n \\ u-d=2 \end{cases}")
        self.assertEqual(text, "u+d=2n\nu-d=2")

    def test_matrix_environment(self):
        text = latex_to_unicode(r"\begin{matrix} a & b \\ c & d \end{matrix}")
        self.assertNotIn("begin", text)
        self.assertNotIn("&", text)
        self.assertIn("a", text)
        self.assertIn("\n", text)

    # ---------------- sympy 默认关闭 ----------------
    def test_sympy_is_off_by_default(self):
        '''sympy 会渲染错（截断 / 改函数名 / 算成布尔值），所以默认不用它。'''
        from promptkit.core.display_sys.render import PREFER_SYMPY
        self.assertFalse(PREFER_SYMPY)

    def test_sympy_output_never_leaks_log_or_atan(self):
        text = render_math(r"\frac{\sqrt5-1}{20}\ln(x^2+\alpha x+1)", display=True)
        self.assertNotIn("log(", text)
        self.assertNotIn("atan(", text)

    def test_sympy_can_still_be_opted_into(self):
        if not HAVE_SYMPY:  # pragma: no cover
            self.skipTest("没装 sympy")
        # 显式打开：多行排版（分数会叠起来）
        art = render_math(r"\frac{a}{b}", display=True, prefer_sympy=True)
        self.assertTrue(art)                       # 不崩就行，具体形状由 sympy 决定

    # ---------------- 用户的整段推导 ----------------
    def test_the_integral_derivation_renders_cleanly(self):
        '''用户那个 ∫dx/(x^5+1) 的推导：一个 LaTeX 命令都不能漏。'''
        source = (
            r"\[" "\n"
            r"\int \frac{dx}{x^5+1}" "\n"
            r"={} \frac15\ln|x+1|" "\n"
            r"+\frac{\sqrt5-1}{20}\ln(x^2+\alpha x+1)" "\n"
            r"-\frac{\sqrt5+1}{20}\ln(x^2+\beta x+1)" "\n"
            r"\\[2mm]" "\n"
            r"+\frac15\sqrt{\frac{5+\sqrt5}{2}}\," "\n"
            r"\arctan\left(\frac{2x+\alpha}{\sqrt{(5+\sqrt5)/2}}\right)" "\n"
            r"+C." "\n"
            r"\]"
        )
        captured = io.StringIO()
        render_markdown(source, stream=captured)
        out = captured.getvalue()

        for leaked in ("\\frac", "\\ln", "\\arctan", "\\sqrt", "\\left",
                       "\\right", "\\[", "\\]", "[2mm]", "{}"):
            self.assertNotIn(leaked, out, f"{leaked} 漏出来了")
        # 也不能被 sympy 改样
        self.assertNotIn("atan(", out)
        self.assertNotIn("log(", out)
        # 该有的都在
        self.assertIn("ln|x+1|", out)
        self.assertIn("arctan", out)
        self.assertIn("α", out)
        self.assertIn("β", out)

    def test_long_formula_is_not_cropped(self):
        '''回归：rich 的 `soft_wrap=True` 会把超宽的代码块行**裁掉**。

        一段长公式在窄终端里后半截会凭空消失（`β` 那一项整个没了）。
        现在改成折行，内容一个字符都不能少。
        '''
        body = (
            r"\[ \int \frac{dx}{x^5+1} = \frac15\ln|x+1|"
            r"+\frac{\sqrt5-1}{20}\ln(x^2+\alpha x+1)"
            r"-\frac{\sqrt5+1}{20}\ln(x^2+\beta x+1) + C \]"
        )
        captured = io.StringIO()
        render_markdown(body, stream=captured)
        out = captured.getvalue()
        for fragment in ("ln|x+1|", "α", "β", "x⁵"):
            self.assertIn(fragment, out, f"{fragment} 被裁掉了")

    def test_no_latex_backslash_survives_anywhere(self):
        '''总闸：任何输入渲染完都不该还有反斜杠。'''
        samples = [
            r"\frac15", r"\boxed{x}", r"\dot{x}", r"\overline{AB}",
            r"\begin{pmatrix}1\\2\end{pmatrix}", r"a \\[1em] b",
            r"\text{中文}", r"\operatorname{sgn}", r"\left\langle x \right\rangle",
            r"\notacommand{y}", r"\sqrt[3]{\frac{a}{b}}",
        ]
        for sample in samples:
            self.assertNotIn("\\", latex_to_unicode(sample), sample)


class TestInOperator(PromptkitTestCase):
    '''`Var("x").in_(...)` —— `in` 是 Python 关键字，当不了方法名。'''

    def test_in_works(self):
        ctx = Workflow(
            NewVar("x", default="exit"),
            If(Var("x").in_(["exit", "quit"])).Then(Set("hit", "yes")).Else(Set("hit", "no")),
        ).run()
        self.assertEqual(ctx["hit"], "yes")

    def test_in_is_false_when_absent(self):
        ctx = Workflow(
            NewVar("x", default="go"),
            If(Var("x").in_(["exit"])).Then(Set("hit", "yes")),
        ).run()
        self.assertIsNone(ctx.get("hit"))

    def test_not_in(self):
        ctx = Workflow(
            NewVar("x", default="go"),
            If(Var("x").not_in(["exit"])).Then(Set("hit", "yes")).Else(Set("hit", "no")),
        ).run()
        self.assertEqual(ctx["hit"], "yes")

    def test_in_a_string(self):
        ctx = Workflow(
            NewVar("x", default="abc"),
            If(Var("x").in_("abcdef")).Then(Set("hit", "yes")).Else(Set("hit", "no")),
        ).run()
        self.assertEqual(ctx["hit"], "yes")

    def test_in_a_dict(self):
        ctx = Workflow(
            NewVar("x", default="k"),
            NewVar("d", default={"k": 1}),
            If(Var("x").in_(Var("d"))).Then(Set("hit", "yes")),
        ).run()
        self.assertEqual(ctx["hit"], "yes")

    def test_repr(self):
        self.assertEqual(repr(Var("x").in_(["a"])), "(Var('x') in ['a'])")


class FakeChatAgent:
    '''模仿真 Agent 的接口，但不起网络。'''

    def __init__(self, reply="嗯"):
        self.start_prompts = [{"role": "system", "content": "默认"}]
        self.history = []
        self.messages = list(self.start_prompts)
        self.prompts = []
        self.tools_seen = []
        self._reply = reply

    def reset(self):
        self.history = []
        self.messages = list(self.start_prompts)

    def rebuild(self):
        self.messages = [*self.start_prompts, *self.history]

    def load_history(self, records, replace=True):
        from promptkit.agent import normalize_history

        messages = normalize_history(records)
        self.history = messages if replace else [*self.history, *messages]
        self.rebuild()
        return len(messages)

    def get_history(self):
        return list(self.history)

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        content = prompt.get("content") if isinstance(prompt, dict) else str(prompt)
        self.prompts.append(content)
        self.tools_seen.append(
            [
                getattr(t, "name", getattr(t, "__name__", repr(t)))
                for t in (tools or [])
            ]
        )
        reply = f"{self._reply}{len(self.prompts)}：{content}"
        self.history.append({"role": "user", "content": content})
        self.history.append({"role": "assistant", "content": reply})
        self.rebuild()          # 真 Agent 的 _append_message 也会重建
        return reply


class TestChatUtilSteps(PromptkitTestCase):
    '''`chat_util` 里的声明式 Step：只往 Context 里写配置。'''

    def test_role(self):
        self.assertEqual(Workflow(Role("导师")).run()["chat_role"], "导师")

    def test_role_last_wins(self):
        '''回归：`Context.declare` 在键已存在时不覆盖，`Role` 必须用赋值。'''
        ctx = Workflow(Role("A"), Role("B")).run()
        self.assertEqual(ctx["chat_role"], "B")

    def test_want(self):
        self.assertEqual(Workflow(Want("找选题")).run()["chat_want"], "找选题")

    def test_rules(self):
        ctx = Workflow(Rules("一", "二")).run()
        self.assertEqual(ctx["chat_rules"], ["一", "二"])

    def test_rules_accumulate_by_default(self):
        '''多次 `Rules(...)` 应该**累积**，不是后者覆盖前者。

        用户天然会一行写一条规矩（`Rules(a)` / `Rules(b)` / ...），
        如果默认覆盖，除最后一条之外全部静默丢失。
        '''
        ctx = Workflow(Rules("一"), Rules("二"), Rules("三")).run()
        self.assertEqual(ctx["chat_rules"], ["一", "二", "三"])

    def test_rules_can_replace(self):
        ctx = Workflow(Rules("一"), Rules("二", replace=True)).run()
        self.assertEqual(ctx["chat_rules"], ["二"])

    def test_tools(self):
        def search(q: str) -> str:
            """搜。"""
            return q

        ctx = Workflow(Tools(search)).run()
        self.assertEqual(ctx["chat_tools"], [search])

    def test_tools_accumulate(self):
        def a(q: str) -> str:
            """A。"""
            return q

        def b(q: str) -> str:
            """B。"""
            return q

        ctx = Workflow(Tools(a), Tools(b)).run()
        self.assertEqual(ctx["chat_tools"], [a, b])

    def test_deliverables(self):
        item = Deliverable("a", save_to="x.md")
        ctx = Workflow(Deliverables(item)).run()
        self.assertEqual(ctx["chat_deliverables"], [item])

    def test_exit_words_defaults(self):
        ctx = Workflow(ExitWords()).run()
        self.assertEqual(ctx["chat_exit_words"], list(DEFAULT_EXIT_WORDS))

    def test_exit_words_custom(self):
        ctx = Workflow(ExitWords("拜拜")).run()
        self.assertEqual(ctx["chat_exit_words"], ["拜拜"])

    def test_behavior_runs_its_steps(self):
        ctx = Workflow(Behavior(Set("hit", 1), Set("hit2", 2))).run()
        self.assertEqual(ctx["hit"], 1)
        self.assertEqual(ctx["hit2"], 2)

    def test_behavior_exposes_children(self):
        self.assertEqual(len(Behavior(Set("a", 1)).children()), 1)

    def test_chat_context_vars_lists_keys(self):
        for key in chat_context_vars():
            self.assertIsInstance(key, str)


class TestSystemPrompt(PromptkitTestCase):
    '''把声明编译成 system prompt 装到 Agent 上。'''

    def test_build_system_prompt(self):
        text = build_system_prompt(
            role="导师",
            want="找选题",
            rules=["专业", "简短"],
            deliverables=[Deliverable("选题建议", description="3-5 个", format="markdown")],
        )
        self.assertIn("你是导师", text)
        self.assertIn("你的目标：找选题", text)
        self.assertIn("- 专业", text)
        self.assertIn("选题建议：3-5 个（markdown 格式）", text)

    def test_build_system_prompt_defaults(self):
        self.assertIn("AI 助手", build_system_prompt())

    def test_system_prompt_installs_on_agent(self):
        agent = FakeChatAgent()
        Workflow(Role("导师"), Want("找选题"), SystemPrompt()).run(agent=agent)
        self.assertEqual(agent.start_prompts[0]["role"], "system")
        self.assertIn("你是导师", agent.start_prompts[0]["content"])

    def test_system_prompt_resets_so_messages_update(self):
        '''回归：只改 `start_prompts` 不 `reset()` 的话，`messages` 还是旧的。'''
        agent = FakeChatAgent()
        agent.history = [{"role": "user", "content": "旧历史"}]
        Workflow(Role("新角色"), SystemPrompt()).run(agent=agent)
        self.assertEqual(agent.messages[0]["content"], agent.start_prompts[0]["content"])
        self.assertEqual(agent.history, [])          # reset 清掉了旧历史

    def test_system_prompt_can_keep_history(self):
        agent = FakeChatAgent()
        agent.history = [{"role": "user", "content": "旧历史"}]
        Workflow(Role("新角色"), SystemPrompt(reset=False)).run(agent=agent)
        self.assertEqual(len(agent.history), 1)     # 没清

    def test_system_prompt_needs_an_agent(self):
        with self.assertRaises(ValueError) as cm:
            Workflow(SystemPrompt()).run()
        self.assertIn("agent", str(cm.exception))

    def test_custom_template_wins(self):
        agent = FakeChatAgent()
        Workflow(Role("导师"), SystemPrompt(template="固定的话")).run(agent=agent)
        self.assertEqual(agent.start_prompts[0]["content"], "固定的话")


class TestChatLoop(PromptkitTestCase):
    '''多轮对话循环。'''

    def run_loop(self, inputs, verbose=Verbosity.SILENT, **kwargs):
        agent = FakeChatAgent()
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=list(inputs)):
            ctx = Workflow(ChatLoop(**kwargs)).run(
                agent=agent, out=out, verbose=verbose
            )
        return ctx, agent, out.getvalue()

    def test_loops_until_exit(self):
        ctx, agent, _ = self.run_loop(["a", "b", "exit"])
        self.assertEqual(len(agent.prompts), 2)      # exit 那轮不调模型
        self.assertEqual(ctx["chat_signal"], "end")

    def test_default_exit_words(self):
        for word in ("exit", "quit", ":q"):
            _, agent, _unused = self.run_loop([word])
            self.assertEqual(len(agent.prompts), 0, word)

    def test_custom_exit_words(self):
        _, agent, _ = self.run_loop(["拜拜"], exit_words=["拜拜"])
        self.assertEqual(len(agent.prompts), 0)

    def test_reply_is_shown(self):
        # 注意用 NORMAL：SILENT 连 Echo 都不打，看不到回答
        _, _, output = self.run_loop(["你好", "exit"], verbose=Verbosity.NORMAL)
        self.assertIn("嗯1", output)

    def test_tools_are_read_from_context(self):
        def search(q: str) -> str:
            """搜。"""
            return q

        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            Workflow(Tools(search), ChatLoop()).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.tools_seen, [["search"]])

    def test_behavior_runs_before_the_model(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["help", "exit"]):
            ctx = Workflow(
                ChatLoop(
                    behavior=[If(Var("usr_input") == "help").Then(Set("helped", 1))]
                )
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(ctx.get("helped"), 1)
        self.assertEqual(len(agent.prompts), 1)

    def test_max_turns(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["a"] * 20):
            with self.assertRaises(LoopLimitExceeded):
                Workflow(ChatLoop(max_turns=3)).run(
                    agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
                )


class TestChatWorkflow(PromptkitTestCase):
    '''`ChatWorkflow` —— 把声明编译成普通 Workflow。'''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmp, name)

    # ---------------- 编译 ----------------
    def test_build_returns_a_plain_workflow(self):
        self.assertIsInstance(ChatWorkflow(role="导师").build(), Workflow)

    def test_build_shows_the_declarations(self):
        tree = repr(ChatWorkflow(role="导师", want="找选题", rules=["专业"]).build())
        for fragment in ("Role(", "Want(", "Rules(", "SystemPrompt()", "chat_loop"):
            self.assertIn(fragment, tree)

    def test_build_system_prompt(self):
        text = ChatWorkflow(
            role="导师",
            want="找选题",
            rules=["专业"],
            deliverables=[Deliverable("选题建议", description="3-5 个")],
        ).build_system_prompt()
        self.assertIn("你是导师", text)
        self.assertIn("找选题", text)
        self.assertIn("选题建议", text)

    def test_deliverables_split_by_when(self):
        chat = ChatWorkflow(
            deliverables=[
                Deliverable("a", when="end"),
                Deliverable("b", when="each_turn"),
            ]
        )
        self.assertEqual([d.name for d in chat.end_deliverables()], ["a"])
        self.assertEqual([d.name for d in chat.each_turn_deliverables()], ["b"])

    def test_deliverable_rejects_bad_when(self):
        with self.assertRaises(ValueError):
            Deliverable("a", when="sometimes")

    def test_deliverable_needs_a_name(self):
        with self.assertRaises(ValueError):
            Deliverable("")

    # ---------------- 端到端 ----------------
    def test_run_installs_system_prompt_and_chats(self):
        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", want="找选题", rules=["专业"])
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            ctx = chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())

        self.assertIn("你是导师", agent.start_prompts[0]["content"])
        self.assertEqual(len(agent.prompts), 1)
        self.assertEqual(ctx["chat_role"], "导师")

    def test_deliverable_is_saved(self):
        target = self.path("out.md")
        agent = FakeChatAgent()
        chat = ChatWorkflow(
            role="导师",
            deliverables=[Deliverable("选题建议", format="markdown", save_to=target)],
        )
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())

        self.assertTrue(os.path.exists(target))
        with open(target, encoding="utf-8") as handle:
            self.assertIn("嗯1", handle.read())

    def test_json_deliverable(self):
        target = self.path("out.json")
        agent = FakeChatAgent()
        chat = ChatWorkflow(
            role="导师",
            deliverables=[Deliverable("数据", format="json", save_to=target)],
        )
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())

        with open(target, encoding="utf-8") as handle:
            self.assertIsInstance(json.load(handle), str)   # 模型回的是字符串

    def test_deliverable_without_save_to_is_skipped(self):
        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", deliverables=[Deliverable("只在内存里")])
        with mock.patch("builtins.input", side_effect=["exit"]):
            ctx = chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(ctx["chat_signal"], "end")

    def test_per_turn_deliverable(self):
        target = self.path("each.md")
        agent = FakeChatAgent()
        chat = ChatWorkflow(
            role="导师",
            deliverables=[
                Deliverable("每轮", format="markdown", save_to=target, when="each_turn")
            ],
        )
        with mock.patch("builtins.input", side_effect=["一", "二", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        with open(target, encoding="utf-8") as handle:
            self.assertIn("嗯2", handle.read())     # 最后一轮的

    def test_tools_are_wired(self):
        def search(q: str) -> str:
            """搜。"""
            return q

        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", tools=[search])
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(agent.tools_seen, [["search"]])

    def test_behavior_is_wired(self):
        agent = FakeChatAgent()
        chat = ChatWorkflow(
            role="导师",
            behavior=[If(Var("usr_input") == "help").Then(Echo("输入 exit 退出"))],
        )
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=["help", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.NORMAL, out=out)
        self.assertIn("输入 exit 退出", out.getvalue())

    def test_needs_an_agent(self):
        '''没有默认 Agent 时要给清晰报错，而不是 AttributeError。'''
        with self.assertRaises(ValueError) as cm:
            ChatWorkflow(role="导师").run()
        self.assertIn("agent", str(cm.exception))

    def test_matches_a_hand_written_workflow(self):
        '''「编译出来的」和「手写的」行为要一致 —— 这就是语法糖的意思。'''
        chat = ChatWorkflow(role="导师", exit_words=["exit"])

        agent_a = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["a", "exit"]):
            chat.run(agent=agent_a, verbose=Verbosity.SILENT, out=io.StringIO())

        agent_b = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["a", "exit"]):
            Workflow(
                Role("导师"),
                SystemPrompt(),
                ChatLoop(exit_words=["exit"]),
            ).run(agent=agent_b, verbose=Verbosity.SILENT, out=io.StringIO())

        self.assertEqual(agent_a.prompts, agent_b.prompts)
        self.assertEqual(
            agent_a.start_prompts[0]["content"],
            agent_b.start_prompts[0]["content"],
        )

    def test_chat_workflow_function_form(self):
        self.assertIsInstance(chat_workflow(role="导师"), ChatWorkflow)


class TestWriteFile(PromptkitTestCase):
    '''`write_file` 工具：让**模型**决定什么时候把成果写进文件。

    这是唯一一个「由大模型驱动写磁盘」的能力，所以沙箱是重点。
    '''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.writer = WriteFile(self.tmp)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def path(self, name):
        return os.path.join(self.tmp, name.replace("/", os.sep))

    def read(self, name):
        with open(self.path(name), encoding="utf-8") as handle:
            return handle.read()

    # ---------------- 模型看到的契约 ----------------
    def test_tool_name_and_description(self):
        '''名字/描述是模型决定要不要调的唯一依据，不能用类的文档冒充。'''
        tool = self.writer.as_tool()
        self.assertEqual(tool.name, "write_file")
        self.assertIn("文件", tool.description)
        self.assertIn("保存", tool.description)

    def test_tool_schema(self):
        tool = self.writer.as_tool()
        self.assertEqual(sorted(tool.parameters["properties"]), ["content", "name"])
        self.assertEqual(sorted(tool.parameters["required"]), ["content", "name"])

    def test_as_tool_matches_tools_writer(self):
        '''`Tools(writer)` 和 `writer.as_tool()` 必须给出同一个定义。

        `as_tools` 对可调用对象走 `Tool.from_function`，靠对象上的
        `__name__` / `__doc__`；这条防止那套约定哪天失效。
        '''
        from promptkit.tools import as_tool

        via_class = self.writer.as_tool()
        via_generic = as_tool(self.writer)
        self.assertEqual(via_class.name, via_generic.name)
        self.assertEqual(via_class.description, via_generic.description)
        self.assertEqual(via_class.parameters, via_generic.parameters)

    # ---------------- 正常写入 ----------------
    def test_writes_a_file(self):
        message = self.writer("a.md", "# 标题")
        self.assertIn("已写入", message)
        self.assertEqual(self.read("a.md"), "# 标题\n")

    def test_creates_parent_directories(self):
        self.writer("deep/nested/a.md", "x")
        self.assertTrue(os.path.exists(self.path("deep/nested/a.md")))

    def test_writes_utf8_and_lf(self):
        self.writer("a.md", "中文 ✅")
        with open(self.path("a.md"), "rb") as handle:
            raw = handle.read()
        self.assertEqual(raw.decode("utf-8"), "中文 ✅\n")
        self.assertNotIn(b"\r", raw)

    def test_adds_a_trailing_newline(self):
        self.writer("a.md", "x")
        self.assertTrue(self.read("a.md").endswith("\n"))

    def test_does_not_double_the_trailing_newline(self):
        self.writer("a.md", "x\n")
        self.assertEqual(self.read("a.md"), "x\n")

    def test_strips_a_whole_content_fence(self):
        '''模型很爱把文件内容写成 ```markdown ... ```，存下去会带上围栏。'''
        self.writer("a.md", "```markdown\n# 标题\n正文\n```")
        self.assertEqual(self.read("a.md"), "# 标题\n正文\n")

    def test_keeps_inner_fences(self):
        '''只剥「整段就是一层围栏」的情况，正文里的代码块不能动。'''
        self.writer("a.md", "# 标题\n\n```python\nprint(1)\n```\n")
        self.assertIn("```python", self.read("a.md"))

    def test_overwrite(self):
        self.writer("a.md", "旧")
        self.writer("a.md", "新")
        self.assertEqual(self.read("a.md"), "新\n")

    def test_no_overwrite_mode(self):
        writer = WriteFile(self.tmp, overwrite=False)
        writer("a.md", "旧")
        message = writer("a.md", "新")
        self.assertIn("已存在", message)
        self.assertEqual(self.read("a.md"), "旧\n")

    # ---------------- 沙箱 ----------------
    def test_rejects_parent_traversal(self):
        for name in ("../x.md", "a/../../x.md", "a/../b.md"):
            message = self.writer(name, "x")
            self.assertIn("错误", message, name)

    def test_rejects_absolute_paths(self):
        for name in ("/etc/passwd", "C:\\Windows\\x.txt", "\\\\server\\share\\x.md"):
            self.assertIn("错误", self.writer(name, "x"), name)

    def test_rejects_bad_characters(self):
        self.assertIn("错误", self.writer("a:b.md", "x"))

    def test_rejects_empty_name(self):
        self.assertIn("错误", self.writer("", "x"))
        self.assertIn("错误", self.writer("   ", "x"))

    def test_nothing_escapes_the_root(self):
        self.writer("../逃.md", "x")
        self.writer("/etc/passwd", "x")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "..", "逃.md")))

    def test_extension_allowlist(self):
        self.assertIn("错误", self.writer("x.exe", "x"))

    def test_custom_allowlist(self):
        writer = WriteFile(self.tmp, allow=("md",))
        self.assertIn("已写入", writer("a.md", "x"))
        self.assertIn("错误", writer("a.json", "x"))

    def test_max_chars(self):
        writer = WriteFile(self.tmp, max_chars=10)
        self.assertIn("错误", writer("a.md", "x" * 50))
        self.assertIn("已写入", writer("a.md", "x" * 5))

    def test_sandbox_off_allows_absolute(self):
        writer = WriteFile(self.tmp, sandbox=False)
        self.assertIn("已写入", writer(self.path("abs.md"), "x"))

    # ---------------- JSON ----------------
    def test_valid_json_is_accepted(self):
        self.writer("a.json", '{"a": 1}')
        self.assertEqual(json.loads(self.read("a.json")), {"a": 1})

    def test_invalid_json_is_rejected(self):
        message = self.writer("a.json", "{坏掉的")
        self.assertIn("错误", message)
        self.assertIn("JSON", message)
        self.assertFalse(os.path.exists(self.path("a.json")))

    def test_json_validation_can_be_disabled(self):
        writer = WriteFile(self.tmp, validate_json=False)
        self.assertIn("已写入", writer("a.json", "{坏掉的"))

    # ---------------- 清单 ----------------
    def test_written_list(self):
        self.writer("a.md", "x")
        self.writer("b/c.json", "{}")
        self.assertEqual(self.writer.written, ["a.md", "b/c.json"])

    def test_written_deduplicates(self):
        self.writer("a.md", "1")
        self.writer("a.md", "2")
        self.assertEqual(self.writer.written, ["a.md"])

    def test_paths_helper(self):
        self.writer("a.md", "x")
        self.assertEqual(self.writer.paths(), ["a.md"])
        self.assertTrue(os.path.isabs(self.writer.paths(absolute=True)[0]))

    def test_index_file(self):
        self.writer("a.md", "x")
        with open(self.path("_index.json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["files"][0]["path"], "a.md")

    def test_index_can_be_disabled(self):
        writer = WriteFile(self.tmp, index_file=None)
        writer("a.md", "x")
        self.assertFalse(os.path.exists(self.path("_index.json")))

    def test_failures_are_strings_not_exceptions(self):
        '''工具必须返回字符串，让模型看到错误后自己改，而不是把工作流打挂。'''
        for name in ("../x.md", "x.exe", "", "a.json"):
            self.assertIsInstance(self.writer(name, "{坏"), str, name)

    # ---------------- 和 workflow 串起来 ----------------
    def test_tools_writer_passes_it_to_the_agent(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            Workflow(Tools(self.writer), ChatLoop(exit_words=["exit"])).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.tools_seen, [["write_file"]])

    def test_chatworkflow_can_use_it(self):
        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", tools=[self.writer], exit_words=["exit"])
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(agent.tools_seen, [["write_file"]])

    def test_shared_instance_exists(self):
        self.assertIsInstance(WRITE_FILE, WriteFile)
        self.assertTrue(DEFAULT_ALLOWED_EXTENSIONS)


class TestHistoryRoundTrip(PromptkitTestCase):
    '''`Agent.rebuild` / `load_history` / `normalize_history`。

    这是「退出保留上下文」的底层：`export_history` 有了，缺的是反向的 load。
    '''

    def test_rebuild_keeps_history(self):
        '''`rebuild` 只重建 messages，不清历史 —— 和 `reset` 的区别就在这。'''
        agent = Agent(api_key="k")
        agent.history = [{"role": "user", "content": "在"}]
        agent.start_prompts = [{"role": "system", "content": "新设定"}]
        agent.rebuild()
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.messages[0]["content"], "新设定")
        self.assertEqual(agent.messages[1]["content"], "在")

    def test_reset_clears_history(self):
        agent = Agent(api_key="k")
        agent.history = [{"role": "user", "content": "在"}]
        agent.reset()
        self.assertEqual(agent.history, [])
        self.assertEqual(agent.messages, agent.start_prompts)

    def test_load_history_from_list(self):
        agent = Agent(api_key="k")
        count = agent.load_history([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        self.assertEqual(count, 2)
        self.assertEqual([m["content"] for m in agent.history], ["a", "b"])

    def test_load_history_drops_system(self):
        '''`system` 属于 `start_prompts`，混进历史会被当成「上一轮我说过」。'''
        agent = Agent(api_key="k")
        count = agent.load_history([
            {"role": "system", "content": "不该进历史"},
            {"role": "user", "content": "a"},
        ])
        self.assertEqual(count, 1)
        self.assertEqual(agent.history[0]["role"], "user")

    def test_load_history_drops_junk(self):
        agent = Agent(api_key="k")
        count = agent.load_history([{"role": "nope"}, "字符串", None, {"role": "user"}])
        self.assertEqual(count, 1)

    def test_load_history_keeps_tool_fields(self):
        agent = Agent(api_key="k")
        agent.load_history([
            {"role": "tool", "content": "结果", "tool_call_id": "c1", "name": "search"},
        ])
        self.assertEqual(agent.history[0]["tool_call_id"], "c1")
        self.assertEqual(agent.history[0]["name"], "search")

    def test_load_history_rebuilds_messages(self):
        agent = Agent(api_key="k")
        agent.start_prompts = [{"role": "system", "content": "设定"}]
        agent.load_history([{"role": "user", "content": "a"}])
        self.assertEqual([m["role"] for m in agent.messages], ["system", "user"])

    def test_load_history_append_mode(self):
        agent = Agent(api_key="k")
        agent.history = [{"role": "user", "content": "旧"}]
        agent.load_history([{"role": "user", "content": "新"}], replace=False)
        self.assertEqual([m["content"] for m in agent.history], ["旧", "新"])

    def test_load_history_from_json_file(self):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "h.json")
        agent = Agent(api_key="k")
        agent.history = [{"role": "user", "content": "a"}]
        agent.export_history(path)
        other = Agent(api_key="k")
        self.assertEqual(other.load_history(path), 1)
        self.assertEqual(other.history[0]["content"], "a")

    def test_load_history_from_jsonl_file(self):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "h.jsonl")
        agent = Agent(api_key="k")
        agent.history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        agent.export_history(path, fmt="jsonl")
        other = Agent(api_key="k")
        self.assertEqual(other.load_history(path), 2)

    def test_load_history_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Agent(api_key="k").load_history(os.path.join(tempfile.mkdtemp(), "nope.json"))

    def test_normalize_history_accepts_dict(self):
        cleaned = normalize_history({"messages": [{"role": "user", "content": "a"}]})
        self.assertEqual(len(cleaned), 1)

    def test_normalize_history_accepts_history_key(self):
        cleaned = normalize_history({"history": [{"role": "user", "content": "a"}]})
        self.assertEqual(len(cleaned), 1)

    def test_normalize_history_rejects_junk(self):
        self.assertEqual(normalize_history(None), [])
        self.assertEqual(normalize_history(42), [])


class TestSession(PromptkitTestCase):
    '''`Session`：把上下文存到磁盘，下次接着聊。'''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "sessions", "tutor.json")
        self.session = Session(self.path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def filled(self, turns=2):
        agent = Agent(api_key="k")
        agent.start_prompts = [{"role": "system", "content": "你是导师"}]
        agent.rebuild()
        for index in range(turns):
            agent.history.append({"role": "user", "content": f"问{index}"})
            agent.history.append({"role": "assistant", "content": f"答{index}"})
        agent.rebuild()
        return agent

    # ---------------- 基本 ----------------
    def test_does_not_exist_at_first(self):
        self.assertFalse(self.session.exists())
        self.assertEqual(self.session.turns(), 0)

    def test_save_creates_parent_dirs(self):
        self.session.save(self.filled())
        self.assertTrue(os.path.exists(self.path))

    def test_save_and_load_round_trip(self):
        self.session.save(self.filled(2))
        other = Agent(api_key="k")
        restored = self.session.load(other)
        self.assertEqual(restored, 4)
        self.assertEqual(len(other.history), 4)
        self.assertEqual(other.history[0]["content"], "问0")

    def test_saved_payload(self):
        self.session.save(self.filled(2), role="导师")
        payload = self.session.read()
        self.assertEqual(payload["version"], SESSION_VERSION)
        self.assertEqual(payload["turns"], 2)
        self.assertEqual(payload["role"], "导师")
        self.assertEqual(payload["system_prompt"], "你是导师")
        self.assertIn("saved_at", payload)

    def test_load_missing_file_is_a_no_op(self):
        '''没找到会话时不该顺手把 agent 里已有的历史清掉。'''
        agent = self.filled(1)
        self.assertEqual(self.session.load(agent), 0)
        self.assertEqual(len(agent.history), 2)      # 原样保留

    def test_load_missing_file_keeps_history_when_not_replacing(self):
        agent = self.filled(1)
        self.session.load(agent, replace=False)
        self.assertEqual(len(agent.history), 2)

    def test_load_appends(self):
        self.session.save(self.filled(1))
        agent = self.filled(1)
        self.session.load(agent, replace=False)
        self.assertEqual(len(agent.history), 4)

    def test_clear(self):
        self.session.save(self.filled())
        self.assertTrue(self.session.exists())
        self.session.clear()
        self.assertFalse(self.session.exists())

    def test_turns(self):
        self.session.save(self.filled(3))
        self.assertEqual(self.session.turns(), 3)

    # ---------------- 兼容 export_history ----------------
    def test_reads_a_bare_array(self):
        '''`export_history` 写的是裸数组，Session 也要能读。'''
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump([{"role": "user", "content": "a"}], handle)
        agent = Agent(api_key="k")
        self.assertEqual(self.session.load(agent), 1)

    def test_no_bom_no_crlf(self):
        self.session.save(self.filled())
        with open(self.path, "rb") as handle:
            raw = handle.read()
        self.assertNotIn(b"\r", raw)

    # ---------------- 上下文管理器 ----------------
    def test_open_saves_on_exit(self):
        agent = self.filled(1)
        with self.session.open(agent):
            pass
        self.assertTrue(self.session.exists())

    def test_open_saves_even_on_exception(self):
        '''聊到一半 Ctrl+C 不该把整段对话丢掉。'''
        agent = self.filled(1)
        with self.assertRaises(RuntimeError):
            with self.session.open(agent):
                raise RuntimeError("中途炸了")
        self.assertTrue(self.session.exists())
        self.assertEqual(self.session.read()["turns"], 1)

    def test_open_loads_first(self):
        self.session.save(self.filled(1))
        agent = Agent(api_key="k")
        with self.session.open(agent):
            self.assertEqual(len(agent.history), 2)

    def test_as_session_accepts_string_and_instance(self):
        self.assertIsInstance(as_session("a.json"), Session)
        self.assertIs(as_session(self.session), self.session)
        self.assertIsNone(as_session(None))
        with self.assertRaises(TypeError):
            as_session(42)

    def test_repr(self):
        self.assertIn("Session", repr(self.session))


class TestChatWorkflowSession(PromptkitTestCase):
    '''`ChatWorkflow(session=...)`：开跑前 load、结束时 save。'''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "tutor.json")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def chat(self, **kwargs):
        return ChatWorkflow(role="导师", exit_words=["exit"], session=self.path, **kwargs)

    def run_chat(self, inputs, agent=None):
        agent = agent or FakeChatAgent()
        with mock.patch("builtins.input", side_effect=list(inputs)):
            self.chat().run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        return agent

    def test_creates_the_session_file(self):
        self.run_chat(["你好", "exit"])
        self.assertTrue(os.path.exists(self.path))

    def test_second_run_continues(self):
        '''核心诉求：第二次运行要接着第一次的上下文。'''
        first = self.run_chat(["第一个问题", "exit"])
        self.assertEqual(len(first.history), 2)

        second = self.run_chat(["第二个问题", "exit"])
        self.assertEqual(len(second.history), 4)
        self.assertEqual(second.history[0]["content"], "第一个问题")
        self.assertEqual(second.history[2]["content"], "第二个问题")

    def test_third_run_keeps_accumulating(self):
        self.run_chat(["一", "exit"])
        self.run_chat(["二", "exit"])
        third = self.run_chat(["三", "exit"])
        self.assertEqual(len(third.history), 6)

    def test_restored_history_is_sent_to_the_model(self):
        '''不是只留在文件里 —— 要真的出现在发给模型的消息列表里。'''
        self.run_chat(["上一轮", "exit"])
        second = self.run_chat(["这一轮", "exit"])
        contents = [m.get("content") for m in second.messages]
        self.assertIn("上一轮", contents)
        self.assertIn("这一轮", contents)

    def test_system_prompt_is_fresh_and_not_duplicated(self):
        self.run_chat(["一", "exit"])
        second = self.run_chat(["二", "exit"])
        systems = [m for m in second.messages if m["role"] == "system"]
        self.assertEqual(len(systems), 1)
        self.assertIn("你是导师", systems[0]["content"])

    def test_reset_true_starts_fresh(self):
        self.run_chat(["一", "exit"])
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["二", "exit"]):
            self.chat().run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO(), reset=True
            )
        self.assertEqual(len(agent.history), 2)     # 只剩这一轮的
        self.assertEqual(agent.history[0]["content"], "二")

    def test_saves_even_when_the_run_raises(self):
        '''真正的异常（不是 Ctrl+C —— 那个现在算正常退出）也要把会话存下来。'''
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["hi", RuntimeError("炸了")]):
            with self.assertRaises(RuntimeError):
                self.chat().run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertTrue(os.path.exists(self.path))

    def test_session_can_be_passed_to_run(self):
        path = os.path.join(self.tmp, "other.json")
        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", exit_words=["exit"])
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            chat.run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO(), session=path
            )
        self.assertTrue(os.path.exists(path))

    def test_no_session_means_no_files(self):
        agent = FakeChatAgent()
        chat = ChatWorkflow(role="导师", exit_words=["exit"])
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            chat.run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.tmp), [])

    def test_metadata_records_the_role(self):
        self.run_chat(["hi", "exit"])
        self.assertEqual(Session(self.path).read().get("role"), "导师")


class TestHandWrittenSession(PromptkitTestCase):
    '''手写 `Workflow` 也能指定 session：`wf.run(session=...)`。

    以前只有 `ChatWorkflow.run` 有这能力，手写 workflow 想保留上下文
    得自己在 `run()` 外面包一圈 try/finally。
    '''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "s.tutor.json")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def build(self):
        return Workflow(
            Role("导师"),
            Want("找选题"),
            ExitWords("exit"),
            SystemPrompt(),
            ChatLoop(),
        )

    def run_wf(self, inputs, agent=None, **kwargs):
        agent = agent or FakeChatAgent()
        with mock.patch("builtins.input", side_effect=list(inputs)):
            self.build().run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO(), **kwargs
            )
        return agent

    def test_creates_the_session_file(self):
        self.run_wf(["hi", "exit"], session=self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_continues_across_runs(self):
        first = self.run_wf(["第一次", "exit"], session=self.path)
        self.assertEqual(len(first.history), 2)

        second = self.run_wf(["第二次", "exit"], session=self.path)
        self.assertEqual(len(second.history), 4)
        self.assertIn("第一次", second.history[0]["content"])

    def test_system_message_is_not_duplicated(self):
        self.run_wf(["一", "exit"], session=self.path)
        second = self.run_wf(["二", "exit"], session=self.path)
        systems = [m for m in second.messages if m["role"] == "system"]
        self.assertEqual(len(systems), 1)

    def test_reset_true_starts_fresh(self):
        self.run_wf(["一", "exit"], session=self.path)
        agent = FakeChatAgent()
        self.run_wf(["二", "exit"], agent=agent, session=self.path, reset=True)
        self.assertEqual(len(agent.history), 2)

    def test_saves_even_when_the_run_raises(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["hi", RuntimeError("炸了")]):
            with self.assertRaises(RuntimeError):
                self.build().run(
                    agent=agent,
                    verbose=Verbosity.SILENT,
                    out=io.StringIO(),
                    session=self.path,
                )
        self.assertTrue(os.path.exists(self.path))

    def test_session_object_also_works(self):
        self.run_wf(["hi", "exit"], session=Session(self.path))
        self.assertTrue(os.path.exists(self.path))

    def test_no_session_leaves_no_file(self):
        self.run_wf(["hi", "exit"])
        self.assertEqual(os.listdir(self.tmp), [])

    def test_needs_an_agent(self):
        with self.assertRaises(ValueError) as cm:
            self.build().run(session=self.path)
        self.assertIn("agent", str(cm.exception))

    def test_chatworkflow_uses_the_same_code_path(self):
        '''两条路（手写 / ChatWorkflow）不能各自实现一套。'''
        chat = ChatWorkflow(role="导师", exit_words=["exit"], session=self.path)
        with mock.patch("builtins.input", side_effect=["hi", "exit"]):
            chat.run(agent=FakeChatAgent(), verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertTrue(os.path.exists(self.path))
        # ChatWorkflow 会带上 role 元数据；手写的那条没有，说明是同一个 Session 类
        self.assertEqual(Session(self.path).read().get("role"), "导师")


class TestDeclarativeListForms(PromptkitTestCase):
    '''声明式 Step 同时接受「一个列表」和「一串参数」。

    这是最容易写错的：`Deliverables([Deliverable(...)])` 传的是一个列表，
    不拆开的话 items 会变成 `[[Deliverable(...)]]` —— 交付物**永远不保存**
    而且一声不响。
    '''

    def test_deliverables_list_form(self):
        item = Deliverable("选题建议", save_to="x.md")
        ctx = Workflow(Deliverables([item])).run()
        self.assertEqual(ctx["chat_deliverables"], [item])

    def test_deliverables_varargs_form(self):
        a, b = Deliverable("a"), Deliverable("b")
        ctx = Workflow(Deliverables(a, b)).run()
        self.assertEqual(ctx["chat_deliverables"], [a, b])

    def test_rules_list_form(self):
        self.assertEqual(Workflow(Rules(["一", "二"])).run()["chat_rules"], ["一", "二"])

    def test_tools_list_form(self):
        def search(q: str) -> str:
            """搜。"""
            return q

        self.assertEqual(Workflow(Tools([search])).run()["chat_tools"], [search])

    def test_exit_words_list_form(self):
        self.assertEqual(
            Workflow(ExitWords(["拜拜", "再见"])).run()["chat_exit_words"],
            ["拜拜", "再见"],
        )

    def test_flatten_single_helper(self):
        self.assertEqual(flatten_single((["a", "b"],)), ("a", "b"))
        self.assertEqual(flatten_single(("a", "b")), ("a", "b"))
        self.assertEqual(flatten_single(([1, 2],)), (1, 2))
        self.assertEqual(flatten_single(("a",)), ("a",))


class TestExitWordsAreHonored(PromptkitTestCase):
    '''回归：`ChatLoop` 曾经**忽略** `ExitWords`。

    判断用的是焊死的默认列表，`ExitWords("拜拜")` 写进 Context 也被无视 ——
    文档却说「会被自动读出来」。现在判断引用 `Var("chat_exit_words")`。
    '''

    def test_exit_words_from_context(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["拜拜", "exit"]):
            Workflow(ExitWords("拜拜"), ChatLoop()).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.prompts, [])       # 一进来就该退出

    def test_explicit_argument_still_wins(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["拜拜"]):
            Workflow(ExitWords("别的词"), ChatLoop(exit_words=["拜拜"])).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.prompts, [])

    def test_default_is_used_without_exit_words(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["exit"]):
            Workflow(ChatLoop()).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.prompts, [])


class TestDeliverableDirectoryTarget(PromptkitTestCase):
    '''`save_to` 是目录时自动补文件名。'''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.folder = os.path.join(self.tmp, "Desktop")
        os.makedirs(self.folder)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_directory_gets_a_filename(self):
        item = Deliverable("选题建议", format="markdown", save_to=self.folder)
        target = MakeFile(item).target()
        self.assertEqual(os.path.dirname(target), self.folder)
        self.assertEqual(os.path.basename(target), "选题建议.md")

    def test_directory_that_does_not_exist_yet(self):
        item = Deliverable("报告", format="json", save_to=os.path.join(self.tmp, "out"))
        self.assertTrue(MakeFile(item).target().endswith(os.path.join("out", "报告.json")))

    def test_file_path_is_kept(self):
        path = os.path.join(self.tmp, "a.md")
        self.assertEqual(MakeFile(Deliverable("x", save_to=path)).target(), path)

    def test_trailing_separator_means_directory(self):
        item = Deliverable("报告", format="markdown", save_to=self.tmp + os.sep)
        self.assertEqual(
            MakeFile(item).target(), os.path.join(self.tmp, "报告.md")
        )

    def test_extension_for(self):
        self.assertEqual(extension_for("markdown"), ".md")
        self.assertEqual(extension_for("json"), ".json")
        self.assertEqual(extension_for("csv"), ".csv")
        self.assertEqual(extension_for("yaml"), ".yaml")
        self.assertEqual(extension_for("html"), ".html")
        self.assertEqual(extension_for("随便"), ".txt")

    def test_it_really_writes(self):
        item = Deliverable("选题建议", format="markdown", save_to=self.folder)
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                ExitWords("exit"),
                ChatLoop(after_turn=[MakeFile(item)]),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), ["选题建议.md"])

    def test_deliverables_alone_does_not_save(self):
        '''回归：`Deliverables(...)` 只是声明，落盘要靠 `SaveDeliverables()`。

        用户写这行很容易以为它自己会存 —— 它不会。
        '''
        item = Deliverable("选题建议", format="markdown", save_to=self.folder)
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                Deliverables(item), ExitWords("exit"), ChatLoop()
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), [])


class TestSystemPromptInterpolation(PromptkitTestCase):
    '''system prompt 里的 `{{变量}}` 必须被替换。

    回归：`Rules("团队成员信息：{{basic_info}}")` 之前**原样**进了
    system prompt —— 因为 system prompt 直接塞进 `agent.start_prompts`，
    不走 `Prompt.render()`。模型看到的是 `{{basic_info}}` 这串字符，
    于是反馈「这里是空的」。
    '''

    def test_rule_placeholder_is_filled(self):
        agent = FakeChatAgent()
        Workflow(
            NewVar("basic_info", default={}),
            Set(Var("basic_info")["学生类型"], "研究生"),
            Role("导师"),
            Rules("团队成员信息：{{basic_info}}"),
            SystemPrompt(),
        ).run(agent=agent)
        content = agent.start_prompts[0]["content"]
        self.assertIn("研究生", content)
        self.assertNotIn("{{", content)

    def test_want_and_role_placeholders(self):
        agent = FakeChatAgent()
        Workflow(
            NewVar("who", default="小王"),
            Role("{{who}} 的导师"),
            Want("帮 {{who}} 找选题"),
            SystemPrompt(),
        ).run(agent=agent)
        content = agent.start_prompts[0]["content"]
        self.assertIn("小王 的导师", content)
        self.assertIn("帮 小王 找选题", content)

    def test_deliverable_description_placeholder(self):
        agent = FakeChatAgent()
        Workflow(
            NewVar("topic", default="咖啡"),
            Deliverables([
                Deliverable("选题建议", description="围绕 {{topic}} 展开")
            ]),
            SystemPrompt(),
        ).run(agent=agent)
        self.assertIn("围绕 咖啡 展开", agent.start_prompts[0]["content"])

    def test_custom_template_is_interpolated(self):
        agent = FakeChatAgent()
        Workflow(
            NewVar("name", default="小明"),
            SystemPrompt(template="你在和 {{name}} 对话"),
        ).run(agent=agent)
        self.assertEqual(agent.start_prompts[0]["content"], "你在和 小明 对话")

    def test_single_braces_are_left_alone(self):
        '''JSON 示例里的单花括号不能被当成占位符。'''
        agent = FakeChatAgent()
        Workflow(
            Rules('输出格式：{"label": "退款"}'),
            SystemPrompt(),
        ).run(agent=agent)
        self.assertIn('{"label": "退款"}', agent.start_prompts[0]["content"])

    def test_dollar_is_literal_in_the_system_prompt(self):
        '''system prompt 是散文，`$` 该当字面量（公式 / 价格）。'''
        agent = FakeChatAgent()
        Workflow(
            Rules("预算上限 $5000，公式用 $x_1$ 表示"),
            SystemPrompt(),
        ).run(agent=agent)
        content = agent.start_prompts[0]["content"]
        self.assertIn("$5000", content)
        self.assertIn("$x_1$", content)

    def test_missing_variable_is_loud(self):
        '''写错变量名要报错，而不是留个 `{{...}}` 让模型困惑。'''
        agent = FakeChatAgent()
        with self.assertRaises(KeyError) as cm:
            Workflow(Rules("信息：{{nope}}"), SystemPrompt()).run(agent=agent)
        self.assertIn("nope", str(cm.exception))

    def test_build_system_prompt_without_variables_keeps_placeholder(self):
        '''`build_system_prompt()` 不带 variables 时是纯文本模板预览。'''
        text = build_system_prompt(rules=["信息：{{basic_info}}"])
        self.assertIn("{{basic_info}}", text)

    def test_build_system_prompt_with_variables(self):
        text = build_system_prompt(
            rules=["信息：{{info}}"], variables={"info": {"a": 1}}
        )
        self.assertIn("'a': 1", text)
        self.assertNotIn("{{", text)

    def test_chatworkflow_build_system_prompt_is_raw(self):
        '''`ChatWorkflow.build_system_prompt()` 不接 Context，所以是原样模板。'''
        chat = ChatWorkflow(role="导师", rules=["信息：{{basic_info}}"])
        self.assertIn("{{basic_info}}", chat.build_system_prompt())

    def test_full_user_scenario(self):
        '''用户那个脚本：4 条 Rules + 问卷填的 basic_info 都要进 system prompt。'''
        agent = FakeChatAgent()
        with mock.patch(
            "builtins.input", side_effect=["研究生", "4", "统计", "y", "exit"]
        ):
            Workflow(
                Str("usr_input"),
                NewVar("basic_info", default={}),
                Set(Var("basic_info")["学生类型"], Read(">>> ")),
                Set(Var("basic_info")["队伍人数"], Read(">>> ")),
                Set(Var("basic_info")["专业背景"], Read(">>> ")),
                Set(Var("basic_info")["数据分析成员"], Read(">>> ")),
                Role("市场调研分析大赛的导师"),
                Want("帮参赛者找到合适的选题方向"),
                Rules("像导师一样专业、耐心"),
                Rules("我们要参加的比赛为“正大杯”市场调研分析大赛。"),
                Rules("请像一位导师一样，使用较为专业、准确的语言进行回答"),
                Rules("团队成员基本信息：{{basic_info}}"),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())

        content = agent.start_prompts[0]["content"]
        self.assertIn("正大杯", content)                 # 第 2 条没丢
        self.assertIn("较为专业、准确", content)          # 第 3 条没丢
        self.assertIn("研究生", content)                 # 占位符填上了
        self.assertIn("统计", content)
        self.assertNotIn("{{", content)


class TestReadOnEOF(PromptkitTestCase):
    '''`Read(on_eof="break")`：输入结束 / Ctrl+C 当成「跳出循环」。

    回归：用户聊了 13 轮，按 Ctrl+D 结束，结果 `Desktop/选题建议.md`
    **没有生成** —— 因为 `Read` 抛 `EOFError`，整个工作流异常退出，
    循环后面的 `SaveDeliverables()` 根本没机会跑。
    '''

    def test_default_still_raises(self):
        '''默认行为不变：`Read()` 遇到 EOF 就抛，别偷偷改了别处的语义。'''
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(EOFError):
                Workflow(Set("x", Read())).run()

    def test_default_still_raises_on_ctrl_c(self):
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                Workflow(Set("x", Read())).run()

    def test_break_mode_ends_the_loop(self):
        wf = Workflow(
            NewVar("n", default=0),
            While(Var("n") < 99).Do(
                Set("x", Read(on_eof="break")),
                Inc("n"),
            ),
            Set("done", 1),
        )
        with mock.patch("builtins.input", side_effect=["a", EOFError]):
            result = wf.run()
        self.assertEqual(result["n"], 1)         # 只跑了第一轮
        self.assertEqual(result["done"], 1)      # 循环后面的步骤照跑

    def test_break_mode_on_ctrl_c(self):
        wf = Workflow(
            NewVar("n", default=0),
            While(Var("n") < 99).Do(
                Set("x", Read(on_eof="break")),
                Inc("n"),
            ),
            Set("done", 1),
        )
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            result = wf.run()
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["done"], 1)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            Read(on_eof="nope")

    def test_repr_shows_the_mode(self):
        self.assertEqual(repr(Read()), "Read('')")
        self.assertIn("on_eof='break'", repr(Read(on_eof="break")))

    # ---------------- 端到端：交付物不该被中断吃掉 ----------------
    def test_deliverable_survives_ctrl_c(self):
        folder = tempfile.mkdtemp()
        desk = os.path.join(folder, "Desktop")
        os.makedirs(desk)
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", KeyboardInterrupt]):
            Workflow(
                Str("usr_input"),
                Deliverables([
                    Deliverable("选题建议", format="markdown", save_to=desk)
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
                SaveDeliverables(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(desk), ["选题建议.md"])

    def test_deliverable_survives_eof(self):
        folder = tempfile.mkdtemp()
        desk = os.path.join(folder, "Desktop")
        os.makedirs(desk)
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", EOFError]):
            Workflow(
                Str("usr_input"),
                Deliverables([
                    Deliverable("选题建议", format="markdown", save_to=desk)
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
                SaveDeliverables(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(desk), ["选题建议.md"])

    def test_each_turn_deliverable_is_even_safer(self):
        '''`when="each_turn"` 每轮就写，连 Ctrl+C 都不用等。'''
        folder = tempfile.mkdtemp()
        desk = os.path.join(folder, "Desktop")
        os.makedirs(desk)
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["第一轮", KeyboardInterrupt]):
            Workflow(
                Str("usr_input"),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(
                    after_turn=[
                        MakeFile(
                            Deliverable(
                                "选题建议",
                                format="markdown",
                                save_to=desk,
                                when="each_turn",
                            )
                        )
                    ]
                ),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        with open(os.path.join(desk, "选题建议.md"), encoding="utf-8") as handle:
            self.assertIn("嗯1", handle.read())


class TestSessionVariables(PromptkitTestCase):
    '''会话不只存对话历史，也能存 Context 变量。

    典型场景：聊天机器人开头有 4 个问卷（`Read` 填 `basic_info`），
    光存历史的话每次启动都要重问一遍。`variables=["basic_info"]` 就够了。
    '''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "s.json")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def build(self):
        return Workflow(
            NewVar("basic_info", default={}),
            If(Var("basic_info") == {}).Then(
                Set(Var("basic_info")["类型"], Read(">>> ")),
            ),
            Role("导师"),
            Rules("成员信息：{{basic_info}}"),
            ExitWords("exit"),
            SystemPrompt(),
            ChatLoop(),
        )

    def run_wf(self, inputs, agent=None, **kwargs):
        agent = agent or FakeChatAgent()
        options = {"session": self.path, **kwargs}      # 显式传的覆盖默认
        with mock.patch("builtins.input", side_effect=list(inputs)) as fake:
            self.build().run(
                agent=agent,
                verbose=Verbosity.SILENT,
                out=io.StringIO(),
                **options,
            )
        return agent, fake.call_count

    # ---------------- Session 层 ----------------
    def test_save_and_load_variables(self):
        session = Session(self.path)
        agent = FakeChatAgent()
        session.save(agent, variables={"basic_info": {"a": 1}})
        self.assertEqual(session.load_variables(), {"basic_info": {"a": 1}})

    def test_load_variables_without_file(self):
        self.assertEqual(Session(self.path).load_variables(), {})

    def test_non_jsonable_variable_falls_back_to_repr(self):
        class Weird:
            def __repr__(self):
                return "<怪>"

        Session(self.path).save(FakeChatAgent(), variables={"x": Weird()})
        self.assertEqual(Session(self.path).load_variables()["x"], "<怪>")

    def test_variables_are_not_saved_when_empty(self):
        Session(self.path).save(FakeChatAgent())
        self.assertNotIn("variables", Session(self.path).read())

    # ---------------- 端到端 ----------------
    def test_questionnaire_is_asked_once(self):
        '''核心诉求：第一次问 4 个问题，第二次直接进对话。'''
        # 第一次：问卷 1 问 + 1 轮对话 + exit
        agent, calls = self.run_wf(
            ["研究生", "你好", "exit"], variables=["basic_info"]
        )
        self.assertEqual(calls, 3)
        self.assertEqual(agent.history[0]["content"], "你好")
        self.assertEqual(len(agent.history), 2)

        # 第二次：问卷跳过，只剩 exit 这一个输入
        agent2, calls2 = self.run_wf(["exit"], variables=["basic_info"])
        self.assertEqual(calls2, 1)
        self.assertEqual(len(agent2.history), 2)       # 上次那轮接回来了
        self.assertEqual(agent2.history[0]["content"], "你好")

    def test_variables_reach_the_system_prompt(self):
        self.run_wf(["研究生", "exit"], variables=["basic_info"])
        agent, _ = self.run_wf(["exit"], variables=["basic_info"])
        self.assertIn("研究生", agent.start_prompts[0]["content"])

    def test_variables_land_in_the_session_file(self):
        self.run_wf(["研究生", "exit"], variables=["basic_info"])
        payload = Session(self.path).read()
        self.assertEqual(payload["variables"]["basic_info"], {"类型": "研究生"})

    def test_without_variables_the_questionnaire_repeats(self):
        '''不传 variables 就还是老行为 —— 每次都要重问。'''
        self.run_wf(["研究生", "exit"])
        _, calls = self.run_wf(["本科", "exit"])
        self.assertEqual(calls, 2)

    def test_reset_reasks(self):
        self.run_wf(["研究生", "exit"], variables=["basic_info"])
        _, calls = self.run_wf(["本科", "exit"], variables=["basic_info"], reset=True)
        self.assertEqual(calls, 2)                     # 重新问了

    def test_session_object_carries_the_variable_list(self):
        agent, calls = self.run_wf(
            ["研究生", "你好", "exit"],
            session=Session(self.path, variables=["basic_info"]),
        )
        self.assertEqual(calls, 3)
        self.assertEqual(Session(self.path).load_variables()["basic_info"], {"类型": "研究生"})


class TestEachTurnInHandWrittenWorkflow(PromptkitTestCase):
    '''回归：手写 Workflow 里 `Deliverable(when="each_turn")` 曾经**完全没效果**。

    循环里没人管它，循环后的 `SaveDeliverables()` 又按 `when="end"` 把它过滤掉 ——
    文件一个都不会出现，而且一声不响。
    '''

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.folder = os.path.join(self.tmp, "Desktop")
        os.makedirs(self.folder)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_chatloop_auto_saves_each_turn(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                Deliverables([
                    Deliverable("选题建议", format="markdown", save_to=self.folder,
                                when="each_turn")
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
                SaveDeliverables(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), ["选题建议.md"])

    def test_each_turn_updates_the_file(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["第一轮", "第二轮", "exit"]):
            Workflow(
                Deliverables([
                    Deliverable("选题建议", format="markdown", save_to=self.folder,
                                when="each_turn")
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        with open(os.path.join(self.folder, "选题建议.md"), encoding="utf-8") as handle:
            self.assertIn("嗯2", handle.read())        # 最后一轮的

    def test_end_deliverables_still_work(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                Deliverables([
                    Deliverable("报告", format="markdown", save_to=self.folder)
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
                SaveDeliverables(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), ["报告.md"])

    def test_no_double_write(self):
        '''`each_turn` 的交付物不该被循环后的 `SaveDeliverables()` 再存一次。'''
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            ctx = Workflow(
                Deliverables([
                    Deliverable("a", format="markdown", save_to=self.folder,
                                when="each_turn"),
                    Deliverable("b", format="markdown", save_to=self.folder),
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(),
                SaveDeliverables(),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(sorted(os.listdir(self.folder)), ["a.md", "b.md"])

    def test_savedeliverables_when_argument(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                Deliverables([
                    Deliverable("a", format="markdown", save_to=self.folder,
                                when="each_turn")
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(after_turn=[SaveDeliverables(when="each_turn")]),
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), ["a.md"])

    def test_explicit_after_turn_disables_the_default(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                Deliverables([
                    Deliverable("a", format="markdown", save_to=self.folder,
                                when="each_turn")
                ]),
                ExitWords("exit"),
                SystemPrompt(),
                ChatLoop(after_turn=[]),          # 显式空 = 什么都不存
            ).run(agent=agent, verbose=Verbosity.SILENT, out=io.StringIO())
        self.assertEqual(os.listdir(self.folder), [])


# ==================== 多厂商客户端（client_sys） ====================
class FakeSDK:
    '''假的 openai.OpenAI：只实现 chat.completions.create。'''

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("假 SDK 没有更多回复了")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    '''假的 urllib 响应：能迭代（SSE 行）、能 read()、能 close()。'''

    def __init__(self, lines=(), body=b"{}"):
        self.lines = list(lines)
        self.body = body
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.body

    def close(self):
        self.closed = True


def anthropic_client_with(reply):
    '''造一个不发网络的 AnthropicClient，并记录每次请求体。

    `reply` 可以是：响应 dict（非流式）、SSE 行列表（流式）、
    或者 `callable(body, stream)`。
    '''
    seen = []

    def transport(url, body, stream):
        seen.append({"url": url, "body": body, "stream": stream})
        return reply(body, stream) if callable(reply) else reply

    return AnthropicClient(api_key="sk-ant-test", transport=transport), seen


def sse(*payloads):
    '''把 dict 变成 SSE 行（data + 空行）。'''
    lines = []
    for payload in payloads:
        lines.append("data: " + json.dumps(payload, ensure_ascii=False))
        lines.append("")
    return lines


class TestVendorRegistry(PromptkitTestCase):
    def test_builtin_vendors(self):
        for name in ("deepseek", "openai", "anthropic", "openai-compatible"):
            self.assertIn(name, vendor_names())

    def test_default_vendor_is_still_deepseek(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                self.assertEqual(detect_vendor(), "deepseek")

    def test_model_prefix_decides_the_vendor(self):
        self.assertEqual(detect_vendor(model="claude-sonnet-4-5"), "anthropic")
        self.assertEqual(detect_vendor(model="deepseek-chat"), "deepseek")
        self.assertEqual(detect_vendor(model="gpt-4o"), "openai")
        self.assertEqual(detect_vendor(model="o3-mini"), "openai")
        # OpenRouter 那种带厂商前缀的写法
        self.assertEqual(detect_vendor(model="anthropic/claude-3-5-haiku"), "anthropic")
        self.assertEqual(detect_vendor(model="deepseek/deepseek-chat"), "deepseek")

    def test_base_url_wins_over_model_name(self):
        self.assertEqual(
            detect_vendor(model="gpt-4o", base_url="https://api.anthropic.com"),
            "anthropic",
        )
        self.assertEqual(detect_vendor(base_url="https://api.openai.com/v1"), "openai")
        self.assertEqual(detect_vendor(base_url="https://api.deepseek.com"), "deepseek")

    def test_unknown_base_url_means_openai_compatible(self):
        self.assertEqual(
            detect_vendor(base_url="http://localhost:11434/v1"), "openai-compatible"
        )

    def test_env_keys_are_sniffed_only_when_nothing_is_said(self):
        with mock.patch("promptkit.config.load_env_file"):
            with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=True):
                self.assertEqual(detect_vendor(), "anthropic")
            with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=True):
                self.assertEqual(detect_vendor(), "openai")
            # DeepSeek 排第一：两个都设了也还是它
            with mock.patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "k", "DEEPSEEK_API_KEY": "k"},
                clear=True,
            ):
                self.assertEqual(detect_vendor(), "deepseek")

    def test_env_file_can_decide_the_vendor(self):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".env", delete=False, encoding="utf-8"
        )
        try:
            handle.write("ANTHROPIC_API_KEY=sk-ant-file\n")
            handle.close()
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch(
                    "promptkit.config.load_env_file",
                    lambda: load_env_file(handle.name),
                ):
                    self.assertEqual(detect_vendor(), "anthropic")
        finally:
            os.unlink(handle.name)

    def test_explicit_key_does_not_sniff_env(self):
        # 显式给了密钥就说明用户心里有默认厂商了，别再跟着环境变量漂
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=True):
            self.assertEqual(detect_vendor(api_key="sk-x"), "deepseek")

    def test_alias_and_case_are_accepted(self):
        for name in ("claude", "ANTHROPIC", "Anthropic"):
            self.assertEqual(get_spec(name).name, "anthropic")
        self.assertEqual(get_spec("openai_compatible").name, "openai-compatible")
        self.assertEqual(get_spec("ds").name, "deepseek")

    def test_unknown_vendor_lists_the_choices(self):
        with self.assertRaises(ValueError) as cm:
            get_spec("mistral")
        message = str(cm.exception)
        self.assertIn("deepseek", message)
        self.assertIn("anthropic", message)

    def test_resolve_target_fills_vendor_defaults(self):
        spec, model, base = resolve_target(vendor="anthropic")
        self.assertEqual(spec.name, "anthropic")
        self.assertEqual(model, "claude-sonnet-4-5")
        self.assertEqual(base, "https://api.anthropic.com")

    def test_openai_compatible_needs_model_and_base_url(self):
        with self.assertRaises(ValueError) as cm:
            resolve_target(vendor="openai-compatible", base_url="http://x/v1")
        self.assertIn("model", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            resolve_target(vendor="openai-compatible", model="qwen2.5")
        self.assertIn("base_url", str(cm.exception))

    def test_key_order_is_argument_then_env(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "from-env"}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                self.assertEqual(resolve_api_key("anthropic", "from-arg"), "from-arg")
                self.assertEqual(resolve_api_key("anthropic"), "from-env")

    def test_universal_key_fallback(self):
        with mock.patch.dict("os.environ", {"PROMPTKIT_API_KEY": "sk-any"}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                self.assertEqual(resolve_api_key("deepseek"), "sk-any")

    def test_missing_key_names_the_vendor_variable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                with self.assertRaises(ValueError) as cm:
                    resolve_api_key("anthropic")
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_missing_key_hints_local_endpoints(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                with self.assertRaises(ValueError) as cm:
                    resolve_api_key("openai-compatible")
        self.assertIn("ollama", str(cm.exception))

    def test_key_can_come_from_the_env_file(self):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".env", delete=False, encoding="utf-8"
        )
        try:
            handle.write("ANTHROPIC_API_KEY=sk-from-file\n")
            handle.close()
            with mock.patch.dict("os.environ", {}, clear=True):
                # 把「读哪个 .env」换成临时文件；读法还是真的那个函数
                with mock.patch(
                    "promptkit.config.load_env_file",
                    lambda: load_env_file(handle.name),
                ):
                    self.assertEqual(resolve_api_key("anthropic"), "sk-from-file")
        finally:
            os.unlink(handle.name)

    def test_register_vendor_adds_a_provider(self):
        from promptkit.client_sys import registry

        spec = ClientSpec(
            name="test-gateway",
            label="测试网关",
            kind="openai",
            env_vars=("TEST_GATEWAY_KEY",),
            default_model="gateway-mini",
            default_base_url="https://gateway.test/v1",
        )
        try:
            register_vendor(spec)
            self.assertEqual(get_spec("test-gateway").name, "test-gateway")
            with mock.patch.dict(
                "os.environ", {"TEST_GATEWAY_KEY": "sk-gw"}, clear=True
            ):
                with mock.patch("promptkit.config.load_env_file"):
                    client = make_client("test-gateway")
            self.assertIsInstance(client, OpenAIClient)
            self.assertEqual(client.base_url, "https://gateway.test/v1")
            self.assertEqual(resolve_target(vendor="test-gateway")[1], "gateway-mini")
        finally:
            registry.PROVIDERS.pop("test-gateway", None)
            registry._ALIASES.pop("test-gateway", None)

    def test_register_vendor_rejects_an_unknown_kind(self):
        with self.assertRaises(ValueError) as cm:
            register_vendor(ClientSpec(name="weird", label="W", kind="grpc"))
        self.assertIn("factory", str(cm.exception))


class TestOpenAIClient(PromptkitTestCase):
    def test_create_sends_chat_completions_fields(self):
        sdk = FakeSDK(["reply"])
        client = OpenAIClient(api_key="k", base_url="https://x/v1", sdk=sdk)
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(client.create(model="m", messages=messages), "reply")
        self.assertEqual(
            sdk.calls[0], {"model": "m", "messages": messages, "stream": False}
        )

    def test_tools_and_stream_are_forwarded(self):
        sdk = FakeSDK([iter([])])
        client = OpenAIClient(api_key="k", base_url="https://x/v1", sdk=sdk)
        tools = [{"type": "function", "function": {"name": "f"}}]
        client.create(model="m", messages=[], tools=tools, stream=True)
        self.assertEqual(sdk.calls[0]["tools"], tools)
        self.assertTrue(sdk.calls[0]["stream"])

    def test_params_are_passed_through_untouched(self):
        sdk = FakeSDK(["ok"])
        client = OpenAIClient(api_key="k", base_url="https://x/v1", sdk=sdk)
        client.create(
            model="m",
            messages=[],
            temperature=0.2,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}},
        )
        call = sdk.calls[0]
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["reasoning_effort"], "low")
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})

    def test_chat_completions_facade(self):
        client = OpenAIClient(api_key="k", base_url="https://x/v1", sdk=FakeSDK())
        self.assertIs(client.chat, client)
        self.assertIs(client.completions, client)

    def test_close_closes_the_sdk(self):
        sdk = FakeSDK()
        with OpenAIClient(api_key="k", base_url="https://x/v1", sdk=sdk) as client:
            self.assertIsInstance(client, BaseClient)
        self.assertTrue(sdk.closed)

    def test_make_client_builds_a_deepseek_client(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-ds"}, clear=True):
            client = make_client("deepseek")
        try:
            self.assertIsInstance(client, OpenAIClient)
            self.assertEqual(client.base_url, "https://api.deepseek.com")
            self.assertEqual(client.api_key, "sk-ds")
            self.assertEqual(client.spec.default_params["reasoning_effort"], "high")
        finally:
            client.close()

    def test_make_client_needs_a_base_url_for_compatible(self):
        with self.assertRaises(ClientError) as cm:
            make_client("openai-compatible", api_key="k")
        self.assertIn("base_url", str(cm.exception))


class TestAnthropicTranslation(PromptkitTestCase):
    def test_system_is_hoisted_and_messages_are_blockified(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "规则"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "在"},
            ],
        )
        body = seen[0]["body"]
        self.assertEqual(body["system"], "规则")
        self.assertEqual(
            body["messages"],
            [
                {"role": "user", "content": [{"type": "text", "text": "你好"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "在"}]},
            ],
        )

    def test_multiple_system_messages_are_joined(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "system", "content": "一"},
                {"role": "system", "content": "二"},
                {"role": "user", "content": "x"},
            ],
        )
        self.assertEqual(seen[0]["body"]["system"], "一\n\n二")

    def test_tool_calls_become_tool_use_blocks(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "user", "content": "天气"},
                {
                    "role": "assistant",
                    "content": "我查一下",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "北京"}',
                            },
                        }
                    ],
                },
            ],
        )
        blocks = seen[0]["body"]["messages"][1]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "我查一下"})
        self.assertEqual(
            blocks[1],
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": {"city": "北京"},
            },
        )

    def test_tool_results_become_user_blocks(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "user", "content": "天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "晴"},
            ],
        )
        self.assertEqual(
            seen[0]["body"]["messages"][-1],
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "晴"}
                ],
            },
        )

    def test_consecutive_same_role_turns_merge(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "user", "content": "a"},
                {"role": "tool", "tool_call_id": "c1", "content": "1"},
                {"role": "tool", "tool_call_id": "c2", "content": "2"},
            ],
        )
        messages = seen[0]["body"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]["content"]), 3)

    def test_broken_arguments_are_kept_raw(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "{oops"}}
                    ],
                },
            ],
        )
        blocks = seen[0]["body"]["messages"][-1]["content"]
        self.assertEqual(blocks[0]["input"], {"_raw": "{oops"})

    def test_empty_assistant_message_is_dropped(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "y"},
            ],
        )
        messages = seen[0]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertEqual(len(messages[0]["content"]), 2)

    def test_tools_become_input_schema(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[{"role": "user", "content": "x"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查天气。",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        self.assertEqual(
            seen[0]["body"]["tools"],
            [
                {
                    "name": "get_weather",
                    "description": "查天气。",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )

    def test_max_tokens_defaults_to_4096_and_can_be_overridden(self):
        client, seen = anthropic_client_with({})
        client.create(model="claude", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(seen[0]["body"]["max_tokens"], 4096)
        client.create(
            model="claude", messages=[{"role": "user", "content": "x"}], max_tokens=32
        )
        self.assertEqual(seen[1]["body"]["max_tokens"], 32)

    def test_stop_maps_to_stop_sequences(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[{"role": "user", "content": "x"}],
            stop=["END"],
            temperature=0.3,
            top_p=0.9,
        )
        body = seen[0]["body"]
        self.assertEqual(body["stop_sequences"], ["END"])
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["top_p"], 0.9)

    def test_tool_choice_is_translated(self):
        client, seen = anthropic_client_with({})
        client.create(
            model="claude",
            messages=[{"role": "user", "content": "x"}],
            tool_choice="required",
        )
        self.assertEqual(seen[0]["body"]["tool_choice"], {"type": "any"})

        client.create(
            model="claude",
            messages=[{"role": "user", "content": "x"}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        self.assertEqual(seen[1]["body"]["tool_choice"], {"type": "tool", "name": "f"})

    def test_unknown_params_are_rejected_with_a_list(self):
        client, _ = anthropic_client_with({})
        with self.assertRaises(ClientError) as cm:
            client.create(
                model="claude",
                messages=[{"role": "user", "content": "x"}],
                response_format={"type": "json_object"},
            )
        self.assertIn("response_format", str(cm.exception))

    def test_thinking_is_rejected_with_an_explanation(self):
        client, _ = anthropic_client_with({})
        with self.assertRaises(ClientError) as cm:
            client.create(
                model="claude",
                messages=[{"role": "user", "content": "x"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
        self.assertIn("extended thinking", str(cm.exception))

    def test_model_is_required(self):
        client, _ = anthropic_client_with({})
        with self.assertRaises(ClientError) as cm:
            client.create(model="", messages=[])
        self.assertIn("model", str(cm.exception))

    def test_endpoint_handles_the_v1_suffix(self):
        self.assertEqual(
            AnthropicClient(api_key="k").endpoint,
            "https://api.anthropic.com/v1/messages",
        )
        self.assertEqual(
            AnthropicClient(api_key="k", base_url="https://proxy/v1").endpoint,
            "https://proxy/v1/messages",
        )
        self.assertEqual(
            AnthropicClient(api_key="k", base_url="https://proxy/").endpoint,
            "https://proxy/v1/messages",
        )


class TestAnthropicResponse(PromptkitTestCase):
    def test_text_thinking_and_usage(self):
        client, _ = anthropic_client_with(
            {
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "想一下"},
                    {"type": "text", "text": "你好"},
                ],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            }
        )
        reply = client.create(model="claude", messages=[{"role": "user", "content": "x"}])
        self.assertIsInstance(reply, Completion)
        self.assertEqual(reply.text, "你好")
        self.assertEqual(reply.message.reasoning_content, "想一下")
        self.assertEqual(reply.usage.prompt_tokens, 9)
        self.assertEqual(reply.usage.completion_tokens, 4)
        self.assertEqual(reply.usage.total_tokens, 13)
        self.assertEqual(reply.choices[0].finish_reason, "stop")
        self.assertEqual(reply.model, "claude-sonnet-4-5")

    def test_tool_use_becomes_tool_calls(self):
        client, _ = anthropic_client_with(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "北京"},
                    }
                ],
                "stop_reason": "tool_use",
            }
        )
        reply = client.create(model="claude", messages=[{"role": "user", "content": "x"}])
        call = reply.message.tool_calls[0]
        self.assertEqual(call.id, "toolu_1")
        self.assertEqual(call.function.name, "get_weather")
        self.assertEqual(json.loads(call.function.arguments), {"city": "北京"})
        self.assertEqual(reply.choices[0].finish_reason, "tool_calls")
        # Agent 记历史时用的就是这个形状
        self.assertEqual(
            call.model_dump(),
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
            },
        )

    def test_stop_reason_mapping(self):
        cases = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        for anthropic_reason, openai_reason in cases.items():
            client, _ = anthropic_client_with(
                {"content": [{"type": "text", "text": "x"}], "stop_reason": anthropic_reason}
            )
            reply = client.create(model="claude", messages=[])
            self.assertEqual(reply.choices[0].finish_reason, openai_reason)

    def test_empty_usage_is_not_invented(self):
        client, _ = anthropic_client_with({"content": [{"type": "text", "text": "x"}]})
        reply = client.create(model="claude", messages=[])
        self.assertIsNone(reply.usage)


class TestAnthropicStreaming(PromptkitTestCase):
    def test_text_deltas(self):
        client, _ = anthropic_client_with(
            sse(
                {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "你"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "好"},
                },
                {"type": "message_stop"},
            )
        )
        chunks = list(
            client.create(
                model="claude", messages=[{"role": "user", "content": "x"}], stream=True
            )
        )
        pieces = [c.delta.content for c in chunks if c.delta is not None]
        self.assertEqual(pieces, ["你", "好"])
        self.assertEqual(chunks[0].usage.prompt_tokens, 3)

    def test_tool_use_deltas(self):
        client, _ = anthropic_client_with(
            sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city"'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ': "北京"}'},
                },
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            )
        )
        chunks = list(
            client.create(model="claude", messages=[], stream=True)
        )
        calls = [chunk.delta.tool_calls[0] for chunk in chunks if chunk.delta and chunk.delta.tool_calls]
        self.assertEqual(calls[0].id, "toolu_1")
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(
            "".join(call.function.arguments for call in calls), '{"city": "北京"}'
        )
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_thinking_deltas(self):
        client, _ = anthropic_client_with(
            sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "嗯"},
                }
            )
        )
        chunks = list(client.create(model="claude", messages=[], stream=True))
        self.assertEqual(chunks[0].delta.reasoning_content, "嗯")

    def test_error_event_raises(self):
        client, _ = anthropic_client_with(
            sse(
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                }
            )
        )
        with self.assertRaises(ClientError) as cm:
            list(client.create(model="claude", messages=[], stream=True))
        self.assertIn("overloaded_error", str(cm.exception))

    def test_iter_sse_multiline_data(self):
        lines = [
            ": 心跳",
            "event: content_block_delta",
            'data: {"type": "content_block_delta", "index": 0,',
            'data:  "delta": {"type": "text_delta", "text": "嗨"}}',
            "",
        ]
        events = list(iter_sse(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "content_block_delta")
        self.assertEqual(events[0][1]["delta"]["text"], "嗨")

    def test_iter_sse_skips_junk(self):
        self.assertEqual(list(iter_sse([": ping", "", "data: not json", ""])), [])

    def test_http_transport_posts_the_right_request(self):
        fake = FakeHTTPResponse(body=b'{"content": [{"type": "text", "text": "ok"}]}')
        client = AnthropicClient(api_key="sk-ant", timeout=5)
        with mock.patch("urllib.request.urlopen", return_value=fake) as opener:
            reply = client.create(model="claude", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(reply.text, "ok")
        request = opener.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.get_header("X-api-key"), "sk-ant")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(json.loads(request.data)["model"], "claude")
        self.assertTrue(fake.closed)

    def test_streaming_transport_closes_the_response(self):
        lines = [
            b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}\n',
            b"\n",
        ]
        fake = FakeHTTPResponse(lines)
        client = AnthropicClient(api_key="k")
        with mock.patch("urllib.request.urlopen", return_value=fake):
            chunks = list(client.create(model="claude", messages=[], stream=True))
        self.assertEqual([c.delta.content for c in chunks if c.delta], ["hi"])
        self.assertTrue(fake.closed)

    def test_http_errors_show_the_api_message(self):
        error = urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages",
            401,
            "Unauthorized",
            {},
            io.BytesIO(
                b'{"type": "error", "error": {"type": "authentication_error",'
                b' "message": "invalid x-api-key"}}'
            ),
        )
        client = AnthropicClient(api_key="bad")
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ClientError) as cm:
                client.create(model="claude", messages=[{"role": "user", "content": "x"}])
        self.assertIn("401", str(cm.exception))
        self.assertIn("invalid x-api-key", str(cm.exception))


class TestAgentMultiVendor(PromptkitTestCase):
    def test_plain_agent_is_still_deepseek(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-ds"}, clear=True):
            agent = Agent(register_as_default=False)
        self.assertEqual(agent.vendor, "deepseek")
        self.assertEqual(agent.model, "deepseek-flash")
        self.assertEqual(agent.base_url, "https://api.deepseek.com")
        self.assertEqual(agent.api_key, "sk-ds")
        self.assertIsInstance(agent.client, OpenAIClient)
        # DeepSeek 的思考参数跟着厂商走，不是 Agent 写死的
        self.assertEqual(agent.params["reasoning_effort"], "high")
        self.assertEqual(agent.params["extra_body"], {"thinking": {"type": "enabled"}})

    def test_vendor_switches_model_base_url_and_client(self):
        agent = Agent(vendor="anthropic", api_key="sk-ant", register_as_default=False)
        self.assertEqual(agent.vendor, "anthropic")
        self.assertEqual(agent.model, "claude-sonnet-4-5")
        self.assertEqual(agent.base_url, "https://api.anthropic.com")
        self.assertIsInstance(agent.client, AnthropicClient)
        self.assertEqual(agent.client.api_key, "sk-ant")
        self.assertEqual(agent.params, {"max_tokens": 4096})

    def test_vendor_can_be_inferred_from_the_model(self):
        agent = Agent(model="claude-sonnet-4-5", api_key="k", register_as_default=False)
        self.assertEqual((agent.vendor, agent.model), ("anthropic", "claude-sonnet-4-5"))
        agent = Agent(model="gpt-4o", api_key="k", register_as_default=False)
        self.assertEqual(agent.vendor, "openai")
        self.assertEqual(agent.base_url, "https://api.openai.com/v1")

    def test_vendor_can_be_inferred_from_the_base_url(self):
        agent = Agent(
            base_url="http://localhost:11434/v1",
            model="qwen2.5",
            api_key="ollama",
            register_as_default=False,
        )
        self.assertEqual(agent.vendor, "openai-compatible")
        self.assertIsInstance(agent.client, OpenAIClient)
        self.assertEqual(agent.client.base_url, "http://localhost:11434/v1")

    def test_params_and_max_tokens_are_merged_over_vendor_defaults(self):
        agent = Agent(
            vendor="anthropic",
            api_key="k",
            params={"temperature": 0.2},
            max_tokens=99,
            timeout=3,
            register_as_default=False,
        )
        self.assertEqual(agent.params["temperature"], 0.2)
        self.assertEqual(agent.params["max_tokens"], 99)
        self.assertEqual(agent.client.timeout, 3)

    def test_missing_key_names_the_vendor_variable(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("promptkit.config.load_env_file"):
                with self.assertRaises(ValueError) as cm:
                    Agent(vendor="anthropic")
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_ask_with_anthropic_text_reply(self):
        agent = Agent(vendor="anthropic", api_key="k", register_as_default=False)
        seen = []

        def transport(url, body, stream):
            seen.append(body)
            return {
                "content": [{"type": "text", "text": "你好呀"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }

        agent.client.transport = transport
        handler, events = collect_events()
        agent.bus.on(handler)

        self.assertEqual(agent.ask("嗨"), "你好呀")
        self.assertEqual(events[-1].type, EventType.MESSAGE_END)
        # 系统提示被提到顶层，不进 messages
        self.assertEqual(seen[0]["system"], "You are a helpful assistant.")
        self.assertEqual(agent.history, [
            {"role": "user", "content": "嗨"},
            {"role": "assistant", "content": "你好呀"},
        ])

    def test_ask_with_anthropic_tool_loop(self):
        def get_weather(city: str) -> str:
            """查天气。"""
            return f"{city} 晴"

        replies = [
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "北京"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
            {"content": [{"type": "text", "text": "北京 晴"}], "stop_reason": "end_turn"},
        ]
        seen = []

        def transport(url, body, stream):
            seen.append(body)
            return replies.pop(0)

        agent = Agent(vendor="anthropic", api_key="k", register_as_default=False)
        agent.client.transport = transport
        handler, events = collect_events()
        agent.bus.on(handler)

        self.assertEqual(agent.ask("天气？", tools=[get_weather]), "北京 晴")

        # 工具真的跑了，结果以 tool_result 块回给模型
        results = [e.data for e in events if e.type == EventType.TOOL_RESULT]
        self.assertEqual(results[0]["name"], "get_weather")
        self.assertEqual(results[0]["result"], "北京 晴")
        self.assertEqual([m["role"] for m in seen[1]["messages"]], ["user", "assistant", "user"])
        self.assertEqual(
            seen[1]["messages"][-1]["content"][0],
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "北京 晴"},
        )
        # 工具定义翻译成 input_schema
        self.assertEqual(
            seen[0]["tools"][0]["input_schema"]["properties"],
            {"city": {"type": "string"}},
        )
        # 历史里留下的是 OpenAI 形状（换厂商不用转换）
        self.assertEqual(agent.history[1]["tool_calls"][0]["id"], "toolu_1")

    def test_ask_stream_with_anthropic_sse(self):
        lines = sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "你"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "好"},
            },
            {"type": "message_stop"},
        )
        agent = Agent(vendor="anthropic", api_key="k", register_as_default=False)
        agent.client.transport = lambda url, body, stream: lines

        events = list(agent.ask_stream("hi"))
        deltas = [e.data["text"] for e in events if e.type == EventType.MESSAGE_DELTA]
        self.assertEqual(deltas, ["你", "好"])
        self.assertEqual(events[-1].type, EventType.MESSAGE_END)
        self.assertEqual(events[-1].data["text"], "你好")
        self.assertEqual(agent.history[-1]["content"], "你好")

    def test_openai_vendor_does_not_leak_deepseek_params(self):
        agent = Agent(vendor="openai", api_key="k", register_as_default=False)
        sdk = FakeSDK([fake_message(role="assistant", content="hi")])
        agent.client.sdk = sdk

        self.assertEqual(agent.ask("hello"), "hi")
        call = sdk.calls[0]
        self.assertEqual(call["model"], "gpt-4o-mini")
        self.assertNotIn("reasoning_effort", call)
        self.assertNotIn("extra_body", call)

    def test_structured_output_over_anthropic(self):
        class Intent(BaseModel):
            label: str

        agent = Agent(vendor="anthropic", api_key="k", register_as_default=False)
        agent.client.transport = lambda url, body, stream: {
            "content": [{"type": "text", "text": '{"label": "退款"}'}],
            "stop_reason": "end_turn",
        }
        value = Executor(agent=agent).run(
            Prompt(role="user", content="意图？"), Intent
        )
        self.assertEqual(value.label, "退款")


from promptkit.coding import (  # noqa: E402
    ApplyChanges,
    ApplyDiff,
    ChangesOutput,
    CodeFormatError,
    CodingStep,
    CommandError,
    DiffError,
    DiffOutput,
    FilePatch,
    FixLoop,
    FullFileOutput,
    GitCommit,
    GitDiff,
    ListDir,
    LintStep,
    ReadFileStep,
    RunCommand,
    RunTests,
    Sandbox,
    SandboxError,
    SearchCode,
    WriteFileStep,
    apply_hunks,
    apply_patch,
    apply_patch_text,
    default_coding_tools,
    format_hits,
    get_output,
    git,
    parse_patch,
    run_command as coding_run_command,
    strip_code_fence,
    validate_patch,
)
from promptkit.coding.tools import (  # noqa: E402
    ApplyDiffTool,
    ListDirTool,
    ReadFileTool,
    RunCommandTool,
    RunTestsTool,
    SearchCodeTool,
    WriteFileTool,
)

from promptkit.orchestrate import (  # noqa: E402
    AggregateStep,
    DelegateTool,
    ExecutePlan,
    LLMPlanner,
    Orchestrator,
    OrchestratorConfig,
    Plan,
    PlanError,
    PlanResult,
    PlanStep,
    Planner,
    RoleFactory,
    Task,
    TaskError,
    TaskResult,
    TemplatePlanner,
    format_task_results,
    plan_from,
    structured_answer,
    sub_agent,
)

# ==================== 记忆系统（memory） ====================
class MemoryTestCase(PromptkitTestCase):
    '''记忆测试的底子：临时文件 + 自动清理。'''

    def temp_path(self, name="memory.db"):
        import shutil

        folder = tempfile.mkdtemp(prefix="pk-mem-")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return os.path.join(folder, name)

    def temp_dir(self):
        '''一个临时目录（当 `PROMPTKIT_DATA_DIR` 用）。'''
        import shutil

        folder = tempfile.mkdtemp(prefix="pk-data-")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return folder


class RecordingMemory(Memory):
    '''记录每一次调用的假记忆（断言「谁在什么时候读写」用）。'''

    def __init__(self, values=None, hits=None):
        self.values = dict(values or {})
        self.hits = list(hits or [])
        self.events = []
        self.recalls = []

    def read(self, key, default=None):
        self.events.append(("read", key))
        return self.values.get(str(key), default)

    def write(self, key, value):
        self.events.append(("write", key))
        self.values[str(key)] = value

    def recall(self, query, top_k=5):
        self.events.append(("recall", query))
        self.recalls.append((query, top_k))
        return list(self.hits)

    def forget(self, key):
        self.events.append(("forget", key))
        return self.values.pop(str(key), None) is not None

    def keys(self):
        return list(self.values)


def walk_steps(step):
    '''遍历一棵步骤树。'''
    yield step
    for child in step.children():
        yield from walk_steps(child)


class ReverseReranker(Reranker):
    def __init__(self):
        self.calls = []

    def rerank(self, query, items, top_k=None):
        self.calls.append((query, len(items), top_k))
        return list(reversed(list(items)))[:top_k]


class FakeChromaCollection:
    '''假的 chromadb Collection：只实现适配器用到的那几个方法。'''

    def __init__(self):
        self.rows = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for index, item_id in enumerate(ids):
            self.rows[str(item_id)] = {
                "embedding": list(embeddings[index]),
                "document": documents[index],
                "metadata": dict(metadatas[index]),
            }

    def query(self, query_embeddings, n_results, where, include):
        vector = list(query_embeddings[0])
        matched = [
            (item_id, row)
            for item_id, row in self.rows.items()
            if _metadata_matches(row["metadata"], where)
        ]
        matched.sort(
            key=lambda pair: cosine(vector, pair[1]["embedding"]), reverse=True
        )
        matched = matched[:n_results]
        return {
            "ids": [[item_id for item_id, _ in matched]],
            "documents": [[row["document"] for _, row in matched]],
            "metadatas": [[dict(row["metadata"]) for _, row in matched]],
            "distances": [
                [1.0 - cosine(vector, row["embedding"]) for _, row in matched]
            ],
        }

    def get(self, where=None, include=None):
        return {
            "ids": [
                item_id
                for item_id, row in self.rows.items()
                if _metadata_matches(row["metadata"], where)
            ]
        }

    def delete(self, ids=None, where=None):
        doomed = {str(item) for item in (ids or [])}
        if where:
            doomed |= {
                item_id
                for item_id, row in self.rows.items()
                if _metadata_matches(row["metadata"], where)
            }
        for item_id in doomed:
            self.rows.pop(item_id, None)


def _metadata_matches(metadata, where):
    if not where:
        return True
    return all(metadata.get(key) == value for key, value in where.items())


class _FakePoint:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = list(vector)
        self.payload = dict(payload)


class _FakeFilter:
    def __init__(self, must=None):
        self.must = list(must or [])


class _FakeFieldCondition:
    def __init__(self, key, match):
        self.key = key
        self.match = match


class _FakeMatchValue:
    def __init__(self, value):
        self.value = value


class _FakeHasIdCondition:
    def __init__(self, has_id):
        self.has_id = list(has_id)


class FakeQdrantModels:
    '''假的 qdrant_client.models。'''

    PointStruct = _FakePoint
    Filter = _FakeFilter
    FieldCondition = _FakeFieldCondition
    MatchValue = _FakeMatchValue
    HasIdCondition = _FakeHasIdCondition


class FakeQdrantClient:
    '''假的 QdrantClient：只实现适配器用到的那几个方法。'''

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.points = {}
        self.upserts = []
        self.deletes = []

    def upsert(self, collection_name, points):
        self.upserts.append(list(points))
        for point in points:
            self.points[str(point.id)] = point

    def _matches(self, point, condition):
        for item in getattr(condition, "must", []):
            if isinstance(item, _FakeHasIdCondition):
                if str(point.id) not in {str(value) for value in item.has_id}:
                    return False
            elif point.payload.get(item.key) != item.match.value:
                return False
        return True

    def query_points(self, collection_name, query, limit, query_filter, with_payload):
        matched = [
            (item_id, point)
            for item_id, point in self.points.items()
            if self._matches(point, query_filter)
        ]
        matched.sort(key=lambda pair: cosine(query, pair[1].vector), reverse=True)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=item_id,
                    payload=dict(point.payload),
                    score=cosine(query, point.vector),
                )
                for item_id, point in matched[:limit]
            ]
        )

    def delete(self, collection_name, points_selector):
        self.deletes.append(points_selector)
        for item_id in [
            item_id
            for item_id, point in self.points.items()
            if self._matches(point, points_selector)
        ]:
            self.points.pop(item_id, None)

    def count(self, collection_name, count_filter, exact):
        return SimpleNamespace(
            count=len(
                [
                    1
                    for point in self.points.values()
                    if self._matches(point, count_filter)
                ]
            )
        )


class KeywordEmbedding:
    '''按关键词打分的假向量：结果**完全确定**，不受哈希碰撞影响。'''

    def __init__(self, keywords):
        self.keywords = list(keywords)
        self.dimension = len(self.keywords)

    def embed(self, text):
        lowered = str(text or "")
        return [1.0 if keyword in lowered else 0.0 for keyword in self.keywords]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class TestMemoryInterface(MemoryTestCase):
    def test_in_memory_roundtrip(self):
        memory = InMemoryMemory(namespace="u1")
        memory.write("level", "中级")
        self.assertEqual(memory.read("level"), "中级")
        self.assertEqual(memory.namespace, "u1")
        self.assertTrue(memory.has("level"))
        self.assertIn("level", memory)
        self.assertFalse(memory.has("missing"))
        self.assertEqual(memory.read("missing", "默认"), "默认")
        self.assertEqual(len(memory), 1)
        self.assertTrue(memory.forget("level"))
        self.assertFalse(memory.forget("level"))
        self.assertEqual(memory.read("level"), None)

    def test_in_memory_can_hold_none(self):
        # 关键区分：值真的是 None，和「没有这个键」
        memory = InMemoryMemory()
        memory.write("x", None)
        self.assertTrue(memory.has("x"))
        self.assertIsNone(memory.read("x", "默认"))

    def test_default_recall_is_not_implemented(self):
        with self.assertRaises(NotImplementedError) as cm:
            InMemoryMemory().recall("x")
        self.assertIn("VectorMemory", str(cm.exception))

    def test_default_forget_is_not_implemented(self):
        class PureReadWrite(Memory):
            def read(self, key, default=None):
                return default

            def write(self, key, value):
                pass

        with self.assertRaises(NotImplementedError):
            PureReadWrite().forget("x")

    def test_item_text_accepts_three_shapes(self):
        self.assertEqual(format_memories([MemoryItem(id="a", text="甲")]), "[1] 甲")
        self.assertEqual(format_memories([{"text": "乙"}]), "[1] 乙")
        self.assertEqual(format_memories(["丙"]), "[1] 丙")

    def test_format_memories_options(self):
        items = [
            {"text": "甲", "score": 0.912},
            {"text": ""},                      # 空内容跳过
            MemoryItem(id="b", text="乙" * 20),
        ]
        self.assertEqual(format_memories(items, numbered=False), "甲\n" + "乙" * 20)
        self.assertIn("相似度 0.91", format_memories(items, with_score=True))
        self.assertTrue(format_memories(items, max_chars=5).splitlines()[1].endswith("…"))

    def test_memory_item_to_dict(self):
        item = MemoryItem(id="a#0", text="甲", metadata={"key": "a"}, score=0.123456)
        self.assertEqual(item.key, "a")
        self.assertEqual(
            item.to_dict(),
            {"text": "甲", "score": 0.1235, "id": "a#0", "metadata": {"key": "a"}},
        )
        self.assertEqual(item.to_dict(include_metadata=False), {"text": "甲", "score": 0.1235, "id": "a#0"})


class TestTokenBudget(MemoryTestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)
        self.assertEqual(estimate_tokens("你好世界"), 4)
        # 混排：中文按字、英文按 4 字符
        self.assertEqual(estimate_tokens("你好abcd"), 3)

    def test_trim_keeps_order_and_drops_overflow(self):
        budget = TokenBudget(max_tokens=4)
        items = [{"text": "一二"}, {"text": "三四"}, {"text": "五六七八"}]
        kept = budget.trim(items)
        self.assertEqual(kept, items[:2])
        self.assertLessEqual(budget.count_items(kept), 4)

    def test_trim_truncates_the_first_when_it_alone_overflows(self):
        budget = TokenBudget(max_tokens=3)
        kept = budget.trim([{"text": "一二三四五六七八九十"}])
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0]["text"].endswith("…"))
        self.assertLessEqual(budget.count(kept[0]["text"]), 3)

    def test_trim_preserves_item_type(self):
        budget = TokenBudget(max_tokens=2)
        kept = budget.trim([MemoryItem(id="a", text="一二三四")])
        self.assertIsInstance(kept[0], MemoryItem)
        self.assertTrue(kept[0].text.endswith("…"))

    def test_custom_tokenizer(self):
        budget = TokenBudget(max_tokens=2, tokenizer=lambda text: len(text.split()))
        self.assertEqual(budget.count("a b c"), 3)
        kept = budget.trim([{"text": "a b"}, {"text": "c d"}])
        self.assertEqual(len(kept), 1)

    def test_invalid_budget(self):
        for value in (0, -1, "10"):
            with self.assertRaises(ValueError):
                TokenBudget(max_tokens=value)

    def test_count_items(self):
        self.assertEqual(
            TokenBudget().count_items([{"text": "abcd"}, {"text": "efgh"}]), 2
        )


class TestKVMemory(MemoryTestCase):
    def test_roundtrip_of_json_types(self):
        memory = KVMemory(self.temp_path(), namespace="u1")
        values = {
            "text": "中级",
            "number": 42,
            "flag": True,
            "nothing": None,
            "list": [1, 2, 3],
            "dict": {"a": 1},
        }
        for key, value in values.items():
            memory.write(key, value)
        for key, value in values.items():
            self.assertEqual(memory.read(key), value)
        memory.close()

    def test_missing_returns_default(self):
        memory = KVMemory(self.temp_path())
        self.assertEqual(memory.read("nope"), None)
        self.assertEqual(memory.read("nope", "兜底"), "兜底")
        self.assertFalse(memory.has("nope"))
        memory.close()

    def test_overwrite_and_forget(self):
        memory = KVMemory(self.temp_path())
        memory.write("level", "初级")
        memory.write("level", "高级")
        self.assertEqual(memory.read("level"), "高级")
        self.assertTrue(memory.forget("level"))
        self.assertFalse(memory.forget("level"))
        self.assertEqual(memory.read("level"), None)
        memory.close()

    def test_keys_items_count_clear(self):
        memory = KVMemory(self.temp_path(), namespace="u1")
        memory.write("b", 2)
        memory.write("a", 1)
        self.assertEqual(memory.keys(), ["a", "b"])        # 有序，方便 diff
        self.assertEqual(memory.items(), {"a": 1, "b": 2})
        self.assertEqual(memory.count(), 2)
        self.assertEqual(len(memory), 2)
        memory.clear()
        self.assertEqual(memory.count(), 0)
        memory.close()

    def test_namespaces_are_isolated(self):
        path = self.temp_path()
        alice = KVMemory(path, namespace="alice")
        bob = KVMemory(path, namespace="bob")
        alice.write("level", "中级")
        bob.write("level", "入门")
        self.assertEqual(alice.read("level"), "中级")
        self.assertEqual(bob.read("level"), "入门")
        alice.clear()
        self.assertEqual(alice.count(), 0)
        self.assertEqual(bob.read("level"), "入门")
        self.assertEqual(alice.namespaces(), ["bob"])
        alice.close()
        bob.close()

    def test_persists_across_instances(self):
        path = self.temp_path()
        first = KVMemory(path)
        first.write("level", "中级")
        first.close()
        second = KVMemory(path)
        self.assertEqual(second.read("level"), "中级")
        second.close()

    def test_in_memory_database(self):
        memory = KVMemory(":memory:")
        memory.write("x", 1)
        self.assertEqual(memory.read("x"), 1)
        memory.close()

    def test_pydantic_and_dataclass_values(self):
        import dataclasses

        class Point(BaseModel):
            x: int

        @dataclasses.dataclass
        class Plain:
            y: int

        memory = KVMemory(self.temp_path())
        memory.write("point", Point(x=1))
        memory.write("plain", Plain(y=2))
        self.assertEqual(memory.read("point"), {"x": 1})
        self.assertEqual(memory.read("plain"), {"y": 2})
        memory.close()

    def test_unserializable_value_raises(self):
        memory = KVMemory(self.temp_path())
        with self.assertRaises(MemoryError) as cm:
            memory.write("bad", object())
        self.assertIn("JSON", str(cm.exception))
        memory.close()

    def test_context_manager_closes(self):
        with KVMemory(self.temp_path()) as memory:
            memory.write("x", 1)
        self.assertIsNone(memory._conn)


class TestSessionMemory(MemoryTestCase):
    def test_reads_the_bound_agent(self):
        agent = FakeChatAgent()
        agent.history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "在"},
        ]
        agent.start_prompts = [{"role": "system", "content": "你是导师"}]
        agent.rebuild()
        memory = SessionMemory(agent)

        self.assertEqual(memory.read("history")[0]["content"], "你好")
        self.assertEqual(memory.read("turns"), 1)
        self.assertEqual(memory.read("last")["content"], "在")
        self.assertEqual(memory.read("last_user"), "你好")
        self.assertEqual(memory.read("last_assistant"), "在")
        self.assertEqual(memory.read("system"), "你是导师")
        self.assertIn("用户：你好", memory.read("text"))
        self.assertEqual(memory.read("nope", "兜底"), "兜底")

    def test_messages_include_system_only_when_asked(self):
        agent = FakeChatAgent()
        agent.history = [{"role": "user", "content": "hi"}]
        agent.start_prompts = [{"role": "system", "content": "sys"}]
        agent.rebuild()
        # `messages` 就是「真正发给模型的那串」，system 当然在里面
        self.assertEqual(SessionMemory(agent).read("messages")[0]["role"], "system")
        # `history` 默认不含 system
        self.assertEqual(SessionMemory(agent).read("history")[0]["role"], "user")
        with_system = SessionMemory(agent, include_system=True)
        self.assertEqual(with_system.read("history")[0]["role"], "system")
        self.assertIn("系统：sys", with_system.read("text"))

    def test_uses_the_default_agent_when_none(self):
        agent = FakeChatAgent()
        agent.history = [{"role": "user", "content": "默认"}]
        Agent.set_default(agent)
        self.assertEqual(SessionMemory().read("last_user"), "默认")

    def test_without_any_agent_raises(self):
        Agent.set_default(None)
        with self.assertRaises(MemoryError) as cm:
            SessionMemory().read("history")
        self.assertIn("Agent()", str(cm.exception))

    def test_write_is_rejected(self):
        with self.assertRaises(MemoryError) as cm:
            SessionMemory(FakeChatAgent()).write("history", [])
        self.assertIn("只读", str(cm.exception))

    def test_recall_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            SessionMemory(FakeChatAgent()).recall("x")


class TestChunkingAndCosine(MemoryTestCase):
    def test_chunk_text_sizes_and_overlap(self):
        text = "".join(str(i % 10) for i in range(25))
        chunks = memory_chunk_text(text, size=10, overlap=3)
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))
        # 相邻块重叠 3 个字符：第二块的前 3 个字符 == 第一块的后 3 个
        self.assertEqual(chunks[0][-3:], chunks[1][:3])
        self.assertEqual(chunks[0] + chunks[1][3:], text[:17])

    def test_chunk_text_edge_cases(self):
        self.assertEqual(memory_chunk_text(""), [])
        self.assertEqual(memory_chunk_text(None), [])
        self.assertEqual(memory_chunk_text("   "), [])          # 全是空白不切
        self.assertEqual(memory_chunk_text("abc", size=10, overlap=0), ["abc"])

    def test_chunk_text_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            memory_chunk_text("abc", size=0)
        with self.assertRaises(ValueError):
            memory_chunk_text("abc", size=5, overlap=5)

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)
        self.assertEqual(cosine([], [1]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1, 2], [1])


class TestInMemoryVectorStore(MemoryTestCase):
    def test_upsert_search_and_ranking(self):
        store = InMemoryVectorStore()
        store.upsert("u1", MemoryItem(id="a", text="甲", vector=[1.0, 0.0]))
        store.upsert("u1", MemoryItem(id="b", text="乙", vector=[0.0, 1.0]))
        hits = store.search("u1", [1.0, 0.0], top_k=2)
        self.assertEqual(hits[0].id, "a")
        self.assertAlmostEqual(hits[0].score, 1.0)
        self.assertEqual(store.count("u1"), 2)

    def test_namespace_isolation(self):
        store = InMemoryVectorStore()
        store.upsert("u1", MemoryItem(id="a", text="甲", vector=[1.0]))
        store.upsert("u2", MemoryItem(id="b", text="乙", vector=[1.0]))
        self.assertEqual([item.id for item in store.search("u1", [1.0])], ["a"])
        self.assertEqual(store.count("u2"), 1)
        store.clear("u1")
        self.assertEqual(store.count("u1"), 0)
        self.assertEqual(store.count("u2"), 1)

    def test_delete_by_ids_and_by_key(self):
        store = InMemoryVectorStore()
        store.upsert("u1", MemoryItem(id="a#0", text="甲", vector=[1.0], metadata={"key": "a"}))
        store.upsert("u1", MemoryItem(id="a#1", text="甲2", vector=[1.0], metadata={"key": "a"}))
        store.upsert("u1", MemoryItem(id="b#0", text="乙", vector=[1.0], metadata={"key": "b"}))
        self.assertEqual(store.delete("u1", ids=["a#0"]), 1)
        self.assertEqual(store.delete("u1", key="a"), 1)
        self.assertEqual(store.count("u1"), 1)

    def test_upsert_copies_metadata(self):
        store = InMemoryVectorStore()
        metadata = {"key": "a"}
        store.upsert("u1", MemoryItem(id="a#0", text="甲", vector=[1.0], metadata=metadata))
        metadata["key"] = "改掉了"
        self.assertEqual(store.search("u1", [1.0])[0].key, "a")


class TestVectorMemory(MemoryTestCase):
    def make_memory(self, **kwargs):
        options = {
            "chunk_size": 40,
            "overlap": 5,
            "namespace": "u1",
        }
        options.update(kwargs)
        return VectorMemory(HashEmbedding(dimension=128), **options)

    def test_write_chunks_and_count(self):
        memory = self.make_memory()
        text = "退款政策：" + "无理由退款。" * 20
        memory.write("notes", text)
        self.assertGreater(memory.count(), 1)
        self.assertEqual(memory.count(), len(memory_chunk_text(text, 40, 5)))

    def test_recall_prefers_relevant_chunks(self):
        memory = self.make_memory()
        memory.write("refund", "退款政策：7 天内无理由退款，需要订单号。")
        memory.write("weather", "今天多云转晴，最高气温 26 度，适合出门。")
        hits = memory.recall("退款需要订单号吗", top_k=2)
        self.assertTrue(hits)
        self.assertIn("退款", hits[0]["text"])
        self.assertGreater(hits[0]["score"], 0.0)
        self.assertIn("metadata", hits[0])

    def test_min_score_filters_noise(self):
        memory = VectorMemory(KeywordEmbedding(["退款", "天气"]), namespace="u1")
        memory.write("refund", "退款政策：7 天内无理由退款。")
        self.assertEqual(memory.recall("量子色动力学胶子禁闭"), [])
        # 放宽阈值就又能看到「零相关」的东西
        loose = VectorMemory(
            KeywordEmbedding(["退款", "天气"]), namespace="u1", min_score=None
        )
        loose.write("refund", "退款政策：7 天内无理由退款。")
        self.assertTrue(loose.recall("量子色动力学胶子禁闭", top_k=1))

    def test_overwrite_removes_stale_chunks(self):
        memory = self.make_memory()
        memory.write("notes", "一" * 200)
        before = memory.count()
        memory.write("notes", "短")
        self.assertEqual(memory.count(), 1)
        self.assertLess(memory.count(), before)
        self.assertEqual(memory.recall("短", top_k=1)[0]["text"], "短")

    def test_write_empty_clears(self):
        memory = self.make_memory()
        memory.write("notes", "有内容")
        self.assertEqual(memory.count(), 1)
        memory.write("notes", "")
        self.assertEqual(memory.count(), 0)

    def test_forget(self):
        memory = self.make_memory()
        memory.write("notes", "有内容")
        self.assertTrue(memory.forget("notes"))
        self.assertFalse(memory.forget("notes"))
        self.assertEqual(memory.recall("有内容"), [])

    def test_read_is_not_supported(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.make_memory().read("notes")
        self.assertIn("KVMemory", str(cm.exception))

    def test_budget_is_applied(self):
        memory = self.make_memory(budget=TokenBudget(max_tokens=10))
        memory.write("notes", "退款政策：" + "无理由退款。" * 20)
        hits = memory.recall("退款", top_k=5)
        self.assertEqual(len(hits), 1)                    # 只留得下一条
        self.assertLessEqual(
            estimate_tokens(hits[0]["text"]), 10
        )

    def test_reranker_is_called_and_can_reorder(self):
        reranker = ReverseReranker()
        memory = self.make_memory(reranker=reranker)
        memory.write("a", "退款政策：7 天无理由。")
        memory.write("b", "退款需要订单号。")
        hits = memory.recall("退款", top_k=2)
        self.assertTrue(reranker.calls)
        query, candidate_count, top_k = reranker.calls[0]
        self.assertEqual((query, top_k), ("退款", 2))
        self.assertGreaterEqual(candidate_count, 1)
        self.assertEqual(len(hits), 2)

    def test_max_age_filters_old_items(self):
        store = InMemoryVectorStore()
        memory = self.make_memory(store=store, max_age=10)
        memory.write("notes", "退款政策")
        for item in store._items.values():
            item.metadata["ts"] = 1.0                      # 很久以前
        self.assertEqual(memory.recall("退款"), [])

    def test_decay_pushes_old_items_down(self):
        import time

        store = InMemoryVectorStore()
        memory = self.make_memory(store=store, decay_half_life=1.0, min_score=None)
        memory.write("old", "退款政策 甲")
        memory.write("new", "退款政策 乙")
        for (_, item_id), item in store._items.items():
            item.metadata["ts"] = 1.0 if item_id.startswith("old") else time.time()
        hits = memory.recall("退款政策", top_k=2)
        self.assertEqual(hits[0]["metadata"]["key"], "new")

    def test_clear(self):
        memory = self.make_memory()
        memory.write("a", "甲")
        memory.clear()
        self.assertEqual(memory.count(), 0)

    def test_embedding_client_is_required(self):
        with self.assertRaises(ValueError):
            VectorMemory(None)


class TestMemorySteps(MemoryTestCase):
    def test_read_memory(self):
        memory = RecordingMemory({"level": "中级"})
        ctx = Workflow(
            ReadMemory(memory, "level", output_key="level"),
        ).run()
        self.assertEqual(ctx["level"], "中级")
        self.assertEqual(memory.events, [("read", "level")])
        self.assertIn("read_memory:level", ctx.trace)

    def test_read_memory_defaults_output_to_key_name(self):
        memory = RecordingMemory()
        ctx = Workflow(ReadMemory(memory, "level", default="新手")).run()
        self.assertEqual(ctx["level"], "新手")

    def test_read_memory_dynamic_key(self):
        memory = RecordingMemory({"u1_level": "高级"})
        ctx = Workflow(
            ReadMemory(memory, Var("who") + "_level", output_key="level"),
        ).run(who="u1")
        self.assertEqual(ctx["level"], "高级")

    def test_write_memory_literal_expression_and_from_key(self):
        memory = RecordingMemory()
        ctx = Workflow(
            WriteMemory(memory, "a", "字面量"),
            WriteMemory(memory, "b", Var("n") * 2),
            WriteMemory(memory, "c", from_key="n"),
        ).run(n=3)
        self.assertEqual(memory.values, {"a": "字面量", "b": 6, "c": 3})
        self.assertIn("write_memory:a", ctx.trace)

    def test_write_memory_skips_missing_variable(self):
        memory = RecordingMemory()
        ctx = Workflow(WriteMemory(memory, "c", from_key="nope")).run()
        self.assertEqual(memory.values, {})
        self.assertIn("write_memory:skip", ctx.trace)

    def test_write_memory_can_write_none_when_missing(self):
        memory = RecordingMemory()
        Workflow(WriteMemory(memory, "c", from_key="nope", missing="write")).run()
        self.assertEqual(memory.values, {"c": None})

    def test_write_memory_needs_a_source(self):
        with self.assertRaises(ValueError):
            WriteMemory(RecordingMemory(), "k")

    def test_forget_memory(self):
        memory = RecordingMemory({"level": "中级"})
        ctx = Workflow(ForgetMemory(memory, "level")).run()
        self.assertEqual(memory.events, [("forget", "level")])
        self.assertIn("forget_memory:level:ok", ctx.trace)

    def test_steps_validate_the_memory_object(self):
        with self.assertRaises(TypeError):
            ReadMemory(object(), "k")
        with self.assertRaises(TypeError):
            WriteMemory(object(), "k", 1)
        with self.assertRaises(TypeError):
            ForgetMemory(object(), "k")

    def test_memory_step_compiles_into_plain_steps(self):
        memory = RecordingMemory({"a": 1, "b": 2})
        step = MemoryStep(memory, read=["a", "b"], write=["w"])
        ctx = Workflow(step).run(w="值")
        self.assertEqual([s.name for s in step.children()], ["read_memory", "read_memory", "write_memory"])
        self.assertEqual(ctx["a"], 1)
        self.assertEqual(memory.values["w"], "值")
        self.assertIn("read=2", step.describe())
        self.assertIn("write=1", step.describe())

    def test_memory_step_with_recall_config(self):
        memory = RecordingMemory(hits=[{"text": "退款政策", "score": 0.9}])
        step = MemoryStep(
            memory,
            recall=RecallConfig(query_key="q", output_key="hits", top_k=3),
        )
        ctx = Workflow(step).run(q="退款")
        self.assertIn("退款政策", ctx["hits"])
        self.assertEqual(memory.recalls, [("退款", 3)])
        self.assertIn("recall", step.describe())

    def test_memory_step_accepts_a_recall_dict(self):
        memory = RecordingMemory(hits=[{"text": "x"}])
        step = MemoryStep(memory, recall={"query_key": "q", "output_key": "hits"})
        ctx = Workflow(step).run(q="问题")
        self.assertEqual(ctx["hits"], "[1] x")

    def test_memory_step_tuple_write(self):
        memory = RecordingMemory()
        Workflow(MemoryStep(memory, write=[("key_a", "var_a")])).run(var_a="值")
        self.assertEqual(memory.values["key_a"], "值")

    def test_memory_step_needs_something_to_do(self):
        with self.assertRaises(ValueError):
            MemoryStep(RecordingMemory())

    def test_memory_step_counts_as_one_trace_entry(self):
        memory = RecordingMemory({"a": 1})
        ctx = Workflow(MemoryStep(memory, read=["a"])).run()
        self.assertEqual(ctx.trace.count("memory"), 1)


class TestRecallMemory(MemoryTestCase):
    HITS = [
        {"text": "退款政策：7 天无理由", "score": 0.9},
        {"text": "发货时间：48 小时内", "score": 0.2},
    ]

    def test_default_output_is_readable_text(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(
            RecallMemory(memory, query=Var("q"), output_key="recalled", top_k=2),
        ).run(q="退款")
        self.assertEqual(
            ctx["recalled"], "[1] 退款政策：7 天无理由\n[2] 发货时间：48 小时内"
        )
        self.assertEqual(memory.recalls, [("退款", 2)])
        self.assertIn("recall_memory:2", ctx.trace)

    def test_raw_output_is_structured(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(
            RecallMemory(memory, query="退款", output_key="raw", raw=True)
        ).run()
        self.assertEqual(ctx["raw"], self.HITS)

    def test_query_key_reads_the_context(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(
            RecallMemory(memory, query_key="usr_input", output_key="hits"),
        ).run(usr_input="发货")
        self.assertEqual(memory.recalls[0][0], "发货")
        self.assertTrue(ctx["hits"])

    def test_literal_query(self):
        memory = RecordingMemory(hits=self.HITS)
        Workflow(RecallMemory(memory, query="写死的查询")).run()
        self.assertEqual(memory.recalls[0][0], "写死的查询")

    def test_empty_query_skips_recall(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(RecallMemory(memory, query_key="q", output_key="hits")).run(q="  ")
        self.assertEqual(memory.recalls, [])
        self.assertEqual(ctx["hits"], "")
        self.assertIn("recall_memory:skip", ctx.trace)

    def test_min_score_filters(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(
            RecallMemory(memory, query="x", output_key="hits", min_score=0.5)
        ).run()
        self.assertEqual(ctx["hits"], "[1] 退款政策：7 天无理由")

    def test_budget_trims(self):
        memory = RecordingMemory(hits=self.HITS)
        ctx = Workflow(
            RecallMemory(
                memory, query="x", output_key="hits", budget=TokenBudget(max_tokens=4)
            )
        ).run()
        self.assertNotIn("发货时间", ctx["hits"])

    def test_max_chars_truncates_each_item(self):
        memory = RecordingMemory(hits=[{"text": "甲" * 50, "score": 0.9}])
        ctx = Workflow(
            RecallMemory(memory, query="x", output_key="hits", max_chars=10)
        ).run()
        self.assertEqual(ctx["hits"], "[1] " + "甲" * 9 + "…")

    def test_needs_a_query(self):
        with self.assertRaises(ValueError):
            RecallMemory(RecordingMemory())

    def test_needs_a_recall_capable_memory(self):
        with self.assertRaises(TypeError):
            RecallMemory(InMemoryMemory(), query="x")

    def test_top_k_validation(self):
        with self.assertRaises(ValueError):
            RecallMemory(RecordingMemory(), query="x", top_k=0)

    def test_describe(self):
        step = RecallMemory(RecordingMemory(), query_key="q", top_k=3)
        self.assertIn("top_k=3", step.describe())
        self.assertIn("-> recalled", step.describe())


class TestRecallMemoryTool(MemoryTestCase):
    def make_tool(self, memory=None, **kwargs):
        memory = memory if memory is not None else RecordingMemory(
            hits=[
                {"text": "退款政策：7 天无理由", "score": 0.9, "metadata": {"owner": "u1", "phone": "123"}},
                {"text": "发货时间：48 小时内", "score": 0.4, "metadata": {"owner": "u1"}},
            ]
        )
        return RecallMemoryTool(memory, **kwargs), memory

    def test_returns_results_and_calls_recall(self):
        tool, memory = self.make_tool()
        result = tool.run("退款")
        self.assertEqual(result["query"], "退款")
        self.assertEqual(result["count"], 2)
        self.assertEqual(memory.recalls, [("退款", 3)])
        self.assertEqual(result["results"][0]["text"], "退款政策：7 天无理由")

    def test_tool_call_returns_json_text(self):
        tool, _ = self.make_tool()
        payload = json.loads(tool.call({"query": "退款", "top_k": 2}))
        self.assertEqual(payload["count"], 2)

    def test_top_k_is_capped(self):
        tool, memory = self.make_tool()
        tool.run("x", top_k=99)
        self.assertEqual(memory.recalls[0][1], 5)          # max_top_k 默认 5

    def test_call_limit(self):
        tool, memory = self.make_tool(max_calls=2)
        tool.run("a")
        tool.run("b")
        third = tool.run("c")
        self.assertIn("error", third)
        self.assertEqual(len(memory.recalls), 2)
        self.assertEqual(tool.calls, 2)
        tool.reset()
        self.assertEqual(tool.calls, 0)
        self.assertEqual(tool.run("d")["count"], 2)

    def test_empty_query_is_rejected(self):
        tool, memory = self.make_tool()
        self.assertIn("error", tool.run("   "))
        self.assertEqual(memory.recalls, [])

    def test_long_items_are_truncated(self):
        tool, _ = self.make_tool(
            RecordingMemory(hits=[{"text": "甲" * 100, "score": 0.9}]),
            max_item_chars=10,
        )
        text = tool.run("x")["results"][0]["text"]
        self.assertEqual(len(text), 10)
        self.assertTrue(text.endswith("…"))

    def test_metadata_is_hidden_by_default(self):
        tool, _ = self.make_tool()
        self.assertNotIn("metadata", tool.run("退款")["results"][0])

    def test_metadata_can_be_included_and_redacted(self):
        tool, _ = self.make_tool(include_metadata=True, redact=["phone"])
        metadata = tool.run("退款")["results"][0]["metadata"]
        self.assertEqual(metadata, {"owner": "u1"})

    def test_budget_is_applied(self):
        tool, _ = self.make_tool(budget=TokenBudget(max_tokens=6))
        self.assertEqual(tool.run("退款")["count"], 1)

    def test_kv_memory_gives_a_clear_error(self):
        tool = RecallMemoryTool(InMemoryMemory())
        result = tool.run("x")
        self.assertIn("error", result)

    def test_schema(self):
        tool, _ = self.make_tool()
        schema = tool.schema()["function"]
        self.assertEqual(schema["name"], "recall_memory")
        self.assertEqual(schema["parameters"]["required"], ["query"])

    def test_invalid_limits(self):
        for kwargs in ({"max_top_k": 0}, {"max_calls": 0}, {"max_item_chars": 0}):
            with self.assertRaises(ValueError):
                RecallMemoryTool(RecordingMemory(), **kwargs)

    def test_audit_log(self):
        tool, _ = self.make_tool()
        with self.assertLogs("promptkit.memory", level="INFO") as captured:
            tool.run("退款")
        self.assertIn("recall_memory", captured.output[0])


class TestLayeredMemory(MemoryTestCase):
    def make(self, **kwargs):
        kv = kwargs.pop("kv", None) or KVMemory(self.temp_path(), namespace="u1")
        vector = kwargs.pop("vector", None)
        if vector is None:
            vector = VectorMemory(HashEmbedding(dimension=64), namespace="u1")
        memory = LayeredMemory(kv=kv, vector=vector, **kwargs)
        self.addCleanup(kv.close)
        return memory

    def test_read_write_recall_forget(self):
        memory = self.make()
        memory.write("level", "中级")
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertEqual(memory.read("level"), "中级")
        hits = memory.recall("退款", top_k=1)
        self.assertIn("退款", hits[0]["text"])
        self.assertTrue(memory.forget("notes"))
        self.assertEqual(memory.read("notes"), None)
        self.assertEqual(memory.recall("退款", top_k=1), [])

    def test_mirror_off_skips_the_vector_store(self):
        memory = self.make(mirror=False)
        memory.write("level", "中级")
        self.assertEqual(memory.read("level"), "中级")
        self.assertEqual(memory.recall("中级", top_k=1), [])

    def test_kv_only(self):
        memory = LayeredMemory(kv=KVMemory(self.temp_path()))
        self.addCleanup(memory.kv.close)
        memory.write("k", "v")
        self.assertEqual(memory.read("k"), "v")
        with self.assertRaises(NotImplementedError):
            memory.recall("v")

    def test_vector_only(self):
        memory = LayeredMemory(vector=VectorMemory(HashEmbedding(dimension=32)))
        memory.write("k", "退款")
        self.assertTrue(memory.recall("退款", top_k=1))
        with self.assertRaises(NotImplementedError):
            memory.read("k")

    def test_requires_at_least_one_backend(self):
        with self.assertRaises(ValueError):
            LayeredMemory()

    def test_repr_mentions_both_backends(self):
        text = repr(self.make())
        self.assertIn("KVMemory", text)
        self.assertIn("VectorMemory", text)
        self.assertIn("u1", text)

    def test_long_term_memory_without_embedding_is_kv_only(self):
        memory = long_term_memory("u1", path=self.temp_path())
        self.addCleanup(memory.kv.close)
        memory.write("level", "中级")
        self.assertEqual(memory.read("level"), "中级")
        with self.assertRaises(NotImplementedError):
            memory.recall("中级")

    def test_long_term_memory_with_embedding(self):
        memory = long_term_memory(
            "u1",
            path=self.temp_path(),
            vector_path=self.temp_path("vectors"),
            embed_client=HashEmbedding(dimension=64),
        )
        self.addCleanup(memory.close)
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertTrue(memory.recall("退款", top_k=1))


class TestChatWorkflowMemory(MemoryTestCase):
    def run_chat(self, chat, inputs, **kwargs):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=[*inputs, "exit"]):
            ctx = chat.run(
                agent=agent,
                verbose=Verbosity.SILENT,
                out=io.StringIO(),
                **kwargs,
            )
        return ctx, agent

    def test_read_happens_before_the_system_prompt(self):
        memory = RecordingMemory({"level": "中级"})
        chat = ChatWorkflow(
            role="课程导师",
            rules=["学员水平：{{level}}"],
            memory=memory,
            memory_config=MemoryConfig(read=["level"]),
        )
        ctx, agent = self.run_chat(chat, ["你好"])
        self.assertEqual(ctx["level"], "中级")
        self.assertIn("学员水平：中级", agent.start_prompts[0]["content"])

    def test_recall_runs_every_turn_and_is_injected_into_the_prompt(self):
        memory = VectorMemory(HashEmbedding(dimension=128), namespace="u1",
                              chunk_size=60, overlap=10)
        memory.write("notes", "退款政策：7 天内无理由退款，需要订单号。")
        chat = ChatWorkflow(
            role="客服",
            memory=memory,
            memory_config=MemoryConfig(
                recall=RecallConfig(query_key="usr_input", output_key="related", top_k=1)
            ),
        )
        ctx, agent = self.run_chat(chat, ["退款怎么办"])
        self.assertIn("相关片段", agent.prompts[0])
        self.assertIn("退款政策", agent.prompts[0])
        self.assertIn("退款政策", ctx["related"])

    def test_recall_template_can_be_turned_off_for_raw_output(self):
        memory = RecordingMemory(hits=[{"text": "退款", "score": 0.9}])
        chat = ChatWorkflow(
            role="客服",
            memory=memory,
            memory_config=MemoryConfig(
                recall=RecallConfig(query_key="usr_input", output_key="related", raw=True)
            ),
        )
        self.assertIsNone(chat.turn_template())
        ctx, agent = self.run_chat(chat, ["问题"])
        self.assertEqual(ctx["related"], [{"text": "退款", "score": 0.9}])

    def test_write_on_end_happens_after_the_loop(self):
        memory = RecordingMemory()
        chat = ChatWorkflow(
            role="导师",
            memory=memory,
            memory_config=MemoryConfig(write=["topics_covered"], write_on="end"),
        )
        ctx, _ = self.run_chat(chat, ["第一轮"], topics_covered="退款政策")
        self.assertEqual(memory.values["topics_covered"], "退款政策")
        self.assertGreater(
            ctx.trace.index("write_memory:topics_covered"),
            ctx.trace.index("while:done"),
        )

    def test_write_on_each_turn_happens_inside_the_loop(self):
        memory = RecordingMemory()
        chat = ChatWorkflow(
            role="导师",
            memory=memory,
            memory_config=MemoryConfig(write=["topics_covered"], write_on="each_turn"),
        )
        ctx, _ = self.run_chat(chat, ["第一轮"], topics_covered="退款政策")
        self.assertEqual(memory.values["topics_covered"], "退款政策")
        self.assertLess(
            ctx.trace.index("write_memory:topics_covered"),
            ctx.trace.index("while:done"),
        )

    def test_deliverables_still_save_each_turn_with_memory(self):
        memory = RecordingMemory()
        folder = os.path.dirname(self.temp_path("x.md"))
        chat = ChatWorkflow(
            role="导师",
            deliverables=[
                Deliverable("笔记", save_to=folder, when="each_turn"),
            ],
            memory=memory,
            memory_config=MemoryConfig(read=["level"]),
        )
        self.run_chat(chat, ["第一轮"], level="中级")
        self.assertEqual(os.listdir(folder), ["笔记.md"])

    def test_memory_without_config_raises(self):
        with self.assertRaises(ValueError):
            ChatWorkflow(memory=RecordingMemory())

    def test_config_without_memory_raises(self):
        with self.assertRaises(ValueError):
            ChatWorkflow(memory_config=MemoryConfig(read=["x"]))

    def test_empty_config_raises(self):
        with self.assertRaises(ValueError):
            ChatWorkflow(memory=RecordingMemory(), memory_config=MemoryConfig())

    def test_config_must_be_a_memory_config(self):
        with self.assertRaises(TypeError):
            ChatWorkflow(memory=RecordingMemory(), memory_config=object())

    def test_config_accepts_a_dict(self):
        chat = ChatWorkflow(
            memory=RecordingMemory({"level": "中级"}),
            memory_config={"read": ["level"]},
        )
        ctx, _ = self.run_chat(chat, ["你好"])
        self.assertEqual(ctx["level"], "中级")

    def test_build_puts_reads_before_the_system_prompt(self):
        chat = ChatWorkflow(
            role="导师",
            memory=RecordingMemory(),
            memory_config=MemoryConfig(read=["level"]),
        )
        descriptions = [step.describe() for step in chat.build().steps]
        self.assertLess(
            next(i for i, text in enumerate(descriptions) if "read=" in text),
            next(i for i, text in enumerate(descriptions) if "SystemPrompt" in text),
        )

    def test_memory_is_visible_in_the_step_tree(self):
        chat = ChatWorkflow(
            role="导师",
            memory=RecordingMemory(),
            memory_config=MemoryConfig(
                read=["level"],
                recall=RecallConfig(query_key="usr_input", output_key="related"),
                write=["topics_covered"],
            ),
        )
        tree = chat.build().render()
        self.assertIn("MemoryStep", tree)
        self.assertIn("RecallMemory", tree)
        self.assertIn("WriteMemory", tree)


class TestChatLoopTurnTemplate(MemoryTestCase):
    def test_default_turn_content_is_the_input(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(ChatLoop()).run(
                agent=agent, verbose=Verbosity.SILENT, out=io.StringIO()
            )
        self.assertEqual(agent.prompts[0], "你好")

    def test_custom_turn_template(self):
        agent = FakeChatAgent()
        with mock.patch("builtins.input", side_effect=["你好", "exit"]):
            Workflow(
                ChatLoop(turn_template="{{usr_input}}\n\n附加：{{extra}}"),
            ).run(
                agent=agent,
                verbose=Verbosity.SILENT,
                out=io.StringIO(),
                extra="参考记忆",
            )
        self.assertEqual(agent.prompts[0], "你好\n\n附加：参考记忆")


class TestMemoryAdapters(MemoryTestCase):
    def test_hash_embedding_is_deterministic_and_normalized(self):
        embedder = HashEmbedding(dimension=64)
        left = embedder.embed("退款政策")
        right = embedder.embed("退款政策")
        self.assertEqual(left, right)
        self.assertEqual(embedder.dimension, 64)
        self.assertAlmostEqual(sum(value * value for value in left), 1.0, places=6)
        self.assertEqual(embedder.embed(""), [0.0] * 64)

    def test_hash_embedding_reflects_lexical_similarity(self):
        embedder = HashEmbedding(dimension=256)
        query = embedder.embed("退款政策")
        near = embedder.embed("退款政策说明")
        far = embedder.embed("今天天气很好")
        self.assertGreater(cosine(query, near), cosine(query, far))
        self.assertGreater(cosine(query, far), -1.0)

    def test_hash_embedding_validation(self):
        with self.assertRaises(ValueError):
            HashEmbedding(dimension=4)
        with self.assertRaises(ValueError):
            HashEmbedding(ngram=0)

    def test_openai_embedding_uses_the_client(self):
        api = SimpleNamespace(calls=[], create=None)

        def create(**kwargs):
            api.calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(len(text))])
                    for index, text in enumerate(kwargs["input"])
                ]
            )

        api.create = create
        embedder = OpenAIEmbedding(client=SimpleNamespace(embeddings=api))
        vectors = embedder.embed_many(["a", "bb", "ccc"])
        self.assertEqual(vectors, [[1.0], [2.0], [3.0]])
        self.assertEqual(api.calls[0]["model"], "text-embedding-3-small")
        self.assertEqual(api.calls[0]["input"], ["a", "bb", "ccc"])

    def test_openai_embedding_batches(self):
        calls = []

        def create(**kwargs):
            calls.append(list(kwargs["input"]))
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.0]) for _ in kwargs["input"]]
            )

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        embedder = OpenAIEmbedding(client=client, batch_size=2)
        embedder.embed_many(["a", "b", "c"])
        self.assertEqual(calls, [["a", "b"], ["c"]])

    def test_openai_embedding_known_dimension(self):
        embedder = OpenAIEmbedding(client=SimpleNamespace(embeddings=SimpleNamespace()))
        self.assertEqual(embedder.dimension, 1536)

    def test_sentence_transformer_embedding(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def get_sentence_embedding_dimension(self):
                return 3

            def encode(self, texts, normalize_embeddings=True):
                self.calls.append((list(texts), normalize_embeddings))
                return [[float(len(text)), 1.0, 0.0] for text in texts]

        model = FakeModel()
        embedder = SentenceTransformerEmbedding(model=model)
        self.assertEqual(embedder.dimension, 3)
        self.assertEqual(embedder.embed("ab"), [2.0, 1.0, 0.0])
        self.assertEqual(model.calls[0][1], True)

    def test_sentence_transformer_needs_encode(self):
        with self.assertRaises(TypeError):
            SentenceTransformerEmbedding(model=object())

    def test_callable_embedding(self):
        from promptkit.memory.adapters import CallableEmbedding

        embedder = CallableEmbedding(lambda text: [float(len(text))], dimension=1)
        self.assertEqual(embedder.embed("abc"), [3.0])

    def test_chroma_store(self):
        collection = FakeChromaCollection()
        store = ChromaVectorStore(collection)
        store.upsert("u1", MemoryItem(id="a#0", text="退款政策", vector=[1.0, 0.0],
                                      metadata={"key": "a", "nested": {"x": 1}}))
        store.upsert("u2", MemoryItem(id="b#0", text="别的用户", vector=[1.0, 0.0],
                                      metadata={"key": "b"}))
        self.assertEqual(store.count("u1"), 1)

        hits = store.search("u1", [1.0, 0.0], top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].text, "退款政策")
        self.assertNotIn("namespace", hits[0].metadata)
        self.assertEqual(hits[0].score, 1.0)
        self.assertIsInstance(hits[0].metadata["nested"], str)   # 复杂值被 JSON 化

        self.assertEqual(store.delete("u1", key="a"), 1)
        self.assertEqual(store.count("u1"), 0)
        self.assertEqual(store.count("u2"), 1)
        store.clear("u2")
        self.assertEqual(store.count("u2"), 0)

    def test_qdrant_store(self):
        models = FakeQdrantModels()
        client = FakeQdrantClient()
        store = QdrantVectorStore(client, "notes", models=models)
        store.upsert("u1", MemoryItem(id="a#0", text="退款政策", vector=[1.0, 0.0],
                                      metadata={"key": "a"}))
        store.upsert("u2", MemoryItem(id="b#0", text="别人的", vector=[1.0, 0.0],
                                      metadata={"key": "b"}))

        payload = client.points[next(iter(client.points))].payload
        self.assertEqual(payload["text"], "退款政策")
        self.assertEqual(payload["namespace"], "u1")
        self.assertEqual(payload["key"], "a")

        hits = store.search("u1", [1.0, 0.0], top_k=2)
        self.assertEqual([item.text for item in hits], ["退款政策"])
        self.assertEqual(store.count("u1"), 1)
        store.delete("u1", key="a")
        self.assertTrue(client.deletes)

    def test_cohere_reranker(self):
        client = SimpleNamespace(
            calls=[],
            rerank=None,
        )

        def rerank(model, query, documents, top_n):
            client.calls.append((model, query, documents, top_n))
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=index, relevance_score=1.0 - index * 0.1)
                    for index in reversed(range(min(top_n, len(documents))))
                ]
            )

        client.rerank = rerank
        reranker = CohereReranker(client, model="rerank-v3")
        items = [MemoryItem(id="a", text="甲"), MemoryItem(id="b", text="乙")]
        result = reranker.rerank("q", items, top_k=2)
        self.assertEqual([item.id for item in result], ["b", "a"])
        self.assertEqual(client.calls[0][0], "rerank-v3")
        self.assertEqual(client.calls[0][2], ["甲", "乙"])

    def test_cohere_reranker_ignores_bad_indexes(self):
        client = SimpleNamespace(
            rerank=lambda **kwargs: SimpleNamespace(
                results=[SimpleNamespace(index=5, relevance_score=0.9)]
            )
        )
        self.assertEqual(
            CohereReranker(client).rerank("q", [MemoryItem(id="a", text="甲")]), []
        )

    def test_cross_encoder_reranker(self):
        class FakeCross:
            def predict(self, pairs):
                return [float(len(text)) for _, text in pairs]

        items = [MemoryItem(id="a", text="短"), MemoryItem(id="b", text="长很多")]
        result = CrossEncoderReranker(FakeCross()).rerank("q", items, top_k=1)
        self.assertEqual(result[0].id, "b")

    def test_llm_reranker(self):
        agent = FakeAgent(['{"order": [2, 0, 1]}'])
        items = [
            MemoryItem(id="a", text="甲"),
            MemoryItem(id="b", text="乙"),
            MemoryItem(id="c", text="丙"),
        ]
        result = LlmReranker(agent).rerank("q", items, top_k=2)
        self.assertEqual([item.id for item in result], ["c", "a"])
        self.assertGreater(result[0].score, result[1].score)

    def test_llm_reranker_fills_missing_indexes(self):
        agent = FakeAgent(['{"order": [1]}'])
        items = [MemoryItem(id="a", text="甲"), MemoryItem(id="b", text="乙"),
                 MemoryItem(id="c", text="丙")]
        result = LlmReranker(agent).rerank("q", items)
        self.assertEqual([item.id for item in result], ["b", "a", "c"])

    def test_llm_reranker_falls_back_to_numbers(self):
        agent = FakeAgent(["排序结果：2 0"])
        items = [MemoryItem(id="a", text="甲"), MemoryItem(id="b", text="乙"),
                 MemoryItem(id="c", text="丙")]
        result = LlmReranker(agent).rerank("q", items)
        self.assertEqual([item.id for item in result], ["c", "a", "b"])

    def test_llm_reranker_needs_an_agent(self):
        Agent.set_default(None)
        items = [MemoryItem(id="a", text="甲"), MemoryItem(id="b", text="乙")]
        with self.assertRaises(ValueError):
            LlmReranker().rerank("q", items)

    def test_noop_reranker(self):
        items = [MemoryItem(id="a", text="甲"), MemoryItem(id="b", text="乙")]
        self.assertEqual(len(NoopReranker().rerank("q", items, top_k=1)), 1)


class TestStoragePaths(MemoryTestCase):
    def test_data_dir_honors_the_env_override(self):
        folder = self.temp_dir()
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": folder}):
            self.assertEqual(default_data_dir(), os.path.abspath(folder))

    def test_windows_default_dir(self):
        with mock.patch.dict(
            os.environ,
            {"PROMPTKIT_DATA_DIR": "", "APPDATA": r"C:\Users\me\AppData\Roaming"},
            clear=True,
        ):
            with mock.patch("os.name", "nt"):
                self.assertEqual(
                    default_data_dir(),
                    os.path.join(r"C:\Users\me\AppData\Roaming", "promptkit"),
                )

    @unittest.skipIf(os.name == "nt", "POSIX 路径语义在 Windows 上不适用（ntpath）")
    def test_macos_default_dir(self):
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": ""}, clear=True):
            with mock.patch("os.name", "posix"), mock.patch("sys.platform", "darwin"):
                with mock.patch("os.path.expanduser", return_value="/Users/me"):
                    self.assertEqual(
                        default_data_dir(),
                        "/Users/me/Library/Application Support/promptkit",
                    )

    @unittest.skipIf(os.name == "nt", "POSIX 路径语义在 Windows 上不适用（ntpath）")
    def test_linux_respects_xdg(self):
        with mock.patch.dict(
            os.environ, {"PROMPTKIT_DATA_DIR": "", "XDG_DATA_HOME": "/data"}, clear=True
        ):
            with mock.patch("os.name", "posix"), mock.patch("sys.platform", "linux"):
                self.assertEqual(default_data_dir(), "/data/promptkit")

    @unittest.skipIf(os.name == "nt", "POSIX 路径语义在 Windows 上不适用（ntpath）")
    def test_linux_falls_back_to_local_share(self):
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": ""}, clear=True):
            with mock.patch("os.name", "posix"), mock.patch("sys.platform", "linux"):
                with mock.patch("os.path.expanduser", return_value="/home/me"):
                    self.assertEqual(
                        default_data_dir(), "/home/me/.local/share/promptkit"
                    )

    def test_default_paths_follow_the_namespace(self):
        folder = self.temp_dir()
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": folder}):
            self.assertTrue(
                default_kv_path("user_001").endswith(os.path.join("memory", "kv", "user_001.db"))
            )
            self.assertTrue(
                default_vector_dir("user_001").endswith(os.path.join("memory", "vector", "user_001"))
            )

    def test_namespace_is_made_file_safe(self):
        # 中文原样保留（好认）
        self.assertEqual(safe_namespace_name("张三"), "张三")
        # 路径分隔符不能出现在文件名里，而且不能和 a_b 撞车
        unsafe = safe_namespace_name("a/b")
        self.assertNotIn("/", unsafe)
        self.assertNotIn(os.sep, unsafe)
        self.assertNotEqual(unsafe, "a_b")
        self.assertEqual(unsafe, safe_namespace_name("a/b"))     # 稳定
        # `..` 这种不能原样用
        self.assertNotIn("..", safe_namespace_name(".."))

    def test_expand_path(self):
        from promptkit.memory.paths import expand_path

        self.assertEqual(expand_path("/tmp/x"), os.path.abspath("/tmp/x"))
        self.assertTrue(os.path.isabs(expand_path("~/x")))


class TestStorageConfig(MemoryTestCase):
    def test_namespace_is_required(self):
        with self.assertRaises(TypeError):
            StorageConfig(backend="sqlite")            # type: ignore[call-arg]

    def test_namespace_cannot_be_blank(self):
        with self.assertRaises(ValueError) as cm:
            StorageConfig(backend="sqlite", namespace="   ")
        self.assertIn("namespace", str(cm.exception))

    def test_backend_is_normalized(self):
        self.assertEqual(
            StorageConfig(backend=" SQLite ", namespace="u1").backend, "sqlite"
        )

    def test_path_is_expanded(self):
        config = StorageConfig(backend="sqlite", namespace="u1", path="~/data/a.db")
        self.assertTrue(os.path.isabs(config.path))
        self.assertTrue(config.path.endswith(os.path.join("data", "a.db")))

    def test_remote_backend_needs_a_url(self):
        with self.assertRaises(ValueError):
            StorageConfig(backend="qdrant", namespace="u1")

    def test_local_backend_rejects_url(self):
        with self.assertRaises(ValueError):
            StorageConfig(backend="sqlite", namespace="u1", url="http://x")

    def test_chroma_needs_path_or_url(self):
        with self.assertRaises(ValueError):
            StorageConfig(backend="chroma", namespace="u1")
        StorageConfig(backend="chroma", namespace="u1", path=self.temp_dir())
        StorageConfig(backend="chroma", namespace="u1", url="http://localhost:8000")

    def test_options_must_be_a_dict(self):
        with self.assertRaises(TypeError):
            StorageConfig(backend="sqlite", namespace="u1", options="nope")

    def test_paths_and_collection_name(self):
        folder = self.temp_dir()
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": folder}):
            config = StorageConfig(backend="sqlite", namespace="user_001")
            self.assertEqual(config.kv_path(), default_kv_path("user_001"))
            self.assertEqual(config.vector_path(), default_vector_dir("user_001"))
            self.assertIn("user_001", config.collection_name())

            explicit = StorageConfig(
                backend="sqlite", namespace="u1", path=os.path.join(folder, "x.db")
            )
            self.assertEqual(explicit.kv_path(), os.path.join(folder, "x.db"))

            named = StorageConfig(
                backend="qdrant", namespace="u1", url="http://x",
                options={"collection": "tutor"},
            )
            self.assertEqual(named.collection_name(), "tutor")
            self.assertIn("qdrant", named.describe())
            self.assertIn("u1", named.describe())


class TestCreateMemory(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.data_dir = self.temp_dir()
        patcher = mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": self.data_dir})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sqlite_backend_lands_in_the_default_dir(self):
        memory = create_memory(StorageConfig(backend="sqlite", namespace="user_001"))
        self.addCleanup(memory.close)
        self.assertIsInstance(memory, KVMemory)
        self.assertEqual(memory.path, default_kv_path("user_001"))
        self.assertTrue(os.path.isdir(os.path.dirname(memory.path)))
        memory.write("level", "中级")
        self.assertEqual(memory.read("level"), "中级")

    def test_sqlite_backend_rejects_a_folder(self):
        with self.assertRaises(ValueError) as cm:
            create_memory(
                StorageConfig(backend="sqlite", namespace="u1", path=self.temp_dir())
            )
        self.assertIn("文件夹", str(cm.exception))

    def test_jsonl_backend_is_persistent(self):
        config = StorageConfig(backend="jsonl", namespace="u1")
        memory = create_memory(config, embed_client=HashEmbedding(dimension=64))
        self.assertIsInstance(memory, VectorMemory)
        self.assertIsInstance(memory.store, FolderVectorStore)
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertTrue(memory.recall("退款", top_k=1))
        memory.close()

        again = create_memory(config, embed_client=HashEmbedding(dimension=64))
        self.addCleanup(again.close)
        self.assertEqual(again.count(), len(FolderVectorStore(config.vector_path())))
        self.assertTrue(again.recall("退款", top_k=1))

    def test_vector_backends_need_an_embedder(self):
        with self.assertRaises(ValueError) as cm:
            create_memory(StorageConfig(backend="jsonl", namespace="u1"))
        self.assertIn("embed_client", str(cm.exception))

    def test_memory_and_session_backends(self):
        self.assertIsInstance(
            create_memory(StorageConfig(backend="memory", namespace="u1")),
            InMemoryMemory,
        )
        self.assertIsInstance(
            create_memory(StorageConfig(backend="session", namespace="u1")),
            SessionMemory,
        )

    def test_layered_backend(self):
        memory = create_memory(
            StorageConfig(backend="layered", namespace="u1"),
            embed_client=HashEmbedding(dimension=64),
        )
        self.addCleanup(memory.close)
        self.assertIsInstance(memory, LayeredMemory)
        memory.write("level", "中级")
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertEqual(memory.read("level"), "中级")
        self.assertTrue(memory.recall("退款", top_k=1))

    def test_dict_config_and_kwargs_forwarding(self):
        memory = create_memory(
            {"backend": "sqlite", "namespace": "u1"}, timeout=1.5
        )
        self.addCleanup(memory.close)
        self.assertEqual(memory.timeout, 1.5)

    def test_unknown_backend_lists_the_choices(self):
        with self.assertRaises(ValueError) as cm:
            create_memory(StorageConfig(backend="redis", namespace="u1"))
        message = str(cm.exception)
        self.assertIn("sqlite", message)
        self.assertIn("register_memory_backend", message)

    def test_register_a_custom_backend(self):
        memory = InMemoryMemory(namespace="u1")
        try:
            register_memory_backend(
                "dummy", lambda config, **kwargs: memory, aliases=("dummy_alias",)
            )
            self.assertIn("dummy", BACKENDS)
            self.assertIs(
                create_memory(StorageConfig(backend="dummy", namespace="u1")), memory
            )
            self.assertIs(
                create_memory(StorageConfig(backend="dummy-alias", namespace="u1")),
                memory,
            )
        finally:
            BACKENDS.pop("dummy", None)
            BACKENDS.pop("dummy_alias", None)

    def test_chroma_backend_uses_the_client(self):
        module = self.fake_chromadb()
        config = StorageConfig(
            backend="chroma", namespace="u1", path=self.temp_dir(),
            options={"collection": "tutor"},
        )
        with mock.patch.dict(sys.modules, {"chromadb": module}):
            memory = create_memory(config, embed_client=HashEmbedding(dimension=64))
        self.addCleanup(memory.close)
        self.assertIsInstance(memory.store, ChromaVectorStore)
        self.assertEqual(memory.store.collection.name, "tutor")
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertTrue(memory.recall("退款", top_k=1))

    def test_qdrant_backend_uses_the_client(self):
        import types

        module = types.ModuleType("qdrant_client")
        module.QdrantClient = FakeQdrantClient
        module.models = FakeQdrantModels()
        config = StorageConfig(
            backend="qdrant", namespace="u1", url="http://localhost:6333",
            options={"api_key": "secret"},
        )
        with mock.patch.dict(sys.modules, {"qdrant_client": module}):
            memory = create_memory(config, embed_client=HashEmbedding(dimension=64))
        self.addCleanup(memory.close)
        self.assertIsInstance(memory.store, QdrantVectorStore)
        self.assertEqual(memory.store.client.kwargs["url"], "http://localhost:6333")
        self.assertEqual(memory.store.client.kwargs["api_key"], "secret")
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertTrue(memory.recall("退款", top_k=1))

    # ---------------- 辅助 ----------------
    @staticmethod
    def fake_chromadb():
        import types

        module = types.ModuleType("chromadb")

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_or_create_collection(self, name, metadata=None):
                collection = FakeChromaCollection()
                collection.name = name
                collection.metadata = metadata
                return collection

        module.PersistentClient = FakeClient
        module.HttpClient = FakeClient
        return module


class TestKVLifecycle(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.temp_path()

    def test_backup_is_a_usable_copy(self):
        memory = KVMemory(self.path, namespace="u1")
        memory.write("level", "中级")
        target = memory.backup(os.path.join(self.temp_dir(), "backup", "u1.db"))
        memory.close()

        self.assertTrue(os.path.isfile(target))
        restored = KVMemory(target, namespace="u1")
        self.addCleanup(restored.close)
        self.assertEqual(restored.read("level"), "中级")

    def test_export_and_import_roundtrip(self):
        memory = KVMemory(self.path, namespace="u1")
        memory.write("level", "中级")
        memory.write("topics", ["退款"])
        exported = memory.export_json(os.path.join(self.temp_dir(), "u1.json"))
        memory.clear()
        self.assertEqual(memory.count(), 0)

        self.assertEqual(memory.import_json(exported), 2)
        self.assertEqual(memory.read("level"), "中级")
        self.assertEqual(memory.read("topics"), ["退款"])
        memory.close()

    def test_import_replaces_when_asked(self):
        source = KVMemory(self.temp_path("source.db"), namespace="u1")
        source.write("level", "中级")
        exported = source.export_json(os.path.join(self.temp_dir(), "u1.json"))
        source.close()

        target = KVMemory(self.path, namespace="u1")
        self.addCleanup(target.close)
        target.write("stale", "旧值")
        target.import_json(exported, replace=True)
        self.assertEqual(target.keys(), ["level"])

    def test_import_writes_into_its_own_namespace(self):
        memory = KVMemory(self.path, namespace="u1")
        memory.write("level", "中级")
        exported = memory.export_json(os.path.join(self.temp_dir(), "u1.json"))
        memory.close()

        other = KVMemory(self.path, namespace="u2")
        self.addCleanup(other.close)
        other.import_json(exported)
        self.assertEqual(other.read("level"), "中级")
        self.assertEqual(other.namespaces(), ["u1", "u2"])

    def test_import_rejects_foreign_files(self):
        bad = os.path.join(self.temp_dir(), "bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write('{"items": "nope"}')
        memory = KVMemory(self.path)
        self.addCleanup(memory.close)
        with self.assertRaises(MemoryError):
            memory.import_json(bad)

    def test_vacuum_keeps_the_data(self):
        memory = KVMemory(self.path, namespace="u1")
        for index in range(50):
            memory.write(f"k{index}", index)
        for index in range(50):
            memory.forget(f"k{index}")
        memory.write("level", "中级")
        memory.vacuum()
        self.assertEqual(memory.items(), {"level": "中级"})
        memory.close()

    def test_destroy_removes_the_file(self):
        memory = KVMemory(self.path, namespace="u1")
        memory.write("level", "中级")
        self.assertGreater(memory.size_bytes(), 0)
        self.assertTrue(memory.destroy())
        self.assertFalse(os.path.exists(memory.path))
        self.assertFalse(memory.destroy())          # 第二次：本来就没有
        self.assertEqual(memory.size_bytes(), 0)

    def test_destroy_on_an_in_memory_database(self):
        memory = KVMemory(":memory:")
        self.assertFalse(memory.destroy())
        self.assertEqual(memory.size_bytes(), 0)

    def test_default_path_creates_parents(self):
        with mock.patch.dict(os.environ, {"PROMPTKIT_DATA_DIR": self.temp_dir()}):
            memory = KVMemory(namespace="fresh_user")
            self.addCleanup(memory.close)
            self.assertEqual(memory.path, default_kv_path("fresh_user"))
            self.assertTrue(os.path.isdir(os.path.dirname(memory.path)))
            memory.write("k", 1)
            self.assertEqual(memory.read("k"), 1)


class TestFolderVectorStore(MemoryTestCase):
    def make(self, **kwargs):
        path = self.temp_path("vectors")
        store = FolderVectorStore(path, **kwargs)
        self.addCleanup(store.close)
        return store, path

    def test_upsert_search_delete(self):
        store, _ = self.make()
        store.upsert("u1", MemoryItem(id="a#0", text="退款", vector=[1.0, 0.0],
                                      metadata={"key": "a"}))
        store.upsert("u1", MemoryItem(id="b#0", text="天气", vector=[0.0, 1.0],
                                      metadata={"key": "b"}))
        self.assertEqual(store.count("u1"), 2)
        hits = store.search("u1", [1.0, 0.0], top_k=1)
        self.assertEqual(hits[0].id, "a#0")
        self.assertEqual(store.delete("u1", key="a"), 1)
        self.assertEqual(store.count("u1"), 1)
        self.assertEqual(store.delete("u1", ids=["b#0"]), 1)
        self.assertEqual(store.count("u1"), 0)

    def test_namespaces_share_one_folder_without_mixing(self):
        store, _ = self.make()
        store.upsert("u1", MemoryItem(id="a", text="甲的", vector=[1.0]))
        store.upsert("u2", MemoryItem(id="b", text="乙的", vector=[1.0]))
        self.assertEqual(store.count("u1"), 1)
        self.assertEqual([item.id for item in store.search("u2", [1.0])], ["b"])
        store.clear("u1")
        self.assertEqual(store.count("u2"), 1)

    def test_data_survives_a_restart(self):
        first, path = self.make()
        first.upsert("u1", MemoryItem(id="a", text="退款政策", vector=[1.0, 0.0],
                                     metadata={"key": "a"}))
        first.close()

        second = FolderVectorStore(path)
        self.addCleanup(second.close)
        self.assertEqual(second.count("u1"), 1)
        self.assertEqual(second.search("u1", [1.0, 0.0])[0].text, "退款政策")

    def test_appends_then_compacts(self):
        store, path = self.make(compact_every=2)
        for index in range(4):
            store.upsert("u1", MemoryItem(id=f"n{index}", text=f"第{index}条", vector=[1.0]))
        with open(store.index_path, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 4)              # 压实过，没有重复行
        self.assertEqual(store.count("u1"), 4)

        store.flush()
        with open(store.index_path, encoding="utf-8") as handle:
            self.assertEqual(len([line for line in handle if line.strip()]), 4)

    def test_deleted_rows_do_not_come_back_after_compaction(self):
        store, path = self.make(compact_every=1)
        store.upsert("u1", MemoryItem(id="a", text="甲", vector=[1.0]))
        store.delete("u1")                           # 不给过滤条件 = 清空这个 namespace
        store.flush()
        again = FolderVectorStore(path)
        self.addCleanup(again.close)
        self.assertEqual(again.count("u1"), 0)

    def test_corrupt_lines_are_skipped(self):
        store, path = self.make()
        store.upsert("u1", MemoryItem(id="a", text="甲", vector=[1.0]))
        store.close()
        with open(store.index_path, "a", encoding="utf-8") as handle:
            handle.write("这不是 JSON\n")
            handle.write('{"op": "put"}\n')          # 缺 id
        again = FolderVectorStore(path)
        self.addCleanup(again.close)
        self.assertEqual(again.count("u1"), 1)
        self.assertEqual(again.search("u1", [1.0])[0].text, "甲")

    def test_size_and_repr(self):
        store, _ = self.make()
        store.upsert("u1", MemoryItem(id="a", text="甲", vector=[1.0]))
        self.assertGreater(store.size_bytes(), 0)
        self.assertIn("rows=1", repr(store))

    def test_vector_memory_with_a_folder(self):
        path = self.temp_path("vm-vectors")
        memory = VectorMemory(HashEmbedding(dimension=64), namespace="u1", path=path)
        memory.write("notes", "退款政策：7 天无理由退款。")
        self.assertTrue(memory.recall("退款", top_k=1))
        memory.close()

        again = VectorMemory(HashEmbedding(dimension=64), namespace="u1", path=path)
        self.addCleanup(again.close)
        self.assertTrue(again.recall("退款", top_k=1))

    def test_store_and_path_are_exclusive(self):
        with self.assertRaises(ValueError):
            VectorMemory(
                HashEmbedding(dimension=8),
                store=InMemoryVectorStore(),
                path=self.temp_path("x"),
            )

    def test_close_on_an_in_memory_store_is_a_noop(self):
        memory = VectorMemory(HashEmbedding(dimension=8))
        memory.close()                                # 不炸就行


class TestMemoryLayering(MemoryTestCase):
    def test_core_steps_do_not_import_memory_at_module_level(self):
        '''依赖方向：`core.steps` 不认识 memory 包（只在函数里延迟导入）。

        否则 `import promptkit.core` 就会把 SQLite、向量库那些一起拖进来，
        「步骤是底层」这条线就断了。
        '''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^(?:from\s+(promptkit[\w.]*)\s+import|import\s+(promptkit[\w.]*))",
            re.MULTILINE,
        )
        offenders = []
        for path in sorted((root / "core" / "steps").rglob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                dep = match.group(1) or match.group(2)
                if "promptkit.memory" in dep:
                    offenders.append(f"{path.name}: {dep}")
        self.assertEqual(offenders, [])


# ==================== 编码任务（coding） ====================
CALC = "def add(a, b):\n    return a - b\n"
FIX_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class CodingTestCase(MemoryTestCase):
    '''编码任务测试的底子：一个临时项目目录 + 读写小工具。'''

    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()

    def write(self, relative, text):
        path = os.path.join(self.root, *relative.split("/"))
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def read(self, relative):
        with open(
            os.path.join(self.root, *relative.split("/")), encoding="utf-8"
        ) as handle:
            return handle.read()

    def sandbox(self, **kwargs):
        kwargs.setdefault("allow_write", True)
        return Sandbox(self.root, **kwargs)


class CounterStep(Step):
    '''每执行一次就 +1（FixLoop 测试用）。'''

    name = "tick"

    def __init__(self, key="n"):
        self.key = key

    def execute(self, context):
        context[self.key] = context.get(self.key, 0) + 1


class TestDiffParsing(CodingTestCase):
    def test_parses_one_file(self):
        patches = parse_patch(FIX_DIFF)
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual((patch.old_path, patch.new_path), ("calc.py", "calc.py"))
        self.assertFalse(patch.is_creation)
        self.assertEqual(len(patch.hunks), 1)
        hunk = patch.hunks[0]
        self.assertEqual((hunk.old_start, hunk.old_count), (1, 2))
        self.assertEqual((hunk.new_start, hunk.new_count), (1, 2))
        self.assertEqual(hunk.old_lines(), ["def add(a, b):", "    return a - b"])
        self.assertEqual(hunk.new_lines(), ["def add(a, b):", "    return a + b"])

    def test_parses_multiple_files(self):
        text = FIX_DIFF + (
            "--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        )
        self.assertEqual([p.path for p in parse_patch(text)], ["calc.py", "other.py"])

    def test_strips_code_fences_and_prose(self):
        fenced = "好的，修改如下：\n```diff\n" + FIX_DIFF + "```\n"
        self.assertEqual(len(parse_patch(fenced)), 1)
        self.assertTrue(strip_code_fence("```\nabc\n```") == "abc")

    def test_ignores_git_style_headers(self):
        text = (
            "diff --git a/x.py b/x.py\n"
            "index 1234567..89abcde 100644\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1 +1 @@\n-a\n+b\n"
        )
        patch = parse_patch(text)[0]
        self.assertEqual(patch.path, "x.py")

    def test_creation_and_deletion(self):
        created = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
        patch = parse_patch(created)[0]
        self.assertTrue(patch.is_creation)
        self.assertEqual(patch.path, "new.py")

        deleted = "--- a/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x = 1\n"
        patch = parse_patch(deleted)[0]
        self.assertTrue(patch.is_deletion)
        self.assertEqual(patch.path, "old.py")

    def test_no_newline_marker_is_ignored(self):
        text = (
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n"
        )
        self.assertEqual(len(parse_patch(text)[0].hunks), 1)

    def test_empty_context_lines_are_tolerated(self):
        # 模型常把「空的上下文行」写成空行而不是一个空格
        text = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n"
        hunk = parse_patch(text)[0].hunks[0]
        self.assertEqual(hunk.old_lines(), ["a", "", "b"])

    def test_hunk_line_count_mismatch(self):
        text = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n a\n-b\n+c\n"
        with self.assertRaises(DiffError) as cm:
            parse_patch(text)
        self.assertIn("行数对不上", str(cm.exception))

    def test_file_without_hunks(self):
        with self.assertRaises(DiffError) as cm:
            parse_patch("--- a/x.py\n+++ b/x.py\n")
        self.assertIn("没有任何 @@ hunk", str(cm.exception))

    def test_missing_file_header(self):
        with self.assertRaises(DiffError) as cm:
            parse_patch("@@ -1 +1 @@\n-a\n+b\n")
        self.assertIn("--- 旧文件", str(cm.exception))

    def test_empty_patch(self):
        for text in ("", "   ", "```\n```"):
            with self.assertRaises(DiffError):
                parse_patch(text)

    def test_unsafe_paths_are_rejected(self):
        for header in ("/etc/passwd", "C:/Windows/x.py", "a/../../x.py"):
            text = f"--- {header}\n+++ {header}\n@@ -1 +1 @@\n-a\n+b\n"
            with self.assertRaises(DiffError):
                validate_patch(text)

    def test_path_checks_can_be_skipped(self):
        text = "--- /abs/x.py\n+++ /abs/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(len(validate_patch(text, check_paths=False)), 1)

    def test_file_patch_repr(self):
        self.assertIn("FilePatch", repr(parse_patch(FIX_DIFF)[0]))


class TestDiffApply(CodingTestCase):
    def test_replaces_lines(self):
        self.assertEqual(apply_patch_text(CALC, FIX_DIFF), "def add(a, b):\n    return a + b\n")

    def test_line_numbers_can_be_off(self):
        source = "# 头部注释\n# 又一行\n\n" + CALC
        text = (
            "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n-    return a - b\n+    return a + b\n"
        )
        result = apply_patch_text(source, text)
        self.assertIn("return a + b", result)
        self.assertTrue(result.startswith("# 头部注释"))

    def test_multiple_hunks(self):
        source = "a = 1\nb = 2\nc = 3\nd = 4\n"
        text = (
            "--- a/x.py\n+++ b/x.py\n"
            "@@ -1,1 +1,1 @@\n-a = 1\n+a = 10\n"
            "@@ -4,1 +4,1 @@\n-d = 4\n+d = 40\n"
        )
        self.assertEqual(apply_patch_text(source, text), "a = 10\nb = 2\nc = 3\nd = 40\n")

    def test_insertion_into_empty_file(self):
        text = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
        self.assertEqual(apply_patch_text("", text), "x = 1\ny = 2\n")

    def test_deletion_to_empty(self):
        text = "--- a/x.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x = 1\n-y = 2\n"
        self.assertEqual(apply_patch_text("x = 1\ny = 2\n", text), "\n")

    def test_missing_context_raises(self):
        with self.assertRaises(DiffError) as cm:
            apply_patch_text("完全不同\n", FIX_DIFF, path="calc.py")
        self.assertIn("上下文找不到", str(cm.exception))
        self.assertIn("calc.py", str(cm.exception))

    def test_trailing_newline_is_preserved(self):
        self.assertTrue(apply_patch_text("a\n", "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n").endswith("b\n"))
        self.assertEqual(
            apply_patch_text("a", "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"), "b"
        )

    def test_crlf_source_is_normalized(self):
        result = apply_patch_text("a\r\nb\r\n", "--- a/x\n+++ b/x\n@@ -2 +2 @@\n-b\n+c\n")
        self.assertEqual(result, "a\nc\n")

    def test_apply_hunks_directly(self):
        hunk = parse_patch(FIX_DIFF)[0].hunks[0]
        self.assertEqual(apply_hunks(CALC, [hunk]), "def add(a, b):\n    return a + b\n")

    def test_empty_patch_text_is_rejected(self):
        with self.assertRaises(DiffError):
            apply_patch_text(CALC, "   ")


class TestSandbox(CodingTestCase):
    def test_read_and_write_inside_base(self):
        sandbox = self.sandbox()
        self.write("calc.py", CALC)
        self.assertEqual(sandbox.read_text("calc.py"), CALC)
        sandbox.write_text("sub/new.py", "x = 1\n")
        self.assertEqual(self.read("sub/new.py"), "x = 1\n")

    def test_write_is_denied_by_default(self):
        sandbox = Sandbox(self.root)
        with self.assertRaises(SandboxError):
            sandbox.write_text("x.py", "x = 1")

    def test_escapes_are_blocked(self):
        sandbox = self.sandbox()
        for path in ("../x.py", "a/../../x.py", "/etc/passwd", "C:/Windows/x.py"):
            with self.assertRaises(SandboxError):
                sandbox.resolve(path)

    def test_sensitive_files_are_blocked(self):
        sandbox = self.sandbox()
        self.write(".env", "SECRET=1")
        for path in (".env", ".env.local"):
            with self.assertRaises(SandboxError):
                sandbox.read_text(path)
        with self.assertRaises(SandboxError):
            sandbox.write_text("id_rsa", "x")

    def test_sensitive_files_can_be_whitelisted(self):
        self.write(".env.example", "KEY=xxx")
        sandbox = self.sandbox(allow=[".env.example"])
        self.assertEqual(sandbox.read_text(".env.example"), "KEY=xxx")

    def test_read_is_truncated(self):
        self.write("big.txt", "甲" * 50)
        sandbox = Sandbox(self.root, max_read_chars=10)
        text = sandbox.read_text("big.txt")
        self.assertTrue(text.startswith("甲" * 10))
        self.assertIn("已截断", text)

    def test_missing_file_message(self):
        with self.assertRaises(SandboxError) as cm:
            self.sandbox().read_text("nope.py")
        self.assertIn("不存在", str(cm.exception))

    def test_list_dir_skips_noise_and_filters(self):
        self.write("a.py", "x = 1\n")
        self.write("b.txt", "hello\n")
        self.write("pkg/c.py", "y = 2\n")
        os.makedirs(os.path.join(self.root, "__pycache__"), exist_ok=True)
        sandbox = self.sandbox()
        files = sandbox.list_dir()
        self.assertIn("a.py", files)
        self.assertIn("pkg/c.py", files)
        self.assertNotIn("__pycache__", " ".join(files))
        self.assertEqual(sandbox.list_dir(pattern="*.py"), ["a.py", "pkg/c.py"])

    def test_search(self):
        self.write("calc.py", "def add(a, b):\n    return a - b\n")
        sandbox = self.sandbox()
        hits = sandbox.search("def add")
        self.assertEqual(hits[0]["path"], "calc.py")
        self.assertEqual(hits[0]["line"], 1)
        # `add(a` 不是合法正则（括号没闭合），这时按普通字符串搜
        self.assertEqual(sandbox.search("add(a")[0]["path"], "calc.py")

    def test_command_allowlist(self):
        sandbox = Sandbox(self.root, allow_commands=["python"])
        sandbox.check_command("python -c \"print(1)\"")
        with self.assertRaises(CommandError):
            sandbox.check_command("ls")
        with self.assertRaises(CommandError):
            sandbox.check_command("python -c \"print(1)\" ; rm -rf /")

    def test_audit_log(self):
        self.write("calc.py", CALC)
        sandbox = self.sandbox()
        with self.assertLogs("promptkit.coding", level="INFO") as captured:
            sandbox.read_text("calc.py")
            sandbox.write_text("calc.py", CALC)
        self.assertIn("read calc.py", "\n".join(captured.output))

    def test_repr(self):
        self.assertIn("只读", repr(Sandbox(self.root)))
        self.assertIn("读写", repr(self.sandbox()))


class TestCodeFormats(CodingTestCase):
    def test_diff_output_passes_structure(self):
        self.assertEqual(get_output("diff").name, "diff")
        self.assertEqual(DiffOutput().parse(FIX_DIFF).strip(), FIX_DIFF.strip())

    def test_diff_output_dry_run_success(self):
        self.write("calc.py", CALC)
        text = DiffOutput().parse(FIX_DIFF, sandbox=self.sandbox(), path="calc.py")
        self.assertIn("return a + b", text)

    def test_diff_output_dry_run_catches_misalignment(self):
        self.write("calc.py", "完全不同\n")
        with self.assertRaises(CodeFormatError) as cm:
            DiffOutput().parse(FIX_DIFF, sandbox=self.sandbox(), path="calc.py")
        self.assertIn("干跑失败", str(cm.exception))

    def test_diff_output_requires_existing_file_unless_creation(self):
        with self.assertRaises(CodeFormatError) as cm:
            DiffOutput().parse(
                FIX_DIFF, sandbox=self.sandbox(), path="missing.py"
            )
        self.assertIn("读不到", str(cm.exception))

    def test_full_file_output(self):
        code = FullFileOutput().parse("```python\ndef f():\n    return 1\n```")
        self.assertEqual(code, "def f():\n    return 1\n")

    def test_full_file_syntax_error(self):
        with self.assertRaises(CodeFormatError) as cm:
            FullFileOutput().parse("def f(:\n    pass\n")
        self.assertIn("语法错误", str(cm.exception))

    def test_full_file_custom_syntax_check(self):
        output = FullFileOutput(
            language="js", syntax_check=lambda text: (_ for _ in ()).throw(ValueError("坏代码"))
        )
        with self.assertRaises(CodeFormatError):
            output.parse("function f() {}")
        # 非 python 且没有 checker：不做语法检查
        self.assertEqual(
            FullFileOutput(language="js").parse("function f() {}\n"), "function f() {}\n"
        )

    def test_changes_output(self):
        import json as _json

        raw = _json.dumps(
            {
                "changes": [
                    {"path": "a.py", "action": "create", "content": "x = 1\n"},
                    {"path": "b.py", "action": "delete"},
                ]
            }
        )
        data = ChangesOutput().parse(raw)
        self.assertEqual(len(data["changes"]), 2)
        self.assertEqual(data["changes"][0]["path"], "a.py")

    def test_changes_output_accepts_a_bare_list(self):
        import json as _json

        raw = _json.dumps([{"path": "a.py", "content": "x = 1\n"}])
        self.assertEqual(ChangesOutput().parse(raw)["changes"][0]["action"], "modify")

    def test_changes_output_validation(self):
        import json as _json

        cases = [
            (_json.dumps({"changes": []}), "changes 是空的"),
            (_json.dumps({"changes": [{"action": "create", "content": "x"}]}), "path"),
            (
                _json.dumps({"changes": [{"path": "a.py", "action": "rename", "content": "x"}]}),
                "不合法",
            ),
            (
                _json.dumps({"changes": [{"path": "a.py", "action": "create"}]}),
                "必须给 content",
            ),
            (
                _json.dumps(
                    {"changes": [
                        {"path": "a.py", "action": "create", "content": "x"},
                        {"path": "a.py", "action": "delete"},
                    ]}
                ),
                "出现了两次",
            ),
            ("不是 JSON", "JSON"),
        ]
        for raw, expected in cases:
            with self.assertRaises(CodeFormatError) as cm:
                ChangesOutput().parse(raw)
            self.assertIn(expected, str(cm.exception))

    def test_changes_modify_requires_existing_file_when_sandbox_given(self):
        import json as _json

        raw = _json.dumps(
            {"changes": [{"path": "nope.py", "action": "modify", "content": "x = 1\n"}]}
        )
        self.assertEqual(len(ChangesOutput().parse(raw)["changes"]), 1)  # 没沙箱不查
        with self.assertRaises(CodeFormatError):
            ChangesOutput().parse(raw, sandbox=self.sandbox())

    def test_unknown_format(self):
        with self.assertRaises(ValueError) as cm:
            get_output("yaml")
        self.assertIn("diff", str(cm.exception))


class TestCodingStep(CodingTestCase):
    def test_writes_the_validated_patch(self):
        self.write("calc.py", CALC)
        agent = FakeAgent([FIX_DIFF])
        ctx = Workflow(
            CodingStep(
                task="把减法改成加法",
                context={"calc.py": CALC},
                path="calc.py",
                base_dir=self.root,
            )
        ).run(agent=agent)
        self.assertIn("return a + b", ctx["patch"])
        self.assertEqual(ctx.trace[-1], "coding:1")
        self.assertEqual(len(agent.prompts), 1)

    def test_prompt_contains_task_files_error_and_format(self):
        agent = FakeAgent([FIX_DIFF])
        ctx_workflow = Workflow(
            CodingStep(
                task=Var("task"),
                context={"src/calc.py": CALC},
                error=Var("last_error"),
                cautions=[Custom("不要修改测试文件")],
                base_dir=self.root,
            )
        )
        ctx_workflow.run(
            agent=agent, task="修复 add", last_error={"output": "AssertionError: 2 != 5"}
        )
        prompt = prompt_text(agent.prompts[0])
        self.assertIn("修复 add", prompt)
        self.assertIn("src/calc.py", prompt)
        self.assertIn("return a - b", prompt)
        self.assertIn("AssertionError: 2 != 5", prompt)
        self.assertIn("unified diff", prompt)
        self.assertIn("不要修改测试文件", prompt)

    def test_retries_after_bad_patch(self):
        self.write("calc.py", CALC)
        agent = FakeAgent(["这不是补丁", FIX_DIFF])
        ctx = Workflow(
            CodingStep(task="改", path="calc.py", base_dir=self.root)
        ).run(agent=agent)
        self.assertIn("return a + b", ctx["patch"])
        self.assertEqual(len(agent.prompts), 2)
        self.assertIn("你上一次的输出不能用", prompt_text(agent.prompts[1]))
        self.assertEqual(ctx.trace[-1], "coding:2")

    def test_gives_up_after_max_retries(self):
        self.write("calc.py", CALC)
        agent = FakeAgent(["坏", "还是坏"])
        with self.assertRaises(CodeFormatError) as cm:
            Workflow(
                CodingStep(task="改", path="calc.py", base_dir=self.root, max_retries=1)
            ).run(agent=agent)
        self.assertIn("连续 2 次", str(cm.exception))
        self.assertEqual(agent.replies, [])

    def test_dry_run_rejects_a_misaligned_patch(self):
        self.write("calc.py", "完全不同\n")
        agent = FakeAgent([FIX_DIFF, FIX_DIFF])
        with self.assertRaises(CodeFormatError):
            Workflow(
                CodingStep(task="改", path="calc.py", base_dir=self.root, max_retries=1)
            ).run(agent=agent)
        self.assertIn("干跑失败", prompt_text(agent.prompts[1]))

    def test_verify_can_be_disabled(self):
        self.write("calc.py", "完全不同\n")
        ctx = Workflow(
            CodingStep(task="改", path="calc.py", base_dir=self.root, verify=False)
        ).run(agent=FakeAgent([FIX_DIFF]))
        self.assertIn("return a + b", ctx["patch"])

    def test_full_file_format(self):
        agent = FakeAgent(["```python\ndef f():\n    return 1\n```"])
        ctx = Workflow(
            CodingStep(task="写个函数", output_format="full_file", output_key="code")
        ).run(agent=agent)
        self.assertEqual(ctx["code"], "def f():\n    return 1\n")
        self.assertIn("完整的文件内容", prompt_text(agent.prompts[0]))

    def test_json_format(self):
        import json as _json

        agent = FakeAgent(
            [_json.dumps({"changes": [{"path": "a.py", "action": "create", "content": "x = 1\n"}]})]
        )
        ctx = Workflow(
            CodingStep(task="新建文件", output_format="json", output_key="changes",
                        base_dir=self.root)
        ).run(agent=agent)
        self.assertEqual(ctx["changes"]["changes"][0]["path"], "a.py")

    def test_custom_output_object(self):
        output = FullFileOutput(language="js")
        ctx = Workflow(
            CodingStep(task="写 js", output=output, output_key="code")
        ).run(agent=FakeAgent(["function f() {}\n"]))
        self.assertEqual(ctx["code"], "function f() {}\n")
        self.assertEqual(Workflow(CodingStep(task="x", output=output)).run(
            agent=FakeAgent(["x\n"])
        )["patch"], "x\n")

    def test_needs_an_agent(self):
        Agent.set_default(None)
        with self.assertRaises(ValueError) as cm:
            Workflow(CodingStep(task="x")).run()
        self.assertIn("agent", str(cm.exception))

    def test_validation(self):
        with self.assertRaises(ValueError):
            CodingStep(task="x", max_retries=-1)
        with self.assertRaises(TypeError):
            CodingStep(task="x", output=object())
        with self.assertRaises(ValueError):
            CodingStep(task="x", output_format="yaml")

    def test_describe(self):
        self.assertIn("Coding(diff", CodingStep(task="改一行").describe())
        self.assertIn("-> patch", CodingStep(task="改一行").describe())
        self.assertIn("src/a.py", CodingStep(task="改", path="src/a.py").describe())


class TestCodingIO(CodingTestCase):
    def test_read_file_step(self):
        self.write("calc.py", CALC)
        ctx = Workflow(
            ReadFileStep("calc.py", base_dir=self.root, max_chars=10)
        ).run()
        self.assertIn("已截断", ctx["source"])

        ctx = Workflow(
            ReadFileStep(Var("p"), output_key="code", base_dir=self.root)
        ).run(p="calc.py")
        self.assertEqual(ctx["code"], CALC)

    def test_read_file_step_blocks_escape(self):
        with self.assertRaises(SandboxError):
            Workflow(ReadFileStep("../x.py", base_dir=self.root)).run()

    def test_write_file_step(self):
        ctx = Workflow(
            WriteFileStep("out/a.txt", content=Var("text"), output_key="written",
                          base_dir=self.root)
        ).run(text="你好")
        self.assertEqual(self.read("out/a.txt"), "你好")
        self.assertEqual(ctx["written"]["path"], "out/a.txt")
        self.assertEqual(ctx["written"]["chars"], 2)

    def test_apply_diff_single_file(self):
        self.write("calc.py", CALC)
        ctx = Workflow(
            ApplyDiff("calc.py", diff=Var("patch"), base_dir=self.root)
        ).run(patch=FIX_DIFF)
        self.assertEqual(self.read("calc.py"), "def add(a, b):\n    return a + b\n")
        self.assertEqual(ctx["apply_result"]["files"], ["calc.py"])
        self.assertEqual(ctx["apply_result"]["hunks"], 1)

    def test_apply_diff_keyword_path(self):
        self.write("calc.py", CALC)
        Workflow(
            ApplyDiff(path="calc.py", diff=FIX_DIFF, base_dir=self.root)
        ).run()
        self.assertIn("return a + b", self.read("calc.py"))

    def test_apply_diff_multi_file(self):
        self.write("a.py", "x = 1\n")
        self.write("b.py", "y = 2\n")
        patch = (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 10\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y = 2\n+y = 20\n"
        )
        ctx = Workflow(ApplyDiff(diff=patch, base_dir=self.root)).run()
        self.assertEqual(sorted(ctx["apply_result"]["files"]), ["a.py", "b.py"])
        self.assertEqual(self.read("a.py"), "x = 10\n")
        self.assertEqual(self.read("b.py"), "y = 20\n")

    def test_apply_diff_creates_and_deletes(self):
        self.write("old.py", "x = 1\n")
        patch = (
            "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
            "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
        )
        ctx = Workflow(ApplyDiff(diff=patch, base_dir=self.root)).run()
        self.assertEqual(ctx["apply_result"]["created"], ["new.py"])
        self.assertEqual(ctx["apply_result"]["deleted"], ["old.py"])
        self.assertTrue(os.path.exists(os.path.join(self.root, "new.py")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "old.py")))

    def test_apply_diff_backup(self):
        self.write("calc.py", CALC)
        Workflow(
            ApplyDiff("calc.py", diff=FIX_DIFF, base_dir=self.root, backup=True)
        ).run()
        self.assertEqual(self.read("calc.py.orig"), CALC)

    def test_apply_diff_failure_leaves_file_untouched(self):
        self.write("calc.py", "完全不同\n")
        with self.assertRaises(DiffError):
            Workflow(ApplyDiff("calc.py", diff=FIX_DIFF, base_dir=self.root)).run()
        self.assertEqual(self.read("calc.py"), "完全不同\n")

    def test_apply_diff_needs_a_diff(self):
        with self.assertRaises(ValueError):
            ApplyDiff("calc.py")

    def test_apply_changes(self):
        self.write("gone.py", "x = 1\n")
        changes = {
            "changes": [
                {"path": "a.py", "action": "create", "content": "x = 1\n"},
                {"path": "gone.py", "action": "delete"},
            ]
        }
        ctx = Workflow(
            ApplyChanges(Var("changes"), base_dir=self.root)
        ).run(changes=changes)
        self.assertEqual(ctx["apply_result"]["files"], ["a.py"])
        self.assertEqual(ctx["apply_result"]["deleted"], ["gone.py"])
        self.assertEqual(self.read("a.py"), "x = 1\n")
        self.assertFalse(os.path.exists(os.path.join(self.root, "gone.py")))

    def test_list_dir_and_search_code_steps(self):
        self.write("pkg/a.py", "def calculate():\n    return 1\n")
        self.write("pkg/b.txt", "hello\n")
        ctx = Workflow(
            ListDir("pkg", pattern="*.py", output_key="files", base_dir=self.root),
            SearchCode("def calculate", output_key="hits", base_dir=self.root),
        ).run()
        self.assertEqual(ctx["files"], ["pkg/a.py"])
        self.assertEqual(ctx["hits"][0]["path"], "pkg/a.py")

    def test_apply_patch_helper(self):
        self.write("calc.py", CALC)
        result = apply_patch(self.sandbox(), FIX_DIFF, path="calc.py")
        self.assertEqual(result["files"], ["calc.py"])
        self.assertIn("return a + b", self.read("calc.py"))


class TestCommandExecution(CodingTestCase):
    def test_run_command_success(self):
        result = coding_run_command('python -c "print(42)"')
        self.assertTrue(result["passed"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["output"].strip(), "42")
        self.assertFalse(result["timed_out"])

    def test_run_command_failure(self):
        result = coding_run_command('python -c "import sys; sys.exit(3)"')
        self.assertFalse(result["passed"])
        self.assertEqual(result["returncode"], 3)
        self.assertIn("退出码 3", result["output"])

    def test_run_command_timeout(self):
        result = coding_run_command('python -c "import time; time.sleep(5)"', timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["passed"])
        self.assertIn("超时", result["output"])

    def test_run_command_empty(self):
        with self.assertRaises(CommandError):
            coding_run_command("   ")

    def test_output_truncation(self):
        result = coding_run_command(
            'python -c "print(1234567890)"', max_output_chars=5
        )
        self.assertIn("已截断", result["output"])

    def test_run_tests_step(self):
        ctx = Workflow(
            RunTests('python -c "print(1)"', base_dir=self.root)
        ).run()
        self.assertTrue(ctx["test_result"]["passed"])
        self.assertEqual(ctx["test_result"]["command"], 'python -c "print(1)"')

    def test_run_tests_defaults(self):
        step = RunTests("x")
        self.assertEqual(step.output_key, "test_result")
        self.assertEqual(step.timeout, 600.0)
        self.assertEqual(step.name, "run_tests")
        self.assertEqual(LintStep("y").output_key, "lint_result")

    def test_allowlist_is_enforced(self):
        with self.assertRaises(CommandError):
            Workflow(
                RunCommand("echo hi", allowlist=["python"], base_dir=self.root)
            ).run()

    def test_shell_metacharacters_are_rejected_with_allowlist(self):
        with self.assertRaises(CommandError):
            coding_run_command(
                'python -c "print(1)" ; echo x',
                sandbox=Sandbox(self.root, allow_commands=["python"]),
            )

    def test_cwd(self):
        os.makedirs(os.path.join(self.root, "sub"), exist_ok=True)
        ctx = Workflow(
            RunCommand(
                'python -c "import os; print(os.path.basename(os.getcwd()))"',
                cwd="sub",
                base_dir=self.root,
            )
        ).run()
        self.assertIn("sub", ctx["command_result"]["output"])

    def test_cwd_escape_is_blocked(self):
        with self.assertRaises(SandboxError):
            Workflow(
                RunCommand('python -c "print(1)"', cwd="..", base_dir=self.root)
            ).run()


class TestFixLoop(CodingTestCase):
    def test_stops_when_condition_holds(self):
        ctx = Workflow(
            FixLoop([CounterStep()], until=Var("n") >= 3, max_rounds=5)
        ).run()
        self.assertEqual(ctx["n"], 3)
        self.assertEqual(ctx["fix_round"], 3)
        self.assertFalse(ctx["fix_exhausted"])
        self.assertIn("fix_loop:done", ctx.trace)

    def test_marks_exhausted(self):
        ctx = Workflow(
            FixLoop([CounterStep()], until=Var("n") >= 10, max_rounds=2)
        ).run()
        self.assertEqual(ctx["n"], 2)
        self.assertTrue(ctx["fix_exhausted"])
        self.assertIn("fix_loop:exhausted", ctx.trace)

    def test_can_raise_instead(self):
        with self.assertRaises(LoopLimitExceeded):
            Workflow(
                FixLoop(
                    [CounterStep()],
                    until=Var("n") >= 10,
                    max_rounds=2,
                    on_exhausted="raise",
                )
            ).run()

    def test_break_stops_without_marking_exhausted(self):
        ctx = Workflow(
            FixLoop([CounterStep(), Break()], until=Var("n") >= 10, max_rounds=3)
        ).run()
        self.assertEqual(ctx["n"], 1)
        self.assertFalse(ctx["fix_exhausted"])
        self.assertIn("fix_loop:done", ctx.trace)

    def test_keys_can_be_disabled(self):
        ctx = Workflow(
            FixLoop(
                [CounterStep()],
                until=Var("n") >= 1,
                round_key=None,
                exhausted_key=None,
            )
        ).run()
        self.assertNotIn("fix_round", ctx)
        self.assertNotIn("fix_exhausted", ctx)

    def test_do_chaining(self):
        step = FixLoop(until=Var("n") >= 2, max_rounds=3).Do(CounterStep())
        ctx = Workflow(step).run()
        self.assertEqual(ctx["n"], 2)

    def test_empty_body_is_reported_at_run_time(self):
        with self.assertRaises(ValueError) as cm:
            Workflow(FixLoop(until=True)).run()
        self.assertIn("body 是空的", str(cm.exception))

    def test_tree(self):
        step = FixLoop([CounterStep()], until=Var("n") >= 1)
        self.assertEqual(len(step.children()), 1)
        self.assertIn("FixLoop", step.describe())
        self.assertIn("tick", step.render())

    def test_validation(self):
        for kwargs in (
            {"body": [CounterStep()]},                                   # 没有 until
            {"body": [CounterStep()], "until": True, "max_rounds": 0},
            {"body": [CounterStep()], "until": True, "on_exhausted": "explode"},
        ):
            with self.assertRaises((ValueError, TypeError)):
                FixLoop(**kwargs)


class TestGitSteps(CodingTestCase):
    def setUp(self):
        super().setUp()
        import shutil

        if shutil.which("git") is None:
            self.skipTest("没有 git")
        self.git = lambda *args: git(list(args), cwd=self.root)
        self.assertEqual(self.git("init", "-q")["returncode"], 0)
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "tester")
        self.write("calc.py", CALC)
        self.git("add", "-A")
        self.git("commit", "-m", "init")

    def test_git_diff_shows_changes(self):
        self.write("calc.py", "def add(a, b):\n    return a + b\n")
        ctx = Workflow(GitDiff(output_key="diff", base_dir=self.root)).run()
        self.assertIn("-    return a - b", ctx["diff"])
        self.assertIn("+    return a + b", ctx["diff"])
        self.assertTrue(ctx["diff_result"]["passed"])

    def test_git_commit(self):
        self.write("calc.py", "def add(a, b):\n    return a + b\n")
        ctx = Workflow(
            GitCommit(message="fix add", base_dir=self.root)
        ).run()
        self.assertTrue(ctx["commit_result"]["passed"])
        log = self.git("log", "--oneline")
        self.assertIn("fix add", log["output"])
        self.assertIn("init", log["output"])

    def test_commit_message_is_required(self):
        with self.assertRaises(CommandError):
            Workflow(GitCommit(message="  ", base_dir=self.root)).run()

    def test_empty_repo_has_empty_diff(self):
        other = self.temp_dir()
        self.assertEqual(git(["init", "-q"], cwd=other)["returncode"], 0)
        ctx = Workflow(GitDiff(base_dir=other)).run()
        self.assertEqual(ctx["git_diff"], "")
        self.assertTrue(ctx["git_diff_result"]["passed"])


class TestCodingTools(CodingTestCase):
    def test_read_file_tool(self):
        self.write("calc.py", CALC)
        tool = ReadFileTool(base_dir=self.root)
        self.assertEqual(tool.run("calc.py"), CALC)
        self.assertEqual(tool.call({"path": "calc.py"}), CALC)
        self.assertTrue(tool.run("../secret").startswith("错误："))

    def test_write_file_tool_is_read_only_by_default(self):
        tool = WriteFileTool(base_dir=self.root)
        self.assertTrue(tool.run("x.py", "x = 1\n").startswith("错误："))
        self.assertFalse(os.path.exists(os.path.join(self.root, "x.py")))

        writable = WriteFileTool(base_dir=self.root, allow_write=True)
        self.assertIn("已写入", writable.run("x.py", "x = 1\n"))
        self.assertEqual(self.read("x.py"), "x = 1\n")

    def test_apply_diff_tool(self):
        self.write("calc.py", CALC)
        tool = ApplyDiffTool(base_dir=self.root, allow_write=True)
        result = tool.run(FIX_DIFF)
        self.assertEqual(result["files"], ["calc.py"])
        self.assertIn("return a + b", self.read("calc.py"))

        readonly = ApplyDiffTool(base_dir=self.root)
        self.write("calc.py", CALC)
        self.assertTrue(readonly.run(FIX_DIFF).startswith("错误："))

    def test_list_and_search_tools(self):
        self.write("pkg/a.py", "def calculate():\n    return 1\n")
        self.assertIn("pkg/a.py", ListDirTool(base_dir=self.root).run("pkg"))
        hits = SearchCodeTool(base_dir=self.root).run("def calculate")
        self.assertIn("pkg/a.py:1:", hits)

    def test_run_command_tool_and_allowlist(self):
        tool = RunCommandTool(base_dir=self.root, allow_commands=["python"])
        result = tool.run('python -c "print(7)"')
        self.assertTrue(result["passed"])
        self.assertEqual(result["output"].strip(), "7")
        self.assertTrue(tool.run("ls").startswith("错误："))

    def test_run_tests_tool(self):
        tool = RunTestsTool(
            base_dir=self.root,
            command='python -c "print(1)"',
            allow_commands=["python"],
        )
        self.assertIn("通过", tool.run())

    def test_default_coding_tools(self):
        self.write("calc.py", CALC)
        tools = default_coding_tools(self.root)
        self.assertEqual(
            [tool.name for tool in tools],
            ["read_file", "list_dir", "search_code", "apply_diff", "run_tests"],
        )
        self.assertTrue(tools[0].call({"path": "calc.py"}).startswith("def add"))
        self.assertTrue(tools[3].call({"diff": FIX_DIFF}).startswith("错误："))  # 只读

    def test_default_coding_tools_can_write(self):
        self.write("calc.py", CALC)
        tools = default_coding_tools(self.root, allow_write=True)
        tools[3].call({"diff": FIX_DIFF})
        self.assertIn("return a + b", self.read("calc.py"))

    def test_tool_schemas(self):
        for tool in default_coding_tools(self.root):
            function = tool.schema()["function"]
            self.assertEqual(function["name"], tool.name)
            self.assertIn("parameters", function)

    def test_tool_audit_log(self):
        self.write("calc.py", CALC)
        with self.assertLogs("promptkit.coding", level="INFO"):
            ReadFileTool(base_dir=self.root).run("calc.py")


class TestCodeSearch(CodingTestCase):
    '''代码搜索：上下文、每文件上限、files_only、排除、预算。'''

    SOURCE = (
        "alpha = 1\n"
        "target = 2\n"
        "beta = 3\n"
        "target = 4\n"
        "gamma = 5\n"
        "target = 6\n"
        "delta = 7\n"
    )

    def test_context_lines_are_included(self):
        self.write("mod.py", self.SOURCE)
        hits = Sandbox(self.root).search("target", context=1)
        self.assertEqual([hit["line"] for hit in hits], [2, 4, 6])
        first = hits[0]
        self.assertEqual(first["before"], ["alpha = 1"])
        self.assertEqual(first["after"], ["beta = 3"])

    def test_context_is_clipped_at_file_edges(self):
        self.write("edge.py", "target\n")
        hit = Sandbox(self.root).search("target", context=3)[0]
        self.assertEqual(hit["before"], [])
        self.assertEqual(hit["after"], [])

    def test_max_per_file(self):
        self.write("mod.py", self.SOURCE)
        hits = Sandbox(self.root).search("target", max_per_file=2)
        self.assertEqual(len(hits), 2)

    def test_files_only_returns_counts(self):
        self.write("a.py", "target = 1\ntarget = 2\n")
        self.write("b.py", "target = 3\n")
        files = Sandbox(self.root).search("target", files_only=True)
        self.assertEqual(
            {item["path"]: item["matches"] for item in files},
            {"a.py": 2, "b.py": 1},
        )

    def test_exclude_globs(self):
        self.write("src/a.py", "target\n")
        self.write("tests/b.py", "target\n")
        hits = Sandbox(self.root).search("target", exclude=("tests/*",))
        self.assertEqual([hit["path"] for hit in hits], ["src/a.py"])

    def test_file_budget(self):
        for index in range(5):
            self.write(f"m{index}.py", "target\n")
        hits = Sandbox(self.root, max_search_files=2).search("target")
        self.assertEqual(len(hits), 2)

    def test_hit_budget(self):
        self.write("mod.py", "target\n" * 20)
        hits = Sandbox(self.root, max_search_hits=3, max_per_file=10).search("target")
        self.assertEqual(len(hits), 3)

    def test_long_lines_are_truncated(self):
        self.write("long.py", "x" * 400 + "target\n")
        hit = Sandbox(self.root, max_line_chars=20).search("target")[0]
        self.assertLessEqual(len(hit["text"]), 20)

    def test_empty_query(self):
        with self.assertRaises(SandboxError):
            Sandbox(self.root).search("   ")

    def test_invalid_regex_falls_back_to_literal(self):
        self.write("mod.py", "compute(a, b)\n")
        hits = Sandbox(self.root).search("compute(a")
        self.assertEqual(hits[0]["path"], "mod.py")

    def test_format_hits_text(self):
        self.write("mod.py", self.SOURCE)
        hits = Sandbox(self.root).search("target", context=1)
        text = format_hits(hits)
        self.assertIn("mod.py:2: target = 2", text)
        self.assertIn("| alpha = 1", text)
        self.assertEqual(format_hits([]), "（没有匹配）")

    def test_format_hits_for_files_only(self):
        self.write("a.py", "target\n")
        text = format_hits(Sandbox(self.root).search("target", files_only=True))
        self.assertEqual(text, "a.py（1 处）")

    def test_search_code_step_with_text_and_context(self):
        self.write("mod.py", self.SOURCE)
        ctx = Workflow(
            SearchCode(
                "target",
                base_dir=self.root,
                context=1,
                as_text=True,
                output_key="where",
            )
        ).run()
        self.assertIn("mod.py:4: target = 4", ctx["where"])
        self.assertIn("| beta = 3", ctx["where"])

    def test_search_code_step_structured_is_the_default(self):
        self.write("mod.py", self.SOURCE)
        ctx = Workflow(
            SearchCode("target", base_dir=self.root, max_per_file=1)
        ).run()
        self.assertEqual(ctx["hits"][0]["path"], "mod.py")
        self.assertEqual(ctx["hits"][0]["line"], 2)
        # 结构化结果可以直接喂给 ReadFileStep 式的取值表达式
        self.assertEqual(Var("hits")[0]["path"].eval(ctx), "mod.py")

    def test_search_code_step_files_only(self):
        self.write("a.py", "target\n")
        ctx = Workflow(
            SearchCode("target", base_dir=self.root, files_only=True)
        ).run()
        self.assertEqual(ctx["hits"], [{"path": "a.py", "matches": 1}])

    def test_search_tool_exposes_context(self):
        self.write("mod.py", self.SOURCE)
        tool = SearchCodeTool(base_dir=self.root)
        text = tool.call({"query": "target", "context": 1})
        self.assertIn("mod.py:2: target = 2", text)
        self.assertIn("| alpha = 1", text)
        schema = tool.schema()["function"]["parameters"]["properties"]
        self.assertIn("context", schema)

    def test_search_tool_is_sandboxed(self):
        outside = self.temp_dir()
        with open(os.path.join(outside, "secret.py"), "w", encoding="utf-8") as handle:
            handle.write("target\n")
        tool = SearchCodeTool(base_dir=self.root)
        self.assertEqual(tool.call({"query": "target"}), "（没有匹配）")


class TestCodingLayering(CodingTestCase):
    def test_core_does_not_import_coding(self):
        '''`coding` 是上层：core 不能反过来依赖它（否则 import core 就拖进沙箱/进程）。'''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^(?:from\s+(promptkit[\w.]*)\s+import|import\s+(promptkit[\w.]*))",
            re.MULTILINE,
        )
        offenders = []
        for path in sorted((root / "core").rglob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                dep = match.group(1) or match.group(2)
                if "promptkit.coding" in dep:
                    offenders.append(f"{path.name}: {dep}")
        self.assertEqual(offenders, [])


# ==================== 多代理编排（orchestrate） ====================
import time  # noqa: E402  （并发 / 超时测试要用）


class OrchAgent:
    '''编排测试用的假 agent：按顺序回话、可选发 USAGE、可选慢、可选失败。'''

    def __init__(self, replies=None, *, tokens=0, delay=0.0, fail_times=0):
        self.replies = list(replies or ["好的"])
        self.tokens = tokens
        self.delay = delay
        self.fail_times = fail_times
        self.prompts = []
        self.tools_seen = []
        self.resets = 0
        self.calls = 0

    def reset(self):
        self.resets += 1

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        from promptkit.core.event_sys import Event, EventType

        self.calls += 1
        text = prompt["content"] if isinstance(prompt, dict) else str(prompt)
        self.prompts.append(text)
        self.tools_seen.append(tools)
        if self.calls <= self.fail_times:
            raise RuntimeError(f"第 {self.calls} 次挂了")
        if bus is not None and self.tokens:
            bus.emit(Event(EventType.USAGE, {"total_tokens": self.tokens}))
        if self.delay:
            time.sleep(self.delay)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class TestTask(PromptkitTestCase):
    def test_defaults(self):
        task = Task("research")
        self.assertEqual(task.description, "research")
        self.assertEqual(task.key, "research")
        self.assertEqual(task.depends_on, ())
        self.assertEqual(task.max_retries, 1)
        self.assertEqual(task.on_failure, "retry")

    def test_output_key_wins(self):
        self.assertEqual(Task("a", output_key="result").key, "result")

    def test_empty_name(self):
        with self.assertRaises(PlanError):
            Task("   ")

    def test_bad_on_failure(self):
        with self.assertRaises(PlanError):
            Task("a", on_failure="explode")

    def test_fallback_needs_an_agent(self):
        with self.assertRaises(PlanError):
            Task("a", on_failure="fallback")

    def test_negative_retries(self):
        with self.assertRaises(PlanError):
            Task("a", max_retries=-1)

    def test_bad_timeout(self):
        with self.assertRaises(PlanError):
            Task("a", timeout=0)

    def test_deps_are_normalised(self):
        task = Task("a", depends_on=["b", "b", " c "], input_keys=["x", "x"])
        self.assertEqual(task.depends_on, ("b", "c"))
        self.assertEqual(task.input_keys, ("x",))

    def test_to_spec(self):
        spec = Task("a", "做点事", depends_on=("b",), output_key="out").to_spec()
        self.assertEqual(spec["name"], "a")
        self.assertEqual(spec["depends_on"], ["b"])
        self.assertEqual(spec["output_key"], "out")
        self.assertIn("on_failure", spec)

    def test_repr(self):
        self.assertIn("a", repr(Task("a", depends_on=("b",))))


class TestPlan(PromptkitTestCase):
    def test_empty_plan(self):
        with self.assertRaises(PlanError):
            Plan([])

    def test_duplicate_names(self):
        with self.assertRaises(PlanError) as cm:
            Plan((Task("a"), Task("a")))
        self.assertIn("重复", str(cm.exception))

    def test_unknown_dependency(self):
        with self.assertRaises(PlanError) as cm:
            Plan((Task("a", depends_on=("nope",)),))
        self.assertIn("不在计划里", str(cm.exception))

    def test_self_dependency(self):
        with self.assertRaises(PlanError) as cm:
            Plan((Task("a", depends_on=("a",)),))
        self.assertIn("自己", str(cm.exception))

    def test_cycle(self):
        with self.assertRaises(PlanError) as cm:
            Plan((Task("a", depends_on=("b",)), Task("b", depends_on=("a",))))
        self.assertIn("环", str(cm.exception))

    def test_topological_order_is_stable(self):
        plan = Plan(
            (Task("c", depends_on=("b",)), Task("a"), Task("b", depends_on=("a",)))
        )
        self.assertEqual([task.name for task in plan.topological()], ["a", "b", "c"])

    def test_waves(self):
        plan = Plan(
            (
                Task("a"),
                Task("b", depends_on=("a",)),
                Task("c", depends_on=("a",)),
                Task("d", depends_on=("b", "c")),
            )
        )
        self.assertEqual(
            [[task.name for task in layer] for layer in plan.waves()],
            [["a"], ["b", "c"], ["d"]],
        )

    def test_from_template_with_dicts(self):
        plan = Plan.from_template([{"name": "a"}, {"name": "b", "depends_on": ["a"]}])
        self.assertEqual(plan.names, ["a", "b"])

    def test_from_template_max_tasks(self):
        with self.assertRaises(PlanError):
            Plan.from_template([Task("a"), Task("b")], max_tasks=1)

    def test_navigation(self):
        plan = Plan((Task("a"), Task("b", depends_on=("a",))), task="原文")
        self.assertEqual(len(plan), 2)
        self.assertEqual([task.name for task in plan], ["a", "b"])
        self.assertEqual(plan["b"].name, "b")
        self.assertEqual(plan.task, "原文")
        self.assertIn("a -> b", plan.describe())

    def test_plan_from(self):
        plan = Plan((Task("a"),))
        self.assertIs(plan_from(plan), plan)
        self.assertEqual(plan_from([Task("a")]).names, ["a"])
        self.assertEqual(plan_from({"tasks": [{"name": "x"}]}).names, ["x"])
        with self.assertRaises(PlanError):
            plan_from("不是计划")


class TestOrchestrator(PromptkitTestCase):
    def test_runs_in_dependency_order_and_passes_outputs(self):
        research = OrchAgent(["调研结果"])
        writer = OrchAgent(["报告正文"])
        plan = Plan.from_template(
            [
                Task("research", "调研一下", agent=research),
                Task("write", "根据 {{research}} 写报告", agent=writer,
                     depends_on=("research",), output_key="report"),
            ]
        )
        result = Orchestrator().run(plan)
        self.assertTrue(result.ok)
        self.assertEqual(writer.prompts[0], "根据 调研结果 写报告")
        self.assertEqual(result.to_context(), {"research": "调研结果", "report": "报告正文"})

    def test_agents_get_their_own_prompt(self):
        first, second = OrchAgent(["一"]), OrchAgent(["二"])
        plan = Plan.from_template([Task("a", agent=first), Task("b", agent=second)])
        Orchestrator().run(plan)
        self.assertEqual(first.prompts, ["a"])
        self.assertEqual(second.prompts, ["b"])

    def test_upstream_outputs_are_attached_when_not_referenced(self):
        research = OrchAgent(["调研结果"])
        writer = OrchAgent(["报告"])
        plan = Plan.from_template(
            [
                Task("research", "调研一下", agent=research),
                Task("write", "写一份报告", agent=writer, depends_on=("research",)),
            ]
        )
        Orchestrator().run(plan)
        self.assertIn("上游任务的结果", writer.prompts[0])
        self.assertIn("调研结果", writer.prompts[0])

    def test_upstream_is_not_duplicated_when_referenced(self):
        research = OrchAgent(["调研结果"])
        writer = OrchAgent(["报告"])
        plan = Plan.from_template(
            [
                Task("research", "调研一下", agent=research),
                Task("write", "根据 {{research}} 写报告", agent=writer,
                     depends_on=("research",)),
            ]
        )
        Orchestrator().run(plan)
        self.assertEqual(writer.prompts[0], "根据 调研结果 写报告")

    def test_dependent_is_skipped_when_dependency_fails(self):
        failing = OrchAgent(fail_times=99)
        child = OrchAgent(["不该被调用"])
        plan = Plan.from_template(
            [
                Task("a", agent=failing, max_retries=0),
                Task("b", agent=child, depends_on=("a",)),
            ]
        )
        result = Orchestrator().run(plan)
        self.assertEqual(result.by_name("a").status, "failed")
        self.assertEqual(result.by_name("b").status, "skipped")
        self.assertIn("依赖", result.by_name("b").error)
        self.assertEqual(child.calls, 0)

    def test_retry_then_success(self):
        agent = OrchAgent(["好了"], fail_times=1)
        result = Orchestrator().run(
            Plan.from_template([Task("a", agent=agent, max_retries=2)])
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.results[0].attempts, 2)
        self.assertIn("上一次执行出错了", agent.prompts[1])

    def test_retries_are_bounded(self):
        agent = OrchAgent(fail_times=99)
        result = Orchestrator().run(
            Plan.from_template([Task("a", agent=agent, max_retries=2)])
        )
        self.assertEqual(result.results[0].status, "failed")
        self.assertEqual(result.results[0].attempts, 3)
        self.assertEqual(agent.calls, 3)

    def test_fallback_agent(self):
        primary = OrchAgent(fail_times=99)
        backup = OrchAgent(["备用答案"])
        result = Orchestrator().run(
            Plan.from_template(
                [
                    Task(
                        "a",
                        agent=primary,
                        max_retries=0,
                        on_failure="fallback",
                        fallback_agent=backup,
                    )
                ]
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.results[0].output, "备用答案")
        self.assertIn("fallback", result.results[0].error)

    def test_abort_stops_the_plan(self):
        agent = OrchAgent(fail_times=99)
        plan = Plan.from_template(
            [
                Task("a", agent=agent, max_retries=0, on_failure="abort"),
                Task("b", agent=OrchAgent(["x"])),
            ]
        )
        with self.assertRaises(TaskError) as cm:
            Orchestrator().run(plan)
        partial = cm.exception.result
        self.assertEqual(partial.by_name("a").status, "failed")
        self.assertEqual(partial.by_name("b").status, "skipped")
        self.assertIn("中止", partial.by_name("b").error)

    def test_stop_on_error(self):
        agent = OrchAgent(fail_times=99)
        plan = Plan.from_template(
            [Task("a", agent=agent, max_retries=0), Task("b", agent=OrchAgent(["x"]))]
        )
        with self.assertRaises(TaskError):
            Orchestrator(OrchestratorConfig(stop_on_error=True)).run(plan)

    def test_skip_policy_marks_skipped(self):
        '''`skip` = 不重试、标 skipped（依赖它的任务也会被跳过）。'''
        agent = OrchAgent(fail_times=99)
        result = Orchestrator().run(
            Plan.from_template([Task("a", agent=agent, on_failure="skip")])
        )
        self.assertEqual(result.results[0].status, "skipped")
        self.assertEqual(agent.calls, 1)

    def test_abort_does_not_retry(self):
        agent = OrchAgent(fail_times=99)
        with self.assertRaises(TaskError):
            Orchestrator().run(
                Plan.from_template(
                    [Task("a", agent=agent, on_failure="abort", max_retries=3)]
                )
            )
        self.assertEqual(agent.calls, 1)

    def test_budget(self):
        agents = [OrchAgent(tokens=100) for _ in range(3)]
        plan = Plan.from_template(
            [Task(f"t{index}", agent=agents[index]) for index in range(3)]
        )
        result = Orchestrator(OrchestratorConfig(max_total_tokens=150)).run(plan)
        self.assertEqual(
            [item.status for item in result.results],
            ["success", "success", "skipped"],
        )
        self.assertIn("预算", result.by_name("t2").error)
        self.assertEqual(result.tokens, 200)

    def test_missing_agent(self):
        Agent.set_default(None)
        result = Orchestrator().run(Plan.from_template([Task("a")]))
        self.assertEqual(result.results[0].status, "failed")
        self.assertIn("没有可用的 agent", result.results[0].error)

    def test_missing_variable(self):
        result = Orchestrator().run(
            Plan.from_template([Task("a", "写 {{不存在}}", agent=OrchAgent(["x"]))])
        )
        self.assertEqual(result.results[0].status, "failed")
        self.assertIn("变量不存在", result.results[0].error)

    def test_max_tasks(self):
        plan = Plan.from_template([Task(f"t{index}") for index in range(3)])
        with self.assertRaises(PlanError):
            Orchestrator(OrchestratorConfig(max_tasks=2)).run(plan)

    def test_isolation_resets_agent(self):
        agent = OrchAgent(["x"])
        Orchestrator().run(Plan.from_template([Task("a", agent=agent)]))
        self.assertEqual(agent.resets, 1)

        agent2 = OrchAgent(["x"])
        Orchestrator(OrchestratorConfig(isolate=False)).run(
            Plan.from_template([Task("a", agent=agent2)])
        )
        self.assertEqual(agent2.resets, 0)

    def test_on_result_callback(self):
        seen = []
        Orchestrator(on_result=lambda task, result: seen.append((task.name, result.status))).run(
            Plan.from_template([Task("a", agent=OrchAgent(["x"]))])
        )
        self.assertEqual(seen, [("a", "success")])

    def test_concurrency(self):
        slow = [OrchAgent(["a"], delay=0.3), OrchAgent(["b"], delay=0.3)]
        plan = Plan.from_template(
            [Task("a", agent=slow[0]), Task("b", agent=slow[1])]
        )
        started = time.perf_counter()
        result = Orchestrator(OrchestratorConfig(max_concurrency=2)).run(plan)
        elapsed = time.perf_counter() - started
        self.assertTrue(result.ok)
        self.assertLess(elapsed, 0.55)      # 串行要 0.6s 以上
        self.assertEqual([item.name for item in result.results], ["a", "b"])

    def test_timeout(self):
        result = Orchestrator(
            OrchestratorConfig(per_task_timeout=0.05)
        ).run(
            Plan.from_template(
                [Task("slow", agent=OrchAgent(["x"], delay=0.4), max_retries=0)]
            )
        )
        self.assertEqual(result.results[0].status, "failed")
        self.assertIn("超时", result.results[0].error)

    def test_agent_factory_receives_task(self):
        seen = []

        def factory(task):
            seen.append(task.name)
            return OrchAgent([f"{task.name} 的结果"])

        result = Orchestrator(agent_factory=factory).run(
            Plan.from_template([Task("a"), Task("b", depends_on=("a",))])
        )
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual(result.to_context(), {"a": "a 的结果", "b": "b 的结果"})

    def test_config_validation(self):
        for kwargs in ({"max_concurrency": 0}, {"max_tasks": 0}, {"max_total_tokens": 0},
                       {"per_task_timeout": -1}):
            with self.assertRaises(ValueError):
                OrchestratorConfig(**kwargs)


class TestPlanResult(PromptkitTestCase):
    def test_views(self):
        result = PlanResult(
            task="x",
            results=[
                TaskResult("a", "success", output="A", tokens=10),
                TaskResult("b", "failed", error="坏了", tokens=5),
                TaskResult("c", "skipped", error="依赖没成功"),
            ],
        )
        self.assertFalse(result.ok)
        self.assertEqual([item.name for item in result.succeeded], ["a"])
        self.assertEqual([item.name for item in result.failed], ["b"])
        self.assertEqual([item.name for item in result.skipped], ["c"])
        self.assertEqual(result.tokens, 15)
        self.assertEqual(result.to_context(), {"a": "A"})
        self.assertIn("失败 1", result.summary())
        self.assertIn("✗ b", result.summary())
        data = result.to_dict()
        self.assertEqual(data["tokens"], 15)
        self.assertEqual(len(data["results"]), 3)
        self.assertIn("PlanResult", repr(result))

    def test_all_success(self):
        result = PlanResult(results=[TaskResult("a", "success", output=1)])
        self.assertTrue(result.ok)


class TestTemplatePlanner(PromptkitTestCase):
    def test_returns_plan(self):
        planner = TemplatePlanner([Task("a"), Task("b", depends_on=("a",))])
        plan = planner.plan("需求", {})
        self.assertEqual(plan.names, ["a", "b"])
        self.assertIn("TemplatePlanner", repr(planner))

    def test_max_tasks(self):
        with self.assertRaises(PlanError):
            TemplatePlanner([Task("a"), Task("b")], max_tasks=1).plan("x", {})

    def test_task_text_is_carried(self):
        plan = TemplatePlanner([Task("a")]).plan("原始需求", {})
        self.assertEqual(plan.task, "原始需求")


class TestLLMPlanner(PromptkitTestCase):
    def planner(self, replies, **kwargs):
        return LLMPlanner(OrchAgent(replies), **kwargs)

    def test_parses_tasks(self):
        import json as _json

        raw = _json.dumps(
            {
                "tasks": [
                    {"name": "research", "description": "调研", "role": "researcher"},
                    {
                        "name": "write",
                        "description": "写报告",
                        "role": "writer",
                        "depends_on": ["research"],
                        "output_key": "report",
                    },
                ],
                "reasoning": "先调研再写",
            }
        )
        plan = self.planner([raw]).plan("做个报告", {})
        self.assertEqual(plan.names, ["research", "write"])
        self.assertEqual(plan["write"].depends_on, ("research",))
        self.assertEqual(plan["write"].output_key, "report")
        self.assertEqual(plan.task, "做个报告")

    def test_truncates_to_max_tasks(self):
        import json as _json

        raw = _json.dumps(
            {"tasks": [{"name": f"t{index}", "description": "x"} for index in range(5)]}
        )
        self.assertEqual(self.planner([raw], max_tasks=2).plan("x", {}).names, ["t0", "t1"])

    def test_drops_unknown_dependencies(self):
        import json as _json

        raw = _json.dumps(
            {
                "tasks": [
                    {"name": "a", "description": "A", "depends_on": ["不存在"]},
                    {"name": "b", "description": "B", "depends_on": ["a"]},
                ]
            }
        )
        plan = self.planner([raw]).plan("x", {})
        self.assertEqual(plan["a"].depends_on, ())
        self.assertEqual(plan["b"].depends_on, ("a",))

    def test_retries_on_bad_json(self):
        import json as _json

        agent = OrchAgent(["不是 JSON", _json.dumps({"tasks": [{"name": "a", "description": "A"}]})])
        plan = LLMPlanner(agent).plan("x", {})
        self.assertEqual(plan.names, ["a"])
        self.assertIn("上一次的输出无法使用", agent.prompts[1])

    def test_empty_plan(self):
        import json as _json

        with self.assertRaises(PlanError):
            self.planner([_json.dumps({"tasks": []})]).plan("x", {})

    def test_prompt_contains_limits_and_context(self):
        import json as _json

        agent = OrchAgent([_json.dumps({"tasks": [{"name": "a", "description": "A"}]})])
        LLMPlanner(agent, max_tasks=3).plan("原始需求", {"usr_input": "内容"})
        prompt = agent.prompts[0]
        self.assertIn("原始需求", prompt)
        self.assertIn("最多 3 个", prompt)
        self.assertIn("usr_input", prompt)

    def test_invalid_on_failure_is_rejected(self):
        import json as _json

        raw = _json.dumps({"tasks": [{"name": "a", "description": "A", "on_failure": "x"}]})
        with self.assertRaises(Exception):
            self.planner([raw]).plan("x", {})


class TestPlanSteps(PromptkitTestCase):
    def test_plan_step_with_template(self):
        ctx = Workflow(
            PlanStep(task="需求", planner=TemplatePlanner([Task("a"), Task("b", depends_on=("a",))]))
        ).run()
        self.assertIsInstance(ctx["plan"], Plan)
        self.assertEqual(ctx["plan"].names, ["a", "b"])
        self.assertIn("plan:2", ctx.trace)

    def test_plan_step_with_agent_uses_llm_planner(self):
        import json as _json

        raw = _json.dumps({"tasks": [{"name": "a", "description": "A"}]})
        agent = OrchAgent([raw])
        ctx = Workflow(PlanStep(task=Var("需求"), planner=agent, max_tasks=2)).run(需求="做个东西")
        self.assertEqual(ctx["plan"].names, ["a"])
        self.assertIn("做个东西", agent.prompts[0])

    def test_plan_step_without_planner(self):
        with self.assertRaises(ValueError):
            Workflow(PlanStep(task="x")).run()

    def test_plan_step_passes_context_keys(self):
        captured = {}

        class Recorder(Planner):
            def plan(self, task, context):
                captured.update(context)
                return Plan((Task("a"),))

        Workflow(
            PlanStep(task="x", planner=Recorder(), context_keys=("name",))
        ).run(name="小明", other="忽略")
        self.assertEqual(captured, {"name": "小明"})

    def test_execute_plan_writes_results_and_outputs(self):
        plan = Plan.from_template([Task("a", agent=OrchAgent(["结果A"]), output_key="out_a")])
        ctx = Workflow(ExecutePlan(plan=plan)).run()
        self.assertEqual(ctx["task_results"][0]["name"], "a")
        self.assertEqual(ctx["task_results"][0]["status"], "success")
        self.assertIsInstance(ctx["plan_result"], PlanResult)
        self.assertEqual(ctx["out_a"], "结果A")
        self.assertIn("execute_plan:1/1", ctx.trace)

    def test_execute_plan_strict(self):
        plan = Plan.from_template([Task("a", agent=OrchAgent(fail_times=99), max_retries=0)])
        with self.assertRaises(TaskError):
            Workflow(ExecutePlan(plan=plan, strict=True)).run()

    def test_execute_plan_from_template_list(self):
        ctx = Workflow(
            ExecutePlan(plan=[Task("a", agent=OrchAgent(["x"]))])
        ).run()
        self.assertTrue(ctx["plan_result"].ok)

    def test_execute_plan_with_factory(self):
        ctx = Workflow(
            ExecutePlan(
                plan=Plan.from_template([Task("a", role="writer")]),
                agent_factory=lambda task: OrchAgent([f"{task.role} 写的"]),
            )
        ).run()
        self.assertEqual(ctx["a"], "writer 写的")

    def test_aggregate_step_text(self):
        agent = OrchAgent(["合并后的答复"])
        ctx = Workflow(
            ExecutePlan(plan=Plan.from_template([Task("a", agent=OrchAgent(["结果A"]))])),
            AggregateStep(agent=agent),
        ).run()
        self.assertEqual(ctx["final"], "合并后的答复")
        self.assertIn("## a [success]", agent.prompts[0])

    def test_aggregate_step_structured(self):
        ctx = Workflow(
            ExecutePlan(plan=Plan.from_template([Task("a", agent=OrchAgent(["结果A"]))])),
            AggregateStep(agent=OrchAgent(['{"status": "ok"}']), output_model=Order),
        ).run()
        self.assertIsInstance(ctx["final"], Order)
        self.assertEqual(ctx["final"].status, "ok")

    def test_aggregate_step_without_results(self):
        with self.assertRaises(ValueError):
            Workflow(AggregateStep(agent=OrchAgent(["x"]))).run()

    def test_format_task_results_marks_failures(self):
        text = format_task_results(
            [
                {"name": "a", "status": "success", "output": "好"},
                {"name": "b", "status": "failed", "error": "坏了"},
            ]
        )
        self.assertIn("## a [success]", text)
        self.assertIn("（错误：坏了）", text)
        self.assertIn("这些任务没有成功：b", text)

    def test_format_task_results_truncates(self):
        text = format_task_results(
            [{"name": "a", "status": "success", "output": "x" * 200}], max_chars=10
        )
        self.assertIn("已截断", text)


class TestDelegateTool(PromptkitTestCase):
    def test_delegates_and_prefixes(self):
        researcher = OrchAgent(["调研结果"])
        tool = DelegateTool({"researcher": researcher}, max_delegations=2)
        self.assertEqual(tool.run("researcher", "调研一下"), "【researcher】调研结果")
        self.assertEqual(researcher.prompts, ["调研一下"])

    def test_unknown_agent(self):
        tool = DelegateTool({"researcher": OrchAgent(["x"])})
        self.assertIn("没有叫", tool.run("nobody", "x"))

    def test_max_delegations(self):
        tool = DelegateTool({"a": OrchAgent(["x"])}, max_delegations=1)
        tool.run("a", "第一次")
        self.assertIn("上限", tool.run("a", "第二次"))
        tool.reset()
        self.assertIn("【a】", tool.run("a", "重置后"))

    def test_depth_limit(self):
        tool = DelegateTool({"a": OrchAgent(["x"])}, max_depth=1, depth=1)
        self.assertIn("深度", tool.run("a", "x"))

    def test_nested_agent_gets_a_deeper_tool(self):
        sub = OrchAgent(["内层"])
        tool = DelegateTool({"a": sub}, max_depth=2)
        tool.run("a", "x")
        passed = sub.tools_seen[0]
        self.assertEqual(len(passed), 1)
        self.assertIsInstance(passed[0], DelegateTool)
        self.assertEqual(passed[0].depth, 1)

    def test_max_depth_one_means_no_further_tool(self):
        sub = OrchAgent(["内层"])
        DelegateTool({"a": sub}, max_depth=1).run("a", "x")
        self.assertIsNone(sub.tools_seen[0])

    def test_agent_factory(self):
        tool = DelegateTool({}, agent_factory=lambda name: OrchAgent([f"{name} 干的"]))
        self.assertEqual(tool.run("coder", "写代码"), "【coder】coder 干的")

    def test_empty_task(self):
        self.assertIn("不能是空的", DelegateTool({"a": OrchAgent(["x"])}).run("a", " "))

    def test_schema_lists_agents(self):
        tool = DelegateTool({"writer": OrchAgent(["x"]), "coder": OrchAgent(["y"])})
        function = tool.schema()["function"]
        self.assertEqual(function["parameters"]["properties"]["agent"]["enum"], ["coder", "writer"])

    def test_needs_agents_or_factory(self):
        with self.assertRaises(ValueError):
            DelegateTool({})
        with self.assertRaises(ValueError):
            DelegateTool({"a": OrchAgent(["x"])}, max_delegations=0)


class TestSubAgents(PromptkitTestCase):
    def test_sub_agent_has_its_own_history(self):
        before = Agent.default()
        agent = sub_agent("你是调研员", api_key="sk-test")
        self.assertEqual(agent.start_prompts[-1]["content"], "你是调研员")
        self.assertEqual(agent.history, [])
        self.assertIs(Agent.default(), before)      # 不抢默认 agent

    def test_sub_agent_needs_a_system_prompt(self):
        with self.assertRaises(ValueError):
            sub_agent("   ", api_key="sk-test")

    def test_role_factory(self):
        factory = RoleFactory({"writer": "你是写手", "researcher": "你是调研员"})
        agent = factory(Task("x", role="writer"))
        self.assertEqual(agent.start_prompts[-1]["content"], "你是写手")
        with self.assertRaises(KeyError):
            factory(Task("y", role="nope"))

    def test_role_factory_default(self):
        factory = RoleFactory({"writer": "你是写手"}, default_system="你是通用助手")
        agent = factory(Task("x", role="unknown"))
        self.assertEqual(agent.start_prompts[-1]["content"], "你是通用助手")

    def test_structured_answer(self):
        agent = OrchAgent(['{"status": "ok"}'])
        value = structured_answer(agent, "给我个结构", Order)
        self.assertEqual(value.status, "ok")

    def test_structured_answer_retries(self):
        agent = OrchAgent(["不是 JSON", '{"status": "ok"}'])
        self.assertEqual(structured_answer(agent, "x", Order).status, "ok")
        self.assertIn("上一次的输出无法使用", agent.prompts[1])

    def test_structured_answer_gives_up(self):
        from promptkit.executor import MaxRetriesExceeded

        with self.assertRaises(MaxRetriesExceeded):
            structured_answer(OrchAgent(["坏"]), "x", Order, max_retries=1)

    def test_structured_answer_does_not_render_braces(self):
        '''任务输出里带 `{{x}}` 时不能被当成模板（这是不用 Executor 的原因）。'''
        agent = OrchAgent(['{"status": "ok"}'])
        structured_answer(agent, "数据：{{x}} 和 {y}", Order)
        self.assertIn("{{x}}", agent.prompts[0])


class TestOrchestrateLayering(PromptkitTestCase):
    def test_core_does_not_import_orchestrate(self):
        '''编排是上层：core 不能反过来依赖它。'''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^(?:from\s+(promptkit[\w.]*)\s+import|import\s+(promptkit[\w.]*))",
            re.MULTILINE,
        )
        offenders = []
        for path in sorted((root / "core").rglob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                dependency = match.group(1) or match.group(2)
                if "promptkit.orchestrate" in dependency:
                    offenders.append(f"{path.name}: {dependency}")
        self.assertEqual(offenders, [])


# ==================== Skill（可注册的能力单元） ====================
from promptkit.skill import (  # noqa: E402
    Skill,
    SkillAlreadyExists,
    SkillContext,
    SkillError,
    SkillExecutionError,
    SkillNotFound,
    SkillRegistry,
    SkillSpec,
    SkillStep,
    SkillTool,
    SkillValidationError,
    build_parameters,
    build_spec,
    result_text,
    skill,
    type_schema,
    validate_args,
)


class TestSkillSchema(PromptkitTestCase):
    def test_basic_types_and_required(self):
        def handler(text: str, count: int, ratio: float, flag: bool) -> None:
            pass

        schema = build_parameters(handler)
        self.assertEqual(
            {name: item["type"] for name, item in schema["properties"].items()},
            {"text": "string", "count": "integer", "ratio": "number", "flag": "boolean"},
        )
        self.assertEqual(
            schema["required"], ["text", "count", "ratio", "flag"]
        )

    def test_defaults_are_optional_and_recorded(self):
        def handler(text: str, max_words: int = 100) -> None:
            pass

        schema = build_parameters(handler)
        self.assertEqual(schema["required"], ["text"])
        self.assertEqual(schema["properties"]["max_words"], {"type": "integer", "default": 100})

    def test_ctx_is_skipped(self):
        def handler(text: str, ctx: SkillContext) -> None:
            pass

        self.assertEqual(list(build_parameters(handler)["properties"]), ["text"])

    def test_varargs_are_skipped(self):
        def handler(text: str, *args, **kwargs) -> None:
            pass

        self.assertEqual(list(build_parameters(handler)["properties"]), ["text"])

    def test_optional_literal_enum_and_collections(self):
        from enum import Enum
        from typing import Literal, Optional

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        def handler(
            a: Optional[str],
            b: Literal["x", "y"],
            c: list[int],
            d: dict[str, int],
            e: Color,
        ) -> None:
            pass

        properties = build_parameters(handler)["properties"]
        self.assertEqual(properties["a"]["type"], ["string", "null"])
        self.assertEqual(properties["b"]["enum"], ["x", "y"])
        self.assertEqual(properties["c"], {"type": "array", "items": {"type": "integer"}})
        self.assertEqual(
            properties["d"], {"type": "object", "additionalProperties": {"type": "integer"}}
        )
        self.assertEqual(properties["e"]["enum"], ["red", "blue"])

    def test_pydantic_model_becomes_object_schema(self):
        def handler(order: Order) -> None:
            pass

        schema = build_parameters(handler)["properties"]["order"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("status", schema["properties"])

    def test_any_and_missing_annotations(self):
        from typing import Any

        def handler(a: Any, b) -> None:
            pass

        properties = build_parameters(handler)["properties"]
        self.assertEqual(properties["a"], {})
        self.assertEqual(properties["b"], {"type": "string"})

    def test_type_schema_fallbacks(self):
        from typing import Any

        self.assertEqual(type_schema(Any), {})
        self.assertEqual(type_schema(list), {"type": "array"})
        self.assertEqual(type_schema(dict), {"type": "object"})

    def test_validate_args(self):
        spec = SkillSpec(
            "s", "描述", lambda text: text,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["text"],
            },
        )
        validate_args(spec, {"text": "a"})
        with self.assertRaises(SkillValidationError) as cm:
            validate_args(spec, {})
        self.assertIn("缺少必填参数", str(cm.exception))
        with self.assertRaises(SkillValidationError) as cm:
            validate_args(spec, {"text": "a", "nope": 1})
        self.assertIn("不认识", str(cm.exception))
        with self.assertRaises(SkillValidationError) as cm:
            validate_args(spec, {"text": 1})
        self.assertIn("应该是 string", str(cm.exception))
        with self.assertRaises(SkillValidationError):
            validate_args(spec, {"text": "a", "n": True})

    def test_validate_args_enum(self):
        spec = SkillSpec(
            "s", "描述", lambda label: label,
            parameters={
                "type": "object",
                "properties": {"label": {"type": "string", "enum": ["a", "b"]}},
                "required": ["label"],
            },
        )
        with self.assertRaises(SkillValidationError) as cm:
            validate_args(spec, {"label": "c"})
        self.assertIn("只能是", str(cm.exception))


class TestSkillSpec(PromptkitTestCase):
    def test_name_validation(self):
        for bad in ("", "1abc", "有 空格", "x" * 65):
            with self.assertRaises(SkillValidationError):
                SkillSpec(bad, "描述", lambda: None)

    def test_description_is_required(self):
        with self.assertRaises(SkillValidationError) as cm:
            SkillSpec("s", "  ", lambda: None)
        self.assertIn("description", str(cm.exception))

    def test_handler_must_be_callable(self):
        with self.assertRaises(TypeError):
            SkillSpec("s", "描述", "不是函数")

    def test_parameters_must_be_object(self):
        with self.assertRaises(SkillValidationError):
            SkillSpec("s", "描述", lambda: None, parameters={"type": "array"})

    def test_openai_schema(self):
        spec = SkillSpec("s", "描述", lambda: None, tags=("a",))
        schema = spec.openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "s")
        self.assertEqual(schema["function"]["parameters"], {"type": "object", "properties": {}})

    def test_to_dict(self):
        spec = SkillSpec("s", "描述", lambda: None, tags=("a",), version="1.2")
        data = spec.to_dict()
        self.assertEqual(data["version"], "1.2")
        self.assertEqual(data["tags"], ["a"])

    def test_repr(self):
        self.assertIn("s", repr(SkillSpec("s", "描述", lambda: None)))


class TestSkillDecorator(PromptkitTestCase):
    def test_bare_decorator(self):
        @skill
        def summarize(text: str) -> str:
            '''把文本压缩一下。'''
            return text[:5]

        self.assertIsInstance(summarize, Skill)
        self.assertEqual(summarize.name, "summarize")
        self.assertEqual(summarize.description, "把文本压缩一下。")
        self.assertEqual(summarize("abcdefg"), "abcde")

    def test_decorator_with_arguments(self):
        @skill(name="s2", description="自定义描述", tags=["a", "b"])
        def thing(text: str) -> str:
            return text

        self.assertEqual(thing.name, "s2")
        self.assertEqual(thing.description, "自定义描述")
        self.assertEqual(thing.tags, ("a", "b"))

    def test_description_falls_back_to_name(self):
        @skill
        def no_doc(text: str) -> str:
            return text

        self.assertIn("no_doc", no_doc.description)

    def test_output_model_becomes_returns(self):
        @skill(output_model=Order)
        def make(status: str) -> Order:
            return Order(status=status)

        self.assertEqual(make.returns["type"], "object")
        self.assertIn("status", make.returns["properties"])

    def test_manual_parameters_win(self):
        params = {"type": "object", "properties": {}, "required": []}

        @skill(parameters=params)
        def thing(text: str) -> str:
            return text

        self.assertEqual(thing.parameters, params)

    def test_registry_auto_registration(self):
        registry = SkillRegistry()

        @skill(registry=registry)
        def auto(text: str) -> str:
            return text

        self.assertIn("auto", registry)

    def test_context_detection(self):
        def plain(text: str) -> str:
            return text

        def with_ctx(text: str, ctx: SkillContext) -> str:
            return text

        self.assertFalse(build_spec(plain).accepts_ctx)
        self.assertTrue(build_spec(with_ctx).accepts_ctx)

    def test_skill_is_callable_without_ctx(self):
        @skill
        def plain(text: str, n: int = 2) -> str:
            return text * n

        self.assertEqual(plain("a"), "aa")
        self.assertEqual(plain(text="b", n=3), "bbb")

    def test_ctx_skill_without_agent_reports_clearly(self):
        @skill
        def needs_model(text: str, ctx: SkillContext) -> str:
            return ctx.ask(text)

        Agent.set_default(None)
        with self.assertRaises(SkillError) as cm:
            needs_model("hi")
        self.assertIn("agent", str(cm.exception))

    def test_two_skills_same_function_name_are_ok_if_named(self):
        @skill(name="a")
        def one(text: str) -> str:
            return text

        @skill(name="b")
        def two(text: str) -> str:
            return text

        self.assertNotEqual(one.name, two.name)

    def test_invalid_function(self):
        with self.assertRaises(TypeError):
            build_spec(42)


class TestSkillRegistry(PromptkitTestCase):
    def skill(self, name="sum", **kwargs):
        @skill(name=name, description=f"{name} 的描述", **kwargs)
        def handler(text: str) -> str:
            return text

        return handler

    def test_register_accepts_skill_spec_and_function(self):
        def plain(text: str) -> str:
            return text

        registry = SkillRegistry()
        registry.register(self.skill("a"))
        registry.register(SkillSpec("b", "描述", lambda: None))
        registry.register(plain)
        self.assertEqual(registry.names, ["a", "b", "plain"])

    def test_duplicate_registration(self):
        registry = SkillRegistry()
        registry.register(self.skill("a"))
        with self.assertRaises(SkillAlreadyExists):
            registry.register(self.skill("a"))

    def test_replace(self):
        registry = SkillRegistry()
        registry.register(self.skill("a"))
        new = self.skill("a")
        registry.register(new, replace=True)
        self.assertIs(registry.get("a"), new)

    def test_get_missing(self):
        with self.assertRaises(SkillNotFound) as cm:
            SkillRegistry().get("nope")
        self.assertIn("没有叫", str(cm.exception))

    def test_find(self):
        registry = SkillRegistry()
        registry.register(self.skill("a"))
        self.assertIsNotNone(registry.find("a"))
        self.assertIsNone(registry.find("b"))

    def test_unregister(self):
        registry = SkillRegistry([self.skill("a")])
        registry.unregister("a")
        self.assertEqual(len(registry), 0)
        with self.assertRaises(SkillNotFound):
            registry.unregister("a")

    def test_list_by_tag_and_query(self):
        registry = SkillRegistry(
            [self.skill("a", tags=["nlp"]), self.skill("b", tags=["file"])]
        )
        self.assertEqual([item.name for item in registry.list(tag="nlp")], ["a"])
        self.assertEqual([item.name for item in registry.list(query="B")], ["b"])

    def test_dunder(self):
        registry = SkillRegistry([self.skill("a"), self.skill("b")])
        self.assertEqual(len(registry), 2)
        self.assertIn("a", registry)
        self.assertEqual([item.name for item in registry], ["a", "b"])
        self.assertIn("SkillRegistry", repr(registry))

    def test_register_rejects_other_types(self):
        with self.assertRaises(TypeError):
            SkillRegistry().register(42)


class TestSkillExport(PromptkitTestCase):
    def test_to_openai_tools(self):
        @skill(name="a", description="A", tags=["x"])
        def a(text: str) -> str:
            return text

        registry = SkillRegistry([a])
        tools = registry.to_openai_tools()
        self.assertEqual(tools[0]["function"]["name"], "a")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["text"])

    def test_to_tools_returns_bound_tools(self):
        @skill(name="a", description="A")
        def a(text: str) -> str:
            return text.upper()

        registry = SkillRegistry([a])
        ctx = SkillContext()
        tools = registry.to_tools(ctx)
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(tools[0], SkillTool)
        self.assertEqual(tools[0].call({"text": "hi"}), "HI")
        self.assertIs(ctx.skills, registry)

    def test_tool_errors_become_text(self):
        @skill(name="boom", description="一定炸")
        def boom(text: str) -> str:
            raise RuntimeError("炸了")

        tool = SkillRegistry([boom]).to_tools()[0]
        self.assertIn("错误：", tool.call({"text": "x"}))
        self.assertIn("炸了", tool.call({"text": "x"}))

    def test_tool_validation_errors_become_text(self):
        @skill(name="a", description="A")
        def a(text: str) -> str:
            return text

        tool = SkillRegistry([a]).to_tools()[0]
        self.assertIn("错误：", tool.call({}))

    def test_tool_serialises_pydantic(self):
        @skill(name="a", description="A")
        def a(text: str) -> Order:
            return Order(status=text)

        tool = SkillRegistry([a]).to_tools()[0]
        self.assertEqual(tool.call({"text": "ok"}), '{"status": "ok"}')

    def test_result_text(self):
        self.assertEqual(result_text("x"), "x")
        self.assertEqual(result_text({"a": 1}), '{"a": 1}')
        self.assertEqual(result_text(Order(status="s")), '{"status": "s"}')
        self.assertEqual(result_text(3), "3")


class TestSkillToolReentrancy(PromptkitTestCase):
    def test_skill_tool_can_call_the_agent_without_corrupting_history(self):
        '''Skill 当工具用时，内部 `ctx.ask` 是**另一次对话**：

        工具执行期间 agent 的历史是副本，跑完丢掉 —— 否则消息序列会变成
        「assistant(tool_calls) -> user -> assistant -> tool」，API 直接 400。
        '''
        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("nested", "{}")],
                ),
                fake_message(role="assistant", content="内层回答"),
                fake_message(role="assistant", content="外层最终回答"),
            ]
        )

        def nested(**_kwargs):
            return agent.ask("内层问题")

        reply = agent.ask("外层问题", tools=[nested], max_tool_rounds=3)
        self.assertEqual(reply, "外层最终回答")
        self.assertEqual(
            [message["role"] for message in agent.history],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertTrue(agent.history[1].get("tool_calls"))
        self.assertEqual(agent.history[2]["content"], "内层回答")


class TestSkillFunctionAttributes(PromptkitTestCase):
    '''先给函数挂字段，再套装饰器 —— 接口字段读出来，其余进 metadata。'''

    def build(self):
        def summarize(text: str, max_words: int = 100) -> str:
            '''把文本压缩到指定字数。'''
            return text[:max_words]

        summarize.tags = ["nlp", "text"]
        summarize.version = "1.2"
        summarize.owner = "content-team"
        summarize.cost = "cheap"
        summarize.metadata = {"docs_url": "https://example.com"}
        summarize._internal = "不该进 metadata"
        return skill(summarize)

    def test_interface_fields_are_read(self):
        built = self.build()
        self.assertEqual(built.tags, ("nlp", "text"))
        self.assertEqual(built.spec.version, "1.2")
        self.assertEqual(built.description, "把文本压缩到指定字数。")

    def test_arbitrary_attributes_become_metadata(self):
        built = self.build()
        self.assertEqual(built.metadata["owner"], "content-team")
        self.assertEqual(built.metadata["cost"], "cheap")

    def test_metadata_dict_merges_with_attributes(self):
        built = self.build()
        self.assertEqual(built.metadata["docs_url"], "https://example.com")
        self.assertEqual(built.metadata["owner"], "content-team")

    def test_private_attributes_are_skipped(self):
        self.assertNotIn("_internal", self.build().metadata)

    def test_explicit_arguments_win(self):
        def handler(text: str) -> str:
            '''原始描述'''
            return text

        handler.tags = ["old"]
        handler.description = "挂上去的描述"
        handler.owner = "old-team"
        built = skill(handler, tags=["new"], description="显式描述", metadata={"owner": "new-team"})
        self.assertEqual(built.tags, ("new",))
        self.assertEqual(built.description, "显式描述")
        self.assertEqual(built.metadata["owner"], "new-team")

    def test_attributes_can_supply_parameters_and_returns(self):
        def handler(text: str) -> str:
            return text

        handler.parameters = {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要处理的文本"}},
            "required": ["text"],
        }
        handler.returns = {"type": "string"}
        built = skill(handler)
        self.assertEqual(
            built.parameters["properties"]["text"]["description"], "要处理的文本"
        )
        self.assertEqual(built.returns, {"type": "string"})

    def test_output_model_attribute(self):
        def handler(status: str) -> Order:
            return Order(status=status)

        handler.output_model = Order
        built = skill(handler)
        self.assertEqual(built.returns["type"], "object")
        self.assertIn("status", built.returns["properties"])

    def test_version_defaults_to_0_1(self):
        def handler(text: str) -> str:
            return text

        self.assertEqual(skill(handler).spec.version, "0.1")

    def test_metadata_is_not_in_the_api_schema(self):
        built = self.build()
        function = built.schema()["function"]
        self.assertNotIn("metadata", function)
        self.assertEqual(function["name"], "summarize")

    def test_to_dict_includes_metadata(self):
        data = self.build().spec.to_dict()
        self.assertEqual(data["metadata"]["cost"], "cheap")

    def test_with_metadata_can_extend_after_decoration(self):
        built = self.build()
        same = built.with_metadata(owner="platform", level=2)
        self.assertIs(same, built)
        self.assertEqual(built.metadata["owner"], "platform")
        self.assertEqual(built.metadata["level"], 2)

    def test_metadata_is_writable(self):
        built = self.build()
        built.metadata["owner"] = "hotfix"
        self.assertEqual(built.metadata["owner"], "hotfix")

    def test_registry_routes_by_metadata(self):
        built = self.build()

        def other(text: str) -> str:
            '''另一个'''
            return text

        other = skill(other).with_metadata(owner="platform")
        registry = SkillRegistry([built, other])
        names = [item.name for item in registry.list(metadata={"owner": "platform"})]
        self.assertEqual(names, ["other"])
        self.assertEqual(
            [item.name for item in registry.list(metadata={"cost": "cheap"})],
            ["summarize"],
        )

    def test_decorating_an_already_decorated_skill_keeps_metadata(self):
        built = self.build()
        again = skill(built)
        self.assertEqual(again.metadata["owner"], "content-team")
        self.assertEqual(again.tags, ("nlp", "text"))

    def test_skill_attributes_are_still_callable(self):
        built = self.build()
        self.assertEqual(built("abcdef", max_words=3), "abc")


class TestSkillStep(PromptkitTestCase):
    '''SkillStep：在 DSL 里调 Skill（FunctionStep 是 Step，@skill 套不上）。'''

    def skill(self, name="summarize", fn=None):
        if fn is None:
            def fn(text: str) -> str:
                return text[:5]
        return skill(name=name, description=f"{name} 的描述")(fn)

    def test_runs_and_writes_to_skill_name_by_default(self):
        ctx = Workflow(SkillStep(self.skill(), text="提示词工程")).run()
        self.assertEqual(ctx["summarize"], "提示词工程")

    def test_explicit_output_key(self):
        ctx = Workflow(
            SkillStep(self.skill(), text="abcdefg", output_key="short")
        ).run()
        self.assertEqual(ctx["short"], "abcde")

    def test_inputs_accept_expressions(self):
        ctx = Workflow(
            SkillStep(self.skill(), text=Var("source"), output_key="short")
        ).run(source="12345678")
        self.assertEqual(ctx["short"], "12345")

    def test_resolves_by_name_with_registry(self):
        registry = SkillRegistry([self.skill()])
        ctx = Workflow(
            SkillStep("summarize", text="abcdefg", registry=registry, output_key="short")
        ).run()
        self.assertEqual(ctx["short"], "abcde")

    def test_name_without_registry(self):
        with self.assertRaises(SkillNotFound) as cm:
            Workflow(SkillStep("summarize", text="x")).run()
        self.assertIn("registry", str(cm.exception))

    def test_unknown_name_in_registry(self):
        with self.assertRaises(SkillNotFound):
            Workflow(SkillStep("nope", registry=SkillRegistry())).run()

    def test_plain_function_is_wrapped(self):
        def shout(text: str) -> str:
            '''吼一声。'''
            return text.upper()

        ctx = Workflow(SkillStep(shout, text="hi")).run()
        self.assertEqual(ctx["shout"], "HI")

    def test_passes_the_workflow_agent_into_the_skill_context(self):
        agent = OrchAgent(["模型答复"])

        @skill(name="needs_model", description="需要模型")
        def needs_model(text: str, ctx: SkillContext = None) -> str:
            return ctx.ask(f"处理：{text}")

        ctx = Workflow(
            SkillStep(needs_model, text="你好", output_key="reply")
        ).run(agent=agent)
        self.assertEqual(ctx["reply"], "模型答复")
        self.assertIn("处理：你好", agent.prompts[0])

    def test_custom_ctx_wins(self):
        agent = OrchAgent(["自定义 ctx 的答复"])

        @skill(name="needs_model", description="需要模型")
        def needs_model(text: str, ctx: SkillContext = None) -> str:
            return ctx.ask(text)

        ctx = Workflow(
            SkillStep(needs_model, text="hi", ctx=SkillContext(agent=agent))
        ).run()
        self.assertEqual(ctx["needs_model"], "自定义 ctx 的答复")

    def test_missing_required_argument(self):
        with self.assertRaises(SkillValidationError):
            Workflow(SkillStep(self.skill())).run()

    def test_inputs_dict_and_duplicate_arguments(self):
        ctx = Workflow(
            SkillStep(self.skill(), inputs={"text": "abcdefg"}, output_key="short")
        ).run()
        self.assertEqual(ctx["short"], "abcde")
        with self.assertRaises(ValueError) as cm:
            SkillStep(self.skill(), inputs={"text": "a"}, text="b")
        self.assertIn("同时出现", str(cm.exception))

    def test_requires_a_skill(self):
        with self.assertRaises(ValueError):
            SkillStep()

    def test_describe_and_render(self):
        step = SkillStep(self.skill(), text=Var("source"), output_key="short")
        self.assertIn("summarize(text=source)", step.describe())
        self.assertIn("-> short", step.describe())
        self.assertIn("SkillStep", step.render())

    def test_returns_structured_objects(self):
        @skill(name="make", description="造一个")
        def make(status: str) -> Order:
            return Order(status=status)

        ctx = Workflow(SkillStep(make, status="ok")).run()
        self.assertIsInstance(ctx["make"], Order)


class TestChatWorkflowSkills(PromptkitTestCase):
    '''`ChatWorkflow(skills=[...])`：DSL 函数 / Skill 变成模型能调的工具。'''

    def make_lab(self, **kwargs):
        from promptkit.core import Function, LLMStep, Prompt

        return Function(
            args=("ref_path",),
            name="make_lab",
            description="根据学习主题生成实验任务清单",
            **kwargs,
        ).Do(
            LLMStep(
                Prompt(role="user", content="给出实验清单，参考：{{ref_path}}")
            )
        )

    def test_function_tool_schema(self):
        from promptkit.skill.chat import as_tools, function_parameters

        tool = as_tools([self.make_lab()])[0]
        self.assertEqual(tool.name, "make_lab")
        self.assertEqual(tool.description, "根据学习主题生成实验任务清单")
        self.assertEqual(tool.parameters["required"], ["ref_path"])
        self.assertEqual(
            function_parameters(self.make_lab())["properties"]["ref_path"]["type"],
            "string",
        )

    def test_function_tool_executes_with_the_holder_agent(self):
        from promptkit.skill.chat import AgentHolder, as_tools

        agent = OrchAgent(["实验清单：1. 跑基线"])
        tool = as_tools([self.make_lab()], holder=AgentHolder(agent))[0]
        self.assertEqual(tool.call({"ref_path": "https://x"}), "实验清单：1. 跑基线")
        self.assertIn("参考：https://x", agent.prompts[0])

    def test_skill_tool_uses_the_holder_agent(self):
        from promptkit.skill.chat import AgentHolder, as_tools

        @skill(name="ask_model", description="问一下模型")
        def ask_model(text: str, ctx: SkillContext = None) -> str:
            return ctx.ask(text)

        agent = OrchAgent(["答复"])
        tool = as_tools([ask_model], holder=AgentHolder(agent))[0]
        self.assertEqual(tool.call({"text": "你好"}), "答复")
        self.assertEqual(agent.prompts, ["你好"])

    def test_errors_become_text(self):
        from promptkit.skill.chat import AgentHolder, as_tools

        tool = as_tools([self.make_lab()], holder=AgentHolder(OrchAgent(["x"])))[0]
        self.assertIn("错误：", tool.call({}))            # 缺参数

        def boom(text: str) -> str:
            '''炸'''
            raise RuntimeError("坏了")

        broken = as_tools([boom], holder=AgentHolder(OrchAgent(["x"])))[0]
        self.assertIn("坏了", broken.call({"text": "x"}))

    def test_holder_falls_back_to_the_default_agent(self):
        from promptkit.skill.chat import as_tools

        @skill(name="plain", description="纯函数")
        def plain(text: str) -> str:
            return text.upper()

        self.assertEqual(as_tools([plain])[0].call({"text": "hi"}), "HI")

    def test_tools_are_passed_through_and_lists_flattened(self):
        from promptkit.skill.chat import as_tools

        existing = Tool(name="echo", description="回声", parameters={"type": "object", "properties": {}}, func=lambda **_: "x")
        tools = as_tools([[existing], [plain := skill(name="p", description="p")(lambda text: text)]])
        self.assertEqual([tool.name for tool in tools], ["echo", "p"])
        self.assertIs(tools[0], existing)

    def test_rejects_unknown_skill_types(self):
        from promptkit.skill.chat import as_tools

        with self.assertRaises(TypeError) as cm:
            as_tools([42])
        self.assertIn("FunctionDef", str(cm.exception))

    def test_chat_workflow_registers_skills_as_tools(self):
        from unittest import mock

        chat = ChatWorkflow(role="助手", skills=[self.make_lab()])
        built = chat.build()
        with mock.patch("builtins.input", return_value="exit"):
            ctx = built.run(agent=OrchAgent(["x"]), verbose=False)
        names = [getattr(tool, "name", "") for tool in ctx.get("chat_tools") or []]
        self.assertIn("make_lab", names)

    def test_agent_can_call_a_skill_tool_in_chat(self):
        from unittest import mock

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        tool_call("make_lab", '{"ref_path": "https://example.com"}')
                    ],
                ),
                fake_message(role="assistant", content="实验清单：1. 跑基线"),
                fake_message(role="assistant", content="已经生成实验清单"),
            ]
        )
        chat = ChatWorkflow(role="学习助手", skills=[self.make_lab()])
        with mock.patch("builtins.input", side_effect=["帮我做个实验", "exit"]):
            ctx = chat.build().run(agent=agent, verbose=False)

        self.assertEqual(ctx["model_reply"], "已经生成实验清单")
        tool_messages = [
            message["content"]
            for message in agent.history
            if message.get("role") == "tool"
        ]
        self.assertTrue(any("实验清单：1. 跑基线" in str(item) for item in tool_messages))

    def test_chat_workflow_without_skills_is_unchanged(self):
        from unittest import mock

        chat = ChatWorkflow(role="助手")
        built = chat.build()
        with mock.patch("builtins.input", return_value="exit"):
            ctx = built.run(agent=OrchAgent(["x"]), verbose=False)
        self.assertIsNone(ctx.get("chat_tools"))


class TestChatWorkflowSkillsMemory(MemoryTestCase):
    '''记忆怎么进 Skill：ChatWorkflow(memory=...) -> AgentHolder -> SkillContext。'''

    def make_memory(self):
        return KVMemory(self.temp_path("memory.db"))

    def test_skill_gets_the_chat_memory(self):
        from promptkit.skill.chat import AgentHolder, as_tools

        memory = self.make_memory()

        @skill(name="remember", description="记一条")
        def remember(topic: str, ctx: SkillContext) -> str:
            done = ctx.read("topics", [])
            done.append(topic)
            ctx.remember("topics", done)
            return f"记了 {len(done)} 个"

        chat = ChatWorkflow(role="助手", skills=[remember], memory=memory)
        tool = as_tools(chat.skills, holder=AgentHolder(memory=chat.memory))[0]
        self.assertEqual(tool.call({"topic": "决策树"}), "记了 1 个")
        self.assertEqual(memory.read("topics"), ["决策树"])

    def test_holder_can_pick_memory_from_a_context(self):
        from promptkit.skill.chat import AgentHolder

        memory = self.make_memory()
        holder = AgentHolder()
        holder.update({"memory": memory})
        self.assertIs(holder.memory, memory)

    def test_memory_without_config_is_allowed_when_skills_exist(self):
        memory = self.make_memory()

        @skill(name="s", description="s")
        def s(text: str) -> str:
            return text

        chat = ChatWorkflow(role="助手", skills=[s], memory=memory)
        self.assertIsNone(chat.memory_config)
        self.assertIs(chat.memory, memory)

    def test_memory_without_config_is_still_rejected_without_skills(self):
        with self.assertRaises(ValueError) as cm:
            ChatWorkflow(role="助手", memory=self.make_memory())
        self.assertIn("memory_config", str(cm.exception))

    def test_skill_without_memory_reports_clearly(self):
        ctx = SkillContext()
        with self.assertRaises(SkillError) as cm:
            ctx.remember("a", 1)
        self.assertIn("没有 memory", str(cm.exception))
        with self.assertRaises(SkillError):
            ctx.read("a")

    def test_recall_needs_a_vector_memory(self):
        ctx = SkillContext(memory=self.make_memory())
        with self.assertRaises(SkillError) as cm:
            ctx.recall("决策树")
        self.assertIn("VectorMemory", str(cm.exception))

    def test_dsl_function_body_can_use_memory(self):
        from promptkit.core import Function, FunctionStep
        from promptkit.skill.chat import AgentHolder, as_tools

        memory = self.make_memory()
        counter = Function(args=("topic",), name="count_topic", description="记一下")

        def body(ctx):
            done = ctx["memory"].read("topics", [])
            done.append(ctx["topic"])
            ctx["memory"].write("topics", done)
            return len(done)

        counter.Do(FunctionStep(body, output_key="count"))

        tool = as_tools([counter], holder=AgentHolder(agent=OrchAgent(["x"]), memory=memory))[0]
        self.assertEqual(tool.call({"topic": "集成学习"}), "1")
        self.assertEqual(memory.read("topics"), ["集成学习"])


class TestStateInheritance(MemoryTestCase):
    '''技能「继承状态」：Function(inherit=...) + AgentHolder.state 快照。'''

    def holder(self, state=None):
        from promptkit.skill.chat import AgentHolder

        return AgentHolder(agent=OrchAgent(["看到了"]), state=state or {})

    def build(self, prompt_text, **kwargs):
        from promptkit.core import Function, LLMStep, Prompt

        return Function(
            args=("q",), name="probe", description="探针", output_key="out", **kwargs
        ).Do(LLMStep(Prompt(role="user", content=prompt_text), output_key="out"))

    def test_inherit_all(self):
        from promptkit.skill.chat import as_tools

        function = self.build("档案={{profile}} 主题={{topics}}", inherit=True)
        holder = self.holder({"profile": "研一", "topics": ["决策树"]})
        tool = as_tools([function], holder=holder)[0]
        tool.call({"q": "x"})
        prompt = holder.agent.prompts[0]
        self.assertIn("研一", prompt)
        self.assertIn("决策树", prompt)

    def test_inherit_selected_keys(self):
        from promptkit.skill.chat import as_tools

        function = self.build("主题={{topics}}", inherit=("topics",))
        holder = self.holder({"profile": "研一", "topics": ["决策树"]})
        as_tools([function], holder=holder)[0].call({"q": "x"})
        self.assertIn("决策树", holder.agent.prompts[0])
        self.assertNotIn("研一", holder.agent.prompts[0])

    def test_no_inherit_by_default(self):
        from promptkit.skill.chat import as_tools

        function = self.build("{{profile}}")
        holder = self.holder({"profile": "研一"})
        result = as_tools([function], holder=holder)[0].call({"q": "x"})
        self.assertIn("错误：", result)
        self.assertIn("profile", result)

    def test_inherit_validation(self):
        from promptkit.core import Function

        with self.assertRaises(Exception):
            Function(args=(), name="x", inherit=42)

    def test_bound_function_becomes_a_zero_arg_tool(self):
        from promptkit.skill.chat import as_tools

        function = self.build("参考={{q}}", inherit=True)
        holder = self.holder({"profile": "研一"})
        tool = as_tools([function(q="https://ref")], holder=holder)[0]
        self.assertEqual(tool.name, "probe")
        self.assertEqual(tool.parameters, {"type": "object", "properties": {}})
        self.assertEqual(tool.call({}), "看到了")
        self.assertIn("参考=https://ref", holder.agent.prompts[0])

    def test_state_snapshot_skips_framework_keys(self):
        from promptkit.skill.chat import AgentHolder

        holder = AgentHolder()
        holder.snapshot(
            {"profile": "研一", "agent": object(), "memory": object(), "chat_tools": []}
        )
        self.assertEqual(holder.state, {"profile": "研一"})

    def test_skill_context_can_read_state(self):
        from promptkit.skill.chat import AgentHolder, as_tools

        @skill(name="whoami", description="看看档案")
        def whoami(ctx: SkillContext) -> str:
            return f"档案：{ctx.get('profile', '空')}"

        holder = AgentHolder(agent=OrchAgent(["x"]), state={"profile": "研一"})
        self.assertEqual(as_tools([whoami], holder=holder)[0].call({}), "档案：研一")

    def test_chat_workflow_snapshots_state_each_turn(self):
        from unittest import mock

        from promptkit.core import Function, FunctionStep

        EchoInput = Function(
            args=(), name="echo_input", description="回显本轮输入", inherit=True
        ).Do(FunctionStep(lambda ctx: ctx["usr_input"], output_key="out"))

        agent = fake_agent(
            [
                fake_message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call("echo_input", "{}")],
                ),
                fake_message(role="assistant", content="收到"),
            ]
        )
        chat = ChatWorkflow(role="助手", skills=[EchoInput])
        with mock.patch("builtins.input", side_effect=["今天学了决策树", "exit"]):
            chat.build().run(agent=agent, verbose=False)

        tool_messages = [
            str(message.get("content"))
            for message in agent.history
            if message.get("role") == "tool"
        ]
        self.assertTrue(any("今天学了决策树" in item for item in tool_messages))
        self.assertIn("skills_state", str(chat.build().render()))


class TestSkillExecute(PromptkitTestCase):
    def test_executes_with_ctx_and_records_trace(self):
        @skill(name="echo", description="回声")
        def echo(text: str, ctx: SkillContext) -> str:
            return f"{text}!"

        registry = SkillRegistry([echo])
        ctx = SkillContext()
        self.assertEqual(registry.execute("echo", {"text": "hi"}, ctx=ctx), "hi!")
        self.assertEqual(ctx.trace[0]["event"], "echo")
        self.assertEqual(ctx.trace[0]["status"], "ok")

    def test_creates_a_default_context(self):
        @skill(name="echo", description="回声")
        def echo(text: str) -> str:
            return text

        registry = SkillRegistry([echo])
        self.assertEqual(registry.execute("echo", {"text": "hi"}), "hi")

    def test_validation_can_be_skipped(self):
        @skill(name="echo", description="回声")
        def echo(text: str) -> str:
            return text

        registry = SkillRegistry([echo])
        with self.assertRaises(SkillValidationError):
            registry.execute("echo", {})
        # validate=False 会直接进 handler，缺参数就让 Python 自己报错（包成执行错误）
        with self.assertRaises(SkillExecutionError):
            registry.execute("echo", {}, validate=False)

    def test_handler_exception_is_wrapped(self):
        @skill(name="boom", description="炸")
        def boom(text: str) -> str:
            raise ValueError("坏了")

        registry = SkillRegistry([boom])
        ctx = SkillContext()
        with self.assertRaises(SkillExecutionError) as cm:
            registry.execute("boom", {"text": "x"}, ctx=ctx)
        self.assertIn("坏了", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertEqual(ctx.trace[0]["status"], "error")

    def test_unknown_skill(self):
        with self.assertRaises(SkillNotFound):
            SkillRegistry().execute("nope", {})


class TestSkillContext(PromptkitTestCase):
    def test_ask_and_structured(self):
        agent = OrchAgent(['{"status": "ok"}'])
        ctx = SkillContext(agent=agent)
        self.assertEqual(ctx.model, agent.model if hasattr(agent, "model") else None)
        self.assertEqual(ctx.structured("给我个结构", Order).status, "ok")
        self.assertIn("给我个结构", agent.prompts[0])

    def test_structured_does_not_render_braces(self):
        agent = OrchAgent(['{"status": "ok"}'])
        SkillContext(agent=agent).structured("数据：{{x}}", Order)
        self.assertIn("{{x}}", agent.prompts[0])

    def test_ask(self):
        agent = OrchAgent(["答复"])
        self.assertEqual(SkillContext(agent=agent).ask("问题"), "答复")

    def test_run_workflow(self):
        agent = OrchAgent(["不该被调用"])
        ctx = SkillContext(agent=agent)
        result = ctx.run(Workflow(Set("x", "值"), Echo("x={{x}}")), )
        self.assertEqual(result["x"], "值")
        self.assertEqual(ctx.trace[-1]["event"], "workflow")

    def test_call_skill_needs_registry(self):
        with self.assertRaises(SkillError) as cm:
            SkillContext().call_skill("x")
        self.assertIn("注册表", str(cm.exception))

    def test_tool_lookup(self):
        ctx = SkillContext(tools={"echo": lambda text: text})
        self.assertTrue(callable(ctx.tool("echo")))
        with self.assertRaises(SkillError):
            ctx.tool("nope")

    def test_require_agent_uses_default(self):
        agent = Agent(api_key="sk-test", register_as_default=False)
        Agent.set_default(agent)
        try:
            self.assertIs(SkillContext().require_agent(), agent)
        finally:
            Agent.set_default(None)

    def test_require_agent_without_anything(self):
        Agent.set_default(None)
        with self.assertRaises(SkillError):
            SkillContext().require_agent()

    def test_from_agent_and_repr(self):
        agent = Agent(api_key="sk-test", register_as_default=False)
        ctx = SkillContext.from_agent(agent)
        self.assertIs(ctx.agent, agent)
        self.assertIs(ctx.client, agent.client)
        self.assertIn("SkillContext", repr(ctx))

    def test_trace_records(self):
        ctx = SkillContext()
        entry = ctx.record("custom", value=1)
        self.assertEqual(entry, {"event": "custom", "value": 1})
        self.assertEqual(ctx.trace, [entry])


class TestSkillCallingSkill(PromptkitTestCase):
    def test_composite_skill(self):
        @skill(name="summarize", description="压缩")
        def summarize(text: str, limit: int = 3) -> str:
            return text[:limit]

        @skill(name="report", description="先摘要再下结论")
        def report(text: str, ctx: SkillContext) -> str:
            brief = ctx.call_skill("summarize", text=text, limit=2)
            return f"{brief}…很好"

        registry = SkillRegistry([summarize, report])
        self.assertEqual(
            registry.execute("report", {"text": "abcdef"}), "ab…很好"
        )

    def test_ctx_registry_is_wired(self):
        seen = {}

        @skill(name="inner", description="内层")
        def inner(ctx: SkillContext) -> str:
            seen["registry"] = ctx.skills
            return "ok"

        registry = SkillRegistry([inner])
        registry.execute("inner", {})
        self.assertIs(seen["registry"], registry)


class TestSkillEntryPoints(PromptkitTestCase):
    def test_loads_register_function_and_skill(self):
        from unittest import mock

        def register(registry):
            @skill(name="plugged", description="插件")
            def plugged(text: str) -> str:
                return text

            registry.register(plugged)

        class Entry:
            def __init__(self, name, target):
                self.name = name
                self.target = target

            def load(self):
                return self.target

        entries = [
            Entry("plugged", register),
            Entry("direct", SkillSpec("direct", "直接", lambda: None)),
            Entry("broken", None),
        ]
        with mock.patch("importlib.metadata.entry_points", return_value=entries):
            registry = SkillRegistry()
            loaded = registry.load_entry_points()
        self.assertEqual(loaded, 2)
        self.assertIn("plugged", registry)
        self.assertIn("direct", registry)
        self.assertEqual(len(registry.errors), 1)
        self.assertEqual(registry.errors[0]["name"], "broken")

    def test_broken_register_function_is_recorded(self):
        from unittest import mock

        class Entry:
            name = "bad"

            def load(self):
                def register(registry):
                    raise RuntimeError("注册失败")

                return register

        with mock.patch("importlib.metadata.entry_points", return_value=[Entry()]):
            registry = SkillRegistry()
            self.assertEqual(registry.load_entry_points(), 0)
        self.assertIn("注册失败", registry.errors[0]["error"])

    def test_no_plugins(self):
        from unittest import mock

        with mock.patch("importlib.metadata.entry_points", return_value=[]):
            self.assertEqual(SkillRegistry().load_entry_points(), 0)


class TestSkillLayering(PromptkitTestCase):
    def test_core_does_not_import_skill(self):
        '''Skill 是上层：core 不能反过来依赖它。'''
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        pattern = re.compile(
            r"^(?:from\s+(promptkit[\w.]*)\s+import|import\s+(promptkit[\w.]*))",
            re.MULTILINE,
        )
        offenders = []
        for path in sorted((root / "core").rglob("*.py")):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                dependency = match.group(1) or match.group(2)
                if "promptkit.skill" in dependency:
                    offenders.append(f"{path.name}: {dependency}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
