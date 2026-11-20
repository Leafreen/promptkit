"""aipykit：一个极简的大模型提示词 / 工作流工具包。

公开 API 分两层，两层都受支持：

1. **根命名空间聚合**（推荐）—— 所有子包的公开符号都汇总到这里，
   不用记深层路径：

       from aipykit import Agent, Workflow, Prompt, Tool, Echo
       from aipykit import *            # 692 个公开名字全部可用

2. **子包导入**（一直可用，保持向后兼容）—— 老的深层路径不受影响：

       from aipykit.core import Workflow
       from aipykit.mcp import MCPTool

实现要点
--------
- 根 `__init__` 本身不导入任何子包；每个符号都在第一次访问时才从对应的
  子包取，所以 ``import aipykit`` 几乎零开销，可选依赖缺失也不会导致整体
  导入失败。
- ``__all__`` 与 ``_EXPORT_GROUPS`` 是唯一的事实来源：测试会对照各子包
  的 ``__all__`` 校验有无遗漏或多余。
- **重名规则**：同名符号按 `_EXPORT_GROUPS` 的先后顺序归属，先到先得。
  例如 ``Message`` 来自 agent（即 ``aipykit.types.Message``）、
  ``Session`` 来自 ``aipykit.session``、``ContentPart`` 来自 ``aipykit.core``。
  想要被覆盖的那一份，直接走它所在的子包即可。
- 例外：``features`` / ``skill`` 在根命名空间指向**模块**，这样才能
  ``from aipykit import features; features.enable(...)``，也不会破坏
  ``import aipykit.skill.package`` 这类深层导入；``@skill`` 装饰器用
  ``from aipykit.skill import skill`` 取。
- 可选依赖（如 MCP SDK、TUI 依赖）未安装时，显式访问对应符号会抛出带
  安装提示的 ``AttributeError``；``from aipykit import *`` 会跳过它们，
  其余功能照常可用。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: 立即子包 / 顶层模块：支持 `aipykit.core`、`aipykit.config` 这类惰性访问。
_SUBMODULES: dict[str, str] = {
    "agent": "aipykit.agent",
    "agent_factory": "aipykit.agent_factory",
    "api": "aipykit.api",
    "cautions": "aipykit.cautions",
    "client_sys": "aipykit.client_sys",
    "coding": "aipykit.coding",
    "config": "aipykit.config",
    "conversation": "aipykit.conversation",
    "core": "aipykit.core",
    "dsl_libs": "aipykit.dsl_libs",
    "executor": "aipykit.executor",
    "features": "aipykit.features",
    "knowledge": "aipykit.knowledge",
    "mcp": "aipykit.mcp",
    "memory": "aipykit.memory",
    "orchestrate": "aipykit.orchestrate",
    "prompt": "aipykit.prompt",
    "session": "aipykit.session",
    "skill": "aipykit.skill",
    "tui": "aipykit.tui",
    "types": "aipykit.types",
}

#: 这些名字优先解析成**模块**（即使子包里也有同名符号）。
#: `skill` 若解析成装饰器，`import aipykit.skill.package as x` 会拿不到包。
_MODULE_ALIASES = frozenset({"features", "skill"})

#: 公开符号 -> 来源模块。顺序即重名优先级。
_EXPORT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ---- aipykit.prompt ----
    ("aipykit.prompt", (
        "Prompt", "render_messages", "render_text", "resolve_variable",
        "SUPPORTED_ROLES",
    )),
    # ---- aipykit.agent ----
    ("aipykit.agent", (
        "Agent", "Message", "ToolLoopExceeded", "run_prompt", "load_env_file",
        "normalize_history", "ENV_VAR", "ENV_FILE",
    )),
    # ---- aipykit.agent_factory ----
    ("aipykit.agent_factory", (
        "SubAgentFactory", "SubAgentError", "UnknownRoleError",
    )),
    # ---- aipykit.executor ----
    ("aipykit.executor", (
        "Executor", "run_structured", "OutputValidationError", "MaxRetriesExceeded",
    )),
    # ---- aipykit.config ----
    ("aipykit.config", (
        "PROJECT_ROOT", "DEFAULT_START_PROMPTS", "DEFAULT_MODEL", "DEFAULT_BASE_URL",
        "DEFAULT_REASONING_EFFORT", "DEFAULT_EXTRA_BODY", "DOUBAO_API_KEY_ENV",
        "DOUBAO_BASE_URL", "DOUBAO_TEXT_MODEL", "DOUBAO_IMAGE_MODEL",
        "DOUBAO_VIDEO_MODEL", "MULTIMODAL_BASE_URL", "MULTIMODAL_ENDPOINT",
        "MULTIMODAL_TIMEOUT", "resolve_api_key", "api_key_source",
    )),
    # ---- aipykit.session ----
    ("aipykit.session", (
        "Session", "SESSION_VERSION", "as_session", "set_system_prompt_reset",
    )),
    # ---- aipykit.knowledge ----
    ("aipykit.knowledge", (
        "FORMAT_VERSION", "KNOWLEDGE_REFERENCE", "Hit", "KnowledgeBase", "chunk_text",
        "token_weight", "tokenize",
    )),
    # ---- aipykit.features ----
    ("aipykit.features", (
        "FEATURE_ENV", "FeatureDisabled", "disable", "enable", "enabled", "features",
        "require", "status", "using",
    )),
    # ---- aipykit.core（DSL 核心，大部分步骤 / 表达式 / 显示都在这儿）----
    ("aipykit.core", (
        "Workflow", "Context", "Step", "coerce_steps", "ContentPart", "TextPart",
        "ImagePart", "AudioPart", "VideoPart", "FilePart", "part_from_string",
        "part_from_dict", "part_from_any", "If", "IfStep", "While", "WhileStep",
        "HookPoint", "LoopHookEvent", "LoopHookHandle", "LoopHooks", "HookFailure",
        "loop_hooks", "as_hook_point", "LOOP_HOOKS_KEY", "LOOP_HOOK_POINTS",
        "LOOP_START", "BEFORE_CONDITION", "BEFORE_ITERATION", "AFTER_ITERATION",
        "LOOP_LIMIT", "LOOP_ERROR", "LOOP_END", "For", "ForStep", "ForEach",
        "ForEachStep", "DoWhile", "DoWhileStep", "Switch", "SwitchStep", "Case",
        "Break", "Continue", "Label", "LabelStep", "Goto", "GotoStep", "OnError",
        "Condition", "as_condition", "And", "Or", "Not", "When", "Parallel", "Intent",
        "Similar", "Classify", "Sem", "SemanticCondition", "Judge", "LLMJudge",
        "JudgeRequest", "Verdict", "SemanticLabel", "SemanticError", "UnknownVerdict",
        "SemanticBudgetExceeded", "SEMANTIC_STATE_KEY", "SEMANTIC_JUDGE_KEY",
        "DEFAULT_THRESHOLD", "DEFAULT_MARGIN", "DEFAULT_BUDGET", "UNKNOWN_RAISE",
        "UNKNOWN_ELSE", "FALLBACK_LABELS", "RELATIONS", "NONE_RELATION",
        "is_fallback_label", "Var", "NewVar", "Set", "VarDecl", "SetItemStep",
        "SetAttrStep", "SetStep", "Inc", "Dec", "CompoundAssignStep", "VarType",
        "Number", "Str", "Expr", "as_expr", "Const", "VarExpr", "BinaryExpr",
        "UnaryExpr", "AgentReplyExpr", "WhenExpr", "Call", "Transform", "transform",
        "register_transform", "get_transform", "transform_name",
        "registered_transforms", "CallExpr", "ItemExpr", "AttrExpr", "Len", "LLMStep",
        "ImageGenStep", "VideoGenStep", "STTStep", "TTSStep", "MessageStep",
        "FunctionStep", "ReadMemory", "WriteMemory", "ForgetMemory", "RecallMemory",
        "MemoryStep", "UseMemory", "format_recalled", "Choose", "ChooseExpr", "Confirm",
        "ConfirmExpr", "Read", "ReadExpr", "ReadFile", "ReadFileExpr", "Save",
        "SaveStep", "SaveJSON", "SaveHistory", "SaveHistoryStep", "Echo", "EchoStep",
        "Function", "FunctionDef", "BoundStep", "Return", "ReturnStep", "ThisFunc",
        "ThisFuncCall", "ThisFuncExpr", "SkipStep", "Event", "EventType", "EventBus",
        "console_display", "json_display", "collect_events", "filter_events",
        "set_console_suppressed", "Verbosity", "as_verbosity", "render_markdown",
        "render_math", "render_plain", "plain_text", "strip_markdown",
        "latex_to_unicode", "looks_like_math", "needs_art", "get_console", "RENDERERS",
        "HAVE_RICH", "HAVE_TEXICODE", "HAVE_MERMAID", "HAVE_DOT_RENDERER",
        "DIAGRAM_GUIDE", "substitute_math", "substitute_diagrams", "install_hint",
        "setup_logging", "close_logging", "get_logger", "log_display", "LOGGER_NAME",
        "DEFAULT_LOG_PATH", "DEFAULT_FORMAT", "Node", "format_tree", "short",
        "settings", "Settings", "using_settings", "reset_settings", "settings_names",
        "Caution", "CautionGroup", "Custom", "join_cautions", "resolve_cautions",
        "set_default_cautions", "add_default_caution", "get_default_cautions",
        "clear_default_cautions", "using_default_cautions", "Formatter",
        "FormatterError", "JsonFormatter", "XmlFormatter", "RegexFormatter",
        "YamlFormatter", "get_formatter", "WorkflowError", "UndefinedVariable",
        "LoopLimitExceeded", "RecursionLimitExceeded", "ArgumentError", "VarTypeError",
        "UnknownTransformError", "UnserializableCallError", "UnknownLabel",
        "GotoSignal", "ReturnSignal", "CHAT_ROLE_KEY", "CHAT_WANT_KEY",
        "CHAT_RULES_KEY", "CHAT_DELIVERABLES_KEY", "CHAT_TOOLS_KEY", "CHAT_EXIT_KEY",
        "CHAT_SIGNAL_KEY", "CHAT_INPUT_KEY", "CHAT_REPLY_KEY", "DEFAULT_EXIT_WORDS",
        "DEFAULT_COMPACT_PROMPT", "Role", "Want", "Rules", "Tools", "Deliverables",
        "ExitWords", "Behavior", "SystemPrompt", "ChatLoop", "ChatTurn", "Compact",
        "CompactStep", "MakeFile", "SaveDeliverables", "build_system_prompt",
        "save_format", "chat_context_vars", "Deliverable", "ChatWorkflow",
        "chat_workflow", "TimeStamp", "Now", "StartTimer", "StopTimer", "WaitAlarm",
        "ProactiveAlarms", "EnableTUI", "TuiUnavailableError", "SummonSubAgentTool",
    )),
    # ---- aipykit.types ----
    ("aipykit.types", (
        "JSONScalar", "JSONValue", "MessageRole", "MessageContent", "ToolCallFunction",
        "ToolCall", "PromptMessage", "PromptInput", "RecallHit", "SessionPayload",
        "ToolLike", "SupportsWrite", "MemoryLike", "AgentLike",
    )),
    # ---- aipykit.core.tool ----
    ("aipykit.core.tool", (
        "Tool", "tool", "as_tool", "as_tools", "schema_from_signature", "RUN_COMMAND",
        "WEB_SEARCH", "WebSearch", "WriteFile", "WRITE_FILE",
        "DEFAULT_ALLOWED_EXTENSIONS", "MAX_CONTENT_CHARS", "GET_TIME", "TIMER",
        "ALARMS", "AlarmBoard", "AlarmRunner", "clock_text", "now_text", "drain_alarms",
        "start_timer", "stop_timer", "stop_all_timers", "elapsed_seconds",
        "running_timers", "timer", "ReadFileTool", "WriteFileTool", "ApplyDiffTool",
        "ListDirTool", "SearchCodeTool", "RunCommandTool", "RunTestsTool",
        "default_coding_tools", "SkillTool", "result_text", "RecallMemoryTool",
        "WriteMemoryTool", "MEMORY_REFERENCE", "WRITE_MEMORY",
    )),
    # ---- aipykit.core.steps ----
    ("aipykit.core.steps", (
        "execute_step", "Branch", "T", "OUTPUT_STREAM_KEY", "DEFAULT_ENCODING",
        "INPUT_FN_KEY", "input_reader", "output_stream", "BoundCondition",
        "ConditionLike", "var_type_name", "Export",
    )),
    # ---- aipykit.core.steps.actions ----
    ("aipykit.core.steps.actions", (
        "PlanStep", "PlanWithTool", "ExecutePlan", "AggregateStep",
        "format_task_results", "CodingStep", "SkillStep",
    )),
    # ---- aipykit.core.steps.logics ----
    ("aipykit.core.steps.logics", (
        "InRange", "ErrorHandler",
    )),
    # ---- aipykit.core.model_action ----
    ("aipykit.core.model_action", (
        "WORKFLOW_CAUTIONS_KEY", "as_caution", "flatten_cautions",
    )),
    # ---- aipykit.core.event_sys ----
    ("aipykit.core.event_sys", (
        "ChildBus", "EventHandler", "extract_reasoning", "emit", "bus_of",
        "EVENT_BUS_KEY", "THINKING_START", "THINKING_DELTA", "THINKING_END",
        "TOOL_CALL_START", "TOOL_CALL_DELTA", "TOOL_CALL_END", "TOOL_RESULT",
        "MESSAGE_DELTA", "MESSAGE_END", "USER_MESSAGE", "STEP_START", "STEP_END",
        "NOTIFICATION", "SEMANTIC_JUDGE", "ERROR",
    )),
    # ---- aipykit.conversation ----
    ("aipykit.conversation", (
        "SESSION_KEY", "BranchLimitExceeded", "Branches", "Fork",
        "InMemorySessionStore", "Merge", "Restore", "RewindBeyondCompression",
        "RewindBeyondLimit", "RewindConfig", "SQLiteSessionStore", "SessionAppend",
        "SessionClear", "SessionError", "SessionInsert", "SessionPop", "SessionRemove",
        "SessionRollback", "SessionState", "SessionStore", "SessionTruncate",
        "SessionUpdate", "Snapshot", "SnapshotReadonly",
    )),
    # ---- aipykit.client_sys ----
    ("aipykit.client_sys", (
        "Choice", "Chunk", "Completion", "FunctionCall", "Usage", "BaseClient",
        "ChatClient", "ClientError", "ClientSpec", "UnsupportedModality",
        "part_data_url", "ReasoningSpec", "OFF_ALIASES", "expand_reasoning",
        "reasoning_params", "OpenAIClient", "DoubaoClient", "AnthropicClient",
        "iter_sse", "DEFAULT_API_VERSION", "PROVIDERS", "DEFAULT_VENDOR",
        "FALLBACK_ENV_VAR", "detect_vendor", "get_spec", "make_client",
        "register_vendor", "resolve_target", "vendor_names", "with_params",
    )),
    # ---- aipykit.api ----
    ("aipykit.api", (
        "Hooks", "HookEvent", "HookType", "HookHandle", "HookAbort", "HOOKS_KEY",
        "use_hooks", "Assert", "AssertStep", "AssertionFailed", "AssertionResult",
        "AssertionAdapter", "CallableAssertionAdapter", "Assertions",
        "assertion_adapters",
    )),
    # ---- aipykit.cautions ----
    ("aipykit.cautions", (
        "OutputFormat", "NoExtraText", "NoMarkdown", "Language", "MaxLength",
        "NoFabrication", "Citations", "Grounded", "Concise", "StepByStep", "AskFirst",
        "Tone", "Style", "NoText", "Quality", "SchemaHint", "JsonSchemaHint",
        "json_only", "json_strict", "json_with_schema", "concise_text", "code_only",
        "polite_reply", "grounded", "tutor", "socratic", "analyst", "translator",
        "JSON_ONLY", "JSON_STRICT", "PLAIN_TEXT", "CODE_ONLY", "GROUNDED", "TUTOR",
        "SOCRATIC", "ANALYST",
    )),
    # ---- aipykit.coding ----
    ("aipykit.coding", (
        "CodingError", "CodeFormatError", "DiffError", "SandboxError", "CommandError",
        "Sandbox", "CODING_LOGGER", "Hunk", "FilePatch", "parse_patch",
        "validate_patch", "apply_hunks", "apply_patch_text", "strip_code_fence",
        "CodeOutput", "DiffOutput", "FullFileOutput", "ChangesOutput", "get_output",
        "validate_changes", "OUTPUT_FORMATS", "FixLoop", "ReadFileStep",
        "WriteFileStep", "ApplyDiff", "ApplyChanges", "ListDir", "SearchCode",
        "apply_patch", "format_hits", "run_command", "CommandResult", "RunCommand",
        "RunTests", "LintStep", "git", "GitDiff", "GitCommit",
    )),
    # ---- aipykit.memory ----
    ("aipykit.memory", (
        "Memory", "MemoryError", "MemoryItem", "InMemoryMemory", "item_text",
        "replace_item_text", "format_memories", "empty", "KVMemory", "SessionMemory",
        "VectorMemory", "InMemoryVectorStore", "FolderVectorStore", "LayeredMemory",
        "long_term_memory", "as_memory", "StorageConfig", "create_memory",
        "register_memory_backend", "BACKENDS", "DATA_DIR_ENV", "default_data_dir",
        "default_kv_path", "default_vector_dir", "memory_dir", "safe_namespace_name",
        "expand_path", "EmbeddingClient", "VectorStore", "Reranker", "cosine",
        "TokenBudget", "estimate_tokens", "MemoryConfig", "RecallConfig",
        "WRITE_ON_VALUES", "MEMORY_LOGGER",
    )),
    # ---- aipykit.memory.adapters ----
    ("aipykit.memory.adapters", (
        "CallableEmbedding", "HashEmbedding", "OpenAIEmbedding",
        "SentenceTransformerEmbedding", "ChromaVectorStore", "QdrantVectorStore",
        "NoopReranker", "LlmReranker", "CohereReranker", "CrossEncoderReranker",
    )),
    # ---- aipykit.orchestrate ----
    ("aipykit.orchestrate", (
        "Task", "TaskResult", "Plan", "PlanResult", "Planner", "TemplatePlanner",
        "LLMPlanner", "coerce_planner", "TaskSpec", "TaskPlanModel", "Orchestrator",
        "OrchestratorConfig", "sub_agent", "RoleFactory", "structured_answer",
        "as_text", "plan_from", "DelegateTool", "PlanTool", "API_VERSION",
        "SUPPORTED_VERSIONS", "Request", "Response", "handle", "run", "plan",
        "PlanSpec", "ErrorCode", "OrchestrationError", "PlanError", "TaskError",
        "BudgetExceeded", "DepthExceeded", "DelegateError",
    )),
    # ---- aipykit.skill ----
    ("aipykit.skill", (
        "skill", "build_spec", "SkillSpec", "Skill", "as_skill", "as_skills",
        "MarkdownSkillTool", "as_skill_package", "SkillPackage", "ScriptSpec",
        "load_skills_from_dir", "merge_tools", "resolve_python", "PYTHON_ENV_VAR",
        "SkillRegistry", "SkillContext", "build_parameters", "type_schema",
        "validate_args", "SkillError", "SkillValidationError", "SkillAlreadyExists",
        "SkillNotFound", "SkillExecutionError", "SkillDependencyError",
        "InvalidSkillPackageError", "DependencyInstallError", "ScriptNotFoundError",
        "ResourceNotFoundError", "PathEscapeError", "DuplicateToolError",
        "PythonInterpreterError", "orchestrate_run", "orchestrate_plan",
        "orchestrate_validate", "OrchestrateSkillSet", "register_orchestrate_skills",
    )),
    # ---- aipykit.mcp（MCP SDK 是可选依赖）----
    ("aipykit.mcp", (
        "MCPError", "MCPConnectionError", "MCPToolNotFound", "MCPTool",
        "build_mcp_tools", "load_mcp_tools", "load_mcp_from_config", "HookRegistry",
        "ToolHookResult", "get_default_hook_registry", "set_default_hook_registry",
        "mcp_permission_gate", "register_mcp_security",
    )),
    # ---- aipykit.tui（TUI 依赖是可选依赖）----
    ("aipykit.tui", (
        "start_tui", "TuiSession",
    )),
)

#: 名字 -> 来源模块（重名时保留先出现的那个）。
_EXPORTS: dict[str, str] = {}
for _module, _names in _EXPORT_GROUPS:
    for _name in _names:
        _EXPORTS.setdefault(_name, _module)
del _module, _names, _name

__all__: list[str] = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """按需从对应子包取符号（PEP 562）。

    - 子包名（``core`` / ``memory`` …）惰性 import 成模块；
    - 公开符号按 ``_EXPORTS`` 从来源模块取，取到后缓存进 ``globals()``；
    - 可选依赖缺失时转成带提示的 ``AttributeError``，
      这样 ``from aipykit import *`` 会跳过它而不是整体失败。
    """
    if name in _MODULE_ALIASES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module

    module_name = _EXPORTS.get(name)
    if module_name is None:
        module_name = _SUBMODULES.get(name)
        if module_name is None:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}；"
                f"用 dir(aipykit) 查看全部 {len(__all__)} 个公开名字"
            )
        module = import_module(module_name)
        globals()[name] = module
        return module

    try:
        value = getattr(import_module(module_name), name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] == "aipykit":
            raise
        raise AttributeError(
            f"aipykit.{name} 需要可选依赖 {exc.name!r}；"
            f"请按 pyproject.toml 的 optional-dependencies 安装后再用"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全看到全部公开符号与子包。"""
    return sorted({*globals(), *__all__, *_SUBMODULES})
