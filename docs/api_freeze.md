# promptkit 公开 API 冻结清单（阶段 0）

> 本文件是**契约**，不是教程。重构期间只允许在「新增」方向扩展，
> 已有名字的语义 / 签名变化必须先进入 `deprecated` 区并在下个版本移除。
>
> 标记：
> - `stable`：承诺向后兼容；改签名 = 破坏性变更。
> - `deprecated`：仍可用，但会在下一个大版本移除；新代码不得使用。
> - `experimental`：可能变；不强求兼容。

## 顶层包

| 导入路径 | 状态 | 说明 |
| --- | --- | --- |
| `promptkit.core` | stable | DSL 门面（Step / Workflow / 表达式 / 聊天 / 设置） |
| `promptkit.agent` | stable | `Agent` / `run_prompt` / `load_env_file` |
| `promptkit.prompt` | stable | `Prompt` / `render_text` |
| `promptkit.session` | stable | 旧 `Session`（会话文件读写） |
| `promptkit.conversation` | stable | 新对话状态层（`Session` / `Message` / CRUD Step） |
| `promptkit.memory` | stable | 记忆接口与实现 |
| `promptkit.skill` | stable | Skill 定义、注册表、聊天桥接 |
| `promptkit.coding` | stable | 编码 Agent 工具集 |
| `promptkit.orchestrate` | experimental | 多 Agent 编排；`API_VERSION` 仍在演进 |
| `promptkit.mcp` | experimental | 依赖可选包 `mcp`，只支持 Streamable HTTP |
| `promptkit.knowledge` | stable | 本地知识库（已移出实验开关） |
| `promptkit.features` | stable | 通用实验开关机制（当前无内置功能） |
| `promptkit.api` | stable | `Hooks` / 断言 |
| `promptkit.cautions` | stable | 内置约束与预设 |
| `promptkit.client_sys` | stable | 多厂商客户端与注册表 |
| `promptkit.types` | stable | 公共类型别名 |

## DSL 核心原语（`promptkit.core`）

### 步骤与执行

| 名字 | 状态 |
| --- | --- |
| `Step` / `coerce_steps` / `execute_step` | stable |
| `Workflow` / `Context` | stable |
| `FunctionStep` / `MessageStep` | stable |

### 逻辑与表达式

| 名字 | 状态 |
| --- | --- |
| `If` / `While` / `For` / `ForEach` / `DoWhile` / `Switch` | stable |
| `Break` / `Continue` / `Label` / `Goto` | stable |
| `Var` / `NewVar` / `Set` / `Inc` / `Dec` | stable |
| `Expr` / `as_expr` / `Const` / `VarExpr` / `BinaryExpr` | stable |
| `Call` | **deprecated（阶段 3 起）** — 改用 `Transform` |
| `Len` / `ItemExpr` / `AttrExpr` | stable |
| `Intent` / `Similar` / `Classify` / `Judge` / `LLMJudge` | stable |
| `LoopHooks` / `loop_hooks` | stable |

### 函数

| 名字 | 状态 |
| --- | --- |
| `Function(args=..., name=...)` | **deprecated（阶段 4 起）** — 改用位置形参 |
| `FunctionDef.Do` / `BoundStep` | stable |
| `Return` / `ThisFunc` | stable |

### 聊天

| 名字 | 状态 |
| --- | --- |
| `ChatWorkflow` / `chat_workflow` | stable |
| `ChatLoop` / `ChatTurn` / `Behavior` | stable |
| `Role` / `Want` / `Rules` / `Tools` / `Deliverables` / `ExitWords` | stable |
| `SystemPrompt` / `Compact` | stable |
| `MultilineInput` | stable（若不存在则按阶段 1 新增） |
| `OnError` | **新增（阶段 5）** |

## 对话状态层（`promptkit.conversation`）

| 名字 | 状态 |
| --- | --- |
| `Message`（frozen + uuid4 id） | stable |
| `Session`（CRUD / clone / store） | stable（签名在阶段 2 扩展） |
| `SessionState` | stable |
| `SessionStore` / `InMemorySessionStore` / `SQLiteSessionStore` | stable |
| `SessionAppend` / `SessionInsert` / `SessionRemove` / `SessionTruncate` / `SessionUpdate` / `SessionClear` | stable |
| `SessionPop` | **deprecated（阶段 2 起）** — 改用 `SessionRollback` |
| `SessionRollback` | **新增（阶段 2）** |
| `Branch` / `Branches` / `Fork` / `Merge` / `Restore` / `Snapshot` | stable |

阶段 2 会新增（不破坏旧方法的前提下）：

```text
Session.append(role, content, metadata=None) -> str
Session.insert(index, role, content, metadata=None) -> str
Session.update(message_id, content=None, metadata=None) -> None
Session.at(index) -> Message | None
Session.list(start=0, end=None) -> list[Message]
Session.last(n=1) -> list[Message]
Session.count() -> int
Session.to_list() -> list[dict]
Session.snapshot() -> dict
Session.from_snapshot(snap, store=None) -> Session
```

## 设置与配置

| 名字 | 状态 |
| --- | --- |
| `Settings`（frozen 快照） | stable（阶段 1 改为 frozen） |
| `settings` 读属性 | stable |
| `settings.xxx = ...` 赋值 | **deprecated（阶段 1 起）** — 改用 `Workflow(max_iterations=...)` |
| `using_settings` / `reset_settings` / `settings_names` | stable |
| `config.load_env_file` / `ENV_VAR` / `ENV_FILE` | stable |

## 工具与集成

| 名字 | 状态 |
| --- | --- |
| `Tool` / `tool` / `as_tool` / `as_tools` / `ToolRegistry` | stable |
| `RUN_COMMAND` / `WEB_SEARCH` / `WriteFile` | stable |
| `ReadFileTool` / `WriteFileTool` / `ApplyDiffTool` / `ListDirTool` / `SearchCodeTool` / `RunCommandTool` / `RunTestsTool` | stable |
| `MCPTool` / `load_mcp_tools` / `load_mcp_from_config` | experimental |

## 记忆

| 名字 | 状态 |
| --- | --- |
| `Memory` / `MemoryItem` / `InMemoryMemory` | stable |
| `KVMemory` / `SessionMemory` / `VectorMemory` / `LayeredMemory` | stable |
| `EmbeddingClient` / `VectorStore` / `Reranker` | stable |
| `TokenBudget` / `estimate_tokens` | stable |
| `RecallMemoryTool` / `WriteMemoryTool` | stable |

## Skill

| 名字 | 状态 |
| --- | --- |
| `skill` / `SkillSpec` / `build_spec` | stable |
| `SkillRegistry` / `SkillContext` | stable |
| `SkillStep` / `SkillTool` / `as_tools` | stable |

## 编码

| 名字 | 状态 |
| --- | --- |
| `CodingStep` / `FixLoop` | stable |
| `Sandbox` / `run_command` | stable |
| `parse_patch` / `apply_hunks` / `apply_patch_text` | stable |
| `CodingError` / `DiffError` / `SandboxError` / `CommandError` | stable |

## 客户端

| 名字 | 状态 |
| --- | --- |
| `BaseClient` / `ChatClient` / `ClientError` | stable |
| `OpenAIClient` / `DoubaoClient` / `AnthropicClient` | stable |
| `PROVIDERS` / `register_vendor` / `make_client` | stable |
| `resolve_api_key` / `resolve_target` / `detect_vendor` | stable |
| 多模态：`BaseMultimodalClient` / `make_multimodal_client` / `MULTIMODAL_PROVIDERS` | stable |

## 编排（experimental）

| 名字 | 状态 |
| --- | --- |
| `Task` / `Plan` / `Orchestrator` / `Planner` | experimental |
| `sub_agent` / `DelegateTool` / `PlanTool` | experimental |
| `API_VERSION` / `SUPPORTED_VERSIONS` | experimental |

## 阶段路线图与兼容承诺

| 阶段 | 内容 | 会触发的 deprecation |
| --- | --- | --- |
| 0 | API 冻结、基线测试、密钥外置 | 无 |
| 1 | Settings 去全局化、ChatLoop 阶段化 | `settings.xxx = ...` |
| 2 | Session 新签名、`SessionPop` → `SessionRollback` | `SessionPop`、旧 `Session` 位置参数写法 |
| 3 | `Call` → `Transform` | `Call`（lambda / 闭包直接报错） |
| 4 | `Function` 去 `args` 元组与 `name` | `Function(args=..., name=...)` |
| 5 | `OnError`、Trace 增强 | 无 |

**基线测试**：`tests/baseline/`。每个阶段结束必须全绿，且
`examples/course_assistant.py --fake`（课程学习助手，tutor 场景）仍可跑通。
