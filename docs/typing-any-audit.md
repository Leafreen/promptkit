# `Any` 类型审计与重构方案

> 范围：`src/promptkit/` 全部 118 个 `.py` 文件。
> 生成方式：`ast` 逐节点扫描（`.tmp/any_audit.json`）+ mypy 基线。
> 状态：**只出清单和方案，尚未改代码**。

## 1. 结论

- 项目是 **Python**，对应关系是：TypeScript `any` ≈ `typing.Any`；
  `unknown` 没有直接对应，最接近的是 **`object`**（配合 `isinstance` 收窄或 `cast`）。
- 共 **1047 处 `Any`**，分布在 **87 个文件**（另有 31 个文件无 `Any`）：
  - 函数参数 **641** 处
  - 返回值 **207** 处
  - 类/对象字段 **124** 处
  - 局部变量 / 模块变量 **71** 处
  - 非注解用法（`cast(Any, ...)` 等）**4** 处
- 其中约 **80% 可以精确化**：JSON 边界、消息、工具 schema、技能规格、记忆元数据。
- 约 **20% 是刻意的动态边界**（DSL 求值结果、`Context` 任意运行时对象、装饰器透传），
  这些应换成 `object` + `Protocol`/`cast`，而不是硬造一个假精确类型。
- 当前 mypy 基线（`check_untyped_defs = false`）：**53 errors / 20 files**。
  其中若干是真实隐患（缺 import、缺 return、返回类型错配），应在第一阶段先修掉。

## 2. 现状数据

### 2.1 按位置

| 位置 | 数量 | 典型写法 |
| --- | ---: | --- |
| 参数 | 641 | `def ask(self, tools: Optional[Sequence[Any]] = None)` |
| 返回 | 207 | `def eval(self, context: Context) -> Any` |
| 字段 | 124 | `params: Dict[str, Any]` |
| 局部/模块 | 71 | `_MISSING: Any = object()` |
| 运行时（非注解） | 4 | `cast(Any, ...)` |

### 2.2 按代码形态（粗略）

| 形态 | 数量 |
| --- | ---: |
| 裸 `Any` 注解（`: Any` / `-> Any`） | 466 + 163 |
| `Dict[str, Any]` | 246 |
| `List[Any]` / `Sequence[Any]` / `Tuple[..., Any]` | 69 |
| `Optional[Any]` | 17 |
| `Callable[..., Any]` | 13 |

### 2.3 mypy 基线（53 errors）

| 错误码 | 数量 | 说明 |
| --- | ---: | --- |
| `arg-type` | 12 | 参数类型不匹配（多为 `Any \| None`） |
| `name-defined` | 11 | `render.py` 缺 `List` 导入；`chat_util.py` 缺 `Tuple` 导入 |
| `union-attr` | 11 | `Session \| None`、`Planner \| None` 未判空 |
| `assignment` | 4 | `render.py` 的 frozenset/元组赋值；`hooks.py` 元组长度 |
| `attr-defined` | 4 | `BaseModel.label`；`Step.prompt_content` |
| `operator` | 3 | `if_step.py` 元组 +；`embedding.py` 可选 int |
| `return-value` | 2 | `semantic.py` 的 `Intent` / `PlanResult` |
| 其他 | 6 | `index` / `no-redef` / `return` / `misc` / `var-annotated` |

**必须优先修的真实问题**：

1. `core/display_sys/render.py` 缺 `List` 导入（10 处）。
2. `core/steps/actions/chat_util.py` 缺 `Tuple` 导入（1 处）。
3. `core/tool/builtin/write_file.py` 缺 `return`。
4. `core/steps/logics/semantic.py`：`Classify._wrap` 收到 `SemanticCondition` 却声明 `Intent`；
   `BaseModel` 上访问 `.label` / `.confidence`。
5. `core/steps/workflow.py`：`session` 可能为 `None` 却直接 `.load()`。
6. `memory/adapters/embedding.py`：`dimension: Optional[int]` 参与 `*` / `%`。
7. `memory/adapters/store.py`：`expand_path(str | None)`。

## 3. 类型工具包（建议先落地）

新增 `src/promptkit/types.py`（不依赖任何子包，避免循环导入），各层再从这里取。

### 3.1 JSON 与消息

```python
# src/promptkit/types.py
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Protocol, Sequence, TypeAlias, TypedDict, Union

JSONScalar: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]

MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool", "function"]
ContentPart: TypeAlias = Dict[str, JSONValue]        # 多模态分段
MessageContent: TypeAlias = Union[str, None, List[ContentPart]]

class ToolCallFunction(TypedDict):
    name: str
    arguments: str

class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction

class _MessageRequired(TypedDict):
    role: MessageRole
    content: MessageContent

class Message(_MessageRequired, total=False):
    '''OpenAI 风格消息；可选键用 `total=False` 继承（兼容 Python 3.10）。'''
    tool_calls: List[ToolCall]
    tool_call_id: str
    name: str

class RecallHit(TypedDict):
    text: str
    score: float

class SessionPayload(TypedDict):
    version: int
    messages: List[Message]
    variables: Dict[str, JSONValue]
    saved_at: str
```

### 3.2 鸭子类型协作者 → `Protocol`

```python
ToolLike: TypeAlias = Union["Tool", Callable[..., object]]

class SupportsWrite(Protocol):
    def write(self, key: str, value: object) -> None: ...

class MemoryLike(SupportsWrite, Protocol):
    def read(self, key: str, default: object = None) -> object: ...
    def recall(self, query: str, top_k: int = 5) -> Sequence[Dict[str, object]]: ...

class AgentLike(Protocol):
    history: List[Message]
    start_prompts: List[Message]
    def ask(self, prompt: object, tools: Optional[Sequence[ToolLike]] = None, **kwargs: object) -> str: ...
    def reset(self) -> None: ...
```

- `MemoryLike` 放在 `types.py`：`chat_util` / `Compact` 只依赖协议，不必 import 具体记忆包。
- `ToolLike` 里的 `"Tool"` 用 `TYPE_CHECKING` 前向引用，避免 `types.py → core.tool.base` 的循环。
- `StepLike` 留在 `core/steps/interface.py`：

```python
StepLike: TypeAlias = Union[Step, Sequence[Step]]
```

### 3.3 装饰器 → `ParamSpec`

`tool()` / `skill()` 这类「可带参可不带参」的装饰器，用重载 + `ParamSpec` 保留下游函数签名：

```python
P = ParamSpec("P")
R = TypeVar("R")

@overload
def tool(func: Callable[P, R]) -> Tool: ...
@overload
def tool(*, name: str = ..., description: str = ...) -> Callable[[Callable[P, R]], Tool]: ...
```

不要继续用 `*args: Any, **kwargs: Any`——那会把装饰后函数的签名整个抹掉。

### 3.4 领域 TypedDict（按层定义，不放进 `types.py`）

| 层 | 类型 | 字段 |
| --- | --- | --- |
| 工具 | `ToolParameter` | `type`, `description`, `enum`, `items`, `default` |
| 工具 | `ToolSchema` | `type: Literal["object"]`, `properties: Dict[str, ToolParameter]`, `required: List[str]` |
| 技能 | `SkillSpecData` | `name`, `description`, `parameters: ToolSchema`, `returns: ToolSchema \| None`, `examples: List[Dict[str, JSONValue]]`, `metadata: Dict[str, JSONValue]` |
| 编排 | `TaskData` / `PlanData` | `id`, `title`, `instruction`, `depends_on: List[str]` |
| 客户端 | `ChatRequest` | `model`, `messages: List[Message]`, `temperature: float`, `tools: List[ToolSchema]` |
| 客户端 | `ChatResponse` | `id`, `choices`, `usage` |

## 4. 替换决策树

```text
这处 Any 是注解吗？
├─ 否（cast / isinstance / 运行时） → 保留但改成 cast(Any→具体类型) 或收窄
└─ 是
   ├─ 数据进出 API / 文件 / 记忆元数据 / session → JSONValue 或对应 TypedDict
   ├─ Context 里的运行时对象（agent / memory / step） → object + 具名 getter
   ├─ DSL 求值结果（Expr.eval / Var / Call）
   │    → object（真正的 unknown）+ 使用处 isinstance 收窄 / cast
   ├─ 鸭子类型协作者（memory / agent / tool / assertion） → Protocol
   ├─ 装饰器透传函数 → ParamSpec + TypeVar
   └─ 以上都不是 → 重新设计具体类型；**不再新增 Any**
```

三条硬规则：

1. **JSON 边界不允许 `Any`**：一律 `JSONValue` / `TypedDict`，反序列化后用 `isinstance` 校验。
2. **`object` 不是终点**：取用时必须 `isinstance` 收窄或 `cast(具体类型, value)`；
   直接 `obj.attr` 会被 mypy 拦住，这正是我们要的。
3. **pydantic 模型字段保持真实类型**：`coding/`、`semantic.py` 的 `BaseModel`
   参与运行时校验，不能改成 `object` / `JSONValue`。

## 5. 逐文件清单

| 文件 | 总数 | 参数 | 返回 | 字段 | 局部/模块 | 目标策略 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `core/steps/logics/expr_sys/expr.py` | 53 | 40 | 10 | 0 | 3 | DSL 值域：Value=object + cast / ConditionLike |
| `skill/spec.py` | 43 | 24 | 11 | 5 | 3 | 技能层：SkillSpec TypedDict / Callable[..., object] |
| `agent.py` | 38 | 22 | 6 | 3 | 7 | 对外门面：Message TypedDict / ToolLike / AgentLike |
| `client_sys/anthropic_client.py` | 35 | 22 | 9 | 0 | 4 | 协议层：请求/响应 TypedDict + JSONValue |
| `skill/context.py` | 32 | 13 | 11 | 8 | 0 | 技能层：SkillSpec TypedDict / Callable[..., object] |
| `core/steps/actions/chat_util.py` | 30 | 28 | 2 | 0 | 0 | Step 输入：StepLike / DSLLiteral / AgentLike |
| `core/steps/actions/memory.py` | 27 | 10 | 2 | 15 | 0 | MemoryLike / DSLLiteral |
| `core/steps/logics/semantic.py` | 27 | 25 | 1 | 1 | 0 | DSL 值域：LabelLike / ConditionLike |
| `api/hooks.py` | 24 | 18 | 1 | 5 | 0 | HookHandler = Callable[..., object] |
| `core/tool/builtin/web_search.py` | 24 | 13 | 8 | 1 | 2 | JSONValue / 后端响应 TypedDict |
| `core/steps/workflow.py` | 23 | 21 | 0 | 0 | 2 | StepLike / RunOptions TypedDict |
| `core/tool/memory.py` | 22 | 13 | 3 | 3 | 3 | JSONValue / ToolSchema / MemoryLike |
| `memory/storage.py` | 22 | 21 | 0 | 1 | 0 | MemoryLike / JSONValue |
| `skill/markdown.py` | 22 | 11 | 7 | 1 | 3 | SkillSpec TypedDict / Callable[..., object] |
| `memory/base.py` | 21 | 10 | 7 | 2 | 2 | MemoryLike / JSONValue |
| `skill/chat.py` | 21 | 11 | 8 | 1 | 1 | SkillSpec TypedDict / Callable[..., object] |
| `core/steps/actions/orchestrate.py` | 20 | 4 | 3 | 13 | 0 | 编排 Step：StepLike + TaskData |
| `core/steps/chat.py` | 20 | 7 | 0 | 10 | 3 | 工作流编排：StepLike / 配置 TypedDict |
| `core/tool/base.py` | 20 | 13 | 2 | 2 | 3 | ToolSchema TypedDict / Callable[..., object] |
| `orchestrate/api.py` | 20 | 11 | 2 | 7 | 0 | TaskData / PlanData / JSONValue |
| `core/steps/logics/var.py` | 18 | 14 | 4 | 0 | 0 | DSL 值域：object + cast |
| `orchestrate/orchestrator.py` | 18 | 11 | 6 | 0 | 1 | TaskData / PlanData |
| `skill/schema.py` | 18 | 9 | 4 | 0 | 3 | SkillSpec TypedDict / ParamSpec |
| `skill/orchestrate_skill.py` | 17 | 10 | 6 | 0 | 1 | SkillSpec TypedDict |
| `api/assertions.py` | 16 | 15 | 0 | 1 | 0 | AssertionLike Protocol |
| `orchestrate/dsl.py` | 16 | 8 | 5 | 2 | 1 | TaskData / PlanData |
| `core/tool/coding.py` | 15 | 4 | 11 | 0 | 0 | JSONValue / ToolSchema |
| `core/display_sys/render.py` | 14 | 10 | 3 | 0 | 1 | object（渲染任意内容） |
| `core/steps/logics/function.py` | 14 | 10 | 1 | 1 | 2 | DSL 值域：object + cast |
| `coding/io.py` | 13 | 3 | 2 | 8 | 0 | JSONValue / TypedDict |
| `core/steps/logics/loop_hooks.py` | 12 | 8 | 0 | 3 | 1 | object + HookHandler |
| `orchestrate/task.py` | 12 | 4 | 3 | 5 | 0 | TaskData TypedDict |
| `session.py` | 12 | 6 | 4 | 0 | 2 | SessionPayload / Message |
| `coding/formats.py` | 11 | 7 | 3 | 0 | 1 | JSONValue / object |
| `executor.py` | 11 | 10 | 1 | 0 | 0 | Message / ToolLike |
| `knowledge.py` | 11 | 6 | 3 | 1 | 1 | JSONValue / TypedDict |
| `orchestrate/agents.py` | 11 | 8 | 3 | 0 | 0 | TaskData / AgentLike |
| `core/model_action/caution_related/caution.py` | 10 | 8 | 2 | 0 | 0 | 约束值：object / Callable[..., str] |
| `core/steps/actions/coding.py` | 10 | 1 | 2 | 7 | 0 | StepLike / object |
| `core/steps/logics/expr_sys/vartypes.py` | 10 | 8 | 2 | 0 | 0 | DSL 值域：object + cast |
| `memory/adapters/store.py` | 10 | 5 | 3 | 0 | 2 | MemoryLike / Sequence[float] |
| `core/steps/actions/interactive.py` | 9 | 5 | 3 | 1 | 0 | StepLike / Callable[..., object] |
| `memory/vector.py` | 9 | 6 | 2 | 1 | 0 | MemoryLike / Sequence[float] |
| `orchestrate/planner.py` | 9 | 9 | 0 | 0 | 0 | TaskData / PlanData |
| `prompt.py` | 9 | 6 | 2 | 0 | 1 | Message / JSONValue |
| `skill/registry.py` | 9 | 5 | 4 | 0 | 0 | Callable[..., object] / SkillLike |
| `core/model_action/caution_related/formatter.py` | 8 | 1 | 6 | 0 | 1 | object（格式化任意值） |
| `core/steps/logics/loop.py` | 8 | 8 | 0 | 0 | 0 | StepLike / object |
| `memory/kv.py` | 8 | 4 | 3 | 0 | 1 | JSONValue / MemoryLike |
| `memory/session.py` | 8 | 5 | 3 | 0 | 0 | JSONValue / MemoryLike |
| `client_sys/openai_client.py` | 7 | 4 | 1 | 0 | 2 | ChatRequest / ChatResponse |
| `core/steps/actions/skill.py` | 7 | 6 | 0 | 1 | 0 | StepLike / Callable[..., object] |
| `memory/budget.py` | 7 | 4 | 2 | 0 | 1 | int / MockTokenizer Protocol |
| `memory/layered.py` | 7 | 5 | 2 | 0 | 0 | MemoryLike / JSONValue |
| `orchestrate/plan_tool.py` | 7 | 5 | 1 | 0 | 1 | TaskData / PlanData |
| `client_sys/client.py` | 6 | 4 | 1 | 1 | 0 | ChatRequest / ChatResponse |
| `client_sys/types.py` | 6 | 1 | 3 | 1 | 1 | ChatRequest / ChatResponse |
| `core/steps/actions/llm.py` | 6 | 0 | 0 | 4 | 1 | AgentLike / ToolLike |
| `orchestrate/delegate.py` | 6 | 3 | 1 | 1 | 1 | TaskData / AgentLike |
| `coding/exec.py` | 5 | 1 | 1 | 2 | 0 | JSONValue / object |
| `core/context.py` | 5 | 2 | 3 | 0 | 0 | Dict[str, object] + 具名 getter |
| `core/display_sys/display.py` | 5 | 4 | 0 | 0 | 1 | object / EventHandler |
| `core/settings.py` | 5 | 3 | 2 | 0 | 0 | SettingsValue = Union[int, float, str, bool, None] |
| `core/steps/logics/conditions.py` | 5 | 4 | 1 | 0 | 0 | ConditionLike |
| `core/steps/logics/while_step.py` | 5 | 5 | 0 | 0 | 0 | ConditionLike / StepLike |
| `core/tool/skill.py` | 5 | 5 | 0 | 0 | 0 | ToolSchema / JSONValue |
| `memory/adapters/rerank.py` | 5 | 4 | 0 | 0 | 1 | Sequence[RecallHit] |
| `cautions/builtin.py` | 4 | 3 | 1 | 0 | 0 | Callable[..., str] |
| `client_sys/registry.py` | 4 | 2 | 1 | 0 | 1 | ClientFactory Protocol |
| `coding/sandbox.py` | 4 | 0 | 1 | 0 | 3 | object / JSONValue |
| `core/event_sys/events.py` | 4 | 3 | 1 | 0 | 0 | HookHandler / JSONValue |
| `core/steps/actions/recall.py` | 4 | 2 | 0 | 2 | 0 | MemoryLike / Sequence[RecallHit] |
| `core/steps/logics/switch.py` | 4 | 4 | 0 | 0 | 0 | object + cast |
| `core/display_sys/runlog.py` | 3 | 1 | 0 | 0 | 2 | object / JSONValue |
| `core/model_action/defaults.py` | 3 | 3 | 0 | 0 | 0 | CautionLike / Sequence[CautionLike] |
| `coding/git.py` | 2 | 0 | 0 | 2 | 0 | object |
| `coding/loop.py` | 2 | 1 | 0 | 1 | 0 | object |
| `core/steps/logics/if_step.py` | 2 | 2 | 0 | 0 | 0 | ConditionLike / StepLike |
| `memory/adapters/embedding.py` | 2 | 2 | 0 | 0 | 0 | Sequence[float] / Sequence[Sequence[float]] |
| `memory/folder_store.py` | 2 | 2 | 0 | 0 | 0 | Sequence[float] |
| `orchestrate/errors.py` | 2 | 1 | 1 | 0 | 0 | object（异常上下文） |
| `cautions/presets.py` | 1 | 1 | 0 | 0 | 0 | Callable[..., str] |
| `core/display_sys/tree.py` | 1 | 1 | 0 | 0 | 0 | object |
| `core/steps/interface.py` | 1 | 1 | 0 | 0 | 0 | StepLike 定义处 |
| `core/steps/logics/goto.py` | 1 | 1 | 0 | 0 | 0 | str / object |
| `core/tool/builtin/write_file.py` | 1 | 0 | 0 | 0 | 1 | object |
| `memory/config.py` | 1 | 0 | 0 | 1 | 0 | int / str |

## 6. 分阶段实施

| 阶段 | 文件 | `Any` 数 | 内容 | 验收门禁 |
| --- | --- | ---: | --- | --- |
| P0 | 20（现有错误） | — | 修 53 个 mypy 基线错误：缺 import、缺 return、`Session \| None`、adapter 参数 | `mypy` 0 error；`pytest` 全绿 |
| P1 | 8 | 125 | 新增 `types.py`；门面层：`agent` / `session` / `prompt` / `executor` / `knowledge` / `chat` / `workflow` / `interface` | 这 8 个文件 `disallow_any_explicit = true` |
| P2 | 12 | 159 | DSL 值域：`expr` / `semantic` / `var` / `function` / `conditions` / `loop*` / `vartypes` | `pytest` 全绿；新增 `object` + `cast` 静态回归测试 |
| P3 | 26 | 302 | `core/tool/*` + `memory/*` + 动作 Step（`chat_util` / `memory` / `orchestrate` / `coding` / `interactive` / `skill` / `recall` / `llm`）：`JSONValue`、`ToolSchema`、`MemoryLike` | mypy 无新增错误；记忆相关测试全绿 |
| P4 | 27 | 358 | `skill/*` + `orchestrate/*` + `client_sys/*` + `coding/*` | 各层 TypedDict 落地；mypy 逐模块收紧 |
| P5 | 14 | 103 | `display` / `api` / `cautions` / `context` / `settings` 余量 | `disallow_any_explicit = true` 全项目；`check_untyped_defs = true` |

合计：125 + 159 + 302 + 358 + 103 = **1047**。

每阶段只改注解，不改逻辑；门禁固定为：

```powershell
python -m mypy
python -m pytest tests -q
```

## 7. 行为等价性与风险

- **注解在运行时基本无副作用**：源码普遍带 `from __future__ import annotations`，
  注解不会在 import 时求值。新增的 `TypeAlias` / `Protocol` / `TypedDict` 只是
  `typing` 对象，不改变任何执行路径。
- **唯二会读注解的地方**：`core/tool/base.py` 与 `skill/schema.py` 都只对
  **用户传入的函数**做 `__annotations__` / `get_origin` 检查，不读库内部函数；
  因此本重构不影响自动 schema 生成。
- **pydantic 模型例外**：`coding/`、`semantic.py` 里的 `BaseModel` 字段参与运行时
  校验，必须保留真实类型，不能替换为 `object`。
- **`object` 会让调用方需要 `cast`**：这是预期收益（把隐式信任变成显式断言）。
  若某处确实无法收窄，用 `cast` 并加一行注释说明来源，而不是退回 `Any`。
- **Python 3.10 兼容**：用 `TypeAlias` 而不是 `type` 语句；可选键用
  `TypedDict` 的 `total=False` 继承，而不是 `NotRequired`（3.11+）。

## 8. 下一步

建议按 P0 → P1 顺序开工：先把 53 个基线错误清零（其中 6 类是真实隐患），
再落地 `types.py` 与门面层的类型替换。P2 之后的每一阶段都可以独立提交、独立回滚。
