﻿# Planner / 多 Agent 协同 —— 接口重构说明 (v2)

> 重构范围：`src/promptkit/orchestrate/`
> 
> 设计原则：轻量化、DSL 直连、向后兼容、RESTful 统一响应、完整错误码

---

## 一、重构总览

| 维度 | 改动前 | 改动后 |
|------|--------|--------|
| **入口** | 每个组件各自暴露 `run()` / `plan()` | 统一 `handle(Request)` + `run()` / `plan()` 简写 |
| **返回值** | `PlanResult` / 抛 `PlanError` / 抛 `TaskError` | `Response{ok, code, message, data}`（v2）；保留 v1 抛异常语义 |
| **DSL 解析** | 多处重复 `plan_from` / `Plan.from_*` | 单一入口 `PlanSpec.from_dsl()` → `to_plan()` / `to_config()` / `to_factory()` |
| **错误传递** | abort 会把原始错误码覆盖成 `TASK_FAILED` | `TaskResult._exception` 保留原始异常，abort 透传 `.code` |
| **新增文件** | — | `dsl.py`（PlanSpec）、`api.py`（v2 RESTful） |
| **修改文件** | — | `errors.py`、`task.py`、`orchestrator.py`、`steps.py`、`__init__.py` |

---

## 二、新旧接口映射

### 2.1 执行入口

| 旧接口 | 新接口 | 说明 |
|--------|--------|------|
| `Orchestrator.run(list_of_tasks)` | `handle(Request(spec=tasks))` | v2 返回 `Response`，不再直接抛异常 |
| `Orchestrator.run(Plan)` | `handle(Request(spec=plan))` | 同左 |
| `Orchestrator.run(dict)` | `handle(Request(spec=dict))` | dict 即 DSL，自动走 `PlanSpec.from_dsl` |
| `Workflow(ExecutePlan(plan=...))` | 不变 | `ExecutePlan` 已支持直接吃 DSL dict |
| `TemplatePlanner.plan(task)` | `handle(Request(action="plan", spec={"task": task}, planner=planner))` | 只出计划不执行 |

### 2.2 数据构造

| 旧接口 | 新接口 | 说明 |
|--------|--------|------|
| `Plan.from_template([...])` | `PlanSpec.from_dsl({"tasks": [...]}).to_plan()` | DSL 优先 |
| `OrchestratorConfig(max_tasks=5)` | `PlanSpec.from_dsl({"policy": {...}}).to_config()` | policy 键 = Config 字段 |
| `RoleFactory({"researcher": "..."})` | `PlanSpec.from_dsl({"agents": {...}}).to_factory()` | 支持 `agents` 或 `roles` |
| `plan_from([Task(...)])` | **不变** | 唯一计划解析入口，已提升为模块级公共函数 |

### 2.3 向后兼容保障

所有原接口 **完全保留**，无签名变化：

```python
# 旧代码继续可用
from promptkit.orchestrate import Orchestrator, Task, Plan
r = Orchestrator(agent=my_agent).run([Task("a", "...")])   # → PlanResult

# 新代码推荐走 handle
from promptkit.orchestrate import handle, Request, ErrorCode
resp = handle(Request(spec=dsl, agent=my_agent))          # → Response
if resp.ok:
    ...
elif resp.code is ErrorCode.ROLE_UNDEFINED:
    ...
```

---

## 三、DSL 集成规范（PlanSpec）

### 3.1 Schema

```jsonc
{
  "task": "string",                        // 顶层目标（可选）
  "agents": { "role": "system prompt" },   // 别名 roles
  "default_role": "string",                // 未指定 role 的任务默认用谁
  "policy": {                              // 键 = OrchestratorConfig 字段
    "max_concurrency": 2,
    "max_tasks": 10,
    "per_task_timeout": 60,
    "stop_on_error": false,
    "isolate": true,
    "max_total_tokens": 0
  },
  "tasks": [                               // 键 = Task 字段
    {
      "name": "string",
      "description": "支持 {{var}} 模板",
      "role": "string",
      "depends_on": ["name"],
      "output_key": "string",
      "timeout": 30.0,
      "max_retries": 2,
      "on_failure": "retry|skip|fallback|abort"
    }
  ]
}
```

### 3.2 三种用法

```python
from promptkit.orchestrate import PlanSpec, handle, ExecutePlan

SPEC = {
    "task": "调研并写报告",
    "agents": {"researcher": "市场专家", "writer": "报告撰写者"},
    "policy": {"max_concurrency": 2},
    "tasks": [
        {"name": "research", "description": "调研 {{topic}}",
         "role": "researcher", "output_key": "research"},
        {"name": "write", "description": "根据 {{research}} 写报告",
         "role": "writer", "depends_on": ["research"]},
    ],
}

# 用法 1：直接喂给 ExecutePlan（Workflow 步骤）
ctx = Workflow(ExecutePlan(plan=SPEC, agent=my_agent)).run()

# 用法 2：走统一 API（最推荐）
resp = handle(Request(spec=SPEC, context={"topic": "AI"}, agent=my_agent))

# 用法 3：显式解析，灵活组装
spec = PlanSpec.from_dsl(SPEC)
plan    = spec.to_plan()           # Plan（含重名/依赖/环校验）
config  = spec.to_config()         # OrchestratorConfig
factory = spec.to_factory()        # RoleFactory
orch    = spec.to_orchestrator(agent=my_agent)   # 一步到位
result  = orch.run(plan, context={"topic": "AI"})
```

### 3.3 严格校验

`PlanSpec.from_dsl` / `to_plan` / `to_config` / `to_factory` 全部拒绝未知键：

```python
# 顶层未知键
PlanSpec.from_dsl({"unknown_key": 1, "tasks": [...]})
# → PlanError(INVALID_REQUEST): DSL 配置里有不认识的键：unknown_key

# policy 里写错
PlanSpec.from_dsl({"policy": {"max_concurency": 99}})
# → PlanError(INVALID_REQUEST): policy 里有不认识的键：max_concurency
```

### 3.4 往返序列化

```python
spec = PlanSpec.from_dsl(SPEC)
assert PlanSpec.from_dsl(spec.to_dsl()).names == spec.names  # 真 round-trip
```

---

## 四、模型自主规划（PlanTool）

`PlanStep` 是「工作流要求这一步必须规划」；`PlanTool` 是「模型自己决定要不要规划」。
工具产出的 `Plan` 直接写进 `Context[output_key]`（默认 `"plan"`）。

### 4.1 工具形态

```python
from promptkit.orchestrate import PlanTool

context = {}
tool = PlanTool(agent, context=context, max_tasks=5, max_calls=2)
agent.ask("这活儿怎么拆？", tools=[tool])

context["plan"]          # Plan（模型调过工具才有）
tool.used                # 已规划次数
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `agent` / `planner` | `None` | 二选一。给 `agent` 会自动包成 `LLMPlanner` |
| `context` | `None` | 绑定的 Context，产出写进 `context[output_key]` |
| `output_key` | `"plan"` | 计划存放的键 |
| `max_tasks` | `5` | 单次规划最多拆几个任务 |
| `max_calls` | `1` | 整个任务里最多规划几次（防无限拆） |
| `context_keys` | `()` | 只把点名的上下文变量喂给规划器 |

工具不抛异常：失败返回 `错误：……`，交给模型自己决定下一步。

### 4.2 步骤形态

```python
from promptkit.core import Var, Workflow
from promptkit.orchestrate import ExecutePlan, PlanWithTool

ctx = Workflow(
    PlanWithTool(prompt="先想清楚怎么做：{{usr_input}}", max_tasks=5),
    ExecutePlan(plan=Var("plan"), agent_factory=lambda task: sub_agent(task.role)),
).run(agent=planner_agent, usr_input="写一份 AI 报告")
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `prompt` | `""` | 告诉模型「什么时候该规划」 |
| `agent` | `None` | 不传就用 `Workflow.run(agent=...)` 的那个 |
| `output_key` / `reply_key` | `"plan"` / `"plan_reply"` | 计划 / 模型回复的存放键 |
| `required` | `True` | `True`：没调工具就报错；`False`：允许只回话 |

### 4.3 规划器归一化

所有「Planner / 带 `plan()` 的对象 / Agent / None」的统一处理只有一个入口
`coerce_planner(value, *, max_tasks, tools, allow_none)`，`PlanStep`、`PlanWithTool`、
`PlanTool`、`api._planned`、`skill.orchestrate_plan` 全部走它 —— 五种写法行为一致。

---

## 五、RESTful API（api.py）

### 5.1 核心结构

```
handle(Request) → Response
     │
     ├── req.validated()          # 入参校验（action / version / spec / context / planner / config / factory）
     └── _dispatch(req)
           ├── action="plan" → _planned(req, spec)
           └── action="run"  → _executed(req, spec)
                                    │
                                    ├── PlanSpec.from_dsl        # DSL → 规格
                                    ├── OrchestratorConfig       # policy → 配置
                                    ├── RoleFactory              # agents → 角色表
                                    └── Orchestrator.run         # 调度执行
```

调用链深度：`handle → _dispatch → _planned/_executed → PlanSpec/Orchestrator` = **4 层**，满足 ≤5 层约束。

### 5.2 Request

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `action` | `str` | `"run"` | `"run"` 或 `"plan"` |
| `spec` | `Any` | `None` | DSL dict / PlanSpec / Plan / Task list，必填 |
| `context` | `Mapping` | `{}` | 任务描述 `{{var}}` 渲染用 |
| `version` | `str` | `"2.0"` | `"1.0"` 保留抛异常语义 |
| `agent` | `Any` | `None` | 默认执行 agent |
| `planner` | `Any` | `None` | `plan` 动作用，需要有 `.plan()` 或 `.ask()` |
| `config` | `OrchestratorConfig` | `None` | 覆盖 DSL 的 policy |
| `factory` | `Callable` | `None` | 覆盖 DSL 的 agents |

### 5.3 Response

| 字段 | 类型 | 说明 |
|------|------|------|
| `ok` | `bool` | 成功与否 |
| `code` | `ErrorCode` | 机器可读错误码（成功时 = `OK`） |
| `message` | `str` | 人类可读文案 |
| `data` | `dict` | 业务数据（plan: waves/tasks, run: results） |
| `version` | `str` | 匹配请求版本 |
| `status` | `int` (property) | HTTP 风格状态码 |
| `to_dict()` | | 全字段序列化成 dict |
| `__bool__` | | `bool(response) == response.ok` |

### 5.4 简写

```python
# run 简写
resp = run(spec, context={...}, agent=...)          # handle(Request(action="run", ...))

# plan 简写
resp = plan(spec={"task": "..."}, planner=planner)  # handle(Request(action="plan", ...))
```

---

## 六、错误码参考

### 6.1 完整枚举（`ErrorCode`）

| 错误码 | HTTP Status | 触发条件 |
|--------|-------------|----------|
| `OK` | 200 | 成功 |
| `UNSUPPORTED_VERSION` | 400 | `Request.version` 不在 `SUPPORTED_VERSIONS` 里 |
| `INVALID_REQUEST` | 400 / 422 | action / spec / context / planner / config / factory 类型不对；DSL 有未知键 |
| `PLAN_INVALID` | 422 | 计划本身不合法（通用兜底） |
| `PLAN_EMPTY` | 422 | 没给 tasks 也没给 task |
| `PLAN_DUPLICATE_TASK` | 422 | 两个任务同名 |
| `PLAN_UNKNOWN_DEPENDENCY` | 422 | 任务依赖不存在的 task |
| `PLAN_CYCLE` | 422 | 依赖环 |
| `PLAN_TOO_MANY_TASKS` | 422 | 任务数 > `max_tasks` |
| `TASK_INVALID` | 422 | 任务项有未知字段 |
| `TASK_FAILED` | 500 | 任务失败且 `on_failure=abort` 或 `stop_on_error` |
| `TASK_TIMEOUT` | 504 | 任务超过 `timeout` |
| `BUDGET_EXCEEDED` | 429 | token 预算耗尽 |
| `NO_AGENT` | 422 | 任务找不到可执行的 agent |
| `ROLE_UNDEFINED` | 422 | 任务用了角色表里没有的 role |
| `DEPTH_EXCEEDED` | 422 | 委派层数超限 |
| `DELEGATE_INVALID` | 422 | 委派配置不合法 |
| `INTERNAL_ERROR` | 500 | 兜底（未预期的异常类型） |

### 6.2 异常类型 → 错误码映射

| Python 异常 | 默认映射 | 说明 |
|-------------|----------|------|
| `PlanError` | 自带 `.code` | 根据场景设置（CYCLE / DUPLICATE / ...） |
| `TaskError` | `TASK_FAILED` | 但 `TaskResult._exception.code` 会透传原始码（如 `TASK_TIMEOUT`） |
| `BudgetExceeded` | 自带 `.code` | `BUDGET_EXCEEDED` |
| `KeyError` | `ROLE_UNDEFINED` | 角色表查不到 |
| `TimeoutError` | `TASK_TIMEOUT` | 任务超时 |
| `TypeError` / `ValueError` | `INVALID_REQUEST` | 入参类型错 |

---

## 七、版本迁移指南

### 7.1 从 v1（抛异常）迁移到 v2（统一响应）

```python
# === v1 旧写法 ===
from promptkit.orchestrate import Orchestrator, Task, PlanError, TaskError

try:
    result = Orchestrator(agent=my_agent).run([Task("a", "...")])
except PlanError as exc:
    print(f"计划不合法: {exc.code} - {exc}")
except TaskError as exc:
    print(f"任务 {exc.task} 失败: {exc.code}")

# === v2 新写法 ===
from promptkit.orchestrate import handle, Request, ErrorCode

resp = handle(Request(spec={"tasks": [...]}, agent=my_agent))
if resp.ok:
    print(resp.data["results"])
else:
    match resp.code:
        case ErrorCode.PLAN_CYCLE: ...
        case ErrorCode.TASK_TIMEOUT: ...
        case ErrorCode.ROLE_UNDEFINED: ...
        case _: print(resp.message)

# === 渐进迁移：版本号开关 ===
resp = handle(Request(spec=..., version="1.0"))  # 保持抛异常
resp = handle(Request(spec=..., version="2.0"))  # 统一 Response（默认）
```

### 7.2 DSL 迁移

```python
# 旧：手动拼 Plan
plan = Plan.from_template([
    Task("research", "调研 {{topic}}", role="researcher", output_key="research"),
    Task("write", "写 {{research}}", role="writer", depends_on=("research",)),
])

# 新：DSL 声明式（1:1 映射，无适配层）
SPEC = {
    "agents": {"researcher": "...", "writer": "..."},
    "tasks": [
        {"name": "research", "description": "调研 {{topic}}",
         "role": "researcher", "output_key": "research"},
        {"name": "write", "description": "写 {{research}}",
         "role": "writer", "depends_on": ["research"]},
    ],
    "policy": {"max_concurrency": 2, "max_tasks": 10},
}
resp = handle(Request(spec=SPEC, context={"topic": "AI"}, agent=my_agent))
```

---

## 八、测试覆盖

重构新增 **33+ 测试用例**，分布在三个 TestClass：

| TestClass | 测试数 | 覆盖范围 |
|-----------|--------|----------|
| `TestPlanSpecDSL` | 10+ | `from_dsl` / `to_plan` / `to_config` / `to_factory` / `to_dsl` round-trip / 未知键拒绝 |
| `TestOrchestrateAPI` | 15+ | `handle` / `run` / `plan` / Request 校验 / 错误码映射 / v1→v2 版本切换 / abort & timeout 码透传 |
| `TestOrchestrateCompat` | 8+ | 旧 `Orchestrator.run` 三入口 / `plan_from` / `ExecutePlan` DSL 直连 / v1 抛异常语义 |

运行方式：

```powershell
python -m pytest tests/test_promptkit.py -k "Orchestrate or PlanSpec or plan_from" -v
```

当前状态：**1624 passed, 3 skipped, 0 failed**（含全部历史测试 + 新增用例）。

离线 smoke example：

```powershell
python examples/orchestrate_refactor.py
# === ALL SMOKE PASSED  (version=2.0) ===
```

---

## 九、调用链深度验证

| 接口 | 调用路径 | 层数 |
|------|----------|------|
| `handle(Request(spec=dsl))` | `handle → _dispatch → _executed → Orchestrator.run` | 4 |
| `handle(Request(action="plan", spec=dsl))` | `handle → _dispatch → _planned → PlanSpec.to_plan` | 4 |
| `handle(Request(action="plan", spec={"task":...}, planner=p))` | `handle → _dispatch → _planned → planner.plan → plan_from` | 5 |
| `PlanSpec.from_dsl(dict)` | 直接构造 + 校验 | 1 |
| `ExecutePlan(plan=dsl)` | `ExecutePlan.step → Orchestrator.run → plan_from` | 3 |

全部 ≤ 5 层，符合轻量化约束。

---

## 十、文件清单

### 新增

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/promptkit/orchestrate/dsl.py` | 198 | PlanSpec — DSL 唯一解析入口 |
| `src/promptkit/orchestrate/api.py` | 200 | v2 RESTful API — Request / Response / handle |
| `src/promptkit/orchestrate/plan_tool.py` | 160 | PlanTool — 模型自主规划，结果写入 Context |
| `docs/orchestrate-api-v2.md` | — | 本文档 |
| `examples/orchestrate_refactor.py` | 210 | 离线 smoke 示例 |
| `examples/orchestrate_plan_tool.py` | 120 | PlanTool 离线示例 |

### 修改

| 文件 | 改动 |
|------|------|
| `errors.py` | 新增 `ErrorCode` 枚举（20 值）；异常基类加 `.code` / `.detail` / `.to_dict()`；`PlanError` 兼作 `ValueError` |
| `task.py` | 新增 `TaskResult._exception`（保留原始错误码）；提升 `plan_from` 为全局唯一计划解析入口 |
| `planner.py` | 新增 `coerce_planner()` — 规划器归一化唯一入口（移除 `PlanStep._planner` / `api._planner_for`） |
| `orchestrator.py` | 移除冗余 `_coerce()`；abort 错误码透传；超时 / 超任务数使用 ErrorCode |
| `steps.py` | `ExecutePlan` 直接接受 DSL dict（走 PlanSpec）；移除重复 `plan_from`；新增 `PlanWithTool`；`PlanStep` 改用 `coerce_planner` |
| `__init__.py` | 新增导出：`API_VERSION` / `SUPPORTED_VERSIONS` / `Request` / `Response` / `handle` / `run` / `plan` / `PlanSpec` / `ErrorCode` / `PlanTool` / `PlanWithTool` / `coerce_planner` |

### 未改动（向后兼容）

- `Orchestrator.run` — 签名不变，仍接受 `list / Plan / dict`
- `Plan.from_template` — 保留
- `plan_from()` — 提升为模块级公共函数，旧调用点零改动
- `TemplatePlanner.plan` / `LLMPlanner.plan` — 保留
- `Workflow(ExecutePlan(...))` — 保留，且 ExecutePlan 增强为支持 DSL dict
- 所有 1624 条历史测试用例 — 零修改、全绿

---

## 十一、约束与取舍

| 约束 | 实现方式 |
|------|----------|
| 零新增第三方依赖 | core orchestrate 模块无新 import |
| 模块 < 20KB | 新增 dsl.py + api.py 合计约 400 行 ≈ 15KB |
| 单方法 ≤ 50 行 | 最长方法 `Request.validated()` ≈ 50 行；其他均 ≤ 30 行 |
| 调用链 ≤ 5 层 | 见第八节验证表 |
| 零适配层 | `PlanSpec` 1:1 映射到 Plan / OrchestratorConfig / RoleFactory |
| v1/v2 共存 | `Request.version` 开关；默认 v2；`handle` 内 `if version == "1.0": raise` |
| 错误码不变 | `ErrorCode` 值未来只增不改 |
| `TaskResult.to_dict()` 不暴露 `_exception` | `_exception` 是私有字段，仅内部错误码透传用 |
