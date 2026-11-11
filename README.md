# PromptKit

一个极简的 LLM 提示词 / 工作流 Python DSL：把逻辑写成步骤树，让 Agent 自己调工具。

PromptKit 把提示词、结构化输出、控制流、约束、工具调用、多模态、记忆这些
「写 LLM 应用绕不开的东西」做成一套小而可组合的库：提示词是带 `{{变量}}`
的对象，工作流是一棵可以 `print` 出来的步骤树，模型输出由 Pydantic 校验并
自动重试，约束（Caution）和解析方式（Formatter）都是可复用的对象。

- 包名 `promptkit`，版本 0.1.0，要求 Python ≥ 3.10
- 核心依赖只有 `openai>=1.40` 和 `pydantic>=2.0`
- 项目采用 src 布局，包在 `src/promptkit/`

## 功能特性

**工作流 DSL**

- `Workflow` / `LLMStep` / `FunctionStep` 组合出顺序执行的步骤树，`Workflow`
  本身也是 `Step`，可任意嵌套复用
- 完整控制流：`If / ElseIf / Else`、`Switch / Case`、`While / DoWhile / For /
  ForEach`、`Break / Continue`、`Label / Goto`、`When`（三元）
- `Function('$a', '$b').Do(...)` 定义带参数、可复用的步骤，支持 `Return` 和
  `ThisFunc` 递归；惰性表达式（`Var("n") + 1` 运行时才求值）
- 语义条件：`Classify` / `Intent` / `Similar` / `Sem(...)` 运算符写法，
  闭集判定，由 `LLMJudge` 显式注入
- 声明式聊天机器人 `ChatWorkflow`（角色 / 目标 / 规则 / 交付物编译成普通
  `Workflow`）、会话持久化 `Session`

**LLM 输出控制**

- `Prompt` 对象 + `{{var}}` 变量模板（支持 `{{a.b.0}}` 点路径，缺变量报错）
- `Executor` 结构化输出：Pydantic 模型校验，解析失败带错误信息自动重试
- `Caution` 约束体系：原子约束 + 预置组合，全局默认 → Workflow → Step 三层合并
- `Formatter` 可换解析方式：JSON / XML / YAML / 正则

**工具与 MCP**

- 普通 Python 函数直接放进 `tools=[...]`，函数签名自动生成 JSON Schema
- 内置工具：联网搜索 `WEB_SEARCH`（多后端可插拔）、沙箱写文件 `WriteFile`、
  跨平台跑命令 `RUN_COMMAND`、查时间 `GET_TIME`、到点提醒 `TIMER`
- MCP 客户端：`load_mcp_tools` 发现工具、`register_mcp_security` 白名单
  安全门、`MCPToolStep` 在工作流里显式调用（仅支持 Streamable HTTP 传输）
- 编码任务原语（`promptkit.coding`）：`CodingStep` / `ApplyDiff` / `RunTests` /
  `FixLoop` / `Sandbox`，构成「改代码 → 跑测试 → 失败再改」闭环
- 多代理编排（`promptkit.orchestrate`）：`Plan` / `ExecutePlan` /
  `AggregateStep`，任务拆分、上下文隔离、失败降级、token 预算
- Skill（`promptkit.skill`）：把「提示词 + 流程 + 工具 + 输出契约」封装成
  可注册、可描述、可执行的能力单元

**多模态**

- `ImageGenStep` / `VideoGenStep`（文生图 / 文生视频 / 以图生图 / 以图生视频）、
  `STTStep` / `TTSStep`
- `Prompt(TextPart(...), ImagePart(...))` 多部件内容，视觉模型直接看图
- `Save` 支持媒体部件直接落盘，文件路径支持 `{{i}}` 模板

**记忆与知识库**

- 三层记忆：工作记忆（`Context`）/ 短期记忆（会话）/ 长期记忆
  （`KVMemory` SQLite、`VectorMemory` 语义召回、`LayeredMemory` 组合）
- 向量记忆默认 `HashEmbedding` 词法级向量，零依赖开箱即用，可换
  OpenAI / 本地模型 / Chroma / Qdrant 适配器
- 知识库（`promptkit.knowledge`，BM25 词法检索，零依赖开箱即用）

**可观测性**

- 事件系统：执行与显示解耦，`EventBus` + `EventType`，`Agent` / `Step`
  只发事件、不 `print`
- `verbose` 分级（`Verbosity.SILENT / NORMAL / VERBOSE / DEBUG`）；
  结果走 stdout，过程走 stderr
- 开箱 handler：`console_display` / `json_display` / `log_display`
- 终端渲染：`Echo(render="markdown")`（装 `rich` 后带排版），LaTeX 公式
  零依赖转 Unicode

## 安装

### 环境要求

- Python ≥ 3.10
- 必需依赖：`openai>=1.40`、`pydantic>=2.0`

### 安装方式

```bash
# 开发用（editable，改代码立刻生效）
pip install -e .

# 作为正式包使用
pip install .

# 也可以只装运行时依赖
pip install -r requirements.txt
```

### 可选 extras

不装也能跑，只是对应功能退化或缺席：

| extra | 装上得到 | 不装会怎样 |
| --- | --- | --- |
| `rich` | 终端 Markdown 排版、代码高亮、表格 | 退回纯文本（去掉标记） |
| `math` | sympy 多行数学排版 | 用内置的零依赖 Unicode 渲染（默认） |
| `render` | `rich` + `sympy` | 同上 |
| `yaml` | `YamlFormatter` | 用别的解析器 |
| `mcp` | MCP 客户端（`mcp>=1.0`） | 无法连接 MCP Server |
| `dev` | `pytest`、`mypy` | 无法跑测试 / 类型检查 |

```bash
pip install -e ".[render]"   # 终端排版 + 数学公式
pip install -e ".[mcp]"      # 接 MCP Server
```

⚠️ `math` extra 里 sympy 的 `parse_latex` 还要求 **antlr4 运行时**且版本挑剔
（sympy 1.14 硬性要求 `antlr4-python3-runtime==4.11`），见 [常见问题](#常见问题-faq)。

### 配置密钥（.env）

`config.load_env_file()` 会按「当前工作目录 → 项目根目录 → 包目录」查找
`.env`（项目根的 `.env` 和 `src/promptkit/.env` 都能被找到）。模板见
`src/promptkit/.env.example`：

```ini
# 文本模型密钥（用哪家填哪家；Agent() 默认 deepseek）
DEEPSEEK_API_KEY=sk-xxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxx
# 豆包 / 火山方舟（DOUBAO_API_KEY 优先，回退 ARK_API_KEY）
# DOUBAO_API_KEY=xxxxxxxx

# 通用密钥：所有厂商都能用（优先级低于厂商专属变量）
# PROMPTKIT_API_KEY=sk-xxxx

# 多模态网关（生图 / 生视频 / STT / TTS 的通用端点）
# PROMPTKIT_MULTIMODAL_API_KEY=xxxxxxxx
# PROMPTKIT_MULTIMODAL_BASE_URL=https://your-gateway/v1
```

密钥查找顺序：`api_key=` 参数 → 厂商环境变量 → `PROMPTKIT_API_KEY` → `.env`。
`.env` 已被 `.gitignore` 忽略。

## 快速开始

一个 15 行内可跑的最小例子（`Agent()` 零配置，自动读密钥并注册为默认 agent）：

```python
from promptkit.agent import Agent
from promptkit.core import *
from promptkit.prompt import Prompt

agent = Agent()                       # 默认 DeepSeek；换厂商：Agent(vendor="openai")

wf = Workflow(
    Set("topic", Read("想聊什么主题？")),
    LLMStep(Prompt(role="user", content="用一句话介绍：{{topic}}"),
            output_key="intro"),
    Echo("介绍：{{intro}}"),
)

if __name__ == "__main__":
    wf.run()
```

更多可运行示例：

```bash
python examples/interview.py                    # 模拟面试工作流（交互式）
python examples/refund_workflow.py "我要退款"    # 结构化输出 + 条件分支
python examples/cautions.py                     # 约束与 Formatter
python examples/memory.py                       # 记忆系统（不需要 API Key）
```

**下一步读什么**：

- 提示词和变量怎么写 → [核心概念](#核心概念)
- 分支 / 循环 / 函数 → [工作流 DSL 参考](#工作流-dsl-参考)
- 控制模型输出格式 → [约束 Caution](#约束-caution)
- 生图 / 接 MCP / 记忆 → 对应专题章节
- 直接看能跑的代码 → [示例索引](#示例索引)

## 核心概念

### Prompt 与变量模板

变量用**双花括号** `{{key}}` 表示，支持点路径取对象字段 / 字典键 / 列表下标：

```python
from promptkit.prompt import Prompt

Prompt(role="user", content="你好，{{name}}").render(name="小明")
# {'role': 'user', 'content': '你好，小明'}

Prompt(role="user", content="意图：{{intent.label}}").render(intent=Intent(label="退款"))
# 支持 {{user.name}} / {{hits.0.text}} / {{a.b.c}}
```

- `{{key}}` 是**严格**语法：变量不存在抛 `KeyError` 并列出可用变量；
  角色不是 system/user/assistant 抛 `ValueError`。推荐使用。
- 也兼容 `${key}` / `$key`，但**宽松**：找不到变量就原样保留（因为 `$`
  和 LaTeX 公式、价格天生冲突）。想「写错立刻报错」就用 `{{}}`。
- `Echo` 的字符串只认 `{{key}}`：`Echo("分数：{{score}}")` 等价于
  `Echo("分数：", Var("score"))`。
- 单花括号 `{name}` 不会被替换，模型会原样收到 —— 不报错但答非所问，
  是最常见的坑。

### Agent 与厂商

`Agent` 是模型客户端封装（上下文管理 + 工具循环 + 命令行对话），默认 DeepSeek：

```python
from promptkit.agent import Agent

Agent()                                    # deepseek（默认）
Agent(vendor="openai")                     # OPENAI_API_KEY，默认 gpt-4o-mini
Agent(vendor="anthropic")                  # ANTHROPIC_API_KEY，claude-sonnet-4-5
Agent(vendor="doubao")                     # DOUBAO_API_KEY / ARK_API_KEY
Agent(base_url="http://localhost:11434/v1", model="qwen2.5",
      api_key="ollama")                    # Ollama / vLLM 等兼容端点
```

| 厂商 | `vendor=` | 环境变量 | 默认模型 |
| --- | --- | --- | --- |
| DeepSeek | `deepseek`（默认） | `DEEPSEEK_API_KEY` | `deepseek-flash` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `anthropic`（别称 `claude`） | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| 豆包 | `doubao` | `DOUBAO_API_KEY` / `ARK_API_KEY` | `doubao-seed-2-1-pro-260915` |
| 兼容端点 | `openai-compatible`（必须给 `base_url` 和 `model`） | `OPENAI_API_KEY` | — |

厂商识别顺序：`vendor=` 显式 > `base_url` 主机名 > `model` 前缀 > 环境里
现成的密钥 > DeepSeek。厂商特有参数走 `params=`（如
`Agent(params={"reasoning_effort": "low"})`）。内部统一说 Chat Completions
方言，Anthropic 的协议差异由 `AnthropicClient` 翻译（`urllib` 直连，不需要
装 anthropic SDK）。

### LLMStep 的三件事

`LLMStep` 是「调一次模型」的步骤，只关心三件事：

```python
LLMStep(
    prompt=Prompt(role="user", content="判断意图：{{text}}"),  # ① 提示词
    output_model=Intent,          # ② 结构化输出（可选）
    output_key="intent",          #    结果写进 Context 的哪个键
    cautions=[JSON_ONLY],         # ③ 约束（可选）
)
```

提示词里的 `{{var}}` 从 Context 取值。agent 从哪来：`LLMStep(..., agent=agent)`，
或统一写在 `wf.run(agent=agent)`。

### 结构化输出

`output_model` 给一个 Pydantic 模型，`Executor` 会从模型输出里抽取 JSON
（裸 JSON、```` ```json ```` 代码块、夹在解释文字里的 JSON 都能识别），
解析或校验失败时**带着错误信息自动重试**，`settings.max_retries` 次后抛
`MaxRetriesExceeded`：

```python
from pydantic import BaseModel

class Intent(BaseModel):
    label: str

LLMStep(Prompt(role="user", content="判断意图：{{text}}"),
        Intent, output_key="intent")
```

解析方式可换（`Formatter`）：`JsonFormatter()`（默认）/ `XmlFormatter()` /
`YamlFormatter()`（需 pyyaml）/ `RegexFormatter(r"情绪：(?P<label>\w+)")`，
或字符串 `"xml"` / `"regex:<pattern>"`。解析失败的重试提示会带上格式化器
的名字。`LLMStep` 是泛型的，`LLMStep[Intent](...)` 让 IDE / mypy 能检查。

完整示例见 [examples/refund_workflow.py](examples/refund_workflow.py)、
[examples/cautions.py](examples/cautions.py)。

## 工作流 DSL 参考

`from promptkit.core import *` 导出整套 DSL。**变量必须先声明**
（`NewVar` / `Number` / `Str`），读没声明过的变量抛 `UndefinedVariable`；
声明了类型后每次赋值（`Set`、外部传参、函数返回值）都会自动转换。

### 内置步骤速查表

| 语法 | 说明 |
| --- | --- |
| `NewVar("x", type=None, default=...)` | 声明变量；`Number("n")` / `Str("s")` 声明带类型的（默认 `0` / `""`） |
| `Var("x")` / `Var(name="x", val=expr)` | 读变量 / 写变量；支持点路径 `Var("a.b")` |
| `Set("x", expr)` | 写变量（按声明类型转换） |
| `Set(Var("d")["k"], expr)` | 写容器的下标（dict / list，支持嵌套和表达式键） |
| `Inc("i", by=1)` / `Dec("i", by=1)` | `i++` / `i += n` / `i--` |
| `LLMStep(prompt, output_model=None, output_key="output")` | 调模型；给了 `output_model` 就做结构化校验 |
| `FunctionStep(func, output_key=None)` | 包装任意 Python 函数为步骤 |
| `Function('$a', '$b', name=...).Do(...)` | 定义带参数的可复用步骤；形参用 `$` 标记（声明记号，内部去掉）；参数强校验（多传 / 少传 / 重复抛 `ArgumentError`） |
| `Return(expr)` | 函数返回值，调用方用 `.result` 取（属性，不加括号） |
| `ThisFunc(Var("n") - 1).result` | 函数体内递归调用自己（上限 `settings.max_recursion_depth`，默认 100） |
| `Transform(name, *args, **kwargs)` | 具名数据转换表达式（`join` / `len` / `json_dumps` …；用 `@transform("name")` 注册扩展）；**不是步骤**，要包进 `Set` / `Echo` |
| `If(c).Then(...).ElseIf(c).Then(...).Else(...)` | 多路分支；条件支持常量 / lambda / `Condition` 子类 / `And` / `Or` / `Not` |
| `When(cond, a, b)` | 三元表达式 `cond ? a : b` |
| `Switch(v).Case(a, b).Then(...).Default(...)` | switch-case；默认不穿透，`Case(1, fallthrough=True)` 显式穿透 |
| `While(cond, max_iterations=N).Do(...)` | 条件循环；超限抛 `LoopLimitExceeded` |
| `DoWhile(cond).Do(...)` | 先执行一次再判断 |
| `For(init, cond, step).Do(...)` | C 风格 `for(init; cond; step)`，如 `For(Set("i",0), Var("i")<5, Inc("i"))` |
| `ForEach("x", iterable).Do(...)` | 遍历 range / 列表 / 表达式 |
| `Break()` / `Continue()` | 跳出 / 跳过本轮（作用于最内层循环；`Switch` 不吞 `Break`） |
| `OnError(*steps, handler=..., fallback=..., output_key="error")` | 兜底一组步骤的异常：默认吞掉并写 `output_key`，`reraise=True` 处理完再抛 |
| `Label("x")` / `Goto("x")` | 跳到**顶层**标签（重试循环、跳出深层嵌套）；上限 `max_jumps` |
| `Read(prompt="")` | 读一行标准输入（惰性表达式）；`.Until(标记)` 多行读取；`.OnBreak()` Ctrl+C/D 不炸工作流 |
| `ReadFile(path)` | 读整个文件（惰性，UTF-8） |
| `Confirm("继续？")` | 是非题（y/n/是/否），`.Default(True)` 兜底空输入 |
| `Choose("吃啥？", "A", "B")` | 编号菜单选一个，`.Default(2)` 兜底 |
| `Echo(*parts, render="text"\|"markdown", title=...)` | 打印 / 渲染；支持 `{{var}}` 插值 |
| `Save(*values, file=..., format="text"\|"json", append=...)` | 写文件（UTF-8 + LF，自动建父目录；`file` 支持 `{{i}}` 模板；媒体部件直接落盘） |
| `SaveJSON(value, file=...)` | 写成合法 JSON 文件 |
| `SaveHistory(file=..., include_variables=True)` | 对话历史 + 步骤轨迹 + 变量存成 JSON |
| `MessageStep(role, content)` | 往 `context["messages"]` 追加消息 |
| `Len(expr)` | 长度（`len(expr)` 会立刻求值，不行） |
| `SkipStep()` | 空步骤（写显式空分支） |
| `expr[i]` / `expr[a:b]` / `expr.attr("n")` / `expr.method("m", *a)` | 下标 / 切片 / 取属性 / 调方法（都是惰性） |
| `print(wf)` / `step.render()` | 打印整棵步骤树（纯展示，不执行） |

### 表达式与条件

对 `Var(...)` 做比较 / 运算得到的是**惰性表达式**，运行时才求值：

```python
(Var("n") > 1) & (Var("n") < 5)     # ✅ 组合条件必须加括号（& 优先级高于比较）
If(Var("p").attr("age") >= 18).Then(...)   # 条件里可以下标、取属性
```

`Expr` 不能放进 Python 的 `if` / `and` / `or`（`bool(expr)` 抛
`TypeError`，避免静默错误）。变量里可以放任意 Python 对象（原样存储，
不拷贝），容器读写用 `expr[i]` 读、`Set(Var("d")["k"], v)` 写。

### 函数与递归

```python
Ask = Function('$job', name="question").Do(
    Prompt(role="user", content="面试一个 {{job}} 岗位，问一个问题。"),
)

Ask(job="后端")            # BoundStep，放进 Workflow
Ask(job=Var("job"))        # 参数也可以是表达式，每次执行重新求值
Ask(job="后端").result     # 当表达式用：执行并取回复（提示词函数自动写 output_key）
```

`Function` 的参数只在步骤体内部可见（执行完自动恢复），不污染外层 Context。
`Return` 会穿过任意深度的 `If` / `While` 冒泡到函数边界；提示词函数不需要
显式 `Return`。`.agent_reply()` 是 `.result` 的旧名字（兼容别名，仍可用）。

### 语义条件

`If` / `Switch` 也能接语义判断（闭集判定，模型只能选给定标签），
**必须显式提供 judge**：

```python
from promptkit.core import Classify, LLMJudge, Sem, Set, Similar, Switch, Var, Workflow

wf = Workflow(
    Set("sentiment", Classify(Var("usr_input"), "积极的", "消极的", "其他")),
    Switch(Var("sentiment"))
        .Case("积极的").Then(Echo("积极的"))
        .Default(Echo("判断不出来")),
)
wf.run(agent=agent, judge=LLMJudge(agent))   # judge 注入一次，所有语义判断共享
```

```python
If(Similar(Var("a"), Var("b"), "same", "contains")).Then(...)  # same/contains/entails/contradicts/independent
If(Sem(Var("a")) == Var("b")).Then(...)   # 运算符写法：== / >= / >> / ^
```

可控性设计：闭集标签（选到闭集外报错）、模型只产出 `label + confidence +
reason`（真假由代码算）、确定性规则先短路、置信度阈值 + margin、unknown
默认走 Else、每次 run 共享调用预算（默认 32）并缓存、注入防护（文本放在
`<<< >>>` 里）。语义条件只适合软判断，硬约束（金额、权限）用确定性条件。

### 循环 Hook 与约束

`While(...).Hook(扩展点, fn)` 支持 7 个扩展点（`LOOP_START` /
`BEFORE_CONDITION` / `BEFORE_ITERATION` / `AFTER_ITERATION` / `LOOP_LIMIT` /
`LOOP_ERROR` / `LOOP_END`），钩子签名 `fn(event: LoopHookEvent) ->
Optional[bool]`，返回 `False` 可短路。钩子异常不会炸主循环，记进
`hooks.failures`。完整示例见 [examples/while_hooks.py](examples/while_hooks.py)。

### 聊天机器人与会话

`ChatWorkflow` 把「它是谁、要干什么、有什么规矩、要产出什么」声明出来，
编译成普通 `Workflow`：

```python
from promptkit.core import ChatWorkflow, Deliverable
from promptkit.core.tool import WEB_SEARCH

chat = ChatWorkflow(
    role="市场调研分析大赛的导师",
    want="帮参赛者找到合适的选题方向",
    rules=["像导师一样专业、耐心", "每次回复不超过 300 字"],
    deliverables=[Deliverable("选题建议", format="markdown",
                              save_to="outputs/选题.md")],
    tools=[WEB_SEARCH],
    session="sessions/tutor.json",     # 退出保存、下次接着聊（异常退出也存）
)
chat.run(agent=agent)
```

- `session=` 负责「开跑前 load、结束时 save」；手写 `Workflow` 用
  `wf.run(agent=agent, session="...")`，同一份代码
- 会话还能记住 Context 变量（问卷只问一次）：
  `wf.run(..., session="s.json", variables=["basic_info"])`
- 长对话自动压缩：`ChatWorkflow(compact_after=40, compact_keep=8)`
- 让模型感知时间：`ChatWorkflow(timestamps=True)`（每轮用户消息自带
  `[11-09 15:30]` 前缀；手写 `ChatLoop(timestamps=True)` 同理，模板里可用
  `{{chat_time}}`）。也可以把步骤塞进循环体：`ChatLoop(TimeStamp(), ...)`
- 计时器主动推送：`ChatWorkflow(proactive_alarms=True)` —— 到点立刻让模型
  说一句，不等用户输入（底层是 `AlarmRunner` 后台线程 + `Agent` 回合锁）。
  步骤写法：`ChatLoop(ProactiveAlarms())`
- 声明式 Step（`Role` / `Want` / `Rules` / `Tools` / `Deliverables` /
  `ExitWords` / `SystemPrompt` / `ChatLoop` / `Compact` / `SaveDeliverables`）
  也可手写组合；`Rules` 等**多次调用会累积**，重置传 `replace=True`
- ⚠️ `Deliverables(...)` 只是声明，手写 Workflow 时循环后要自己加
  `SaveDeliverables()`（`ChatWorkflow.build()` 会自动加）
- ⚠️ 同名不同物：`promptkit.core.Role` 是声明式 Step，
  `promptkit.cautions.Role` 是约束文本，`import *` 会互相覆盖

完整示例见 [examples/chat_bot.py](examples/chat_bot.py)、
[examples/course_assistant.py](examples/course_assistant.py)、
[examples/conversation_crud.py](examples/conversation_crud.py)。

## 约束 Caution

`Caution` 把「输出要求」变成对象：Prompt 只写任务，约束单独管理，渲染时
拼到内容末尾。

```python
from promptkit.cautions import Language, NoExtraText, OutputFormat, JSON_STRICT

LLMStep(
    prompt=Prompt(role="user", content="判断用户意图：{{text}}"),
    output_model=Intent, output_key="intent",
    cautions=[OutputFormat("JSON"), NoExtraText(), Language("中文")],
)
```

组合方式：列表、`+` 号、`sum([...])`、预置组再叠加（`GROUNDED + Tone("耐心")`）、
或直接写字符串。文本完全相同的约束自动去重。

### 原子约束

| 类 | 输出示例 |
| --- | --- |
| `OutputFormat(format, extra="")` | `请以 JSON 格式输出。` |
| `NoExtraText()` | `除非另有指令，不要生成任何解释、前后缀或其他内容。` |
| `NoMarkdown()` | `不要使用 Markdown 标记……直接输出纯文本。` |
| `Language(lang)` | `请使用中文回答。` |
| `MaxLength(n)` | `回答不超过 200 字。` |
| `NoFabrication()` | `如果信息不足或不确定，直接说不知道，不要编造或猜测。` |
| `Citations(style="[编号]")` | `引用资料时用 [编号] 标出出处……` |
| `Grounded()` | `只根据上面提供的资料回答……` |
| `Concise()` | `回答尽量简短直接……` |
| `StepByStep()` | `请分步骤说明，每一步都给出理由或依据。` |
| `AskFirst(at_most=3)` | `如果缺少必要信息，先问我（一次最多 3 个问题）……` |
| `Tone(tone)` | `语气要求：礼貌。` |
| `Role(role)` | `你现在扮演：资深客服。` |
| `Style(style)` | 风格约束（生图 / 生视频也用） |
| `NoText()` | 画面中不要出现文字（多模态用） |
| `Quality(...)` | 画质约束（多模态用） |
| `SchemaHint(dict)` / `JsonSchemaHint(Model)` | 输出字段说明（后者从 Pydantic 自动生成） |
| `Custom(text)` | 原样输出 |

### 预置约束组

`cautions=[JSON_STRICT]` 直接用；都是 `CautionGroup`，可继续 `+`：

| 组 | 内容 | 场景 |
| --- | --- | --- |
| `json_only()` / `JSON_ONLY` | JSON + 无废话 + 语言 | 简单结构化输出 |
| `json_strict()` / `JSON_STRICT` | JSON + NoMarkdown + 无废话 + NoFabrication + 语言 | 模型老爱包代码块或编字段 |
| `json_with_schema(schema)` | JSON + 字段说明 + 无废话 + 语言 | 字段多、要求明确 |
| `plain_text()` / `PLAIN_TEXT` | 无废话 + 长度 + 语言 | 普通文本回复 |
| `concise_text()` | NoMarkdown + Concise + 长度 + 语言 | 简短纯文本 |
| `code_only()` / `CODE_ONLY` | 代码 + 语言 + 无废话 | 只要代码 |
| `polite_reply()` | 纯文本 + 礼貌语气 | 客服 |
| `grounded()` / `GROUNDED` | Grounded + Citations + NoFabrication + 语言 | 知识库 / 联网搜索 |
| `tutor(persona)` / `TUTOR` | Role + Tone + StepByStep + NoFabrication | 教学 / 辅导 |
| `socratic()` / `SOCRATIC` | Role + AskFirst + StepByStep + Tone | 追问式引导 |
| `analyst()` / `ANALYST` | Role + StepByStep + Concise + NoFabrication | 数据分析 |
| `translator()` | 译文 + 无废话 + NoMarkdown + 语言 | 翻译 |

### 三层合并，就近优先

优先级从低到高：**全局默认 → Workflow → Step**：

```python
from promptkit.core import set_default_cautions

set_default_cautions([NoExtraText(), Language("中文")])   # ① 全局
wf = Workflow(..., cautions=[Tone("简洁")])                # ② Workflow 级
LLMStep(..., cautions=[MaxLength(50)])                     # ③ Step 级
LLMStep(..., use_default_cautions=False)                   # 单步豁免全局默认
```

同一套约束也作用于 `ImageGenStep` / `VideoGenStep`（`cautions=[Style("水彩画"),
NoText()]`，三层合并语义一致）。Caution 是**事前引导**，`output_model` +
重试是**事后兜底**，两者互补。

⚠️ 全局默认是进程级可变状态，测试里用 `using_default_cautions(...)` /
`clear_default_cautions()` 收尾。

## 多模态

生成与理解走统一的多模态客户端 `BaseMultimodalClient`（内置豆包和通用网关
两种实现），不经过文本 `Agent`：

```python
from promptkit.client_sys.multimodal import make_multimodal_client
from promptkit.core.steps.actions.multimodal import ImageGenStep, VideoGenStep

client = make_multimodal_client("doubao")   # 密钥读 DOUBAO_API_KEY / ARK_API_KEY

wf = Workflow(
    ImageGenStep("image_prompt", "cat_image", client=client, size="1024x1024"),
)
wf.run(image_prompt="一只橘猫趴在机械键盘上睡觉")
```

四个步骤：

| 步骤 | 作用 | 输出 |
| --- | --- | --- |
| `ImageGenStep(prompt_key, output_key, client=..., size=...)` | 文生图 / 以图生图 | `ImagePart` |
| `VideoGenStep(prompt_key, output_key, client=..., duration=...)` | 文生视频 / 以图生视频 | `VideoPart` |
| `STTStep(audio_key, output_key, client=...)` | 语音转文字 | `str` |
| `TTSStep(text_key, output_key, client=..., voice=...)` | 文字转语音 | `AudioPart` |

`prompt_key` 既可以是 Context 变量名，也可以直接给 `Prompt`。**以图生图 /
以图生视频**就是把图片部件放进 Prompt：

```python
from promptkit.core.content import ImagePart, TextPart
from promptkit.prompt import Prompt

ImageGenStep(
    Prompt(role="user", content=[
        TextPart(text="把这只猫变成水彩画"),
        ImagePart(Var("cat_image")),        # 参考图
    ]),
    "result",
    cautions=[Style("水彩画"), NoText()],   # 和 LLMStep 同一套约束
)
```

生成的媒体部件可以直接进下一轮 `LLMStep`（视觉模型看图）：

```python
LLMStep(
    Prompt(role="user", content=[
        TextPart(text="用 50 字给这张图写一句注脚："),
        Var("cat_image"),                   # ImagePart 自动传给 vision model
    ]),
    output_key="caption",
)
```

**落盘**：`Save` 认识媒体部件，路径里 `{{i}}` 按上下文展开：

```python
Save(Var("cat_image"), file="cats/cat_{{i}}.png")
```

完整示例见 [examples/image_generation_workflow.py](examples/image_generation_workflow.py)、
[examples/multimodal_chat.py](examples/multimodal_chat.py)。

## 工具调用与 MCP

### 工具调用（function calling）

把普通 Python 函数放进 `tools=[...]`，函数名、docstring、类型标注自动生成
JSON Schema；`Agent.ask()` 内部自动跑「模型要工具 → 执行 → 结果喂回 →
直到给出最终回答」的循环：

```python
def query_order(order_id: str) -> dict:
    """查询订单的状态和金额。order_id 是订单号。"""
    return {"status": "已发货", "amount": 199.0}

LLMStep(Prompt(role="user", content="帮我看看订单 67890，能退就退了。"),
        tools=[query_order], output_key="answer")
```

行为约定（都有测试）：工具抛异常不中断对话（错误文本回给模型）、模型要不
存在的工具回错误文本、超过 `max_tool_rounds`（默认 8）抛
`ToolLoopExceeded`、工具返回 dict/list 自动 `json.dumps`。自定义名字 /
描述 / schema 用 `Tool(...)` 显式构造或 `@tool` 装饰器。

内置工具：

| 工具 | 说明 |
| --- | --- |
| `WEB_SEARCH` | 联网搜索；配了 Key（博查 / 智谱 / Tavily / Serper / Brave）就用 Key，否则走免密钥降级链 `百度 → Bing 中国 → DuckDuckGo`；顺序可用 `PROMPTKIT_SEARCH_BACKENDS` 调整 |
| `WriteFile` / `WRITE_FILE` | 让**模型自己决定**写文件；沙箱限制根目录、扩展名白名单、单次 20 万字符 |
| `RUN_COMMAND` | 跑 shell 命令（跨平台、UTF-8 容错、带超时） |
| `GET_TIME` | 查当前时间（`YYYY-MM-DD HH:MM:SS 周X`），模型能主动问「现在几点」 |
| `TIMER` | 计时器：`timer("start", seconds=600, name="work")` 到点后往对话里插一条 `<15:30>到了。`，模型下一轮即可感知 |

计时器有两种投递方式：

- **排队（默认）**：到点后立即打印并排队，Agent 在**下一次**调模型前把提醒
  作为 `user` 消息插进历史（用户下次说话时模型会先看到它）。
- **主动推送**：`ChatLoop(proactive_alarms=True)` / `ChatWorkflow(...)`，或
  把 `ProactiveAlarms()` 塞进循环体，或直接 `AlarmRunner(agent).start()` ——
  到点立刻让模型跑一轮，不用等用户输入。
  `AlarmRunner` 是后台守护线程，`Agent` 内有回合锁保证同一时刻只有一轮调用；
  对话结束（正常 / 异常）时会自动停掉。

入参：`action`（`start` / `stop` / `elapsed` / `list`）、`seconds`、`name`、
`message`（自定义提醒文本，默认 `<HH:MM>到了。`）。

不在聊天里也能用 DSL 原语（写进步骤树就一定执行，不依赖模型是否调工具）：

```python
from promptkit.core import (
    Workflow, TimeStamp, StartTimer, WaitAlarm, StopTimer,
)

wf = Workflow(
    TimeStamp(),                       # chat_time = "11-09 15:30"（短格式）
    StartTimer(600, name="contest"),   # 10 分钟后提醒
    WaitAlarm(timeout=700, output_key="alarm"),  # 阻塞等提醒
    StopTimer("contest"),
)
```

| 原语 | 做什么 |
| --- | --- |
| `TimeStamp(output_key="chat_time")` | 写短时间戳，适合塞进 prompt |
| `Now(output_key="now")` | 写完整时间（含星期），支持 `timezone=` |
| `StartTimer(seconds, name=..., message=...)` | 起计时器，到点提醒进 `ALARMS` |
| `StopTimer(name=...)` | 停计时器 |
| `WaitAlarm(timeout=None, output_key="alarm")` | 阻塞等一条提醒；超时写空串、不抛异常 |
| `ProactiveAlarms()` | 放进 `ChatLoop` 循环体，等价于 `proactive_alarms=True` |

⚠️ `WriteFile` 的沙箱（`sandbox=True`，默认开）是唯一「模型驱动写磁盘」的
通道，别关。搜索结果会被塞进提示词而网页内容不可信 —— 别把搜索结果当
指令执行。

完整示例见 [examples/tool_calling.py](examples/tool_calling.py)、
[examples/web_search.py](examples/web_search.py)、
[examples/auto_files.py](examples/auto_files.py)。

### 编码任务（promptkit.coding）

编码任务里模型产出的是**要执行的代码变更**，核心闭环是「读上下文 → 生成
变更（结构 + 干跑校验）→ 应用 → 跑测试 → 失败带错误再改」：

```python
from promptkit.coding import ApplyDiff, CodingStep, FixLoop, ReadFileStep, RunTests

wf = Workflow(
    ReadFileStep("calc.py", output_key="source", base_dir=root),
    RunTests("python -m unittest discover -s . -q", base_dir=root,
             allowlist=["python"]),
    FixLoop(
        [CodingStep(task="修好 add", context={"calc.py": Var("source")},
                    error=Var("test_result"), path="calc.py", base_dir=root),
         ApplyDiff("calc.py", diff=Var("patch"), base_dir=root),
         RunTests("python -m unittest discover -s . -q", base_dir=root,
                  allowlist=["python"])],
        until=Var("test_result")["passed"], max_rounds=3,
    ),
)
```

要点：三种输出格式（`diff` / `full_file` / `json`，都校验「能不能执行」）；
`Sandbox` 守安全边界（只收相对路径、敏感文件拒绝、命令白名单、审计日志，
**不是容器级沙箱**）；`default_coding_tools(root, allow_write=True,
allow_commands=["python"])` 把同一批能力交给 Agent 自主使用。示例见
[examples/coding_agent.py](examples/coding_agent.py)、
[examples/coding_chat.py](examples/coding_chat.py)。

### 多代理编排（promptkit.orchestrate）

```python
from promptkit.orchestrate import (AggregateStep, ExecutePlan,
                                   OrchestratorConfig, Plan, RoleFactory, Task)

plan = Plan.from_template([
    Task("research", "给出 3 条理由", role="researcher", output_key="research"),
    Task("write", "根据理由写文案：\n{{research}}", role="writer",
         depends_on=("research",), output_key="draft"),
])

workflow = Workflow(
    ExecutePlan(plan=plan,
                agent_factory=RoleFactory({"researcher": "你是调研员……",
                                           "writer": "你是文案……"}),
                config=OrchestratorConfig(max_total_tokens=20000)),
    AggregateStep(results_key="task_results", agent=editor,
                  instructions="合成一份说明。"),
)
```

三种模式按序采用：① 固定模板 `Plan.from_template` → ② LLM 规划
`PlanStep(planner=agent)` → ③ 动态委派 `DelegateTool`。护栏：`max_tasks`
（默认 10）/ `max_concurrency`（默认 1，`ExecutePlan(parallel=True)` 是语法
糖）/ `max_total_tokens` / `per_task_timeout` / `max_delegations` /
`max_depth`；失败策略 `retry`（默认）/ `skip` / `abort` / `fallback`。示例见
[examples/orchestrate.py](examples/orchestrate.py) 等同主题脚本。

### Skill（promptkit.skill）

Skill 是完整能力单元（提示词 + 控制流 + 工具 + 输出契约），对外只有一份描述：

```python
from promptkit.skill import SkillContext, skill

@skill(name="classify_sentiment", description="判断文本情感倾向",
       output_model=Sentiment)
def classify_sentiment(text: str, ctx: SkillContext) -> Sentiment:
    return ctx.structured(f"判断这段话的情感：{text}", Sentiment)
```

`@skill` 从函数签名生成参数 Schema（`ctx` 跳过），`output_model` 的 Schema
填进 `returns`；返回值是可直接调用、也可注册的 `Skill`。用法三选：
`SkillStep(skill, text=Var("text"), output_key="s")` 放进工作流、
`registry.to_tools(ctx)` 交给 Agent 自己挑、`ChatWorkflow(skills=[...])`
挂进聊天。没有全局注册表，注册是显式的；插件通过
`entry-points."promptkit.skills"` 加载。外部 `.md` 技能（一段说明 + 参数表）
用 `as_skill("skills/x.md")` / `as_skills("skills/")` 加载后直接当 Tool 用。
示例见
[examples/skills.py](examples/skills.py)、
[examples/skill_refund.py](examples/skill_refund.py)、
[examples/markdown_skill.py](examples/markdown_skill.py)（外部 `.md` 技能）、
[examples/skill_with_script.py](examples/skill_with_script.py)（技能里带脚本）。

目录式技能包（`SKILL.md` 或 `AGENT.md` + `references/` + `assets/` +
`scripts/`）用 `as_skill_package("skills/xxx")` 加载

```python
from promptkit.skill import as_skill_package

pkg = as_skill_package("skills/ppt-master")

# 工具路由：正文按需加载，模型匹配时自己调技能工具
ChatWorkflow(role="PPT 助手", tools=pkg.tools())

# 提示词路由：让「什么时候该用这个技能」始终在 system prompt 里
ChatWorkflow(
    role="PPT 助手",
    rules=[pkg.routing_prompt()],
    tools=pkg.tools(),
)
```

`pkg.routing_prompt()` 只放技能名、描述和触发规则，控制 token；确实需要整份
入口常驻时用 `rules=[pkg.instructions]`（`{{参数}}` 会保持原样，可用
`pkg.skill.render(...)` 替换）。

资源路径限定在技能目录内，二进制只返回元信息；脚本从 `argparse` 推断参数
Schema，只允许执行加载时登记过的 `.py`（不走 shell，带超时和输出截断）。
frontmatter 支持 `dependencies`（缺失会标注，`strict_dependencies=True` 时
直接抛 `SkillDependencyError`）和 `allowed-tools`（兼容 `Read` / `Glob` /
`Bash` 写法）。示例见
[examples/skill_package.py](examples/skill_package.py)。

技能脚本用哪个解释器不需要手工配置，`resolve_python` 按下面的顺序取：
显式 `python=` → 环境变量 `PROMPTKIT_PYTHON` → 技能包根目录的 `.venv` /
`venv` → 当前解释器 → PATH。换一台电脑直接 `pip install .` 就能用；Windows
商店那种退出码 9009 的假 python 会在探测阶段被自动跳过。

### MCP 接入

`promptkit.mcp` 是 MCP 客户端，**只支持 Streamable HTTP 传输**（不支持
stdio / sse），需要 `pip install -e ".[mcp]"`：

```python
import asyncio
from promptkit.mcp import load_mcp_tools, register_mcp_security
from promptkit.mcp.hooks import get_default_hook_registry

# ① 发现工具并注册（async；单个失败不中断整体）
asyncio.run(load_mcp_tools(registry, "http://localhost:3000/mcp", "filesystem"))
#    工具名带服务前缀：fs__read_file、fs__list_dir …

# ② 安全门：白名单之外一律 PermissionError；白名单为空 = 全部拒绝
register_mcp_security(get_default_hook_registry(),
                      {"fs__read_file", "fs__list_dir"})

# ③ 两种用法
agent.ask("读一下 config.yaml", tools=registry.all())     # Agent 自主调用
MCPToolStep(tool=registry.get("fs__read_file"),            # 工作流显式调用
            input_keys={"path": "target_path"},
            output_key="file_content")
```

MCPTool 和内置 Tool 走同一套 `tool_calls` 流程，Agent 循环零改动。完整演示
（含安全 gate 各场景）见 [examples/mcp_demo.py](examples/mcp_demo.py)、
[examples/mcp_agent.py](examples/mcp_agent.py)。

## 记忆与知识库

三层记忆，别混用：

| 层 | 生命周期 | 实现 |
| --- | --- | --- |
| 工作记忆 | 单次 `Workflow.run` | `Context` |
| 短期记忆 | 单次会话 | `SessionMemory`（只读包 `Agent.history`）/ `Session` 文件 |
| 长期记忆 | 跨会话 | `KVMemory`（SQLite）/ `VectorMemory` / `LayeredMemory` |

设计原则：**显式优于魔法**（读写 / 召回都写在步骤里：`ReadMemory` /
`WriteMemory` / `RecallMemory` / `MemoryStep`，给模型的记忆工具只给读）、
接口薄（`Memory` 只有 `read` / `write` / `recall` / `forget`）、适配器隔离
（embedding / store / rerank 都是接口）、可遗忘。

```python
from promptkit.memory import VectorMemory
from promptkit.memory.adapters import HashEmbedding

vec = VectorMemory(HashEmbedding(dimension=256), namespace="learner_001")
vec.write("notes", "……很长的笔记，会按块切分……")
vec.recall("退款怎么处理", top_k=3)
# [{'text': '……', 'score': 0.83, 'id': 'notes#0', 'metadata': {...}}]
```

- `HashEmbedding` 是零依赖词法级向量，开箱即用；换真语义就换
  `OpenAIEmbedding` 等适配器，接口一样
- `KVMemory(namespace="u1")` 落 SQLite（默认在系统数据目录
  `%APPDATA%\promptkit\memory\` 等，可用 `PROMPTKIT_DATA_DIR` 整体改）；
  `VectorMemory` 不传 `path=` 时是进程内索引
- 召回后过 `TokenBudget` 裁剪，别让长笔记吃掉上下文
- 声明式接入：`ChatWorkflow(memory=..., memory_config=MemoryConfig(read=[...],
  recall=RecallConfig(...), write=[...]))`；`store="data/u001"` 把 session
  和长期记忆接在一起（含压缩摘要沉淀）
- `StorageConfig` 里 `namespace` **必填**（多用户隔离全靠它）

示例见 [examples/memory.py](examples/memory.py)、
[examples/memory_tutor.py](examples/memory_tutor.py)、
[examples/handwritten_memory_bot.py](examples/handwritten_memory_bot.py)。

**知识库**（`promptkit.knowledge`）开箱即用：

```python
from promptkit.knowledge import KnowledgeBase
```

示例见 [examples/knowledge_base.py](examples/knowledge_base.py)、
[examples/persistent_kb.py](examples/persistent_kb.py)。

## 可观测性

核心思路：**执行和显示解耦**。`Agent` / `Executor` / `Step` 只发事件、不
`print`；显示由你挂的 handler 决定，同一套代码能喂给控制台、日志、
WebSocket、GUI。

```python
from promptkit.core import EventBus, EventType
from promptkit.core.display_sys import console_display

bus = EventBus()
bus.on(console_display(show_thinking=True, show_tools=True))
agent = Agent(bus=bus)
```

事件类型（`EventType` 同时是 `str` 子类）：

| 事件 | 含义 |
| --- | --- |
| `THINKING_START` / `THINKING_DELTA` / `THINKING_END` | 思考过程（DeepSeek / o1 / Claude 的思考都能识别） |
| `TOOL_CALL_START` / `TOOL_CALL_DELTA` / `TOOL_CALL_END` | 工具调用（流式下参数一段段拼） |
| `TOOL_RESULT` | 工具返回值 |
| `MESSAGE_DELTA` / `MESSAGE_END` | 最终回复（流式逐 token，`END` 带完整文本） |
| `STEP_START` / `STEP_END` | 每个 Step 的开始 / 结束 |
| `ERROR` | 某一步抛异常（`Break` / `Continue` / `Return` / `Goto` 算控制流，不发） |

四种用法：

```python
answer = agent.ask("你好")                     # ① 默认静默
answer = agent.ask("你好", verbose=True)       # ② 这一次打印
for event in agent.ask_stream("你好"): ...     # ③ 自己迭代事件（流式）
ctx = wf.run(agent=agent, bus=bus)             # ④ 工作流级别（on_event= 也行）
```

**结果走 stdout，过程走 stderr**，重定向即分离：
`python x.py > result.txt 2> trace.txt`。`wf.run(..., out=buf)` 可改结果去向。
`verbose` 分级用 `Verbosity.SILENT / NORMAL / VERBOSE / DEBUG`（`True` 等价
`VERBOSE`）。日志落盘：`wf.run(..., on_event=log_display("logs/run.log"))`；
`json_display(stream=...)` 输出一行一条 JSON 适合日志 / WebSocket。
需要**结构化步骤 trace**时给 `wf.run(..., _trace_steps=True)`：每个 Step 前后
往 `ctx.trace` 追加 `{phase, step, type, duration_ms, status, error}`；
`_dump_trace=True` 会额外把它写成 `logs/trace_<时间>.jsonl`（路径回填到
`ctx["_trace_file"]`）。默认关闭，所以原有 `ctx.trace` 的字符串标记不受影响。

完整示例见 [examples/streaming_events.py](examples/streaming_events.py)、
[examples/console_output.py](examples/console_output.py)。

## 全局设置 settings

DSL 的上限 / 默认值集中在一个全局对象上；**在「用的时候」读**，所以改全局
值对已构造好的工作流也生效：

```python
from promptkit.core import settings, using_settings, reset_settings

settings.max_retries = 3                          # 全局改
with using_settings(max_iterations=5):            # 临时改，退出自动还原
    wf.run()
```

| 设置 | 默认 | 管什么 |
| --- | --- | --- |
| `max_iterations` | `100` | `While` / `DoWhile` 迭代上限 |
| `max_for_iterations` | `10000` | `For` / `ForEach` 迭代上限 |
| `max_recursion_depth` | `100` | `Function` 递归上限 |
| `max_steps` | `50` | 一个 `Workflow` 最多几步 |
| `max_jumps` | `10000` | `Goto` 连续跳转上限 |
| `max_retries` | `2` | `LLMStep` 结构化输出重试次数 |
| `max_tool_rounds` | `8` | 工具调用循环上限 |
| `auto_schema` | `False` | `LLMStep` 默认注入 JSON Schema |
| `encoding` | `"utf-8"` | `Save` / `ReadFile` / `SaveHistory` 编码 |

显式传参永远优先（`While(cond, max_iterations=10)` 用 10）。写错设置名在
调用时立刻 `TypeError` 并列出可用项。⚠️ 进程级可变状态，测试用
`using_settings(...)` / `reset_settings()` 收尾。

## 项目结构

```
promptkit/
├─ src/promptkit/          # 包本体（src 布局）
│  ├─ agent.py             # Agent：请求、工具循环、发事件（自己不 print）
│  ├─ prompt.py            # Prompt 对象与 {{var}} 渲染
│  ├─ executor.py          # 结构化输出：JSON 抽取 + 重试
│  ├─ config.py            # 密钥查找 / .env 读取 / 模型名
│  ├─ session.py           # Session：会话持久化
│  ├─ features.py          # 功能开关（实验性功能默认关闭）
│  ├─ knowledge.py         # 知识库（实验性）
│  ├─ core/                # 工作流核心：Context / Step / Workflow / DSL /
│  │                       #   表达式 / 事件 / 显示 / Caution / Formatter / 工具
│  ├─ cautions/            # 内置原子约束 + 预置组合
│  ├─ client_sys/          # 多厂商客户端（OpenAI 兼容 / Anthropic / 豆包 / 多模态）
│  ├─ memory/              # 记忆系统：KV / 向量 / 分层 / 适配器
│  ├─ coding/              # 编码任务：沙箱 / diff / 执行 / FixLoop
│  ├─ orchestrate/         # 多代理编排：Task / Plan / Orchestrator / DelegateTool
│  ├─ skill/               # Skill：@skill / SkillContext / SkillRegistry
│  ├─ mcp/                 # MCP 客户端：load_mcp_tools / 安全门（仅 HTTP 传输）
│  ├─ api/                 # 对外 API 汇总
│  ├─ conversation/        # 会话管理
│  ├─ dsl_libs/            # DSL 辅助库
│  └─ .env.example         # 密钥模板
├─ examples/               # 36 个可运行示例（见下节索引）
├─ tests/                  # pytest；含 multimodal / mcp / conversation /
│                          #   integration 子目录
├─ docs/                   # 专题文档（见文末）
├─ pyproject.toml          # 打包配置 + mypy 基线
└─ requirements.txt        # 运行时依赖
```

`import promptkit.core` 不会把 SQLite、向量库、子进程、编排这些拖进来
（分层有测试盯着）。每一层 `__init__.py` 都再导出自己整个子树，所以
`from promptkit.core import LLMStep` 和
`from promptkit.core.steps.actions.llm import LLMStep` 是同一个对象。

## 示例索引

`examples/` 下全部可直接 `python examples/xxx.py` 运行（真模型示例需要
配好密钥）：

**入门**

- [interview.py](examples/interview.py) — 模拟面试工作流（交互式，While 循环）
- [refund_workflow.py](examples/refund_workflow.py) — 结构化输出 + 条件分支
- [emotion_analysis.py](examples/emotion_analysis.py) — 情绪分析
- [multi_vendor.py](examples/multi_vendor.py) — 同一工作流换多家厂商跑
- [chat_bot.py](examples/chat_bot.py) — 聊天机器人
- [chat_session.py](examples/chat_session.py) — 会话持久化
- [conversation_crud.py](examples/conversation_crud.py) /
  [conversation_branching.py](examples/conversation_branching.py) — 会话管理

**工作流与输出控制**

- [cautions.py](examples/cautions.py) — Caution 约束与 Formatter
- [while_hooks.py](examples/while_hooks.py) — 循环 Hook 扩展点
- [hooks_and_assertions.py](examples/hooks_and_assertions.py) — 断言与钩子
- [console_output.py](examples/console_output.py) — 渲染 / 分级 / 日志落盘
- [streaming_events.py](examples/streaming_events.py) — 事件流

**工具与编码**

- [tool_calling.py](examples/tool_calling.py) — function calling
- [web_search.py](examples/web_search.py) — 联网搜索
- [auto_files.py](examples/auto_files.py) — WriteFile 模型自主写文件
- [list_desktop.py](examples/list_desktop.py) — RUN_COMMAND
- [coding_agent.py](examples/coding_agent.py) — 改代码 → 跑测试 → 修复闭环
- [coding_chat.py](examples/coding_chat.py) — 多轮对话里把应用做出来
- [mcp_demo.py](examples/mcp_demo.py) / [mcp_agent.py](examples/mcp_agent.py) — MCP

**多模态**

- [image_generation_workflow.py](examples/image_generation_workflow.py) — 生图三场景
- [multimodal_chat.py](examples/multimodal_chat.py) — 多模态对话

**记忆与知识库**

- [memory.py](examples/memory.py) — 记忆系统（不需要 API Key）
- [memory_tutor.py](examples/memory_tutor.py) — 会记住你的答疑教练
- [handwritten_memory_bot.py](examples/handwritten_memory_bot.py) — 手写记忆机器人
- [knowledge_base.py](examples/knowledge_base.py) /
  [persistent_kb.py](examples/persistent_kb.py) — 知识库（实验性，需 enable）

**编排与 Skill**

- [orchestrate.py](examples/orchestrate.py) — 多代理：固定模板 + 汇总（`--plan` 换 LLM 规划）
- [orchestrate_plan.py](examples/orchestrate_plan.py) /
  [orchestrate_plan_tool.py](examples/orchestrate_plan_tool.py) /
  [orchestrate_skill.py](examples/orchestrate_skill.py) /
  [orchestrate_refactor.py](examples/orchestrate_refactor.py) — 编排各场景
- [skills.py](examples/skills.py) — Skill 注册表与互调
- [skill_refund.py](examples/skill_refund.py) — 业务 Skill：规则 + 工作流
- [skill_package.py](examples/skill_package.py) — 目录式技能包：资源按需读 + 脚本参数自动推断
- [course_assistant.py](examples/course_assistant.py) — `skills=[DSL 函数]` 挂进聊天

## 常见问题 FAQ

**Q：变量写了 `{name}` 没替换成值，模型答非所问？**
变量是**双花括号** `{{name}}`。单花括号不报错也不替换，是最容易踩的坑。
`$name` / `${name}` 也能用但宽松（找不到就原样保留）；要「写错立刻报错」
就用 `{{}}`。

**Q：`Echo(Var("x"))` 和 `Echo("{{x}}")` 有区别吗？**
没有，`Echo` 字符串支持 `{{key}}` 插值（只认 `{{}}`，不认 `$key`，避免
`价格是 $100` 被误伤）。但 `Echo` 不直接输出 `Condition` / 布尔表达式结果
——要按条件选文本用 `When(cond, a, b)`。

**Q：`Var("n") > 1 and Var("m") < 2` 为什么报错？**
`and` / `or` / `not` 不能重载，`Expr` 的 `bool()` 会抛 `TypeError`。
组合条件写成 `(Var("n") > 1) & (Var("m") < 2)`（`&` 优先级高于比较，括号
必须加）。判断放进 `If(...)` / `While(...)` / `Condition(...)`。

**Q：`Transform("name", ...)` 直接放进 `Workflow(...)` 报 `TypeError`？**
`Transform` 是表达式不是步骤（求值出一个值，但没有「执行」语义）。包进
`Set("x", Transform("join", Var("lines")))` / `Echo(Transform("upper", Var("s")))`，
或只为了副作用用 `FunctionStep(lambda ctx: f())`。同理 `Set("x", SomeStep())`
也是错的 —— 步骤不是值。旧的 `Call(f, ...)` 仍可用（会发 `DeprecationWarning`），
但它只能翻译注册过的具名函数。

**Q：`Save` 路径想用循环变量 `i`，f-string 不行？**
`Save` 的 `file=` 在构造时是普通字符串，f-string 在构建工作流时就定死了。
用模板：`Save(Var("img"), file="cats/cat_{{i}}.png")`，`{{i}}` 在保存时按
上下文展开。

**Q：MCP 连不上本地 stdio server？**
`promptkit.mcp` **只支持 Streamable HTTP 传输**（规格约束），不支持
stdio / sse。`load_mcp_tools(..., transport=...)` 传别的会直接报错。

**Q：装了 `math` extra 后 sympy 报错 / 公式被改错？**
sympy 的 `parse_latex` 需要 `antlr4-python3-runtime` 且版本挑剔（sympy
1.14 硬性要求 `==4.11`），装上后按它报错里的版本号补装即可。另外 sympy
解析是有损的（`ln` 变 `log(x, E)`、`\sqrt5` 解析失败、多行公式放弃），
所以默认走内置的零依赖 Unicode 渲染；要 sympy 排版显式
`render_math(..., prefer_sympy=True)`。

**Q：`python examples/xxx.py` 报 9009 / 「不是内部或外部命令」？**
这是环境 PATH 问题（终端找不到 `python`），不是项目问题。Windows 上先确认
Python 已加入 PATH，或用完整解释器路径运行。技能包内部已自动探测可用解释器
（`resolve_python`），也可以用环境变量 `PROMPTKIT_PYTHON` 直接指定。

**Q：改了 `agent.start_prompts` 没生效？**
`agent.messages` 是构造时算好的缓存。改完要 `agent.reset()`（清空历史）
或 `agent.rebuild()`（保留历史）；`SystemPrompt()` 默认帮你做。接会话时
`session=` 存在会自动改成 `reset=False`，不会抹掉装回来的历史。

**Q：聊天开头问卷每次启动都重问？**
`Session` 默认只存对话历史，Context 变量不在 `messages` 里。传
`wf.run(..., session="s.json", variables=["basic_info"])` 让指定变量跨进程
保存，配合 `If(Var("basic_info") == {})` 只问一次。

## 测试与开发

```bash
pip install -e ".[dev]"

pytest                        # 跑全部测试（tests/，不需要网络和 API Key）
python tests/test_promptkit.py   # 冒烟测试

mypy                          # 类型检查（基线见 pyproject.toml [tool.mypy]）
```

- 测试在 `tests/`，含 `multimodal/`、`mcp/`、`conversation/`、
  `integration/` 子目录与 `conftest.py`
- mypy 基线是**宽松的**（`check_untyped_defs = false` 等），按
  [docs/typing-any-audit.md](docs/typing-any-audit.md) 的分批计划逐步收紧
- 打包：`pip install build && python -m build`（产物在 `dist/`）

## 贡献指南

1. **提 Issue**：先搜索已有 issue；bug 报告请附最小复现、完整报错、运行
   环境（OS / Python 版本 / 厂商与模型）。
2. **提 PR**：分支自 `main`，改动聚焦单一目的；涉及行为变更请同步更新
   相关示例或测试。
3. **代码风格**：遵循现有代码风格（中文注释、模块尾部职责说明）；类型注解
   遵守 `pyproject.toml` 的 mypy 宽松基线，新代码尽量带注解，不要为提高
   严格度而大改旧文件。
4. **测试**：新功能必须有测试；测试不得依赖网络或真实 API Key（用
   FakeAgent / 脚本化 agent，参考现有测试写法）。
5. **提交信息**：用简洁中文或英文说明「为什么」而非罗列「是什么」；
   一个 commit 只做一件事。

## 许可证

本仓库暂未包含 LICENSE 文件。在采用、分发或二次开发前，请联系作者 /
维护者确认许可条款。

## 相关文档

- [docs/orchestrate-api-v2.md](docs/orchestrate-api-v2.md) — 编排 API v2 设计
- [docs/typing-any-audit.md](docs/typing-any-audit.md) — 类型注解审计与 mypy 收紧计划
