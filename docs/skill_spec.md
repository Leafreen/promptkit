# Skill 包约定（skill_spec）

> 版本：v1（阶段 0 定稿）
> 状态：**目标契约**。阶段 1~3 逐步实现；与现有实现的差异见文末「迁移对照」。

一个 **skill 包就是一个目录**。任何符合本约定的目录，丢进来就能用：

```python
from promptkit.skill import SkillPackage

pkg = SkillPackage.load("skills/ppt-master")
pkg.install_deps()
tools = pkg.tools(python=PY, cwd="./output", timeout=600)
```

## 1. 目录结构

```text
my-skill/
├── SKILL.md              # 必需：frontmatter + 正文说明书
├── requirements.txt      # 可选：pip 依赖
├── scripts/              # 可选：可独立执行的 Python 脚本
│   ├── build.py
│   └── render.py
├── resources/            # 可选：参考文档、模板、数据（推荐目录名）
│   ├── guide.md
│   └── template.pptx
└── output/               # 可选：默认产物目录（见 §5.4）
```

**兼容约定**：`references/`、`assets/`、`templates/` 也按资源目录索引
（现有技能包如 ppt-master 大量使用这三个名字）。资源目录一律**只读**；
`resources/` 与其余三者同时存在时全部索引，去重按相对路径。

不属于约定的目录（如 `workflows/`、`tests/`）不索引、不报错。

## 2. SKILL.md

### 2.1 frontmatter 字段表

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `name` | ✅ | str | 包名，kebab-case（`^[a-z0-9]+(-[a-z0-9]+)*$`），命名空间用 |
| `description` | ✅ | str | 一句话说明「做什么 + 什么时候用」 |
| `entrypoint` | | str | 主入口脚本（相对包根，如 `scripts/main.py`），仅声明用途 |
| `python_version` | | str | 如 `">=3.10"`；加载时校验当前解释器，不满足抛 `InvalidSkillPackageError` |
| `dependencies` | | list[str] | pip 包名；与 `requirements.txt` 取并集 |
| `tools` | | list[str] | 工具白名单（见 §5.5）；不声明则按默认启用表 |

- 缺 `name` → `InvalidSkillPackageError("SKILL.md 缺少 name 字段")`
- 缺 `description` → `InvalidSkillPackageError("SKILL.md 缺少 description 字段")`
- frontmatter 未闭合 → `InvalidSkillPackageError`，错误信息含行号
- frontmatter 支持 YAML 子集：字符串、列表、块标量（`>` / `|`）；不引入新依赖，手写解析

### 2.2 正文

正文就是给模型看的说明书：流程、命令行用法、注意事项。加载时不进上下文，
模型调 `read_skill` 才读（渐进披露）。

## 3. scripts/ 约定

- 脚本必须可独立执行：`python scripts/xxx.py --args ...`
- **不逐个挂成工具**（352 个脚本会把 tools schema 撑爆，模型反而选不准）
- `list_scripts()` 返回清单（名字 + 首行 docstring 作为描述）
- `run_script(name, args)` 按需执行：
  - `name` 是 `scripts/` 下的相对路径（允许子目录，如 `svg/convert.py`），
    必须是 `.py`，越界抛 `PathEscapeError`，不存在抛 `ScriptNotFoundError`
  - 非零退出码**不抛异常**，返回 `{stdout, stderr, returncode}` 的 JSON，
    让模型看到错误自行修正
- 以下文件不进清单：`__init__.py`、`_*.py`、`test_*.py`、`*_test.py`，
  以及 `tests/` `test/` `docs/` `doc/` `__pycache__/` 目录下的任何文件

## 4. resources/ 约定

- 纯数据文件：文档、模板、图片、字体；**不含可执行代码**
- `list_resources()` 返回全部资源相对路径（JSON）
- `read_resource(path)`：
  - `path` 相对包根，越界抛 `PathEscapeError`
  - 文本直接返回内容；单文件超 100KB 截断并标注 `[已截断]`
  - 二进制返回 `[二进制文件，大小 N 字节]`，不读内容

## 5. 工具集

### 5.1 生成规则（按需生成，没有的目录不生成对应工具）

| 条件 | 工具 | 默认 |
|---|---|---|
| 总是 | `read_skill(section="")` | ✅ |
| 任一资源目录存在 | `list_resources()` | ✅ |
| 任一资源目录存在 | `read_resource(path)` | ✅ |
| 总是 | `write_file(path, content)` | ✅ |
| `scripts/` 存在 | `list_scripts()` | ✅ |
| `scripts/` 存在 | `run_script(name, args="")` | ✅ |
| 仅 `enable={"run_command"}` | `run_command(cmd)` | ❌ |

### 5.2 `tools()` 签名

```python
def tools(
    self,
    *,
    python: str | None = None,        # run_script 用的解释器；None = sys.executable
    cwd: str | Path | None = None,    # 工作目录（run_script 的 cwd、write_file 的根）
    timeout: int = 300,
    enable: set[str] | None = None,   # 白名单：只启用这些
    disable: set[str] | None = None,  # 黑名单：去掉这些
    namespace: bool = False,          # True 时工具名加 "{name}__" 前缀（多包合并用）
    allow_commands: list[str] | None = None,  # run_command 的命令白名单
) -> list[Tool]:
```

### 5.3 环境事实写进工具描述

每个工具的描述必须自带环境事实，用户 prompt 不再重复：
`run_script` 描述里写死「Python 解释器是 X，工作目录是 Y」；
`write_file` 描述里写死「文件写到 Z 下」。

### 5.4 读写分离

`cwd` 是**唯一**的工作目录概念：

- `run_script` 以 `cwd` 为工作目录执行
- `write_file` 以 `cwd` 为根目录，越界拒绝
- `cwd=None` 时 = 包根下的 `output/`（没有就自动创建）
- 技能包本体（SKILL.md / scripts / resources）始终只读

技能目录装在只读位置时，调用方传一个可写 `cwd` 即可，包不用改。

### 5.5 默认最小权限

- `run_script` 默认启用（只能跑 `scripts/` 下登记在册的 `.py`）
- `run_command` 默认**关闭**；`enable` 后还必须配 `allow_commands`
  （如 `["python", "git"]`），未配置时拒绝所有命令，
  错误信息提示「请配置 allow_commands」
- `enable` / `disable` 对全部工具生效；两者同给时先 enable 后 disable

## 6. 门面 API

```python
@dataclass
class SkillPackage:
    root: Path
    name: str
    description: str
    skill_md: str                 # SKILL.md 正文（不含 frontmatter）
    frontmatter: dict
    has_scripts: bool
    has_resources: bool

    @classmethod
    def load(cls, root: str | Path) -> "SkillPackage": ...
    def tools(self, **kwargs) -> list[Tool]: ...        # 见 §5.2
    def system_prompt(self, *, max_lines: int = 50) -> str: ...
    def install_deps(self, python: str | None = None) -> None: ...
```

- `install_deps()`：存在 `requirements.txt` 时执行 `pip install -r`，
  失败抛 `DependencyInstallError`；没有则空操作
- `system_prompt()`：生成注入 system prompt 的片段（包名、描述、
  SKILL.md 前 `max_lines` 行、包根路径、可用工具清单、脚本清单）
- 批量加载：`load_skills_from_dir(root, pattern="*", recursive=False)`，
  单个包失败记 warning 不中断

## 7. 安全边界

- 所有文件操作统一走 `_safe_path(path, base)`：`resolve` 后必须
  `is_relative_to(base)`，否则抛 `PathEscapeError`；符号链接按
  resolve 后的真实路径判定
- `run_script` 只能跑 `scripts/` 内的 `.py`
- **这不是容器级沙箱**：脚本仍以当前用户权限运行，只加载可信技能包

## 8. 错误体系

所有异常含 `code` / `message` / `hint` / `docs_url` 四个字段，
挂在现有 `SkillError` 下（`promptkit.skill.errors`）：

| 异常 | 基类 | code | 何时抛 |
|---|---|---|---|
| `InvalidSkillPackageError` | `SkillValidationError` | `SKILL_INVALID` | 目录不存在、SKILL.md 缺失/不合法 |
| `DependencyInstallError` | `SkillDependencyError` | `SKILL_DEP_INSTALL` | `install_deps` 失败 |
| `ScriptNotFoundError` | `SkillError` | `SKILL_SCRIPT_NOT_FOUND` | `run_script` 名字不存在 |
| `ResourceNotFoundError` | `SkillError` | `SKILL_RESOURCE_NOT_FOUND` | `read_resource` 路径不存在 |
| `PathEscapeError` | `SkillValidationError` | `SKILL_PATH_ESCAPE` | 任何越界路径 |
| `DuplicateToolError` | `SkillError` | `SKILL_DUP_TOOL` | 多包合并重名 |

`docs_url` 指向 `docs/errors/skill-*.md`（随异常一并新增）。

## 9. 多包合并

```python
tools = [*pkg_a.tools(namespace=True), *pkg_b.tools(namespace=True)]
# 工具名变成 ppt-master__run_script、data-analysis__run_script
```

`namespace=True` 时工具名 = `{name}__{tool}`（包名的 `-` 换 `_`）。
合并后重名抛 `DuplicateToolError`，错误信息写明哪两个包冲突。

## 10. 示例包

| 包 | 路径 | 用途 |
|---|---|---|
| hello-skill | `examples/skills/hello-skill/` | 最小合规包，每阶段的冒烟验证 |
| ppt-master | `examples/skills/ppt-master/` | 精简脱敏版，验证真实形态（嵌套脚本、references 兼容） |

## 迁移对照（现状 → 本约定）

| 现状（`as_skill_package`） | 本约定 | 迁移 |
|---|---|---|
| `as_skill_package(dir)` | `SkillPackage.load(dir)` | 旧函数保留，加 `@deprecated` 转发 |
| 脚本逐个挂工具 `pkg_script` | `list_scripts` + `run_script(name)` | 阶段 2 切换默认行为 |
| `list_skill_resources` / `read_skill_resource` | `list_resources` / `read_resource` | 新工具名按 §5.1 |
| 资源目录 `references/assets/templates` | `resources/` 为主，三者兼容 | 兼容索引，不用改包 |
| 技能说明工具（进 tools 列表） | `read_skill` | 等价替换 |
| `command_tool()` 手工挂载 | `run_command`，默认关闭 + 白名单 | 阶段 5 |
| `write_tool()` 手工挂载 | `write_file`，总是生成 | 阶段 2 |
| 环境事实手写进 prompt | `system_prompt()` 自动注入 | 阶段 3 |
