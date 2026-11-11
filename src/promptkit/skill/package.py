# skill/package.py
'''目录式技能包（SKILL.md + references/ + assets/ + scripts/）。

`as_skill` 只把 SKILL.md 当一段说明读进来；技能包在这一层之上补四件事：

1. **渐进披露**：把资源清单附在说明末尾，模型需要细节时再读。
2. **资源访问**：只读、限定在技能目录内；二进制只返回元信息。
3. **脚本执行**：从 argparse 推断参数 Schema，包成普通 Tool。
4. **命令执行**：`command_tool()` 给大型技能用的通用命令工具（固定工作目录、
   放宽超时、重写 python 前缀），不必把几百个脚本逐个挂成工具。

    from promptkit.skill import as_skill_package

    pkg = as_skill_package("skills/ppt-master", scripts=[])
    ChatWorkflow(
        role="PPT 助手",
        rules=[pkg.routing_prompt()],          # 让「何时用」始终可见
        tools=[*pkg.tools(), pkg.command_tool(python=sys.executable)],
    )

两条注入路线：

- **工具路由**（默认，省 token）：模型在工具列表里看到技能名和描述，
  匹配时调用技能工具读取正文；正文不会一开始就进上下文。
- **提示词路由**（入口必须常驻）：`rules=[pkg.routing_prompt()]` 注入短触发块；
  `rules=[pkg.instructions]` 则注入完整入口（占位符保持原样）。

目录约定：

    ppt-master/
    ├── SKILL.md / AGENT.md   # 入口说明（二选一，SKILL.md 优先）
    ├── references/           # 参考资料，按需读取
    ├── assets/               # 图片、字体等素材
    ├── templates/            # 模板文件
    └── scripts/*.py          # 可执行脚本

frontmatter 额外字段（都是可选的）：

    allowed-tools: [list_resources, read_resource, run_script]
    dependencies: [python-pptx, pillow]
    scripts:
      build_deck:
        path: scripts/build_deck.py
        description: 根据大纲生成 pptx

`allowed-tools` 会兼容 Anthropic 技能里的 `Read` / `Glob` / `Bash` 写法，
映射到本模块的 `read_resource` / `list_resources` / `run_script`；不认识的
名字只记 warning，不会中断加载。

安全边界：资源路径必须落在技能目录内；脚本只能跑加载时登记过的 `.py`；
不走 shell、带超时和输出截断。它不是容器级沙箱 —— 脚本本身仍以当前用户
权限运行，只应加载可信技能。
'''
from __future__ import annotations

import ast
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from promptkit.core.tool import Tool
from promptkit.core.tool.base import ToolSchema
from promptkit.skill.errors import (
    DependencyInstallError,
    DuplicateToolError,
    InvalidSkillPackageError,
    PathEscapeError,
    ScriptNotFoundError,
    SkillDependencyError,
    SkillError,
    SkillValidationError,
)
from promptkit.skill.markdown import MarkdownSkillTool, _split_frontmatter, as_skill
from promptkit.skill.python import resolve_python

__all__ = [
    "ScriptSpec",
    "SkillPackage",
    "as_skill_package",
    "load_skills_from_dir",
    "merge_tools",
]

logger = logging.getLogger("promptkit.skill")

#: 入口文件的候选名（`AGENT.md` 是部分技能包的写法）
_ENTRY_NAMES = (
    "SKILL.md",
    "skill.md",
    "Skill.md",
    "AGENT.md",
    "agent.md",
    "Agent.md",
    "AGENTS.md",
    "Agents.md"
)

#: 资源工具 / 脚本工具的类别（用于 allowed-tools / tools 白名单）
_TOOL_ALIASES: Dict[str, str] = {
    # 本模块的规范名
    "skill": "read_skill",
    "readskill": "read_skill",
    "readresource": "read_resource",
    "listresources": "list_resources",
    "listscripts": "list_scripts",
    "runscript": "run_script",
    "runcommand": "run_command",
    "writefile": "write_file",
    # Anthropic 技能里的常见叫法
    "read": "read_resource",
    "readfile": "read_resource",
    "glob": "list_resources",
    "ls": "list_resources",
    "list": "list_resources",
    "listfiles": "list_resources",
    "bash": "run_script",
    "shell": "run_script",
    "execute": "run_script",
    "run": "run_script",
    "write": "write_file",
    "edit": "write_file",
}

#: pip 包名 -> 实际 import 名
_IMPORT_ALIASES: Dict[str, str] = {
    "python-pptx": "pptx",
    "python-docx": "docx",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "pymupdf": "fitz",
    "python-dateutil": "dateutil",
    "python-slugify": "slugify",
    "pydantic": "pydantic",
}

#: 可以直接按文本读的扩展名；不在表里的按二进制处理
_TEXT_SUFFIXES = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".htm",
        ".css",
        ".xml",
        ".svg",
        ".tex",
        ".sql",
        ".sh",
        ".ps1",
    }
)

_MAX_RESOURCE_CHARS = 200_000
_MAX_RESOURCE_BYTES = 4_000_000
_MAX_LISTED_RESOURCES = 200
_MAX_APPENDIX_ITEMS = 30
_MAX_OUTPUT_CHARS = 20_000
_DEFAULT_TIMEOUT = 120.0
#: scripts/ 下这些目录装的是库代码而非命令行入口，自动发现时跳过
_NON_ENTRY_DIRS = frozenset({"tests", "test", "docs", "doc", "__pycache__"})

#: 技能包名约定：kebab-case（小写字母/数字，用 - 连接）
_PACKAGE_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
#: python_version 写法：`>=3.10`、`==3.11.2`、`3.12` 等
_PYTHON_VERSION_RE = re.compile(
    r"^\s*(?P<op>>=|<=|>|<|==|!=)?\s*"
    r"(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?\s*$"
)


# ============================ 脚本 ============================
@dataclass(frozen=True)
class ScriptSpec:
    '''一个可执行脚本：`name` 是相对 scripts/ 的路径，`path` 是绝对路径。'''

    name: str
    path: Path
    description: str


# ============================ 技能包 ============================
@dataclass
class SkillPackage:
    '''一个目录式技能的运行时视图。'''

    root: Path
    skill: MarkdownSkillTool
    skill_md: str = ""               # SKILL.md 正文（不含 frontmatter / 附录）
    scripts: Tuple[ScriptSpec, ...] = ()
    references: Tuple[Path, ...] = ()
    assets: Tuple[Path, ...] = ()
    templates: Tuple[Path, ...] = ()
    resources: Tuple[Path, ...] = ()   # 新约定的 resources/ 目录
    dependencies: Tuple[str, ...] = ()
    missing_dependencies: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()      # 旧字段：allowed-tools 的 resource 类别
    tool_whitelist: Tuple[str, ...] = ()     # frontmatter `tools` 的全量白名单
    entrypoint: str = ""                     # frontmatter `entrypoint`（仅声明用途）
    workdir: Path = field(default_factory=lambda: Path("."))
    timeout: float = _DEFAULT_TIMEOUT
    max_output_chars: int = _MAX_OUTPUT_CHARS
    max_resource_chars: int = _MAX_RESOURCE_CHARS

    def __post_init__(self) -> None:
        self.timeout = max(0.1, float(self.timeout))
        self.max_output_chars = max(100, int(self.max_output_chars))
        self.max_resource_chars = max(100, int(self.max_resource_chars))

    # ---------------- 门面（skill_spec v1 §6） ----------------
    @classmethod
    def load(cls, root: Union[str, Path]) -> "SkillPackage":
        '''严格加载一个技能包。

        相比宽松的 `as_skill_package()`，这里额外校验 frontmatter：

        - `name` 必填，且是 kebab-case（`^[a-z0-9]+(-[a-z0-9]+)*$`）
        - `description` 必填
        - `python_version`（如有）要满足当前解释器

        任一不满足抛 `InvalidSkillPackageError`。校验通过后走完整的
        资源/脚本发现逻辑。
        '''
        resolved = _resolve_root(root)
        entry = _entry_path(resolved)
        try:
            text = entry.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise InvalidSkillPackageError(f"读不了入口文件 {entry}：{exc}") from exc

        try:
            front, body = _split_frontmatter(text, entry)
        except SkillValidationError as exc:
            raise InvalidSkillPackageError(str(exc)) from exc
        if not body.strip():
            raise InvalidSkillPackageError(
                f"入口文件 {entry} 只有 frontmatter，没有正文"
            )
        _validate_package_frontmatter(front, entry)

        return as_skill_package(resolved)

    @property
    def name(self) -> str:
        return self.skill.name

    @property
    def description(self) -> str:
        return self.skill.description

    @property
    def frontmatter(self) -> Dict[str, Any]:
        return self.skill.frontmatter

    @property
    def has_scripts(self) -> bool:
        return bool(self.scripts)

    @property
    def has_resources(self) -> bool:
        return bool(
            self.references or self.assets or self.templates or self.resources
        )

    def system_prompt(
        self,
        *,
        max_lines: int = 50,
        python: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
    ) -> str:
        '''生成注入 system prompt 的片段（包名、描述、根目录、环境事实、
        正文节选、工具清单、脚本清单）。环境事实由此统一给出，用户 prompt
        不再手写；这里只计算路径，不会创建工作目录。'''
        body_lines = self.skill_md.splitlines()
        if max_lines > 0 and len(body_lines) > max_lines:
            excerpt = "\n".join(body_lines[:max_lines])
            excerpt += f"\n…（正文共 {len(body_lines)} 行，用 read 类工具按需读取）"
        else:
            excerpt = self.skill_md

        workdir = (
            Path(cwd).expanduser().resolve()
            if cwd is not None
            else (self.root / "output").resolve()
        )
        env: List[str] = [
            f"技能：{self.name}",
            f"描述：{self.description}",
            f"根目录：{self.root}",
            f"Python 解释器：{resolve_python(python, root=self.root)}",
            f"工作目录：{workdir}",
        ]
        if self.entrypoint:
            env.append(f"入口脚本：{self.entrypoint}")
        lines: List[str] = [*env, "", excerpt.strip()]
        tool_names = self._default_tool_names()
        if tool_names:
            lines.append("")
            lines.append("可用工具：" + "、".join(tool_names))
        if self.scripts:
            lines.append("")
            lines.append("脚本：")
            for spec in self.scripts:
                lines.append(f"- `{spec.name}` — {spec.description}")
        return "\n".join(lines)

    def install_deps(self, python: Optional[str] = None) -> None:
        '''安装技能包根目录下 `requirements.txt` 声明的依赖。

        没有该文件时是空操作；安装失败抛 `DependencyInstallError`。
        '''
        requirements = self.root / "requirements.txt"
        if not requirements.is_file():
            return
        interpreter = resolve_python(python, root=self.root)
        completed = subprocess.run(
            [interpreter, "-m", "pip", "install", "-r", str(requirements)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise DependencyInstallError(
                f"安装 {requirements.name} 失败（退出码 {completed.returncode}）",
                hint="检查网络 / pip 源；可先用命令行 pip install -r 手动确认",
            )

    # ---------------- 对外 ----------------
    def tools(
        self,
        *,
        python: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
        enable: Optional[Set[str]] = None,
        disable: Optional[Set[str]] = None,
        namespace: bool = True,
        allow_commands: Optional[List[str]] = None,
    ) -> List[Tool]:
        '''返回挂给模型的工具（skill_spec v1 §5）。

        脚本不逐个挂成工具，而是 `list_scripts` + `run_script(name, args)`；
        `read_skill` / `write_file` 总是生成。`cwd` 是唯一工作目录：
        `run_script` 在它下面执行、`write_file` 写到它下面，None 时用
        包根下的 `output/`。

        `run_command` 默认不生成；显式给出 `allow_commands`（白名单）或
        在 `enable` 里点名它时才会启用。`enable` 是白名单语义（只留这些），
        `disable` 最后生效；`namespace=True` 给工具名加 `{name}__` 前缀。
        '''
        interpreter = resolve_python(python, root=self.root)
        workdir = self._resolve_cwd(cwd)
        limit = max(0.1, float(timeout) if timeout is not None else self.timeout)

        candidates: Dict[str, Tool] = {}
        candidates["read_skill"] = self._read_skill_tool()
        if self.has_resources:
            candidates["list_resources"] = self._list_resources_tool()
            candidates["read_resource"] = self._read_resource_tool()
        candidates["write_file"] = self.write_tool(root=workdir)
        if self.has_scripts:
            candidates["list_scripts"] = self._list_scripts_tool()
            candidates["run_script"] = self._run_script_tool(
                interpreter, workdir, limit
            )
        # §5.5：run_command 默认关闭；给白名单或 enable 点名才生成
        candidates["run_command"] = self._run_command_tool(
            interpreter, workdir, limit, allow_commands
        )

        enabled: Set[str] = set(candidates) - {"run_command"}
        if enable is not None:
            enabled = set(enable) & set(candidates)
        if allow_commands:
            # 显式给了白名单就是明确要用命令工具，不必再写一遍 enable
            enabled.add("run_command")
        if self.tool_whitelist:
            enabled &= set(self.tool_whitelist)
        if disable is not None:
            enabled -= set(disable)

        prefix = self._namespace_prefix() if namespace else ""
        tools: List[Tool] = []
        for canonical, tool in candidates.items():
            if canonical not in enabled:
                continue
            if prefix:
                tool.name = f"{prefix}{canonical}"
            tools.append(tool)
        return tools

    def _read_skill(self, section: str = "") -> str:
        '''读技能说明书正文；`section` 传章节标题时只返回该节。'''
        text = self.skill_md
        if not section:
            return text
        heading = re.compile(
            rf"^#{1,6}\s+{re.escape(section)}\s*$", re.MULTILINE
        )
        match = heading.search(text)
        if match is None:
            return text
        start = match.end()
        rest = text[start:]
        following = re.compile(r"^#{1,6}\s+", re.MULTILINE)
        next_heading = following.search(rest)
        end = start + (next_heading.start() if next_heading else len(rest))
        return text[match.start():end].strip() or text

    def _read_skill_tool(self) -> Tool:
        return Tool(
            name="read_skill",
            description=(
                f"读取技能 {self.name} 的说明书（{self.description}）。"
                "section 留空读全文，也可传章节标题只读某一节。"
            ),
            parameters=cast(
                ToolSchema,
                {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": "要读的章节标题；留空读全文",
                        }
                    },
                },
            ),
            func=self._read_skill,
        )

    def _list_resources_tool(self) -> Tool:
        return Tool(
            name="list_resources",
            description=(
                "列出技能包里的资源（参考文档/素材/模板/数据/脚本）路径。"
                "需要了解有哪些可用文件时先调它。"
            ),
            parameters=cast(
                ToolSchema,
                {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "reference",
                                "asset",
                                "template",
                                "resource",
                                "script",
                            ],
                            "description": "只列某一类；不传则全部列出",
                        }
                    },
                },
            ),
            func=self.list_resources,
        )

    def _read_resource_tool(self) -> Tool:
        return Tool(
            name="read_resource",
            description=(
                "读取技能包内的文本资源（参考文档、说明等）。路径相对技能根"
                "目录；图片、pptx、pdf 等二进制只返回元信息，需要处理时把"
                "路径传给脚本。"
            ),
            parameters=cast(
                ToolSchema,
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对技能根目录的路径，例如 references/style.md",
                        }
                    },
                    "required": ["path"],
                },
            ),
            func=self.read_resource,
        )

    def _list_scripts(self) -> str:
        items = [
            {"path": spec.name, "description": spec.description}
            for spec in self.scripts
        ]
        return json.dumps(items, ensure_ascii=False)

    def _list_scripts_tool(self) -> Tool:
        return Tool(
            name="list_scripts",
            description=(
                f"列出技能 {self.name} 可执行的脚本（相对 scripts/ 的路径 + "
                "一句话说明）。先看它再 run_script。"
            ),
            parameters=cast(ToolSchema, {"type": "object", "properties": {}}),
            func=self._list_scripts,
        )

    def _run_script_tool(self, interpreter: str, workdir: Path, limit: float) -> Tool:
        def call(name: str, args: str = "") -> str:
            return self.run_script(
                name, args, python=interpreter, cwd=workdir, timeout=limit
            )

        return Tool(
            name="run_script",
            description=(
                f"执行技能 {self.name} 的脚本（相对 scripts/ 的 .py 路径）。"
                f"Python 解释器是 {interpreter}，工作目录是 {workdir}。"
                "args 传命令行参数字符串；非零退出码不抛错，"
                "会返回 stdout / stderr / returncode。"
            ),
            parameters=cast(
                ToolSchema,
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "scripts/ 下的脚本相对路径，如 build.py 或 svg/convert.py",
                        },
                        "args": {
                            "type": "string",
                            "description": "命令行参数字符串，如 --out deck.pptx；无参数留空",
                        },
                    },
                    "required": ["name"],
                },
            ),
            func=call,
        )

    def _run_command_tool(
        self,
        interpreter: str,
        workdir: Path,
        limit: float,
        allow_commands: Optional[List[str]],
    ) -> Tool:
        '''`run_command`：默认关闭，启用后还必须过 allow_commands 白名单。'''
        base = self.command_tool(
            python=interpreter, timeout=max(600.0, limit), root=workdir
        )
        allowed = [name for name in map(_command_head, allow_commands or []) if name]
        allowed_text = "、".join(allowed) if allowed else "（未配置）"

        def call(command: str = "") -> str:
            text = str(command or "").strip()
            if not text:
                return "错误：command 不能为空"
            if not allowed:
                return (
                    "错误：run_command 已启用但没有配置 allow_commands，已拒绝执行。"
                    "请在 tools(enable={'run_command'}, allow_commands=['python']) "
                    "里显式声明允许的命令。"
                )
            head = _command_head(text)
            if head not in allowed:
                return (
                    f"错误：命令 {head!r} 不在 allow_commands 白名单里"
                    f"（允许：{allowed_text}）"
                )
            return str(base.func(text))

        return Tool(
            name="run_command",
            description=(
                f"在目录 {workdir} 里执行一条命令（工作目录已自动切换，"
                f"Python 解释器是 {interpreter}）。"
                f"只允许白名单里的命令：{allowed_text}，其他命令一律拒绝。"
                "写文件内容请用 write_file，不要用 echo / python -c 拼文件。"
            ),
            parameters=cast(
                ToolSchema,
                {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的完整命令行",
                        }
                    },
                    "required": ["command"],
                },
            ),
            func=call,
        )

    def _resolve_cwd(self, cwd: Optional[Union[str, Path]]) -> Path:
        '''解析唯一工作目录：`cwd` 显式给就用它，None 用包根下的 output/。'''
        if cwd is not None:
            target = Path(cwd).expanduser().resolve()
        else:
            target = (self.root / "output").resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SkillValidationError(f"无法创建工作目录 {target}：{exc}") from exc
        return target

    def _namespace_prefix(self) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", self.name).strip("_") + "__"

    def _default_tool_names(self) -> List[str]:
        '''按 §5.1 的生成规则给出规范名列表（顺序与 `tools()` 一致）。'''
        names = ["read_skill"]
        if self.has_resources:
            names.extend(["list_resources", "read_resource"])
        names.append("write_file")
        if self.has_scripts:
            names.extend(["list_scripts", "run_script"])
        if self.tool_whitelist:
            names = [name for name in names if name in self.tool_whitelist]
        return names

    def command_tool(
        self,
        *,
        name: str = "run_command",
        description: Optional[str] = None,
        python: Optional[str] = None,
        timeout: Optional[float] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> Tool:
        '''包一个「在指定目录里执行命令」的工具。

        大型技能的说明是按命令行写的（`python scripts/xxx.py --参数 ...`），
        把几百个脚本逐个挂成工具反而不现实。手写一个 run_command 包装要处理
        四件琐事，这里一次做完：

        - 工作目录固定到 `root`（默认技能目录），相对路径不用模型自己拼
        - 超时放宽（组版 / 转换脚本 30 秒不够），默认 600 秒且不低于 `timeout`
        - 命令开头的 `python` / `python3` / `py` 重写成 `python=` 给的解释器
          （绕开 Windows 上退出码 9009 的假指针）
        - 输出按 `max_output_chars` 截断，避免一条命令灌爆上下文

        参数：

        - `name` / `description`：工具名与描述，可按技能改写
        - `python`：重写 `python` 前缀用的解释器；None 表示自动探测
          （技能包 `.venv` → 当前解释器 → PATH）
        - `timeout`：单条命令的超时秒数
        - `root`：命令的工作目录。技能目录只读或想单独放产物时，传一个可写
          目录；脚本用绝对路径调用即可。None 表示用技能目录（`workdir`）

            # 读技能、写到项目目录，两边解耦
            pkg.command_tool(python=sys.executable, root="D:/projects/out")
        '''
        from promptkit.core.tool.builtin.run_command import (
            run_command as run_shell,
        )

        workdir = self._resolve_root(root)
        limit = max(
            0.1,
            float(timeout) if timeout is not None else max(600.0, self.timeout),
        )
        max_chars = self.max_output_chars
        interpreter = resolve_python(python, root=self.root)

        def rewrite_python(command: str) -> str:
            if not interpreter:
                return command
            text = command.strip()
            # python3 要排在 python 前面，否则 `python3 x.py` 会被误拆
            for alias in ("python3", "python", "py"):
                if text == alias or text.startswith(alias + " "):
                    return f'"{interpreter}"' + text[len(alias):]
            return text

        def enter_workdir(command: str) -> str:
            quoted = f'"{workdir}"'
            cd = f"cd /d {quoted}" if os.name == "nt" else f"cd {quoted}"
            return f"{cd} && {rewrite_python(command)}"

        def run(command: str) -> str:
            return _truncate(run_shell(enter_workdir(command), int(limit)), max_chars)

        shell_hint = (
            "当前是 Windows，用 cmd 语法（dir、type、%USERPROFILE%），"
            "不要用 ls / cat / ~"
            if os.name == "nt"
            else "当前是 POSIX shell，用 ls / cat / $HOME"
        )
        default_description = (
            f"在目录 {workdir} 里执行一条命令（已自动切换工作目录）。"
            f"{shell_hint}。单条命令最长 {int(limit)} 秒。"
            "只用来跑命令；写文件内容请用 write_file，"
            "不要在这里用 python -c / echo 拼文件内容（有命令行长度限制）。"
        )
        if interpreter:
            default_description += (
                f"命令开头的 python / python3 / py 会被换成 {interpreter}。"
            )
        return Tool.from_function(
            run,
            name=name,
            description=default_description if description is None else description,
        )

    def write_tool(
        self,
        *,
        name: str = "write_file",
        description: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> Tool:
        '''包一个「在指定目录里写文件」的工具（整文件覆盖，自动建目录）。

        不挂这个工具时，模型只能用 `run_command` 里的 `python -c "..."`
        一行行拼文件内容——Windows 命令行有 8191 字符上限，大文件必炸，
        转义也会烧掉大量轮次。SVG / 生成脚本这类大文件必须走这里。

        参数：

        - `name` / `description`：工具名与描述，可按技能改写
        - `root`：允许写入的根目录，默认技能目录；传别的目录即可把产物
          和只读的技能素材分开。越界路径仍会被拒绝

            tools = [*pkg.tools(), pkg.command_tool(root=out), pkg.write_tool(root=out)]
        '''
        from promptkit.core.tool.coding import WriteFileTool

        target = self._resolve_root(root)
        tool = WriteFileTool(str(target), allow_write=True)
        if name != "write_file":
            tool.name = name
        if description is not None:
            tool.description = description
        else:
            tool.description += (
                f"文件写到 {target} 下；"
                "大文件（SVG、脚本）一律用这个工具写，"
                "不要塞进 run_command 的命令行里。"
            )
        return tool

    def _resolve_root(self, root: Optional[Union[str, Path]]) -> Path:
        '''解析读写根目录；None 表示用技能目录，显式传入时会自动建好。'''
        if root is None:
            return self.workdir
        target = Path(root).expanduser().resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SkillValidationError(f"无法创建目录 {target}：{exc}") from exc
        return target

    def check_dependencies(self) -> Dict[str, bool]:
        '''逐个检查声明的依赖是否可导入。'''
        return {name: _is_installed(name) for name in self.dependencies}

    @property
    def instructions(self) -> str:
        '''完整入口说明（已追加资源清单）。

        占位符（`{{topic}}` 之类）保持原样；需要替换时用
        `pkg.skill.render(topic=...)`。
        '''
        return self.skill.instructions

    def routing_prompt(self) -> str:
        '''一段适合放进 system prompt 的短触发块。

        `ChatWorkflow(rules=[pkg.routing_prompt()], tools=pkg.tools())` 可以让
        「什么时候该用这个技能」始终可见，正文仍然按需加载，不必把整份
        `SKILL.md` / `AGENT.md` 塞进上下文。
        '''
        lines: List[str] = [f"## 可用技能：{self.skill.name}"]
        if self.skill.description:
            lines.append(self.skill.description)
        lines.append(
            "当用户任务与这个技能匹配时，先调用工具 `read_skill` "
            "读取完整说明，再按说明使用配套工具；不要凭空假设文件内容。"
        )
        helpers: List[str] = []
        if self._allows("list_resources"):
            helpers.append("`list_resources`")
        if self._allows("read_resource"):
            helpers.append("`read_resource`")
        if self.scripts:
            helpers.append("`run_script`")
        if helpers:
            lines.append("配套工具：" + "、".join(helpers) + "。")
        return "\n".join(lines)

    def list_resources(self, kind: str = "") -> str:
        '''列出资源；返回 JSON 文本（工具结果统一是字符串）。'''
        if not self._allows("list_resources"):
            return "错误：当前技能的 allowed-tools 没有放行列资源清单"
        if kind and kind not in (
            "reference",
            "asset",
            "template",
            "resource",
            "script",
        ):
            return (
                "错误：kind 只能是 reference / asset / template / resource / "
                f"script，得到 {kind!r}"
            )

        groups: Tuple[Tuple[str, Tuple[Path, ...]], ...] = (
            ("reference", self.references),
            ("asset", self.assets),
            ("template", self.templates),
            ("resource", self.resources),
            ("script", tuple(spec.path for spec in self.scripts)),
        )
        items: List[Dict[str, Any]] = []
        for category, paths in groups:
            if kind and category != kind:
                continue
            for path in paths[:_MAX_LISTED_RESOURCES]:
                items.append(
                    {
                        "path": path.relative_to(self.root).as_posix(),
                        "kind": category,
                        "size": _file_size(path),
                    }
                )
        if not items:
            return "技能包里没有可列出的资源。"
        return json.dumps(items, ensure_ascii=False)

    def read_resource(self, path: str, max_chars: Optional[int] = None) -> str:
        '''读取一个文本资源；路径越界或二进制都会给出明确提示。'''
        if not self._allows("read_resource"):
            return "错误：当前技能的 allowed-tools 没有放行资源读取"

        try:
            target = self._resolve(path)
        except SkillValidationError as exc:
            return f"错误：{exc}"

        relative = target.relative_to(self.root).as_posix()
        size = _file_size(target)
        if target.suffix.lower() not in _TEXT_SUFFIXES:
            return json.dumps(
                {
                    "path": relative,
                    "size": size,
                    "binary": True,
                    "hint": "这是二进制资源，请把相对路径传给脚本处理，不要直接读取",
                },
                ensure_ascii=False,
            )
        if size > _MAX_RESOURCE_BYTES:
            return (
                f"错误：{relative} 有 {size} 字节，超过单次读取上限 "
                f"{_MAX_RESOURCE_BYTES}；请改用脚本处理，或传更小的 max_chars 由脚本分片"
            )

        try:
            raw = target.read_bytes()
        except OSError as exc:
            return f"错误：读不了 {relative}：{exc}"

        text = _decode(raw)
        limit = self.max_resource_chars if max_chars is None else int(max_chars)
        limit = max(1, min(limit, self.max_resource_chars))
        if len(text) > limit:
            return (
                text[:limit]
                + f"\n…（已截断，共 {len(text)} 字符；需要更多请传 max_chars）"
            )
        return text

    def run_script(
        self,
        name: str,
        args: str = "",
        *,
        python: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        '''执行 scripts/ 下的一个脚本，返回 JSON（skill_spec v1 §3）。

        `name` 是相对 scripts/ 的 .py 路径（如 `build.py` 或 `svg/convert.py`），
        只接受登记在册的脚本；`args` 是命令行参数字符串。非零退出码不抛
        异常，统一返回 `{"returncode", "stdout", "stderr"}`。
        '''
        target = self._resolve_script(name)
        interpreter = resolve_python(python, root=self.root)
        workdir = self._resolve_cwd(cwd)
        limit = max(0.1, float(timeout) if timeout is not None else self.timeout)

        if isinstance(args, str):
            argv = shlex.split(args)
        else:
            argv = [str(item) for item in args]

        command = [interpreter, str(target), *argv]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            logger.info("skill script %s 超时（%.1fs）", name, limit)
            return json.dumps(
                {"returncode": None, "stdout": "", "stderr": f"超时（>{limit:g} 秒）"},
                ensure_ascii=False,
            )
        except OSError as exc:
            logger.info("skill script %s 启动失败：%s", name, exc)
            return json.dumps(
                {"returncode": None, "stdout": "", "stderr": f"无法启动脚本：{exc}"},
                ensure_ascii=False,
            )

        duration = time.perf_counter() - started
        logger.info(
            "skill script %s 完成：exit=%s duration=%.2fs",
            name,
            completed.returncode,
            duration,
        )
        return json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": _truncate(completed.stdout or "", self.max_output_chars),
                "stderr": _truncate(completed.stderr or "", self.max_output_chars),
            },
            ensure_ascii=False,
        )

    def _resolve_script(self, name: str) -> Path:
        '''把相对 scripts/ 的脚本名解析成绝对路径；越界/不存在都抛异常。'''
        text = str(name or "").strip()
        if not text:
            raise ScriptNotFoundError(
                "脚本名不能为空", hint="用 list_scripts 查可用脚本"
            )
        candidate = Path(text)
        if candidate.is_absolute() or bool(candidate.drive):
            raise PathEscapeError(f"脚本名必须是 scripts/ 下的相对路径：{name}")
        if candidate.suffix.lower() != ".py":
            raise ScriptNotFoundError(f"脚本必须是 .py：{name}")

        script_root = (self.root / "scripts").resolve()
        target = (script_root / candidate).resolve()
        if not target.is_relative_to(script_root):
            raise PathEscapeError(f"脚本路径越界：{name}")
        if not target.is_file():
            raise ScriptNotFoundError(
                f"脚本不存在：{name}", hint="用 list_scripts 查可用脚本"
            )
        registered = {spec.path for spec in self.scripts}
        if target not in registered:
            raise ScriptNotFoundError(
                f"脚本不在可执行清单里：{name}（可能是被过滤的库文件）"
            )
        return target

    def describe(self) -> str:
        '''一句话说明这个包暴露了什么，方便 CLI / 调试打印。'''
        tools = "、".join(tool.name for tool in self.tools())
        return (
            f"技能包 {self.skill.name}（{self.root}）："
            f"{len(self.references)} 份参考、{len(self.assets)} 个素材、"
            f"{len(self.scripts)} 个脚本；工具：{tools}"
        )

    # ---------------- 内部 ----------------
    def _allows(self, category: str) -> bool:
        '''运行时放行判断：`tools` 白名单优先，其次兼容旧的 allowed-tools。'''
        if category == "skill":
            category = "read_skill"
        if self.tool_whitelist:
            return category in self.tool_whitelist
        if category == "read_skill":
            return True
        return not self.allowed_tools or category in self.allowed_tools

    def _resolve(self, raw: str) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise SkillValidationError("path 不能为空")

        candidate = Path(text)
        if candidate.is_absolute():
            raise SkillValidationError("只能用相对技能根目录的路径，不接受绝对路径")

        root = self.root.resolve()
        target = (root / candidate).resolve()
        if target == root or not target.is_relative_to(root):
            raise SkillValidationError(f"路径越界：{raw}")
        if not target.is_file():
            raise SkillValidationError(f"资源不存在：{raw}")

        parts = target.relative_to(root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            raise SkillValidationError(f"不读隐藏文件 / 缓存目录：{raw}")
        return target

    def appendix(self) -> str:
        '''追加到 SKILL.md 末尾的资源清单与使用约定。'''
        lines: List[str] = [
            "",
            "---",
            "",
            "## 技能包资源（运行时自动附加）",
            "",
            f"- 根目录：`{self.root}`",
        ]
        self._append_files(lines, "参考文档", self.references, self.root)
        self._append_files(lines, "素材", self.assets, self.root)
        self._append_files(lines, "模板", self.templates, self.root)
        if self.scripts:
            lines.append("")
            lines.append("### 脚本")
            for spec in self.scripts[:_MAX_APPENDIX_ITEMS]:
                relative = spec.path.relative_to(self.root).as_posix()
                lines.append(f"- `{spec.name}` — {spec.description}（`{relative}`）")
            if len(self.scripts) > _MAX_APPENDIX_ITEMS:
                lines.append(f"- …共 {len(self.scripts)} 个脚本")

        if self.dependencies:
            lines.append("")
            lines.append("### 依赖状态")
            for name in self.dependencies:
                if name in self.missing_dependencies:
                    lines.append(
                        f"- `{name}`：**缺失**（请先 `pip install {name}`）"
                    )
                else:
                    lines.append(f"- `{name}`：已安装")
            if self.missing_dependencies:
                lines.append(
                    "缺少依赖时脚本可能直接失败；遇到这种情况先向用户报告需要安装的包。"
                )

        lines.append("")
        lines.append("### 使用约定")
        if self._allows("list_resources"):
            lines.append(
                "- 需要知道有哪些资料时调用 `list_resources`，可用 kind 过滤。"
            )
        if self._allows("read_resource"):
            lines.append(
                '- 读文本资料用 `read_resource(path="references/x.md")`，'
                "不要凭空臆测文件内容。"
            )
        lines.append(
            "- 图片、pptx、pdf 等二进制资源不要直接读，把相对路径传给脚本。"
        )
        if self.scripts:
            names = "、".join(f"`{spec.name}`" for spec in self.scripts[:5])
            lines.append(
                f"- 执行脚本先 `list_scripts` 查名字，再 `run_script(name=..., "
                f"args=...)` 传命令行参数；可用脚本：{names}"
            )
        return "\n".join(lines)

    @staticmethod
    def _append_files(
        lines: List[str],
        title: str,
        paths: Tuple[Path, ...],
        root: Path,
    ) -> None:
        if not paths:
            return
        lines.append("")
        lines.append(f"### {title}")
        for path in paths[:_MAX_APPENDIX_ITEMS]:
            lines.append(f"- `{path.relative_to(root).as_posix()}`")
        if len(paths) > _MAX_APPENDIX_ITEMS:
            lines.append(f"- …共 {len(paths)} 个文件")


# ============================ 加载 ============================
def as_skill_package(
    source: Any,
    *,
    name: Optional[str] = None,
    entry: Optional[str] = None,
    description: Optional[str] = None,
    tags: Sequence[str] = (),
    scripts: Optional[Any] = None,
    workdir: Optional[Any] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_output_chars: int = _MAX_OUTPUT_CHARS,
    max_resource_chars: int = _MAX_RESOURCE_CHARS,
    strict_dependencies: bool = False,
) -> SkillPackage:
    '''加载一个目录式技能包。

    参数：

    - `source`：技能目录（里面必须有 `SKILL.md`、`AGENT.md` 或对应小写版本）
    - `name` / `description` / `tags`：覆盖 SKILL.md 里的描述
    - `scripts`：覆盖 frontmatter 的 `scripts`；可传路径、路径列表或
      `{工具名: 路径}` / `{工具名: {path, description}}`
    - `workdir`：脚本的工作目录，默认技能根目录
    - `timeout` / `max_output_chars` / `max_resource_chars`：执行与读取上限
    - `strict_dependencies`：为 True 时，缺依赖直接抛
      `SkillDependencyError`；默认只在资源清单里标注

    脚本默认扫描 `scripts/**/*.py`（跳过 `__init__.py` 和下划线开头的文件），
    参数从 `argparse.add_argument` 推断；推断不出来时退化成
    `args: string[]` 通用工具。
    '''
    root = _resolve_root(source)
    entry_path = _entry_path(root) if entry is None else entry

    base = as_skill(
        entry_path,
        name=name,
        description=description,
        tags=tags,
    )
    frontmatter = base.frontmatter

    dependencies, missing = _dependency_state(
        frontmatter, base.name, strict_dependencies, root=root
    )
    # `allowed-tools` 是旧字段（resource 类别）；`tools` 是全量工具白名单。
    allowed = _parse_allowed_tools(
        _first_present(frontmatter, "allowed-tools", "allowed_tools")
    )
    whitelist = _parse_allowed_tools(frontmatter.get("tools"))
    entrypoint = _resolve_entrypoint(root, frontmatter.get("entrypoint"))

    raw_scripts = (
        scripts
        if scripts is not None
        else frontmatter.get("scripts")
    )
    script_specs = _build_scripts(root, raw_scripts)

    package = SkillPackage(
        root=root,
        skill=base,
        skill_md=base.body,
        scripts=script_specs,
        references=_discover(root, "references"),
        assets=_discover(root, "assets"),
        templates=_discover(root, "templates"),
        resources=_discover(root, "resources"),
        dependencies=dependencies,
        missing_dependencies=missing,
        allowed_tools=allowed,
        tool_whitelist=whitelist,
        entrypoint=entrypoint,
        workdir=_resolve_workdir(root, workdir),
        timeout=float(timeout),
        max_output_chars=int(max_output_chars),
        max_resource_chars=int(max_resource_chars),
    )

    # 说明正文末尾追加资源清单：模型先看到「有什么」，需要时再读。
    package.skill = _with_appendix(base, package.appendix())
    return package


def load_skills_from_dir(
    root: Union[str, Path],
    *,
    pattern: str = "*",
    recursive: bool = False,
) -> List[SkillPackage]:
    '''批量加载目录下的技能包（skill_spec §6）。

    只尝试看起来像技能包的子目录（含入口文件）；单个包校验失败只记
    warning，不中断其余包的加载。
    '''
    base = _resolve_root(root)
    candidates = base.rglob(pattern) if recursive else base.glob(pattern)
    packages: List[SkillPackage] = []
    for path in sorted(candidates):
        if not path.is_dir():
            continue
        if not any((path / name).is_file() for name in _ENTRY_NAMES):
            continue
        try:
            packages.append(SkillPackage.load(path))
        except SkillError as exc:
            logger.warning("跳过技能包 %s：%s", path, exc)
    return packages


def merge_tools(
    packages: Sequence[SkillPackage],
    *,
    python: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None,
    namespace: bool = True,
) -> List[Tool]:
    '''把多个技能包的工具合并成一个列表（skill_spec §9）。

    默认 `namespace=True`（工具名带 `{name}__` 前缀）；仍重名时抛
    `DuplicateToolError`，信息里写明是哪两个包冲突。
    '''
    merged: List[Tool] = []
    owners: Dict[str, str] = {}
    for package in packages:
        for tool in package.tools(
            python=python, cwd=cwd, timeout=timeout, namespace=namespace
        ):
            if tool.name in owners:
                raise DuplicateToolError(
                    f"工具名冲突：{tool.name!r} 同时来自 {owners[tool.name]!r} "
                    f"和 {package.name!r}",
                    hint="用 namespace=True 加包名前缀，或用 disable 去掉冲突项",
                )
            owners[tool.name] = package.name
            merged.append(tool)
    return merged


def _resolve_root(source: Any) -> Path:
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError(
            "技能包需要目录路径，"
            f"得到 {type(source).__name__}"
        )
    path = Path(source).expanduser()
    if not path.is_dir():
        raise InvalidSkillPackageError(f"技能包目录不存在：{path}")
    return path.resolve()


def _entry_path(root: Path) -> Path:
    for entry in _ENTRY_NAMES:
        candidate = root / entry
        if candidate.is_file():
            return candidate
    raise InvalidSkillPackageError(
        f"目录 {root} 里没有 {' / '.join(_ENTRY_NAMES)}，不是标准技能包"
    )


def _validate_package_frontmatter(
    frontmatter: Mapping[str, Any], entry: Path
) -> None:
    '''`SkillPackage.load()` 的严格 frontmatter 校验（skill_spec §2.1）。'''
    name = frontmatter.get("name")
    if name is None or not str(name).strip():
        raise InvalidSkillPackageError(
            f"{entry} 的 frontmatter 缺少 name 字段",
            hint="在 SKILL.md 顶部 frontmatter 写 name: <kebab-case 名称>",
        )
    name_text = str(name).strip()
    if not _PACKAGE_NAME_RE.match(name_text):
        suggestion = re.sub(r"[^a-z0-9]+", "-", name_text.lower()).strip("-")
        raise InvalidSkillPackageError(
            f"技能名 {name_text!r} 不符合 kebab-case：{entry}",
            hint=(
                "name 只用小写字母/数字，用 - 连接，例如 "
                f"{suggestion or 'my-skill'}"
            ),
        )

    description = frontmatter.get("description")
    if description is None or not str(description).strip():
        raise InvalidSkillPackageError(
            f"{entry} 的 frontmatter 缺少 description 字段",
            hint="写一行 description，说明技能做什么、什么时候用",
        )

    _check_python_version(frontmatter.get("python_version"), entry)


def _check_python_version(raw: Any, entry: Path) -> None:
    '''校验 frontmatter 里的 python_version（如 `>=3.10`）；无法解析只记 warning。'''
    if raw is None:
        return
    text = str(raw).strip()
    if not text:
        return
    match = _PYTHON_VERSION_RE.match(text)
    if match is None:
        logger.warning("python_version %r 无法解析，已忽略", text)
        return

    op = match.group("op") or "=="
    required = (
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
    )
    current = sys.version_info[:3]
    satisfied = {
        ">=": current >= required,
        "<=": current <= required,
        ">": current > required,
        "<": current < required,
        "==": current == required,
        "!=": current != required,
    }[op]
    if not satisfied:
        raise InvalidSkillPackageError(
            f"{entry.name} 要求 python_version {text}，"
            f"当前解释器是 {'.'.join(map(str, current))}",
            hint="换一个满足版本的解释器运行，或改掉 SKILL.md 里的 python_version",
        )


def _dependency_state(
    frontmatter: Mapping[str, Any],
    skill_name: str,
    strict: bool,
    root: Optional[Path] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    '''返回 (声明的依赖, 缺失的依赖)。

    依赖 = frontmatter 的 `dependencies` ∪ `requirements.txt` 里的包名
    （§2.1 取并集，去重保序）；严格模式下缺依赖直接抛。
    '''
    declared = _parse_dependencies(
        _first_present(frontmatter, "dependencies", "requirements", "requires")
    )
    if root is not None:
        merged = list(declared)
        for name in _requirements_names(root):
            if name not in merged:
                merged.append(name)
        declared = tuple(merged)

    missing = tuple(item for item in declared if not _is_installed(item))
    if strict and missing:
        command = " ".join(missing)
        raise SkillDependencyError(
            f"技能 {skill_name!r} 缺少依赖：{', '.join(missing)}；"
            f"请先 `pip install {command}`，或关闭 strict_dependencies"
        )
    return declared, missing


def _requirements_names(root: Path) -> Tuple[str, ...]:
    '''从 requirements.txt 提取包名；跳过注释、空行和 `-r` / `-e` 等选项行。'''
    path = root / "requirements.txt"
    if not path.is_file():
        return ()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ()

    names: List[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for name in _parse_dependencies(line):
            if name not in names:
                names.append(name)
    return tuple(names)


def _resolve_entrypoint(root: Path, raw: Any) -> str:
    '''校验 frontmatter `entrypoint`：只记录相对包根的路径，非法时 warning。

    规格里它「仅声明用途」，因此这里不因路径问题中断加载。
    '''
    text = str(raw or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive:
        logger.warning("entrypoint 必须是相对包根的路径，已忽略：%s", text)
        return ""
    target = (root / candidate).resolve()
    if not target.is_relative_to(root):
        logger.warning("entrypoint 越界，已忽略：%s", text)
        return ""
    if not target.is_file():
        logger.warning("entrypoint 指向的文件不存在，已忽略：%s", text)
        return ""
    return target.relative_to(root).as_posix()


def _resolve_workdir(root: Path, workdir: Any) -> Path:
    if workdir is None:
        return root
    working = Path(workdir).expanduser().resolve()
    working.mkdir(parents=True, exist_ok=True)
    return working


def _with_appendix(
    base: MarkdownSkillTool, appendix: str
) -> MarkdownSkillTool:
    return MarkdownSkillTool(
        name=base.name,
        description=base.description,
        parameters=dict(base.parameters),
        body=base.body.rstrip() + "\n" + appendix,
        source=base.source,
        frontmatter=base.frontmatter,
        version=base.version,
        tags=base.tags,
    )


def _first_present(frontmatter: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in frontmatter:
            return frontmatter[key]
    return None


# ============================ 资源 ============================
def _discover(root: Path, dirname: str) -> Tuple[Path, ...]:
    base = root / dirname
    if not base.is_dir():
        return ()
    paths = [
        path
        for path in base.rglob("*")
        if path.is_file() and not _is_hidden(path, root)
    ]
    return tuple(sorted(paths, key=lambda item: str(item).lower()))


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.startswith(".") or part == "__pycache__" for part in parts)


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _file_size(path: Path) -> int:
    '''文件可能刚被删掉；列表工具不该因此抛异常。'''
    try:
        return path.stat().st_size
    except OSError:
        return -1


# ============================ 依赖 ============================
def _parse_dependencies(raw: Any) -> Tuple[str, ...]:
    result: List[str] = []
    for item in _string_items(raw):
        name = re.split(r"[<>=!~\[; ]", item, maxsplit=1)[0].strip()
        if name and name not in result:
            result.append(name)
    return tuple(result)


def _is_installed(package: str) -> bool:
    key = package.strip().lower()
    # pip 包名常用 `-`，import 名常用 `_`；点号属于真实子模块，不能动。
    module = _IMPORT_ALIASES.get(key) or key.replace("-", "_")
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# ============================ allowed-tools ============================
def _parse_allowed_tools(raw: Any) -> Tuple[str, ...]:
    '''把 `allowed-tools` / `tools` 里的名字归一化成规范工具名。'''
    if raw is None:
        return ()
    result: List[str] = []
    for item in _string_items(raw):
        key = re.sub(r"[\s_.-]+", "", item.lower())
        category = _TOOL_ALIASES.get(key)
        if category is None:
            logger.warning("工具白名单里的 %r 没有对应工具，已忽略", item)
            continue
        if category not in result:
            result.append(category)
    if not result:
        # 明确声明了白名单，却没有任何能识别的工具：
        # 按「什么都不放行」处理，避免退化成默认全开。
        return ("__none__",)
    return tuple(result)


def _string_items(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = re.split(r"[,，]", raw)
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    raise SkillValidationError(
        f"期望字符串或字符串列表，得到 {type(raw).__name__}"
    )


# ============================ 脚本发现 ============================
def _build_scripts(root: Path, raw: Any) -> Tuple[ScriptSpec, ...]:
    entries: List[Tuple[str, Path, Optional[str]]] = []

    if raw is None:
        script_root = root / "scripts"
        if script_root.is_dir():
            for path in sorted(script_root.rglob("*.py")):
                subdirs = path.relative_to(script_root).parts[:-1]
                lowered = {part.lower() for part in subdirs}
                if (
                    path.is_file()
                    and path.name != "__init__.py"
                    and not path.name.startswith("_")
                    # 测试文件与常见非入口目录是库代码，不是命令行入口
                    and not path.name.startswith("test_")
                    and not path.name.endswith("_test.py")
                    and not lowered & _NON_ENTRY_DIRS
                    and not any(part.startswith("_") for part in subdirs)
                    and not _is_hidden(path, root)
                ):
                    local = Path(*subdirs, path.name).as_posix()
                    entries.append((local, path, None))
    elif isinstance(raw, (str, os.PathLike)):
        path = _script_path(root, raw)
        entries.append((_script_relative_name(root, path), path, None))
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            path = _script_path(root, item)
            entries.append((_script_relative_name(root, path), path, None))
    elif isinstance(raw, Mapping):
        for _key, value in raw.items():
            description: Optional[str] = None
            if isinstance(value, (str, os.PathLike)):
                path = _script_path(root, value)
            elif isinstance(value, Mapping):
                location = value.get("path") or value.get("file") or value.get("script")
                if not location:
                    raise SkillValidationError(
                        f"frontmatter scripts.{_key} 缺少 path"
                    )
                path = _script_path(root, location)
                if value.get("description"):
                    description = str(value["description"])
            else:
                raise SkillValidationError(
                    f"frontmatter scripts.{_key} 必须是路径或映射"
                )
            entries.append((_script_relative_name(root, path), path, description))
    else:
        raise SkillValidationError(
            "scripts 必须是路径、路径列表或 {名字: 路径} 映射，"
            f"得到 {type(raw).__name__}"
        )

    seen: Dict[str, Path] = {}
    specs: List[ScriptSpec] = []
    for local, path, description in entries:
        if local in seen and seen[local] != path:
            raise SkillValidationError(
                f"脚本重名：{local!r}（{seen[local]} 与 {path}）"
            )
        seen[local] = path
        specs.append(_parse_script(path, local, description))
    return tuple(specs)


def _script_relative_name(root: Path, path: Path) -> str:
    '''脚本在 scripts/ 下的相对路径（posix），如 build.py 或 svg/convert.py。'''
    try:
        return path.relative_to(root / "scripts").as_posix()
    except ValueError:
        return path.name


def _script_path(root: Path, raw: Any) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise SkillValidationError("脚本路径不能为空")
    candidate = Path(text)
    target = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if target == root or not target.is_relative_to(root):
        raise SkillValidationError(f"脚本路径越界：{raw}")
    if not target.is_file():
        raise SkillValidationError(f"脚本不存在：{raw}")
    if target.suffix.lower() != ".py":
        raise SkillValidationError(f"脚本必须是 .py：{raw}")
    if _is_hidden(target, root):
        raise SkillValidationError(f"不加载隐藏目录里的脚本：{raw}")
    return target


# ============================ 脚本解析 ============================
def _parse_script(
    path: Path, name: str, description: Optional[str] = None
) -> ScriptSpec:
    '''从脚本 docstring 首段提取描述；不解析 argparse（脚本按需用
    `run_script(name, args)` 调用，不再逐个推断参数 Schema）。'''
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        docstring = ast.get_docstring(tree) or ""
    except (OSError, SyntaxError, UnicodeDecodeError):
        docstring = ""

    script_description = (
        description
        or _first_paragraph(docstring)
        or f"运行技能脚本 {path.name}"
    )
    return ScriptSpec(name=name, path=path, description=script_description)


def _first_paragraph(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped:
            return stripped
    return ""


# ============================ 输出 ============================
def _command_head(command: str) -> str:
    '''取命令首词并归一化（去引号 / 路径 / .exe），用于白名单比较。'''
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        parts = text.split()
    if not parts:
        return ""
    head = parts[0].strip("\"'")
    if "/" in head or "\\" in head:
        head = Path(head).name
    if head.lower().endswith(".exe"):
        head = head[:-4]
    return head


def _truncate(text: str, limit: int) -> str:
    limit = max(1, int(limit))
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（输出已截断，共 {len(text)} 字符）"
