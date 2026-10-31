# skill/markdown.py
'''从外部 Markdown 文件加载技能（不改代码就能扩展能力）。

一个 `.md` 技能就是「一段说明 + 一份参数表」。模型调用它时拿到的是
**整理好的操作说明**，然后按说明继续干活 —— 不会为此再启动一次模型调用。

    from promptkit.skill import as_skill, as_skills

    Tools(as_skill("skills/weekly-report.md"))                  # 手写 Workflow
    ChatWorkflow(skills=[as_skill("skills/weekly-report.md")])  # 声明式

    Tools(*as_skills("skills/"))                                # 一次装一个目录

文件格式（frontmatter 可选）：

    ---
    name: weekly_report
    description: 把零散的工作记录整理成结构化周报
    parameters:
      period:
        type: string
        description: 覆盖的时间范围，如 2026-W43
    ---
    # 周报规范
    ……
    请按 {{period}} 汇总。

没有 frontmatter 时：名字取文件名，描述取第一个标题 / 段落，参数为空。
只解析**简单 YAML**（嵌套映射 / 标量 / 标量列表）；复杂结构请显式传参。
'''
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from promptkit.skill.errors import SkillValidationError
from promptkit.core.tool import Tool
from promptkit.core.tool.base import ToolSchema

__all__ = ["MarkdownSkillTool", "as_skill", "as_skills"]

#: 函数名规则：和 `SkillSpec` / OpenAI 工具名保持一致
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
#: 正文里的参数占位符：`{{period}}` / `{{ period }}`
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")
#: 名字里不能出现的字符
_BAD_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

#: 目录模式优先找这些文件名
_ENTRY_NAMES = ("SKILL.md", "skill.md", "Skill.md")


class MarkdownSkillTool(Tool):
    '''一个 Markdown 技能的 Tool 视图。

    对模型来说它就是普通工具：有名字、描述、参数 Schema。
    调用后返回技能正文（`{{参数}}` 已被替换），模型接着按正文做事。
    '''

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        body: str,
        source: Path,
        frontmatter: Optional[Mapping[str, Any]] = None,
        version: str = "0.1",
        tags: Sequence[str] = (),
    ):
        self.body = body
        self.source = Path(source)
        self.frontmatter: Dict[str, Any] = dict(frontmatter or {})
        self.version = str(version)
        self.tags: Tuple[str, ...] = tuple(str(tag) for tag in (tags or ()))
        super().__init__(
            name=name,
            description=description,
            parameters=cast(ToolSchema, parameters),
            func=self.run,
        )

    @property
    def instructions(self) -> str:
        '''技能原文（未替换占位符）。想直接拼进提示词时用它。'''
        return self.body

    def render(self, **arguments: Any) -> str:
        '''把 `{{参数}}` 替换成实参；没传的占位符原样保留。'''
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in arguments:
                return match.group(0)
            value = arguments[key]
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False, default=str)
            return str(value)

        return _PLACEHOLDER.sub(replace, self.body)

    def run(self, **arguments: Any) -> str:
        '''工具调用：返回渲染后的技能正文。'''
        text = self.render(**arguments)
        if not text.strip():
            return f"错误：技能 {self.name} 的内容为空"
        return text

    def __repr__(self) -> str:
        return f"<MarkdownSkillTool {self.name!r} source={str(self.source)!r}>"


def as_skill(
    source: Any,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    tags: Sequence[str] = (),
) -> MarkdownSkillTool:
    '''读取一个 `.md` 技能文件，返回可直接放进 `Tools(...)` / `skills=[...]` 的工具。

    参数都是可选的，用来覆盖文件里写的内容：

        WEEKLY = as_skill("skills/weekly-report.md")
        Tools(WEEKLY)
        ChatWorkflow(skills=[WEEKLY])

    `source` 可以是：
    - 一个 `.md` 文件
    - 一个目录（找 `SKILL.md`；没有且只有一个 `.md` 时就用它）

    文件在**加载时读一次**；改完 `.md` 重新构造即可，不用改 Python 代码。
    '''
    path = _resolve_path(source)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SkillValidationError(f"读不了技能文件 {path}：{exc}") from None

    front, body = _split_frontmatter(text, path)
    if not body.strip():
        raise SkillValidationError(f"技能文件 {path} 只有 frontmatter，没有正文")

    skill_name = _clean_name(name or front.get("name") or path.stem, path)
    skill_description = (
        description
        or str(front.get("description") or "").strip()
        or _first_paragraph(body)
        or f"调用 {skill_name}"
    )
    schema = _build_parameters(
        parameters if parameters is not None else front.get("parameters"), body, path
    )

    return MarkdownSkillTool(
        name=skill_name,
        description=skill_description,
        parameters=schema,
        body=body.strip(),
        source=path,
        frontmatter=front,
        version=str(version or front.get("version") or "0.1"),
        tags=tags or _as_tags(front.get("tags")),
    )


def as_skills(source: Any, *, recursive: bool = False) -> List[MarkdownSkillTool]:
    '''一次加载多个 `.md` 技能。

        Tools(*as_skills("skills/"))                 # 目录下所有 .md
        Tools(*as_skills("skills/", recursive=True)) # 含子目录

    `source` 也可以是单个文件或一串路径。目录里一个 `.md` 都没有时明确报错，
    免得「装了个空目录还以为生效了」。
    '''
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        if path.is_dir():
            pattern = "**/*.md" if recursive else "*.md"
            files = sorted(
                (item for item in path.glob(pattern) if item.is_file()),
                key=lambda item: str(item).lower(),
            )
            if not files:
                raise SkillValidationError(f"目录 {path} 里没有 .md 技能文件")
            return [as_skill(item) for item in files]
        return [as_skill(path)]

    if isinstance(source, Iterable):
        return [as_skill(item) for item in source]

    raise TypeError(
        "as_skills 需要路径、目录或路径列表，"
        f"得到 {type(source).__name__}"
    )


# ============================ 路径与名字 ============================
def _resolve_path(source: Any) -> Path:
    '''文件原样返回；目录则找入口文件。'''
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError(
            "as_skill 需要 .md 文件路径或目录，"
            f"得到 {type(source).__name__}"
        )

    path = Path(source).expanduser()
    if path.is_dir():
        for entry in _ENTRY_NAMES:
            candidate = path / entry
            if candidate.is_file():
                return candidate
        markdown_files = sorted(item for item in path.glob("*.md") if item.is_file())
        if len(markdown_files) == 1:
            return markdown_files[0]
        if not markdown_files:
            raise SkillValidationError(
                f"目录 {path} 里没有 {_ENTRY_NAMES[0]} 或任何 .md 文件"
            )
        names = "、".join(item.name for item in markdown_files[:5])
        raise SkillValidationError(
            f"目录 {path} 里有多个 .md（{names}…），"
            "请直接指定文件，或改用 as_skills()"
        )
    if not path.is_file():
        raise SkillValidationError(f"技能文件不存在：{path}")
    if path.suffix.lower() != ".md":
        raise SkillValidationError(f"技能文件必须是 .md：{path}")
    return path


def _clean_name(raw: Any, path: Path) -> str:
    '''把文件名 / frontmatter 里的名字变成合法的工具名。'''
    text = str(raw or "").strip()
    if _NAME.match(text):
        return text

    cleaned = _BAD_NAME_CHARS.sub("_", text).strip("_.-")[:64]
    if cleaned and not _NAME.match(cleaned):
        cleaned = f"skill_{cleaned}"
    if not _NAME.match(cleaned or ""):
        raise SkillValidationError(
            f"技能名 {text!r} 不能用作工具名（需要字母/下划线开头，"
            f"只能用字母/数字/下划线/点/横线）：{path}；"
            "请在 frontmatter 写 name，或把文件改成英文名"
        )
    return cleaned


def _first_paragraph(body: str) -> str:
    '''没有 description 时：第一个标题 / 第一段正文。'''
    for line in body.splitlines():
        text = line.strip()
        if not text or text.startswith("<!--"):
            continue
        if text.startswith("#"):
            title = text.lstrip("#").strip()
            if title:
                return title
            continue
        if text.startswith("```"):
            continue
        return text
    return ""


# ============================ frontmatter ============================
def _split_frontmatter(text: str, path: Path) -> Tuple[Dict[str, Any], str]:
    '''拆出 `---` 包住的 frontmatter 和正文。'''
    stripped = text.lstrip("\ufeff")
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, stripped

    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            raw = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1:])
            return _parse_frontmatter(raw, path), body

    raise SkillValidationError(f"技能文件 {path} 的 frontmatter 没有结束的 ---")


def _parse_frontmatter(raw: str, path: Path) -> Dict[str, Any]:
    '''只解析简单 YAML：嵌套映射、标量、标量列表。

    项目不把 pyyaml 当必需依赖，所以这里手写一个小解析器；遇到
    不支持的结构就明确报错，而不是悄悄解析错。
    '''
    entries: List[Tuple[int, str]] = []
    for line in raw.expandtabs(2).splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        entries.append((len(line) - len(line.lstrip(" ")), text))
    if not entries:
        return {}

    value, index = _parse_block(entries, 0, entries[0][0], path)
    if index != len(entries):
        raise SkillValidationError(
            f"技能文件 {path} 的 frontmatter 缩进不合法：{entries[index][1]!r}"
        )
    if not isinstance(value, dict):
        raise SkillValidationError(f"技能文件 {path} 的 frontmatter 必须是映射")
    return value


def _parse_block(
    entries: List[Tuple[int, str]], index: int, indent: int, path: Path
) -> Tuple[Any, int]:
    if entries[index][1].startswith("- "):
        return _parse_list(entries, index, indent, path)
    return _parse_map(entries, index, indent, path)


def _parse_map(
    entries: List[Tuple[int, str]], index: int, indent: int, path: Path
) -> Tuple[Dict[str, Any], int]:
    result: Dict[str, Any] = {}
    while index < len(entries):
        current_indent, text = entries[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise SkillValidationError(
                f"技能文件 {path} 的 frontmatter 缩进不合法：{text!r}"
            )

        key, separator, rest = text.partition(":")
        if not separator or not key.strip():
            raise SkillValidationError(
                f"技能文件 {path} 的 frontmatter 不是 key: value：{text!r}"
            )
        key = key.strip()
        rest = rest.strip()
        index += 1

        if rest:
            result[key] = _parse_scalar(rest)
            continue
        if index < len(entries) and entries[index][0] > current_indent:
            child, index = _parse_block(entries, index, entries[index][0], path)
            result[key] = child
        else:
            result[key] = {}
    return result, index


def _parse_list(
    entries: List[Tuple[int, str]], index: int, indent: int, path: Path
) -> Tuple[List[Any], int]:
    result: List[Any] = []
    while index < len(entries):
        current_indent, text = entries[index]
        if current_indent < indent or not text.startswith("- "):
            break
        if current_indent > indent:
            raise SkillValidationError(
                f"技能文件 {path} 的 frontmatter 缩进不合法：{text!r}"
            )
        item = text[2:].strip()
        if ":" in item and not item.startswith(("'", '"')):
            raise SkillValidationError(
                f"技能文件 {path} 的 frontmatter 暂不支持「列表里嵌套映射」："
                f"{text!r}；请改用标量列表，或在代码里直接传 parameters"
            )
        result.append(_parse_scalar(item))
        index += 1
    return result, index


def _parse_scalar(text: str) -> Any:
    '''标量：引号字符串、布尔、null、数字、`[a, b]` 列表、单行 JSON。'''
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_scalar(part) for part in inner.split(",")] if inner else []
    if value.startswith("{") and value.endswith("}"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


# ============================ 参数 ============================
def _build_parameters(raw: Any, body: str, path: Path) -> Dict[str, Any]:
    '''把 frontmatter 的 `parameters` 变成 JSON Schema。

    两种写法都认：
    - 简写：`period: {type: string, description: ...}` → 自动包成 object
    - 完整：`type: object` + `properties` / `required`
    '''
    if raw is None:
        return {"type": "object", "properties": {}}
    if not isinstance(raw, dict):
        raise SkillValidationError(
            f"技能文件 {path} 的 parameters 必须是映射，得到 {type(raw).__name__}"
        )

    if raw.get("type") == "object":
        schema = dict(raw)
        schema.setdefault("properties", {})
        return schema

    properties: Dict[str, Any] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(spec, dict):
            item = {
                name: spec[name]
                for name in ("type", "description", "enum", "default")
                if name in spec
            }
            item.setdefault("type", "string")
        elif isinstance(spec, str):
            item = {"type": "string", "description": spec}
        else:
            item = {"type": "string"}
        properties[key] = item

    schema = {"type": "object", "properties": properties}
    required = [key for key in properties if _uses(body, key)]
    if required:
        schema["required"] = required
    return schema


def _uses(body: str, key: str) -> bool:
    return re.search(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", body) is not None


def _as_tags(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, Iterable):
        return tuple(str(item) for item in raw if str(item).strip())
    return (str(raw),)
