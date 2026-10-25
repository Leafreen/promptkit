# coding/formats.py
'''三种「可执行的模型输出」：diff / full_file / json。

和对话任务最大的区别：这里解析出来的东西**马上会被程序拿去执行**。
所以每种格式都带两层校验：

1. **结构**：diff 能不能解析、hunk 行数对不对；json 有没有 path / action
2. **可执行性**：补丁能不能真的 apply 到目标文件（给了 `sandbox` + `path` 就干跑一次）；
   Python 代码能不能 `ast.parse` 通过

校验失败抛 `CodeFormatError` —— `CodingStep` 会把它塞回提示词重试。
'''
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from promptkit.coding.diff import apply_hunks, strip_code_fence, validate_patch
from promptkit.coding.errors import CodeFormatError
from promptkit.core.model_action.caution_related.formatter import get_formatter

__all__ = [
    "CodeOutput",
    "DiffOutput",
    "FullFileOutput",
    "ChangesOutput",
    "get_output",
    "validate_changes",
    "OUTPUT_FORMATS",
]

#: 支持的动作（`changes` 格式）
ACTIONS = frozenset({"create", "modify", "delete"})


class CodeOutput(ABC):
    '''一种代码输出格式：规定怎么说 + 怎么解析校验。'''

    #: 格式名（`CodingStep(output_format=...)` 用）
    name: str = "code"

    @abstractmethod
    def instruction(self) -> str:
        '''塞进提示词里的「输出格式要求」。'''

    @abstractmethod
    def parse(
        self,
        raw: str,
        *,
        sandbox: Any = None,
        path: Optional[str] = None,
        language: str = "python",
    ) -> Any:
        '''解析 + 校验；不通过抛 `CodeFormatError`。'''

    def describe(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class DiffOutput(CodeOutput):
    '''统一 diff：改现有文件最省 token，但模型最容易写错格式。

    给了 `sandbox` + `path` 时会**干跑**一次（读文件 + 应用 hunk，不写回），
    所以在真正落盘之前就能发现「上下文对不上」。
    '''

    name = "diff"

    def instruction(self) -> str:
        return (
            "只输出一个 unified diff（不要输出完整文件、不要解释、不要 Markdown 代码块）。\n"
            "格式要求：\n"
            "1. 每个文件以 `--- 相对路径` 和 `+++ 相对路径` 两行开头\n"
            "2. 后接 `@@ -旧起始行,旧行数 +新起始行,新行数 @@`\n"
            "3. hunk 内每行以空格（上下文）、`+`（新增）、`-`（删除）开头\n"
            "4. 旧文件不存在（新建文件）时写 `--- /dev/null`\n"
            "5. 行数必须和 @@ 头里写的一致，否则补丁不合法"
        )

    def parse(
        self,
        raw: str,
        *,
        sandbox: Any = None,
        path: Optional[str] = None,
        language: str = "python",
    ) -> str:
        text = strip_code_fence(raw)
        # 给了 path 就说明目标文件由调用方指定，不校验补丁头里的路径
        patches = validate_patch(text, check_paths=path is None)
        if sandbox is None or path is None:
            return text

        hunks = [hunk for patch in patches for hunk in patch.hunks]
        creation = any(patch.is_creation for patch in patches)
        try:
            source = sandbox.read_text(path)
        except Exception:
            if not creation:
                raise CodeFormatError(
                    f"目标文件 {path!r} 读不到，而补丁也不是「新建文件」"
                    "（新建要在补丁头里写 `--- /dev/null`）"
                ) from None
            source = ""
        try:
            apply_hunks(source, hunks, path=path)
        except Exception as exc:  # DiffError
            raise CodeFormatError(
                f"补丁应用不到 {path!r}（干跑失败）：{exc}"
            ) from None
        return text


class FullFileOutput(CodeOutput):
    '''完整文件：新建 / 整文件重写时最稳，代价是大文件费 token。

    `language="python"` 时会做语法检查（`ast.parse`），其它语言可以传
    `syntax_check=callable(text)`（比如 `lambda t: subprocess.run(["node","--check",...])`）。
    '''

    name = "full_file"

    def __init__(
        self,
        language: Optional[str] = None,
        *,
        syntax_check: Optional[Callable[[str], None]] = None,
    ):
        #: None = 「用调用方（`CodingStep.language`）说的语言」，默认当 Python
        self.language = language
        self.syntax_check = syntax_check

    def instruction(self) -> str:
        return (
            "只输出**完整的文件内容**（不是 diff），第一个字符就是文件内容，"
            "不要解释、不要 Markdown 代码块、不要```。"
        )

    def parse(
        self,
        raw: str,
        *,
        sandbox: Any = None,
        path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        code = strip_code_fence(raw)
        if not code.strip():
            raise CodeFormatError("输出是空的，看起来没有给出代码")

        target_language = (self.language or language or "python").lower()
        if self.syntax_check is not None:
            try:
                self.syntax_check(code)
            except Exception as exc:
                raise CodeFormatError(f"语法检查没过：{exc}") from None
        elif target_language.startswith("python") or target_language in ("py", ""):
            try:
                ast.parse(code, filename=path or "<生成的代码>")
            except SyntaxError as exc:
                raise CodeFormatError(
                    f"Python 语法错误（第 {exc.lineno} 行）：{exc.msg}\n"
                    f"  {(exc.text or '').rstrip()}"
                ) from None

        return code if code.endswith("\n") else code + "\n"


class ChangesOutput(CodeOutput):
    '''多文件变更的 JSON：结构化、可逐条校验，适合重构。

        {"changes": [
            {"path": "src/a.py", "action": "modify", "content": "..."},
            {"path": "src/b.py", "action": "create", "content": "..."}
        ]}
    '''

    name = "json"

    def instruction(self) -> str:
        return (
            "只输出一个 JSON 对象（不要解释、不要 Markdown 代码块）：\n"
            '{"changes": [{"path": "相对路径", "action": "create|modify|delete",'
            ' "content": "完整文件内容"}]}\n'
            "规则：\n"
            "1. `path` 是相对路径，不要 `..`、不要绝对路径\n"
            "2. `action` 只能是 create / modify / delete\n"
            "3. create / modify 必须给 `content`（完整文件内容，不是 diff）\n"
            "4. delete 不需要 `content`\n"
            "5. 一个文件一条，不要重复"
        )

    def parse(
        self,
        raw: str,
        *,
        sandbox: Any = None,
        path: Optional[str] = None,
        language: str = "python",
    ) -> Dict[str, Any]:
        try:
            data = get_formatter(None).parse(raw)
        except Exception as exc:  # FormatterError
            raise CodeFormatError(f"不是合法的 JSON：{exc}") from None

        if isinstance(data, list):
            data = {"changes": data}
        if not isinstance(data, dict) or not isinstance(data.get("changes"), list):
            raise CodeFormatError(
                'JSON 顶层必须是 {"changes": [...]}（或直接是一个 changes 数组）'
            )
        return {"changes": validate_changes(data["changes"], sandbox=sandbox)}


def validate_changes(
    changes: Sequence[Any], *, sandbox: Any = None
) -> List[Dict[str, Any]]:
    '''校验多文件变更列表（`ChangesOutput` 和 `ApplyChanges` 共用）。'''
    result: List[Dict[str, Any]] = []
    seen: set = set()
    for index, item in enumerate(changes or (), start=1):
        if not isinstance(item, dict):
            raise CodeFormatError(f"第 {index} 条 change 不是对象：{item!r}")
        target = item.get("path")
        if not isinstance(target, str) or not target.strip():
            raise CodeFormatError(f"第 {index} 条 change 缺少 path")
        target = target.strip()
        action = str(item.get("action") or "modify").strip().lower()
        if action not in ACTIONS:
            raise CodeFormatError(
                f"第 {index} 条 change 的 action={action!r} 不合法，"
                f"只能是 {' / '.join(sorted(ACTIONS))}"
            )
        if target in seen:
            raise CodeFormatError(f"文件 {target!r} 出现了两次")
        seen.add(target)

        content = item.get("content")
        if action == "delete":
            if content not in (None, ""):
                raise CodeFormatError(f"{target!r} 是 delete，不该带 content")
            result.append({"path": target, "action": action})
            continue
        if not isinstance(content, str) or not content.strip():
            raise CodeFormatError(
                f"{target!r} 是 {action}，必须给 content（完整文件内容）"
            )
        if sandbox is not None and action == "modify" and not sandbox.exists(target):
            raise CodeFormatError(
                f"{target!r} 标记为 modify，但文件不存在；新建请用 action=create"
            )
        result.append({"path": target, "action": action, "content": content})

    if not result:
        raise CodeFormatError("changes 是空的，等于什么都没改")
    return result


#: 格式名 -> 实现类
OUTPUT_FORMATS: Dict[str, type] = {
    "diff": DiffOutput,
    "patch": DiffOutput,
    "full_file": FullFileOutput,
    "full": FullFileOutput,
    "file": FullFileOutput,
    "json": ChangesOutput,
    "changes": ChangesOutput,
}


def get_output(name: str = "diff", **kwargs: Any) -> CodeOutput:
    '''按名字取一个输出格式；名字不认识时给出可用列表。'''
    key = str(name or "").strip().lower().replace("-", "_")
    factory = OUTPUT_FORMATS.get(key)
    if factory is None:
        raise ValueError(
            f"不认识的代码输出格式 {name!r}；可用：{', '.join(sorted(OUTPUT_FORMATS))}"
        )
    return factory(**kwargs)
