'''让**模型自己决定**什么时候把成果写进文件。

为什么是一个 tool 而不是一个 Step：

- `Step`（`Save` / `SaveDeliverables`）是**你**在写工作流时决定「第 3 步存这里」
- `tool` 是**模型**在跑的时候决定「用户要一份选题建议，我应该落盘」

后者才是「在合适的地方自动生成文件」。模型判断「这轮该产出文件了」，
就调 `write_file`，框架负责安全地写下去。

    from promptkit.default_tools import WriteFile
    from promptkit.core import Tools

    writer = WriteFile("outputs")          # 只允许写在 outputs/ 下面
    wf = Workflow(
        Tools(writer),                     # 直接传实例，自动变成工具
        SystemPrompt(),
        ChatLoop(),
    )
    wf.run(agent=agent)
    print(writer.written)                  # ['选题建议.md', ...]

⚠️ 这是**由大模型驱动的文件写入**，所以默认带沙箱：
限制根目录、拒绝 `..` 和绝对路径、扩展名白名单、大小上限。
不要为了省事把 `sandbox=False` 打开。
'''
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from promptkit.tools import Tool

__all__ = [
    "WriteFile",
    "WRITE_FILE",
    "TOOL_NAME",
    "TOOL_DESCRIPTION",
    "DEFAULT_ALLOWED_EXTENSIONS",
    "MAX_CONTENT_CHARS",
]

#: 工具名（模型看到的名字）
TOOL_NAME = "write_file"

#: 工具描述（模型看到的说明）。写清楚「什么时候该调」比写清楚参数更重要 ——
#: 模型是照着这句话决定要不要落盘的。
TOOL_DESCRIPTION = (
    "把内容保存成一个文件。当用户需要一份可以带走的成果"
    "（报告、Markdown 文档、表格、JSON 数据、代码文件等）时调用它。"
    "name 是相对路径（如 '选题建议.md' 或 'data/资料.json'），"
    "content 是文件的完整内容；同一路径重复写会覆盖。"
)

#: 允许写出来的扩展名。想放宽就自己传 `allow=`。
DEFAULT_ALLOWED_EXTENSIONS: Tuple[str, ...] = (
    "md", "txt", "json", "jsonl", "csv", "tsv", "yaml", "yml",
    "html", "xml", "py", "js", "css", "sql", "sh", "log", "tex", "mmd",
)

#: 单次写入的字符上限（防止模型一次吐几百 KB 把磁盘塞满）
MAX_CONTENT_CHARS = 200_000

#: 文件名里不允许出现的字符（Windows 上 `:` `*` `?` 之类直接非法）
_BAD_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

#: 整个内容被一层代码围栏包住时，把那层剥掉。
#: 模型很爱把文件内容写成 ```markdown ... ```，直接存会带上围栏。
_WHOLE_FENCE = re.compile(r"^\s*```[\w+-]*[ \t]*\r?\n(.*?)\r?\n?```\s*$", re.S)


def _strip_fence(text: str) -> str:
    '''把「整个内容就是一层代码围栏」的情况剥掉那层围栏。'''
    match = _WHOLE_FENCE.match(text)
    return match.group(1) if match else text


class WriteFile:
    '''把内容写进文件——**由模型决定何时调用**。

    用法上它就是一个可调用对象，`Tools(writer)` / `LLMStep(tools=[writer])`
    都能直接吃（`as_tool` 认 `callable`，参数从 `__call__` 的签名和 docstring 推）。

    参数
    ----
    - `root`：沙箱根目录。所有文件都必须落在这下面，父目录自动创建。
    - `allow`：扩展名白名单（小写，不带点）。`None` 表示用默认那一串。
    - `overwrite`：同名文件要不要覆盖。`False` 时返回错误让模型换个名字。
    - `max_chars`：单次写入的字符上限。
    - `encoding`：默认 UTF-8（和项目其它文件 IO 一致）。
    - `validate_json`：`.json` 结尾的内容必须是合法 JSON，否则返回错误让模型改。
    - `index_file`：在根目录维护一份 `_index.json` 清单，记录写了哪些文件。
    - `sandbox`：关掉就**不做**根目录/路径校验 —— 强烈不建议。
    '''

    def __init__(
        self,
        root: str = "outputs",
        *,
        allow: Optional[Sequence[str]] = None,
        overwrite: bool = True,
        max_chars: int = MAX_CONTENT_CHARS,
        encoding: Optional[str] = None,
        validate_json: bool = True,
        index_file: Optional[str] = "_index.json",
        sandbox: bool = True,
    ):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.allow = tuple(
            item.lower().lstrip(".")
            for item in (DEFAULT_ALLOWED_EXTENSIONS if allow is None else allow)
        )
        self.overwrite = overwrite
        self.max_chars = max_chars
        self.encoding = encoding
        self.validate_json = validate_json
        self.index_file = index_file
        self.sandbox = sandbox
        #: 这个实例写过哪些文件（相对 `root`，按写入顺序，去重）
        self.written: List[str] = []

        # `as_tool(writer)` 走的是 `Tool.from_function`，它从对象上读
        # `__name__` / `__doc__`。实例属性会盖住类属性，所以在这里按「工具」
        # 而不是「类」来定义名字和描述 —— 否则模型看到的名字是 `tool`，
        # 描述是类文档（讲实现而不是讲什么时候该调）。
        self.__name__ = TOOL_NAME
        self.__doc__ = TOOL_DESCRIPTION

    # ---------------- 模型看到的那个函数 ----------------
    def __call__(self, name: str, content: str) -> str:
        '''保存文件。'''  # 模型看到的是 TOOL_DESCRIPTION，不是这里
        return self.write(name, content)

    # ---------------- 真正的实现 ----------------
    def write(self, name: str, content: str) -> str:
        '''写文件（模型调用走 `__call__`，两者等价）。返回给模型看的一句话。'''
        text = "" if content is None else str(content)
        text = _strip_fence(text)

        target, problem = self._resolve(name)
        if problem:
            return problem

        extension = os.path.splitext(target)[1].lower().lstrip(".")
        if self.allow and extension not in self.allow:
            return (
                f"错误：不允许写 .{extension or '（无扩展名）'} 文件。"
                f"允许的扩展名：{', '.join(self.allow)}"
            )
        if len(text) > self.max_chars:
            return (
                f"错误：内容太长（{len(text)} 字符，上限 {self.max_chars}）。"
                "请分成多个文件写。"
            )
        if extension == "json" and self.validate_json:
            checked = self._check_json(text)
            if checked:
                return checked
        if not self.overwrite and os.path.exists(target):
            return f"错误：{name!r} 已存在，换一个文件名。"

        payload = text if text.endswith("\n") else text + "\n"
        try:
            folder = os.path.dirname(target)
            if folder:
                os.makedirs(folder, exist_ok=True)
            # newline="\n"：写 LF，别让 Windows 翻成 CRLF（和项目其它文件 IO 一致）
            with open(
                target,
                "w",
                encoding=self.encoding or "utf-8",
                newline="\n",
            ) as handle:
                handle.write(payload)
        except OSError as exc:
            return f"错误：写入失败：{exc}"

        relative = os.path.relpath(target, self.root).replace(os.sep, "/")
        if relative not in self.written:
            self.written.append(relative)
        self._update_index(relative, len(payload))

        return (
            f"已写入 {relative}（{len(payload)} 字符，共 {len(self.written)} 个文件）"
        )

    # ---------------- 路径安全 ----------------
    def _resolve(self, name: str) -> Tuple[str, Optional[str]]:
        '''把模型给的名字解析成绝对路径；不安全就返回 `(路径, 错误信息)`。'''
        raw = str(name or "").strip().strip('"').strip("'")
        if not raw:
            return "", "错误：文件名不能为空。"

        if not self.sandbox:
            return os.path.abspath(raw), None

        # 先判绝对路径：`C:\x` 里那个 `:` 其实也会被下面的字符检查拦住，
        # 但那样报的是「有不允许的字符」，对模型来说毫无指导意义。
        #
        # 注意不能只信 `os.path.isabs`：Windows 上 `isabs('/etc/passwd')`
        # 居然是 False（没盘符就不算绝对），所以再显式挡一道前导分隔符。
        if (
            os.path.isabs(raw)
            or os.path.splitdrive(raw)[0]
            or raw[:1] in ("/", "\\")
        ):
            return "", f"错误：请用相对路径（比如 '报告.md'），不要用绝对路径：{raw!r}"

        if _BAD_NAME_CHARS.search(raw):
            return "", f"错误：文件名里有不允许的字符：{raw!r}"

        # 逐段检查 `..`，别让 `a/../../etc/passwd` 逃出去
        parts = raw.replace("\\", "/").split("/")
        if any(part in ("..",) for part in parts):
            return "", f"错误：路径里不能有 '..'：{raw!r}"

        target = os.path.normpath(os.path.join(self.root, raw))
        # 再确认一次：解析完仍在 root 下面（符号链接之类也能挡住）
        if os.path.commonpath([self.root, target]) != self.root:
            return "", f"错误：只能写到 {os.path.basename(self.root)}/ 目录下面。"
        if target == self.root:
            return "", "错误：文件名不能为空。"
        return target, None

    def _check_json(self, text: str) -> Optional[str]:
        '''`.json` 文件必须是合法 JSON —— 让模型当场改，比事后排查强。'''
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return (
                f"错误：内容不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列："
                f"{exc.msg}）。请修正后重试。"
            )
        return None

    def _update_index(self, relative: str, size: int) -> None:
        '''维护一份 `_index.json` 清单，方便事后查「这次生成了哪些文件」。'''
        if not self.index_file:
            return
        path = os.path.join(self.root, self.index_file)
        try:
            data: Dict[str, Any] = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    data = loaded
            files = data.get("files")
            if not isinstance(files, list):
                files = []
            files = [item for item in files if item.get("path") != relative]
            files.append({"path": relative, "chars": size})
            data["files"] = files
            data["count"] = len(files)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except (OSError, ValueError):
            # 清单只是方便，写不进去不该影响真正的交付
            pass

    # ---------------- 其它 ----------------
    def as_tool(self) -> Tool:
        '''显式拿一个 `Tool`（等价于 `as_tool(writer)`）。'''
        return Tool.from_function(
            self, name=TOOL_NAME, description=TOOL_DESCRIPTION
        )

    def paths(self, absolute: bool = False) -> List[str]:
        '''写过哪些文件；`absolute=True` 时给绝对路径。'''
        if not absolute:
            return list(self.written)
        return [os.path.join(self.root, item.replace("/", os.sep)) for item in self.written]

    def __repr__(self) -> str:
        return f"<WriteFile root={self.root!r} written={len(self.written)}>"


#: 开箱即用的共享实例（和 `RUN_COMMAND` / `WEB_SEARCH` 一个风格）。
#: ⚠️ 它是**共享**的，`written` 会跨次累积；需要干净的清单就自己
#: `WriteFile("outputs")` 建一个。
WRITE_FILE = WriteFile()
