# core/steps/actions/interactive.py
'''跟外部打交道的「输入 / 输出」步骤。

- `Read` / `ReadFile`：读标准输入 / 读文件（都是**惰性表达式**）
- `Echo`：打印
- `Save`：把值写进文件
- `SaveHistory`：把对话历史 + 步骤轨迹存成 JSON
'''
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 model_action
    from promptkit.core.context import Context
from promptkit.core.settings import settings
from promptkit.core.steps.interface import Step
from promptkit.core.steps.logics.expr_sys.expr import Expr, as_expr

# 结果输出流挂在 Context 上的键名（Workflow.run(out=...) 会写进来）
OUTPUT_STREAM_KEY = "_output_stream"

#: 当前显示分级挂在 Context 上的键名（Workflow.run(verbose=...) 会写进来）
VERBOSITY_KEY = "_verbosity"


def verbosity_of(context: Context) -> Any:
    '''当前这次运行的显示分级（默认 NORMAL）。'''
    from promptkit.core.display_sys.render import as_verbosity

    return as_verbosity(context.get(VERBOSITY_KEY))

#: 文本文件默认编码。真正用的是 `settings.encoding`，这个常量只是兜底默认值。
#: 不指定编码的话 Python 会用系统编码（Windows 上是 GBK），
#: 中文以外的东西（emoji、日文…）直接抛 UnicodeEncodeError，
#: 而且写出来的文件换台机器就读不了。
DEFAULT_ENCODING = "utf-8"


def current_encoding(explicit: Optional[str] = None) -> str:
    '''解析实际要用的编码：显式值优先，否则取全局设置。'''
    return settings.encoding if explicit is None else explicit


def output_stream(context: Context) -> Any:
    '''「结果」往哪儿写：默认 stdout，可以用 Workflow.run(out=...) 换掉。'''
    return context.get(OUTPUT_STREAM_KEY) or sys.stdout


def _prepare_path(path: str) -> str:
    '''把文件路径规整好，顺手把缺的目录建出来。'''
    full = os.path.abspath(os.path.expanduser(str(path)))
    folder = os.path.dirname(full)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return full


def _write_text(path: str, text: str, *, append: bool, encoding: str) -> str:
    '''写文本，返回真正写入的绝对路径。

    `newline="\\n"` 是故意的：Python 默认会把 `\\n` 翻译成平台换行符
    （Windows 上变成 `\\r\\n`），那样同一个工作流在两台机器上写出的字节就不一样，
    拿去做 diff / 打包 / 喂给别的工具都会出岔子。这里统一写 LF。
    '''
    full = _prepare_path(path)
    with open(
        full, "a" if append else "w", encoding=encoding, newline="\n"
    ) as handle:
        handle.write(text)
    return full


# =============================== 读 ===============================
class ReadExpr(Expr):
    '''从标准输入读一行。惰性求值，所以构造工作流时不会阻塞等输入。

        Set("job", Read("请输入面试的岗位："))

    `on_eof` 决定「输入结束」时怎么办：

    - `"raise"`（默认）：原样抛 `EOFError` / `KeyboardInterrupt`
    - `"break"`：当成「对话到此为止」，跳出当前循环 —— 于是循环**后面**的
      步骤（比如 `SaveDeliverables()`）还能正常执行

    Ctrl+D（EOF）和 Ctrl+C 在提示符处基本都是「我不想聊了」，
    默认行为会让整个工作流抛异常退出、后面的收尾步骤全部跳过 ——
    实测就是「聊完了，但交付物没写出来」。聊天循环默认用 `"break"`。
    '''

    def __init__(self, prompt: str = "", strip: bool = True, on_eof: str = "raise"):
        if on_eof not in ("raise", "break"):
            raise ValueError(f"on_eof 只能是 'raise' 或 'break'，得到 {on_eof!r}")
        self.prompt = prompt
        self.strip = strip
        self.on_eof = on_eof

    def eval(self, context: Context) -> str:
        try:
            text = input(self.prompt)
        except (EOFError, KeyboardInterrupt):
            if self.on_eof != "break":
                raise
            # 先换个行，免得 `^C` 和后面的输出粘在同一行
            try:
                sys.stderr.write("\n")
                sys.stderr.flush()
            except Exception:  # noqa: BLE001 - 流被换掉也不该影响收尾
                pass
            from promptkit.core.errors import BreakSignal

            raise BreakSignal() from None
        return text.strip() if self.strip else text

    def __repr__(self) -> str:
        if self.on_eof != "raise":
            return f"Read({self.prompt!r}, on_eof={self.on_eof!r})"
        return f"Read({self.prompt!r})"


class ReadFileExpr(Expr):
    '''读取**整个文件**。惰性求值，所以构造工作流时不会真的去读。

        Set("配置", ReadFile("config.json"))
        Set("文档", Read("docs/intro.md", from_file=True))   # 等价写法

    `strip=True`（默认）会把首尾空白去掉；想原样保留就传 `strip=False`。
    文件读不到会抛 `FileNotFoundError`（不会悄悄返回空串）。
    '''

    def __init__(
        self, file: str, strip: bool = True, encoding: Optional[str] = None
    ):
        self.file = file
        self.strip = strip
        self.encoding = encoding

    def eval(self, context: Context) -> str:
        try:
            with open(self.file, encoding=current_encoding(self.encoding)) as handle:
                text = handle.read()
        except OSError as exc:
            raise FileNotFoundError(
                f"读不到文件 {self.file!r}：{exc}"
            ) from None
        return text.strip() if self.strip else text

    def __repr__(self) -> str:
        return f"ReadFile({self.file!r})"


# =============================== 打印 ===============================
@dataclass
class EchoStep(Step):
    '''打印若干表达式 / 字面量。

    `render` 决定怎么显示：

    - `"text"`（默认）：原样打印
    - `"markdown"`：交给渲染器排版（装了 rich 就是富文本，没装就去掉标记）

    两条路径都会把 `$...$` / `$$...$$` 里的 LaTeX 转成 Unicode 数学（零依赖）。
    `title` 会在上面加一条分隔线。
    '''

    parts: List[Expr] = field(default_factory=list)
    sep: str = ""
    render_method: str = "text"
    title: Optional[str] = None
    name: str = "echo"

    def execute(self, context: Context) -> None:
        from promptkit.core.display_sys.render import Verbosity

        if verbosity_of(context) < Verbosity.NORMAL:
            context.trace.append(self.name)
            return

        text = self.sep.join(str(part.eval(context)) for part in self.parts)
        stream = output_stream(context)
        if self.render_method in ("text", "plain"):
            from promptkit.core.display_sys.render import render_plain

            render_plain(text, stream=stream, title=self.title)
        else:
            from promptkit.core.display_sys.render import RENDERERS, render_markdown

            renderer = RENDERERS.get(self.render_method, render_markdown)
            renderer(text, stream=stream, title=self.title)
        context.trace.append(self.name)

    def describe(self) -> str:
        parts = ", ".join(repr(part) for part in self.parts)
        if self.render_method not in ("text", "plain"):
            return f"Echo({parts}, render={self.render_method!r})"
        return f"Echo({parts})"


# =============================== 写 ===============================
@dataclass
class SaveStep(Step):
    '''把若干值写进文件。

        Save("结果：", Var("n"), file="out.txt")
        Save(Var("报告"), file="report.md", append=True)
        Save(Var("info"), file="info.json", format="json")

    几个约定：

    - `format="text"`（默认）一行一个值，会被 `str()` 转一遍
    - `format="json"` 只接受**一个**值，用 `json.dumps` 写（dict / Pydantic 都行）
    - 写的是 **UTF-8**
    - 父目录不存在会**自动创建**
    - 文本模式末尾补一个换行（除非本来就是空字符串）
    '''

    values: List[Expr] = field(default_factory=list)
    file: str = "output.txt"
    append: bool = False
    format: str = "text"
    encoding: Optional[str] = None
    name: str = "save"

    def __post_init__(self) -> None:
        if self.format not in ("text", "json"):
            raise ValueError(f"format 只能是 'text' 或 'json'，得到 {self.format!r}")
        if self.format == "json" and len(self.values) != 1:
            raise ValueError(
                f"format='json' 只能存一个值，收到 {len(self.values)} 个"
            )

    def _as_text(self, context: Context) -> str:
        if self.format == "json":
            import json

            payload = _plain(self.values[0].eval(context))
            return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        text = "\n".join(str(value.eval(context)) for value in self.values)
        return text + "\n" if text else ""

    def execute(self, context: Context) -> None:
        text = self._as_text(context)
        _write_text(
            self.file,
            text,
            append=self.append,
            encoding=current_encoding(self.encoding),
        )
        context.trace.append(f"{self.name}:{self.file}")

    def describe(self) -> str:
        parts = ", ".join(repr(value) for value in self.values)
        mode = " >>" if self.append else ""
        return f"Save({parts}){mode} -> {self.file!r}"


#: 这些键是运行时内部状态，不该写进历史文件
_INTERNAL_KEYS = frozenset({OUTPUT_STREAM_KEY, "_event_bus", "agent"})


def _plain(value: Any, depth: int = 0) -> Any:
    '''把任意值转成能 `json.dump` 的东西；转不了就用 `repr` 兜底。'''
    if depth > 12:  # 环 / 太深的结构，别再往下钻
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _plain(item, depth + 1)
            for key, item in value.items()
            if key not in _INTERNAL_KEYS
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain(item, depth + 1) for item in value]

    # Pydantic 模型
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return _plain(dumper(), depth + 1)
        except Exception:  # noqa: BLE001 - 兜底到 repr
            pass
    return repr(value)


@dataclass
class SaveHistoryStep(Step):
    '''把对话历史 + 步骤轨迹存成 JSON。

    存的是三样东西：

    - `messages`：对话消息（`context["messages"]`）
    - `trace`：步骤执行轨迹
    - `variables`：上下文里的变量（`include_variables=True` 才有）

    内部的运行时对象（`agent` / 事件总线 / 输出流）会被跳过；
    Pydantic 对象会转成 dict；实在转不了的用 `repr` 兜底 ——
    所以**什么上下文都不会导致写不出来**。
    '''

    file: str = "history.json"
    include_variables: bool = False
    encoding: Optional[str] = None
    indent: Optional[int] = 2
    name: str = "save_history"

    def execute(self, context: Context) -> None:
        payload: dict = {
            "trace": _plain(list(context.trace)),
            "messages": _plain(context.get("messages", [])),
        }
        if self.include_variables:
            payload["variables"] = _plain(context.to_dict())

        _write_text(
            self.file,
            json.dumps(payload, ensure_ascii=False, indent=self.indent) + "\n",
            append=False,
            encoding=current_encoding(self.encoding),
        )
        context.trace.append(f"{self.name}:{self.file}")

    def describe(self) -> str:
        extra = " + 变量" if self.include_variables else ""
        return f"SaveHistory({self.file!r}){extra}"


# =============================== 工厂 ===============================
def Read(  # noqa: N802
    prompt: str = "",
    *,
    strip: bool = True,
    from_file: bool = False,
    encoding: Optional[str] = None,
    on_eof: str = "raise",
) -> Union[ReadExpr, ReadFileExpr]:
    '''读输入，当表达式用。

        Set("job", Read("岗位："))                    # 读一行标准输入
        Set("文本", Read("a.txt", from_file=True))     # 读整个文件
        Read(">>> ", on_eof="break")                  # Ctrl+D / Ctrl+C 跳出循环

    `from_file=True` 时第一个参数是**文件路径**（不是提示语）。
    嫌这个写法别扭就直接用 `ReadFile("a.txt")`。
    '''
    if from_file:
        return ReadFileExpr(prompt, strip, encoding)
    return ReadExpr(prompt, strip, on_eof)


def ReadFile(  # noqa: N802
    path: str, *, strip: bool = True, encoding: Optional[str] = None
) -> ReadFileExpr:
    '''读整个文件，当表达式用：`Set("配置", ReadFile("config.json"))`。'''
    return ReadFileExpr(path, strip, encoding)


def Save(  # noqa: N802
    *values: Any,
    file: str = "output.txt",
    append: bool = False,
    format: str = "text",
    encoding: Optional[str] = None,
) -> SaveStep:
    '''把值写进文件。

        Save("结果：", Var("n"), file="out.txt")
        Save(Var("报告"), file="report.md", append=True)      # 追加
        Save(Var("info"), file="info.json", format="json")    # 结构化
    '''
    return SaveStep(
        values=[as_expr(value) for value in values],
        file=file,
        append=append,
        format=format,
        encoding=encoding,
    )


def SaveJSON(value: Any, file: str = "output.json", **kwargs: Any) -> SaveStep:  # noqa: N802
    '''把**一个**值写成 JSON：`SaveJSON(Var("info"), "info.json")`。

    等价于 `Save(value, file=file, format="json")`，只是名字更直白。
    '''
    return Save(value, file=file, format="json", **kwargs)


def SaveHistory(  # noqa: N802
    file: str = "history.json",
    *,
    include_variables: bool = False,
    encoding: Optional[str] = None,
    indent: Optional[int] = 2,
) -> SaveHistoryStep:
    '''把对话历史和步骤轨迹存成 JSON。

        SaveHistory("run1.json")
        SaveHistory("run1.json", include_variables=True)
    '''
    return SaveHistoryStep(
        file=file,
        include_variables=include_variables,
        encoding=encoding,
        indent=indent,
    )


def Echo(  # noqa: N802
    *parts: Any, sep: str = "", render: str = "text", title: Optional[str] = None
) -> EchoStep:
    '''打印：`Echo("面试官：", Var("question"))`。

    字符串里的 `{{name}}` 会自动换成 context 里的变量：

        Echo("情感分数：{{score}}")      # 等价于 Echo("情感分数：", Var("score"))

    渲染 Markdown / 数学公式：

        Echo(Var("model_reply"), render="markdown")
        Echo(Var("model_reply"), render="markdown", title="选题建议")
    '''
    from promptkit.core.steps.logics.expr_sys.expr import TemplateExpr

    return EchoStep(
        parts=[TemplateExpr(p) if isinstance(p, str) else as_expr(p) for p in parts],
        sep=sep,
        render_method=render,
        title=title,
    )


__all__ = [
    "DEFAULT_ENCODING",
    "OUTPUT_STREAM_KEY",
    "Echo",
    "EchoStep",
    "Read",
    "ReadExpr",
    "ReadFile",
    "ReadFileExpr",
    "Save",
    "SaveHistory",
    "SaveHistoryStep",
    "SaveStep",
    "output_stream",
]
