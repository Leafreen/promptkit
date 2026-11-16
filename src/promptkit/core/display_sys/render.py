# core/display_sys/render.py
'''终端渲染：Markdown、数学公式、分级显示。

外部依赖**全可选**，没有也能跑：

| 装了什么 | 得到什么 | 没装会怎样 |
| --- | --- | --- |
| `rich` | Markdown 排版、代码高亮、表格 | 退回纯文本（原文照打） |
| `TeXicode`（`txc`） | 复杂公式更完整的 Unicode art | 用内置的 Unicode 映射 |

    pip install "promptkit[render]"

核心库不 print；这一层只负责「把文本变成终端能看的样子」，
往哪儿写由调用方给（默认 stdout，也可以是文件 / StringIO）。
'''
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unicodedata
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from promptkit.core.settings import settings

__all__ = [
    "Verbosity",
    "as_verbosity",
    "HAVE_RICH",
    "HAVE_TEXICODE",
    "get_console",
    "latex_to_unicode",
    "texicode_render",
    "RENDERERS",
    "render_math",
    "render_markdown",
    "render_plain",
    "plain_text",
    "strip_markdown",
    "substitute_math",
    "looks_like_math",
    "needs_art",
    "install_hint",
]


# ============================ 显示分级 ============================
class Verbosity(IntEnum):
    '''终端显示到什么程度。

    比 `verbose=True/False` 多两档：能分清「要结果」和「要排查」。

        wf.run(verbose=Verbosity.NORMAL)    # 清爽：只有对话
        wf.run(verbose=Verbosity.DEBUG)     # 排查：连原始事件都打
    '''

    #: 什么都不打（跑批处理，结果从 `ctx` 里取）
    SILENT = 0
    #: 默认：只打结果 —— `Echo`、模型的回答
    NORMAL = 1
    #: 再加步骤轨迹、工具调用、思考过程
    VERBOSE = 2
    #: 再加原始事件数据（参数、token 用量…）
    DEBUG = 3


def as_verbosity(value: Any) -> Verbosity:
    '''把各种写法统一成 `Verbosity`。

    兼容老的 `verbose=True/False`：`True` -> VERBOSE，`False`/`None` -> NORMAL。
    '''
    if value is None:
        return Verbosity.NORMAL
    if isinstance(value, Verbosity):
        return value
    if isinstance(value, bool):
        return Verbosity.VERBOSE if value else Verbosity.NORMAL
    if isinstance(value, int):
        try:
            return Verbosity(value)
        except ValueError:
            raise ValueError(
                f"verbose 只能是 0~{int(Verbosity.DEBUG)} 或 Verbosity 成员，"
                f"得到 {value!r}"
            ) from None
    raise TypeError(f"verbose 只能是 bool / int / Verbosity，得到 {value!r}")


# ============================ 可选依赖 ============================
def _try_import_rich() -> Tuple[Any, Any]:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        return None, None
    return Console, Markdown


_Console, _Markdown = _try_import_rich()
HAVE_RICH = _Console is not None

#: 按输出流缓存 Console（同一个流别反复创建）
_CONSOLES: Dict[int, Any] = {}


def install_hint(what: str = "render") -> str:
    '''缺少可选依赖时给用户的提示。'''
    return f'需要额外依赖：pip install "promptkit[{what}]"'


def get_console(stream: Any = None) -> Any:
    '''拿一个（缓存的）rich Console；没装 rich 返回 None。

    **不要加 `soft_wrap=True`**：它的意思是「不自动换行，交给终端」，
    而在代码块里 rich 会因此把超宽的行**裁掉**而不是折行 ——
    一段长公式后半截会凭空消失（实测 `-(√(5)+1)/(20)ln(x²+β x+1)`
    整个 β 项被吃掉）。宁可折行难看，也不能丢内容。
    '''
    if not HAVE_RICH:
        return None
    stream = sys.stdout if stream is None else stream
    key = id(stream)
    console = _CONSOLES.get(key)
    if console is None:
        console = _Console(file=stream, highlight=False)
        _CONSOLES[key] = console
    return console


# ============================ LaTeX -> Unicode ============================
#: 常用符号的直译（不依赖任何库）
_SYMBOLS = {
    # 希腊字母
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # 运算
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝",
    # 大算符
    "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "oint": "∮",
    "partial": "∂", "nabla": "∇", "infty": "∞", "sqrt": "√",
    # 集合 / 逻辑
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "forall": "∀", "exists": "∃",
    "neg": "¬", "land": "∧", "lor": "∨",
    # 箭头
    "to": "→", "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓",
    # 其它
    "dots": "…", "ldots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "angle": "∠", "perp": "⊥", "parallel": "∥", "degree": "°",
    "circ": "∘", "star": "⋆", "bullet": "•",
    "prime": "′", "therefore": "∴", "because": "∵",
    # 证明结束符
    "blacksquare": "■", "square": "□", "qed": "∎", "checkmark": "✓",
    # 逻辑 / 蕴含
    "implies": "⟹", "Longleftrightarrow": "⟺", "Longleftarrow": "⟸",
    "Longrightarrow": "⟹", "iff": "⟺", "land": "∧", "lor": "∨", "lnot": "¬",
    # 括号 / 定界符
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "langle": "⟨", "rangle": "⟩", "lvert": "|", "rvert": "|",
    "vert": "|", "mid": "|", "Vert": "‖",
    # 微积分 / 线性代数
    "lim": "lim", "log": "log", "ln": "ln", "exp": "exp", "det": "det",
    "max": "max", "min": "min", "sup": "sup", "inf": "inf",
}

_SUPERSCRIPT = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
    ")": "⁾", "n": "ⁿ", "i": "ⁱ",
}
_SUBSCRIPT = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
    ")": "₎", "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "n": "ₙ",
    "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ",
    "x": "ₓ",
}

_IGNORED_SYMBOLS = ("\\,", "\\;", "\\!", "\\:", "\\ ", "~")

_GROUP = r"\{([^{}]*)\}"


#: 常见重音：\hat{y} -> ŷ。用组合字符，靠终端自己叠。
_ACCENTS = {
    "hat": "\u0302", "bar": "\u0304", "vec": "\u20d7", "tilde": "\u0303",
    "dot": "\u0307", "ddot": "\u0308", "overline": "\u0304", "widehat": "\u0302",
}


def _apply_accents(text: str) -> str:
    pattern = re.compile(r"\\(hat|bar|vec|tilde|dot|ddot|overline|widehat)" + _GROUP)
    for _ in range(4):  # \hat{\beta}_1 这种嵌套，转几轮就干净了
        new = pattern.sub(lambda m: m.group(2) + _ACCENTS[m.group(1)], text)
        if new == text:
            break
        text = new
    return text


def latex_to_unicode(latex: str) -> str:
    '''把 LaTeX 翻译成可读的 Unicode 文本。**零依赖**。

    走的是「分词 -> 递归渲染」，不是一堆积木式正则。这点很关键：
    正则只能一个个补丁（`\\frac{a}{b}` 能用、`\\frac15` 就漏），
    而 TeX 的参数本来就可以不写花括号、可以任意嵌套。

    覆盖 LLM 常见写法的子集：

    - 结构：`\\frac` `\\binom` `\\sqrt` `\\hat` 之类，参数支持 `{...}` 和单字符
    - 上下标：`x^2` `x^{n+1}` `x_i` `x_{ij}`
    - 环境：`\\begin{cases}...\\end{cases}`（`&` 丢、`\\\\` 换行）
    - 换行：`\\\\` 和 `\\\\[2mm]`（`[2mm]` 里的间距丢掉）
    - 函数名：`\\ln` `\\arctan` `\\lim` … 原样输出名字
    - 符号：希腊字母 / 运算 / 定界符，查表转 Unicode

    **认不出的命令不会漏出反斜杠** —— 去掉 `\\` 只留名字，读起来仍然像文字
    （`\\arctan` -> `arctan`，`\\weirdcmd` -> `weirdcmd`）。比留一堆 `\\foo`
    有用得多，也比「静默当成变量名」安全。
    '''
    tokens = _tokenize(latex or "")
    text, _ = _render_sequence(tokens, 0)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return unicodedata.normalize("NFC", text).strip()


# ============================ 分词 ============================
#: `(类型, 值)`。类型：`cmd` 命令名、`char` 普通字符、`sym` 反斜杠转义的单字符、
#: `{` `}` `[` `]` `^` `_` `&` 结构符号。
_Token = Tuple[str, str]

#: 反斜杠后跟这些字符时是「转义的单个符号」而不是命令
_ESCAPED_CHARS = frozenset("\\{}[]&%$#_^ ,;:!")

#: 这些符号转义后要还原成字面字符
_LITERAL_CHARS = frozenset("{}[]&%$#_^")


def _tokenize(latex: str) -> List[_Token]:
    '''把 LaTeX 切成 token。反斜杠 + 字母 = 命令，反斜杠 + 符号 = 字面符号。'''
    tokens: List[_Token] = []
    index = 0
    total = len(latex)
    while index < total:
        char = latex[index]
        if char == "\\":
            after = index + 1
            if after < total and latex[after].isalpha():
                end = after
                while end < total and latex[end].isalpha():
                    end += 1
                tokens.append(("cmd", latex[after:end]))
                index = end
            elif after < total:
                tokens.append(("sym", latex[after]))
                index = after + 1
            else:
                index = after
        elif char in "{}[]^_&":
            tokens.append((char, char))
            index += 1
        elif char.isspace():
            end = index
            while end < total and latex[end].isspace():
                end += 1
            tokens.append(("char", " "))
            index = end
        else:
            tokens.append(("char", char))
            index += 1
    return tokens


# ============================ 渲染 ============================
#: `\frac` 一族 -> `(a)/(b)`
_FRACTIONS = frozenset({"frac", "dfrac", "tfrac", "cfrac"})
#: `\binom` 一族 -> `C(a, b)`
_BINOMIALS = frozenset({"binom", "dbinom", "tbinom"})
#: 去掉命令只看内容
_TRANSPARENT = frozenset(
    {"boxed", "fbox", "text", "mathrm", "operatorname", "mbox", "textbf",
     "textit", "textsf", "textrm", "textnormal", "mathnormal", "boldsymbol",
     "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "ensuremath"}
)
#: 只影响排版、直接丢掉
_IGNORED = frozenset(
    {"left", "right", "middle", "displaystyle", "textstyle", "scriptstyle",
     "limits", "nolimits", "big", "Big", "bigg", "Bigg", "bigl", "bigr",
     "Bigl", "Bigr", "biggl", "biggr", "mathstrut", "phantom", "hphantom",
     "vphantom", "nonumber", "notag", "label", "tag", "quad", "qquad",
     "enspace", "thinspace", "medspace", "thickspace", "nobreakspace"}
)
#: 这些命令后面紧跟一个「被转义的符号」，那个符号要原样输出
_DELIMITER_COMMANDS = frozenset({"left", "right", "middle", "big", "Big",
                                 "bigg", "Bigg", "bigl", "bigr", "Bigl",
                                 "Bigr"})

#: 重音命令 -> 组合字符
_ACCENT_MARKS = {
    "hat": "\u0302", "widehat": "\u0302", "bar": "\u0304", "overline": "\u0304",
    "vec": "\u20d7", "tilde": "\u0303", "widetilde": "\u0303",
    "dot": "\u0307", "ddot": "\u0308",
}

#: 函数名：原样输出（`\arctan` -> `arctan`，不改写成 `atan`）
_FUNCTIONS = frozenset(
    {"ln", "log", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc",
     "arcsin", "arccos", "arctan", "arccot", "sinh", "cosh", "tanh", "coth",
     "lim", "max", "min", "sup", "inf", "det", "dim", "ker", "deg", "gcd",
     "arg", "Pr", "mod", "bmod"}
)

#: 多行环境：行与行之间换真换行
_MULTILINE_ENVS = frozenset(
    {"cases", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
     "aligned", "align", "align*", "gathered", "gather", "gather*", "split",
     "array", "eqnarray", "eqnarray*", "subarray", "smallmatrix"}
)


def _as_script(body: str, table: Dict[str, str]) -> Optional[str]:
    '''把上下标内容转成 Unicode；有转不了的字符就返回 None。'''
    out = []
    for char in body.strip():
        mapped = table.get(char)
        if mapped is None:
            return None
        out.append(mapped)
    return "".join(out) or None


def _script(prefix: str, body: str, table: Dict[str, str]) -> str:
    '''上下标：能全转就用 Unicode，转不了就保留 `^`/`_` 加括号。'''
    if not body:
        return ""
    mapped = _as_script(body, table)
    if mapped is not None:
        return mapped
    # 转不了就老实写成 ^(...) / _(...) —— 宁可难看，也别改变含义
    single = body if len(body) == 1 else f"({body})"
    return prefix + single


def _skip_optional(tokens: List[_Token], pos: int) -> int:
    '''跳过 `[...]`（`\\\\[2mm]`、`\\sqrt[3]` 的可选参数）。返回新位置。'''
    if pos >= len(tokens) or tokens[pos][0] != "[":
        return pos
    depth = 0
    while pos < len(tokens):
        kind = tokens[pos][0]
        if kind == "[":
            depth += 1
        elif kind == "]":
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    return pos


def _read_optional(tokens: List[_Token], pos: int) -> Tuple[str, int]:
    '''读 `[...]` 的内容，返回 `(文本, 新位置)`。'''
    if pos >= len(tokens) or tokens[pos][0] != "[":
        return "", pos
    text, end = _render_sequence(tokens, pos + 1, stop=("]",))
    if end < len(tokens) and tokens[end][0] == "]":
        end += 1
    return text, end


def _read_argument(tokens: List[_Token], pos: int) -> Tuple[str, int]:
    '''读一个参数：`{...}` 整组，或者**下一个单 token**。

    单 token 是 TeX 的简写：`\\frac15` 就是 `\\frac{1}{5}`，`\\frac ab` 就是
    `\\frac{a}{b}`。以前只认 `{...}`，所以 `\\frac15` 整个漏掉 ——
    那句 `=\\frac15\\ln|x+1|` 就是这么残的。

    **先跳过空白**：TeX 里控制词后面的空格只是分隔符，`\\frac ab` 的两个参数
    是 `a` 和 `b`，不是「空格」和 `a`。
    '''
    while pos < len(tokens) and tokens[pos] == ("char", " "):
        pos += 1
    if pos >= len(tokens):
        return "", pos
    kind, value = tokens[pos]
    if kind == "{":
        text, end = _render_sequence(tokens, pos + 1, stop=("}",))
        if end < len(tokens) and tokens[end][0] == "}":
            end += 1
        return text, end
    if kind == "cmd":
        return _render_command(tokens, pos + 1, value)
    if kind == "sym":
        return _LITERAL_CHARS_FOR(value), pos + 1
    if kind == "char":
        return value, pos + 1
    return "", pos + 1


def _LITERAL_CHARS_FOR(value: str) -> str:
    r'''`\,` `\;` 之类：间距丢掉，其余还原成字面字符。'''
    if value in (" ", ",", ";", ":", "!", "\\"):
        return "\n" if value == "\\" else " "
    return value


def _render_command(
    tokens: List[_Token], pos: int, name: str
) -> Tuple[str, int]:
    '''渲染一个命令。`pos` 已经指向命令名之后。'''
    if name in _FRACTIONS:
        numerator, pos = _read_argument(tokens, pos)
        denominator, pos = _read_argument(tokens, pos)
        return f"({numerator})/({denominator})", pos

    if name in _BINOMIALS:
        top, pos = _read_argument(tokens, pos)
        bottom, pos = _read_argument(tokens, pos)
        return f"C({top}, {bottom})", pos

    if name == "sqrt":
        index, pos = _read_optional(tokens, pos)
        radicand, pos = _read_argument(tokens, pos)
        prefix = _script("^", index, _SUPERSCRIPT) if index else ""
        return f"{prefix}√({radicand})", pos

    if name in _ACCENT_MARKS:
        base, pos = _read_argument(tokens, pos)
        return base + _ACCENT_MARKS[name], pos

    if name in _TRANSPARENT:
        return _read_argument(tokens, pos)

    if name in ("begin", "end"):
        env, pos = _read_argument(tokens, pos)
        if name == "end":
            return "", pos
        # 渲染到 \end 为止，再把这个 \end{env} 吃掉
        body, pos = _render_sequence(tokens, pos, stop_commands=("end",))
        if pos < len(tokens) and tokens[pos] == ("cmd", "end"):
            pos += 1
            _name, pos = _read_argument(tokens, pos)
        if env not in _MULTILINE_ENVS:
            body = body.replace("\n", " ")
        return body, pos

    if name in _IGNORED:
        return "", pos

    if name in _FUNCTIONS:
        return name, pos

    if name in _SYMBOLS:
        return _SYMBOLS[name], pos

    # 认不出的命令：去掉反斜杠只留名字，再加个空格免得跟后面的撞在一起
    return name + " ", pos


def _render_sequence(
    tokens: List[_Token],
    pos: int,
    stop: Tuple[str, ...] = (),
    stop_commands: Tuple[str, ...] = (),
) -> Tuple[str, int]:
    '''渲染 `tokens[pos:]` 直到遇到 `stop` 里的 token 类型或 `stop_commands` 里的命令。

    `stop_commands` 命中时**不消费**那个 token，交给调用方处理（`\\end{env}` 要用）。
    '''
    out: List[str] = []
    while pos < len(tokens):
        kind, value = tokens[pos]

        if kind in stop:
            return "".join(out), pos
        if kind == "cmd" and value in stop_commands:
            return "".join(out), pos

        pos += 1

        if kind == "cmd":
            text, pos = _render_command(tokens, pos, value)
            out.append(text)
        elif kind == "{":
            text, pos = _render_sequence(tokens, pos, stop=("}",))
            if pos < len(tokens) and tokens[pos][0] == "}":
                pos += 1
            out.append(text)
        elif kind == "}":
            continue  # 多余的右括号，丢掉
        elif kind == "^":
            argument, pos = _read_argument(tokens, pos)
            out.append(_script("^", argument, _SUPERSCRIPT))
        elif kind == "_":
            argument, pos = _read_argument(tokens, pos)
            out.append(_script("_", argument, _SUBSCRIPT))
        elif kind == "sym" and value == "\\":
            pos = _skip_optional(tokens, pos)   # \\[2mm] 里的间距不要
            out.append("\n")
        elif kind == "sym":
            out.append(_LITERAL_CHARS_FOR(value))
        elif kind == "char":
            out.append(value)
        elif kind == "&":
            out.append(" ")
        elif kind in ("[", "]"):
            out.append(value)

    return "".join(out), pos



#: 需要「多行排版」才算好看的命令。
#: 简单公式（`y = \beta_0 + \beta_1 x_1`）不值得多开一次子进程，
#: 内置映射的 `β₀` 反而更紧凑；所以只在这些结构出现时才请 TeXicode 出马。
#:
#: 注意用 `(?![A-Za-z])` 而不是 `\b`：`\int_0^1` 里 `t` 和 `_` 之间**没有**词边界
#: （下划线算词字符），用 `\b` 会漏掉最常见的带上下限写法。
_NEEDS_ART = re.compile(
    r"\\(frac|dfrac|tfrac|int|iint|oint|sum|prod|lim|sqrt|binom"
    r"|matrix|cases|overbrace|underbrace)(?![A-Za-z])"
)


def needs_art(latex: str) -> bool:
    '''这条公式值不值得请 TeXicode 排版（多行分数 / 积分 / 求和…）。'''
    return bool(_NEEDS_ART.search(latex or ""))


# ============================ TeXicode 增强 ============================
#: TeXicode（`txc`）是外部 CLI：TeX -> Unicode art，输出比内置映射更完整漂亮。
#: 它是**可选增强**，没装完全不影响 —— 内置渲染器始终可用、作为兜底。
_TEXICODE_BIN = "txc"
#: 纯 Python 的转换很快，5 秒足够兜底（挂起时放弃，回退内置）。
_TEXICODE_TIMEOUT = 5.0


def _detect_texicode() -> bool:
    '''TeXicode CLI（`txc` / `txc.exe`）在不在 PATH 上。'''
    return shutil.which(_TEXICODE_BIN) is not None


#: import 时探测一次；之后再装 TeXicode 需要重启进程才能被识别。
HAVE_TEXICODE = _detect_texicode()


def texicode_render(latex: str) -> Optional[str]:
    '''用 TeXicode CLI 把一段 LaTeX 转成 Unicode art。

    任何失败（不可用 / 超时 / 非零退出 / 空输出）都返回 None，
    让调用方回退到内置渲染，绝不因一个外部工具把公式渲染弄崩。

    走 `subprocess.run([...])` 传参、不开 shell，避免命令注入。
    '''
    if not HAVE_TEXICODE:
        return None
    try:
        proc = subprocess.run(
            [_TEXICODE_BIN, latex],
            capture_output=True,
            text=True,
            timeout=_TEXICODE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


#: 公式渲染结果缓存：同一份 Markdown 反复渲染时不用重复解析
_MATH_CACHE: Dict[Tuple[str, bool, bool], str] = {}
_MATH_CACHE_LIMIT = 512


def render_math(
    latex: str,
    *,
    display: bool = False,
    prefer_texicode: Optional[bool] = None,
) -> str:
    r'''渲染一段公式。

    默认走**内置渲染器**（`latex_to_unicode`），零依赖、任何环境都能用。
    它是 LaTeX 的**保真翻译**：`\ln` 不会变成 `log(x, E)`，`\arctan` 不会变
    `atan`，`\frac15` 不会静默截断，恒等式也不会被算成一个布尔值。

    TeXicode 是可选增强：`prefer_texicode=True` 且本机装了 `txc` 时，
    复杂公式交给它渲染出更完整的 Unicode art；失败 / 没装都自动回退内置。
    也可以全局 `settings.prefer_texicode = True`。

    `prefer_texicode` 参数显式传值时优先；传 `None` 时读
    `settings.prefer_texicode`。
    '''
    use_texicode = (
        settings.prefer_texicode if prefer_texicode is None else prefer_texicode
    )
    key = (latex, display, use_texicode)
    cached = _MATH_CACHE.get(key)
    if cached is not None:
        return cached

    result = ""
    # TeXicode 优先：输出最忠实原式。只在「复杂公式 + 已开启 + 装了 CLI」时才
    # 请它出马，简单公式（x^2、\beta_0）用内置映射更快，也省一次子进程。
    if use_texicode and HAVE_TEXICODE and needs_art(latex):
        result = texicode_render(latex) or ""
    if not result:
        result = latex_to_unicode(latex) or latex.strip()

    if len(_MATH_CACHE) >= _MATH_CACHE_LIMIT:
        _MATH_CACHE.clear()
    _MATH_CACHE[key] = result
    return result


# ============================ Markdown ============================
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

#: LaTeX 正统定界符：`\[...\]` 和 `\(...\)`。
#: 这是 DeepSeek 等模型最常用的写法，必须支持 —— 否则模型写 `\(P\)`，
#: rich 会把 `\(` 当 Markdown 转义，最后只剩 `(P)`，公式全成原文。
#:
#: 开头的 `(?<!\\)` 不能省：`\\[2mm]` 是「换行 + 间距」，里面的 `\[`
#: 会被当成数学块开头，导致整个块**提前结束**、后半截公式退化成普通文字。
_BACKSLASH_DISPLAY_MATH = re.compile(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", re.S)
_BACKSLASH_INLINE_MATH = re.compile(r"(?<!\\)\\\((.+?)(?<!\\)\\\)")

#: 出现了这些才算「像公式」
_MATH_SIGNAL = re.compile(
    r"[\\^_=]|\w\s*[-+*/]\s*\w"
)

#: 单个短词也算公式：`$n$` / `$x_1$` / `$2n$` / `$\beta$`。
#: 必须以字母或反斜杠开头，或者是「数字+字母」（`2n`）——
#: `$100` / `$1.5` 这种纯数字就跟钱分不开，一律不当公式。
_SIMPLE_MATH = re.compile(
    r"^(?:[A-Za-z\\][\w\\^_{}()]*|\d+[A-Za-z][\w\\^_{}()]*)$"
)

#: 中日韩字符 —— 数学公式里不会有
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def looks_like_math(body: str) -> bool:
    '''判断 `$...$` 里装的到底是不是公式。

    只看「有一对 `$`」就当成公式是**会出事的** —— 中文里
    `价格是 $100，折扣 $50` 这种句子很常见，硬转会把钱数搅碎。
    所以要求：不含中日韩字符、长度有限，并且
    「出现数学信号」或「就是一个短的数学记号」（`$n$` / `$2n$`）。

    `\\(...\\)` / `\\[...\\]` 不走这个判断 —— 那是 LaTeX 自己的定界符，
    不会有歧义，一律当公式。
    '''
    if not body or len(body) > 200:
        return False
    if _CJK.search(body):
        return False
    return bool(_MATH_SIGNAL.search(body) or _SIMPLE_MATH.match(body))


#: 这些命令模型有时会写在**公式外面**（尤其 \blacksquare 当证明结束符）。
#: 它们不可能是正常文字，所以无条件替换，不管在不在定界符里。
_STANDALONE_SYMBOLS = {
    "blacksquare": "■",
    "qed": "∎",
    "square": "□",
    "checkmark": "✓",
}
_STANDALONE = re.compile(
    r"\\(" + "|".join(_STANDALONE_SYMBOLS) + r")(?![A-Za-z])"
)


def _replace_standalone(text: str) -> str:
    '''替换那些「写在公式外面也合法」的符号命令。'''
    return _STANDALONE.sub(lambda m: _STANDALONE_SYMBOLS[m.group(1)], text)


def _substitute_math(
    text: str, *, fence: bool = True, prefer_texicode: Optional[bool] = None
) -> str:
    '''把 Markdown 里的公式定界符换成终端能显示的文本。

    支持四种写法：

    | 定界符 | 含义 | 严格程度 |
    | --- | --- | --- |
    | `\\[...\\]` | 块级（LaTeX 正统） | 一律当公式 |
    | `\\(...\\)` | 行内（LaTeX 正统） | 一律当公式 |
    | `$$...$$` | 块级（Markdown 习惯） | 一律当公式 |
    | `$...$` | 行内（Markdown 习惯） | 先过 `looks_like_math()` |

    **必须在交给 rich 之前做**：Markdown 把 `\\(` 当转义序列，rich 会把它
    输出成裸的 `(`，定界符就没了 —— 模型写 `\\(P\\)` 最后只剩 `(P)`。

    `$...$` 之所以要额外判断：它和钱（`$100`）、和普通文字里的美元符号冲突；
    `\\(...\\)` 没有这个问题。
    '''

    def display(match: "re.Match[str]") -> str:
        body = render_math(
            match.group(1), display=True, prefer_texicode=prefer_texicode
        )
        # 包成代码块：rich 会用等宽字体原样输出（数学排版本来就依赖对齐）
        return f"\n```\n{body}\n```\n" if fence else f"\n{body}\n"

    def inline(match: "re.Match[str]") -> str:
        raw = match.group(1)
        if not looks_like_math(raw):
            return match.group(0)  # 不是公式，连 $ 一起原样留着
        return f"`{render_math(raw, prefer_texicode=prefer_texicode)}`"

    # 先处理「写在公式外面也合法」的符号（\blacksquare 之类）
    text = _replace_standalone(text)

    # LaTeX 正统定界符：无歧义，直接转
    text = _BACKSLASH_DISPLAY_MATH.sub(display, text)
    text = _BACKSLASH_INLINE_MATH.sub(
        lambda m: f"`{render_math(m.group(1), prefer_texicode=prefer_texicode)}`",
        text,
    )
    # Markdown 习惯的美元符号
    text = _DISPLAY_MATH.sub(display, text)
    return _INLINE_MATH.sub(inline, text)


def substitute_math(
    text: str, *, fence: bool = True, prefer_texicode: Optional[bool] = None
) -> str:
    '''公开入口：把 Markdown 里的公式定界符替换成终端可读文本。

    `_substitute_math` 的公开包装，供 TUI 等外部模块调用。默认 `fence=True`
    （块级公式包成代码块）；`prefer_texicode=True` 开启 TeXicode 增强。
    '''
    return _substitute_math(text, fence=fence, prefer_texicode=prefer_texicode)


def strip_markdown(text: str) -> str:
    '''没装 rich 时的兜底：把明显碍眼的标记去掉，其余原样保留。'''
    if not text:
        return ""
    text = _substitute_math(text, fence=False)
    text = re.sub(r"^```.*$", "", text, flags=re.M)          # 代码围栏
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # 标题
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)              # 粗体
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", text)  # 斜体
    text = re.sub(r"`([^`]+)`", r"\1", text)                  # 行内代码
    text = re.sub(r"^\s{0,3}[-*+]\s+", "· ", text, flags=re.M)  # 无序列表
    return text


def render_markdown(
    text: Any,
    *,
    stream: Any = None,
    title: Optional[str] = None,
    console: Any = None,
) -> str:
    '''把 Markdown 渲染到 `stream`，返回真正输出的文本。

    - 装了 `rich`：排版 + 代码高亮 + 表格，数学公式先转成 Unicode
    - 没装：退回纯文本（去掉碍眼的标记），公式照样是 Unicode

    无论哪条路径，数学公式都会被预处理 —— 这是零依赖的。
    '''
    body = "" if text is None else str(text)
    stream = sys.stdout if stream is None else stream

    if HAVE_RICH:
        console = console or get_console(stream)
        if console is not None:
            if title:
                console.rule(f"[bold cyan]{title}[/bold cyan]")
            console.print(_Markdown(_substitute_math(body)))
            return body

    # 兜底：没有 rich
    if title:
        print(f"{title}\n{'─' * max(4, min(len(title) + 2, 60))}", file=stream)
    print(strip_markdown(body), file=stream)
    return body


def render_plain(
    text: Any, *, stream: Any = None, title: Optional[str] = None
) -> str:
    '''按纯文本输出（但仍会把数学公式转成 Unicode）。

    走 `print()`，所以 `mock.patch("builtins.print")` 能照常截获，
    重定向 `sys.stdout` 也照常生效。
    '''
    body = "" if text is None else str(text)
    stream = sys.stdout if stream is None else stream
    if title:
        print(f"{title}\n{'─' * max(4, min(len(title) + 2, 60))}", file=stream)
    print(plain_text(body), file=stream)
    return body


#: `render=` 参数认得的值
RENDERERS = {
    "text": render_plain,
    "plain": render_plain,
    "markdown": render_markdown,
    "md": render_markdown,
}


def plain_text(text: Any) -> str:
    '''只做公式替换，返回字符串（不输出）。'''
    return _substitute_math("" if text is None else str(text), fence=False)
