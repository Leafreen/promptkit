from dataclasses import dataclass
from typing import Any, Dict, List
import re

# 可用的提示词角色
SUPPORTED_ROLES = ("system", "user", "assistant")

# 变量占位符。两种语法的**严格程度不一样**：
#
#   {{key}}    规范写法，严格：变量不存在就报错（支持点路径 {{answer.text}}）
#   ${key}     $$key   兼容写法，宽松：变量不存在就**原样留着**
#
# 为什么美元符号要宽松 —— 它和这些东西天生冲突：
#
#   LaTeX 公式      $x_1$ / $$y = ax + b$$      （现在是常见需求）
#   价格            $100 / $1.5
#   普通文字        「请用 $ 或 $$ 包裹公式」      （$或$ 会被当成变量 `或`）
#
# 只要 $ 后面跟着一个词，就必然有歧义。硬报错会让这些完全正常的提示词
# 直接跑不起来，所以「找不到就不动它」。想要「写错立刻报错」就用 {{}}。
_VAR_PATTERN = re.compile(
    r"\{\{\s*([\w.]+)\s*\}\}"      # {{key}}  严格
    r"|\$\{(\w+)\}"                # ${key}   宽松
    r"|\$(\w+)(?![\w$])"           # $key     宽松；后面紧跟 $ 的当公式，不认
)

# 只认 {{key}}：给普通文本用（$ 后面的数字很容易是「价格」而不是变量名）
_BRACE_PATTERN = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _walk(value: Any, part: str, full_key: str) -> Any:
    '''从对象 / 字典 / 列表里取出一个路径片段。'''
    if isinstance(value, dict):
        if part in value:
            return value[part]
    elif isinstance(value, (list, tuple)):
        index = part[1:] if part.startswith("-") else part
        if index.isdigit():
            position = int(part)
            if -len(value) <= position < len(value):
                return value[position]
    elif not part.startswith("_") and hasattr(value, part):
        # 下划线开头的挡掉，避免 {{x.__class__}} 这种意外
        return getattr(value, part)

    raise KeyError(
        f"变量 {full_key!r} 里的 {part!r} 取不到："
        f"{type(value).__name__} 上没有这个字段"
    )


def resolve_variable(key: str, variables: Dict[str, Any]) -> Any:
    '''按名字取变量，支持点路径。

    `key` 精确命中就用它（所以真的有个 key 叫 "a.b" 也不会被拆开）；
    否则按 `.` 拆开逐段走属性 / 字典键 / 列表下标：

        {{intent}}            整个对象
        {{intent.label}}      Pydantic 字段 / 对象属性
        {{hits.0.text}}       列表下标
    '''
    if key in variables:
        return variables[key]

    if "." in key:
        head, _, rest = key.partition(".")
        if head in variables:
            value: Any = variables[head]
            for part in rest.split("."):
                if part:
                    value = _walk(value, part, key)
            return value

    available = ", ".join(sorted(map(str, variables))) or "（空）"
    raise KeyError(f"变量 {key!r} 不存在；当前可用：{available}")


def render_text(
    text: str, variables: Dict[str, Any], *, allow_dollar: bool = True
) -> str:
    '''把文本里的变量占位符替换成变量的值。

    两种语法的严格程度不同（见 `_VAR_PATTERN` 上面的说明）：

    - `{{key}}`：变量不存在 -> 抛 `KeyError`（带上可用变量名）
    - `$key` / `${key}`：变量不存在 -> **原样留着**，因为 `$` 和公式 / 价格冲突

    `allow_dollar=False` 时不认任何 `$` 写法，只认 `{{key}}`。
    '''
    pattern = _VAR_PATTERN if allow_dollar else _BRACE_PATTERN

    def replace(match: "re.Match[str]") -> str:
        braced = match.group(1)
        if braced is not None:
            return str(resolve_variable(braced, variables))

        # 美元符号写法：拿不准就放过，别把公式和价格搅碎
        key = match.group(2) or match.group(3)
        if key not in variables:
            return match.group(0)
        return str(variables[key])

    return pattern.sub(replace, text)


@dataclass(frozen=True)
class Prompt:
    '''
    提示词对象，支持可变提示词。

    - 变量使用 {{key}} 表示，key 是变量名（也兼容 ${key} / $key）
    - 支持点路径：{{user.name}} / {{hits.0.text}}
    - 使用 render(**kwargs) 渲染该对象，返回 OpenAI 支持的消息格式

    示例：
        >>> Prompt(role="user", content="你好，{{name}}").render(name="小明")
        {'role': 'user', 'content': '你好，小明'}
    '''

    role: str
    content: str

    def render(self, **kwargs: Any) -> Dict[str, str]:
        '''渲染该对象，返回支持的消息格式。'''
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(
                f"暂不支持该角色：{self.role!r}，可选：{', '.join(SUPPORTED_ROLES)}"
            )
        return {"role": self.role, "content": render_text(self.content, kwargs)}

    def _render_text(self, text: str, kwargs: Dict[str, Any]) -> str:
        '''渲染文本，替换其中的变量占位符。'''
        return render_text(text, kwargs)


def render_messages(prompts: List[Prompt], **kwargs: Any) -> List[Dict[str, str]]:
    '''批量渲染多个 Prompt，返回消息列表。'''
    return [p.render(**kwargs) for p in prompts]
