'''格式化器（Formatter）：把模型的原始文本变成结构化数据。

和 Pydantic 解耦：Formatter 只管「文本 -> dict / list」，
`output_model` 只管「dict -> 有类型的对象」。

    LLMStep(..., output_model=Intent)                       # 默认 JsonFormatter
    LLMStep(..., output_model=Intent, formatter=RegexFormatter(r"(?P<label>\\w+)"))
'''
import json
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Pattern, Tuple

# 匹配 ```json ... ``` 或 ``` ... ``` 代码块
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class FormatterError(ValueError):
    '''原始文本没法解析成结构化数据。'''


class Formatter(ABC):
    '''把一段原始文本解析成结构化数据。'''

    #: 展示用的短名字（会出现在步骤树里）
    name: str = "formatter"

    @abstractmethod
    def parse(self, raw: str) -> Any:
        '''解析；失败时抛 FormatterError。'''

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class JsonFormatter(Formatter):
    '''最常用的：从模型输出里尽力抽取 JSON。

    依次尝试：整段文本 -> Markdown 代码块 -> 文本中第一段 `{...}` / `[...]`。
    '''

    name = "json"

    def parse(self, raw: str) -> Any:
        if not raw or not raw.strip():
            raise FormatterError("模型返回了空内容")

        candidate = raw.strip()

        fence = _FENCE_RE.search(candidate)
        if fence:
            candidate = fence.group(1).strip()

        parsed, err = _try_loads(candidate)
        if err is None:
            return parsed

        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                parsed, err = _try_loads(candidate[start : end + 1])
                if err is None:
                    return parsed

        raise FormatterError(f"无法从模型输出中解析出 JSON：{candidate[:200]!r}")


class XmlFormatter(Formatter):
    '''把 `<output><a>1</a></output>` 解析成 `{"a": "1"}`。'''

    name = "xml"

    def __init__(self, root_tag: Optional[str] = None):
        self.root_tag = root_tag

    def parse(self, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise FormatterError("模型返回了空内容")

        fence = _FENCE_RE.search(text)
        if fence:
            text = fence.group(1).strip()

        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise FormatterError(f"XML 解析失败：{exc}：{text[:200]!r}") from None

        if self.root_tag and root.tag != self.root_tag:
            raise FormatterError(f"根标签应该是 <{self.root_tag}>，得到 <{root.tag}>")

        result: Dict[str, Any] = {}
        for child in root:
            result[child.tag] = child.text
        return result


class RegexFormatter(Formatter):
    '''用命名捕获组取值：`RegexFormatter(r"意图：(?P<label>\\w+)")`。'''

    name = "regex"

    def __init__(self, pattern: str):
        self.pattern: Pattern[str] = re.compile(pattern, re.DOTALL)

    def parse(self, raw: str) -> Dict[str, Any]:
        match = self.pattern.search(raw or "")
        if not match:
            raise FormatterError(f"正则没有匹配上：{self.pattern.pattern!r}")
        groups = match.groupdict()
        if not groups:
            raise FormatterError(
                f"正则里没有命名捕获组，拿不到字段：{self.pattern.pattern!r}"
            )
        return groups

    def __repr__(self) -> str:
        return f"RegexFormatter({self.pattern.pattern!r})"


class YamlFormatter(Formatter):
    '''YAML 输出。需要装 pyyaml（`pip install pyyaml`）。'''

    name = "yaml"

    def parse(self, raw: str) -> Any:
        try:
            import yaml
        except ImportError:  # pragma: no cover - 取决于环境
            raise FormatterError(
                "YamlFormatter 需要 pyyaml：pip install pyyaml"
            ) from None

        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:yaml)?\s*|\s*```$", "", text, flags=re.DOTALL)
        try:
            data = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001 - yaml 的异常类型不固定
            raise FormatterError(f"YAML 解析失败：{exc}") from None
        if data is None:
            raise FormatterError("模型返回了空内容")
        return data


def get_formatter(formatter: Any = None) -> Formatter:
    '''`None` -> JsonFormatter；字符串 -> 按名字找；对象 -> 原样返回。

    字符串支持：`"json"` / `"xml"` / `"yaml"`，以及 `"regex:<pattern>"`。
    '''
    if formatter is None:
        return JsonFormatter()
    if isinstance(formatter, Formatter):
        return formatter
    if isinstance(formatter, str):
        key = formatter.strip().lower()
        if key == "json":
            return JsonFormatter()
        if key == "xml":
            return XmlFormatter()
        if key == "yaml":
            return YamlFormatter()
        if key.startswith("regex:"):
            return RegexFormatter(formatter.split(":", 1)[1])
        raise FormatterError(
            f"不认识的 formatter {formatter!r}；"
            "可用：json / xml / yaml / regex:<pattern>，或直接传 Formatter 实例"
        )
    raise TypeError(
        f"formatter 必须是 Formatter 或字符串，得到 {type(formatter).__name__}"
    )


def _try_loads(text: str) -> Tuple[Any, Optional[json.JSONDecodeError]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, exc


__all__ = [
    "Formatter",
    "FormatterError",
    "JsonFormatter",
    "XmlFormatter",
    "RegexFormatter",
    "YamlFormatter",
    "get_formatter",
]
