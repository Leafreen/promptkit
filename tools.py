'''工具（function calling）。

最简单的用法就是直接传 Python 函数，签名和 docstring 会自动变成 JSON Schema：

    def get_weather(city: str) -> str:
        """查询某个城市的天气。"""
        return f"{city}：晴，25℃"

    agent.ask("北京天气怎么样？", tools=[get_weather])

需要自定义名字 / 描述 / 参数时，用 Tool 显式构造：

    Tool("get_weather", "查询天气", {"type": "object", "properties": {...}}, get_weather)
'''
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin

# Python 类型 -> JSON Schema 类型
_JSON_TYPES: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    '''把 Python 类型标注翻译成 JSON Schema 的 type。'''
    origin = get_origin(annotation)
    if origin is Union:  # Optional[X] / X | None
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
        return "string"
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"
    return _JSON_TYPES.get(annotation, "string")


def _doc_summary(func: Callable[..., Any]) -> str:
    '''取 docstring 的第一段作为工具描述。'''
    doc = inspect.getdoc(func) or ""
    return doc.split("\n\n")[0].strip()


def schema_from_signature(func: Callable[..., Any]) -> Dict[str, Any]:
    '''从函数签名生成 parameters 的 JSON Schema。'''
    try:
        hints = inspect.get_annotations(func, eval_str=True)
    except Exception:  # 标注里有求值不了的东西就退回原始标注
        hints = getattr(func, "__annotations__", {})

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue  # *args / **kwargs 不进 schema
        properties[param_name] = {"type": _json_type(hints.get(param_name, str))}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    '''一个可以被模型调用的工具。'''

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("工具必须有名字")
        if not callable(self.func):
            raise TypeError(f"工具 {self.name!r} 的 func 必须可调用，得到 {self.func!r}")

    def schema(self) -> Dict[str, Any]:
        '''转成 API 需要的 {"type": "function", "function": {...}}。'''
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: Dict[str, Any]) -> str:
        '''执行工具，返回值统一转成字符串。'''
        return self.to_text(self.func(**arguments))

    @staticmethod
    def to_text(result: Any) -> str:
        '''工具结果必须是字符串；dict / list 用 JSON 表示。'''
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Tool":
        '''用函数的名字 / docstring / 签名自动生成工具定义。'''
        if not callable(func):
            raise TypeError(f"from_function 需要可调用对象，得到 {func!r}")
        return cls(
            name=name or getattr(func, "__name__", "tool"),
            description=(
                description if description is not None else _doc_summary(func)
            ),
            parameters=(
                parameters if parameters is not None else schema_from_signature(func)
            ),
            func=func,
        )


def _unimplemented(name: str) -> Callable[..., str]:
    def call(**_kwargs: Any) -> str:
        raise NotImplementedError(
            f"工具 {name!r} 只有声明、没有 Python 实现；"
            "请在 tools 里传 Tool(...) 或真正的函数"
        )

    return call


def as_tool(value: Any) -> Tool:
    '''把 Tool / 普通函数 / 原始 dict 统一成 Tool。'''
    if isinstance(value, Tool):
        return value
    if callable(value):
        return Tool.from_function(value)
    if isinstance(value, dict):
        fn = value.get("function", value)
        return Tool(
            name=fn.get("name", ""),
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
            func=_unimplemented(fn.get("name", "")),
        )
    raise TypeError(
        f"tools 里只能是 Tool、函数或 dict，得到 {type(value).__name__}"
    )


def as_tools(tools: Any) -> List[Tool]:
    '''把一串工具统一成 List[Tool]。'''
    if tools is None:
        return []
    return [as_tool(item) for item in tools]


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    '''装饰器写法：@tool 或 @tool(name="...")。'''

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_function(
            f, name=name, description=description, parameters=parameters
        )

    return wrap(func) if func is not None else wrap
