# skill/schema.py
'''从函数签名生成 JSON Schema —— 手写 `parameters` 容易和实现走样。

支持的类型：`str / int / float / bool / list[T] / dict[str, T] / Optional[T] /
Literal[...] / Enum / Pydantic 模型`；`ctx` 参数会被跳过（它是注入的，不是入参）。
'''
from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

from promptkit.skill.errors import SkillValidationError

__all__ = ["build_parameters", "type_schema", "validate_args", "JSON_TYPES"]

JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

#: 这些参数名不进入 schema（由框架注入）
INJECTED = ("ctx", "context")


def build_parameters(func: Any, *, skip: Tuple[str, ...] = INJECTED) -> Dict[str, Any]:
    '''从函数签名生成 `{"type": "object", "properties": {...}, "required": [...]}`。'''
    try:
        hints = inspect.get_annotations(func, eval_str=True)
    except Exception:      # noqa: BLE001 - 标注里有求值不了的就用原始标注
        hints = dict(getattr(func, "__annotations__", {}))

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in skip or _is_context_annotation(hints.get(name)):
            continue

        schema = type_schema(hints.get(name, str))
        if param.default is not inspect.Parameter.empty:
            default = _json_default(param.default)
            if default is not None:
                schema = {**schema, "default": default}
        else:
            required.append(name)
        properties[name] = schema

    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return parameters


def type_schema(annotation: Any) -> Dict[str, Any]:
    '''把一个 Python 标注转成 JSON Schema 片段。'''
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in JSON_TYPES:
        return {"type": JSON_TYPES[annotation]}
    if annotation is list or annotation is tuple:
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}
    if annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        options = [arg for arg in args if arg is not type(None)]
        nullable = len(options) != len(args)
        if len(options) == 1:
            schema = type_schema(options[0])
            if nullable:
                schema = _nullable(schema)
            return schema
        return {"anyOf": [type_schema(arg) for arg in options]}

    if origin is not None and str(origin).endswith("Literal"):
        values = list(args)
        kinds = {type(value) for value in values}
        if len(kinds) == 1 and next(iter(kinds)) in JSON_TYPES:
            return {"type": JSON_TYPES[next(iter(kinds))], "enum": values}
        return {"enum": values}

    if origin in (list, tuple, set):
        array_schema: Dict[str, Any] = {"type": "array"}
        if args:
            array_schema["items"] = type_schema(args[0])
        return array_schema

    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            schema["additionalProperties"] = type_schema(args[1])
        return schema

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            values = [item.value for item in annotation]
            kind = {type(value) for value in values}
            if len(kind) == 1 and next(iter(kind)) in JSON_TYPES:
                return {"type": JSON_TYPES[next(iter(kind))], "enum": values}
            return {"enum": values}
        if hasattr(annotation, "model_json_schema"):      # Pydantic v2
            try:
                return annotation.model_json_schema()
            except Exception:      # noqa: BLE001
                return {"type": "object"}

    return {"type": "string"}


def validate_args(spec: Any, args: Dict[str, Any]) -> None:
    '''按 spec 的 `parameters` 检查调用参数：必填、未知字段、基本类型。

    不引入 jsonschema 依赖：只做「能挡住 90% 手误」的那部分。
    '''
    parameters = getattr(spec, "parameters", None) or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []

    missing = [name for name in required if name not in args]
    if missing:
        raise SkillValidationError(
            f"Skill {spec.name!r} 缺少必填参数：{', '.join(missing)}"
        )

    unknown = [name for name in args if name not in properties]
    if unknown:
        allowed = ", ".join(sorted(properties)) or "（无）"
        raise SkillValidationError(
            f"Skill {spec.name!r} 不认识这些参数：{', '.join(unknown)}；可用：{allowed}"
        )

    for name, value in args.items():
        if value is None:
            continue
        _check_type(spec.name, name, value, properties[name])


def _check_type(skill_name: str, param: str, value: Any, schema: Dict[str, Any]) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        options = expected
    elif expected:
        options = [expected]
    else:
        options = []

    if "anyOf" in schema:
        options = [
            item.get("type") for item in schema["anyOf"] if item.get("type")
        ]

    if not options:
        return

    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "array": lambda item: isinstance(item, (list, tuple)),
        "object": lambda item: isinstance(item, dict),
        "null": lambda item: item is None,
    }
    allowed = [name for name in options if name in checks]
    if allowed and not any(checks[name](value) for name in allowed):
        raise SkillValidationError(
            f"Skill {skill_name!r} 的参数 {param!r} 应该是 {' / '.join(allowed)}，"
            f"得到 {type(value).__name__}"
        )

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise SkillValidationError(
            f"Skill {skill_name!r} 的参数 {param!r} 只能是 {enum}，得到 {value!r}"
        )


def _is_context_annotation(annotation: Any) -> bool:
    if annotation is None:
        return False
    name = getattr(annotation, "__name__", "") or getattr(
        getattr(annotation, "__class__", None), "__name__", ""
    )
    return name in ("SkillContext", "Context") and "skill" in str(
        getattr(annotation, "__module__", "")
    )


def _nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return {**schema, "type": [kind, "null"]}
    if isinstance(kind, list) and "null" not in kind:
        return {**schema, "type": [*kind, "null"]}
    return schema


def _json_default(value: Any) -> Optional[Any]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return list(value)
    return None      # 复杂默认值不写进 schema，免得误导模型
