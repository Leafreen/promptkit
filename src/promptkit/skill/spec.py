# skill/spec.py
'''`SkillSpec`（描述）和 `Skill`（可调用对象），以及 `@skill` 装饰器。

一个 Skill 就是「一个能说清楚自己是什么、怎么调、返回什么」的函数：
`name` + `description` + `parameters`（JSON Schema）+ `handler`。

    @skill(name="classify_sentiment", tags=["nlp"])
    def classify_sentiment(text: str, ctx: SkillContext) -> Sentiment:
        """对文本做情感分类。"""
        return ctx.structured(f"文本：{text}，输出 label 和 confidence", Sentiment)
'''
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from promptkit.skill.errors import SkillValidationError
from promptkit.skill.schema import build_parameters

__all__ = ["SkillSpec", "Skill", "skill", "build_spec"]

#: OpenAI 的函数名规则（也是我们愿意接受的名字）
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


@dataclass
class SkillSpec:
    '''一个 Skill 的全部描述。数据而已，不执行任何东西。'''

    name: str
    description: str
    handler: Callable[..., Any]
    #: JSON Schema；不给就按签名生成
    parameters: Dict[str, Any] = field(default_factory=dict)
    #: 输出契约（可选）：Pydantic 的 schema，或手写 JSON Schema
    returns: Optional[Dict[str, Any]] = None
    tags: Tuple[str, ...] = ()
    examples: Tuple[Dict[str, Any], ...] = ()
    version: str = "0.1"
    #: 任意业务字段（owner / cost / docs_url …）。函数上的公开属性会自动收到这里。
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: handler 接不接受注入的 `ctx`（装饰器自动判断）
    accepts_ctx: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not _NAME.match(self.name):
            raise SkillValidationError(
                f"Skill 名字不合法：{self.name!r}；"
                "只能由字母/数字/下划线/点/横线组成，且以字母或下划线开头（最长 64）"
            )
        self.description = str(self.description or "").strip()
        if not self.description:
            raise SkillValidationError(
                f"Skill {self.name!r} 需要一个 description —— 模型就是靠它决定调不调的"
            )
        if not callable(self.handler):
            raise TypeError(f"Skill {self.name!r} 的 handler 必须可调用")
        if not self.parameters:
            self.parameters = {"type": "object", "properties": {}}
        if self.parameters.get("type") != "object":
            raise SkillValidationError(
                f"Skill {self.name!r} 的 parameters 必须是 object 类型"
            )
        self.tags = tuple(str(tag) for tag in (self.tags or ()) if str(tag).strip())
        self.examples = tuple(dict(item) for item in (self.examples or ()))
        self.version = str(self.version or "0.1")
        self.metadata = {
            str(key): value
            for key, value in (self.metadata or {}).items()
            if str(key).strip()
        }

    # ---------------- 对外描述 ----------------
    def openai_schema(self) -> Dict[str, Any]:
        '''转成 OpenAI tools 里的一项。'''
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "tags": list(self.tags),
            "version": self.version,
            "examples": list(self.examples),
            "metadata": dict(self.metadata),
        }

    # ---------------- 执行 ----------------
    def invoke(self, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        if self.accepts_ctx:
            return self.handler(*args, ctx=ctx, **kwargs)
        return self.handler(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<SkillSpec {self.name!r} v{self.version}>"


class Skill:
    '''`@skill` 装饰之后拿到的东西：既是函数（可调用），又带着自己的描述。'''

    def __init__(self, spec: SkillSpec):
        self.spec = spec

    # 描述转发，方便 `skill.name` / `skill.parameters` 直接用
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.spec.parameters

    @property
    def returns(self) -> Optional[Dict[str, Any]]:
        return self.spec.returns

    @property
    def tags(self) -> Tuple[str, ...]:
        return self.spec.tags

    @property
    def metadata(self) -> Dict[str, Any]:
        '''任意业务字段（可读写）：`skill.metadata["owner"] = "team-a"`。'''
        return self.spec.metadata

    def with_metadata(self, **extra: Any) -> "Skill":
        '''链式补字段：`@skill(...).with_metadata(owner="team-a")`。'''
        for key, value in extra.items():
            if not str(key).startswith("_"):
                self.spec.metadata[str(key)] = value
        return self

    def __call__(self, *args: Any, ctx: Any = None, **kwargs: Any) -> Any:
        '''直接调用（测试 / 自己代码里用）。

        不传 `ctx` 时给一个空的 —— 用到模型能力时会报「没有可用的 agent」，
        而不是莫名其妙地 AttributeError。
        '''
        if ctx is None:
            from promptkit.skill.context import SkillContext

            ctx = SkillContext()
        return self.spec.invoke(ctx, *args, **kwargs)

    def schema(self) -> Dict[str, Any]:
        return self.spec.openai_schema()

    def __repr__(self) -> str:
        return f"<Skill {self.name!r} tags={list(self.tags)}>"


def skill(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    returns: Optional[Dict[str, Any]] = None,
    tags: Sequence[str] = (),
    examples: Sequence[Dict[str, Any]] = (),
    version: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    output_model: Any = None,
    registry: Any = None,
    replace: bool = False,
) -> Any:
    '''把函数变成 Skill。既能 `@skill` 也能 `@skill(name=..., tags=[...])`。

    也可以**先给函数挂字段，再套装饰器**：

        def summarize(text: str, max_words: int = 100) -> str:
            """把文本压缩到指定字数。"""

        summarize.tags = ["nlp"]
        summarize.version = "1.2"
        summarize.owner = "content-team"      # 任意额外字段 -> spec.metadata
        summarize.metadata = {"cost": "cheap"}

        summarize = skill(summarize)          # 等价于 @skill

    - `name` 默认取函数名；`description` 默认取 docstring 第一段
    - `parameters` 不给就**按签名自动生成**
    - `output_model` 给了，`returns` 自动取它的 JSON Schema
    - `registry` 给了就顺手注册进去
    - 函数上的公开属性（`tags` / `version` / `parameters` / `returns` /
      `description` / `name` / `examples` / `metadata` / `output_model`）会被读；
      其余公开属性都收进 `metadata`
    - 显式传的关键字参数**优先于**函数属性
    '''

    def wrap(fn: Callable[..., Any]) -> Skill:
        built = build_spec(
            fn,
            name=name,
            description=description,
            parameters=parameters,
            returns=returns,
            tags=tags,
            examples=examples,
            version=version,
            metadata=metadata,
            output_model=output_model,
        )
        wrapper = Skill(built)
        if registry is not None:
            registry.register(wrapper, replace=replace)
        return wrapper

    if func is None:
        return wrap
    return wrap(func)


def build_spec(
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    returns: Optional[Dict[str, Any]] = None,
    tags: Sequence[str] = (),
    examples: Sequence[Dict[str, Any]] = (),
    version: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    output_model: Any = None,
) -> SkillSpec:
    '''从函数（或 `Skill` / `SkillSpec`）造一个 `SkillSpec`。

    优先级：**显式参数 > 函数上的属性 > 签名 / docstring 推导**。
    '''
    if isinstance(func, Skill):
        spec = func.spec
        return _merge(spec, name=name, description=description, parameters=parameters,
                      returns=returns, tags=tags, examples=examples,
                      version=version, metadata=metadata, output_model=output_model)
    if isinstance(func, SkillSpec):
        return _merge(func, name=name, description=description, parameters=parameters,
                      returns=returns, tags=tags, examples=examples,
                      version=version, metadata=metadata, output_model=output_model)
    if not callable(func):
        raise TypeError(f"skill 需要可调用对象，得到 {type(func).__name__}")

    known, extra = _extras_from_function(func)

    skill_name = (
        name
        or known.get("name")
        or getattr(func, "__name__", "")
        or "skill"
    )
    skill_description = description if description is not None else known.get("description")
    if skill_description is None:
        doc = inspect.getdoc(func) or ""
        skill_description = doc.split("\n\n")[0].strip()
    if not skill_description:
        skill_description = f"调用 {skill_name}"

    parameters = parameters if parameters is not None else known.get("parameters")
    if parameters is None:
        parameters = build_parameters(func)

    created_returns = returns if returns is not None else known.get("returns")
    model = output_model if output_model is not None else known.get("output_model")
    if created_returns is None and model is not None:
        dump = getattr(model, "model_json_schema", None)
        if callable(dump):
            created_returns = dump()

    merged_tags = tuple(tags or ()) or tuple(known.get("tags") or ())
    merged_examples = tuple(examples or ()) or tuple(known.get("examples") or ())
    merged_version = version or known.get("version") or "0.1"
    merged_metadata: Dict[str, Any] = {
        **extra,
        **dict(known.get("metadata") or {}),
        **dict(metadata or {}),
    }

    return SkillSpec(
        name=skill_name,
        description=skill_description,
        handler=func,
        parameters=dict(parameters),
        returns=created_returns,
        tags=merged_tags,
        examples=merged_examples,
        version=merged_version,
        metadata=merged_metadata,
        accepts_ctx=_accepts_ctx(func),
    )


def _merge(spec: SkillSpec, **overrides: Any) -> SkillSpec:
    '''给已有的 spec 叠一层显式参数（重复装饰 / `skill(skill_obj)` 时用）。'''
    metadata = dict(spec.metadata)
    metadata.update(dict(overrides.get("metadata") or {}))
    model = overrides.get("output_model")
    returns = overrides.get("returns")
    if returns is None and model is not None:
        dump = getattr(model, "model_json_schema", None)
        if callable(dump):
            returns = dump()
    return SkillSpec(
        name=overrides.get("name") or spec.name,
        description=overrides.get("description") or spec.description,
        handler=spec.handler,
        parameters=overrides.get("parameters") or spec.parameters,
        returns=returns if returns is not None else spec.returns,
        tags=tuple(overrides.get("tags") or ()) or spec.tags,
        examples=tuple(overrides.get("examples") or ()) or spec.examples,
        version=overrides.get("version") or spec.version,
        metadata=metadata,
        accepts_ctx=spec.accepts_ctx,
    )


#: 这些名字是接口字段，不算「额外字段」
_KNOWN_ATTRS = frozenset(
    {
        "name",
        "description",
        "parameters",
        "returns",
        "tags",
        "examples",
        "version",
        "metadata",
        "output_model",
    }
)


def _extras_from_function(
    func: Callable[..., Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    '''把函数上挂的公开属性分成「接口字段」和「额外字段」。

    只读 `func.__dict__`（也就是你自己 `func.xxx = ...` 挂上去的那些），
    下划线开头的跳过 —— 那是给装饰器 / 框架用的，不该混进 metadata。
    '''
    known: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    raw = getattr(func, "__dict__", None) or {}
    for key, value in list(raw.items()):
        if key.startswith("_"):
            continue
        if key in _KNOWN_ATTRS:
            known[key] = value
        else:
            extra[key] = value
    return known, extra


def _accepts_ctx(func: Callable[..., Any]) -> bool:
    '''handler 里有没有 `ctx`（或标注成 SkillContext 的参数）。'''
    try:
        signature = inspect.signature(func)
        hints = inspect.get_annotations(func, eval_str=True)
    except (TypeError, ValueError):      # 内建函数之类
        return False

    for name, param in signature.parameters.items():
        if name == "ctx":
            return True
        annotation = hints.get(name)
        module = str(getattr(annotation, "__module__", ""))
        if getattr(annotation, "__name__", "") == "SkillContext" and "skill" in module:
            return True
    return False
