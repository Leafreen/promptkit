# core/steps/actions/multimodal.py
'''多模态步骤：生成 / 理解，统一收成一个 Step。

底层只有 `MultimodalStep` 一个类，上层是四个函数式 DSL 入口：

    ImageGen("一只在月球上弹吉他的猫", save_to="cat.png", output_key="cat")
    ImageUnderstand(Var("cat"), prompt="描述这张图", output_key="desc")
    VideoGen("镜头缓慢推进", inputs=[Var("cat")], save_to="demo.mp4")
    VideoUnderstand(Var("demo"), prompt="描述这段视频")

它们都能直接放进 `Workflow(...)`，因为返回的是 `Step` 实例。
新增模态（audio / 3d……）时不用加新 Step，用
`Generate("audio", ...)` / `Understand("audio", ...)` 即可。
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

from promptkit.client_sys.multimodal import (
    Media,
    Modality,
    MultimodalResult,
    Operation,
    normalize_modality,
    normalize_operation,
)
from promptkit.core.steps.interface import Step

if TYPE_CHECKING:  # 只用于类型标注：steps 层不该在运行时依赖 Context
    from promptkit.core.context import Context


@dataclass
class MultimodalStep(Step):
    '''一次多模态调用。

    三个 DSL 入口最终都落到这里：

    - `Generate(modality, prompt, ...)`     生成文本 / 图片 / 视频
    - `Understand(modality, source, ...)`   理解图片 / 视频
    - `Multimodal(operation, modality, ...)` 最底层写法，未来新操作直接用它

    结果写回 `context[output_key]`：生成结果是 `Media`，理解结果是 `str`。
    如果给了 `save_to`，生成结果会先落盘，`Media.path` 就是本地路径。
    '''

    operation: str
    modality: str
    prompt: object = None
    sources: Sequence[object] = ()
    output_key: str = "output"
    save_to: object = None
    model: Optional[str] = None
    agent: object = None
    params: Dict[str, object] = field(default_factory=dict)

    name: str = "multimodal"

    def __post_init__(self) -> None:
        self.operation = normalize_operation(self.operation)
        self.modality = normalize_modality(self.modality)
        if self.sources is None:
            self.sources = ()
        elif not isinstance(self.sources, (list, tuple)):
            self.sources = (self.sources,)
        self.params = dict(self.params or {})

    # ---------------- 执行 ----------------
    def _agent(self, context: "Context") -> Any:
        agent = self.agent if self.agent is not None else context.get("agent")
        if agent is None:
            try:
                from promptkit.agent import Agent

                agent = Agent.default()
            except ImportError:
                agent = None
        if agent is None:
            raise ValueError(
                "MultimodalStep 需要一个 agent：先执行 `agent = Agent()`，"
                "或构造时传 agent=，或 Workflow.run(agent=...)"
            )
        if not hasattr(agent, "multimodal"):
            raise TypeError(
                f"{type(agent).__name__} 没有 multimodal()："
                "多模态步骤需要 promptkit.agent.Agent（或兼容它的实现）"
            )
        return agent

    def _prompt_text(self, context: "Context") -> Optional[str]:
        value = _resolve(self.prompt, context)
        if value is None:
            return None
        if isinstance(value, str) and "{{" in value:
            from promptkit.prompt import render_text

            return render_text(value, context.to_dict(), allow_dollar=False)
        return str(value)

    def _save_path(self, context: "Context") -> Optional[str]:
        value = _resolve(self.save_to, context)
        if value is None:
            return None
        if isinstance(value, str) and "{{" in value:
            from promptkit.prompt import render_text

            value = render_text(value, context.to_dict(), allow_dollar=False)
        return str(value)

    def execute(self, context: "Context") -> None:
        prompt = self._prompt_text(context)
        if self.operation == Operation.GENERATE and not prompt:
            raise ValueError(
                f"{self.describe()} 需要 prompt（生成操作不能是空提示词）"
            )

        inputs: Tuple[Media, ...] = tuple(
            Media.from_any(_resolve(item, context)) for item in self.sources
        )

        result: MultimodalResult = self._agent(context).multimodal(
            self.operation,
            self.modality,
            prompt=prompt,
            inputs=inputs,
            model=self.model,
            save_to=self._save_path(context),
            **self.params,
        )

        context[self.output_key] = result.content
        context.trace.append(f"{self.name}:{self.modality}:{self.operation}")

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        from promptkit.core.display_sys.tree import short

        source = ""
        if self.sources:
            source = f" <- {', '.join(short(item) for item in self.sources)}"
        prompt = f"({short(self.prompt)})" if self.prompt is not None else ""
        save = f" -> {short(self.save_to)}" if self.save_to is not None else ""
        return (
            f"{self.modality}:{self.operation}{prompt}{source}"
            f" -> {self.output_key}{save}"
        )

    def __repr__(self) -> str:
        return (
            f"MultimodalStep({self.operation!r}, {self.modality!r}, "
            f"output_key={self.output_key!r})"
        )


def _resolve(value: object, context: "Context") -> object:
    '''表达式求值；普通值原样返回。'''
    from promptkit.core.steps.logics.expr_sys.expr import Expr

    return value.eval(context) if isinstance(value, Expr) else value


def _sources(value: object) -> Tuple[object, ...]:
    '''单个媒体 / 媒体列表 / None -> tuple。'''
    from promptkit.core.steps.logics.expr_sys.expr import Expr

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or hasattr(value, "path"):
        return (value,)
    if isinstance(value, Expr):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


# ---------------- 函数式 DSL ----------------
def Multimodal(  # noqa: N802
    operation: str,
    modality: str,
    *,
    prompt: object = None,
    sources: object = None,
    output_key: str = "output",
    save_to: object = None,
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''最底层的多模态步骤：操作 / 模态都自己指定。'''
    return MultimodalStep(
        operation=operation,
        modality=modality,
        prompt=prompt,
        sources=_sources(sources),
        output_key=output_key,
        save_to=save_to,
        model=model,
        agent=agent,
        params=dict(params),
    )


def Generate(  # noqa: N802
    modality: str,
    prompt: object,
    *,
    inputs: object = None,
    output_key: str = "output",
    save_to: object = None,
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''生成：`Generate("image", "一只猫", save_to="cat.png")`。'''
    return Multimodal(
        Operation.GENERATE,
        modality,
        prompt=prompt,
        sources=inputs,
        output_key=output_key,
        save_to=save_to,
        model=model,
        agent=agent,
        **params,
    )


def Understand(  # noqa: N802
    modality: str,
    source: object = None,
    prompt: object = None,
    *,
    output_key: str = "output",
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''理解：`Understand("image", Var("cat"), prompt="描述")`。'''
    return Multimodal(
        Operation.UNDERSTAND,
        modality,
        prompt=prompt,
        sources=source,
        output_key=output_key,
        model=model,
        agent=agent,
        **params,
    )


def ImageGen(  # noqa: N802
    prompt: object,
    *,
    inputs: object = None,
    output_key: str = "image",
    save_to: object = None,
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''图片生成（默认豆包 Seedream）。'''
    return Generate(
        Modality.IMAGE,
        prompt,
        inputs=inputs,
        output_key=output_key,
        save_to=save_to,
        model=model,
        agent=agent,
        **params,
    )


def ImageUnderstand(  # noqa: N802
    source: object,
    prompt: object = "请描述这张图片。",
    *,
    output_key: str = "image_description",
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''图片理解（默认豆包 Seed 视觉模型）。'''
    return Understand(
        Modality.IMAGE,
        source,
        prompt,
        output_key=output_key,
        model=model,
        agent=agent,
        **params,
    )


def VideoGen(  # noqa: N802
    prompt: object,
    *,
    inputs: object = None,
    output_key: str = "video",
    save_to: object = None,
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''视频生成（默认豆包 Seedance）。'''
    return Generate(
        Modality.VIDEO,
        prompt,
        inputs=inputs,
        output_key=output_key,
        save_to=save_to,
        model=model,
        agent=agent,
        **params,
    )


def VideoUnderstand(  # noqa: N802
    source: object,
    prompt: object = "请描述这段视频。",
    *,
    output_key: str = "video_description",
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''视频理解（默认豆包 Seed 视觉模型）。'''
    return Understand(
        Modality.VIDEO,
        source,
        prompt,
        output_key=output_key,
        model=model,
        agent=agent,
        **params,
    )


def TextGen(  # noqa: N802
    prompt: object,
    *,
    output_key: str = "text",
    model: Optional[str] = None,
    agent: object = None,
    **params: object,
) -> MultimodalStep:
    '''文本生成：走同一套多模态入口，方便全链路只认一种客户端。'''
    return Generate(
        Modality.TEXT,
        prompt,
        output_key=output_key,
        model=model,
        agent=agent,
        **params,
    )


__all__ = [
    "MultimodalStep",
    "Multimodal",
    "Generate",
    "Understand",
    "ImageGen",
    "ImageUnderstand",
    "VideoGen",
    "VideoUnderstand",
    "TextGen",
]
