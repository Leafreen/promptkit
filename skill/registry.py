# skill/registry.py
'''`SkillRegistry`：注册、查、列、导成工具、执行、从插件加载。

设计上刻意**没有全局单例**：注册表是你显式创建、显式传进 `SkillContext` 的。
想要「装完就能用」就用 entry points（见 `load_entry_points`）。
'''
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from promptkit.skill.context import SkillContext
from promptkit.skill.errors import (
    SkillAlreadyExists,
    SkillExecutionError,
    SkillNotFound,
)
from promptkit.skill.schema import validate_args
from promptkit.skill.spec import Skill, SkillSpec, build_spec
from promptkit.skill.tools import SkillTool

__all__ = ["SkillRegistry", "default_group"]

#: entry points 的分组名
default_group = "promptkit.skills"


class SkillRegistry:
    '''一个 Skill 的集合。

        registry = SkillRegistry()
        registry.register(classify_sentiment)     # `@skill` 装饰过的
        registry.to_openai_tools()                # 给模型看的 schema
        registry.to_tools(ctx)                    # 给 Agent 真能调的工具
    '''

    def __init__(self, skills: Optional[Sequence[Any]] = None):
        self._skills: Dict[str, Skill] = {}
        #: 插件加载失败之类的记录（一条坏插件不该拖垮整个应用）
        self.errors: List[Dict[str, str]] = []
        for item in skills or ():
            self.register(item)

    # ---------------- 注册 ----------------
    def register(self, item: Any, *, replace: bool = False) -> Skill:
        '''接受 `Skill`（装饰过的函数）、`SkillSpec`，或者普通函数。'''
        if isinstance(item, Skill):
            skill = item
        elif isinstance(item, SkillSpec):
            skill = Skill(item)
        elif callable(item):
            skill = Skill(build_spec(item))
        else:
            raise TypeError(
                f"register 只接受 Skill / SkillSpec / 函数，得到 {type(item).__name__}"
            )

        name = skill.name
        if name in self._skills and not replace:
            raise SkillAlreadyExists(
                f"Skill {name!r} 已经注册过了；要覆盖请用 replace=True"
            )
        self._skills[name] = skill
        return skill

    def unregister(self, name: str) -> Skill:
        if name not in self._skills:
            raise SkillNotFound(f"没有叫 {name!r} 的 Skill")
        return self._skills.pop(name)

    # ---------------- 查询 ----------------
    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFound(
                f"没有叫 {name!r} 的 Skill；已注册：{', '.join(self.names) or '（空）'}"
            ) from None

    def find(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list(
        self,
        tag: Optional[str] = None,
        *,
        query: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Skill]:
        '''按标签 / 关键字 / 额外字段筛。

        `metadata={"owner": "team-a"}` 只留下「这些键值都对得上」的 Skill ——
        给「按团队 / 成本 / 版本路由」用。
        '''
        found = [skill for skill in self._skills.values()]
        if tag:
            found = [skill for skill in found if tag in skill.tags]
        if query:
            needle = str(query).lower()
            found = [
                skill
                for skill in found
                if needle in skill.name.lower() or needle in skill.description.lower()
            ]
        if metadata:
            found = [
                skill
                for skill in found
                if all(
                    skill.metadata.get(key) == value
                    for key, value in dict(metadata).items()
                )
            ]
        return sorted(found, key=lambda skill: skill.name)

    @property
    def names(self) -> List[str]:
        return sorted(self._skills)

    # ---------------- 导出 ----------------
    def to_openai_tools(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        '''导出成 OpenAI tools 的 schema。

        注意：这只给**描述**，`as_tools` 把它转成 Tool 之后 `func` 是空实现；
        要让 Agent 真能调，用 `to_tools(ctx)`。
        '''
        return [skill.spec.openai_schema() for skill in self.list(tag=tag)]

    def to_tools(
        self,
        ctx: Optional[SkillContext] = None,
        *,
        tag: Optional[str] = None,
    ) -> List[Any]:
        '''导出成可执行的 `Tool` 列表（需要时给一个 ctx）。'''
        context = ctx if ctx is not None else SkillContext()
        if context.skills is None:
            context.skills = self
        return [
            SkillTool(skill, ctx=context, registry=self)
            for skill in self.list(tag=tag)
        ]

    # ---------------- 执行 ----------------
    def execute(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        ctx: Optional[SkillContext] = None,
        validate: bool = True,
    ) -> Any:
        '''执行一个 Skill。参数会先按 schema 校验，异常会包成 `SkillExecutionError`。'''
        skill = self.get(name)
        values = dict(arguments or {})

        if validate:
            validate_args(skill.spec, values)

        context = ctx if ctx is not None else SkillContext()
        if context.skills is None:
            context.skills = self

        started = time.perf_counter()
        try:
            result = skill.spec.invoke(context, **values)
        except SkillExecutionError:
            raise
        except Exception as exc:      # noqa: BLE001 - 统一包一层，保留 __cause__
            context.record(
                name,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                duration=round(time.perf_counter() - started, 3),
            )
            raise SkillExecutionError(
                f"Skill {name!r} 执行失败：{type(exc).__name__}: {exc}"
            ) from exc

        context.record(
            name,
            status="ok",
            duration=round(time.perf_counter() - started, 3),
        )
        return result

    # ---------------- 插件 ----------------
    def load_entry_points(self, group: str = default_group) -> int:
        '''把第三方包里声明的 Skill 装进来（`promptkit.skills` 组）。

            [project.entry-points."promptkit.skills"]
            my_skills = "my_pkg.skills:register"

        `register` 是一个收 `SkillRegistry` 的函数；也可以直接指向一个 Skill。
        加载失败的插件记进 `self.errors`，不影响其它插件。
        '''
        entries = _entry_points(group)
        loaded = 0
        for entry in entries:
            entry_name = getattr(entry, "name", "?")
            try:
                target = entry.load()
            except Exception as exc:      # noqa: BLE001
                self.errors.append({"name": entry_name, "error": f"加载失败：{exc}"})
                continue

            try:
                if isinstance(target, (Skill, SkillSpec)):
                    self.register(target)
                elif callable(target):
                    target(self)
                else:
                    raise TypeError(
                        f"entry point 指向了 {type(target).__name__}；"
                        "要一个 register(registry) 函数或 Skill"
                    )
            except Exception as exc:      # noqa: BLE001
                self.errors.append({"name": entry_name, "error": str(exc)})
                continue
            loaded += 1
        return loaded

    # ---------------- 杂项 ----------------
    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: Any) -> bool:
        return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        return iter(self.list())

    def __repr__(self) -> str:
        return f"<SkillRegistry {len(self._skills)} 个: {', '.join(self.names)}>"


def _entry_points(group: str) -> Any:
    '''兼容不同 Python 版本的 `entry_points()` 用法。'''
    try:
        from importlib.metadata import entry_points
    except ImportError:      # pragma: no cover - 3.7 之前
        return []

    try:
        return entry_points(group=group)
    except TypeError:      # pragma: no cover - 老版本返回 dict
        return entry_points().get(group, [])
