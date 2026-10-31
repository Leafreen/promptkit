# core/steps/actions/memory.py
'''和长期记忆打交道的 IO 原语：`ReadMemory` / `WriteMemory` / `ForgetMemory`，
外加把三件事组合起来的 `MemoryStep`。

为什么分成四种
--------------
| 想干的事 | 用哪个 |
| --- | --- |
| 按键读一个结构化值 | `ReadMemory` |
| 按键写一个结构化值 | `WriteMemory` |
| 删掉一个键 | `ForgetMemory` |
| 语义召回一堆非结构化内容 | `RecallMemory`（在 `recall.py`） |
| 一次配好「读 + 召回 + 写」 | `MemoryStep` |

**KV 走 IO，向量走检索，组合走 Step**。三者不重叠：
`MemoryStep` 不发明新语义，它只是把上面几步按顺序装起来。

记忆对象是**鸭子类型**的：只要它有 `read` / `write` / `forget` 就行，
不需要继承 `promptkit.memory.Memory`。所以 `core` 不需要 import memory 包
（只在类型标注里用），自己写个 Redis 客户端也能直接塞进来。
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from promptkit.core._missing import MISSING
from promptkit.core.steps.interface import Step, execute_step
from promptkit.core.steps.logics.expr_sys.expr import Expr

if TYPE_CHECKING:  # 只用于类型标注：运行时用鸭子类型，core 不依赖 memory 包
    from promptkit.core.context import Context
    from promptkit.memory.budget import TokenBudget

__all__ = [
    "ReadMemory",
    "WriteMemory",
    "ForgetMemory",
    "MemoryStep",
    "UseMemory",
]


# ============================ 小工具 ============================
def _key_of(key: object, context: Context) -> str:
    '''键可以写死，也可以用表达式算（`Var("who") + "_level"`）。'''
    value = key.eval(context) if isinstance(key, Expr) else key
    if value is None:
        raise ValueError("记忆的键不能是 None")
    return str(value)


def _value_of(value: object, context: Context) -> object:
    '''值是表达式就求值，否则原样（字面量就是字面量）。'''
    return value.eval(context) if isinstance(value, Expr) else value


def _as_list(items: object) -> List[object]:
    '''`read="level"` 和 `read=["level"]` 都当列表。'''
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    if isinstance(items, (list, tuple)):
        return list(items)
    return [items]


def _memory_label(memory: object) -> str:
    '''步骤树里怎么显示这个记忆。'''
    namespace = getattr(memory, "namespace", "")
    name = type(memory).__name__
    return f"{name}({namespace})" if namespace else name


def _require(memory: object, method: str, step: str) -> None:
    '''构造时就检查记忆有没有这个方法，别等跑起来才 AttributeError。'''
    if not callable(getattr(memory, method, None)):
        raise TypeError(
            f"{step} 需要实现了 {method}() 的记忆对象，"
            f"得到 {type(memory).__name__}（记忆是鸭子类型的：有这几个方法就行）"
        )


# ============================ 读 ============================
@dataclass
class ReadMemory(Step):
    '''按键读长期记忆，写进 Context。

        ReadMemory(kv, "level", output_key="level")
        ReadMemory(kv, "preferred_style")          # 变量名就是键名

    读不到时写 `default`（默认 `None`）—— 所以下游 `{{level}}` 不会因为
    「第一次来」而炸掉，你可以用 `If(Var("level") == None)` 走新手分支。
    '''

    memory: object
    key: object
    #: 写进哪个变量；省略就用键名
    output_key: Optional[str] = None
    default: object = None

    name: str = "read_memory"

    def __post_init__(self) -> None:
        _require(self.memory, "read", "ReadMemory")

    def execute(self, context: Context) -> None:
        key = _key_of(self.key, context)
        value = getattr(self.memory, "read")(key, self.default)
        context[self.output_key or key] = value
        context.trace.append(f"{self.name}:{key}")

    def describe(self) -> str:
        return (
            f"ReadMemory({_memory_label(self.memory)}, {self.key!r}"
            f" -> {self.output_key or '同名变量'})"
        )


# ============================ 写 ============================
@dataclass
class WriteMemory(Step):
    '''把值写进长期记忆。

        WriteMemory(kv, "level", "中级")                 # 字面量
        WriteMemory(kv, "level", Var("level"))            # Context 里的变量
        WriteMemory(kv, "level", from_key="level")        # 同上，省一个 Var()

    `from_key` 是给「编译出来的步骤」用的：`MemoryConfig.write=["level"]`
    编译成的就是 `from_key="level"`。

    变量不存在时怎么办由 `missing` 决定：`"skip"`（默认，什么都不写）或
    `"write"`（写 None）。默认跳过是因为「这轮没聊到这个字段」不该把
    上次记的值清成 None。
    '''

    memory: object
    key: object
    #: 要写的值（表达式或字面量）；和 `from_key` 二选一
    value: object = MISSING
    #: 从 Context 的哪个变量取值
    from_key: Optional[str] = None
    #: `"skip"` 变量不存在就跳过 / `"write"` 写 None
    missing: str = "skip"

    name: str = "write_memory"

    def __post_init__(self) -> None:
        _require(self.memory, "write", "WriteMemory")
        if self.value is MISSING and self.from_key is None:
            raise ValueError(
                "WriteMemory 需要 value= 或 from_key=；"
                "想写 Context 里的变量就写 from_key='变量名'"
            )
        if self.missing not in ("skip", "write"):
            raise ValueError(
                f"missing 只能是 'skip' 或 'write'，得到 {self.missing!r}"
            )

    def execute(self, context: Context) -> None:
        key = _key_of(self.key, context)
        if self.from_key is not None:
            if self.from_key not in context:
                if self.missing == "skip":
                    context.trace.append(f"{self.name}:skip")
                    return
                value = None
            else:
                value = context[self.from_key]
        else:
            value = _value_of(self.value, context)

        getattr(self.memory, "write")(key, value)
        context.trace.append(f"{self.name}:{key}")

    def describe(self) -> str:
        source = self.from_key if self.from_key is not None else self.value
        return (
            f"WriteMemory({_memory_label(self.memory)}, {self.key!r}"
            f" <- {source!r})"
        )


# ============================ 忘 ============================
@dataclass
class ForgetMemory(Step):
    '''删掉一个键（含它在向量记忆里的所有 chunk）。

        ForgetMemory(kv, "level")

    「可遗忘」是记忆系统的基本要求：写进去的东西必须能删掉。
    '''

    memory: object
    key: object

    name: str = "forget_memory"

    def __post_init__(self) -> None:
        _require(self.memory, "forget", "ForgetMemory")

    def execute(self, context: Context) -> None:
        key = _key_of(self.key, context)
        removed = bool(getattr(self.memory, "forget")(key))
        context.trace.append(f"{self.name}:{key}:{'ok' if removed else 'miss'}")

    def describe(self) -> str:
        return f"ForgetMemory({_memory_label(self.memory)}, {self.key!r})"


# ============================ 组合 ============================
@dataclass
class MemoryStep(Step):
    '''一次把「读 / 召回 / 写」配好。

        MemoryStep(
            kv,
            read=["level", "preferred_style"],
            recall=RecallConfig(query_key="usr_input", output_key="history"),
            write=["topics_covered"],
        )

    展开成的是普通步骤（`ReadMemory` / `RecallMemory` / `WriteMemory`），
    所以步骤树、事件、`describe()` 全都照常工作 —— 它不发明任何新语义。

    只写 `read=` 就只读；只写 `write=` 就只写。这样 `ChatWorkflow` 才能分别
    把「开跑前读」「每轮召回」「结束时写」放在三个位置。
    '''

    memory: object
    #: 要读的键名（或现成的 `ReadMemory`）
    read: Sequence[object] = ()
    #: `RecallConfig` / dict / 现成的 `RecallMemory`
    recall: object = None
    #: 要写的键名（或现成的 `WriteMemory`）；`(键, 变量名)` 可以分开指定
    write: Sequence[object] = ()
    #: 给召回用的预算（`RecallConfig.budget` 优先）
    budget: Optional["TokenBudget"] = None
    #: 变量不存在时：`"skip"` 跳过 / `"write"` 写 None
    missing: str = "skip"

    name: str = "memory"

    def __post_init__(self) -> None:
        self.steps: Tuple[Step, ...] = tuple(self._compile())
        if not self.steps:
            raise ValueError(
                "MemoryStep 至少要配 read= / recall= / write= 之一；"
                "什么都不做请用 SkipStep()，那样更明确"
            )

    # ---------------- 编译 ----------------
    def _compile(self) -> List[Step]:
        steps: List[Step] = []

        for entry in _as_list(self.read):
            steps.append(
                entry
                if isinstance(entry, ReadMemory)
                else ReadMemory(self.memory, entry)
            )

        if self.recall is not None:
            steps.append(self._recall_step(self.recall))

        for entry in _as_list(self.write):
            steps.append(self._write_step(entry))

        return steps

    def _recall_step(self, recall: object) -> Step:
        from promptkit.memory.config import RecallConfig

        from promptkit.core.steps.actions.recall import RecallMemory

        if isinstance(recall, dict):
            recall = RecallConfig(**recall)
        if isinstance(recall, RecallConfig):
            return RecallMemory(
                self.memory,
                query_key=recall.query_key,
                output_key=recall.output_key,
                top_k=recall.top_k,
                budget=recall.budget or self.budget,
                min_score=recall.min_score,
                raw=recall.raw,
            )
        if isinstance(recall, Step):
            return recall
        raise TypeError(
            "recall= 需要 RecallConfig / dict / RecallMemory，"
            f"得到 {type(recall).__name__}"
        )

    def _write_step(self, entry: object) -> Step:
        if isinstance(entry, WriteMemory):
            return entry
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            key, source = entry
            return WriteMemory(
                self.memory, key, from_key=str(source), missing=self.missing
            )
        return WriteMemory(self.memory, entry, from_key=str(entry), missing=self.missing)

    # ---------------- 执行 ----------------
    def execute(self, context: Context) -> None:
        for step in self.steps:
            execute_step(step, context)
        context.trace.append(self.name)

    def children(self) -> List[Step]:
        return list(self.steps)

    def describe(self) -> str:
        reads = sum(1 for step in self.steps if isinstance(step, ReadMemory))
        writes = sum(1 for step in self.steps if isinstance(step, WriteMemory))
        bits = [f"read={reads}"] if reads else []
        if self.recall is not None:
            bits.append("recall")
        if writes:
            bits.append(f"write={writes}")
        return f"MemoryStep({_memory_label(self.memory)}, {', '.join(bits)})"


class UseMemory(Step):
    '''链式写法：`UseMemory(mem).Read(...).Recall(...).Write(...)`。

        UseMemory(memory).Read("level", "goal").Recall("usr_input").Write(
            "topics_covered"
        )

    和 `MemoryStep(read=..., recall=..., write=...)` 完全等价 —— 只是把
    关键字参数换成链式调用，方便手写 `Workflow` 时读起来像一句话。
    真正执行时仍是编译出来的 `ReadMemory` / `RecallMemory` / `WriteMemory`，
    所以步骤树、事件、trace 都和 `MemoryStep` 一致。

    `memory` 接收 Memory / 路径 / 命名空间，和 `MEMORY_REFERENCE` /
    `WRITE_MEMORY` / `Compact(...).To(...)` 一致：

        UseMemory("data/tutor.db").Read("level")
    '''

    name = "use_memory"

    def __init__(self, memory: object):
        # 路径 / 命名空间要在这里落成记忆对象；延迟导入，core 不硬依赖 memory 包
        from promptkit.memory.layered import as_memory

        self.memory = as_memory(memory)
        self._read: List[object] = []
        self._recall: object = None
        self._write: List[object] = []
        self._steps: Optional[Tuple[Step, ...]] = None

    # ---------------- DSL ----------------
    def Read(self, *keys: object) -> "UseMemory":  # noqa: N802 - 跟随 DSL 命名
        '''按键读，键名同时就是 Context 变量名。'''
        _require(self.memory, "read", "UseMemory.Read")
        self._read.extend(keys)
        self._steps = None
        return self

    def Recall(
        self, query_key: str, output_key: str = "related"
    ) -> "UseMemory":  # noqa: N802 - 跟随 DSL 命名
        '''按 `query_key` 做语义召回，结果写进 `output_key`。'''
        _require(self.memory, "recall", "UseMemory.Recall")
        self._recall = {"query_key": query_key, "output_key": output_key}
        self._steps = None
        return self

    def Write(self, *keys: object) -> "UseMemory":  # noqa: N802 - 跟随 DSL 命名
        '''写回长期记忆；`("键", "变量名")` 可以分开指定。'''
        _require(self.memory, "write", "UseMemory.Write")
        self._write.extend(keys)
        self._steps = None
        return self

    # ---------------- 执行 ----------------
    def _compile(self) -> Tuple[Step, ...]:
        if self._steps is None:
            if not (self._read or self._recall or self._write):
                raise ValueError(
                    "UseMemory 至少要链一个 Read() / Recall() / Write()；"
                    "什么都不做请用 SkipStep()"
                )
            step = MemoryStep(
                self.memory,
                read=self._read,
                recall=self._recall,
                write=self._write,
            )
            self._steps = tuple(step.steps)
        return self._steps

    def execute(self, context: Context) -> None:
        for step in self._compile():
            execute_step(step, context)
        context.trace.append(self.name)

    def children(self) -> List[Step]:
        return list(self._compile())

    def describe(self) -> str:
        bits: List[str] = []
        if self._read:
            bits.append(f"read={len(self._read)}")
        if self._recall is not None:
            bits.append("recall")
        if self._write:
            bits.append(f"write={len(self._write)}")
        return f"UseMemory({_memory_label(self.memory)}, {', '.join(bits)})"
