'''知识库：分块 + 检索 + 拼进提示词。

⚠️ **实验性功能，默认关闭。** import 本模块时会先过一道
`features.require("knowledge")` 守卫，没启用就直接抛 `FeatureDisabled`
（错误信息里写了两种启用方式）：

    from promptkit import features
    features.enable("knowledge")        # 或设环境变量 PROMPTKIT_ENABLE_KNOWLEDGE=1

    from promptkit.knowledge import KnowledgeBase

分工：工作流负责**编排**（什么时候检索、检索不到怎么办、结果怎么进提示词），
本模块只负责**检索本身**。它是普通 Python，通过 `Call(kb.search, ...)`
或 `kb.as_tool()` 接进工作流。

零依赖：BM25 词法检索，中文按「双字 + 单字」切分；不联网、不需要向量库。
要语义检索就替换 `KnowledgeBase.search`（保持签名），工作流层不用改 ——
这正是把检索独立出来的意义。

典型用法：

    kb = KnowledgeBase.from_dir("docs")     # 建库（默认只收 md / txt）
    kb.sync_dir("docs")                     # 增量同步，可反复执行
    hits = kb.search("怎么退款", k=3)        # 检索
    print(kb.format(hits))                  # 拼成可直接贴进提示词的文字
    LLMStep(..., tools=[kb.as_tool()])      # 或让模型自己决定查什么
    kb.save("kb.json")                      # 落盘（原子写）

公开名字见 `__all__`：`KnowledgeBase`、`Hit`、`chunk_text`、`tokenize`、
`token_weight`、`KNOWLEDGE_REFERENCE`、`FORMAT_VERSION`。
'''
import json
import math
import os
import re
import tempfile
import time

from promptkit.features import require

# 守卫：没打开就 import 失败，并告诉你怎么打开
require("knowledge")

from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence  # noqa: E402

from promptkit.tools import Tool  # noqa: E402

#: 持久化文件的格式版本（读旧文件永远兼容，读到更新的会报错）
FORMAT_VERSION = 2
_FORMAT_NAME = "promptkit-knowledge"


def _replace_with_retry(source: str, target: str, attempts: int = 5) -> None:
    '''`os.replace`，在 Windows 上失败就重试几次。

    Windows 上 `os.replace` 会抛 `PermissionError`（WinError 5），
    即使目标文件没被这个进程打开 —— 刚写完关掉的临时文件经常正被杀毒软件
    或索引服务扫，那一刻就改不了名。等几十毫秒通常就好了。
    实测加了重试之后，之前偶发的测试失败不再出现。

    参数：
        source（str）：临时文件路径；成功后它被移动走（原路径不再存在）。
        target（str）：目标路径，被原子地覆盖。
        attempts（int，默认 5）：最多尝试次数（含第一次）。传 0 或负数时
            一次都不尝试、文件原样不动、也不报错 —— 调用方应保证 >= 1。

    返回：
        None —— 成功即返回，没有返回值。

    异常：
        PermissionError：`attempts` 次都失败时抛出**最后一次**的异常。
        OSError：其它 `os.replace` 错误（跨盘、路径不存在等）不重试，直接抛。

    注意：
        退避从 20ms 起步、每次翻倍（20 / 40 / 80 / 160ms…），只有在
        `PermissionError` 时才重试，其它错误立刻抛出以免掩盖真实问题。
    '''
    delay = 0.02
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2

# ---------- 分词 ----------
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

# BM25 参数
_K1 = 1.5
_B = 0.75

#: 单字 token 的权重。bigram 命中说明词真的对上了；单字命中只说明「沾边」。
#: 不给权重的话，「量子纠缠」里的 `子` 会盖过真正的相关块；
#: 完全不用单字的话，查询「退吗」就永远匹配不到「退款」（bigram 不交叉）。
_UNIGRAM_WEIGHT = 0.25


def token_weight(token: str) -> float:
    '''token 的检索权重：多字 token 按原样，单字 token 打折。

    参数：
        token（str）：`tokenize` 产出的任意 token；空串、单个汉字、单个
            英文字母都算「单字」。

    返回：
        float —— `len(token) > 1` 时返回 1.0；否则返回 `_UNIGRAM_WEIGHT`（0.25）。

    注意：
        单字只说明「沾边」，双字命中才说明词真的对上了。不给权重时
        「量子纠缠」里的 `子` 会盖过真正相关的块；完全不用单字时，
        查询「退吗」又匹配不到「退款」（bigram 不交叉）。
    '''
    return 1.0 if len(token) > 1 else _UNIGRAM_WEIGHT


def tokenize(text: str) -> List[str]:
    '''把文本切成检索用的 token：英文按单词、中文按「双字 + 单字」。

    参数：
        text（str）：待切文本。`None`、空串、纯空白/纯标点都得到空列表。

    返回：
        List[str] —— 按出现顺序排列：先把英文/数字/下划线的连续段（已转
        小写）放进结果，再追加每个中文连续片段的单字与双字（同一片段内
        先全部单字、再全部双字）。**不做去重** —— BM25 需要词频。

    切分规则：
        - 英文、数字、下划线：`[a-zA-Z0-9_]+` 为一段，统一转小写
        - 中文汉字：每个连续片段先切单字（兜底召回），再切相邻双字
          （主力）。例如「如何退款」→ 如 / 何 / 退 / 款 / 如何 / 何退 / 退款
        - 其它字符（标点、空白、emoji）直接丢弃

    注意：
        中文没有空格，直接 `split()` 会把整句变成一个 token，检索基本失效；
        双字是够用的近似，单字用来兜住 bigram 对不齐的情况。中英混排时
        两套 token 会一起进结果。
    '''
    if not text:
        return []
    tokens: List[str] = _ASCII_WORD.findall(text.lower())
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)  # 单字：兜底召回
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 双字：主力
    return tokens


# ---------- 分块 ----------
def chunk_text(text: str, max_chars: int = 400, overlap: int = 60) -> List[str]:
    '''把长文本切成检索用的块：先按空行分段，过长的段再按长度滑窗切。

    参数：
        text（str）：原文。`None` 视为空串（返回空列表）。
        max_chars（int，默认 400）：单块长度上限，按**字符数**计（不是字节）。
            必须 > 0。
        overlap（int，默认 60）：相邻块的重叠字符数，避免答案正好被切断在
            块边界。会被**钳制**到 `[0, max_chars - 1]`：负数按 0 处理；
            大于等于 `max_chars` 时取 `max_chars - 1`（保证每次至少前进 1 个字符）。

    返回：
        List[str] —— 分好的块，每块都已 `strip()`，空块被丢弃。
        - 长度 <= `max_chars` 的段落整体作为一块，不再切分
        - 更长的段落按步长 `max_chars - overlap` 滑窗，末块会与前一块重叠
        - 段落之间不会合并成同一块

    异常：
        ValueError：`max_chars <= 0`。

    注意：
        分段依据是**空行**（连续两个换行，中间可以夹空白）；段落内部的
        单个换行会保留在块里。本函数纯按字符数切，不做语义切分、不去重，
        也不保证块与块之间不重复（有 `overlap` 就必然重复）。
    '''
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正")
    overlap = max(0, min(overlap, max_chars - 1))

    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = start + max_chars
            chunks.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = end - overlap
    return [c for c in chunks if c]


@dataclass
class Hit:
    '''一条检索结果。

    字段：
        text（str）：命中的原文块，未做高亮、未做截断。
        source（str，默认 ""）：块来源，通常是文件名。手工 `add()` 时
            没给 `source`，这里就是空串。
        score（float，默认 0.0）：BM25 得分，越大越相关，**没有固定上界**，
            也不保证落在 0~1；只能用来在同一批结果里排序。
        index（int，默认 -1）：该块在 `KnowledgeBase.chunks` 里的下标，
            可用来回查原始块；`-1` 表示不是从知识库检索出来的
            （手工构造的 `Hit`）。
    '''

    text: str
    source: str = ""
    score: float = 0.0
    index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        '''转成可 JSON 序列化的字典。

        返回：
            dict —— 四个键：`text`（str）、`source`（str）、
            `score`（float，**四舍五入到 4 位小数**）、`index`（int）。

        注意：
            只是展示 / 落盘用的快照，四舍五入会丢精度；需要精确分数
            就直接读 `.score`。
        '''
        return {
            "text": self.text,
            "source": self.source,
            "score": round(self.score, 4),
            "index": self.index,
        }

    def __str__(self) -> str:
        '''人类可读形式：`正文（来源）`；来源为空时只输出正文。

        注意：
            给日志 / 调试用，不带得分、不带高亮；要拼提示词请用
            `KnowledgeBase.format(...)`（会带上编号与相关度）。
        '''
        where = f"（{self.source}）" if self.source else ""
        return f"{self.text}{where}"


@dataclass
class KnowledgeBase:
    '''一堆文本块 + 一张倒排索引：建库、检索、持久化。

    典型生命周期：

        kb = KnowledgeBase.from_dir("docs")      # 建库
        kb.add("补充的一段", source="note.md")    # 或手工追加
        kb.sync_dir("docs")                      # 增量同步（可反复执行）
        hits = kb.search("退款", k=3)             # 检索
        kb.save("kb.json")                       # 落盘

    字段：
        chunks（List[str]，默认空）：全部块，下标即 `Hit.index`。
        sources（List[str]，默认空）：与 `chunks` 一一对应的来源名。
            构造时若比 `chunks` 短会自动补空串（见 `__post_init__`）；
            多出来的项保留但不参与检索。
        max_chars（int，默认 400）：`add()` 分块时的单块上限（字符数）。
        overlap（int，默认 60）：`add()` 分块时的重叠字符数（会被钳制，
            规则见 `chunk_text`）。
        ingested（Dict[str, Dict[str, Any]]，默认空）：增量同步台账，
            `source -> {"mtime": float, "size": int}`；只有 `sync_dir`
            会写它。

    派生数据（`_index` 倒排表、`_lengths` 块长度）不通过构造传入，按需重建：
    任何改动 `chunks` 的操作都只把索引标脏，下一次检索时一次性重建 ——
    批量灌数据比每块都重建划算得多。
    '''

    chunks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    max_chars: int = 400
    overlap: int = 60
    #: 增量同步的台账：source -> {"mtime": ..., "size": ...}
    ingested: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 派生数据：按需重建（_dirty 时）
    _index: Dict[str, Dict[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lengths: List[int] = field(default_factory=list, init=False, repr=False)
    _dirty: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        '''把 `sources` 补齐到与 `chunks` 等长（缺的补空串）。

        返回：
            None

        注意：
            只补不裁：`sources` 比 `chunks` 长时原样保留，多出的项不会被
            用到，`remove_source` / `search` 都只按下标取。
        '''
        if len(self.sources) < len(self.chunks):
            self.sources = list(self.sources) + [""] * (
                len(self.chunks) - len(self.sources)
            )

    # ---------------- 索引 ----------------
    def _rebuild(self) -> None:
        '''扫一遍全部块，重建倒排索引与块长度表，并清掉脏标记。

        返回：
            None

        注意：
            复杂度 O(总 token 数)。只在索引脏或块数变化时调用；`add()` 期间
            不做增量更新，避免每块都重建一次。
        '''
        index: Dict[str, Dict[int, int]] = {}
        lengths: List[int] = []
        for position, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk)
            lengths.append(len(tokens))
            for token in tokens:
                postings = index.setdefault(token, {})
                postings[position] = postings.get(position, 0) + 1
        self._index = index
        self._lengths = lengths
        self._dirty = False

    def _ensure_index(self) -> None:
        '''索引脏、或长度表与块数对不上时重建；否则什么都不做。

        返回：
            None
        '''
        if self._dirty or len(self._lengths) != len(self.chunks):
            self._rebuild()

    # ---------------- 建库 ----------------
    def add(self, text: str, source: str = "") -> "KnowledgeBase":
        '''追加一段文本，按 `max_chars` / `overlap` 自动分块。

        参数：
            text（str）：原文。空串或纯空白不会产生任何块。
            source（str，默认 ""）：来源名，会记到这段文本产生的**每一个**
                块上（`Hit.source`），也是 `remove_source` / `sync_dir`
                按来源删除的依据。

        返回：
            KnowledgeBase —— `self`，便于链式 `kb.add(a).add(b)`。

        异常：
            ValueError：`self.max_chars <= 0` 时由 `chunk_text` 抛出。

        注意：
            只把索引标脏，不立刻重建；连续 `add` 之后第一次 `search`
            才付分词成本。
        '''
        for chunk in chunk_text(text, self.max_chars, self.overlap):
            self.chunks.append(chunk)
            self.sources.append(source)
        self._dirty = True
        return self

    def remove_source(self, source: str) -> int:
        '''删掉某个来源的全部块。

        参数：
            source（str）：**精确匹配**的来源名（不做大小写、路径归一化，
                所以 `"docs/a.md"` 和 `"a.md"` 是两个来源）。

        返回：
            int —— 实际删掉的块数。为 0 表示来源不存在：知识库不变，
            索引也不会被标脏。

        注意：
            只改 `chunks` / `sources`，**不动** `ingested` 台账。想连台账
            一起清用 `sync_dir(prune=True)`，或手动 `del kb.ingested[name]` ——
            否则下次 `sync_dir` 会把该文件当成「没变过」而跳过。
        '''
        kept = [
            (text, where)
            for text, where in zip(self.chunks, self.sources)
            if where != source
        ]
        removed = len(self.chunks) - len(kept)
        if removed:
            self.chunks = [text for text, _ in kept]
            self.sources = [where for _, where in kept]
            self._dirty = True
        return removed

    def add_file(self, path: str, encoding: str = "utf-8") -> "KnowledgeBase":
        '''读一个文件并整篇 `add()` 进去。

        参数：
            path（str）：文件路径。来源名取**文件名**（`os.path.basename`），
                不含目录 —— 所以不同目录下的同名文件会被视为同一来源。
            encoding（str，默认 "utf-8"）：解码方式。读取时用
                `errors="replace"`，编码不对**不会**抛 `UnicodeDecodeError`，
                只会出现替换字符。

        返回：
            KnowledgeBase —— `self`，可链式调用。

        异常：
            FileNotFoundError：路径不存在。
            IsADirectoryError：路径是目录。
            PermissionError：没有读权限。

        注意：
            不做增量判断，同一文件调两次就灌两份；要幂等用 `sync_dir`。
        '''
        with open(path, encoding=encoding, errors="replace") as handle:
            return self.add(handle.read(), source=os.path.basename(path))

    def add_dir(self, directory: str, pattern: str = r".*\.(md|txt)$") -> "KnowledgeBase":
        '''把一个目录下的匹配文件全部灌进来（**不递归**子目录）。

        参数：
            directory（str）：目录路径，只扫它的**第一层**文件。
            pattern（str，默认匹配 `.md` / `.txt` 结尾）：正则，编译时
                不区分大小写，用 `search`（不是全匹配）匹配**文件名**，
                所以默认值等价于「后缀是 md 或 txt」。

        返回：
            KnowledgeBase —— `self`。

        异常：
            FileNotFoundError：目录不存在。
            PermissionError：目录不可读。
            ValueError：`pattern` 不是合法正则（`re.compile` 抛出）。

        注意：
            不做增量、不删已消失的文件、也不写 `ingested` 台账 —— 反复调用
            会重复灌。要幂等请用 `sync_dir`。
        '''
        matcher = re.compile(pattern, re.IGNORECASE)
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            if os.path.isfile(full) and matcher.search(name):
                self.add_file(full)
        return self

    def sync_dir(
        self,
        directory: str,
        pattern: str = r".*\.(md|txt)$",
        *,
        force: bool = False,
        prune: bool = True,
        encoding: str = "utf-8",
    ) -> Dict[str, int]:
        '''对着一个目录做**增量**同步，返回各项计数。

        参数：
            directory（str）：目录路径，只扫第一层文件（不递归）。
            pattern（str，默认匹配 `.md` / `.txt` 结尾）：正则，不区分大小写，
                用 `search` 匹配文件名（见 `add_dir`）。
            force（bool，默认 False，仅关键字）：True 时无视台账、把匹配到的
                文件全部重灌；False 时「修改时间（截到毫秒）+ 文件大小」都没
                变的文件直接跳过。
            prune（bool，默认 True，仅关键字）：True 时把**磁盘上已不存在**的
                来源从知识库和台账里一起删掉。
            encoding（str，默认 "utf-8"，仅关键字）：读取编码，`errors="replace"`。

        返回：
            Dict[str, int] —— 四个计数键，单位都是**文件**（不是块）：
            `added`（台账里没有）、`updated`（台账里有但指纹变了）、
            `skipped`（指纹没变）、`removed`（被 prune 掉的来源数）。

        异常：
            FileNotFoundError：目录不存在。
            PermissionError：目录或文件不可读。
            ValueError：`pattern` 不是合法正则。

        注意：
            - **幂等**：同一目录重复跑，第二次全部计入 `skipped`。
            - 指纹只看 mtime + size：内容改了但大小与毫秒级 mtime 都没变
              （极少见）会被误判成没变，用 `force=True` 兜底。
            - `updated` 会先 `remove_source(name)` 再重新 `add`，所以更新过的
              文件其块会排到末尾（`Hit.index` 随之变化）。
            - `prune` 只清本台账里的名字；手工 `add()` 进来的块不受影响。
            - 只认文件名当来源：两个目录下的同名文件会互相覆盖。
        '''
        matcher = re.compile(pattern, re.IGNORECASE)
        stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0}
        present = set()

        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            if not (os.path.isfile(full) and matcher.search(name)):
                continue
            present.add(name)

            info = os.stat(full)
            fingerprint = {
                "mtime": round(info.st_mtime, 3),
                "size": info.st_size,
            }
            previous = self.ingested.get(name)

            if previous == fingerprint and not force:
                stats["skipped"] += 1
                continue

            if previous is None:
                stats["added"] += 1
            else:
                self.remove_source(name)
                stats["updated"] += 1

            with open(full, encoding=encoding, errors="replace") as handle:
                self.add(handle.read(), source=name)
            self.ingested[name] = fingerprint

        if prune:
            for gone in [s for s in self.ingested if s not in present]:
                self.remove_source(gone)
                del self.ingested[gone]
                stats["removed"] += 1

        return stats

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs: Any) -> "KnowledgeBase":
        '''用一堆文本快速建库。

        参数：
            texts（Iterable[str]）：按顺序逐条 `add()`。来源名全部为空串，
                所以这些块不能用 `remove_source` 精准删除。生成器也可以，
                会被一次性消费。
            **kwargs：透传给 `KnowledgeBase(...)`，如 `max_chars` / `overlap`。

        返回：
            KnowledgeBase —— 新实例。

        注意：
            逐条 `add()` 只标脏，索引在第一次检索时一次性重建，建库阶段
            不会反复分词。
        '''
        kb = cls(**kwargs)
        for text in texts:
            kb.add(text)
        return kb

    @classmethod
    def from_dir(cls, directory: str, **kwargs: Any) -> "KnowledgeBase":
        '''用目录建库，等价于 `cls(**kwargs).add_dir(directory)`。

        参数：
            directory（str）：目录路径，只收第一层、后缀 md / txt 的文件。
            **kwargs：透传给 `KnowledgeBase(...)`。注意 `pattern` 不在这里
                透传 —— 要自定义扩展名就直接调 `add_dir` / `sync_dir`。

        返回：
            KnowledgeBase —— 新实例。

        异常：
            FileNotFoundError：目录不存在。

        注意：
            语义是「一次性全量灌」：重复调用会重复灌。要幂等请用
            `KnowledgeBase(**kwargs).sync_dir(directory)`。
        '''
        return cls(**kwargs).add_dir(directory)

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> List[Hit]:
        '''BM25 检索，返回得分最高的若干块。

        参数：
            query（str）：查询串，与块使用同一套 `tokenize` 切词。空串直接
                返回空列表（连索引都不建）。
            k（int，默认 3）：最多返回几条。必须为正整数：传 0 或负数一律返回
                空列表（不会退化成 `[:k]` 的负切片语义去「切掉末尾」）。
            min_score（float，默认 0.0）：得分**严格大于**该值才保留。默认 0
                等价于不过滤（命中块的得分恒为正）。

        返回：
            List[Hit] —— 按 `score` 降序排列，同分时按 `index` 升序；长度
            不超过 `k`。知识库为空、查询为空、或没有任何 token 命中时返回 `[]`。

        执行流程：
            1. 索引脏就重建一次（O(总 token 数)，之后命中缓存）
            2. 查询 token 去重后逐个查倒排表，按 BM25（k1=1.5、b=0.75）累加：
               得分 = idf × 权重 × 词频 × (k1+1) / (词频 + k1×(1-b+b×块长/平均块长))
            3. 单字 token 的权重乘 0.25（见 `token_weight`）；几乎每块都出现的
               token 其 idf 趋近 0，自然被压低

        注意：
            纯词法检索，不做同义改写、不支持跨语言查询。要语义检索就替换本
            方法（保持签名不变），工作流层无需改动。
        '''
        total = len(self.chunks)
        if not total or not query:
            return []
        # `k <= 0` 直接当空结果：负数交给切片会变成「切掉末尾几块」，
        # 既反直觉又让调用方难以察觉（0 则本来就是空）。
        if k <= 0:
            return []

        self._ensure_index()
        average = (sum(self._lengths) / total) or 1.0
        scores: Dict[int, float] = {}

        for token in set(tokenize(query)):
            postings = self._index.get(token)
            if not postings:
                continue
            weight = token_weight(token)
            # 停用词式的 token（几乎每块都有）idf 会趋近 0，自动被压下去
            idf = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings.items():
                length = self._lengths[position] or 1
                denominator = frequency + _K1 * (1 - _B + _B * length / average)
                scores[position] = scores.get(position, 0.0) + (
                    idf * weight * frequency * (_K1 + 1) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            Hit(
                text=self.chunks[position],
                source=self.sources[position],
                score=score,
                index=position,
            )
            for position, score in ranked
            if score > min_score
        ][:k]

    def search_text(self, query: str, k: int = 3) -> str:
        '''检索并把结果拼成能直接贴进提示词的文本。

        参数：
            query（str）：查询串；同时用于「一条都没查到」时的提示文案。
            k（int，默认 3）：最多取几条，语义同 `search`。

        返回：
            str —— 有命中时是带编号的资料块（格式见 `format`）；无命中时是
            `（知识库里没有和「query」相关的内容）`。**永远不会返回空串**。

        注意：
            不抛异常：检索不到也返回可读文本，所以适合直接塞进 prompt，
            或作为工具的返回值 —— 模型能看懂「没查到」而不是拿到空字符串。
        '''
        return self.format(self.search(query, k=k), query=query)

    # ---------------- 拼提示词 ----------------
    @staticmethod
    def format(hits: Sequence[Any], query: str = "") -> str:
        '''把检索结果拼成带编号的资料块文本。

        参数：
            hits（Sequence[Hit | dict]）：检索结果。元素可以是 `Hit`，也可以是
                含 `text` / `source` / `score` 键的 dict（便于处理 JSON 化的
                历史结果）；缺少的键按空串 / `None` 处理。
            query（str，默认 ""）：**只**用于「一条都没有」时的提示文案；
                传入时文案会带上查询词，便于模型理解上下文。

        返回：
            str —— 每条一段、段间空一行：

                [1] 来源：faq.md 相关度：0.83
                原文片段……

            `来源` 为空时该段省略、`score` 不是数字时省略；`hits` 为空时返回
            一句占位说明（不会返回空串）。

        注意：
            只做排版，不截断、不高亮、不排序（顺序即传入顺序，编号从 1 起）。
        '''
        if not hits:
            if query:
                return f"（知识库里没有和「{query}」相关的内容）"
            return "（知识库里没有相关内容）"

        lines = []
        for number, hit in enumerate(hits, start=1):
            if isinstance(hit, dict):
                text = hit.get("text", "")
                source = hit.get("source", "")
                score = hit.get("score")
            else:
                text, source, score = hit.text, hit.source, hit.score
            where = f" 来源：{source}" if source else ""
            shown = f" 相关度：{score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(f"[{number}]{where}{shown}\n{text}")
        return "\n\n".join(lines)

    # ---------------- 当工具用 ----------------
    def as_tool(
        self,
        name: str = "search_knowledge_base",
        description: Optional[str] = None,
        k: int = 3,
    ) -> Tool:
        '''把检索包成 `Tool`，让模型自己决定什么时候查、查什么。

        参数：
            name（str，默认 "search_knowledge_base"）：工具名，模型按这个名字
                调用；同一批工具里不要重名。
            description（str | None，默认 None）：工具描述，模型据此判断该不该
                用。`None` 时用内置文案，强调「必须基于返回片段回答、要说明用了
                哪一段、查不到就直说不知道」。
            k（int，默认 3）：模型不传 `top_k` 时使用的默认条数。

        返回：
            Tool —— 入参为 `query: str`（必填）和 `top_k: int = k`（可选），
            返回 `search_text(query, k=top_k)` 得到的字符串。

        注意：
            - 模型传的 `top_k` 会被钳制到 `[1, 20]`，避免一次把提示词灌爆。
            - 工具闭包持有 `self`，检索用的是同一份索引；但工具本身不缓存结果，
              同一个问题问两次会算两遍。
            - 内层函数的 docstring 就是模型的工具说明，改动文案会直接影响
              模型的调用行为。
        '''

        def search_knowledge_base(query: str, top_k: int = k) -> str:
            '''在知识库里检索和 query 相关的资料。返回带编号的原文片段，请基于它们回答。'''
            return self.search_text(query, k=max(1, min(int(top_k), 20)))

        return Tool.from_function(
            search_knowledge_base,
            name=name,
            description=description
            or (
                "在知识库里检索和 query 相关的资料。"
                "返回带编号的原文片段，回答时必须基于这些片段，"
                "并说明用到了哪一段；查不到就直说不知道。"
            ),
        )

    # ---------------- 持久化 ----------------
    def to_dict(self, include_index: bool = False) -> Dict[str, Any]:
        '''组装成可 JSON 序列化的字典（`save` 的内部表示）。

        参数：
            include_index（bool，默认 False）：True 时把倒排索引与块长度表
                一起写进结果（键为 `index` / `lengths`），下次 `load` 可省掉
                一次全量分词；代价是文件体积大致翻倍，且索引结构随版本变化
                的可能性更大。正常使用保持 False。

        返回：
            dict —— 键：`format`（固定 `"promptkit-knowledge"`）、`version`
            （`FORMAT_VERSION`）、`max_chars`、`overlap`、`chunks`、`sources`、
            `ingested`；`include_index=True` 时另有 `index`、`lengths`。

        注意：
            调用它会**顺带确保索引是最新的**（内部 `_ensure_index`），即使
            `include_index=False` —— 频繁序列化大库时这点开销要算进去。
        '''
        self._ensure_index()
        payload: Dict[str, Any] = {
            "format": _FORMAT_NAME,
            "version": FORMAT_VERSION,
            "max_chars": self.max_chars,
            "overlap": self.overlap,
            "chunks": self.chunks,
            "sources": self.sources,
            "ingested": self.ingested,
        }
        if include_index:
            payload["index"] = self._index
            payload["lengths"] = self._lengths
        return payload

    def save(self, path: str, include_index: bool = False) -> None:
        '''存到磁盘。**原子写**：先写同目录临时文件，再改名覆盖目标。

        参数：
            path（str）：目标文件路径。父目录不存在会自动创建。
            include_index（bool，默认 False）：是否连索引一起存，语义见
                `to_dict`。

        返回：
            None

        异常：
            PermissionError / OSError：目录不可写、磁盘满、改名重试耗尽等。
            TypeError / ValueError：内容无法 JSON 序列化（正常不会发生）。

        注意：
            - 直接往目标文件写，中途崩了会留下半截 JSON、`load` 直接失败；
              这里改用 `os.replace`，同盘上原子 —— 要么旧文件完好，要么
              完整新文件。
            - 写失败时会删掉临时文件再抛原异常，旧文件不受影响。
            - 目标目录里以 `.kb-` 开头、`.tmp` 结尾的文件是过程产物；进程被
              强杀时可能残留，可以安全删除。
            - 文本按 UTF-8 写出且 `ensure_ascii=False`，中文在文件里可读。
        '''
        folder = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(folder, exist_ok=True)
        payload = self.to_dict(include_index=include_index)

        handle, temp = tempfile.mkstemp(dir=folder, prefix=".kb-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            _replace_with_retry(temp, path)
        except BaseException:
            if os.path.exists(temp):
                os.remove(temp)
            raise

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "KnowledgeBase":
        '''从磁盘加载知识库。

        参数：
            path（str）：`save` 写出的文件路径（UTF-8 JSON）。
            **kwargs：覆盖文件里记录的构造参数（目前只支持 `max_chars` /
                `overlap`）；不传就用文件里的值，文件里没有则用类默认值。
                传 `KnowledgeBase` 不认识的键会抛 `TypeError`。

        返回：
            KnowledgeBase —— 新实例。文件里存了索引就直接复用（免重新分词）；
            没存、或索引条数与块数对不上，就在首次检索时重建。

        异常：
            FileNotFoundError：文件不存在。
            json.JSONDecodeError：文件不是合法 JSON（如被手动改坏 / 写到一半）。
            ValueError：文件版本比当前 `FORMAT_VERSION` 新（提示升级或重建）。
            TypeError：`kwargs` 里有不支持的键。

        注意：
            - 没有 `version` 字段的**旧文件**照常能读（按老格式处理）。
            - `ingested` 台账会一起恢复，所以 load 之后可以直接继续 `sync_dir`。
            - 这里不校验 `chunks` 的元素类型，文件被手工改坏成非字符串时，
              错误会在分词阶段才暴露。
        '''
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        version = data.get("version")
        if isinstance(version, int) and version > FORMAT_VERSION:
            raise ValueError(
                f"这个知识库文件是 v{version} 写的，当前只认到 "
                f"v{FORMAT_VERSION}：请升级 promptkit 或重新构建"
            )

        settings = dict(kwargs)
        settings.setdefault("max_chars", data.get("max_chars", 400))
        settings.setdefault("overlap", data.get("overlap", 60))

        kb = cls(
            chunks=list(data.get("chunks", [])),
            sources=list(data.get("sources", [])),
            **settings,
        )
        kb.ingested = dict(data.get("ingested", {}))

        # 存了索引就直接用，省掉一次全量分词
        raw_index = data.get("index")
        if raw_index:
            kb._index = {
                token: {int(position): count for position, count in postings.items()}
                for token, postings in raw_index.items()
            }
            kb._lengths = [int(n) for n in data.get("lengths", [])]
            kb._dirty = len(kb._lengths) != len(kb.chunks)
        return kb

    # 旧名字，保留可用。注意：它是 `to_dict` 的**别名**，返回 dict，不是 JSON 字符串
    dumps = to_dict

    def stats(self) -> Dict[str, Any]:
        '''给 CLI / 日志看的概况。

        返回：
            Dict[str, Any] —— 七个键：
            `chunks`（块数）、`sources`（去重后的非空来源数）、
            `tokens`（倒排表里的唯一 token 数）、`characters`（所有块字符数之和，
            因为块有重叠，会大于原始文本长度）、`max_chars`、`overlap`、
            `ingested`（台账条目数）。

        注意：
            会先确保索引最新，首次调用可能触发一次全量分词；没有缓存。
        '''
        self._ensure_index()
        return {
            "chunks": len(self.chunks),
            "sources": len({s for s in self.sources if s}),
            "tokens": len(self._index),
            "characters": sum(len(c) for c in self.chunks),
            "max_chars": self.max_chars,
            "overlap": self.overlap,
            "ingested": len(self.ingested),
        }

    # ---------------- 杂项 ----------------
    def __len__(self) -> int:
        '''块的数量，`len(kb)`。'''
        return len(self.chunks)

    def __iter__(self) -> Iterator[str]:
        '''按顺序迭代块文本（只有正文，不含来源与得分）。'''
        return iter(self.chunks)

    def __repr__(self) -> str:
        '''`<KnowledgeBase chunks=N sources=M>`，`sources` 为去重后的非空来源数。'''
        unique = len({s for s in self.sources if s})
        return f"<KnowledgeBase chunks={len(self.chunks)} sources={unique}>"


def KNOWLEDGE_REFERENCE(  # noqa: N802 - 跟 DSL 命名（工厂，返回 Tool）
    source: Any,
    *,
    name: str = "search_knowledge_base",
    description: Optional[str] = None,
    k: int = 3,
    **kwargs: Any,
) -> Tool:
    '''把「目录 / 单文件 / KnowledgeBase / 现成 Tool」统一成一个检索工具。

    一行把知识库接进聊天：

        from promptkit.knowledge import KNOWLEDGE_REFERENCE
        from promptkit.default_tools import WEB_SEARCH, WRITE_FILE

        Tools(WEB_SEARCH, KNOWLEDGE_REFERENCE("path/to/knowledge"), WRITE_FILE)

    参数：
        source（Any）：决定怎么建库，四种情况：
            - `str` / `PathLike` 且是**目录** → `KnowledgeBase.from_dir(...)`，
              只收第一层、后缀 md / txt 的文件（不递归）
            - `str` / `PathLike` 且是**文件** → 整篇作为一段文本灌进去，
              不检查扩展名，来源名取文件名
            - `KnowledgeBase` → 直接 `.as_tool()`
            - `Tool` → **原样返回**，此时 `name` / `description` / `k` /
              `kwargs` 全部被忽略
        name（str，默认 "search_knowledge_base"）：工具名，语义见 `as_tool`。
        description（str | None，默认 None）：工具描述；`None` 时用内置文案
            （强调必须基于片段回答、要说明用哪一段、查不到就说不知道）。
        k（int，默认 3）：模型不传 `top_k` 时的默认返回条数。
        **kwargs：仅当 `source` 是路径时**透传给 `KnowledgeBase(...)`**
            （如 `max_chars` / `overlap`）；传 `KnowledgeBase` 或 `Tool` 时忽略。

    返回：
        Tool —— 可直接放进 `Tools(...)` / `LLMStep(tools=[...])` 的检索工具，
        行为与 `KnowledgeBase.as_tool()` 完全一致。

    异常：
        TypeError：`source` 既不是路径、也不是 `KnowledgeBase` / `Tool`。
        FileNotFoundError：路径不存在。
        IsADirectoryError：目录被当作文件读取（一般不会发生，目录分支已先拦截）。
        ValueError：`max_chars <= 0` 等构造参数非法，在建库分块时抛出。

    注意：
        - 传路径时**在本次调用内同步建库**：目录越大越慢，且每次调用都会
          重新分词、不跨调用缓存。大库建议自己建好并 `save`，再用
          `KnowledgeBase.load(...)` 传进来。
        - 需要增量更新时不要用路径形式，自己持有 `KnowledgeBase` 调
          `sync_dir`，再把这个实例传进来。
    '''
    if isinstance(source, Tool):
        return source
    if isinstance(source, KnowledgeBase):
        return source.as_tool(name=name, description=description, k=k)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if os.path.isdir(path):
            kb = KnowledgeBase.from_dir(path, **kwargs)
        else:
            kb = KnowledgeBase(**kwargs)
            kb.add_file(path)
        return kb.as_tool(name=name, description=description, k=k)
    raise TypeError(
        "KNOWLEDGE_REFERENCE 需要目录/文件路径、KnowledgeBase 或 Tool，"
        f"得到 {type(source).__name__}"
    )


__all__ = [
    "KnowledgeBase",
    "Hit",
    "chunk_text",
    "tokenize",
    "token_weight",
    "KNOWLEDGE_REFERENCE",
    "FORMAT_VERSION",
]
