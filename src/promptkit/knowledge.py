'''知识库：分块 + 检索 + 拼进提示词。

⚠️ **实验性功能，默认关闭。** 用之前先打开：

    from promptkit import features
    features.enable("knowledge")        # 或设 PROMPTKIT_ENABLE_KNOWLEDGE=1

    from promptkit.knowledge import KnowledgeBase

wf 负责**编排**（什么时候检索、检索不到怎么办、结果怎么进提示词），
这里负责**检索本身**——这是普通 Python，通过 `Call(kb.search, ...)` 接进工作流。

零依赖：用的是 BM25 词法检索，中文按「双字 + 单字」切。
想要语义检索就把 `KnowledgeBase.search` 换成你的向量检索即可，
工作流那一层不用动（这是把检索独立出来的意义）。

    kb = KnowledgeBase.from_dir("docs")
    hits = kb.search("怎么退款", k=3)
    print(kb.format(hits))          # 直接能贴进提示词
    LLMStep(..., tools=[kb.as_tool()])   # 也可以让模型自己决定检索
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
    '''token 的权重：多字的按原样，单字打折。'''
    return 1.0 if len(token) > 1 else _UNIGRAM_WEIGHT


def tokenize(text: str) -> List[str]:
    '''英文按单词切；中文按「双字 + 单字」切。

    中文没有空格，直接 `split()` 会把整句变成一个 token，检索基本失效。
    双字（bigram）是够用的近似：查询「如何退款」切出「如何/何退/退款」，
    能命中资料里的「退款」。

    单字也切（打分时按 `_UNIGRAM_WEIGHT` 打折），用来兜住 bigram 对不齐的情况：
    查询「还能退吗」的 `退吗` 匹配不上「退款」，但单字 `退` 能。
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
    '''把长文本切成块：先按空行分段，太长的段再按长度切（带重叠）。

    重叠是为了避免答案正好被切断在块边界上。
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
    '''一条检索结果。'''

    text: str
    source: str = ""
    score: float = 0.0
    index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": round(self.score, 4),
            "index": self.index,
        }

    def __str__(self) -> str:
        where = f"（{self.source}）" if self.source else ""
        return f"{self.text}{where}"


@dataclass
class KnowledgeBase:
    '''一堆文本块 + 一张倒排索引。

    可以存到磁盘、下次直接加载（见 `save` / `load`），
    也可以对着一个文档目录做**增量**同步（见 `sync_dir`）——
    只处理新增或改动过的文件，重跑不会重复灌。
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
        if len(self.sources) < len(self.chunks):
            self.sources = list(self.sources) + [""] * (
                len(self.chunks) - len(self.sources)
            )

    # ---------------- 索引 ----------------
    def _rebuild(self) -> None:
        '''重建倒排索引（批量灌完之后做一次，比每块都更新更划算）。'''
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
        if self._dirty or len(self._lengths) != len(self.chunks):
            self._rebuild()

    # ---------------- 建库 ----------------
    def add(self, text: str, source: str = "") -> "KnowledgeBase":
        '''加一段文本（会按 max_chars 自动分块）。'''
        for chunk in chunk_text(text, self.max_chars, self.overlap):
            self.chunks.append(chunk)
            self.sources.append(source)
        self._dirty = True
        return self

    def remove_source(self, source: str) -> int:
        '''删掉某个来源的所有块，返回删了几块。'''
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
        with open(path, encoding=encoding, errors="replace") as handle:
            return self.add(handle.read(), source=os.path.basename(path))

    def add_dir(self, directory: str, pattern: str = r".*\.(md|txt)$") -> "KnowledgeBase":
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

        按「修改时间 + 大小」判断文件有没有变过，所以重跑是幂等的：
        没变的文件直接跳过，不会重复灌进去。

        - `force=True`：无视台账，全部重灌
        - `prune=True`：磁盘上已经删掉的文件，也从知识库里移除
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
        kb = cls(**kwargs)
        for text in texts:
            kb.add(text)
        return kb

    @classmethod
    def from_dir(cls, directory: str, **kwargs: Any) -> "KnowledgeBase":
        return cls(**kwargs).add_dir(directory)

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> List[Hit]:
        '''BM25 检索，返回得分最高的 k 条。'''
        total = len(self.chunks)
        if not total or not query:
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
        '''检索并直接返回能贴进提示词的文本（检索不到会说明原因）。'''
        return self.format(self.search(query, k=k), query=query)

    # ---------------- 拼提示词 ----------------
    @staticmethod
    def format(hits: Sequence[Any], query: str = "") -> str:
        '''把检索结果拼成带编号的资料块。'''
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
        '''包成 Tool，让模型自己决定什么时候检索。'''

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
        '''序列化成可 JSON 的字典。

        `include_index=True` 会把倒排索引一起存下来，加载时免去重新分词；
        代价是文件大致翻倍。默认不存（文件更小、格式更稳）。
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
        '''存到磁盘。**原子写**：先写临时文件再改名。

        直接往目标文件写的话，中途崩了会留下一个半截的 JSON，
        下次 `load` 直接报错、知识库就废了。`os.replace` 在同一个
        文件系统上是原子的——要么是旧文件，要么是完整的新文件。
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
        '''从磁盘加载。旧的（没有 version 字段的）文件也能读。'''
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

    # 旧名字，保留可用
    dumps = to_dict

    def stats(self) -> Dict[str, Any]:
        '''给 CLI / 日志看的概况。'''
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
        return len(self.chunks)

    def __iter__(self) -> Iterator[str]:
        return iter(self.chunks)

    def __repr__(self) -> str:
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
    '''把「目录路径 / 单文件路径 / KnowledgeBase / 现成 Tool」统一成一个检索工具。

    一行把知识库接进聊天：

        from promptkit.knowledge import KNOWLEDGE_REFERENCE
        from promptkit.default_tools import WEB_SEARCH, WRITE_FILE

        Tools(WEB_SEARCH, KNOWLEDGE_REFERENCE("path/to/knowledge"), WRITE_FILE)

    - 传目录 → `KnowledgeBase.from_dir(path)`（默认只收 md / txt）
    - 传文件 → 整个文件作为一段文本灌进去
    - 传 `KnowledgeBase` → 直接 `.as_tool()`
    - 传 `Tool` → 原样返回

    `kwargs` 透传给 `KnowledgeBase` 构造（如 `max_chars` / `overlap`）。
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
