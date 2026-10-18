'''网络搜索工具。

后端可插拔，`WebSearch()` 按环境变量自动挑：

| 后端 | 需要 | 说明 |
| --- | --- | --- |
| 博查 Bocha | `BOCHA_API_KEY` | **国内首选**：专为 LLM 做的中文搜索 API |
| 智谱 | `ZHIPUAI_API_KEY` | 智谱的 web_search API |
| Tavily | `TAVILY_API_KEY` | 英文内容质量最好 |
| Serper | `SERPER_API_KEY` | Google 结果 |
| Brave | `BRAVE_API_KEY` | Brave Search API |
| 百度 | 无 | 免密钥。抓 `baidu.com/s`，**尽力而为** |
| Bing 中国 | 无 | 免密钥。抓 `cn.bing.com`，免密钥里最稳的 |
| DuckDuckGo | 无 | 免密钥。国内通常访问不了 |

没配任何 Key 时走**降级链**：百度 → Bing 中国 → DuckDuckGo，
第一个成功就用。想换顺序：

    PROMPTKIT_SEARCH_BACKENDS=bing,baidu      # 或者只留一个

**零依赖**：只用标准库 `urllib` + `html.parser`。

当工具用：

    from promptkit.default_tools.web_search import WEB_SEARCH
    LLMStep(prompt=..., tools=[WEB_SEARCH], output_key="answer")

自己用：

    search = WebSearch()
    results = search.search("今天北京天气", k=3)
    print(search.format(results))

⚠️ 两点提醒：

1. 搜索结果会被塞进提示词 —— 网页内容**不可信**，可能包含"忽略之前的指令"
   这类注入文本。别把搜索结果直接当指令执行。
2. 免密钥的几个后端都是**抓页面**解析的，对方改版或限流就会失效；
   要稳定请配一个 API Key。
'''
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional

from promptkit.tools import Tool

DEFAULT_K = 5
MAX_K = 20
DEFAULT_TIMEOUT = 15.0

#: 免密钥时依次尝试的后端（可用 PROMPTKIT_SEARCH_BACKENDS 覆盖）
DEFAULT_CHAIN = ("baidu", "bing", "duckduckgo")

#: 有些站点会直接拒绝空 User-Agent
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

#: 搜索引擎更愿意给中文结果
ZH_HEADERS = {"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}

#: 百度对「裸」请求会直接返回一个 1.4KB 的安全验证页（里面一个结果都没有）；
#: 带上这些头它才当你是浏览器，给完整的搜索结果页。
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.baidu.com/",
    "Upgrade-Insecure-Requests": "1",
}


class SearchError(RuntimeError):
    '''搜索失败（网络、超时、鉴权、对方改版……）。'''


# ============================ 数据结构 ============================
@dataclass
class SearchResult:
    '''一条搜索结果。'''

    title: str = ""
    url: str = ""
    snippet: str = ""
    #: 这条结果来自哪个后端
    source: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"{self.title} <{self.url}>"


# ============================ 文本 / 字段工具 ============================
def clean_text(text: Any) -> str:
    '''去标签残留 / HTML 实体 / 多余空白。'''
    if text is None:
        return ""
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clamp_k(value: Any, default: int = DEFAULT_K) -> int:
    '''把模型给的 top_k 收进合理范围，越界不报错、直接夹住。'''
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, MAX_K))


def dedupe(results: List[SearchResult]) -> List[SearchResult]:
    '''按 URL 去重，保持原顺序。'''
    seen = set()
    unique = []
    for result in results:
        key = result.url.rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def dig(data: Any, *paths: str, default: Any = None) -> Any:
    '''按候选路径取值，兼容不同厂商的响应结构。

        dig(data, "data.webPages.value", "webPages.value", "results")

    取不到就继续试下一条，全都不行返回 default。
    '''
    for path in paths:
        node = data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                node = None
                break
        if node:
            return node
    return default


def pick_field(item: Dict[str, Any], *names: str) -> str:
    '''按候选字段名取值（不同厂商叫法不一样）。'''
    if not isinstance(item, dict):
        return ""
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def unwrap_url(href: str, base: str = "https://duckduckgo.com") -> str:
    '''把搜索页的跳转链接还原成真实地址。

    DuckDuckGo 的结果链接长这样：

        //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=...

    真实地址在 `uddg` 参数里；没有这个参数就当普通相对链接处理。
    '''
    if not href:
        return ""
    href = unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = base + href

    if "uddg=" in href:
        query = urllib.parse.urlparse(href).query
        target = urllib.parse.parse_qs(query).get("uddg", [])
        if target and target[0]:
            return target[0]
    return href


# ============================ HTTP ============================
def http_get(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    '''GET 一个 URL，返回文本。

    刻意不发 `Accept-Encoding`：urllib 不会自动解压，
    要了 gzip 反而会拿到一堆二进制。
    '''
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise SearchError(f"HTTP {exc.code}：{exc.reason}（{url}）") from None
    except urllib.error.URLError as exc:
        raise SearchError(f"连接失败：{exc.reason}（{url}）") from None
    except TimeoutError:
        raise SearchError(f"请求超时（>{timeout}s）：{url}") from None
    except OSError as exc:
        raise SearchError(f"网络错误：{exc}（{url}）") from None

    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    '''POST 一段 JSON，解析 JSON 响应。'''
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001 - 读不到就算了
            pass
        raise SearchError(f"HTTP {exc.code}：{exc.reason} {detail}".strip()) from None
    except urllib.error.URLError as exc:
        raise SearchError(f"连接失败：{exc.reason}（{url}）") from None
    except TimeoutError:
        raise SearchError(f"请求超时（>{timeout}s）：{url}") from None
    except OSError as exc:
        raise SearchError(f"网络错误：{exc}（{url}）") from None

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SearchError(f"响应不是合法 JSON：{exc}") from None


# ============================ 后端基类 ============================
class SearchBackend(ABC):
    '''搜索后端：给一个查询，返回若干结果。'''

    #: 展示用的名字
    name: str = "backend"

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    @abstractmethod
    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        '''执行搜索；失败时抛 SearchError。'''

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FallbackBackend(SearchBackend):
    '''依次尝试多个后端，第一个拿到结果的胜出。

    免密钥那几个都是抓页面的，单独用随时可能失效；串起来用稳得多。
    '''

    name = "fallback"

    def __init__(self, backends: Iterable[SearchBackend], **kwargs: Any):
        super().__init__(**kwargs)
        self.backends = list(backends)
        if not self.backends:
            raise ValueError("FallbackBackend 至少要有一个后端")

    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        failures = []
        for backend in self.backends:
            try:
                results = backend.search(query, k=k)
            except SearchError as exc:
                failures.append(f"{backend.name}: {exc}")
                continue
            if results:
                return results
            failures.append(f"{backend.name}: 没有结果")
        raise SearchError(
            "所有后端都失败了 —— " + "；".join(failures)
        )

    def __repr__(self) -> str:
        return f"FallbackBackend({[b.name for b in self.backends]})"


# ============================ 免密钥：抓页面 ============================
class _OrderedParser(HTMLParser):
    '''按出现顺序记两类东西：结果链接、以及链接之间的正文。

    抓页面的几个后端结构差别很大，但「标题链接 + 后面跟着摘要」这个模式是共通的。
    刻意不依赖具体 class 名，只靠顺序配对 —— 对方改版时最坏是解析不到东西
    （返回空列表），而不是吐出一堆垃圾。
    '''

    #: 结果链接所在的标签（比如百度的 h3 里才是结果，导航链接不在里面）
    link_scope = ("h3",)
    #: 这些标签里的文字不算正文
    skip_tags = ("script", "style", "noscript")
    #: 块级标签 —— 到边界就把攒的正文切出来。
    #: 不切的话，最后一条结果的摘要会和页脚导航的文本粘成一段。
    block_tags = (
        "div", "p", "li", "td", "tr", "table", "section", "article", "br",
        "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dl", "dt", "dd",
        "blockquote", "pre", "form", "header", "footer", "nav", "main",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: List[Dict[str, str]] = []
        self._depth = 0
        self._skip = 0
        self._in_link = False
        self._href = ""
        self._title: List[str] = []
        self._buffer: List[str] = []

    # ---- 内部 ----
    def _flush_text(self) -> None:
        text = clean_text("".join(self._buffer))
        self._buffer = []
        if text:
            self.items.append({"kind": "text", "text": text})

    # ---- HTMLParser ----
    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}

        if tag in self.skip_tags:
            self._skip += 1
            return
        if self._skip:
            return

        if tag in self.link_scope:
            self._flush_text()
            self._depth += 1
        elif tag == "a" and self._depth and not self._in_link:
            href = attributes.get("href", "")
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self._flush_text()
                self._in_link = True
                self._href = href
                self._title = []
        elif tag in self.block_tags:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skip_tags:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return

        if tag == "a" and self._in_link:
            self._in_link = False
            self.items.append(
                {
                    "kind": "link",
                    "href": self._href,
                    "text": "".join(self._title),
                }
            )
        elif tag in self.link_scope and self._depth:
            self._flush_text()
            self._depth -= 1
        elif tag in self.block_tags:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_link:
            self._title.append(data)
        elif not self._depth:
            self._buffer.append(data)

    def close(self) -> None:  # noqa: D102 - 收尾
        super().close()
        self._flush_text()


def pair_links_with_text(
    items: List[Dict[str, str]], limit: int = DEFAULT_K
) -> List[SearchResult]:
    '''把「链接 + 紧随其后的正文」配成结果（纯函数）。'''
    results: List[SearchResult] = []
    pending: Optional[SearchResult] = None

    for item in items:
        if item["kind"] == "link":
            if pending is not None:
                results.append(pending)
            pending = SearchResult(title=clean_text(item["text"]), url=item["href"])
        elif item["kind"] == "text" and pending is not None and not pending.snippet:
            pending.snippet = item["text"]
    if pending is not None:
        results.append(pending)

    return results[:limit]


# ---------- 百度 ----------
_BAIDU_TARGET_PATTERNS = (
    # <meta http-equiv="refresh" content="0;url=https://real">
    re.compile(
        r"""http-equiv=["']?refresh["']?[^>]*content=["']?\s*\d+\s*;\s*url=([^"'>\s]+)""",
        re.IGNORECASE,
    ),
    # window.location.replace("https://real") / window.location.href = "..."
    re.compile(
        r"""window\.location(?:\.replace\s*\(|\.href\s*=\s*)["']([^"']+)["']""",
        re.IGNORECASE,
    ),
    # 兜底：URL=... 这种老写法
    re.compile(r"""\bURL=([^"'>\s]+)"""),
)


def extract_baidu_target(html: str) -> str:
    '''从百度 `/link?url=...` 的中转页里抠出真实地址。

    百度把结果链接都包成了自己的跳转页，真实地址在跳转页里
    （meta refresh 或 JS 跳转）。抠不到就返回空串（纯函数，可离线测试）。
    '''
    for pattern in _BAIDU_TARGET_PATTERNS:
        match = pattern.search(html or "")
        if match:
            return unescape(match.group(1)).strip()
    return ""


def resolve_baidu_target(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    '''跟随百度的 `/link?url=...`，拿到真实地址。

    两种机制都要处理：

    1. 百度直接回 **302** —— urllib 会自动跟过去，最终地址在 `response.url`
    2. 百度回一个 **HTML 跳转页** —— 地址藏在 meta refresh / JS 里

    解析不出来就返回空串（调用方保留原链接）。
    '''
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **BROWSER_HEADERS}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final = response.url or ""
            body = response.read(200_000)
    except (urllib.error.URLError, OSError, TimeoutError):
        return ""

    # 情况 1：真的跳到了别处
    if final and "baidu.com/link" not in final and final != url:
        return final

    # 情况 2：停在跳转页上，从页面里抠
    return extract_baidu_target(body.decode("utf-8", errors="replace"))


def parse_baidu(html: str, limit: int = DEFAULT_K) -> List[SearchResult]:
    '''解析百度搜索页（纯函数）。'''
    parser = _OrderedParser()
    parser.feed(html or "")
    parser.close()

    results = []
    for item in pair_links_with_text(parser.items, limit=limit * 3):
        href = item.url
        # 只要百度自己的跳转链接，或者普通外链；排掉导航 / 站内链接
        if "/link?url=" in href:
            item.url = urllib.parse.urljoin("https://www.baidu.com", href)
            item.source = "baidu"
            results.append(item)
        elif href.startswith("http") and "baidu.com" not in href:
            item.source = "baidu"
            results.append(item)
    return dedupe(results)[:limit]


class BaiduBackend(SearchBackend):
    '''免密钥：抓 `baidu.com/s`。

    ⚠️ 两个已知问题：

    1. 百度**没有公开的搜索 API**，这是抓页面；它会限流、要求验证码。
       实测：连着抓几次就会被拦，返回一个 1.4KB 的"安全验证"页 —— 那时会
       解析不到东西并明确报错。**别把它当稳定后端用。**
       另外裸请求一定会被拦，所以这里带了 `BROWSER_HEADERS`。
    2. 百度把结果链接包成了 `/link?url=...` 的跳转页，默认拿到的 `url` 是
       百度的中转地址（点开能到，但不是原始地址）。想要真实地址就设
       `resolve_urls=True`，代价是每条结果多一次请求。
    '''

    name = "baidu"
    ENDPOINT = "https://www.baidu.com/s"

    def __init__(self, *, resolve_urls: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.resolve_urls = resolve_urls

    def _resolve(self, url: str) -> str:
        return resolve_baidu_target(url, timeout=self.timeout) or url

    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        url = self.ENDPOINT + "?" + urllib.parse.urlencode(
            {"ie": "utf-8", "wd": query, "rn": clamp_k(k)}
        )
        html = http_get(url, timeout=self.timeout, headers=BROWSER_HEADERS)
        results = parse_baidu(html, limit=k)
        if not results:
            raise SearchError(
                "百度没有解析出任何结果（可能被限流、要求验证码，或真的没有结果）。"
                "换个后端更稳：PROMPTKIT_SEARCH_BACKENDS=bing，"
                "或配一个 API Key（BOCHA_API_KEY / ZHIPUAI_API_KEY）"
            )
        if self.resolve_urls:
            for item in results:
                item.url = self._resolve(item.url)
        return results


# ---------- Bing 中国 ----------
class _BingParser(HTMLParser):
    '''解析 Bing 的 `li.b_algo` 结果块。

    一个块大概是（**顺序很重要**）：

        <li class="b_algo">
          <div class="b_tpcn"><a class="tilk" href="真实地址">图标 + 域名 + 面包屑</a></div>
          <h2><a href="真实地址">标题</a></h2>
          <div class="b_caption"><p>摘要</p></div>
        </li>

    注意第一个 `<a>` 是**来源归属**（favicon + 域名 + 面包屑），不是标题 ——
    照「块里第一个链接」来取，标题就会变成 "python.orghttps://docs.python.org › …"。
    **标题必须取 `<h2>` 里那个链接。**
    '''

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._block: Optional[Dict[str, str]] = None
        self._nested = 0
        self._skip = 0
        self._h2 = 0
        self._in_title = False
        self._in_snippet = False

    def _start_block(self) -> None:
        self._block = {"title": "", "url": "", "snippet": ""}
        self._nested = 0
        self._h2 = 0
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1
            return
        if self._skip:
            return

        attributes = {k.lower(): (v or "") for k, v in attrs}

        if self._block is None:
            if tag == "li" and "b_algo" in attributes.get("class", "").lower():
                self._start_block()
            return

        if tag == "li":
            self._nested += 1
        elif tag == "h2":
            self._h2 += 1
        elif tag == "a":
            href = attributes.get("href", "")
            if href.startswith("http"):
                if not self._block["url"]:
                    # 兜底：万一没有 h2，至少还有归属链接的地址可用
                    self._block["url"] = href
                if self._h2 and not self._in_title:
                    # 标题在 h2 里，用它覆盖兜底的地址
                    self._block["url"] = href
                    self._in_title = True
        elif tag == "p" and not self._in_snippet:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return

        if tag == "h2":
            self._h2 = max(0, self._h2 - 1)
        elif tag == "a":
            self._in_title = False
        elif tag == "p":
            self._in_snippet = False
        elif tag == "li" and self._block is not None:
            if self._nested:
                self._nested -= 1
            else:
                self.results.append(self._block)
                self._block = None

    def handle_data(self, data: str) -> None:
        if self._skip or self._block is None:
            return
        if self._in_title:
            self._block["title"] += data
        elif self._in_snippet and len(self._block["snippet"]) < 500:
            self._block["snippet"] += data


def parse_bing(html: str, limit: int = DEFAULT_K) -> List[SearchResult]:
    '''解析 cn.bing.com 的搜索页（纯函数）。'''
    parser = _BingParser()
    parser.feed(html or "")
    parser.close()
    return [
        SearchResult(
            title=clean_text(block["title"]),
            url=block["url"],
            snippet=clean_text(block["snippet"]),
            source="bing",
        )
        for block in parser.results
        if block["url"]
    ][:limit]


class BingCnBackend(SearchBackend):
    '''免密钥：抓 `cn.bing.com`。

    免密钥的几个里最稳的一个 —— 结构规矩、给的是真实地址、对中文支持也不错。
    百度抓不到时会自动降级到这里。
    '''

    name = "bing"
    ENDPOINT = "https://cn.bing.com/search"

    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        url = self.ENDPOINT + "?" + urllib.parse.urlencode(
            {"q": query, "count": clamp_k(k)}
        )
        html = http_get(url, timeout=self.timeout, headers=ZH_HEADERS)
        results = parse_bing(html, limit=k)
        if not results:
            raise SearchError(
                "Bing 没有解析出任何结果（可能被限流或改版）。"
                "可以试试配一个 API Key（BOCHA_API_KEY / ZHIPUAI_API_KEY）"
            )
        return results


# ---------- DuckDuckGo ----------
class _DDGLiteParser(HTMLParser):
    '''从 `html.duckduckgo.com/lite` 的页面里抠出结果。

    `<a href>` 里带 `uddg=` 的算一条结果，`<td class="...snippet...">` 算摘要。
    '''

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: List[Dict[str, str]] = []
        self._link: Optional[Dict[str, Any]] = None
        self._snippet: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}

        if tag == "a":
            href = attributes.get("href", "")
            if "uddg=" in href:
                self._link = {"href": href, "text": []}
        elif tag == "td":
            classes = attributes.get("class", "").lower()
            if "snippet" in classes:
                self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link["text"].append(data)
        if self._snippet is not None:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link is not None:
            self.items.append(
                {
                    "kind": "link",
                    "href": self._link["href"],
                    "text": "".join(self._link["text"]),
                }
            )
            self._link = None
        elif tag == "td" and self._snippet is not None:
            self.items.append({"kind": "snippet", "text": "".join(self._snippet)})
            self._snippet = None


def parse_ddg_lite(html: str, limit: int = DEFAULT_K) -> List[SearchResult]:
    '''把 DuckDuckGo lite 的 HTML 解析成结果列表（纯函数）。'''
    parser = _DDGLiteParser()
    parser.feed(html or "")

    results: List[SearchResult] = []
    pending: Optional[SearchResult] = None
    for item in parser.items:
        if item["kind"] == "link":
            if pending is not None:
                results.append(pending)
            pending = SearchResult(
                title=clean_text(item["text"]),
                url=unwrap_url(item["href"]),
                source="duckduckgo",
            )
        elif item["kind"] == "snippet" and pending is not None and not pending.snippet:
            pending.snippet = clean_text(item["text"])
    if pending is not None:
        results.append(pending)

    return dedupe([r for r in results if r.url])[:limit]


class DuckDuckGoBackend(SearchBackend):
    '''免密钥：抓 `html.duckduckgo.com/lite`。

    没有官方 API，是尽力而为；**国内通常访问不了**，所以只在
    降级链的最后一位。
    '''

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/lite/"

    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        url = self.ENDPOINT + "?" + urllib.parse.urlencode({"q": query})
        html = http_get(url, timeout=self.timeout)
        results = parse_ddg_lite(html, limit=k)
        if not results:
            raise SearchError(
                "DuckDuckGo 没有解析出任何结果（改版、限流、网络不通，"
                "或者真的没有结果）。国内通常访问不了它，"
                "可以设 PROMPTKIT_SEARCH_BACKENDS=bing 换个后端"
            )
        return results


# ============================ 带 API Key ============================
class _ApiBackend(SearchBackend):
    '''带密钥后端的公共部分。'''

    ENDPOINT = ""
    ENV = ""
    #: 响应里结果列表可能在的位置（按顺序试）
    RESULT_PATHS: tuple = ()
    TITLE_FIELDS: tuple = ("title",)
    URL_FIELDS: tuple = ("url",)
    SNIPPET_FIELDS: tuple = ("snippet",)

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get(self.ENV, "")
        if not self.api_key:
            raise SearchError(f"{self.name} 需要 API Key：设置环境变量 {self.ENV}")

    @abstractmethod
    def _request(self, query: str, k: int) -> Dict[str, Any]:
        '''发请求，返回原始响应。'''

    def search(self, query: str, k: int = DEFAULT_K) -> List[SearchResult]:
        count = clamp_k(k)
        data = self._request(query, count)
        items = dig(data, *self.RESULT_PATHS, default=[]) or []
        if not isinstance(items, list) or not items:
            raise SearchError(
                f"{self.name} 没返回结果；"
                f"响应顶层字段是 {sorted(data)[:8] if isinstance(data, dict) else type(data).__name__}，"
                "如果接口字段名变了，请按实际响应调整 RESULT_PATHS"
            )
        return [
            SearchResult(
                title=pick_field(item, *self.TITLE_FIELDS),
                url=pick_field(item, *self.URL_FIELDS),
                snippet=pick_field(item, *self.SNIPPET_FIELDS),
                source=self.name,
            )
            for item in items[:count]
        ]

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


class BochaBackend(_ApiBackend):
    '''博查（Bocha）：https://bochaai.com

    国内专为 LLM 做的搜索 API，中文结果质量好，返回结构化 JSON。
    **想稳定用国内搜索，首选这个。**
    '''

    name = "bocha"
    ENDPOINT = "https://api.bochaai.com/v1/web-search"
    ENV = "BOCHA_API_KEY"
    RESULT_PATHS = ("data.webPages.value", "webPages.value", "data.results", "results")
    TITLE_FIELDS = ("name", "title")
    URL_FIELDS = ("url", "link")
    SNIPPET_FIELDS = ("summary", "snippet", "content")

    def _request(self, query: str, k: int) -> Dict[str, Any]:
        return http_post_json(
            self.ENDPOINT,
            {"query": query, "count": k, "summary": True},
            timeout=self.timeout,
            headers=self._auth_headers(),
        )


class ZhipuBackend(_ApiBackend):
    '''智谱 AI 的 web_search：https://open.bigmodel.cn

    和博查一样是结构化 JSON，中文结果好。
    '''

    name = "zhipu"
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/web_search"
    ENV = "ZHIPUAI_API_KEY"
    RESULT_PATHS = ("search_result", "search_results", "data.search_result", "results")
    TITLE_FIELDS = ("title", "name")
    URL_FIELDS = ("link", "url")
    SNIPPET_FIELDS = ("content", "snippet", "summary")

    def _request(self, query: str, k: int) -> Dict[str, Any]:
        return http_post_json(
            self.ENDPOINT,
            {"search_engine": "search_std", "search_query": query},
            timeout=self.timeout,
            headers=self._auth_headers(),
        )


class TavilyBackend(_ApiBackend):
    '''Tavily：https://tavily.com（英文内容质量好）。'''

    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"
    ENV = "TAVILY_API_KEY"
    RESULT_PATHS = ("results", "data.results")
    SNIPPET_FIELDS = ("content", "snippet")

    def _request(self, query: str, k: int) -> Dict[str, Any]:
        return http_post_json(
            self.ENDPOINT,
            {"query": query, "max_results": k, "search_depth": "basic"},
            timeout=self.timeout,
            headers=self._auth_headers(),
        )


class SerperBackend(_ApiBackend):
    '''Serper：https://serper.dev（Google 结果）。'''

    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"
    ENV = "SERPER_API_KEY"
    RESULT_PATHS = ("organic", "data.organic")
    URL_FIELDS = ("link", "url")

    def _request(self, query: str, k: int) -> Dict[str, Any]:
        return http_post_json(
            self.ENDPOINT,
            {"q": query, "num": k},
            timeout=self.timeout,
            headers={"X-API-KEY": self.api_key},
        )


class BraveBackend(_ApiBackend):
    '''Brave Search：https://brave.com/search/api/'''

    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    ENV = "BRAVE_API_KEY"
    RESULT_PATHS = ("web.results", "results")
    SNIPPET_FIELDS = ("description", "snippet")

    def _request(self, query: str, k: int) -> Dict[str, Any]:
        url = self.ENDPOINT + "?" + urllib.parse.urlencode({"q": query, "count": k})
        text = http_get(
            url,
            timeout=self.timeout,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SearchError(f"Brave 响应不是合法 JSON：{exc}") from None


#: 环境变量 -> 后端类，**顺序就是优先级**（国内的两个排在前面）
API_BACKENDS: Dict[str, Any] = {
    BochaBackend.ENV: BochaBackend,
    ZhipuBackend.ENV: ZhipuBackend,
    TavilyBackend.ENV: TavilyBackend,
    SerperBackend.ENV: SerperBackend,
    BraveBackend.ENV: BraveBackend,
}

#: 免密钥的后端，按名字索引（给 PROMPTKIT_SEARCH_BACKENDS 用）
KEYLESS_BACKENDS: Dict[str, Any] = {
    "baidu": BaiduBackend,
    "bing": BingCnBackend,
    "duckduckgo": DuckDuckGoBackend,
}


def pick_backend(**kwargs: Any) -> SearchBackend:
    '''有 API Key 就用付费后端，否则按降级链抓页面。

    降级链可以用 `PROMPTKIT_SEARCH_BACKENDS` 覆盖，逗号分隔：

        PROMPTKIT_SEARCH_BACKENDS=bing          # 只用 Bing
        PROMPTKIT_SEARCH_BACKENDS=bing,baidu    # 换个顺序
    '''
    for env, backend in API_BACKENDS.items():
        if os.environ.get(env):
            return backend(**kwargs)

    raw = os.environ.get("PROMPTKIT_SEARCH_BACKENDS", "")
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if not names:
        names = list(DEFAULT_CHAIN)

    unknown = [name for name in names if name not in KEYLESS_BACKENDS]
    if unknown:
        raise SearchError(
            f"PROMPTKIT_SEARCH_BACKENDS 里有不认识的后端 {unknown}；"
            f"可用：{', '.join(sorted(KEYLESS_BACKENDS))}"
        )

    backends = [KEYLESS_BACKENDS[name](**kwargs) for name in names]
    return backends[0] if len(backends) == 1 else FallbackBackend(backends)


# ============================ 门面 ============================
@dataclass
class WebSearch:
    '''搜索门面：选后端、拿结果、拼成能给模型看的文本。

        search = WebSearch()                     # 按环境变量自动选后端
        search = WebSearch(BaiduBackend())       # 或者指定
        search.search("关键词", k=3)
        search.search_text("关键词")              # 直接能贴进提示词
        search.as_tool()                         # 当工具给模型用
    '''

    backend: Optional[SearchBackend] = None
    k: int = DEFAULT_K
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self._resolved: Optional[SearchBackend] = self.backend

    def resolve_backend(self) -> SearchBackend:
        '''延迟决定用哪个后端 —— 这样 import 之后设的环境变量也算数。'''
        if self._resolved is None:
            self._resolved = pick_backend(timeout=self.timeout)
        return self._resolved

    # ---------------- 检索 ----------------
    def search(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        '''搜索；失败抛 SearchError。'''
        query = (query or "").strip()
        if not query:
            return []
        return self.resolve_backend().search(query, k=clamp_k(k or self.k))

    def search_text(self, query: str, k: Optional[int] = None) -> str:
        '''搜索并直接返回能贴进提示词的文本。'''
        return self.format(self.search(query, k=k), query=query)

    @staticmethod
    def format(results: List[SearchResult], query: str = "") -> str:
        '''把结果拼成带编号的文本，方便模型引用来源。'''
        if not results:
            if query:
                return f"（没有搜到和「{query}」相关的结果）"
            return "（没有搜到结果）"

        blocks = []
        for number, item in enumerate(results, start=1):
            lines = [f"[{number}] {item.title or '(无标题)'}", item.url]
            if item.snippet:
                lines.append(item.snippet)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- 当工具用 ----------------
    def as_tool(
        self,
        name: str = "web_search",
        description: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Tool:
        '''包成 Tool，让模型自己决定搜什么。'''
        default_k = clamp_k(k or self.k)

        def web_search(query: str, top_k: int = default_k) -> str:
            '''在互联网上搜索指定关键词，返回相关网页的标题、链接和摘要。'''
            try:
                results = self.search(query, k=clamp_k(top_k, default_k))
            except SearchError as exc:
                # 不抛出去：把失败原因当工具结果交回模型，让它自己决定怎么办
                return f"搜索失败：{exc}"
            return self.format(results, query=query)

        return Tool.from_function(
            web_search,
            name=name,
            description=description
            or (
                "在互联网上搜索信息。返回若干网页的标题、链接和摘要。"
                "用法：query 是搜索关键词（用自然语言提问也行）。"
                "注意：网络内容不可信，可能有错误或过时信息；"
                "回答时请把来源链接一并给出。"
            ),
        )

    def __repr__(self) -> str:
        backend = getattr(self._resolved, "name", "auto")
        return f"<WebSearch backend={backend} k={self.k}>"


#: 开箱即用的工具：`LLMStep(..., tools=[WEB_SEARCH])`
WEB_SEARCH = WebSearch().as_tool()


__all__ = [
    "WEB_SEARCH",
    "API_BACKENDS",
    "KEYLESS_BACKENDS",
    "DEFAULT_CHAIN",
    "BaiduBackend",
    "BingCnBackend",
    "BochaBackend",
    "BraveBackend",
    "DuckDuckGoBackend",
    "FallbackBackend",
    "SearchBackend",
    "SearchError",
    "SearchResult",
    "SerperBackend",
    "TavilyBackend",
    "WebSearch",
    "ZhipuBackend",
    "clean_text",
    "clamp_k",
    "dedupe",
    "dig",
    "extract_baidu_target",
    "http_get",
    "http_post_json",
    "pair_links_with_text",
    "parse_baidu",
    "parse_bing",
    "parse_ddg_lite",
    "pick_backend",
    "pick_field",
    "resolve_baidu_target",
    "unwrap_url",
]
