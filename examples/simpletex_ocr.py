# examples/simpletex_ocr.py
'''SimpleTex 开放平台 API 调用示例（公式识别 / 通用文档识别）。

只依赖 Python 标准库（urllib），不额外装 requests。

准备
----
1. 在 https://simpletex.cn/user/center 创建「用户授权令牌」（UAT）
2. 设置环境变量：

       set SIMPLETEX_TOKEN=你的UAT

运行
----
    python examples/simpletex_ocr.py formula D:\\docs\\formula.png
    python examples/simpletex_ocr.py formula D:\\docs\\formula.png --standard
    python examples/simpletex_ocr.py document D:\\docs\\page.png
    python examples/simpletex_ocr.py document D:\\docs\\page.png --mode formula

接进 promptkit
--------------
    from promptkit.core.tool import Tool

    tool = Tool.from_function(simpletex_formula, name="simpletex_formula")
    agent.ask("识别这张图片里的公式：D:\\docs\\formula.png", tools=[tool])

关于 APP 鉴权：UAT 方式仅适合开发；生产环境请用 APP ID / Secret 对
data 表单做 MD5 签名（header 里放 app-id / random-str / timestamp / sign）。
'''
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

#: 公式识别（轻量模型，速度更快）
LATEX_OCR_TURBO_URL = "https://server.simpletex.cn/api/latex_ocr_turbo"
#: 公式识别（标准模型，效果略好）
LATEX_OCR_URL = "https://server.simpletex.cn/api/latex_ocr"
#: 通用图片识别（可返回 markdown / LaTeX）
GENERAL_OCR_URL = "https://server.simpletex.cn/api/simpletex_ocr"
#: 文档页图片识别（返回 markdown）
DOC_OCR_URL = "https://server.simpletex.cn/api/doc_ocr"


class SimpleTexError(RuntimeError):
    '''SimpleTex 返回失败或网络错误。'''


# ---------------- 底层：multipart + POST ----------------
def _encode_multipart(
    fields: Mapping[str, str],
    files: Sequence[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    '''把字段和文件拼成 multipart/form-data 请求体。

    Args:
        fields: 普通表单键值对。
        files: `(字段名, 文件名, 二进制, Content-Type)` 列表；同名可重复。

    Returns:
        (请求体字节, Content-Type 头)。
    '''
    boundary = uuid.uuid4().hex
    body = bytearray()
    for key, value in fields.items():
        body.extend(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    for key, filename, data, content_type in files:
        body.extend(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{key}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _guess_content_type(filename: str) -> str:
    suffix = os.path.splitext(filename)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def _post(
    url: str,
    fields: Mapping[str, str],
    files: Sequence[tuple[str, str, bytes, str]],
    token: str | None = None,
) -> dict[str, Any]:
    '''发一次 POST，返回解析后的 JSON；`status=false` 抛 SimpleTexError。'''
    token = token or os.environ.get("SIMPLETEX_TOKEN", "")
    if not token:
        raise SimpleTexError("缺少 SIMPLETEX_TOKEN（用户授权令牌 UAT）")

    body, content_type = _encode_multipart(fields, files)
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"token": token, "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SimpleTexError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise SimpleTexError(f"网络错误: {exc.reason}") from exc

    if not payload.get("status"):
        raise SimpleTexError(f"SimpleTex 调用失败: {payload}")
    return payload


# ---------------- 公开 API ----------------
def simpletex_formula(image_path: str, *, standard: bool = False) -> str:
    '''识别图片中的公式，返回 LaTeX 字符串。

    Args:
        image_path: 本地图片路径（png / jpg 等）。
        standard: True 用标准模型（更准、更慢），默认轻量模型。
    '''
    with open(image_path, "rb") as handle:
        data = handle.read()
    payload = _post(
        LATEX_OCR_URL if standard else LATEX_OCR_TURBO_URL,
        fields={},
        files=[("file", os.path.basename(image_path), data, _guess_content_type(image_path))],
    )
    return str(payload["res"]["latex"])


def simpletex_document(
    image_path: str,
    *,
    mode: str = "document",
    enable_rotation: bool = False,
) -> str:
    '''识别文档 / 图片内容，返回 Markdown（或 LaTeX）。

    Args:
        image_path: 本地图片路径。
        mode: `auto` / `document` / `formula`，默认 document。
        enable_rotation: 是否自动矫正 0/90/180/270 度方向。
    '''
    with open(image_path, "rb") as handle:
        data = handle.read()
    payload = _post(
        GENERAL_OCR_URL,
        fields={"rec_mode": mode, "enable_img_rot": "true" if enable_rotation else "false"},
        files=[("file", os.path.basename(image_path), data, _guess_content_type(image_path))],
    )
    result = payload.get("res", {})
    for key in ("content", "markdown", "latex"):
        if isinstance(result, dict) and result.get(key):
            return str(result[key])
    return json.dumps(result, ensure_ascii=False)


def simpletex_doc_page(image_path: str) -> str:
    '''识别 PDF 页面图片，返回 Markdown。'''
    with open(image_path, "rb") as handle:
        data = handle.read()
    payload = _post(
        DOC_OCR_URL,
        fields={},
        files=[("file", os.path.basename(image_path), data, _guess_content_type(image_path))],
    )
    return str(payload["res"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleTex OCR 示例")
    parser.add_argument("command", choices=["formula", "document", "doc"])
    parser.add_argument("image_path")
    parser.add_argument("--standard", action="store_true", help="公式识别用标准模型")
    parser.add_argument("--mode", default="document", choices=["auto", "document", "formula"])
    args = parser.parse_args()

    try:
        if args.command == "formula":
            print(simpletex_formula(args.image_path, standard=args.standard))
        elif args.command == "document":
            print(simpletex_document(args.image_path, mode=args.mode))
        else:
            print(simpletex_doc_page(args.image_path))
    except SimpleTexError as exc:
        raise SystemExit(f"调用失败：{exc}") from exc


if __name__ == "__main__":
    main()
# 职责：SimpleTex 开放平台 API 示例 —— UAT 鉴权、公式识别、通用文档识别、PDF 页识别
