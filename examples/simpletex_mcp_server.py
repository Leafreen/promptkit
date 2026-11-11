# examples/simpletex_mcp_server.py
'''把 SimpleTex OCR 包装成一个 MCP Server（Streamable HTTP）。

为什么要包一层
--------------
SimpleTex 本身只是普通 HTTP API，不是 MCP Server。promptkit 的 MCP Client
只认 Streamable HTTP，所以需要这个薄封装：把 SimpleTex 的三个接口注册成
MCP 工具，再由 promptkit 发现并调用。

工具列表
--------
- `formula(image_path, standard=False)`  → LaTeX
- `document(image_path, mode, enable_rotation)` → Markdown
- `pdf_page(image_path)` → Markdown

启动
----
    pip install "mcp>=1.8"
    set SIMPLETEX_TOKEN=你的用户授权令牌

    python examples/simpletex_mcp_server.py

默认监听 http://127.0.0.1:8932/mcp，可用环境变量覆盖：
    SIMPLETEX_MCP_HOST / SIMPLETEX_MCP_PORT
'''
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simpletex_ocr import (
    simpletex_doc_page,
    simpletex_document,
    simpletex_formula,
)

HOST = os.environ.get("SIMPLETEX_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("SIMPLETEX_MCP_PORT", "8932"))

# mcp 1.x 用 FastMCP；2.x 改名为 MCPServer（工具装饰器用法一致）
try:  # pragma: no cover - 取决于本机安装的 SDK 版本
    from mcp.server.fastmcp import FastMCP as _MCPServer

    mcp = _MCPServer("simpletex-ocr", host=HOST, port=PORT)
    _RUN_KWARGS: dict[str, object] = {}
except ImportError:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _MCPServer

    mcp = _MCPServer("simpletex-ocr")
    _RUN_KWARGS = {"host": HOST, "port": PORT}


@mcp.tool()
def formula(image_path: str, standard: bool = False) -> str:
    '''识别图片中的数学公式，返回 LaTeX。

    Args:
        image_path: 本地图片路径（png / jpg 等）。
        standard: True 用标准模型（更准更慢），默认轻量模型。
    '''
    return simpletex_formula(image_path, standard=standard)


@mcp.tool()
def document(
    image_path: str,
    mode: str = "document",
    enable_rotation: bool = False,
) -> str:
    '''识别文档 / 图片内容，返回 Markdown。

    Args:
        image_path: 本地图片路径。
        mode: auto / document / formula，默认 document。
        enable_rotation: 是否自动矫正 0/90/180/270 度方向。
    '''
    return simpletex_document(
        image_path, mode=mode, enable_rotation=enable_rotation
    )


@mcp.tool()
def pdf_page(image_path: str) -> str:
    '''识别 PDF 页面截图，返回 Markdown。

    Args:
        image_path: 单页 PDF 渲染出的图片路径。
    '''
    return simpletex_doc_page(image_path)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", **_RUN_KWARGS)
# 职责：SimpleTex OCR → MCP Server 薄封装（三个工具 + Streamable HTTP 启动）
