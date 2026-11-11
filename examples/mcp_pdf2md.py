# examples/mcp_pdf2md.py
'''把 MCP-PDF2MD（PDF → Markdown）接进 promptkit 的完整示例。

关键前提
--------
MCP-PDF2MD 默认以 **stdio** 方式启动（`mcp.run()`），而 promptkit 的 MCP
Client 只支持 **Streamable HTTP**。所以不能直接用 `uv run pdf2md`，必须先用
一个 HTTP 启动脚本把它暴露成 `http://127.0.0.1:8931/mcp`（见下方注释）。

准备
----
1. 安装 promptkit 的 MCP 依赖（拉取官方 mcp SDK）：

       pip install promptkit[mcp]

2. 在 MCP-PDF2MD 仓库根目录放一个 serve_pdf2md_http.py：

       import os
       from pdf2md.server import mcp, set_output_dir

       set_output_dir(os.environ.get("PDF2MD_OUTPUT_DIR", "./downloads"))

       if __name__ == "__main__":
           mcp.run(
               transport="streamable-http",
               host="127.0.0.1",
               port=8931,
           )

3. 启动（需要 Mistral OCR 的 API Key）：

       set MISTRAL_API_KEY=你的key
       uv run --directory C:\\path\\to\\mcp-pdf2md python serve_pdf2md_http.py

运行本示例
----------
    python examples/mcp_pdf2md.py discover            # 只发现工具
    python examples/mcp_pdf2md.py tools               # 等同 discover
    python examples/mcp_pdf2md.py convert D:\\docs\\a.pdf
    python examples/mcp_pdf2md.py workflow D:\\docs\\a.pdf
    python examples/mcp_pdf2md.py ask "把 D:\\docs\\a.pdf 转成 markdown"

`convert` / `workflow` 不花模型 token；`ask` 需要配置 promptkit 的厂商与 Key。
'''
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

#: Server 的 Streamable HTTP endpoint（和 serve_pdf2md_http.py 里的端口一致）
PDF2MD_URL = os.environ.get("PDF2MD_MCP_URL", "http://127.0.0.1:8931/mcp")

#: promptkit 的批量加载配置 —— 格式见 promptkit/mcp/config.py
MCP_CONFIG: dict[str, Any] = {
    "mcp_servers": [
        {
            "name": "pdf2md",
            # 注册出来的工具名会是 pdf2md__convert_pdf_file 这种前缀形式
            "url": PDF2MD_URL,
            "transport": "http",
            # 白名单只注册这两个；留空或省略表示该 Server 的全部工具
            "allowed_tools": ["convert_pdf_file", "convert_pdf_url"],
        }
    ]
}


def load_registry() -> tuple[Any, list[str]]:
    '''连接 Server、发现工具、注册到 ToolRegistry，并打开安全 gate。

    Returns:
        (ToolRegistry 实例, 注册成功的工具名列表)。
    '''
    from promptkit.core.tool.registry import ToolRegistry
    from promptkit.mcp import (
        get_default_hook_registry,
        load_mcp_from_config,
        register_mcp_security,
    )

    registry = ToolRegistry()
    registered = asyncio.run(load_mcp_from_config(registry, MCP_CONFIG))
    if not registered:
        raise SystemExit(
            f"没有发现任何工具。确认 MCP-PDF2MD 已用 HTTP 方式启动：{PDF2MD_URL}"
        )

    # 安全层默认拒绝一切 MCP 工具 —— 必须显式放行发现到的工具
    register_mcp_security(get_default_hook_registry(), set(registered))
    return registry, registered


def cmd_discover() -> None:
    '''只连接 + 发现工具，不调用。'''
    registry, registered = load_registry()
    print(f"已注册 {len(registered)} 个 MCP 工具：")
    for tool in registry.all():
        print(f"  • {tool.name}: {tool.description}")
        print(f"    参数 schema: {tool.parameters}")


def cmd_convert(pdf_path: str) -> None:
    '''直接调用 convert_pdf_file（不经过模型）。'''
    registry, _ = load_registry()
    tool = registry.get("pdf2md__convert_pdf_file")
    if tool is None:
        raise SystemExit("Server 上没有 convert_pdf_file 工具")
    result = tool.call({"file_path": pdf_path})
    print(result)


def cmd_workflow(pdf_path: str) -> None:
    '''用 Workflow 的 MCPToolStep 显式编排：把结果写进 Context 再打印。'''
    from promptkit.core import Echo, NewVar, Var, Workflow
    from promptkit.core.steps.actions.mcp_step import MCPToolStep

    registry, _ = load_registry()
    wf = Workflow(
        NewVar("pdf_path", default=pdf_path),
        MCPToolStep(
            tool=registry.get("pdf2md__convert_pdf_file"),
            input_keys={"file_path": "pdf_path"},
            output_key="pdf_markdown",
        ),
        Echo(Var("pdf_markdown")),
        name="pdf2md_demo",
    )
    wf.run()


def cmd_ask(prompt: str) -> None:
    '''让 Agent 自主决定调用哪个 MCP 工具（需要模型 API Key）。'''
    from promptkit.agent import Agent

    registry, _ = load_registry()
    agent = Agent(
        vendor=os.environ.get("PROMPTKIT_VENDOR", "doubao"),
        api_key=os.environ.get("PROMPTKIT_API_KEY"),
    )
    print(agent.ask(prompt, tools=registry.all()))


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-PDF2MD + promptkit 示例")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("discover", help="发现并列出工具")
    sub.add_parser("tools", help="同 discover")

    p_convert = sub.add_parser("convert", help="直接调用 convert_pdf_file")
    p_convert.add_argument("pdf_path")

    p_workflow = sub.add_parser("workflow", help="Workflow + MCPToolStep")
    p_workflow.add_argument("pdf_path")

    p_ask = sub.add_parser("ask", help="Agent 自主选择工具")
    p_ask.add_argument("prompt")

    args = parser.parse_args()
    if args.command in (None, "discover", "tools"):
        cmd_discover()
    elif args.command == "convert":
        cmd_convert(args.pdf_path)
    elif args.command == "workflow":
        cmd_workflow(args.pdf_path)
    elif args.command == "ask":
        cmd_ask(args.prompt)


if __name__ == "__main__":
    main()
# 职责：MCP-PDF2MD 接入示例 —— 配置、发现工具、直接调用、Workflow 编排、Agent 自主调用
