# examples/simpletex_mcp_client.py
r'''用 promptkit 的 MCP Client 接入 SimpleTex MCP Server。

前置：先启动 examples/simpletex_mcp_server.py（默认 127.0.0.1:8932）。

准备
----
    pip install "promptkit[mcp]"
    # 另一个终端：
    set SIMPLETEX_TOKEN=你的用户授权令牌
    python examples/simpletex_mcp_server.py

运行
----
    python examples/simpletex_mcp_client.py discover
    python examples/simpletex_mcp_client.py formula   D:\\docs\formula.png
    python examples/simpletex_mcp_client.py document  D:\\docs\\page.png
    python examples/simpletex_mcp_client.py pdf       D:\\docs\\page.png
    python examples/simpletex_mcp_client.py workflow  D:\\docs\formula.png
    python examples/simpletex_mcp_client.py ask "识别 D:\\docs\formula.png 里的公式"

`formula/document/pdf/workflow` 不花模型 token；`ask` 需要模型 API Key。
'''
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

#: 与 simpletex_mcp_server.py 的端口保持一致
SIMPLETEX_MCP_URL = os.environ.get(
    "SIMPLETEX_MCP_URL", "http://127.0.0.1:8932/mcp"
)

#: promptkit 的 MCP 批量加载配置
MCP_CONFIG: dict[str, Any] = {
    "mcp_servers": [
        {
            "name": "simpletex",
            "url": SIMPLETEX_MCP_URL,
            "transport": "http",
            # 裸工具名白名单；注册后会带 simpletex__ 前缀
            "allowed_tools": ["formula", "document", "pdf_page"],
        }
    ]
}


def load_registry() -> tuple[Any, list[str]]:
    '''连接 Server、发现工具、注册并打开安全 gate。'''
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
            f"没有发现任何工具。确认 SimpleTex MCP Server 已启动：{SIMPLETEX_MCP_URL}"
        )
    # 安全层默认拒绝 MCP 工具，必须显式放行
    register_mcp_security(get_default_hook_registry(), set(registered))
    return registry, registered


def cmd_discover() -> None:
    registry, registered = load_registry()
    print(f"已注册 {len(registered)} 个 MCP 工具：")
    for tool in registry.all():
        print(f"  • {tool.name}: {tool.description}")


def _leaf_message(exc: BaseException) -> str:
    '''从 ExceptionGroup 里取出最内层的信息，便于人读。'''
    children = getattr(exc, "exceptions", ())
    if children:
        return _leaf_message(children[0])
    return str(exc) or type(exc).__name__


def _call(tool_name: str, arguments: dict[str, object]) -> None:
    registry, _ = load_registry()
    tool = registry.get(tool_name)
    if tool is None:
        raise SystemExit(f"Server 上没有 {tool_name} 工具")
    try:
        print(tool.call(arguments))
    except Exception as exc:
        raise SystemExit(f"调用 {tool_name} 失败：{_leaf_message(exc)}") from exc


def cmd_workflow(image_path: str) -> None:
    '''Workflow + MCPToolStep：显式编排公式识别。'''
    from promptkit.core import Echo, NewVar, Var, Workflow
    from promptkit.core.steps.actions.mcp_step import MCPToolStep

    registry, _ = load_registry()
    wf = Workflow(
        NewVar("image_path", default=image_path),
        MCPToolStep(
            tool=registry.get("simpletex__formula"),
            input_keys={"image_path": "image_path"},
            output_key="latex",
        ),
        Echo(Var("latex")),
        name="simpletex_demo",
    )
    wf.run()


def cmd_ask(prompt: str) -> None:
    '''Agent 自主决定调用哪个 SimpleTex 工具。'''
    from promptkit.agent import Agent

    registry, _ = load_registry()
    agent = Agent(
        vendor=os.environ.get("PROMPTKIT_VENDOR", "doubao"),
        api_key=os.environ.get("PROMPTKIT_API_KEY"),
    )
    print(agent.ask(prompt, tools=registry.all()))


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleTex MCP + promptkit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("discover", help="发现并列出工具")

    for name, help_text in (
        ("formula", "公式 → LaTeX"),
        ("document", "文档/图片 → Markdown"),
        ("pdf", "PDF 页 → Markdown"),
        ("workflow", "Workflow + MCPToolStep"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("image_path")

    p_ask = sub.add_parser("ask", help="Agent 自主选择工具")
    p_ask.add_argument("prompt")

    args = parser.parse_args()
    if args.command in (None, "discover"):
        cmd_discover()
    elif args.command == "formula":
        _call("simpletex__formula", {"image_path": args.image_path})
    elif args.command == "document":
        _call("simpletex__document", {"image_path": args.image_path})
    elif args.command == "pdf":
        _call("simpletex__pdf_page", {"image_path": args.image_path})
    elif args.command == "workflow":
        cmd_workflow(args.image_path)
    elif args.command == "ask":
        cmd_ask(args.prompt)


if __name__ == "__main__":
    main()
# 职责：promptkit MCP Client 接入 SimpleTex MCP Server 的示例（发现/直调/Workflow/Agent）
