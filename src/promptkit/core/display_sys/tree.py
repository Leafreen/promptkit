# core/display_sys/tree.py
'''把步骤树画成文本，方便调试：`print(wf)` 就能看到整棵结构。

`Node` 是 `Step` 接口的一部分 —— 每个 Step 实现 `to_node()` 返回它；
`format_tree` 才是「显示」的那一半。
'''
from dataclasses import dataclass, field
from typing import Any, List

__all__ = ["Node", "format_tree", "short"]


@dataclass
class Node:
    '''展示用的树节点：一行文字 + 若干子节点。'''

    text: str
    children: List["Node"] = field(default_factory=list)

    def add(self, *children: "Node") -> "Node":
        self.children.extend(children)
        return self


def format_tree(root: Node) -> str:
    '''渲染成一棵带缩进线的树。'''
    lines = [root.text]
    _render_children(root.children, "", lines)
    return "\n".join(lines)


def _render_children(children: List[Node], prefix: str, lines: List[str]) -> None:
    for index, child in enumerate(children):
        last = index == len(children) - 1
        lines.append(prefix + ("└─ " if last else "├─ ") + child.text)
        _render_children(child.children, prefix + ("   " if last else "│  "), lines)


def short(value: Any, limit: int = 44) -> str:
    '''把可能很长的对象压成一行短文本。'''
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
