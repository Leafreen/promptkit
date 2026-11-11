---
name: hello-skill
description: 最小示例技能包：向指定名字打招呼，并把问候写进 greeting.txt
---

# Hello Skill

## 使用流程

1. 运行 `python scripts/hello.py --name <名字>`，它会在工作目录写 `greeting.txt`
2. `resources/note.md` 是一份示例资源，演示 `read_resource` 的用法

## 约定演示点

- frontmatter 只含必填的 `name` / `description`
- 脚本用 argparse，可独立执行
- 资源是纯文本，按 `resources/note.md` 相对路径读取
