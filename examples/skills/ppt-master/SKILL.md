---
name: ppt-master
description: >
  精简脱敏示例：AI 驱动的 PPT 制作流程。根据 config.json 生成可编辑的
  PPTX 并校验产物。真实完整版见 https://github.com/hugohe3/ppt-master 。
metadata:
  version: "6.9.0-trimmed"
---

# PPT Master（精简示例包）

## 使用流程（Quick 路线）

1. 写 `config.json`：`{"title": "...", "slides": [{"title": "...", "bullets": ["..."]}]}`
2. `python scripts/build_deck.py --config config.json --out deck.pptx`
3. `python scripts/validate_deck.py --deck deck.pptx` 校验页数与标题

## 参考资料

- `references/shared-standards.md`：共享排版标准
- `references/image-palettes/duotone.md`：双色调色板示例
