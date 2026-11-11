---
name: report_kit
description: 按写作规范把零散要点整理成 Markdown 报告
parameters:
  topic:
    type: string
    description: 报告主题
dependencies: []
scripts:
  report_kit_make_report:
    path: scripts/make_report.py
    description: 根据主题和要点生成 Markdown 报告
---

# 报告生成

先读 `references/style.md` 了解排版要求，再调用 `report_kit_make_report`
生成报告。

- 主题：{{topic}}
- 只使用用户给出的要点，不要自己补充事实
- 生成后告诉用户文件路径，不要贴出报告全文
