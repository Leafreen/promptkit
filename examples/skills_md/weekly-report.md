---
name: weekly_report
description: 把零散的工作记录整理成一份结构化周报
parameters:
  period:
    type: string
    description: 覆盖的时间范围，如 2026-W45
  style: 周报语气，默认「简洁专业」
tags: [writing, report]
---

# 周报规范

请把用户给出的工作记录整理成 {{period}} 的周报，语气{{style}}，结构如下：

1. **本周完成**：按项目分条，每条一行，先结论后细节
2. **进行中**：标注进度百分比和卡点
3. **下周计划**：最多 3 条，每条带预期产出

只输出周报正文，不要客套话。记录里没有提到的事不要编造。
