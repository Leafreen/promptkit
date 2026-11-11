---
name: word_count_report
description: 统计文本文件的字数，并整理成简短汇报
parameters:
  path:
    type: string
    description: 要统计的文本文件路径
---

# 字数统计汇报

先调用 `count_words` 工具统计 {{path}}，拿到 JSON 后整理成三行汇报：

- 总字数（不含空白）
- 总行数
- 估算阅读时长（分钟）

只输出汇报本身，不要贴原始 JSON，不要客套话。
