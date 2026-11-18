"""aipykit：一个极简的大模型提示词 / 工作流工具包。

典型用法（从 aipykit 顶层直接 import）::

    from aipykit import Agent, Workflow, Prompt
    from aipykit import SubAgentFactory, SummonSubAgentTool

所有符号都通过懒加载暴露，``import aipykit`` 本身几乎零开销 ——
真正用到时才从对应子包加载。
"""

from __future__ import annotations

