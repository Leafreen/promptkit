'''使用 Workflow 进行带 AI Agent 的编程：情感分析。'''
import os
import sys

# 让 promptkit 这个包可以被导入：把它的**父目录**放进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptkit.core import *  # noqa: E402,F403
from promptkit.agent import Agent  # noqa: E402

if __name__ == '__main__':
    wf = Workflow()
    agent = Agent()

    EmotionAnalyzer = Function(
        args=('text', ),
        name='analyze_emotion',
        output_key='score',
    ).Do(
        Prompt(
            role='user',
            content='请分析以下文本的情感：{{text}}。只输出 0-1 之间的一个数字，0 表示负面，1 表示正面。',
        ),
    )

    wf.add(
        NewVar('text'),
        Set('text', Read('请输入文本：\n')),
        
        EmotionAnalyzer(text=Var('text')),
        Number('score'),
        Set('score', Var('score')),
        
        Echo('情感分析分数：', (Var('score'))),
        
        If(Var('score') >= 0.5).Then(
            Echo('情感分析结果：正面'),
        ).Else(
            Echo('情感分析结果：负面'),
        ),
    )

    # print(wf)
    wf.run()
