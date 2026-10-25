# skill/errors.py
'''Skill 层的异常。'''
from __future__ import annotations


class SkillError(RuntimeError):
    '''Skill 相关错误的基类。'''


class SkillValidationError(SkillError):
    '''定义不合法（名字 / 描述 / 参数）或调用参数不合法。'''


class SkillAlreadyExists(SkillError):
    '''同名 Skill 已经注册过了。'''


class SkillNotFound(SkillError):
    '''注册表里没有这个 Skill。'''


class SkillExecutionError(SkillError):
    '''Skill 执行时抛了异常（原始异常在 `__cause__` 里）。'''
