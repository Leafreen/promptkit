# core/errors.py
'''DSL 用到的异常。全部继承 WorkflowError，方便一次性捕获。'''


class WorkflowError(Exception):
    '''工作流相关错误的基类。'''


class UndefinedVariable(WorkflowError, KeyError):
    '''读了没有声明过的变量。'''

    def __str__(self) -> str:  # KeyError.__str__ 会加引号，这里去掉
        return Exception.__str__(self)


class LoopLimitExceeded(WorkflowError):
    '''While 循环超过了 max_iterations，通常意味着循环变量没被更新。'''


class ArgumentError(WorkflowError, TypeError):
    '''Function 定义/调用时的参数不匹配。'''


class VarTypeError(WorkflowError, TypeError):
    '''给带类型的变量赋了转不过去的值。'''


class BreakSignal(WorkflowError):
    '''内部信号：BreakStep 抛出，由外层循环捕获。用户不应该看到它。'''


class ContinueSignal(WorkflowError):
    '''内部信号：ContinueStep 抛出，由外层循环捕获。用户不应该看到它。'''


class GotoSignal(WorkflowError):
    '''内部信号：GotoStep 抛出，由持有该标签的 Workflow 捕获。'''

    def __init__(self, target: str):
        super().__init__(target)
        self.target = target


class UnknownLabel(WorkflowError):
    '''goto 了一个不存在的标签。'''


class ReturnSignal(WorkflowError):
    '''内部信号：ReturnStep 抛出，由 BoundStep（函数体）捕获。'''

    def __init__(self, value: object):
        super().__init__(value)
        self.value = value
