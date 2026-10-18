# core/event_sys/callbacks.py
'''事件总线。'''
from typing import Callable, List

from promptkit.core.event_sys.events import Event

EventHandler = Callable[[Event], None]


class EventBus:
    '''事件总线：核心库只 emit，handler 决定怎么显示。

        bus = EventBus()
        bus.on(console_display())          # 要显示就挂 handler
        bus.on(lambda e: logger.info(e))   # 想记日志也行

        @bus.on                            # 也能当装饰器用
        def handler(event):
            ...
    '''

    def __init__(self, *handlers: EventHandler):
        self.handlers: List[EventHandler] = list(handlers)

    def on(self, handler: EventHandler) -> EventHandler:
        '''注册一个 handler，并把它返回（方便当装饰器）。'''
        if not callable(handler):
            raise TypeError(f"handler 必须可调用，得到 {handler!r}")
        self.handlers.append(handler)
        return handler

    def off(self, handler: EventHandler) -> None:
        '''注销一个 handler（不存在就忽略）。'''
        try:
            self.handlers.remove(handler)
        except ValueError:
            pass

    def clear(self) -> None:
        '''清空所有 handler。'''
        self.handlers.clear()

    @property
    def has_handlers(self) -> bool:
        return bool(self.handlers)

    def emit(self, event: Event) -> None:
        '''把事件发给所有 handler。

        遍历的是副本，所以 handler 里再 on/off 也不会把这次遍历搞乱。
        '''
        for handler in list(self.handlers):
            handler(event)

    def __len__(self) -> int:
        return len(self.handlers)

    def __repr__(self) -> str:
        return f"<EventBus handlers={len(self.handlers)}>"


class ChildBus(EventBus):
    '''只转发给指定 handler 的总线，用于「这一次调用临时显示」。'''

    def __init__(self, parent: EventBus, *handlers: EventHandler):
        super().__init__(*handlers)
        self.parent = parent

    def emit(self, event: Event) -> None:
        super().emit(event)
        self.parent.emit(event)
