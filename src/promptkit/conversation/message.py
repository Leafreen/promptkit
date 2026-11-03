# conversation/message.py
'''对话状态层里唯一的消息类型。

本模块**不依赖任何 LLM 相关模块**：它只认识「角色 + 内容 + 元信息」，
至于这些消息最终怎么发给哪一个厂商，是 Agent / client_sys 的事。
'''
from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]

MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool", "function"]
ContentPart: TypeAlias = dict[str, JSONValue]
MessageContent: TypeAlias = str | list[ContentPart]

_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})


def new_message_id() -> str:
    '''生成一条消息的 id（uuid4().hex，去掉短横线方便写日志）。'''
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Message:
    '''一条不可变的对话消息。

    字段：

    - `role`：`system` / `user` / `assistant` / `tool` / `function`
    - `content`：字符串，或多模态 `[{...}, ...]`
    - `id`：自动生成的 uuid4().hex；也可以在反序列化时显式传入
    - `metadata`：其余键（`tool_calls` / `tool_call_id` / `name` /
      `reasoning_content` 等）都放在这里，原样透传给调用方

    `frozen=True` 保证字段不能被原地改写；要改就用 `with_content()` /
    `with_metadata()` 或 `dataclasses.replace()` 生成一条新消息。
    `content` 与 `metadata` 在构造时会做一层防御性拷贝，避免外部持有的
    可变对象被顺手改掉。
    '''

    role: MessageRole
    content: MessageContent
    id: str = field(default_factory=new_message_id)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.content, list):
            object.__setattr__(self, "content", copy.deepcopy(self.content))
        elif not isinstance(self.content, str):
            raise TypeError(
                f"content 必须是 str 或 list，得到 {type(self.content).__name__}"
            )
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    # ---------------- 替换式修改 ----------------
    def with_content(self, content: MessageContent) -> Message:
        '''返回一条内容被替换的新消息（id / metadata 原样保留）。'''
        return replace(self, content=content)

    def with_metadata(self, **values: JSONValue) -> Message:
        '''返回一条 metadata 被合并的新消息（其余字段原样保留）。'''
        merged: dict[str, JSONValue] = dict(self.metadata)
        merged.update(values)
        return replace(self, metadata=merged)

    # ---------------- 与 dict 形状互转 ----------------
    def to_dict(self) -> dict[str, JSONValue]:
        '''转成老接口使用的 `{role, content, ...}` 字典。

        不输出 `id`：id 是会话层的内部标识，发给模型时不该出现。
        '''
        payload: dict[str, JSONValue] = copy.deepcopy(dict(self.metadata))
        payload["role"] = self.role
        payload["content"] = cast(JSONValue, copy.deepcopy(self.content))
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Message:
        '''从 `{role, content, ...}` 字典构造消息。

        不认识的 role 退回 `user`；`content=None` 归一成空字符串；
        `role` / `content` / `id` 之外的键全部进 `metadata`。
        '''
        raw_role = data.get("role")
        role = raw_role if isinstance(raw_role, str) and raw_role in _ROLES else "user"

        raw_content = data.get("content")
        if raw_content is None:
            content: MessageContent = ""
        elif isinstance(raw_content, (str, list)):
            content = cast(MessageContent, raw_content)
        else:
            content = str(raw_content)

        metadata: dict[str, JSONValue] = {
            key: cast(JSONValue, value)
            for key, value in data.items()
            if key not in ("role", "content", "id")
        }
        return cls(
            role=cast(MessageRole, role), content=content, metadata=metadata
        )


# 本文件核心职责：定义不可变 Message（frozen + uuid4().hex id + str/list content + metadata）。
