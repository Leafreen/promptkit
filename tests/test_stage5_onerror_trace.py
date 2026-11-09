'''阶段 5 验收：`OnError` 兜底、结构化 trace、trace 落盘、ChatTurn 每轮兜底。'''
import json
from pathlib import Path

import pytest

from promptkit.core import (
    Echo,
    FunctionStep,
    OnError,
    Return,
    Set,
    Var,
    Workflow,
    WorkflowError,
)
from promptkit.core.steps.actions.chat_util import CHAT_ERROR_KEY, ChatTurn


def _boom(context):
    raise RuntimeError("炸了")


def test_on_error_swallows_and_writes_output():
    wf = Workflow(
        OnError(FunctionStep(_boom), output_key="err"),
        Set("after", "ok"),
    )
    ctx = wf.run()
    assert isinstance(ctx["err"], RuntimeError)
    assert ctx["after"] == "ok"


def test_on_error_handler_result_and_fallback():
    seen = []

    def handler(exc, context):
        seen.append(exc)
        return "已恢复"

    wf = Workflow(
        OnError(
            FunctionStep(_boom),
            handler=handler,
            fallback=Set("fallback_ran", True),
        ),
        Set("done", True),
    )
    ctx = wf.run()
    assert seen and isinstance(seen[0], RuntimeError)
    assert ctx["error"] == "已恢复"
    assert ctx["fallback_ran"] is True
    assert ctx["done"] is True


def test_on_error_reraise():
    wf = Workflow(OnError(FunctionStep(_boom), reraise=True))
    with pytest.raises(RuntimeError, match="炸了"):
        wf.run()


def test_on_error_does_not_swallow_control_flow():
    # Return 逃到工作流外是 WorkflowError，不能被 OnError 当成普通错误吞掉
    wf = Workflow(OnError(Return(1)))
    with pytest.raises(WorkflowError, match="Return"):
        wf.run()


def test_on_error_requires_steps():
    with pytest.raises(ValueError):
        OnError()


def test_trace_records_each_step_when_enabled():
    ctx = Workflow(Set("x", 1), Echo(Var("x"))).run(_trace_steps=True)
    starts = [e for e in ctx.trace if isinstance(e, dict) and e["phase"] == "start"]
    ends = [e for e in ctx.trace if isinstance(e, dict) and e["phase"] == "end"]
    assert [e["step"] for e in starts] == ["set", "echo"]
    assert [e["status"] for e in ends] == ["ok", "ok"]
    assert all(isinstance(e["duration_ms"], float) for e in ends)
    assert all(e["type"] for e in ends)


def test_trace_records_error_status():
    ctx = Workflow(OnError(FunctionStep(_boom))).run(_trace_steps=True)
    errors = [
        e
        for e in ctx.trace
        if isinstance(e, dict) and e.get("phase") == "end" and e["status"] == "error"
    ]
    assert len(errors) == 1
    assert errors[0]["step"] == "function"


def test_trace_is_untouched_by_default():
    ctx = Workflow(Set("x", 1)).run()
    # 默认仍只有步骤自己写的字符串标记，没有结构化 dict
    assert not any(isinstance(e, dict) for e in ctx.trace)
    assert "set:x" in ctx.trace


def test_dump_trace_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = Workflow(Set("x", 1), Set("y", 2)).run(_dump_trace=True)
    path = Path(ctx["_trace_file"])
    assert path.exists()
    assert path.parent.name == "logs"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(
        isinstance(entry, dict) and entry["phase"] == "end" for entry in lines
    )


class _BoomAgent:
    vendor = "fake"
    model = "boom"

    def ask(self, prompt, tools=None, bus=None, **kwargs):
        raise RuntimeError("网络炸了")


def test_chat_turn_captures_error_with_handler():
    seen = []
    turn = ChatTurn(on_error=lambda exc, ctx: seen.append(exc))
    ctx = Workflow(Set("usr_input", "hi"), turn).run(agent=_BoomAgent())
    assert isinstance(ctx[CHAT_ERROR_KEY], RuntimeError)
    assert seen and isinstance(seen[0], RuntimeError)
    assert "chat_turn:error:RuntimeError" in ctx.trace


def test_chat_turn_without_handler_still_raises():
    with pytest.raises(RuntimeError, match="网络炸了"):
        Workflow(Set("usr_input", "hi"), ChatTurn()).run(agent=_BoomAgent())
