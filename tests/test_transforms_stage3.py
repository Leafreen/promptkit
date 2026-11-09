'''阶段 3 验收：`Call` 收敛为可注册、可序列化的 `Transform`。

覆盖：
- 未注册的名字在**构造时**就抛 `UnknownTransformError`
- lambda / 闭包 / 动态表达式抛 `UnserializableCallError`
- 旧的 `Call(f, ...)` 仍能把注册过的函数翻译成 `Transform`（带弃用警告）
- `to_dict()` / `from_dict()` 往返等价（含嵌套）
- 内置转换可用，且 `Transform` 是表达式不是步骤
'''
import json
import warnings

import pytest

from promptkit.core import (
    Call,
    Context,
    Expr,
    Set,
    Step,
    Transform,
    UnserializableCallError,
    UnknownTransformError,
    Var,
    Workflow,
    registered_transforms,
    transform,
)


def _eval(expr, **data):
    return expr.eval(Context(data=data))


def test_unknown_transform_name_fails_at_construction():
    with pytest.raises(UnknownTransformError) as exc:
        Transform("no_such_transform", Var("x"))
    assert "no_such_transform" in str(exc.value)


def test_call_with_lambda_is_unserializable():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(UnserializableCallError):
            Call(lambda x: x * 2, Var("n"))


def test_call_with_closure_is_unserializable():
    factor = 3

    def scale(x):
        return x * factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(UnserializableCallError):
            Call(scale, Var("n"))


def test_call_with_dynamic_expression_is_unserializable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(UnserializableCallError):
            Call(Var("f"), Var("n"))


def test_call_with_unregistered_function_fails():
    def plain(x):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(UnknownTransformError):
            Call(plain, Var("n"))


def test_call_translates_registered_function():
    @transform("test_stage3_double")
    def double(x):
        return x * 2

    with pytest.warns(DeprecationWarning):
        expr = Call(double, Var("n"))
    assert isinstance(expr, Transform)
    assert expr.name == "test_stage3_double"
    assert _eval(expr, n=21) == 42


def test_transform_round_trip():
    expr = Transform("join", Var("items"), sep=",")
    data = expr.to_dict()
    assert data["type"] == "transform"
    assert data["name"] == "join"
    restored = Transform.from_dict(data)
    assert restored.to_dict() == data
    assert _eval(restored, items=["a", "b"]) == "a,b"


def test_nested_transform_round_trip():
    expr = Transform(
        "format",
        "{a}-{b}",
        a=Transform("upper", Var("x")),
        b=Transform("len", Var("y")),
    )
    restored = Transform.from_dict(json.loads(json.dumps(expr.to_dict())))
    assert restored.to_dict() == expr.to_dict()
    assert _eval(restored, x="hi", y=[1, 2]) == "HI-2"


def test_from_dict_rejects_foreign_data():
    with pytest.raises(TypeError):
        Transform.from_dict({"type": "var", "name": "x"})


def test_transform_is_expression_not_step():
    expr = Transform("len", Var("items"))
    assert isinstance(expr, Expr)
    assert not isinstance(expr, Step)
    with pytest.raises(TypeError, match="期望 Step"):
        Workflow(expr)


def test_builtin_transforms_are_registered():
    names = set(registered_transforms())
    expected = {
        "join",
        "split",
        "upper",
        "lower",
        "strip",
        "len",
        "append",
        "extend",
        "list",
        "dict",
        "json_dumps",
        "json_loads",
        "format",
    }
    assert expected <= names


def test_builtin_transforms_evaluate():
    assert _eval(Transform("json_loads", '{"a": 1}')) == {"a": 1}
    assert _eval(Transform("json_dumps", {"a": 1})) == '{"a": 1}'
    assert _eval(Transform("append", Var("xs"), 3), xs=[1, 2]) == [1, 2, 3]
    assert _eval(Transform("strip", Var("s")), s="  hi  ") == "hi"


def test_custom_transform_can_be_used_in_a_workflow():
    @transform("test_stage3_shout")
    def shout(text):
        return text.upper() + "!"

    ctx = Workflow(Set("out", Transform("test_stage3_shout", Var("name")))).run(
        name="hi"
    )
    assert ctx["out"] == "HI!"
