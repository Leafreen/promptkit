'''阶段 4 验收：`Function` 改位置形参，`$` 只是声明记号。'''
import re

import pytest

from promptkit.core import (
    ArgumentError,
    Function,
    Return,
    Set,
    Var,
    Workflow,
)


def test_dollar_prefix_is_stripped():
    add = Function("$a", "$b", name="add").Do(Return(Var("a") + Var("b")))
    assert add.args == ("a", "b")
    ctx = Workflow(Set("sum", add(a=1, b=2).result)).run()
    assert ctx["sum"] == 3


def test_params_without_dollar_are_accepted():
    add = Function("a", "b", name="add").Do(Return(Var("a") + Var("b")))
    assert add.args == ("a", "b")
    assert Workflow(Set("sum", add(1, 2).result)).run()["sum"] == 3


def test_duplicate_after_normalisation_raises():
    with pytest.raises(ArgumentError):
        Function("$a", "a")


def test_empty_param_raises():
    with pytest.raises(ArgumentError):
        Function("")
    with pytest.raises(ArgumentError):
        Function("$")


def test_default_name_is_generated():
    definition = Function("$topic")
    assert re.fullmatch(r"function_[0-9a-f]{8}", definition.name)
    assert definition.output_key == definition.name


def test_optional_name_is_kept():
    definition = Function("$topic", name="make_lab", output_key="output")
    assert definition.name == "make_lab"
    assert definition.output_key == "output"


def test_old_args_kwarg_is_rejected():
    with pytest.raises(TypeError):
        Function(args=("a",))
