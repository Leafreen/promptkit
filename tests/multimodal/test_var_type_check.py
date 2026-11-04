'''Var 的可选类型标注：render 时校验，不匹配抛 TypeError。'''
import pytest

from promptkit.core.content import ImagePart, TextPart
from promptkit.core.steps.logics.expr_sys.expr import VarExpr
from promptkit.core.steps.logics.var import Var
from promptkit.prompt import Prompt


def test_var_without_type_accepts_anything():
    prompt = Prompt(role="user", content=[Var("x")])
    rendered = prompt.render(x="cat.png")
    assert rendered["content"][0]["type"] == "image"  # 无标注时按扩展名推断


def test_var_type_match_passes():
    prompt = Prompt(role="user", content=[Var("img", type=ImagePart)])
    rendered = prompt.render(img="cat.png")
    assert rendered["content"][0]["type"] == "image"


def test_var_type_mismatch_raises_type_error():
    prompt = Prompt(role="user", content=[Var("img", type=ImagePart)])
    with pytest.raises(TypeError, match="期望 ImagePart"):
        prompt.render(img="notes.txt")  # 推断成 TextPart，不匹配


def test_var_type_with_part_instance():
    prompt = Prompt(role="user", content=[Var("img", type=ImagePart)])
    image = ImagePart(source="aGk=", source_type="base64", media_type="image/png")
    assert prompt.render(img=image)["content"][0]["type"] == "image"
    with pytest.raises(TypeError):
        prompt.render(img=TextPart(text="不是图"))


def test_var_type_str_means_text():
    '''`type=str`：值必须是字符串，按文本部件处理（不做扩展名推断）。'''
    prompt = Prompt(role="user", content=[Var("x", type=str)])
    rendered = prompt.render(x="cat.png")
    assert rendered["content"][0] == {"type": "text", "text": "cat.png"}
    with pytest.raises(TypeError, match="期望 str"):
        prompt.render(x=123)


def test_var_expr_check_type_directly():
    expr = VarExpr("n", type=int)
    expr.check_type(42)  # 不抛
    with pytest.raises(TypeError, match="期望 int"):
        expr.check_type("42")
    VarExpr("anything").check_type(object())  # 无标注永不抛


def test_var_assignment_form_rejects_type():
    with pytest.raises(TypeError, match="不支持 type"):
        Var(name="x", val=1, type=int)
