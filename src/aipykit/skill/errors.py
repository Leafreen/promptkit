# skill/errors.py
'''Skill 层的异常。

所有异常都带 `code` / `message` / `hint` / `docs_url` 四个字段，供上层
（CLI、日志、错误提示）统一消费；`str(exc)` 仍是 `message`，保持兼容。

    try:
        pkg = SkillPackage.load("skills/x")
    except InvalidSkillPackageError as exc:
        print(exc.code, exc.message, exc.hint, exc.docs_url)
'''
from __future__ import annotations


class SkillError(RuntimeError):
    '''Skill 相关错误的基类。

    子类通过 `default_code` / `docs_page` 覆盖默认值，构造时仍可用位置参数
    传 message（历史调用方式不变）。
    '''

    default_code = "SKILL_ERROR"
    docs_page = "skill-error"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        hint: str = "",
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code or self.default_code
        self.hint = hint
        self.docs_url = docs_url or f"docs/errors/{self.docs_page}.md"


class SkillValidationError(SkillError):
    '''定义不合法（名字 / 描述 / 参数）或调用参数不合法。'''

    default_code = "SKILL_VALIDATION"


class SkillAlreadyExists(SkillError):
    '''同名 Skill 已经注册过了。'''

    default_code = "SKILL_EXISTS"


class SkillNotFound(SkillError):
    '''注册表里没有这个 Skill。'''

    default_code = "SKILL_NOT_FOUND"


class SkillExecutionError(SkillError):
    '''Skill 执行时抛了异常（原始异常在 `__cause__` 里）。'''

    default_code = "SKILL_EXECUTION"


class SkillDependencyError(SkillError):
    '''技能声明的依赖没有安装（严格模式下加载时直接抛出）。'''

    default_code = "SKILL_DEPENDENCY"


# ============================ 技能包（skill_spec v1）============================
# 下面这组异常对应 docs/skill_spec.md §8 的错误体系；code 供机器识别，
# docs_url 指向 docs/errors/ 下的对应文档。

class InvalidSkillPackageError(SkillValidationError):
    '''目录不是合法技能包：SKILL.md 缺失、frontmatter 缺 name/description、
    name 不合规、python_version 不满足等。'''

    default_code = "SKILL_INVALID"
    docs_page = "skill-invalid-package"


class DependencyInstallError(SkillDependencyError):
    '''`SkillPackage.install_deps()` 安装 requirements.txt 失败。'''

    default_code = "SKILL_DEP_INSTALL"
    docs_page = "skill-dependency-install"


class PythonInterpreterError(SkillError):
    '''找不到可用的 Python 解释器，或显式指定的解释器无法运行。'''

    default_code = "SKILL_PYTHON"
    docs_page = "skill-python-interpreter"


class ScriptNotFoundError(SkillError):
    '''`run_script` 请求的脚本名不在登记清单里。'''

    default_code = "SKILL_SCRIPT_NOT_FOUND"
    docs_page = "skill-script-not-found"


class ResourceNotFoundError(SkillError):
    '''`read_resource` 请求的资源路径不存在。'''

    default_code = "SKILL_RESOURCE_NOT_FOUND"
    docs_page = "skill-resource-not-found"


class PathEscapeError(SkillValidationError):
    '''路径越界：目标不在技能目录（或读写根目录）内。'''

    default_code = "SKILL_PATH_ESCAPE"
    docs_page = "skill-path-escape"


class DuplicateToolError(SkillError):
    '''多技能包合并时工具名冲突。'''

    default_code = "SKILL_DUP_TOOL"
    docs_page = "skill-duplicate-tool"
