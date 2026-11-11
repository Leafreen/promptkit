# Skill 错误码索引

所有 Skill 异常都带 `code` / `message` / `hint` / `docs_url` 四个字段。
本页是全部 `code` 的索引，具体异常的说明见各自文档。

| code | 异常 | 说明 | 文档 |
|---|---|---|---|
| `SKILL_ERROR` | `SkillError` | 基类兜底 | 本页 |
| `SKILL_VALIDATION` | `SkillValidationError` | 定义/参数不合法 | 本页 |
| `SKILL_EXISTS` | `SkillAlreadyExists` | 同名技能已注册 | 本页 |
| `SKILL_NOT_FOUND` | `SkillNotFound` | 注册表里没有该技能 | 本页 |
| `SKILL_EXECUTION` | `SkillExecutionError` | 执行时抛异常 | 本页 |
| `SKILL_DEPENDENCY` | `SkillDependencyError` | 声明依赖缺失 | 本页 |
| `SKILL_INVALID` | `InvalidSkillPackageError` | 技能包目录/入口/frontmatter 不合法 | [skill-invalid-package](skill-invalid-package.md) |
| `SKILL_DEP_INSTALL` | `DependencyInstallError` | 安装 requirements.txt 失败 | [skill-dependency-install](skill-dependency-install.md) |
| `SKILL_SCRIPT_NOT_FOUND` | `ScriptNotFoundError` | 脚本名不在清单 | [skill-script-not-found](skill-script-not-found.md) |
| `SKILL_RESOURCE_NOT_FOUND` | `ResourceNotFoundError` | 资源路径不存在 | [skill-resource-not-found](skill-resource-not-found.md) |
| `SKILL_PATH_ESCAPE` | `PathEscapeError` | 路径越界 | [skill-path-escape](skill-path-escape.md) |
| `SKILL_DUP_TOOL` | `DuplicateToolError` | 工具名冲突 | [skill-duplicate-tool](skill-duplicate-tool.md) |

约定参见 [skill_spec](../../skill_spec.md)。
