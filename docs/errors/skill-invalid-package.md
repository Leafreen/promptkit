# SKILL_INVALID — InvalidSkillPackageError

技能包目录不合法，`SkillPackage.load()` 时抛出。

触发条件：目录不存在、缺少 `SKILL.md`/`AGENT.md`、frontmatter 缺
`name`/`description`、`name` 不是 kebab-case、`python_version` 不满足。

处理：按 `hint` 修正 SKILL.md 的 frontmatter，或换合法目录。
