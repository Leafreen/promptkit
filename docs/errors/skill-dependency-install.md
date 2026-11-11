# SKILL_DEP_INSTALL — DependencyInstallError

`SkillPackage.install_deps()` 执行 `pip install -r requirements.txt` 失败。

处理：检查网络 / pip 源，先手动跑一遍 `pip install -r requirements.txt`
确认能装通；必要时改用带自定义源的镜像。
