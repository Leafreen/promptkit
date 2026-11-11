# SKILL_PYTHON — PythonInterpreterError

找不到可用的 Python 解释器，或显式指定的解释器无法运行。

`resolve_python` 的取值顺序：显式 `python=` → 环境变量 `PROMPTKIT_PYTHON`
→ 技能包根目录的 `.venv` / `venv` → 当前解释器 → PATH。

处理：

- 显式 `python=` 或 `PROMPTKIT_PYTHON` 报错：确认路径存在且能运行
  （`<python> -c "import sys; print(sys.version)"`），或删掉配置改用自动探测
- 自动探测全部失败：把 Python 加入 PATH，或在技能包根目录创建 `.venv`，
  或用 `PROMPTKIT_PYTHON` 指定完整路径
- Windows 应用商店的假 python 会以退出码 9009 结束，探测阶段会自动跳过
