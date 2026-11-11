'''Python 解释器解析：让库在任意机器上开箱可用，不要求预先配置环境。

换一台电脑安装后，技能脚本该用哪个解释器不需要手工指定，按下面的优先级取：

1. 显式传入的 `python=`（业务代码明确指定）
2. 环境变量 `PROMPTKIT_PYTHON`（部署级统一配置）
3. 技能包根目录下的虚拟环境 `.venv` / `venv`（每个技能包可各用各的）
4. 当前进程解释器 `sys.executable`
5. PATH 里的 `python3` / `python` / `py`

前两项写错会直接抛 `PythonInterpreterError`（配置错误应当立刻暴露）；后三项
属于自动探测，候选不可用只记 warning 并继续往下找，全部失败才抛错。Windows
应用商店的假 python 别名会以退出码 9009 结束，在探测阶段就会被跳过，不会等到
真正执行脚本才报错。

    from promptkit.skill import resolve_python

    resolve_python()                       # 自动探测
    resolve_python(root="skills/x")        # 优先用技能包自己的 .venv
    resolve_python("C:/Python312/python.exe")
'''
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

from promptkit.skill.errors import PythonInterpreterError

logger = logging.getLogger(__name__)

#: 指定解释器的环境变量名
ENV_VAR = "PROMPTKIT_PYTHON"

#: 探测超时（秒）；正常解释器启动远小于这个值
_PROBE_TIMEOUT = 15.0

#: 探测结果缓存：key 是 (绝对路径, mtime)，避免每次 tools() 都起子进程
_PROBE_CACHE: Dict[Tuple[str, float], bool] = {}


def _locate(candidate: Union[str, Path]) -> str:
    '''把裸名字（`python3`）解析成绝对路径；带目录的按原样展开。'''
    text = os.path.expanduser(os.fspath(candidate))
    if os.path.dirname(text):
        return os.path.abspath(text)
    return shutil.which(text) or text


def _probe(interpreter: str) -> bool:
    '''判断解释器是否真的能跑起来（不抛异常，只返回结果）。'''
    if not interpreter or not os.path.isfile(interpreter):
        return False
    try:
        stamp = os.path.getmtime(interpreter)
    except OSError:
        stamp = 0.0
    key = (interpreter, stamp)
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        completed = subprocess.run(
            [interpreter, "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        ok = False
    else:
        ok = completed.returncode == 0 and bool(completed.stdout.strip())
    _PROBE_CACHE[key] = ok
    return ok


def _venv_pythons(root: Union[str, Path]) -> Iterator[Path]:
    '''技能包根目录下 `.venv` / `venv` 里可能存在的解释器。'''
    base = Path(root)
    for name in (".venv", "venv"):
        environment = base / name
        if os.name == "nt":
            yield environment / "Scripts" / "python.exe"
        yield environment / "bin" / "python"


def resolve_python(
    python: Optional[Union[str, Path]] = None,
    *,
    root: Optional[Union[str, Path]] = None,
    detect_venv: bool = True,
) -> str:
    '''按优先级返回一个能用的 Python 解释器路径。

    参数：

    - `python`：显式指定的解释器（路径或裸命令名）；不可用时直接抛错
    - `root`：技能包根目录；给了才会探测它下面的 `.venv` / `venv`
    - `detect_venv`：是否探测技能包虚拟环境，默认开启；关掉则只看当前解释器
      和 PATH

    环境变量 `PROMPTKIT_PYTHON` 优先于自动探测，写错时抛
    `PythonInterpreterError` 而不是悄悄退回其他解释器。
    '''
    if python:
        located = _locate(python)
        if _probe(located):
            return located
        raise PythonInterpreterError(
            f"指定的 Python 解释器不可用：{python}",
            hint=f"确认路径存在且能运行；或改用环境变量 {ENV_VAR} 指定",
        )

    configured = os.environ.get(ENV_VAR)
    if configured:
        located = _locate(configured)
        if _probe(located):
            return located
        raise PythonInterpreterError(
            f"环境变量 {ENV_VAR} 指向的解释器不可用：{configured}",
            hint=f"修正或删除环境变量 {ENV_VAR}，让它指向一个可运行的 python",
        )

    candidates: list[Tuple[str, str]] = []
    if detect_venv and root is not None:
        for path in _venv_pythons(root):
            candidates.append((str(path), "技能包虚拟环境"))
    candidates.append((sys.executable, "当前解释器"))
    for name in ("python3", "python", "py"):
        found = shutil.which(name)
        if found:
            candidates.append((found, f"PATH 中的 {name}"))

    for candidate, source in candidates:
        if not candidate:
            continue
        located = _locate(candidate)
        if not os.path.isfile(located):
            # 大多数人不会给每个技能包都建 venv，不存在就安静跳过
            continue
        if _probe(located):
            return located
        logger.warning("跳过不可用的 Python 解释器（%s）：%s", source, candidate)

    raise PythonInterpreterError(
        "找不到可用的 Python 解释器",
        hint=f"设置环境变量 {ENV_VAR}=<python 完整路径>，把 Python 加入 PATH，"
        "或在技能包根目录放一个 .venv",
    )
