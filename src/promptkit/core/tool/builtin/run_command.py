import locale
import platform
import subprocess

from promptkit.core.tool.base import Tool

DEFAULT_TIMEOUT = 30


def _default_encoding() -> str:
    '''本机编码；`locale.getencoding()` 只有 3.11+ 才有，老版本退回首选编码。

    mypy 按 `python_version = "3.10"` 检查，所以不能直接写死调用。
    '''
    getencoding = getattr(locale, "getencoding", None)
    if getencoding is not None:
        return str(getencoding())
    return locale.getpreferredencoding(False)


def _decode(raw: bytes) -> str:
    '''按 utf-8 -> 本机编码 -> 兜底 的顺序解码。

    Windows 上同一个 cmd 有时吐 UTF-8、有时吐 GBK（中文卷标就是这样），
    写死一个编码不是**解码崩掉**就是乱码。崩掉的后果尤其糟：
    subprocess 的读取线程会死，stdout 变成 None，看起来就像「命令没输出」。
    '''
    if not raw:
        return ""
    for encoding in ("utf-8", _default_encoding(), "gbk"):
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ========================= 工具实现 ===============================
def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    '''在当前系统上运行一条 shell 命令，返回它的输出（stdout + stderr）。

    Windows 走 cmd.exe，Linux / macOS 走 /bin/sh。
    '''
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"错误：命令超时（超过 {timeout} 秒仍未结束）：{command}"
    except Exception as exc:  # noqa: BLE001 - 把错误文本交给模型
        return f"错误：命令启动失败 {type(exc).__name__}: {exc}"

    out = _decode(result.stdout).strip()
    err = _decode(result.stderr).strip()

    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    if not parts:
        # 明确说「没有输出」，别让模型以为是工具坏了然后一直重试
        parts.append(f"（命令执行了，但没有任何输出；退出码 {result.returncode}）")
    elif result.returncode != 0:
        parts.append(f"[退出码 {result.returncode}]")
    return "\n".join(parts)


# ========================= 工具注册 ===============================
_SYSTEM = platform.system()
_SHELL_HINT = (
    "Windows 用 cmd 语法（dir、%USERPROFILE%、type），不要用 ls / ~ / cat"
    if _SYSTEM == "Windows"
    else "用 sh 语法（ls、$HOME、cat）"
)

RUN_COMMAND = Tool.from_function(
    run_command,
    name="run_command",
    # 把系统信息写进描述：不写的话模型会默认这是 Linux，
    # 然后一直试 ls ~/Desktop 之类的命令，撞到工具调用上限
    description=(
        "在当前机器上运行一条 shell 命令并返回它的输出。"
        f"当前系统是 {_SYSTEM}（{platform.release()}），{_SHELL_HINT}。"
        "参数 command 是一整行命令。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的完整命令行"},
        },
        "required": ["command"],
    },
)
