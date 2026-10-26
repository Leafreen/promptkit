'''运行配置：密钥查找、模型名、默认系统提示。

和 Agent 分开：这里只管「配置从哪来」，Agent 只管「怎么干活」。
这样换模型 / 换密钥来源不用碰 Agent，也方便单独测试 .env 读取逻辑。

这里的 `DEFAULT_MODEL` / `DEFAULT_BASE_URL` / 思考参数都是 **DeepSeek 的默认值**：
想换厂商不用改这个文件，用 `Agent(vendor="openai")` / `Agent(vendor="anthropic")`，
厂商表在 `client_sys/registry.py`（见 README 第 22 节）。
'''
import os
from typing import Dict, List, Optional

# 包安装目录：源码开发时是 src/promptkit，装到 site-packages 后是 site-packages/promptkit
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_env_file() -> str:
    '''默认 .env 查找位置：优先当前工作目录，其次包目录。

    src 布局下 `.env` 通常放在项目根目录（脚本的 CWD），
    不再和包代码放在一起；安装后兜底退回包目录。
    '''
    cwd_env = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(cwd_env):
        return cwd_env
    return os.path.join(PROJECT_ROOT, ".env")


ENV_FILE = _default_env_file()

ENV_VAR = "DEEPSEEK_API_KEY"

DEFAULT_START_PROMPTS: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# 本项目默认使用的 DeepSeek 模型
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# 思考模式参数（DeepSeek 特有；其他厂商要改这里）
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_EXTRA_BODY: Dict[str, object] = {"thinking": {"type": "enabled"}}


def load_env_file(path: str = ENV_FILE) -> None:
    '''极简 .env 读取：一行一条 KEY=VALUE。

    已存在的环境变量优先，不会被覆盖，所以 export 的优先级高于 .env。
    文件不存在就什么都不做。
    '''
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def resolve_api_key(
    api_key: Optional[str] = None, *, env_path: str = ENV_FILE
) -> str:
    '''按 `参数 -> 环境变量 -> .env` 的顺序找密钥，找不到就抛错。'''
    load_env_file(env_path)
    key = api_key or os.environ.get(ENV_VAR)
    if not key:
        raise ValueError(
            f"没有找到 API Key：请设置环境变量 {ENV_VAR}，"
            f"或在 {ENV_FILE} 里写一行 {ENV_VAR}=sk-xxxx"
        )
    return key


def api_key_source(api_key: Optional[str] = None) -> str:
    '''密钥是从哪来的（排查问题时有用）。'''
    if api_key:
        return "参数"
    if os.environ.get(ENV_VAR):
        return f"环境变量 {ENV_VAR}"
    return f"{ENV_FILE}"
