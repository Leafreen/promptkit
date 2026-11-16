'''用 PromptKit 自己介绍自己：加载 ppt-master 技能包，构建并校验一份 PPT。

    python examples/promptkit_intro_deck.py

流程（三步工作流）：
1. prepare  —— 把大纲写成 config.json
2. build    —— 调技能包里的 build_deck.py 生成 pptx（run_script）
3. validate —— 调 validate_deck.py 校验页数与标题

产物：outputs/promptkit-intro.pptx
'''
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.core import FunctionStep, Verbosity, Workflow  # noqa: E402
from promptkit.skill import SkillPackage  # noqa: E402

SKILL_DIR = PROJECT_DIR / "examples" / "skills" / "ppt-master"
OUT_DIR = PROJECT_DIR / "outputs"
CONFIG_PATH = OUT_DIR / "promptkit-intro.json"
DECK_PATH = OUT_DIR / "promptkit-intro.pptx"

DECK = {
    "title": "PromptKit：用 Python 描述 LLM 工作流",
    "slides": [
        {
            "title": "它解决什么问题",
            "bullets": [
                "提示词、工具定义、流程控制散落在业务代码里，改一处要翻十个文件",
                "换模型厂商要重写调用；加一个工具要同步改提示词和参数表",
                "多步任务靠手写 if/while 拼接，难复用、难测试、难观察",
                "PromptKit 把它们收敛成可组合的声明式步骤",
            ],
        },
        {
            "title": "三个核心抽象",
            "bullets": [
                "Prompt —— 带 {{变量}} 模板的消息，支持 {{user.name}}、{{hits.0.text}}",
                "Agent —— 模型 + 工具 + 约束，默认 deepseek，可切 OpenAI / Anthropic / 豆包",
                "Workflow —— 步骤的容器：LLMStep、If、While、ForEach、Function 随意嵌套",
            ],
        },
        {
            "title": "最小工作流",
            "bullets": [
                "Workflow().add(",
                "    Role(\"技术编辑\"),",
                "    Want(\"把下面的草稿整理成三条要点\"),",
                "    LLMStep(prompt=Prompt(role=\"user\", content=\"{{draft}}\"), output_key=\"brief\"),",
                ")",
                "wf.run(agent=Agent(), verbose=Verbosity.NORMAL)",
            ],
        },
        {
            "title": "工作流 DSL 速查",
            "bullets": [
                "顺序与分支：If(cond).Then(...).Else(...)、Switch、Goto",
                "循环：While、For、ForEach，支持 max_iterations 与 Break/Continue",
                "复用：Function(name=...).Do(...) 定义步骤组，按参数绑定后反复调用",
                "表达式：Var、Read、Transform、SessionPop，以及语义条件 judge",
            ],
        },
        {
            "title": "工具调用与对话",
            "bullets": [
                "Tool(name, description, parameters, func) 描述一个可调用能力",
                "ToolRegistry 统一注册；ChatWorkflow 负责多轮对话与工具回填",
                "MCP 接入带安全门：白名单外一律 PermissionError，白名单为空则全部拒绝",
                "工具 Schema 自动转成各家 function calling 格式",
            ],
        },
        {
            "title": "Skill：把「知识」变成可加载单元",
            "bullets": [
                "纯 Markdown 技能：一段说明 + 一份参数表，as_skill() 即可挂给模型",
                "目录式技能包：SKILL.md + resources/ + scripts/ + requirements.txt",
                "SkillPackage.load(dir) 校验 frontmatter：name 必须是 kebab-case",
                "渐进披露：正文按需读取，资源清单只附在附录里，不占上下文",
            ],
        },
        {
            "title": "技能包的工具集",
            "bullets": [
                "read_skill —— 按需读取正文或某一节",
                "list_resources / read_resource —— 列资源、读资源，限定在包内",
                "list_scripts / run_script —— 脚本不逐个挂工具，按名字 + 字符串参数执行",
                "write_file —— 只写工作目录，技能包本体只读",
                "run_command —— 默认关闭；启用后还必须过 allow_commands 白名单",
            ],
        },
        {
            "title": "安全边界",
            "bullets": [
                "资源路径必须落在技能目录内，越界直接拒绝",
                "脚本只能运行加载时登记过的 .py，不走 shell",
                "命令白名单按首词匹配，python 也匹配绝对路径的 python.exe",
                "超时、输出截断、非零退出码由调用方判断，不静默成功",
            ],
        },
        {
            "title": "多包合并",
            "bullets": [
                "tools(namespace=True) 给工具名加前缀：ppt_master__run_script",
                "不同技能包的 read_skill / write_file 因此不会重名",
                "merge_tools() 一次合并多个包，重名会抛 DuplicateToolError",
                "enable / disable 按规范名裁剪，白名单 tools: [...] 可写进 frontmatter",
            ],
        },
        {
            "title": "快速开始",
            "bullets": [
                "pip install -e .",
                "在 .env 里填一家厂商的 API key（OPENAI_API_KEY 等）",
                "from promptkit.core import Workflow, Prompt, LLMStep",
                "Workflow().add(LLMStep(Prompt(role=\"user\", content=\"你好\"))).run(Agent())",
                "示例：hello-skill、ppt-master、skill_package.py",
            ],
        },
        {
            "title": "这份 PPT 就是示例",
            "bullets": [
                "技能包：examples/skills/ppt-master（SKILL.md + scripts + references）",
                "工作流：prepare → build → validate 三步，全部通过 run_script 执行",
                "产物：outputs/promptkit-intro.pptx，可直接编辑",
                "换一套 config.json 就能生成新的 PPT",
            ],
        },
    ],
}


def prepare(ctx):
    '''把大纲写到 config.json，返回路径。'''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(DECK, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(CONFIG_PATH)


def build(ctx):
    '''调技能包脚本生成 pptx。'''
    result = package.run_script(
        "build_deck.py",
        args=f'--config "{CONFIG_PATH}" --out "{DECK_PATH}"',
        python=sys.executable,
        cwd=OUT_DIR,
    )
    print("[build]", result)
    return result


def validate(ctx):
    '''调技能包脚本校验产物。'''
    result = package.run_script(
        "validate_deck.py",
        args=f'--deck "{DECK_PATH}"',
        python=sys.executable,
        cwd=OUT_DIR,
    )
    print("[validate]", result)
    return result


package = SkillPackage.load(SKILL_DIR)


def main() -> int:
    workflow = Workflow().add(
        FunctionStep(prepare, output_key="config_path", name="prepare"),
        FunctionStep(build, output_key="build_log", name="build"),
        FunctionStep(validate, output_key="validate_log", name="validate"),
    )
    workflow.run(verbose=Verbosity.NORMAL)

    if not DECK_PATH.is_file():
        print("没生成 PPT，检查上面的输出")
        return 1
    print(f"\n完成：{DECK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
