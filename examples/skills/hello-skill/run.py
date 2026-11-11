'''hello-skill 离线自检：加载包 → 打印工具 → 真跑一次脚本（不联网）。

    python examples/skills/hello-skill/run.py

这是「约定是否成立」的最终验证：每阶段结束都必须能跑通。
阶段 2 落地 `tools(cwd=)` 读写分离后，脚本产物写到临时目录，不污染示例包。
'''
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from promptkit.skill import SkillPackage  # noqa: E402


def main() -> int:
    # 阶段 1：新门面冒烟 —— 严格校验 + 门面字段
    package = SkillPackage.load(HERE)
    assert package.name == "hello-skill"
    assert package.has_scripts and package.has_resources
    assert "Hello Skill" in package.skill_md

    # 阶段 2：tools(cwd=) 读写分离 —— 产物写到临时目录，不污染示例包
    workdir = Path(tempfile.mkdtemp(prefix="hello_skill_"))
    try:
        tools = {tool.name: tool for tool in package.tools(cwd=workdir)}
        print("工具：" + "、".join(tools))

        result = tools["run_script"].call(
            {"name": "hello.py", "args": "--name PromptKit"}
        )
        print(f"\nrun_script → {result}")

        greeting = workdir / "greeting.txt"
        assert greeting.is_file(), f"脚本没写 greeting.txt 到 {workdir}"
        print(f"greeting.txt 内容：{greeting.read_text(encoding='utf-8').strip()}")

        resource = package.read_resource("resources/note.md")
        assert "示例资源" in resource, "read_resource 读不到 resources/note.md"
        print("read_resource(resources/note.md) 正常")

        print("\nhello-skill 自检通过 ✔")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
