'''根据主题和要点生成 Markdown 报告。'''
import argparse
from pathlib import Path


def build(topic: str, points, heading: str) -> str:
    lines = [f"{heading} {topic}", ""]
    lines.extend(f"- {point}" for point in points)
    lines.extend(["", "## 下一步", "", "补充数据来源后复核结论。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="根据主题和要点生成 Markdown 报告")
    parser.add_argument("--topic", required=True, help="报告主题")
    parser.add_argument("--points", nargs="+", required=True, help="报告要点")
    parser.add_argument("--out", default="out/report.md", help="输出文件，相对技能目录")
    parser.add_argument("--heading", default="#", help="标题前缀")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    content = build(args.topic, args.points, args.heading)
    if args.dry_run:
        print(content)
        return 0

    target = Path(args.out).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
