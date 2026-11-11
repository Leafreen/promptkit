'''word-count skill 的干活脚本：统计文本文件，输出 JSON。

独立可跑（不依赖 promptkit）：

    python count_words.py <文件>
'''
import json
import re
import sys


def main(argv) -> int:
    if len(argv) != 2:
        print("用法：count_words.py <文件>", file=sys.stderr)
        return 2
    try:
        with open(argv[1], encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    chars = len(re.sub(r"\s", "", text))
    print(json.dumps({
        "chars": chars,                       # 不含空白的字数
        "lines": text.count("\n") + 1,
        "reading_minutes": round(chars / 400, 1),  # 按每分钟 400 字估算
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
