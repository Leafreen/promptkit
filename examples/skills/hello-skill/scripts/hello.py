"""向指定名字打招呼，把问候写进 greeting.txt。"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="向指定名字打招呼")
    parser.add_argument("--name", default="世界", help="跟谁打招呼")
    parser.add_argument("--out", default="greeting.txt", help="输出文件名")
    args = parser.parse_args()

    text = f"你好，{args.name}！这是 hello-skill 的问候。"
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(json.dumps({"wrote": args.out, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
