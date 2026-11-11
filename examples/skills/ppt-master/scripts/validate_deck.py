"""校验 PPTX：打印页数与每页标题（JSON）。"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 PPTX 页数与标题")
    parser.add_argument("--deck", required=True, help="要校验的 pptx 路径")
    args = parser.parse_args()

    try:
        from pptx import Presentation
    except ImportError:
        raise SystemExit("缺少依赖：请先 pip install python-pptx")

    deck = Presentation(args.deck)
    titles = [
        slide.shapes.title.text if slide.shapes.title else ""
        for slide in deck.slides
    ]
    print(json.dumps({"slides": len(titles), "titles": titles}, ensure_ascii=False))


if __name__ == "__main__":
    main()
