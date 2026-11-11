"""根据 config.json 生成可编辑的 PPTX（Quick 路线）。"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 config.json 生成 PPTX")
    parser.add_argument("--config", required=True, help="大纲 JSON 文件")
    parser.add_argument("--out", default="deck.pptx", help="输出 pptx 路径")
    args = parser.parse_args()

    try:
        from pptx import Presentation
    except ImportError:
        raise SystemExit("缺少依赖：请先 pip install python-pptx")

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    deck = Presentation()
    title_slide = deck.slides.add_slide(deck.slide_layouts[0])
    title_slide.shapes.title.text = config.get("title", "未命名")

    for item in config.get("slides", []):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = item.get("title", "")
        body = slide.placeholders[1].text_frame
        for index, bullet in enumerate(item.get("bullets", [])):
            para = body.paragraphs[0] if index == 0 else body.add_paragraph()
            para.text = str(bullet)

    deck.save(args.out)
    print(json.dumps({"saved": args.out, "slides": len(deck.slides)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
