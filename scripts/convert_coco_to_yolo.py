import json
from pathlib import Path
from collections import defaultdict


def convert(ann_path: str | Path, out_dir: str | Path) -> None:
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(ann_path, encoding="utf-8-sig"))
    categories = {cat["id"]: i for i, cat in enumerate(sorted(data["categories"], key=lambda c: c["id"]))}
    images = {img["id"]: img for img in data["images"]}
    ann_per_image = defaultdict(list)
    for ann in data["annotations"]:
        ann_per_image[ann["image_id"]].append(ann)
    for img_id, img in images.items():
        img_width, img_height = img["width"], img["height"]
        file_name = Path(img["file_name"]).stem + ".txt"
        lines = []
        for ann in ann_per_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / img_width
            yc = (y + h / 2) / img_height
            wn = w / img_width
            hn = h / img_height
            cls = categories[ann["category_id"]]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        (out_dir / file_name).write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("ann", type=str, help="Path to COCO annotations JSON")
    parser.add_argument("out", type=str, help="Output directory for YOLO label files")
    args = parser.parse_args()
    convert(args.ann, args.out)
