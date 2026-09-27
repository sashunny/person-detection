"""
Run the trained RT-DETR Palm-FFB model on a folder of images (CPU only).

For each image it saves a copy with
    GREEN boxes = ground truth (if annotations are available)
    RED boxes   = predictions (with confidence)
and writes counts.csv with GT count, predicted count and error per file.

Install once:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install transformers pillow

Usage:
    python predict_local.py --model best --images test_images
    python predict_local.py --model best --images test_images --ann test_images/_annotations.coco.json
    python predict_local.py --model best --images test_images --labels test_labels   # YOLO .txt labels
    python predict_local.py --model best --images test_images --thr 0.4

Ground truth (optional, auto-detected):
    - COCO JSON: "_annotations.coco.json" inside the images folder (Roboflow COCO export), or --ann
    - YOLO txt : one .txt per image (same file name) in --labels folder
    If neither is found, only predictions are drawn and GT columns are left empty.
"""

import argparse
import csv
import json
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GREEN = (0, 220, 0)
RED = (255, 50, 0)


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Palm-FFB detection + counting on a folder of images (CPU).")
    p.add_argument("--model", required=True, help="Folder downloaded from Kaggle (the 'best' folder).")
    p.add_argument("--images", required=True, help="Folder with input images.")
    p.add_argument("--out", default="predictions", help="Output folder (default: predictions).")
    p.add_argument("--ann", default=None, help="COCO JSON with ground truth (optional).")
    p.add_argument("--labels", default=None, help="Folder of YOLO .txt labels (optional).")
    p.add_argument("--thr", type=float, default=None,
                   help="Confidence threshold. Default: value in <model>/count_threshold.json, else 0.5.")
    p.add_argument("--batch", type=int, default=4, help="Images per forward pass (default 4).")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default).")
    return p.parse_args()


# ----------------------------------------------------------------------------- ground truth
def load_coco_gt(ann_path):
    """Returns {file_name: [[x1, y1, x2, y2], ...]} using only categories that have annotations
    (skips the empty Roboflow 'supercategory')."""
    with open(ann_path) as f:
        coco = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {fn: [] for fn in id2file.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if w < 1 or h < 1:
            continue
        gt[id2file[a["image_id"]]].append([x, y, x + w, y + h])
    return gt


def load_yolo_gt(label_dir, file_name, img_w, img_h):
    txt = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
    if not os.path.exists(txt):
        return None
    boxes = []
    with open(txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return boxes


# ----------------------------------------------------------------------------- drawing
def get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw, xy, text, color, font):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    y = max(0, y - th - 6)
    draw.rectangle([x, y, x + tw + 8, y + th + 6], fill=color)
    draw.text((x + 4, y + 3 - t), text, fill=(255, 255, 255), font=font)


def draw_result(img, gt_boxes, pred_boxes, pred_scores, thr):
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(min(w, h) / 300))
    font = get_font(max(12, round(min(w, h) / 45)))
    big = get_font(max(16, round(min(w, h) / 25)))

    for b in gt_boxes or []:
        d.rectangle(b, outline=GREEN, width=lw)
    for b, s in zip(pred_boxes, pred_scores):
        d.rectangle(b, outline=RED, width=lw)
        draw_label(d, (b[0], b[1]), f"{s:.2f}", RED, font)

    gt_txt = "GT: -" if gt_boxes is None else f"GT: {len(gt_boxes)}"
    header = f"Pred: {len(pred_boxes)}   {gt_txt}   (thr {thr:.2f})"
    l, t, r, b = d.textbbox((0, 0), header, font=big)
    d.rectangle([0, 0, r - l + 20, b - t + 16], fill=(0, 0, 0))
    d.text((10, 8 - t), header, fill=(255, 255, 0), font=big)

    legend = "GREEN = Ground Truth   RED = Prediction"
    l, t, r, b = d.textbbox((0, 0), legend, font=font)
    d.rectangle([0, h - (b - t) - 14, r - l + 20, h], fill=(0, 0, 0))
    d.text((10, h - (b - t) - 7 - t), legend, fill=(255, 255, 255), font=font)
    return img


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # threshold
    thr = args.thr
    thr_file = os.path.join(args.model, "count_threshold.json")
    if thr is None and os.path.exists(thr_file):
        with open(thr_file) as f:
            thr = float(json.load(f)["threshold"])
    if thr is None:
        thr = 0.5
    print(f"Confidence threshold: {thr}")

    # model (CPU)
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model).eval()

    # images
    files = sorted(f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found in {args.images}")
    print(f"Found {len(files)} images")

    # ground truth source
    coco_gt = None
    ann = args.ann or os.path.join(args.images, "_annotations.coco.json")
    if os.path.exists(ann):
        coco_gt = load_coco_gt(ann)
        print(f"Ground truth: COCO JSON ({ann})")
    elif args.labels:
        print(f"Ground truth: YOLO labels ({args.labels})")
    else:
        print("Ground truth: none found (predictions only)")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    t0 = time.time()

    for start in range(0, len(files), args.batch):
        batch_files = files[start:start + args.batch]
        images = [Image.open(os.path.join(args.images, f)).convert("RGB") for f in batch_files]

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, threshold=thr, target_sizes=[(im.height, im.width) for im in images]
        )

        for fname, img, res in zip(batch_files, images, results):
            boxes = res["boxes"].tolist()
            scores = res["scores"].tolist()

            if coco_gt is not None:
                gt = coco_gt.get(fname)
            elif args.labels:
                gt = load_yolo_gt(args.labels, fname, img.width, img.height)
            else:
                gt = None

            out_img = draw_result(img, gt, boxes, scores, thr)
            out_name = os.path.splitext(fname)[0] + ".jpg"
            out_img.save(os.path.join(args.out, out_name), quality=92)

            pred_n = len(boxes)
            gt_n = None if gt is None else len(gt)
            rows.append({
                "file": fname,
                "gt_count": "" if gt_n is None else gt_n,
                "pred_count": pred_n,
                "error": "" if gt_n is None else pred_n - gt_n,
                "abs_error": "" if gt_n is None else abs(pred_n - gt_n),
            })
            print(f"[{len(rows):>4}/{len(files)}] {fname}: pred={pred_n}"
                  + ("" if gt_n is None else f"  gt={gt_n}  err={pred_n - gt_n:+d}"))

    elapsed = time.time() - t0

    # CSV
    csv_path = os.path.join(args.out, "counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "gt_count", "pred_count", "error", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)

    # summary
    total_pred = sum(r["pred_count"] for r in rows)
    print("\n================ SUMMARY ================")
    print(f"Images: {len(rows)}   time: {elapsed:.1f}s   ({elapsed / len(rows):.2f}s per image on CPU)")
    print(f"Total predicted bunches: {total_pred}")
    with_gt = [r for r in rows if r["gt_count"] != ""]
    if with_gt:
        total_gt = sum(r["gt_count"] for r in with_gt)
        mae = sum(r["abs_error"] for r in with_gt) / len(with_gt)
        bias = sum(r["error"] for r in with_gt) / len(with_gt)
        exact = sum(r["error"] == 0 for r in with_gt) / len(with_gt)
        within1 = sum(r["abs_error"] <= 1 for r in with_gt) / len(with_gt)
        print(f"Images with GT: {len(with_gt)}   total GT bunches: {total_gt}")
        print(f"Count MAE: {mae:.2f}   bias: {bias:+.2f}  (negative = undercounting)")
        print(f"Exact count: {exact * 100:.1f}%   within ±1: {within1 * 100:.1f}%")
    print(f"\nImages + counts.csv saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
