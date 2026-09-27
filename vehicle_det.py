#!/usr/bin/env python3
"""
Inference script for a vehicle-detection .pt checkpoint of unknown origin.

Step 1 (always runs): inspects the checkpoint and prints what kind of model it is
and the class names stored inside it, so you don't have to guess the labels.
Step 2: runs detection on an image, a folder, a video, or a webcam, saves
annotated output and a detections.csv.

Usage:
    pip install ultralytics opencv-python
    python infer.py --weights model.pt --inspect-only
    python infer.py --weights model.pt --source test.jpg
    python infer.py --weights model.pt --source images_folder/ --conf 0.3
    python infer.py --weights model.pt --source traffic.mp4
    python infer.py --weights model.pt --source 0 --show        # webcam

Note: .pt files are Python pickles, so only load checkpoints you trust.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# 1. Checkpoint inspection
# --------------------------------------------------------------------------- #
def inspect_checkpoint(weights: str) -> str:
    """Print what's inside the .pt and return a guess of its format."""
    print(f"\n=== Inspecting {weights} ===")
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"torch.load failed: {e}")
        print("If the error mentions a missing module such as 'models' or 'utils', "
              "it's probably an original YOLOv5 checkpoint (use --backend yolov5).")
        return "unknown"

    fmt = "unknown"
    names = None

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        model = ckpt.get("ema") or ckpt.get("model")
        if model is not None and hasattr(model, "names"):
            names = model.names
        mod = type(model).__module__ if model is not None else ""
        if mod.startswith("ultralytics"):
            fmt = "ultralytics"
        elif mod.startswith("models"):
            fmt = "yolov5"
        elif model is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            fmt = "state_dict"
        for k in ("train_args", "yaml", "date", "version"):
            if k in ckpt and ckpt[k]:
                val = ckpt[k]
                if isinstance(val, dict):
                    val = {kk: val[kk] for kk in list(val)[:12]}
                print(f"{k}: {val}")
    elif isinstance(ckpt, torch.nn.Module):
        names = getattr(ckpt, "names", None)
        fmt = "full_module"

    print(f"Detected format: {fmt}")
    if names is not None:
        print("Class names stored in the model:")
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for i, n in items:
            print(f"  {i}: {n}")
    else:
        print("No class names found in the checkpoint.")

    if fmt == "state_dict":
        print("\nThis is only a state_dict (weights without the architecture). "
              "You'll need the model class definition from your coworker to use it.")
    print("=" * 40 + "\n")
    return fmt


# --------------------------------------------------------------------------- #
# 2. Inference
# --------------------------------------------------------------------------- #
def run_ultralytics(args, out_dir: Path):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names
    print(f"Loaded with Ultralytics. Classes: {names}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        classes=args.classes,
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        show=args.show,
        stream=True,  # don't hold every video frame in memory
        verbose=False,
    )

    rows, counts = [], {}
    for frame_idx, r in enumerate(results):
        src = Path(r.path).name
        for box in r.boxes:
            cls_id = int(box.cls)
            label = names[cls_id]
            x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
            rows.append([src, frame_idx, cls_id, label, round(float(box.conf), 4), x1, y1, x2, y2])
            counts[label] = counts.get(label, 0) + 1
    return rows, counts


def run_yolov5(args, out_dir: Path):
    """Fallback for checkpoints trained with the original ultralytics/yolov5 repo."""
    model = torch.hub.load("ultralytics/yolov5", "custom", path=args.weights)
    model.conf, model.iou = args.conf, args.iou
    if args.classes is not None:
        model.classes = args.classes
    names = model.names
    print(f"Loaded with YOLOv5 hub. Classes: {names}")

    src = Path(args.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    if any(f.suffix.lower() not in IMG_EXTS for f in files):
        sys.exit("The YOLOv5 fallback here handles images only. For video, use "
                 "yolov5's own detect.py or convert the model to Ultralytics format.")

    rows, counts = [], {}
    results = model([str(f) for f in files], size=args.imgsz)
    results.save(save_dir=str(out_dir), exist_ok=True)
    for f, det in zip(files, results.xyxy):
        for x1, y1, x2, y2, conf, cls_id in det.tolist():
            label = names[int(cls_id)]
            rows.append([f.name, 0, int(cls_id), label, round(conf, 4),
                         round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            counts[label] = counts.get(label, 0) + 1
    return rows, counts


def main():
    ap = argparse.ArgumentParser(description="Vehicle detection inference")
    ap.add_argument("--weights", required=True, help="path to the .pt file")
    ap.add_argument("--source", help="image, folder, video, or webcam index (e.g. 0)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--device", default=None, help="'cpu', '0' for GPU 0, etc.")
    ap.add_argument("--classes", type=int, nargs="+", help="only keep these class ids")
    ap.add_argument("--save-dir", default="runs/infer", help="output folder")
    ap.add_argument("--show", action="store_true", help="display results in a window")
    ap.add_argument("--backend", choices=["auto", "ultralytics", "yolov5"], default="auto")
    ap.add_argument("--inspect-only", action="store_true", help="only print checkpoint info")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"Weights not found: {args.weights}")

    fmt = inspect_checkpoint(args.weights)
    if args.inspect_only:
        return
    if not args.source:
        sys.exit("Give --source (image, folder, video or webcam index).")
    if fmt == "state_dict":
        sys.exit("Can't run a bare state_dict without its model definition.")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        backend = "yolov5" if fmt == "yolov5" else "ultralytics"

    try:
        rows, counts = run_ultralytics(args, out_dir) if backend == "ultralytics" else run_yolov5(args, out_dir)
    except Exception as e:
        if args.backend == "auto" and backend == "ultralytics":
            print(f"Ultralytics loader failed ({e}); trying the YOLOv5 fallback...")
            rows, counts = run_yolov5(args, out_dir)
        else:
            raise

    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "frame", "class_id", "label", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    print(f"\nTotal detections: {len(rows)}")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {n}")
    print(f"Annotated output: {out_dir}")
    print(f"Detections CSV:   {csv_path}")


if __name__ == "__main__":
    main()
