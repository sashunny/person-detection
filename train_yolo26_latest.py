"""
Fine-tune YOLO26 on the PPE dataset made by prepare_datasets.py.

Usage
  pip install -U ultralytics
  python train_yolo26.py --data datasets/ppe/yolo/data.yaml --size m --imgsz 960

  RTX 3070 (8 GB): --size s --batch 8   or   --size m --batch 4..6 at 960 px.
  --batch -1 lets Ultralytics pick the largest batch that fits.

Output (the exact folder is printed at the end; newer Ultralytics puts it under runs/detect/)
  .../yolo26/<name>/weights/best.pt      best checkpoint (by val mAP)
  .../yolo26/<name>/per_class_val.csv    per-class precision / recall / mAP on val
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ppe/yolo/data.yaml")
    ap.add_argument("--size", default="m", choices=list("nsmlx"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30, help="early stop after N epochs w/o gain")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--cache", action="store_true", help="cache images in RAM (faster)")
    args = ap.parse_args()

    name = args.name or f"yolo26{args.size}_{args.imgsz}_s{args.seed}"
    model = YOLO(f"yolo26{args.size}.pt")          # COCO-pretrained weights, auto-downloaded

    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
        batch=args.batch, device=args.device, workers=args.workers, seed=args.seed,
        cache="ram" if args.cache else False,
        project="runs/yolo26", name=name, exist_ok=True,
        amp=True,
        # augmentation: Ultralytics defaults (mosaic, HSV, flip, scale, translate) plus
        # a little extra for a small, fixed-camera dataset
        mixup=0.1, degrees=0.0, close_mosaic=15,
        plots=True,
    )

    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}\nValidating best weights on val ...")
    metrics = YOLO(best).val(data=args.data, split="val", imgsz=args.imgsz,
                             batch=args.batch, device=args.device, plots=True,
                             project="runs/yolo26", name=f"{name}_val", exist_ok=True)

    names = metrics.names
    rows = []
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        rows.append([names[int(cls_idx)], f"{p:.3f}", f"{r:.3f}", f"{ap50:.3f}", f"{ap:.3f}"])
    out_csv = best.parent.parent / "per_class_val.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "mAP50", "mAP50-95"])
        w.writerows(rows)

    print(f"\n{'class':24s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:>6s} {r[2]:>6s} {r[3]:>7s} {r[4]:>9s}")
    print(f"{'ALL':24s} {metrics.box.mp:6.3f} {metrics.box.mr:6.3f} "
          f"{metrics.box.map50:7.3f} {metrics.box.map:9.3f}")
    print(f"\nPer-class results saved to {out_csv}")
    print("Classes with no val instances are not listed (nothing to evaluate).")


if __name__ == "__main__":
    main()
