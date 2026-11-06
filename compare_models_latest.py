"""
Evaluate YOLO26 and RF-DETR on the SAME validation set with the SAME metric code,
so the numbers are directly comparable (each framework's own mAP is computed slightly
differently).

Usage
  python compare_models.py --val datasets/ppe/coco/valid \
      --yolo runs/yolo26/yolo26m_960_s0/weights/best.pt --yolo-imgsz 960 \
      --rfdetr runs/rfdetr/rfdetr_medium_960_s0/checkpoint_best_total.pth \
      --rfdetr-size medium --rfdetr-resolution 960 --conf 0.4

Reports per class:
  AP50, AP50-95   ranking quality over all thresholds (COCO-style, 101-point)
  P, R @ --conf   precision / recall at the threshold you'd actually deploy with (IoU 0.5)
Also writes compare_results.csv.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

IOUS = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------- ground truth / predictions

def load_gt(val_dir):
    data = json.loads((Path(val_dir) / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    gt = {im["file_name"]: defaultdict(list) for im in data["images"]}
    id2name = {im["id"]: im["file_name"] for im in data["images"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        gt[id2name[a["image_id"]]][cats[a["category_id"]]].append([x, y, x + w, y + h])
    return gt, [cats[k] for k in sorted(cats)]


def predict_yolo(weights, images, imgsz, device):
    from ultralytics import YOLO
    model = YOLO(weights)
    preds, times = {}, []
    for p in images:
        t = time.perf_counter()
        r = model.predict(str(p), conf=0.001, imgsz=imgsz, device=device, verbose=False,
                          max_det=300)[0]
        times.append(time.perf_counter() - t)
        names = [model.names[int(c)] for c in r.boxes.cls.tolist()]
        preds[p.name] = list(zip(names, r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
    return preds, times


def predict_rfdetr(weights, images, size, resolution):
    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
           "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[size]
    model = cls(pretrain_weights=weights, resolution=resolution)
    preds, times = {}, []
    for p in images:
        t = time.perf_counter()
        d = model.predict(str(p), threshold=0.001, include_source_image=False)
        times.append(time.perf_counter() - t)
        names = list(d.data["class_name"]) if "class_name" in d.data else \
            [model.class_names[int(c)] for c in d.class_id]
        preds[p.name] = list(zip(names, d.confidence.tolist(), d.xyxy.tolist()))
    return preds, times


# ---------------------------------------------------------------- metrics

def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def evaluate(gt, preds, classes, conf_thr, max_dets=100):
    results = {}
    for c in classes:
        n_gt = sum(len(g[c]) for g in gt.values())
        if n_gt == 0:
            continue
        scores, tp = [], []                       # tp: (n_dets, n_iou) bool
        for img, g in gt.items():
            dets = sorted([d for d in preds.get(img, []) if d[0] == c],
                          key=lambda d: -d[1])[:max_dets]
            if not dets:
                continue
            gboxes = g[c]
            img_tp = np.zeros((len(dets), len(IOUS)), bool)
            if gboxes:
                ious = iou_matrix([d[2] for d in dets], gboxes)
                for t_i, thr in enumerate(IOUS):
                    used = np.zeros(len(gboxes), bool)
                    for d_i in range(len(dets)):           # highest score first
                        cand = np.where(~used & (ious[d_i] >= thr))[0]
                        if len(cand):
                            used[cand[np.argmax(ious[d_i, cand])]] = True
                            img_tp[d_i, t_i] = True
            scores += [d[1] for d in dets]
            tp += list(img_tp)
        if not scores:
            results[c] = dict(n=n_gt, ap50=0.0, ap=0.0, p=0.0, r=0.0)
            continue
        scores, tp = np.array(scores), np.array(tp)
        order = np.argsort(-scores, kind="stable")
        scores, tp = scores[order], tp[order]
        aps = []
        for t_i in range(len(IOUS)):
            ctp = np.cumsum(tp[:, t_i])
            cfp = np.cumsum(~tp[:, t_i])
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            prec = np.maximum.accumulate(prec[::-1])[::-1]       # precision envelope
            rs = np.linspace(0, 1, 101)
            idx = np.searchsorted(rec, rs, side="left")
            aps.append(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))
        keep = scores >= conf_thr
        tp_at = tp[keep, 0].sum()
        p = tp_at / max(keep.sum(), 1)
        r = tp_at / n_gt
        results[c] = dict(n=n_gt, ap50=aps[0], ap=float(np.mean(aps)), p=p, r=r)
    return results


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="datasets/ppe/coco/valid")
    ap.add_argument("--yolo")
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--rfdetr")
    ap.add_argument("--rfdetr-size", default="medium")
    ap.add_argument("--rfdetr-resolution", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.4, help="deployment threshold for P/R")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="compare_results.csv")
    args = ap.parse_args()
    if not (args.yolo or args.rfdetr):
        ap.error("give --yolo and/or --rfdetr")

    gt, classes = load_gt(args.val)
    images = [Path(args.val) / n for n in gt]
    print(f"{len(images)} val images")

    runs = {}
    if args.yolo:
        print("Running YOLO26 ...")
        runs["YOLO26"] = predict_yolo(args.yolo, images, args.yolo_imgsz, args.device)
    if args.rfdetr:
        print("Running RF-DETR ...")
        runs["RF-DETR"] = predict_rfdetr(args.rfdetr, images, args.rfdetr_size,
                                         args.rfdetr_resolution)

    res = {m: evaluate(gt, preds, classes, args.conf) for m, (preds, _) in runs.items()}
    models = list(res)
    cls_list = [c for c in classes if any(c in res[m] for m in models)]

    head = f"{'class':24s} {'n':>5s}" + "".join(
        f" | {m:^29s}" for m in models)
    sub = f"{'':24s} {'':>5s}" + "".join(
        f" | {'AP50':>6s} {'AP':>6s} {'P':>6s} {'R':>6s}" for _ in models)
    print(f"\nP / R at conf >= {args.conf}, IoU 0.5\n" + head + "\n" + sub)
    rows = []
    for c in cls_list:
        line = f"{c:24s} {res[models[0]][c]['n']:5d}"
        row = [c, res[models[0]][c]["n"]]
        for m in models:
            r = res[m][c]
            line += f" | {r['ap50']:6.3f} {r['ap']:6.3f} {r['p']:6.3f} {r['r']:6.3f}"
            row += [round(r["ap50"], 4), round(r["ap"], 4), round(r["p"], 4), round(r["r"], 4)]
        print(line)
        rows.append(row)
    line = f"{'MEAN':24s} {'':5s}"
    for m in models:
        v = [res[m][c] for c in cls_list]
        line += " | " + " ".join(f"{np.mean([x[k] for x in v]):6.3f}"
                                 for k in ("ap50", "ap", "p", "r"))
    print(line)

    print("\nPyTorch latency per image (indicative only; benchmark TensorRT FP16 for deployment):")
    for m, (_, times) in runs.items():
        t = np.array(times[3:] or times) * 1000             # skip warm-up
        print(f"  {m:8s} median {np.median(t):6.1f} ms")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_val"] + [f"{m}_{k}" for m in models
                                         for k in ("AP50", "AP50-95", "P", "R")])
        w.writerows(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
