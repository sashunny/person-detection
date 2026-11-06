"""
Run a trained RF-DETR model on every video in a folder (CPU-friendly) and save:
  <out>/<video>_rfdetr.mp4           annotated video
  <out>/<video>_rfdetr.csv           every detection: frame, time, class, conf, box
  <out>/summary_rfdetr.csv           detections per class per video

Usage (laptop, no GPU)
  pip install -U "rfdetr" opencv-python
  python test_rfdetr_videos.py --weights checkpoint_best_total.pth --size medium \
      --resolution 960 --videos TEST --out TEST_results --fps 2

--size and --resolution must match training.
CPU speed: roughly 1-3 s per frame for RF-DETR-medium at 960 px (slower than YOLO on CPU).
--fps sets how many frames per second of video are processed (the output video plays at
that rate, so it keeps real-time duration). Lower --fps to go faster. Videos already done
are skipped unless --overwrite.
"""

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
import rfdetr

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".flv", ".wmv"}
TAG = "rfdetr"

# fixed colour per class (BGR) - same in the RF-DETR script, so videos are easy to compare
CLASSES = ["person", "truck", "helmet", "vest", "uniform", "non_compliant_clothing", "harness",
           "boots", "mask", "mask_improper", "face", "head", "non_safety_footwear"]
PALETTE = [(255, 180, 0), (128, 128, 128), (0, 200, 0), (0, 220, 255), (200, 120, 0),
           (0, 0, 255), (255, 0, 200), (120, 60, 0), (0, 255, 128), (0, 128, 255),
           (60, 60, 255), (0, 0, 180), (80, 0, 160)]


def color(name):
    if name in CLASSES:
        return PALETTE[CLASSES.index(name)]
    h = sum(map(ord, name))
    return (h * 37 % 255, h * 91 % 255, h * 53 % 255)


def draw(frame, dets, model_name):
    th = max(1, round(frame.shape[0] / 540))           # line thickness scales with resolution
    fs = 0.4 * th
    for name, conf, (x1, y1, x2, y2) in dets:
        c = color(name)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, c, th)
        label = f"{name} {conf:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y_txt = max(p1[1], tht + 4)
        cv2.rectangle(frame, (p1[0], y_txt - tht - 4), (p1[0] + tw + 2, y_txt), c, -1)
        cv2.putText(frame, label, (p1[0] + 1, y_txt - 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(frame, model_name, (10, 30 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * th,
                (255, 255, 255), 2 * th, cv2.LINE_AA)
    return frame


SIZES = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
         "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}


def detect(model, frame, args):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)           # RF-DETR expects RGB
    d = model.predict(rgb, threshold=args.conf, include_source_image=False)
    if "class_name" in d.data:
        names = [str(n) for n in d.data["class_name"]]
    else:
        names = [model.class_names[int(c)] for c in d.class_id]
    return [(n, c, b) for n, c, b in zip(names, d.confidence.tolist(), d.xyxy.tolist())
            if not n.startswith("__")]            # drop RF-DETR's "__background__" placeholder


def process_video(model, path, out_dir, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  could not open {path.name}, skipping")
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(src_fps / args.fps))
    out_fps = src_fps / step
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = out_dir / f"{path.stem}_{TAG}.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    counts, rows, t_infer = Counter(), [], []
    bar = tqdm(total=n_frames // step if n_frames else None, desc=path.name, unit="frame",
               dynamic_ncols=True)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = time.perf_counter()
            dets = detect(model, frame, args)
            t_infer.append(time.perf_counter() - t)
            for name, conf, (x1, y1, x2, y2) in dets:
                counts[name] += 1
                rows.append([idx, f"{idx / src_fps:.2f}", name, f"{conf:.3f}",
                             int(x1), int(y1), int(x2), int(y2)])
            writer.write(draw(frame, dets, f"RF-DETR  conf>={args.conf}"))
            bar.update(1)
        idx += 1
    bar.close()
    cap.release()
    writer.release()

    with open(out_dir / f"{path.stem}_{TAG}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "class", "conf", "x1", "y1", "x2", "y2"])
        wr.writerows(rows)
    ms = 1000 * sum(t_infer) / max(len(t_infer), 1)
    print(f"  -> {out_video.name}: {len(t_infer)} frames, {ms:.0f} ms/frame, "
          f"{dict(counts.most_common())}")
    return len(t_infer), counts


def write_summary(out_dir, videos):
    """Detections per class per video, rebuilt from all per-video CSVs."""
    per_video = {}
    for v in videos:
        f = out_dir / f"{v.stem}_{TAG}.csv"
        if f.exists():
            with open(f, newline="") as fh:
                rd = list(csv.DictReader(fh))
            per_video[v.name] = (len({r["frame"] for r in rd}), Counter(r["class"] for r in rd))
    names = sorted({c for _, cnt in per_video.values() for c in cnt})
    with open(out_dir / f"summary_{TAG}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "frames_with_detections"] + names)
        for v, (nf, cnt) in per_video.items():
            wr.writerow([v, nf] + [cnt[c] for c in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="RF-DETR checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium", choices=list(SIZES), help="same as training")
    ap.add_argument("--resolution", type=int, default=960, help="same as training")
    ap.add_argument("--videos", default="TEST", help="folder with test videos")
    ap.add_argument("--out", default="TEST_results")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=2, help="frames per second of video to process")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default: all cores minus 2, keeps the laptop usable)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n = args.threads or max(1, (os.cpu_count() or 4) - 2)
    torch.set_num_threads(n)
    videos = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} videos, CPU with {n} threads, {args.fps} fps, {args.size} @ {args.resolution}, "
          f"conf {args.conf}")

    model = SIZES[args.size](pretrain_weights=args.weights, resolution=args.resolution,
                             device="cpu")
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}")
        if (out_dir / f"{v.stem}_{TAG}.mp4").exists() and not args.overwrite:
            print("  already done, skipping (use --overwrite to redo)")
            continue
        process_video(model, v, out_dir, args)

    write_summary(out_dir, videos)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
