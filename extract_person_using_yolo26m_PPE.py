"""
Detect people in videos with your PPE YOLO model and save each person as a cropped image,
ready to sort into harness / no_harness / unsure for the harness classifier.

CPU-friendly: videos are processed one at a time and read frame by frame (nothing is loaded
all at once), the model runs only on sampled frames (default: one every 2 s), and 2 CPU
cores are left free. Re-running skips videos that are already done.

Usage (laptop, CPU)
  pip install -U ultralytics
  python extract_person_crops.py --weights best.pt --videos harness_videos/ --out person_crops

Output
  person_crops/on_truck/   people whose feet are inside a truck box  <- the ones that matter
  person_crops/ground/     everyone else
  person_crops/crops.csv   one row per crop: video, frame, time, box, confidence, on_truck
  Crop names: <video>_f<frame>_p<n>.jpg  (the video name is kept so the train/val split can
  be done by video later)

Crops use the same padding as the classifier will at deployment (10% per side by default).
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".flv", ".wmv"}


def feet_on_truck(person, trucks):
    """True if the bottom-centre of the person box lies inside any truck box."""
    x1, y1, x2, y2 = person
    fx, fy = (x1 + x2) / 2, y2
    return any(tx1 <= fx <= tx2 and ty1 <= fy <= ty2 for tx1, ty1, tx2, ty2 in trucks)


def process_video(model, ids, path, out, args, writer):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  could not open {path.name}, skipping")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(fps * args.every))
    stem = path.stem.replace(" ", "_")

    saved, idx = 0, 0
    bar = tqdm(total=n_frames // step if n_frames else None, desc=path.name, unit="frame",
               dynamic_ncols=True)
    while True:
        if not cap.grab():                      # move to next frame without decoding it
            break
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()              # decode only the sampled frames
        if not ok:
            break
        h, w = frame.shape[:2]
        r = model.predict(frame, conf=min(args.conf, args.truck_conf), imgsz=args.imgsz,
                          device="cpu", classes=list(ids.values()), verbose=False)[0]
        persons, trucks = [], []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(c) == ids["person"] and s >= args.conf:
                persons.append((b, s))
            elif int(c) == ids.get("truck") and s >= args.truck_conf:
                trucks.append(b)

        for k, ((x1, y1, x2, y2), s) in enumerate(persons):
            if y2 - y1 < args.min_height:
                continue
            on_truck = feet_on_truck((x1, y1, x2, y2), trucks) if "truck" in ids else None
            if args.on_truck_only and not on_truck:
                continue
            px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
            sub = "on_truck" if on_truck else ("ground" if on_truck is False else "all")
            name = f"{stem}_f{idx:06d}_p{k}.jpg"
            cv2.imwrite(str(out / sub / name), frame[cy1:cy2, cx1:cx2],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            writer.writerow([f"{sub}/{name}", path.name, idx, f"{idx / fps:.2f}",
                             int(x1), int(y1), int(x2), int(y2), f"{s:.3f}", on_truck])
            saved += 1
        bar.update(1)
        bar.set_postfix(crops=saved)
        idx += 1
    bar.close()
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="your PPE YOLO best.pt")
    ap.add_argument("--videos", required=True, help="a video file or a folder of videos")
    ap.add_argument("--out", default="person_crops")
    ap.add_argument("--every", type=float, default=2.0,
                    help="seconds between sampled frames (avoids near-duplicate crops)")
    ap.add_argument("--conf", type=float, default=0.5, help="person confidence")
    ap.add_argument("--truck-conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960, help="same as training")
    ap.add_argument("--pad", type=float, default=0.10, help="crop padding per side")
    ap.add_argument("--min-height", type=int, default=60, help="skip people shorter than this (px)")
    ap.add_argument("--on-truck-only", action="store_true", help="save only people on trucks")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all - 2)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))
    from ultralytics import YOLO

    src = Path(args.videos)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS) \
        if src.is_dir() else [src]
    if not videos or not videos[0].exists():
        raise SystemExit(f"No videos found at {src}")

    model = YOLO(args.weights)
    ids = {n: i for i, n in model.names.items() if n in ("person", "truck")}
    if "person" not in ids:
        raise SystemExit(f"Model has no 'person' class: {list(model.names.values())}")
    if "truck" not in ids:
        print("Model has no 'truck' class - can't tell who is on a truck; all crops go to all/.")

    out = Path(args.out)
    for sub in ("on_truck", "ground") if "truck" in ids else ("all",):
        (out / sub).mkdir(parents=True, exist_ok=True)
    done_file = out / "done_videos.txt"
    done = set(done_file.read_text().split("\n")) if done_file.exists() else set()
    csv_path = out / "crops.csv"
    new_csv = not csv_path.exists()

    print(f"{len(videos)} videos, one frame every {args.every} s, imgsz {args.imgsz}, CPU")
    total, t0 = 0, time.time()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_csv:
            writer.writerow(["crop", "video", "frame", "time_s", "x1", "y1", "x2", "y2",
                             "conf", "on_truck"])
        for i, v in enumerate(videos, 1):
            if str(v) in done:
                print(f"[{i}/{len(videos)}] {v.name}: already done, skipping")
                continue
            print(f"[{i}/{len(videos)}] {v.name}")
            n = process_video(model, ids, v, out, args, writer)
            f.flush()
            with open(done_file, "a") as d:
                d.write(f"{v}\n")
            total += n
            print(f"  {n} crops")

    counts = {s.name: len(list(s.glob("*.jpg"))) for s in out.iterdir() if s.is_dir()}
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {total} new crops. In {out}: {counts}")
    print("Next: sort on_truck/ crops into harness/, no_harness/ and unsure/ folders.")


if __name__ == "__main__":
    main()
