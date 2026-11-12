"""
Run the plant-2 model on video(s): show it live in a resizable window and save the output
video with boxes. Shows only: helmet, no_helmet, boots, no_boots.
Green = compliant (helmet, boots), red/orange = violation (no_helmet, no_boots).

Usage (laptop, CPU)
  pip install -U ultralytics
  python test_plant2_video.py --weights best.pt --video test.mp4
  python test_plant2_video.py --weights best.pt --video TEST_videos/     (every video in a folder)

Keys in the window:  q / Esc = stop   space = pause / resume
The window can be resized or maximised with the mouse; it starts sized to fit your screen.

CPU speed: the model runs on --fps frames per second of video (default 4) and the last
boxes are kept on the frames in between, so the saved video has the original length and
frame rate. Raise --fps for smoother boxes (slower), lower it to finish faster.
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".flv", ".wmv"}

# model class -> (display name, BGR colour)
SHOW = {
    "helmet": ("helmet", (0, 200, 0)),
    "head": ("no_helmet", (0, 0, 255)),
    "boots": ("boots", (200, 200, 0)),
    "non_safety_footwear": ("no_boots", (0, 140, 255)),
}
WINDOW = "PPE test  (q = quit, space = pause)"


def screen_size():
    try:                                   # Windows
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:                                   # Linux / macOS with Tk
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        size = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return size
    except Exception:
        return 1366, 768


def fit(w, h, max_w, max_h):
    s = min(max_w / w, max_h / h, 1.0)
    return int(w * s), int(h * s)


def draw(frame, dets, info):
    th = max(1, round(frame.shape[0] / 540))
    fs = 0.5 * th
    for label, conf, (x1, y1, x2, y2), color in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, th + 1)
        text = f"{label} {conf:.2f}"
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y = max(p1[1], tht + 6)
        cv2.rectangle(frame, (p1[0], y - tht - 6), (p1[0] + tw + 4, y), color, -1)
        cv2.putText(frame, text, (p1[0] + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    # info bar
    s = 0.65 * th                                       # shrink text to fit the width
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    s *= min(1.0, (frame.shape[1] - 40) / tw)
    bar = int(40 * s) + 8
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, bar - 10), cv2.FONT_HERSHEY_SIMPLEX, s,
                (255, 255, 255), th, cv2.LINE_AA)
    return frame


def detect(model, frame, keep_ids, args):
    r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device="cpu",
                      classes=keep_ids, verbose=False)[0]
    out = []
    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        label, color = SHOW[model.names[int(c)]]
        out.append((label, s, b, color))
    return out


def run_video(model, keep_ids, path, args, show):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open {path}")
        return show, False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / args.fps))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_plant2.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if show:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            sw, sh = screen_size()
            cv2.resizeWindow(WINDOW, *fit(w, h, int(sw * 0.85), int(sh * 0.80)))
            cv2.moveWindow(WINDOW, 20, 20)
        except cv2.error:
            print("Can't open a window here (headless OpenCV?) - saving the video only.\n"
                  "For the live view: pip uninstall -y opencv-python-headless && "
                  "pip install opencv-python")
            show = False

    print(f"\n{path.name}: {w}x{h} @ {fps:.0f} fps, {n} frames, model every {step} frames")
    dets, totals, infer_ms, idx, paused, stopped = [], Counter(), 0.0, 0, False, False
    t_start = time.perf_counter()
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = time.perf_counter()
                dets = detect(model, frame, keep_ids, args)
                infer_ms = 1000 * (time.perf_counter() - t)
                totals.update(d[0] for d in dets)
            counts = Counter(d[0] for d in dets)
            pct = f"{100 * idx / n:3.0f}%" if n else ""
            info = (f"{pct}  helmet {counts['helmet']}  no_helmet {counts['no_helmet']}  "
                    f"boots {counts['boots']}  no_boots {counts['no_boots']}   "
                    f"{infer_ms:.0f} ms/inference")
            frame = draw(frame, dets, info)
            writer.write(frame)
            idx += 1
            if not show and idx % 100 == 0:
                print(f"  {pct} done", end="\r")
        if show:
            if not paused:
                cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                stopped = True
                break
            if key == ord(" "):
                paused = not paused
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # window closed
                stopped = True
                break

    cap.release()
    writer.release()
    secs = time.perf_counter() - t_start
    print(f"  saved {out_path}  ({idx} frames in {secs:.0f} s)")
    print(f"  detections (on processed frames): {dict(totals.most_common())}")
    return show, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="plant-2 best.pt")
    ap.add_argument("--video", required=True, help="a video file or a folder of videos")
    ap.add_argument("--out", default="plant2_test_output")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960, help="same as training")
    ap.add_argument("--fps", type=float, default=4, help="model runs per second of video")
    ap.add_argument("--no-show", action="store_true", help="don't open a window, only save")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all - 2)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))
    src = Path(args.video)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS) \
        if src.is_dir() else [src]
    if not videos or not videos[0].exists():
        raise SystemExit(f"No video found at {src}")

    model = YOLO(args.weights)
    missing = [k for k in SHOW if k not in model.names.values()]
    if missing:
        raise SystemExit(f"Model has no class {missing}. Its classes: {list(model.names.values())}")
    keep_ids = [i for i, n in model.names.items() if n in SHOW]

    show = not args.no_show
    if show and os.name != "nt" and not (os.environ.get("DISPLAY") or
                                         os.environ.get("WAYLAND_DISPLAY")):
        print("No screen found (e.g. SSH session) - saving the video only, no live view.")
        show = False
    for v in videos:
        show, stopped = run_video(model, keep_ids, v, args, show)
        if stopped:
            print("Stopped by user.")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
