"""
Test the harness pipeline on video(s): YOLO detector (person + truck) -> crop each person on a
truck (+10% padding) -> harness classifier. Shows it live in a resizable window and saves the
annotated video plus a CSV of every classification.

Each person is tracked across frames; the label shown is the average of their last few
classifier results, and "NO HARNESS - ALERT" appears only when no_harness has held for
--alert-seconds (like the production rules engine would do). Low-confidence results are shown
as "unsure" and never alert.

Usage (laptop, CPU)
  pip install -U ultralytics timm lap        (lap is needed for tracking)
  python test_harness_video.py --detector best.pt \
      --classifier harness_model_v2/harness_classifier.pt --video test.mp4

  --detector     plant-1 PPE model (needs 'person' and 'truck' classes)
  --classifier   harness_classifier.pt; its _meta.json must sit next to it
  --all-persons  classify everyone, not only people on trucks

Keys:  q / Esc = stop   space = pause.  The window can be resized with the mouse.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".flv", ".wmv"}
WINDOW = "Harness test  (q = quit, space = pause)"
GREEN, RED, ORANGE, GREY, BLUE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (160, 160, 160), (255, 140, 0)


# ---------------------------------------------------------------- classifier

class HarnessClassifier:
    def __init__(self, weights, meta_path):
        import timm
        meta = json.loads(Path(meta_path).read_text())
        self.classes = meta["classes"]
        self.size = int(meta["img_size"])
        self.pad = tuple(meta.get("pad_color", (114, 114, 114)))
        self.mean = torch.tensor(meta["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(meta["std"]).view(1, 3, 1, 1)
        self.model = timm.create_model(meta["model_name"], pretrained=False,
                                       num_classes=len(self.classes))
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.model.load_state_dict(sd)
        self.model.eval()
        for c in ("harness", "no_harness"):
            if c not in self.classes:
                raise SystemExit(f"Classifier classes {self.classes} have no '{c}'")

    def letterbox(self, bgr):
        h, w = bgr.shape[:2]
        s = min(self.size / w, self.size / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = np.full((self.size, self.size, 3), self.pad, np.uint8)
        x0, y0 = (self.size - nw) // 2, (self.size - nh) // 2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas

    @torch.no_grad()
    def __call__(self, crops):
        """crops: list of BGR images -> array of class probabilities (n, n_classes)."""
        if not crops:
            return np.zeros((0, len(self.classes)))
        x = torch.from_numpy(np.stack([self.letterbox(c) for c in crops])).permute(0, 3, 1, 2)
        x = (x.float() / 255.0 - self.mean) / self.std
        return self.model(x).softmax(1).numpy()


# ---------------------------------------------------------------- helpers

def screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1366, 768


def feet_on_truck(box, trucks):
    x1, y1, x2, y2 = box
    fx, fy = (x1 + x2) / 2, y2
    return any(tx1 <= fx <= tx2 and ty1 <= fy <= ty2 for tx1, ty1, tx2, ty2 in trucks)


def label_box(frame, box, text, color, th):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th + 1)
    if not text:
        return
    fs = 0.5 * th
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    y = max(y1, tht + 6)
    cv2.rectangle(frame, (x1, y - tht - 6), (x1 + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x1 + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                cv2.LINE_AA)


def info_bar(frame, text, th):
    s = 0.65 * th
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    s *= min(1.0, (frame.shape[1] - 40) / tw)
    bar = int(40 * s) + 8
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, bar - 10), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), th,
                cv2.LINE_AA)


# ---------------------------------------------------------------- main loop

def run_video(det, ids, clf, path, args, show, log):
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
    out_path = out_dir / f"{path.stem}_harness.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    th = max(1, round(h / 540))
    hi, ni = clf.classes.index("harness"), clf.classes.index("no_harness")

    if show:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            sw, sh = screen_size()
            s = min(sw * 0.85 / w, sh * 0.8 / h, 1.0)
            cv2.resizeWindow(WINDOW, int(w * s), int(h * s))
            cv2.moveWindow(WINDOW, 20, 20)
        except cv2.error:
            print("Can't open a window - saving the video only.")
            show = False

    history = defaultdict(lambda: deque(maxlen=args.smooth))   # track id -> recent probs
    no_since = {}                                              # track id -> time no_harness began
    drawn, idx, paused, stopped, alerts = [], 0, False, False, set()
    infer_ms = 0.0
    print(f"\n{path.name}: {w}x{h} @ {fps:.0f} fps, {n} frames, model every {step} frames")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            if idx % step == 0:
                t0 = time.perf_counter()
                kw = dict(conf=args.det_conf, imgsz=args.imgsz, device="cpu",
                          classes=list(ids.values()), verbose=False)
                r = (det.track(frame, persist=True, tracker="bytetrack.yaml", **kw)
                     if args.track else det.predict(frame, **kw))[0]
                boxes = r.boxes
                tids = boxes.id.int().tolist() if (args.track and boxes.id is not None) \
                    else [None] * len(boxes)
                trucks = [b for b, c in zip(boxes.xyxy.tolist(), boxes.cls.tolist())
                          if int(c) == ids.get("truck")]
                people = [(b, t) for b, c, t in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), tids)
                          if int(c) == ids["person"]]
                todo, crops, drawn = [], [], []
                for b, t in people:
                    x1, y1, x2, y2 = b
                    on = args.all_persons or "truck" not in ids or feet_on_truck(b, trucks)
                    if not on or (y2 - y1) < args.min_height:
                        drawn.append((b, "", GREY))
                        continue
                    px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
                    todo.append((b, t))
                    crops.append(frame[cy1:cy2, cx1:cx2])
                probs = clf(crops)
                for (b, t), p in zip(todo, probs):
                    key = t if t is not None else id(b)
                    history[key].append(p)
                    avg = np.mean(history[key], axis=0)
                    cls = clf.classes[int(avg.argmax())]
                    conf = float(avg.max())
                    if cls == "no_harness" and conf >= args.min_conf:
                        no_since.setdefault(key, t_sec)
                    else:
                        no_since.pop(key, None)
                    alert = key in no_since and t_sec - no_since[key] >= args.alert_seconds
                    tag = f"ID{t} " if t is not None else ""
                    if alert:
                        text, color = f"{tag}NO HARNESS - ALERT {conf:.2f}", RED
                        alerts.add(key)
                    elif conf < args.min_conf or cls not in ("harness", "no_harness"):
                        text, color = f"{tag}unsure {conf:.2f}", BLUE
                    elif cls == "harness":
                        text, color = f"{tag}harness {conf:.2f}", GREEN
                    else:
                        text, color = f"{tag}no harness {conf:.2f}", ORANGE
                    drawn.append((b, text, color))
                    log.writerow([path.name, idx, f"{t_sec:.2f}", t, *map(int, b),
                                  f"{p[hi]:.3f}", f"{p[ni]:.3f}", cls, f"{conf:.3f}", int(alert)])
                infer_ms = 1000 * (time.perf_counter() - t0)
            for b, text, color in drawn:
                label_box(frame, b, text, color, th)
            pct = f"{100 * idx / n:3.0f}%  " if n else ""
            info_bar(frame, f"{pct}on truck: {sum(1 for d in drawn if d[1])}   "
                            f"alerts so far: {len(alerts)}   {infer_ms:.0f} ms/frame", th)
            writer.write(frame)
            idx += 1
        if show:
            if not paused:
                cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                stopped = True
                break
            if key == ord(" "):
                paused = not paused
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                stopped = True
                break
    cap.release()
    writer.release()
    print(f"  saved {out_path}; people who triggered an alert: {len(alerts)}")
    return show, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="plant-1 PPE YOLO best.pt")
    ap.add_argument("--classifier", required=True, help="harness_classifier.pt")
    ap.add_argument("--meta", default=None, help="default: <classifier>_meta.json")
    ap.add_argument("--video", required=True, help="video file or folder")
    ap.add_argument("--out", default="harness_test_output")
    ap.add_argument("--fps", type=float, default=4, help="pipeline runs per second of video")
    ap.add_argument("--imgsz", type=int, default=960, help="detector image size (as trained)")
    ap.add_argument("--det-conf", type=float, default=0.4)
    ap.add_argument("--pad", type=float, default=0.10, help="crop padding (as in training)")
    ap.add_argument("--min-height", type=int, default=60, help="skip people smaller than this")
    ap.add_argument("--min-conf", type=float, default=0.7, help="below this -> 'unsure'")
    ap.add_argument("--smooth", type=int, default=5, help="average over last N results per person")
    ap.add_argument("--alert-seconds", type=float, default=3, help="no_harness held this long -> alert")
    ap.add_argument("--all-persons", action="store_true", help="classify people not on trucks too")
    ap.add_argument("--no-track", dest="track", action="store_false",
                    help="don't track people (no smoothing across frames)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 2))
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)

    meta = args.meta or str(Path(args.classifier).with_name(
        Path(args.classifier).stem + "_meta.json"))
    if not Path(meta).exists():
        raise SystemExit(f"Meta file not found: {meta} (use --meta)")
    clf = HarnessClassifier(args.classifier, meta)
    det = YOLO(args.detector)
    ids = {n: i for i, n in det.names.items() if n in ("person", "truck")}
    if "person" not in ids:
        raise SystemExit(f"Detector has no 'person' class: {list(det.names.values())}")
    if "truck" not in ids and not args.all_persons:
        print("Detector has no 'truck' class - classifying everyone.")
    if args.track:
        try:
            import lap  # noqa: F401  (used by Ultralytics' tracker)
        except ImportError:
            print("Package 'lap' not installed - running without tracking (no smoothing or "
                  "alerts over time). Fix: pip install lap")
            args.track = False
    print(f"Classifier classes {clf.classes} @ {clf.size}px; unsure below {args.min_conf}; "
          f"alert after {args.alert_seconds}s of no_harness")

    src = Path(args.video)
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS) \
        if src.is_dir() else [src]
    if not videos or not videos[0].exists():
        raise SystemExit(f"No video found at {src}")
    show = not args.no_show
    if show and os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No screen found - saving the video only.")
        show = False

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "harness_predictions.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["video", "frame", "time_s", "track_id", "x1", "y1", "x2", "y2",
                      "p_harness", "p_no_harness", "smoothed_class", "smoothed_conf", "alert"])
        for v in videos:
            show, stopped = run_video(det, ids, clf, v, args, show, log)
            det.predictor = None                     # fresh tracker for the next video
            if stopped:
                print("Stopped by user.")
                break
    cv2.destroyAllWindows()
    print(f"Per-person results: {Path(args.out) / 'harness_predictions.csv'}")


if __name__ == "__main__":
    main()
