"""
Run the trained truck detector on a video (CPU) and save an annotated 1280x720 copy.

Install once:
    pip install ultralytics opencv-python

Usage:
    python test_truck_video.py --video path/to/cam03.mp4 --model best.pt
    python test_truck_video.py --video cam03.mp4 --model best.pt --conf 0.4 --skip 2

Options:
    --out    output path (default: <video name>_pred.mp4 next to the input)
    --conf   confidence threshold (default 0.4)
    --imgsz  inference size; keep it the same as training (default 640)
    --skip   run the model every N frames and reuse boxes in between (speeds up CPU)
    --no-show  don't open the live preview window (just save the video)

Live preview keys:  q / Esc = stop (video saved up to that point),  space = pause/resume
Note: the preview needs "opencv-python", not "opencv-python-headless".
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

OUT_W, OUT_H = 1280, 720
BOX_COLOR = (0, 165, 255)      # orange (BGR)


def parse_args():
    p = argparse.ArgumentParser(description="Truck detection on a video (CPU).")
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--model", default="best.pt", help="trained weights (best.pt)")
    p.add_argument("--out", default=None, help="output video path (.mp4)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--skip", type=int, default=1, help="run model every N frames")
    p.add_argument("--no-show", action="store_true", help="don't show the live preview window")
    return p.parse_args()


def draw(frame, boxes, names):
    for x1, y1, x2, y2, conf, cls in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, BOX_COLOR, 3)
        label = f"{names[int(cls)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(p1[1], th + 10)
        cv2.rectangle(frame, (p1[0], ty - th - 10), (p1[0] + tw + 8, ty), BOX_COLOR, -1)
        cv2.putText(frame, label, (p1[0] + 4, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Status banner: is a truck present in this frame?
    status = f"TRUCK PRESENT ({len(boxes)})" if boxes else "NO TRUCK"
    color = (0, 200, 0) if boxes else (0, 0, 220)
    cv2.rectangle(frame, (0, OUT_H - 45), (360, OUT_H), (0, 0, 0), -1)
    cv2.putText(frame, status, (12, OUT_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    out_path = Path(args.out) if args.out else video.with_name(f"{video.stem}_pred.mp4")

    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")

    print(f"Input: {video}  ({int(cap.get(3))}x{int(cap.get(4))} @ {fps:.1f} fps, "
          f"{total or '?'} frames)\nOutput: {out_path}  ({OUT_W}x{OUT_H})")

    show = not args.no_show
    win = "Truck detection (q = quit, space = pause)"
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, OUT_W, OUT_H)
        except cv2.error:
            print("No display available (opencv-python-headless?). Saving without preview.")
            show = False

    boxes, i, start = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        if i % args.skip == 0:
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                              device="cpu", verbose=False)[0]
            boxes = [(*b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes]

        draw(frame, boxes, names)
        writer.write(frame)
        i += 1

        if show:
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):                       # pause until space again
                while (key := cv2.waitKey(0) & 0xFF) not in (ord(" "), ord("q"), 27):
                    pass
            if key in (ord("q"), 27):                 # q or Esc
                print("Stopped by user.")
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed
                print("Preview window closed.")
                break

        if i % 50 == 0:
            speed = i / (time.time() - start)
            left = f", ~{(total - i) / speed / 60:.1f} min left" if total else ""
            print(f"  frame {i}/{total or '?'}  ({speed:.1f} fps{left})")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done: {i} frames in {(time.time() - start) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
