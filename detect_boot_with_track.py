"""
Run ONLY the boot detector (no pose/person model) on a video — full-frame
detection of Boot / No_Boot, now with ByteTrack for per-detection tracking
across frames. Displays the video in a resizable window and saves the
annotated output.

Why add tracking here: a single frame's classification can flicker (a
person correctly shows Boot in frame 10 but No_Boot in frame 11 due to
motion blur or a bad angle). ByteTrack assigns a persistent ID to each
tracked boot/no-boot region across frames, so you can optionally smooth
those per-frame flickers into a more stable per-track decision (see
TEMPORAL_VOTING below) — the same principle your coworker's pose-based
script used, just applied directly to the boot detector's own output
instead of to a person track.

Just edit the CONFIG section below and run:
    pip install ultralytics opencv-python --break-system-packages
    python detect_boots_only.py

Press 'q' while the display window is focused to stop early (output
saved up to that point either way).
"""

import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
VIDEO_SOURCE = "/path/to/input_video.mp4"     # or an RTSP URL
OUTPUT_VIDEO_PATH = "/path/to/output_video.mp4"

BOOT_MODEL_PATH = "/path/to/your/best.pt"     # your trained boot_detector weights

BOOT_CLASS_ID = 0                             # 0 = Boot
NO_BOOT_CLASS_ID = 1                          # 1 = No_Boot

BOOT_CONF = 0.4

TRACKER_CONFIG = "bytetrack.yaml"             # bundled with ultralytics, no extra install needed

# --- Temporal voting (optional, smooths per-frame flicker per track) ---
# If a tracked detection flips between Boot/No_Boot frame to frame, this
# uses a rolling vote over its recent history to decide what to actually
# display, instead of trusting each single frame in isolation.
ENABLE_TEMPORAL_VOTING = True
HISTORY_LENGTH = 15         # how many recent frames to remember per track
BOOT_VOTE_THRESHOLD = 0.5   # boot_votes / total_votes >= this -> display as Boot

DISPLAY_WINDOW_WIDTH = 960                    # resized display window size (doesn't affect saved video resolution)
DISPLAY_WINDOW_HEIGHT = 540

DISPLAY_LIVE = True
# ---------------------------------------------------------

boot_model = YOLO(BOOT_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

if DISPLAY_LIVE:
    window_name = "Boot Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
    cv2.resizeWindow(window_name, DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT)

frame_count = 0

# Per-track rolling history of predictions: {track_id: deque([1, 0, 1, ...])}
# 1 = Boot, 0 = No_Boot, for that track's recent frames.
track_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    boot_results = boot_model.track(
        source=frame,
        conf=BOOT_CONF,
        tracker=TRACKER_CONFIG,
        persist=True,       # keep track IDs alive across frames within this run
        verbose=False,
    )[0]

    if boot_results.boxes is not None:
        for box in boot_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # box.id is None if ByteTrack couldn't assign/maintain an ID
            # for this detection this frame (e.g. it just appeared).
            track_id = int(box.id[0]) if box.id is not None else None

            if cls_id not in (BOOT_CLASS_ID, NO_BOOT_CLASS_ID):
                continue  # unknown class, skip

            display_cls_id = cls_id
            display_conf = conf

            if ENABLE_TEMPORAL_VOTING and track_id is not None:
                track_history[track_id].append(1 if cls_id == BOOT_CLASS_ID else 0)
                history = track_history[track_id]
                boot_ratio = sum(history) / len(history)
                display_cls_id = BOOT_CLASS_ID if boot_ratio >= BOOT_VOTE_THRESHOLD else NO_BOOT_CLASS_ID
                display_conf = boot_ratio if display_cls_id == BOOT_CLASS_ID else (1 - boot_ratio)

            if display_cls_id == BOOT_CLASS_ID:
                color = (0, 200, 0)      # green
                label = f"Boot {display_conf:.2f}"
            else:
                color = (0, 0, 255)      # red
                label = f"No Boot {display_conf:.2f}"

            if track_id is not None:
                label += f" ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    writer.write(frame)

    if DISPLAY_LIVE:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
writer.release()
if DISPLAY_LIVE:
    cv2.destroyAllWindows()

print(f"\nDone. {frame_count} frames processed.")
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
