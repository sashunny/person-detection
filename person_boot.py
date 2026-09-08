"""
Combined pipeline: person detection -> crop lower body -> boot detection.
Draws a bounding box + label ONLY when a person IS wearing a safety boot.
Displays the video live while processing and saves the annotated output.

Just edit the CONFIG section below and run:
    pip install ultralytics opencv-python --break-system-packages
    python detect_safety_boots.py

Press 'q' while the display window is focused to stop early (output saved
up to that point either way).
"""

import cv2
from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
VIDEO_SOURCE = "/path/to/input_video.mp4"     # or an RTSP URL, e.g. "rtsp://user:pass@ip:554/stream"
OUTPUT_VIDEO_PATH = "/path/to/output_video.mp4"

PERSON_MODEL_PATH = "yolov8n.pt"              # pretrained, COCO 'person' class
BOOT_MODEL_PATH = "/path/to/your/best.pt"     # your trained boot_detector weights

PERSON_CLASS_ID = 0                           # COCO 'person'
BOOT_CLASS_ID = 0                             # 0 = Boot in your boot model (per earlier training script)

PERSON_CONF = 0.4
BOOT_CONF = 0.4

LOWER_BODY_FRACTION = 0.4   # crop the bottom 40% of each person box (legs/feet) before boot detection

# Person detection boxes often don't fully cover the feet (bbox ends right at
# the ankle, or slightly above). This expands the crop beyond the detected
# box so boots aren't cut off. Values are in pixels.
BOTTOM_OFFSET = 25          # extend crop below the person box's bottom edge
SIDE_OFFSET = 15            # extend crop left/right beyond the person box's width

DISPLAY_LIVE = True         # set False to just process + save without popping up a window
# ---------------------------------------------------------

person_model = YOLO(PERSON_MODEL_PATH)
boot_model = YOLO(BOOT_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- Stage 1: detect persons in the full frame ---
    person_results = person_model.predict(
        source=frame,
        classes=[PERSON_CLASS_ID],
        conf=PERSON_CONF,
        verbose=False,
    )[0]

    for person_box in person_results.boxes:
        px1, py1, px2, py2 = map(int, person_box.xyxy[0].tolist())

        # --- crop lower body region (legs/feet), expanded by offsets ---
        person_height = py2 - py1
        crop_y1 = py1 + int(person_height * (1 - LOWER_BODY_FRACTION))
        crop_y1 = max(crop_y1, py1)

        # expand crop bounds beyond the person box, clamped to frame edges
        crop_x1 = max(px1 - SIDE_OFFSET, 0)
        crop_x2 = min(px2 + SIDE_OFFSET, frame_w)
        crop_y2 = min(py2 + BOTTOM_OFFSET, frame_h)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            continue

        # --- Stage 2: run boot detector on the crop ---
        boot_results = boot_model.predict(
            source=crop,
            conf=BOOT_CONF,
            verbose=False,
        )[0]

        # Check if a Boot (compliant) was detected in this crop
        wearing_boot = False
        boot_conf_value = 0.0
        boot_box_in_crop = None

        for boot_box in boot_results.boxes:
            cls_id = int(boot_box.cls[0])
            conf = float(boot_box.conf[0])
            if cls_id == BOOT_CLASS_ID:
                wearing_boot = True
                if conf > boot_conf_value:
                    boot_conf_value = conf
                    boot_box_in_crop = boot_box.xyxy[0].tolist()

        # --- Draw box ONLY if wearing a safety boot ---
        if wearing_boot and boot_box_in_crop is not None:
            bx1, by1, bx2, by2 = boot_box_in_crop
            # map crop-local coords back to full-frame coords
            # (crop origin is crop_x1, crop_y1 — not px1, py1 — since the
            # crop is expanded beyond the person box by SIDE_OFFSET/BOTTOM_OFFSET)
            abs_x1 = crop_x1 + int(bx1)
            abs_y1 = crop_y1 + int(by1)
            abs_x2 = crop_x1 + int(bx2)
            abs_y2 = crop_y1 + int(by2)

            label = f"Safety Boot {boot_conf_value:.2f}"
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 200, 0), 2)
            cv2.putText(
                frame, label, (abs_x1, max(abs_y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )

    writer.write(frame)

    if DISPLAY_LIVE:
        cv2.imshow("Safety Boot Detection", frame)
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