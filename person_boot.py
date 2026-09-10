"""
Pose-driven boot detection: uses a YOLO pose model to locate ankle
keypoints per person, builds a small ROI around each person's ankles,
and runs the boot detector only on that ROI. Displays the video live
while processing and saves the annotated output.

Just edit the CONFIG section below and run:
    pip install ultralytics opencv-python --break-system-packages
    python detect_boots_pose_roi.py

Press 'q' while the display window is focused to stop early (output
saved up to that point either way).

Keypoint indices follow the standard COCO 17-keypoint layout used by
Ultralytics pose models:
    ... 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
"""

import cv2
from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
VIDEO_SOURCE = "/path/to/input_video.mp4"     # or an RTSP URL, e.g. "rtsp://user:pass@ip:554/stream"
OUTPUT_VIDEO_PATH = "/path/to/output_video.mp4"

POSE_MODEL_PATH = "yolo26n-pose.pt"           # adjust filename if your local checkpoint is named differently
BOOT_MODEL_PATH = "/path/to/your/best.pt"     # your trained boot_detector weights

BOOT_CLASS_ID = 0                             # 0 = Boot
NO_BOOT_CLASS_ID = 1                          # 1 = No_Boot

POSE_CONF = 0.4
BOOT_CONF = 0.4
POSE_IMGSZ = 480                              # lower than the default 640 — pose/keypoint localization
                                               # tolerates lower resolution better than small-object
                                               # detection does, and this is a real CPU speedup with no GPU
KEYPOINT_CONF_THRESHOLD = 0.3                 # ignore ankle keypoints below this confidence

LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16

# ROI built around the ankle keypoint(s), in pixels. Feet extend below and
# forward of the ankle joint, and boots extend a bit above the ankle too,
# so the ROI needs generous margin in both directions.
ROI_SIDE_MARGIN = 40      # left/right margin around ankle x-position(s)
ROI_ABOVE_MARGIN = 30     # margin above the ankle (lower shin / boot shaft)
ROI_BELOW_MARGIN = 60     # margin below the ankle (foot / sole)

DRAW_KEYPOINTS = True     # draw small dots on the detected ankle points, useful for debugging ROI placement
DISPLAY_LIVE = True
# ---------------------------------------------------------

pose_model = YOLO(POSE_MODEL_PATH)
boot_model = YOLO(BOOT_MODEL_PATH)

pose_model.fuse()  # fuses Conv+BatchNorm layers — small free speedup, no accuracy cost
boot_model.fuse()

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

    pose_results = pose_model.predict(
        source=frame,
        conf=POSE_CONF,
        imgsz=POSE_IMGSZ,
        verbose=False,
    )[0]

    if pose_results.keypoints is None:
        writer.write(frame)
        if DISPLAY_LIVE:
            cv2.imshow("Pose-based Boot Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        continue

    kpts_xy = pose_results.keypoints.xy.cpu().numpy()      # (num_people, 17, 2)
    kpts_conf = pose_results.keypoints.conf.cpu().numpy()  # (num_people, 17)

    # --- Pass 1: for every person, build the ankle-based ROI and collect it.
    # We don't run the boot model yet — we batch all ROIs into one call below,
    # which is much faster than calling predict() once per person.
    roi_crops = []
    roi_coords = []  # (roi_x1, roi_y1) origin for each crop, aligned by index with roi_crops

    for person_idx in range(len(kpts_xy)):
        person_kpts = kpts_xy[person_idx]
        person_kconf = kpts_conf[person_idx]

        ankle_points = []
        if person_kconf[LEFT_ANKLE_IDX] >= KEYPOINT_CONF_THRESHOLD:
            ankle_points.append(person_kpts[LEFT_ANKLE_IDX])
        if person_kconf[RIGHT_ANKLE_IDX] >= KEYPOINT_CONF_THRESHOLD:
            ankle_points.append(person_kpts[RIGHT_ANKLE_IDX])

        if not ankle_points:
            continue  # no reliable ankle keypoints for this person — skip

        if DRAW_KEYPOINTS:
            for pt in ankle_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)

        xs = [pt[0] for pt in ankle_points]
        ys = [pt[1] for pt in ankle_points]

        roi_x1 = max(int(min(xs) - ROI_SIDE_MARGIN), 0)
        roi_x2 = min(int(max(xs) + ROI_SIDE_MARGIN), frame_w)
        roi_y1 = max(int(min(ys) - ROI_ABOVE_MARGIN), 0)
        roi_y2 = min(int(max(ys) + ROI_BELOW_MARGIN), frame_h)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            continue

        # visualize the ROI box itself (light gray), useful for debugging
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (180, 180, 180), 1)

        roi_crops.append(roi)
        roi_coords.append((roi_x1, roi_y1))

    # --- Pass 2: run the boot model ONCE on all collected ROIs (batched),
    # instead of once per person. This is the main speedup — a single
    # predict() call on a list of crops has far less overhead than N calls.
    if roi_crops:
        batched_boot_results = boot_model.predict(
            source=roi_crops,
            conf=BOOT_CONF,
            verbose=False,
        )

        for (roi_x1, roi_y1), boot_results in zip(roi_coords, batched_boot_results):
            # Collect ALL boxes above threshold, not just the single best one —
            # a person has two feet, so we want both boots drawn if both are found.
            boot_boxes = []
            no_boot_boxes = []

            for box in boot_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == BOOT_CLASS_ID:
                    boot_boxes.append((conf, box.xyxy[0].tolist()))
                elif cls_id == NO_BOOT_CLASS_ID:
                    no_boot_boxes.append((conf, box.xyxy[0].tolist()))

            for conf, (bx1, by1, bx2, by2) in boot_boxes:
                abs_x1, abs_y1 = roi_x1 + int(bx1), roi_y1 + int(by1)
                abs_x2, abs_y2 = roi_x1 + int(bx2), roi_y1 + int(by2)
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 200, 0), 2)
                cv2.putText(
                    frame, f"Safety Boot {conf:.2f}", (abs_x1, max(abs_y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
                )

            for conf, (nx1, ny1, nx2, ny2) in no_boot_boxes:
                abs_x1, abs_y1 = roi_x1 + int(nx1), roi_y1 + int(ny1)
                abs_x2, abs_y2 = roi_x1 + int(nx2), roi_y1 + int(ny2)
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                cv2.putText(
                    frame, f"No Boot {conf:.2f}", (abs_x1, max(abs_y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

    writer.write(frame)

    if DISPLAY_LIVE:
        cv2.imshow("Pose-based Boot Detection", frame)
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
