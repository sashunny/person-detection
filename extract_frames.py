"""
Extract frames from video(s) at a fixed rate (default: 2 frames per second),
for building an annotation dataset from failure-case / complex-case footage.

Just edit the CONFIG section below and run:
    pip install opencv-python --break-system-packages
    python extract_frames.py
"""

from pathlib import Path
import cv2

# ----------------- CONFIG (edit these) -----------------
INPUT_PATH = "/path/to/video_or_folder"   # a single video file OR a folder containing multiple videos
OUTPUT_DIR = "/path/to/extracted_frames"  # frames get saved here

FRAMES_PER_SECOND = 2      # how many frames to extract per second of video
IMAGE_FORMAT = "jpg"       # "jpg" or "png"
JPEG_QUALITY = 95          # only used if IMAGE_FORMAT == "jpg" (0-100)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
# ---------------------------------------------------------

input_path = Path(INPUT_PATH)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Collect the list of videos to process — either one file, or every video in a folder
if input_path.is_file():
    video_paths = [input_path]
elif input_path.is_dir():
    video_paths = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )
else:
    raise SystemExit(f"INPUT_PATH does not exist: {INPUT_PATH}")

if not video_paths:
    raise SystemExit(f"No video files found at: {INPUT_PATH}")

print(f"Found {len(video_paths)} video(s) to process.\n")

total_saved = 0

for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Could not open {video_path.name} — skipping.")
        continue

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many source frames to skip between each frame we save, to hit
    # the target extraction rate (e.g. source is 25fps, target is 2fps -> skip 12/13 frames)
    frame_interval = max(int(round(source_fps / FRAMES_PER_SECOND)), 1)

    video_stem = video_path.stem  # filename without extension, used as a prefix
    frame_idx = 0
    saved_count = 0

    print(f"Processing {video_path.name} (source: {source_fps:.1f} fps, "
          f"~{total_frames} frames, saving every {frame_interval}th frame)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{video_stem}_frame{frame_idx:06d}.{IMAGE_FORMAT}"
            out_path = output_dir / out_name

            if IMAGE_FORMAT == "jpg":
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                cv2.imwrite(str(out_path), frame)

            saved_count += 1

        frame_idx += 1

    cap.release()
    total_saved += saved_count
    print(f"  -> saved {saved_count} frames from {video_path.name}\n")

print(f"Done. {total_saved} frames saved total to: {output_dir}")
