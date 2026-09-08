"""
Simple person detection over a YOLOv8-format dataset (train/valid/test)
using a lightweight COCO-pretrained YOLO model. No training needed.

Just edit the CONFIG section below and run:
    pip install ultralytics opencv-python --break-system-packages
    python detect_persons_simple.py
"""

from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
DATASET_ROOT = "/path/to/your_dataset"      # folder containing train/ valid/ test/
OUTPUT_DIR = "/path/to/output_folder"       # where annotated images + CSV will go
MODEL_NAME = "yolov8n.pt"                   # pretrained, auto-downloads
CONF_THRESHOLD = 0.25
PERSON_CLASS_ID = 0                         # COCO class id for 'person'
SPLITS = ["train", "valid", "test"]         # change to ["train", "val", "test"] if that's your folder naming
# ---------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

model = YOLO(MODEL_NAME)
output_root = Path(OUTPUT_DIR)
output_root.mkdir(parents=True, exist_ok=True)

csv_rows = []

for split in SPLITS:
    img_dir = Path(DATASET_ROOT) / split / "images"
    if not img_dir.exists():
        print(f"[!] Skipping '{split}' — folder not found: {img_dir}")
        continue

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    print(f"[{split}] {len(images)} images found")

    split_out_dir = output_root / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images, 1):
        results = model.predict(
            source=str(img_path),
            classes=[PERSON_CLASS_ID],
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        # Save annotated image with boxes drawn
        annotated = result.plot()
        out_path = split_out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        num_persons = len(result.boxes)
        mean_conf = float(result.boxes.conf.mean()) if num_persons > 0 else 0.0

        csv_rows.append([split, img_path.name, num_persons, round(mean_conf, 4), str(out_path)])

        if i % 50 == 0 or i == len(images):
            print(f"  processed {i}/{len(images)}")

# Save CSV, sorted so zero-detection images appear first (easiest to check manually)
csv_rows.sort(key=lambda r: (r[2], r[3]))
csv_path = output_root / "detection_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image", "num_persons_detected", "mean_confidence", "annotated_path"])
    writer.writerows(csv_rows)

zero_count = sum(1 for r in csv_rows if r[2] == 0)
print(f"\nDone. {len(csv_rows)} images processed.")
print(f"{zero_count} images had ZERO persons detected — check these first.")
print(f"Annotated images saved under: {output_root}")
print(f"Summary CSV: {csv_path}")