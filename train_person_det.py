"""
Train/fine-tune a lightweight YOLOv8 person detector on your own data.

Just edit the CONFIG section below and run:
    pip install ultralytics --break-system-packages
    python train_person_detector.py

Your data.yaml should look something like:

    train: /path/to/dataset/train/images
    val:   /path/to/dataset/valid/images
    test:  /path/to/dataset/test/images
    nc: 1
    names: ['person']

Make sure your label .txt files only use class index 0 (person).

Note: starting from a COCO-pretrained checkpoint (yolov8n.pt) means the
model already knows 'person' reasonably well out of the box. Fine-tuning
here is meant to adapt it to YOUR specific camera angle/lighting/resolution
(e.g. overhead-mounted cameras, poor RTSP quality) rather than teaching it
the concept of a person from scratch — so you likely need far fewer epochs
and less data than a from-scratch model would.
"""

from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
DATA_YAML = "/path/to/your_person_dataset/data.yaml"
BASE_MODEL = "yolov8n.pt"          # lightweight COCO-pretrained checkpoint to fine-tune from
                                    # options: yolov8n.pt (fastest) / yolov8s.pt (more accurate, still light)
EPOCHS = 60                        # fine-tuning needs fewer epochs than training from scratch
IMG_SIZE = 640
BATCH_SIZE = 16                    # lower this (e.g. 8) if you hit GPU/CPU memory errors
DEVICE = 0                         # 0 = first GPU, "cpu" if no GPU available
PROJECT_DIR = "runs_person_detection"
RUN_NAME = "person_detector_v1"

# Augmentation geared toward camera/RTSP variation (angle, lighting,
# small/far-away people) rather than the Ultralytics defaults.
AUGMENTATION = dict(
    hsv_h=0.015,      # slight hue jitter (lighting/camera variation)
    hsv_s=0.5,
    hsv_v=0.4,        # brightness jitter (poor on-site lighting)
    degrees=5.0,      # small rotation, cameras are rarely perfectly level
    translate=0.1,
    scale=0.4,        # helps with people at varying distances from camera
    fliplr=0.5,
    mosaic=1.0,       # combines 4 images, helps with small/far-away people
)
# ---------------------------------------------------------

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    patience=15,        # early stopping if val metrics stop improving
    save=True,
    plots=True,          # saves PR curve, confusion matrix, etc. for review
    **AUGMENTATION,
)

print("\nTraining complete.")
print(f"Best weights saved at: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
print(f"Training plots/metrics saved at: {PROJECT_DIR}/{RUN_NAME}/")

# Run validation on the val split with the best weights and print metrics
best_model = YOLO(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
metrics = best_model.val(data=DATA_YAML)
print("\nValidation metrics:")
print(f"  mAP50:    {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")