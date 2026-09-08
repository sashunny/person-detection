"""
Train a lightweight YOLOv8 detector for boot detection (2 classes: Boot, No Boot).

Just edit the CONFIG section below and run:
    pip install ultralytics --break-system-packages
    python train_boot_detector.py

Your data.yaml should look something like:

    train: /path/to/dataset/train/images
    val:   /path/to/dataset/valid/images
    test:  /path/to/dataset/test/images
    nc: 2
    names: ['Boot', 'No_Boot']

Make sure class indices in your label .txt files match: 0 = Boot, 1 = No_Boot.
"""

from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
DATA_YAML = "/path/to/your_dataset/data.yaml"   # your dataset config
BASE_MODEL = "yolov8n.pt"                       # lightweight pretrained checkpoint to fine-tune from
                                                 # options: yolov8n.pt (fastest) / yolov8s.pt (more accurate, still light)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16          # lower this (e.g. 8) if you hit GPU/CPU memory errors
DEVICE = 0               # 0 = first GPU, "cpu" if no GPU available
PROJECT_DIR = "runs_boot_detection"
RUN_NAME = "boot_detector_v1"

# Augmentation tuned for real-world RTSP-style degradation (compression,
# blur, lighting variation) rather than the Ultralytics defaults, since
# production frames will be noisier than clean training images.
AUGMENTATION = dict(
    hsv_h=0.015,      # slight hue jitter (lighting/camera variation)
    hsv_s=0.5,        # saturation jitter
    hsv_v=0.4,        # brightness jitter (helps with poor lighting on site)
    degrees=5.0,      # small rotation, cameras are rarely perfectly level
    translate=0.1,
    scale=0.4,        # helps with small-object (boot) scale variation
    fliplr=0.5,
    mosaic=1.0,       # combines 4 images, good for small-object detection
    blur=0.05,        # simulate motion blur / compression softness (Ultralytics >=8.1)
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
    patience=20,        # early stopping if val metrics stop improving
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