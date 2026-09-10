"""
Fine-tune the EXISTING trained boot detector (not a fresh COCO checkpoint)
on a small new batch of annotated failure-case images. This is an
active-learning-style update: start from your current best.pt weights,
keep learning gentle, and don't let 88 training images wreck what the
model already learned from your larger original dataset.

Just edit the CONFIG section below and run:
    pip install ultralytics --break-system-packages
    python finetune_boot_detector.py

Your data.yaml should point to the NEW annotated set only:

    train: /path/to/new_dataset/train/images
    val:   /path/to/new_dataset/valid/images
    nc: 2
    names: ['Boot', 'No_Boot']

IMPORTANT: with only 88 training images, this run is fragile. Read the
notes at the bottom of this file before trusting the result blindly.
"""

from ultralytics import YOLO

# ----------------- CONFIG (edit these) -----------------
DATA_YAML = "/path/to/new_failure_cases_dataset/data.yaml"
EXISTING_MODEL_PATH = "/path/to/your/existing/best.pt"   # your CURRENT trained boot model — NOT yolov26s.pt fresh

EPOCHS = 30                # small dataset -> few epochs. More epochs on 88 images overfits fast.
IMG_SIZE = 640
BATCH_SIZE = 8              # small dataset -> small batch; raise only if you have plenty of images per batch
DEVICE = 0                  # 0 = first GPU, "cpu" if no GPU available
PROJECT_DIR = "runs_boot_finetune"
RUN_NAME = "boot_detector_finetune_v1"

LEARNING_RATE = 0.001       # lower than a fresh-training LR (default ~0.01) — gentle updates,
                             # since we're refining an already-good model, not learning from scratch
FREEZE_LAYERS = 10          # freeze the first N layers (backbone) so early feature extraction
                             # doesn't get overwritten by a tiny dataset — only the head adapts

# Light augmentation on purpose — with only 88 images, heavy augmentation
# can distort the few examples you have rather than generalizing usefully.
AUGMENTATION = dict(
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
    degrees=3.0,
    translate=0.05,
    scale=0.3,
    fliplr=0.5,
    mosaic=0.5,        # reduced from 1.0 — mosaic combines 4 images, which can overwhelm
                        # a tiny dataset's already-limited diversity
)
# ---------------------------------------------------------

model = YOLO(EXISTING_MODEL_PATH)  # load YOUR trained weights, not a fresh pretrained checkpoint

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    lr0=LEARNING_RATE,
    freeze=FREEZE_LAYERS,
    patience=10,          # stop early if val metrics stop improving — important safety net on a small dataset
    save=True,
    plots=True,
    **AUGMENTATION,
)

print("\nFine-tuning complete.")
print(f"Best weights saved at: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
print(f"Training plots/metrics saved at: {PROJECT_DIR}/{RUN_NAME}/")

# Validate on the held-out split with the new weights
best_model = YOLO(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
metrics = best_model.val(data=DATA_YAML)
print("\nValidation metrics (new weights):")
print(f"  mAP50:    {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")

print("""
--- Before you trust this ---
1. Compare these numbers against your ORIGINAL model's mAP on the SAME
   new failure-case val set (run the old best.pt through model.val() on
   this data.yaml too) — you want to see the new weights improve on the
   hard cases, not just report a number in isolation.
2. Also re-validate the NEW weights against your ORIGINAL validation set
   (the one from your first training run). With only 88 training images,
   there's real risk of overfitting to the new cases while quietly
   regressing on the general cases the model used to handle fine. If the
   new model does worse on the old val set, that's a problem even if it
   improved on the failure cases.
3. If step 2 shows regression, try a higher FREEZE_LAYERS (freeze more of
   the backbone) or fewer EPOCHS before accepting these weights.
""")
