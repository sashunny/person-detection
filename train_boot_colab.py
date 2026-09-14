# =========================================================
# Colab notebook — fine-tune boot detector on Drive-hosted data
# Paste each section below into separate Colab cells (the "# %%"
# markers show where a natural cell break is), or run as one script.
# =========================================================

# --- CELL 1: install dependencies ---
# !pip install ultralytics --quiet


# --- CELL 2: mount Drive (skip if already mounted this session) ---
from google.colab import drive
drive.mount('/content/drive')


# --- CELL 3: config ---
DATA_YAML = "/content/drive/MyDrive/boot_detection/dataset/data.yaml"
EXISTING_MODEL_PATH = "/content/drive/MyDrive/boot_detection/models/best.pt"  # coworker's/your current weights

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0              # Colab GPU runtime -> 0. Make sure Runtime > Change runtime type > GPU is selected.
PROJECT_DIR = "/content/drive/MyDrive/boot_detection/runs_finetune_v2"  # saved to DRIVE so it survives disconnects
RUN_NAME = "boot_finetune_v2"

LEARNING_RATE = 0.001
FREEZE_LAYERS = 10

AUGMENTATION = dict(
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=4.0,
    translate=0.1,
    scale=0.4,
    shear=2.0,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.5,
    erasing=0.15,
)


# --- CELL 4: your data.yaml MUST only reference train/val ---
# IMPORTANT: since you don't have a test split, your data.yaml should look
# like this (no "test:" key at all — not even an empty one):
#
#     train: /content/drive/MyDrive/boot_detection/dataset/train/images
#     val:   /content/drive/MyDrive/boot_detection/dataset/valid/images
#     nc: 2
#     names: ['Boot', 'No_Boot']
#
# Ultralytics only evaluates on whatever split(s) exist in data.yaml, so
# leaving "test:" out entirely means model.val() below will only ever
# touch train/val — nothing runs against test data at any point.


# --- CELL 5: train ---
from ultralytics import YOLO

model = YOLO(EXISTING_MODEL_PATH)

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
    patience=10,
    save=True,
    plots=True,
    **AUGMENTATION,
)

print("\nTraining complete.")
print(f"Best weights saved at: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


# --- CELL 6: validate on the validation split only ---
best_model = YOLO(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
val_metrics = best_model.val(data=DATA_YAML, split="val")  # explicit split="val" — never touches test
print("\nValidation metrics:")
print(f"  mAP50:    {val_metrics.box.map50:.4f}")
print(f"  mAP50-95: {val_metrics.box.map:.4f}")


# --- CELL 7: download the fine-tuned weights to your LOCAL laptop ---
# This triggers your browser's normal file-save dialog — saves directly
# to your laptop's Downloads folder, no need to go through Drive's website.
from google.colab import files

best_pt_path = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
files.download(best_pt_path)

print(f"\nDownloading: {best_pt_path}")
print("Once downloaded, use this weights file in your local test-video scripts.")
