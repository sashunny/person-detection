"""
Truck detection - YOLO training script for Kaggle (GPU).

Before running on Kaggle:
  1. Settings -> Accelerator -> GPU (T4 or P100).
  2. Settings -> Internet -> On (needed to pip install and download pretrained weights).
  3. Add your uploaded dataset via "Add Input".

Run in a notebook cell with:   !python train_truck_yolo.py
Or paste the whole file into a cell.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIG - adjust if needed
# ----------------------------------------------------------------------------
INPUT_ROOT = Path("/kaggle/input")      # where Kaggle mounts your dataset
WORK_DIR = Path("/kaggle/working")      # writable output folder
DATA_DIR = WORK_DIR / "dataset"         # dataset is copied here (input is read-only)

MODEL = "yolo26n.pt"    # lightweight nano model; try "yolo26s.pt" if accuracy is low
IMGSZ = 640             # use 960 if trucks are small/far from the camera
EPOCHS = 100
BATCH = 16              # lower to 8 if you get CUDA out-of-memory
PATIENCE = 25           # stop early if val mAP doesn't improve for this many epochs
CLASS_NAMES = ["truck"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TRAIN_NAMES = {"train", "training"}
VAL_NAMES = {"val", "valid", "validation"}


# ----------------------------------------------------------------------------
# 1. Install ultralytics and check the GPU
# ----------------------------------------------------------------------------
def setup():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU found. In Kaggle: Settings -> Accelerator -> GPU, then restart.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ----------------------------------------------------------------------------
# 2. Find the train/val image folders inside /kaggle/input
#    Expects CVAT YOLO export structure: images/<subset>/ and labels/<subset>/
# ----------------------------------------------------------------------------
def find_split_dirs():
    train_dir = val_dir = None
    for d in INPUT_ROOT.rglob("*"):
        if d.is_dir() and d.parent.name == "images":
            name = d.name.lower()
            if name in TRAIN_NAMES:
                train_dir = d
            elif name in VAL_NAMES:
                val_dir = d
    if train_dir is None or val_dir is None:
        found = sorted(str(d) for d in INPUT_ROOT.rglob("images") if d.is_dir())
        sys.exit(
            "Could not find both images/train and images/validation folders.\n"
            f"'images' folders found: {found}\n"
            "If train and val ended up in the same folder after merging, "
            "re-export so each project goes into its own subset folder."
        )
    print(f"Train images: {train_dir}\nVal images:   {val_dir}")
    return train_dir, val_dir


# ----------------------------------------------------------------------------
# 3. Copy into a clean, writable structure and sanity-check labels
# ----------------------------------------------------------------------------
def copy_split(src_img_dir, split):
    src_lbl_dir = src_img_dir.parent.parent / "labels" / src_img_dir.name
    dst_img = DATA_DIR / "images" / split
    dst_lbl = DATA_DIR / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_img = n_boxes = n_background = n_missing = 0
    bad_class = set()
    names = set()

    for img in sorted(src_img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        n_img += 1
        names.add(img.stem)
        shutil.copy2(img, dst_img / img.name)

        lbl = src_lbl_dir / img.relative_to(src_img_dir).with_suffix(".txt")
        dst = dst_lbl / f"{img.stem}.txt"
        if lbl.exists():
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            for l in lines:
                cls = int(l.split()[0])
                if cls >= len(CLASS_NAMES):
                    bad_class.add(cls)
            shutil.copy2(lbl, dst)
            n_boxes += len(lines)
            if not lines:
                n_background += 1
        else:
            dst.write_text("")      # no label file -> treat as empty-bay background
            n_missing += 1
            n_background += 1

    print(f"\n[{split}] images: {n_img} | boxes: {n_boxes} | "
          f"background (no truck): {n_background} ({100 * n_background / max(n_img, 1):.0f}%) | "
          f"missing label files: {n_missing}")
    if bad_class:
        sys.exit(f"[{split}] found class ids {sorted(bad_class)} but CLASS_NAMES has "
                 f"{len(CLASS_NAMES)} class(es). Fix CLASS_NAMES or the labels.")
    return names


def prepare_dataset():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    train_dir, val_dir = find_split_dirs()
    train_names = copy_split(train_dir, "train")
    val_names = copy_split(val_dir, "val")

    overlap = train_names & val_names
    if overlap:
        print(f"\nWARNING: {len(overlap)} file names appear in both train and val, "
              f"e.g. {sorted(overlap)[:3]}. Check they are really different frames.")

    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATA_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    return yaml_path


# ----------------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------------
def train(yaml_path):
    from ultralytics import YOLO

    model = YOLO(MODEL)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=0,
        workers=2,
        cache="ram",          # dataset is small, so cache it for speed
        project=str(WORK_DIR / "runs"),
        name="truck_v1",
        exist_ok=True,
        seed=0,
        plots=True,
        # Augmentation (Ultralytics defaults are good; these suit fixed CCTV)
        fliplr=0.5,
        degrees=0.0,          # cameras don't rotate
        mosaic=1.0,
        close_mosaic=10,
    )

    best = WORK_DIR / "runs" / "truck_v1" / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    # Final validation with the best model
    metrics = YOLO(best).val(data=str(yaml_path), imgsz=IMGSZ, device=0)
    print(f"Val mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f} | "
          f"Precision: {metrics.box.mp:.3f} | Recall: {metrics.box.mr:.3f}")


if __name__ == "__main__":
    setup()
    yaml_path = prepare_dataset()
    train(yaml_path)
