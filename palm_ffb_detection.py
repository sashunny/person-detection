# =============================================================================
# Stage 1 (detection): YOLO26 fine-tuning for oil palm FFB counting on Kaggle
# Kaggle settings: Accelerator = GPU (T4/P100), Internet = ON
# Dataset: Roboflow export in "YOLOv8" / "YOLO26" format, uploaded as a Kaggle dataset:
#   <DATA>/train/images, <DATA>/train/labels
#   <DATA>/valid/images, <DATA>/valid/labels
#   <DATA>/test/images,  <DATA>/test/labels   (optional)
# Each "# %%" block can be its own notebook cell.
# =============================================================================

# %% 1. Install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)

# %% 2. Config + dataset yaml (Roboflow's data.yaml uses relative paths that break on Kaggle)
import os, glob, json
import numpy as np
import yaml
from ultralytics import YOLO

DATA    = "/kaggle/input/palm-ffb-yolo"     # <-- CHANGE to your dataset path
WORK    = "/kaggle/working"
MODEL   = "yolo26s.pt"                       # "yolo26n.pt" = smaller/faster, slightly less accurate
IMGSZ   = 640
EPOCHS  = 150
RUN     = "ffb_yolo26s"

has_test = os.path.isdir(f"{DATA}/test/images")
cfg = {
    "path": DATA,
    "train": "train/images",
    "val": "valid/images",
    "names": {0: "Palm-FFB"},
}
if has_test:
    cfg["test"] = "test/images"
DATA_YAML = f"{WORK}/ffb.yaml"
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(cfg, f)
print(open(DATA_YAML).read())

# %% 3. Train
model = YOLO(MODEL)
model.train(
    data=DATA_YAML,
    single_cls=True,          # treat every box as "bunch" (also works later if boxes carry ripeness classes)
    epochs=EPOCHS,
    patience=50,              # YOLO early stopping uses mAP (fitness), not loss
    imgsz=IMGSZ,
    batch=16,
    cos_lr=True,
    # augmentation suited to top-down photos of bunches
    fliplr=0.5, flipud=0.5, degrees=10, scale=0.5, translate=0.1,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    mosaic=1.0, close_mosaic=15,
    project=f"{WORK}/runs", name=RUN, exist_ok=True,
    seed=42, plots=True,
)

BEST = f"{WORK}/runs/{RUN}/weights/best.pt"

# %% 4. Standard metrics
best = YOLO(BEST)
m = best.val(data=DATA_YAML, split="val", imgsz=IMGSZ, plots=True)
print(f"VALID  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  "
      f"P={m.box.mp:.3f}  R={m.box.mr:.3f}")

# %% 5. Pick the confidence threshold that gives the best COUNT on validation
def label_path(img_path):
    stem = os.path.splitext(img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0]
    return stem + ".txt"

def gt_count(img_path):
    p = label_path(img_path)
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p) if line.strip())

def collect(split):
    imgs = sorted(glob.glob(f"{DATA}/{split}/images/*"))
    gts, scores = [], []
    for r in best.predict(imgs, conf=0.01, iou=0.7, imgsz=IMGSZ, stream=True, verbose=False):
        scores.append(r.boxes.conf.cpu().numpy())
        gts.append(gt_count(r.path))
    return np.array(gts), scores

def count_stats(gts, scores, thr):
    pred = np.array([(s >= thr).sum() for s in scores])
    err = pred - gts
    return np.abs(err).mean(), err.mean(), (err == 0).mean()

gts, scores = collect("valid")
print(f"{'thr':>5} {'count MAE':>10} {'bias':>7} {'exact%':>7}")
rows = []
for thr in np.arange(0.10, 0.81, 0.05):
    mae, bias, exact = count_stats(gts, scores, thr)
    rows.append((round(float(thr), 2), mae, bias, exact))
    print(f"{thr:>5.2f} {mae:>10.2f} {bias:>+7.2f} {exact*100:>6.1f}%")
BEST_THR = min(rows, key=lambda r: (r[1], abs(r[2])))[0]
print(f"\nBest counting threshold: {BEST_THR}")
json.dump({"threshold": BEST_THR}, open(f"{WORK}/runs/{RUN}/count_threshold.json", "w"))

if has_test:
    tg, ts = collect("test")
    mae, bias, exact = count_stats(tg, ts, BEST_THR)
    print(f"TEST  count MAE={mae:.2f}  bias={bias:+.2f}  exact={exact*100:.1f}%  (thr={BEST_THR})")

# %% 6. Save prediction images for a visual check (boxes + scores)
vis_imgs = sorted(glob.glob(f"{DATA}/valid/images/*"))[:20]
best.predict(vis_imgs, conf=BEST_THR, iou=0.7, imgsz=IMGSZ, save=True,
             project=f"{WORK}/runs", name=f"{RUN}_vis", exist_ok=True, verbose=False)
print("Prediction images saved to", f"{WORK}/runs/{RUN}_vis")

# %% 7. Export for Android (TFLite / LiteRT). FP16 keeps accuracy; try int8 later if you need speed.
best.export(format="tflite", imgsz=IMGSZ, half=True)
print("Exported files are next to", BEST)
