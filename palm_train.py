# =============================================================================
# RT-DETR fine-tuning for oil palm FFB (fresh fruit bunch) detection + counting
# Kaggle notebook script. Each "# %%" block can be pasted into its own cell.
#
# Kaggle settings: Accelerator = GPU (T4 or P100), Internet = ON
# Dataset: Roboflow "COCO JSON" export, uploaded as a Kaggle dataset:
#   <DATA_DIR>/train/_annotations.coco.json  + images
#   <DATA_DIR>/valid/_annotations.coco.json  + images
#   <DATA_DIR>/test/_annotations.coco.json   + images   (optional)
# =============================================================================

# %% 1. Install / upgrade packages (RT-DETRv2 needs transformers >= 4.52)
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers>=4.52", "accelerate", "albumentations", "torchmetrics", "pycocotools"],
    check=True,
)

# %% 2. Imports and config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single GPU: DataParallel breaks DETR label lists
os.environ["WANDB_DISABLED"] = "true"

import json
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
import albumentations as A
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

DATA_DIR    = "/kaggle/input/palm-ffb"        # <-- CHANGE to your dataset path
OUTPUT_DIR  = "/kaggle/working/rtdetr-palm-ffb"
MODEL_NAME  = "PekingU/rtdetr_v2_r18vd"        # smaller + faster on phone than r50vd
IMAGE_SIZE  = 640

EPOCHS      = 150
BATCH_SIZE  = 4
GRAD_ACC    = 2        # effective batch 8; with ~200 images, 16 would leave too few updates
LR_HEAD     = 1e-4     # transformer encoder/decoder + heads
LR_BACKBONE = 1e-5     # pretrained CNN backbone: 10x lower (official RT-DETR recipe)
WEIGHT_DECAY = 1e-4
SEED        = 42

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% 3. Read COCO annotations and build label mapping
def load_coco(split):
    with open(os.path.join(DATA_DIR, split, "_annotations.coco.json")) as f:
        coco = json.load(f)
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    return coco, anns_by_img

train_coco, _ = load_coco("train")

# Roboflow exports often include an unused "supercategory" with id 0.
# Keep only categories that actually appear in the annotations, mapped to 0..N-1.
used_cat_ids = sorted({a["category_id"] for a in train_coco["annotations"]})
cat_names    = {c["id"]: c["name"] for c in train_coco["categories"]}
CAT2LABEL    = {cid: i for i, cid in enumerate(used_cat_ids)}
id2label     = {i: cat_names[cid] for cid, i in CAT2LABEL.items()}
label2id     = {v: k for k, v in id2label.items()}
print("Classes:", id2label)

n_train_boxes = len(train_coco["annotations"])
print(f"Train images: {len(train_coco['images'])}, train boxes: {n_train_boxes}")

# %% 4. Augmentations (train only)
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),   # fine for top-down photos of bunches; remove if photos are always upright
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.2),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.3),
)

# %% 5. Dataset
class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, transform=None):
        self.split_dir = os.path.join(DATA_DIR, split)
        coco, self.anns = load_coco(split)
        self.images = coco["images"]
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def load(self, idx):
        """Returns (image_info, RGB numpy image, boxes in COCO xywh pixels, labels)."""
        info = self.images[idx]
        img = np.array(Image.open(os.path.join(self.split_dir, info["file_name"])).convert("RGB"))
        h, w = img.shape[:2]
        boxes, labels = [], []
        for a in self.anns.get(info["id"], []):
            if a["category_id"] not in CAT2LABEL:
                continue
            x, y, bw, bh = a["bbox"]
            # clip to image (Albumentations rejects boxes that leak outside)
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            boxes.append([x0, y0, x1 - x0, y1 - y0])
            labels.append(CAT2LABEL[a["category_id"]])
        return info, img, boxes, labels

    def __getitem__(self, idx):
        info, img, boxes, labels = self.load(idx)
        if self.transform is not None:
            out = self.transform(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = out["image"], list(out["bboxes"]), list(out["labels"])

        target = {
            "image_id": int(info["id"]),
            "annotations": [
                {
                    "bbox": [float(v) for v in b],
                    "category_id": int(l),
                    "area": float(b[2]) * float(b[3]),
                    "iscrowd": 0,
                }
                for b, l in zip(boxes, labels)
            ],
        }
        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }

# %% 6. Processor, model, datasets
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, size={"height": IMAGE_SIZE, "width": IMAGE_SIZE}
)
model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,   # re-initialises the class head for 1 class
)

train_ds = CocoDetDataset("train", processor, transform=train_transform)
val_ds   = CocoDetDataset("valid", processor, transform=None)
print(f"train={len(train_ds)}  valid={len(val_ds)}")

# quick sanity check: one sample must have boxes in normalised cx,cy,w,h
s = train_ds[0]
print("pixel_values:", tuple(s["pixel_values"].shape), "| boxes:", s["labels"]["boxes"][:3])

# %% 7. Optimizer with separate backbone LR
backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
other_params    = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
print(f"backbone tensors: {len(backbone_params)}, other tensors: {len(other_params)}")

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": other_params,    "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY,
)

# %% 8. Train
# Warmup given as an integer step count: newer transformers versions removed warmup_ratio.
import math
steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACC))
WARMUP_STEPS = max(1, int(0.05 * steps_per_epoch * EPOCHS))
print(f"steps/epoch={steps_per_epoch}, total steps={steps_per_epoch * EPOCHS}, warmup={WARMUP_STEPS}")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR_HEAD,              # used only for scheduler bookkeeping
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=0.1,                  # RT-DETR default gradient clipping
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    remove_unused_columns=False,        # required: keep "labels" list
    prediction_loss_only=True,          # eval = loss only; mAP/count eval done below
    dataloader_num_workers=2,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    optimizers=(optimizer, None),       # Trainer builds the cosine scheduler
    callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
)

trainer.train()

BEST_DIR = os.path.join(OUTPUT_DIR, "best")
trainer.save_model(BEST_DIR)
processor.save_pretrained(BEST_DIR)
print("Saved best model to", BEST_DIR)

# %% 9. Inference helpers + evaluation (mAP and COUNT error)
from torchmetrics.detection import MeanAveragePrecision

model = trainer.model.to(device).eval()

@torch.no_grad()
def predict_split(split):
    ds = CocoDetDataset(split, processor, transform=None)
    results = []
    for idx in range(len(ds)):
        info, img, boxes, labels = ds.load(idx)
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        det = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[img.shape[:2]]
        )[0]
        gt = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        gt[:, 2:] += gt[:, :2]                      # xywh -> xyxy
        results.append({
            "file_name": info["file_name"],
            "image": img,
            "scores": det["scores"].float().cpu(),
            "labels": det["labels"].cpu(),
            "boxes": det["boxes"].float().cpu(),
            "gt_boxes": gt,
            "gt_labels": torch.tensor(labels, dtype=torch.long),
        })
    return results


def compute_map(results, score_floor=0.001):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    preds, targets = [], []
    for r in results:
        keep = r["scores"] >= score_floor
        preds.append({"boxes": r["boxes"][keep], "scores": r["scores"][keep], "labels": r["labels"][keep]})
        targets.append({"boxes": r["gt_boxes"], "labels": r["gt_labels"]})
    metric.update(preds, targets)
    m = metric.compute()
    return float(m["map"]), float(m["map_50"])


def count_errors(results, thr):
    pred = np.array([int((r["scores"] >= thr).sum()) for r in results])
    gt   = np.array([len(r["gt_boxes"]) for r in results])
    return np.abs(pred - gt).mean(), (pred - gt).mean(), (pred == gt).mean()


def sweep_thresholds(results):
    rows = []
    for thr in np.arange(0.05, 0.91, 0.05):
        mae, bias, exact = count_errors(results, thr)
        rows.append((round(float(thr), 2), mae, bias, exact))
    print(f"{'thr':>5} {'count MAE':>10} {'bias':>7} {'exact%':>7}")
    for thr, mae, bias, exact in rows:
        print(f"{thr:>5.2f} {mae:>10.2f} {bias:>+7.2f} {exact*100:>6.1f}%")
    best = min(rows, key=lambda r: (r[1], abs(r[2])))
    return best[0]


val_results = predict_split("valid")
val_map, val_map50 = compute_map(val_results)
print(f"\nVALID  mAP50-95={val_map:.3f}  mAP50={val_map50:.3f}\n")
BEST_THR = sweep_thresholds(val_results)
print(f"\nBest counting threshold on valid: {BEST_THR}")

with open(os.path.join(BEST_DIR, "count_threshold.json"), "w") as f:
    json.dump({"threshold": BEST_THR}, f)

# %% 10. Test set (if present) using the threshold chosen on valid
if os.path.exists(os.path.join(DATA_DIR, "test", "_annotations.coco.json")):
    test_results = predict_split("test")
    t_map, t_map50 = compute_map(test_results)
    mae, bias, exact = count_errors(test_results, BEST_THR)
    print(f"TEST  mAP50-95={t_map:.3f}  mAP50={t_map50:.3f}")
    print(f"TEST  count MAE={mae:.2f}  bias={bias:+.2f}  exact={exact*100:.1f}%  (thr={BEST_THR})")
else:
    test_results = None
    print("No test split found; skipping.")

# %% 11. Save visualisations (GREEN = ground truth, RED = prediction)
VIS_DIR = "/kaggle/working/vis"
os.makedirs(VIS_DIR, exist_ok=True)

def save_vis(results, thr, n=15):
    for r in results[:n]:
        im = Image.fromarray(r["image"])
        d = ImageDraw.Draw(im)
        for b in r["gt_boxes"].tolist():
            d.rectangle(b, outline=(0, 255, 0), width=3)
        keep = r["scores"] >= thr
        for b, s in zip(r["boxes"][keep].tolist(), r["scores"][keep].tolist()):
            d.rectangle(b, outline=(255, 60, 0), width=3)
            d.text((b[0] + 4, b[1] + 2), f"{s:.2f}", fill=(255, 60, 0))
        d.text((10, 10), f"GT={len(r['gt_boxes'])}  Pred={int(keep.sum())}", fill=(255, 255, 0))
        im.save(os.path.join(VIS_DIR, os.path.basename(r["file_name"])))

save_vis(test_results if test_results else val_results, BEST_THR)
print("Visualisations saved to", VIS_DIR)
