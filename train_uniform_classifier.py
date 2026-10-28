"""
Uniform vs No-Uniform person-crop classifier  (Kaggle-ready)
===========================================================

Expected data layout (any nesting under DATA_ROOT is fine; folders are found automatically):
    DATA_ROOT/
        uniform/       *.jpg / *.png ...
        no uniform/    *.jpg / *.png ...   ("no_uniform" / "no-uniform" also work)

What it does
  1. Collects images, then splits train/val (default 80/20), stratified by class.
     Near-duplicate crops (consecutive frames) are grouped so they never land on
     both sides of the split -> honest validation numbers.
     If your filenames contain the location/video name, set GROUP_REGEX to split
     by location instead (even better).
  2. Letterboxes every crop to 224x224 (keeps body proportions, gray padding).
  3. Fine-tunes ImageNet-pretrained EfficientNet-B0 (or MobileNetV3-Large):
     a few epochs with the backbone frozen, then full fine-tuning.
  4. Keeps the best epoch, tunes the decision threshold on val, and saves:
        /kaggle/working/uniform_classifier.pt        (weights + preprocessing meta)
        /kaggle/working/uniform_classifier_meta.json
        /kaggle/working/uniform_classifier.onnx      (if export succeeds)
        /kaggle/working/split.csv                    (which file went where)
        /kaggle/working/val_mistakes.csv             (misclassified val crops)

Kaggle notes
  * Turn on a GPU (Settings -> Accelerator -> GPU T4/P100).
  * Turn on Internet once so torchvision can download the pretrained weights.
"""

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ============================== CONFIG ==============================
DATA_ROOT = "/kaggle/input"          # point to your dataset, e.g. "/kaggle/input/uniform-crops"
OUT_DIR = "/kaggle/working"

MODEL_NAME = "efficientnet_b0"       # or "mobilenet_v3_large" (faster, slightly less accurate)
IMG_SIZE = 224
PAD_COLOR = (114, 114, 114)          # letterbox padding color — use the SAME at inference

VAL_FRACTION = 0.2
SEED = 42

# Split grouping (prevents near-identical crops leaking between train and val)
GROUP_REGEX = None                   # e.g. r"^(loc\d+)_" if files look like "loc07_frame123_p2.jpg"
VAL_GROUPS = []                      # optional: exact groups (from GROUP_REGEX) to force into val, e.g. ["loc03","loc11"]
DEDUP_GROUPING = True                # used when GROUP_REGEX is None
DEDUP_MAX_BITS = 10                  # dHash Hamming distance (0-64) to call two crops near-duplicates

BATCH_SIZE = 64
EPOCHS = 25
WARMUP_EPOCHS = 3                    # epochs with frozen backbone (train head only)
LR_HEAD = 1e-3
LR_FULL = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PATIENCE = 6                         # early stopping on val balanced accuracy
NUM_WORKERS = 2

CLASSES = ["no_uniform", "uniform"]  # index 1 = uniform (positive class)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ====================================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_name(s):
    return re.sub(r"[\s_\-]+", "", s.lower())


def find_class_dir(root, wanted):
    """Find a folder whose normalized name matches (e.g. 'no uniform' == 'no_uniform')."""
    target = norm_name(wanted)
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if norm_name(d) == target:
                return Path(dirpath) / d
    raise FileNotFoundError(f"Could not find a '{wanted}' folder under {root}")


def collect_images():
    rows = []
    for label, cls in enumerate(CLASSES):
        cdir = find_class_dir(DATA_ROOT, cls)
        files = sorted(p for p in cdir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"{cls:>11}: {len(files):6d} images  ({cdir})")
        rows += [{"path": str(p), "label": label} for p in files]
    if not rows:
        raise RuntimeError("No images found.")
    return pd.DataFrame(rows)


# ------------------------- near-duplicate grouping -------------------------
def dhash_bits(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def dedup_groups(df):
    """Union near-duplicate crops (within each class) into groups."""
    groups = np.empty(len(df), dtype=object)
    for label in df["label"].unique():
        idx = np.where(df["label"].values == label)[0]
        bits = np.stack([dhash_bits(df["path"].iloc[i]) for i in idx]).astype(np.uint8)
        n = len(idx)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        chunk = 256
        for s in range(0, n, chunk):
            d = (bits[s:s + chunk, None, :] != bits[None, :, :]).sum(-1)
            ii, jj = np.nonzero(d <= DEDUP_MAX_BITS)
            for i, j in zip(ii + s, jj):
                if j > i:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        for k, i in enumerate(idx):
            groups[i] = f"{label}_{find(k)}"
    return groups


def make_split(df):
    names = df["path"].map(lambda p: Path(p).name)
    if GROUP_REGEX:
        groups = names.map(lambda n: (re.search(GROUP_REGEX, n) or [None, "unknown"])[1]).values
        print(f"Grouping by GROUP_REGEX -> {len(set(groups))} groups")
    elif DEDUP_GROUPING:
        print("Grouping near-duplicate crops (dHash)...")
        groups = dedup_groups(df)
        print(f"{len(df)} images -> {len(set(groups))} near-duplicate groups")
    else:
        groups = np.arange(len(df)).astype(str)

    df = df.copy()
    df["group"] = groups

    if GROUP_REGEX and VAL_GROUPS:
        df["split"] = np.where(df["group"].isin(VAL_GROUPS), "val", "train")
    else:
        n_splits = max(2, round(1 / VAL_FRACTION))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        _, val_idx = next(sgkf.split(df, df["label"], df["group"]))
        df["split"] = "train"
        df.loc[df.index[val_idx], "split"] = "val"

    print("\nSplit summary:")
    print(pd.crosstab(df["split"], df["label"].map(dict(enumerate(CLASSES)))))
    return df


# ------------------------------ data ------------------------------
class Letterbox:
    """Resize longest side to `size`, pad to square — keeps person proportions."""
    def __init__(self, size, fill=PAD_COLOR):
        self.size, self.fill = size, fill

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # mild color jitter only — color is often what defines a uniform; hue untouched
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.0),
    Letterbox(IMG_SIZE),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=PAD_COLOR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0),  # simulates partial occlusion
])

val_tf = transforms.Compose([
    Letterbox(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class CropDataset(Dataset):
    def __init__(self, df, tf):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i], i


# ------------------------------ model ------------------------------
def build_model(name):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    else:
        raise ValueError(name)
    return m


def set_backbone_trainable(model, flag):
    for p in model.features.parameters():
        p.requires_grad = flag


# ---------------------------- train / eval ----------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, backbone_frozen=False):
    train = optimizer is not None
    model.train(train)
    if train and backbone_frozen:
        model.features.eval()  # keep BatchNorm stats fixed while backbone is frozen

    total_loss, probs, labels, idxs = 0.0, [], [], []
    for x, y, i in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        probs.append(torch.softmax(logits.float(), 1)[:, 1].detach().cpu())
        labels.append(y.cpu())
        idxs.append(i)
    n = len(loader.dataset)
    return (total_loss / n, torch.cat(probs).numpy(),
            torch.cat(labels).numpy(), torch.cat(idxs).numpy())


def metrics(probs, labels, thr=0.5):
    pred = (probs >= thr).astype(int)
    out = {
        "acc": (pred == labels).mean(),
        "bal_acc": balanced_accuracy_score(labels, pred),
        "f1_uniform": f1_score(labels, pred, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        out["auc"] = float("nan")
    return out


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = make_split(collect_images())
    df.to_csv(f"{OUT_DIR}/split.csv", index=False)
    tr_df, va_df = df[df.split == "train"], df[df.split == "val"].reset_index(drop=True)

    train_dl = DataLoader(CropDataset(tr_df, train_tf), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl = DataLoader(CropDataset(va_df, val_tf), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # class weights handle imbalance between uniform / no_uniform
    counts = np.bincount(tr_df["label"], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    print("Class weights:", weights.cpu().numpy().round(3))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    model = build_model(MODEL_NAME).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    ckpt_path = f"{OUT_DIR}/uniform_classifier.pt"

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            set_backbone_trainable(model, False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
            scheduler = None
            print(f"\n--- Phase 1: head only ({WARMUP_EPOCHS} epochs) ---")
        if epoch == WARMUP_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
            print("\n--- Phase 2: full fine-tuning ---")

        t0 = time.time()
        frozen = epoch <= WARMUP_EPOCHS
        tr_loss, tr_p, tr_y, _ = run_epoch(model, train_dl, criterion, device, optimizer, scaler, frozen)
        va_loss, va_p, va_y, _ = run_epoch(model, val_dl, criterion, device)
        if scheduler:
            scheduler.step()

        tm, vm = metrics(tr_p, tr_y), metrics(va_p, va_y)
        print(f"Ep {epoch:2d} | train loss {tr_loss:.4f} acc {tm['acc']:.3f} | "
              f"val loss {va_loss:.4f} acc {vm['acc']:.3f} bal_acc {vm['bal_acc']:.3f} "
              f"f1 {vm['f1_uniform']:.3f} auc {vm['auc']:.3f} | {time.time() - t0:.0f}s")

        if vm["bal_acc"] > best_score:
            best_score, best_epoch, bad_epochs = vm["bal_acc"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print("   ↳ saved best")
        else:
            bad_epochs += 1
            if epoch > WARMUP_EPOCHS and bad_epochs >= PATIENCE:
                print(f"Early stopping (no improvement for {PATIENCE} epochs).")
                break

    # ------------------- best model: threshold + report -------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, va_p, va_y, va_i = run_epoch(model, val_dl, criterion, device)

    thrs = np.linspace(0.05, 0.95, 91)
    scores = [balanced_accuracy_score(va_y, (va_p >= t).astype(int)) for t in thrs]
    best_thr = float(thrs[int(np.argmax(scores))])

    print(f"\nBest epoch: {best_epoch}")
    for name, t in [("thr=0.50", 0.5), (f"thr={best_thr:.2f} (tuned)", best_thr)]:
        m = metrics(va_p, va_y, t)
        cm = confusion_matrix(va_y, (va_p >= t).astype(int), labels=[0, 1])
        print(f"\n[{name}] acc {m['acc']:.3f}  bal_acc {m['bal_acc']:.3f}  "
              f"f1_uniform {m['f1_uniform']:.3f}  auc {m['auc']:.3f}")
        print("Confusion matrix (rows=true, cols=pred) [no_uniform, uniform]:")
        print(cm)

    pred = (va_p >= best_thr).astype(int)
    wrong = va_df.iloc[va_i[pred != va_y]].copy()
    wrong["prob_uniform"] = va_p[pred != va_y]
    wrong.to_csv(f"{OUT_DIR}/val_mistakes.csv", index=False)
    print(f"\n{len(wrong)} val mistakes saved to val_mistakes.csv — review them for label errors.")

    meta = {
        "model_name": MODEL_NAME, "classes": CLASSES, "img_size": IMG_SIZE,
        "pad_color": PAD_COLOR, "mean": MEAN, "std": STD,
        "threshold": best_thr, "best_epoch": best_epoch, "val_bal_acc": best_score,
        "crop_padding_note": "expand detector box by 10% per side before classifying",
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
    with open(f"{OUT_DIR}/uniform_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    try:
        model.eval().cpu()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(model, dummy, f"{OUT_DIR}/uniform_classifier.onnx",
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        print("ONNX exported.")
    except Exception as e:
        print("ONNX export skipped:", e)

    print(f"\nDone. Weights + meta in {OUT_DIR}")


# ------------------------- inference helper -------------------------
def load_classifier(ckpt_path=f"{OUT_DIR}/uniform_classifier.pt", device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    m = build_model(ckpt["meta"]["model_name"])
    m.load_state_dict(ckpt["state_dict"])
    return m.eval().to(device), ckpt["meta"]


@torch.no_grad()
def classify_crops(model, meta, pil_crops, device="cpu"):
    """pil_crops: list of PIL RGB person crops (already padded 10% like training)."""
    tf = transforms.Compose([Letterbox(meta["img_size"], tuple(meta["pad_color"])),
                             transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])])
    x = torch.stack([tf(c) for c in pil_crops]).to(device)
    p = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
    return [("uniform" if pi >= meta["threshold"] else "no_uniform", float(pi)) for pi in p]


if __name__ == "__main__":
    main()
