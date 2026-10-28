"""
3-class person-crop classifier:  uniform / no_uniform / not_a_person   (Kaggle-ready)
===================================================================================

Data on Kaggle
  Zip your three folders and upload the zip as a Kaggle dataset ("Create Dataset").
  Kaggle extracts it automatically. Any nesting is fine — the folders are found by name:
      uniform/         no_uniform/  (or "no uniform")      not_a_person/  (or "not a person")

What it does
  1. Collects images and splits train/val (default 80/20), stratified by class.
     Near-duplicate crops (consecutive frames) are grouped so they never land on both
     sides of the split -> honest validation numbers. If filenames contain the
     location/video name, set GROUP_REGEX to split by location instead (even better).
  2. Letterboxes every crop to 224x224 (keeps proportions, gray padding).
  3. Fine-tunes ImageNet-pretrained EfficientNet-B0 (or MobileNetV3-Large):
     a few epochs with the backbone frozen, then full fine-tuning.
     Class weights compensate for the smaller not_a_person class.
  4. Keeps the epoch with the best val balanced accuracy (mean per-class recall) and saves:
        /kaggle/working/uniform_classifier.pt         (weights + preprocessing meta)
        /kaggle/working/uniform_classifier_meta.json
        /kaggle/working/uniform_classifier.onnx       (if export succeeds)
        /kaggle/working/split.csv                     (which file went where)
        /kaggle/working/val_mistakes.csv              (misclassified val crops + probabilities)

Decision rule at inference: argmax over the 3 softmax probabilities.

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
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ============================== CONFIG ==============================
DATA_ROOT = "/kaggle/input"          # e.g. "/kaggle/input/uniform-crops"
OUT_DIR = "/kaggle/working"

MODEL_NAME = "efficientnet_b0"       # or "mobilenet_v3_large" (faster, slightly less accurate)
IMG_SIZE = 224
PAD_COLOR = (114, 114, 114)          # letterbox padding color — use the SAME at inference

# Class order defines the output indices. Folder names are matched ignoring case,
# spaces, underscores and hyphens ("not a person" == "not_a_person").
CLASSES = ["no_uniform", "uniform", "not_a_person"]

VAL_FRACTION = 0.2
SEED = 42

# Split grouping (prevents near-identical crops leaking between train and val)
GROUP_REGEX = None                   # e.g. r"^(loc\d+)_" if files look like "loc07_frame123_p2.jpg"
VAL_GROUPS = []                      # optional: groups (from GROUP_REGEX) to force into val, e.g. ["loc03","loc11"]
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ====================================================================

N_CLS = len(CLASSES)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_name(s):
    return re.sub(r"[\s_\-]+", "", s.lower())


def find_class_dir(root, wanted):
    """Find a folder whose normalized name matches (skips macOS zip junk)."""
    target = norm_name(wanted)
    for dirpath, dirnames, _ in os.walk(root):
        if "__MACOSX" in dirpath:
            continue
        for d in dirnames:
            if norm_name(d) == target:
                return Path(dirpath) / d
    raise FileNotFoundError(f"Could not find a '{wanted}' folder under {root}")


def collect_images():
    rows = []
    for label, cls in enumerate(CLASSES):
        cdir = find_class_dir(DATA_ROOT, cls)
        files = sorted(p for p in cdir.rglob("*")
                       if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"))
        print(f"{cls:>13}: {len(files):6d} images  ({cdir})")
        if not files:
            raise RuntimeError(f"No images in {cdir}")
        rows += [{"path": str(p), "label": label} for p in files]
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
        for label, cls in enumerate(CLASSES):
            g = groups[df["label"].values == label]
            print(f"  {cls:>13}: {len(g)} images -> {len(set(g))} unique groups")
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
    """Resize longest side to `size`, pad to square — keeps proportions."""
    def __init__(self, size, fill=PAD_COLOR):
        self.size, self.fill = size, tuple(fill)

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
def build_model(name, n_cls=N_CLS, pretrained=True):
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    elif name == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
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
        probs.append(torch.softmax(logits.float(), 1).detach().cpu())
        labels.append(y.cpu())
        idxs.append(i)
    n = len(loader.dataset)
    return (total_loss / n, torch.cat(probs).numpy(),
            torch.cat(labels).numpy(), torch.cat(idxs).numpy())


def metrics(probs, labels):
    pred = probs.argmax(1)
    recalls = {}
    for c, name in enumerate(CLASSES):
        m = labels == c
        recalls[name] = (pred[m] == c).mean() if m.any() else float("nan")
    return {
        "acc": (pred == labels).mean(),
        "bal_acc": balanced_accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
        "recall": recalls,
    }


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

    # class weights compensate for imbalance (e.g. fewer not_a_person crops)
    counts = np.bincount(tr_df["label"], minlength=N_CLS)
    weights = torch.tensor(counts.sum() / (N_CLS * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    print("Class weights:", dict(zip(CLASSES, weights.cpu().numpy().round(3))))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    model = build_model(MODEL_NAME).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    ckpt_path = f"{OUT_DIR}/uniform_classifier.pt"
    short = {"no_uniform": "noU", "uniform": "U", "not_a_person": "bg"}

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
        rec = " ".join(f"{short.get(k, k)}={v:.2f}" for k, v in vm["recall"].items())
        print(f"Ep {epoch:2d} | train loss {tr_loss:.4f} acc {tm['acc']:.3f} | "
              f"val loss {va_loss:.4f} acc {vm['acc']:.3f} bal_acc {vm['bal_acc']:.3f} "
              f"mF1 {vm['macro_f1']:.3f} | recall {rec} | {time.time() - t0:.0f}s")

        if vm["bal_acc"] > best_score:
            best_score, best_epoch, bad_epochs = float(vm["bal_acc"]), epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print("   ↳ saved best")
        else:
            bad_epochs += 1
            if epoch > WARMUP_EPOCHS and bad_epochs >= PATIENCE:
                print(f"Early stopping (no improvement for {PATIENCE} epochs).")
                break

    # ----------------------- best model: report -----------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, va_p, va_y, va_i = run_epoch(model, val_dl, criterion, device)
    pred = va_p.argmax(1)

    print(f"\nBest epoch: {best_epoch}   val balanced acc: {best_score:.3f}\n")
    print(classification_report(va_y, pred, labels=list(range(N_CLS)),
                                target_names=CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(va_y, pred, labels=list(range(N_CLS)))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    wrong_mask = pred != va_y
    wrong = va_df.iloc[va_i[wrong_mask]].copy()
    wrong["true"] = [CLASSES[t] for t in va_y[wrong_mask]]
    wrong["pred"] = [CLASSES[p] for p in pred[wrong_mask]]
    for c, name in enumerate(CLASSES):
        wrong[f"p_{name}"] = va_p[wrong_mask, c].round(3)
    wrong.drop(columns=["label"]).to_csv(f"{OUT_DIR}/val_mistakes.csv", index=False)
    print(f"\n{len(wrong)} val mistakes saved to val_mistakes.csv — review them for label errors.")

    meta = {
        "model_name": MODEL_NAME, "classes": CLASSES, "img_size": IMG_SIZE,
        "pad_color": list(PAD_COLOR), "mean": MEAN, "std": STD,
        "decision": "argmax", "best_epoch": best_epoch, "val_bal_acc": best_score,
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    m = build_model(meta["model_name"], len(meta["classes"]), pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval().to(device), meta


@torch.no_grad()
def classify_crops(model, meta, pil_crops, device="cpu"):
    """pil_crops: list of PIL RGB crops (padded 10% like training).
    Returns [(class_name, confidence, {class: prob}), ...]."""
    tf = transforms.Compose([Letterbox(meta["img_size"], meta["pad_color"]),
                             transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])])
    x = torch.stack([tf(c) for c in pil_crops]).to(device)
    p = torch.softmax(model(x), 1).cpu().numpy()
    out = []
    for row in p:
        k = int(row.argmax())
        out.append((meta["classes"][k], float(row[k]), dict(zip(meta["classes"], row.round(4).tolist()))))
    return out


if __name__ == "__main__":
    main()


!pip install -q onnx onnxscript onnxruntime

import torch, torch.nn as nn
from torchvision import models

ckpt = torch.load("/kaggle/working/uniform_classifier.pt", map_location="cpu", weights_only=False)
meta = ckpt["meta"]
n = len(meta["classes"])

if meta["model_name"] == "efficientnet_b0":
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
else:
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n)
model.load_state_dict(ckpt["state_dict"])
model.eval()

dummy = torch.randn(1, 3, meta["img_size"], meta["img_size"])
torch.onnx.export(
    model, dummy, "/kaggle/working/uniform_classifier.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)

# Check the ONNX model gives the same outputs as PyTorch, with a batch of 4
import numpy as np, onnxruntime as ort
x = torch.randn(4, 3, meta["img_size"], meta["img_size"])
with torch.no_grad():
    ref = model(x).numpy()
out = ort.InferenceSession("/kaggle/working/uniform_classifier.onnx").run(None, {"input": x.numpy()})[0]
print("ONNX exported. Max difference vs PyTorch:", np.abs(ref - out).max())
