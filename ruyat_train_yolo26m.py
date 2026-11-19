"""
Train the plant-2 model: person, truck, helmet / no helmet (head), boots / no boots
(non_safety_footwear). Fine-tunes your plant-1 YOLO26 model on plant-2 data, optionally mixed with the
plant-1 data (recommended - more data, and the model keeps working at both sites).

Input
  --plant2   CVAT export(s) in "Ultralytics YOLO Detection" or "... Detection Track" format
             (unzipped folder with data.yaml, images/, labels/). Track IDs are stripped.
  --plant1   (optional) the plant-1 YOLO dataset made by prepare_datasets.py
             (datasets/ppe/yolo). Its own train/val split is kept; only the classes
             in --classes are used, all other classes are dropped.
  --weights  model to start from: plant-1 best.pt (recommended: it already knows every class
             below, so each class keeps its learned weights) or e.g. yolo26m.pt
  --classes  classes to train (default: person truck helmet head boots non_safety_footwear);
             add harness here later once plant-2 frames are labelled for it

Every plant-2 frame must be labelled for EVERY class in --classes. An export whose frames
were annotated before a class was added (e.g. no truck boxes at all) teaches the model that
class is background - the script warns about this; fix those exports or leave them out.

Plant-2 frames are split train/val by VIDEO (file name before "frame<number>"), same as
prepare_datasets.py, so near-identical frames never leak into val.

Usage (6 GB GPU)
  pip install -U ultralytics
  python train_plant2.py --plant2 plant2_export/ --plant1 datasets/ppe/yolo \
      --weights runs/detect/runs/yolo26/yolo26m_960_s0/weights/best.pt --batch 2

  Resume after an interruption:
  python train_plant2.py --resume <run folder>/weights/last.pt

Output
  datasets/plant2/            merged dataset (data.yaml, val on both plants and plant 2 only)
  .../yolo26_plant2/<name>/weights/best.pt
  Per-class results on plant-2 val (the number that matters for this plant) and on all val.
"""

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

DEFAULT_CLASSES = ["person", "truck", "helmet", "head", "boots", "non_safety_footwear"]
CLASSES = list(DEFAULT_CLASSES)        # replaced by --classes in main()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- reading datasets

def read_names(data_yaml):
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def read_labels(txt, names):
    """Return [(new_class_idx, cx, cy, w, h)], dropping track IDs and unused classes."""
    out, dropped = [], Counter()
    if not txt or not txt.exists():
        return out, dropped
    for line in txt.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        name = names.get(int(float(parts[0])))
        if name not in CLASSES:
            dropped[name] += 1
            continue
        out.append((CLASSES.index(name), *map(float, parts[1:5])))   # parts[5] = track id
    return out, dropped


def load_cvat_export(folder):
    folder = Path(folder)
    ymls = list(folder.rglob("data.yaml"))
    if not ymls:
        raise SystemExit(f"No data.yaml in {folder} - is this an Ultralytics YOLO export?")
    names = read_names(ymls[0])
    root = ymls[0].parent
    label_index = {p.stem: p for p in (root / "labels").rglob("*.txt")} if (root / "labels").exists() \
        else {p.stem: p for p in root.rglob("*.txt") if p.parent.name != root.name}
    items, dropped = [], Counter()
    for img in sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in IMG_EXTS):
        labels, d = read_labels(label_index.get(img.stem), names)
        dropped.update(d)
        items.append((img, labels))
    if not items:
        raise SystemExit(f"No images under {root / 'images'} - export with 'Save images' ticked.")
    counts = Counter(l[0] for _, ls in items for l in ls)
    print(f"Plant 2 export {folder}: {len(items)} images, CVAT labels {list(names.values())}")
    print("  boxes: " + ", ".join(f"{c} {counts[i]}" for i, c in enumerate(CLASSES)))
    for i, c in enumerate(CLASSES):
        if c not in names.values():
            print(f"  WARNING: this export has no '{c}' label at all - its frames would teach the "
                  f"model that every {c} is background. Re-annotate them or leave this export out.")
        elif counts[i] == 0:
            print(f"  WARNING: no '{c}' boxes in this export. Fine only if no {c} is ever visible "
                  "in these frames; otherwise label them before training.")
    return items, dropped


def load_plant1(folder):
    folder = Path(folder)
    names = read_names(folder / "data.yaml")
    splits, dropped = {}, Counter()
    for split in ("train", "val"):
        items = []
        for img in sorted((folder / "images" / split).glob("*")):
            if img.suffix.lower() in IMG_EXTS:
                labels, d = read_labels(folder / "labels" / split / f"{img.stem}.txt", names)
                dropped.update(d)
                items.append((img, labels))
        splits[split] = items
    print(f"Plant 1 dataset {folder}: {len(splits['train'])} train, {len(splits['val'])} val")
    return splits, dropped


# ---------------------------------------------------------------- split by video

def group_of(name, pattern):
    stem = Path(name).stem
    m = re.match(pattern, stem, flags=re.IGNORECASE)
    return m.group(1) if m else (re.sub(r"[_-]*\d+$", "", stem) or stem)


def split_by_video(items, val_frac, pattern, seed, trials=5000):
    groups = defaultdict(list)
    for it in items:
        groups[group_of(it[0].name, pattern)].append(it)
    if len(groups) < 3:
        raise SystemExit(f"Only {len(groups)} video groups found in plant-2 file names - "
                         "check --group-regex.")
    gcls = {g: Counter(l[0] for _, ls in its for l in ls) for g, its in groups.items()}
    total = Counter()
    for c in gcls.values():
        total.update(c)
    rng, names = random.Random(seed), sorted(groups)
    best, best_score = None, float("inf")
    for _ in range(trials):
        rng.shuffle(names)
        val, n = set(), 0
        for g in names:
            if n >= val_frac * len(items):
                break
            val.add(g)
            n += len(groups[g])
        if len(val) == len(names):
            continue
        vc = Counter()
        for g in val:
            vc.update(gcls[g])
        score = abs(n / len(items) - val_frac) * 2
        for k in total:
            share = vc[k] / total[k]
            score += abs(share - val_frac) + (1.0 if share in (0, 1) else 0)
        if score < best_score:
            best, best_score = set(val), score
    train = [it for g in groups if g not in best for it in groups[g]]
    val = [it for g in best for it in groups[g]]
    print(f"Plant 2 split by video: {len(groups) - len(best)} groups train, {len(best)} groups val")
    return train, val


# ---------------------------------------------------------------- writing

def write_split(out, split, items, prefix, link):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img, labels in items:
        name = f"{prefix}{img.name}"                    # prefix avoids name clashes
        dst = out / "images" / split / name
        if link:
            try:
                dst.hardlink_to(img)
            except OSError:
                shutil.copy2(img, dst)
        else:
            shutil.copy2(img, dst)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))


def build_dataset(args):
    p2, dropped = [], Counter()
    for f in args.plant2:
        items, d = load_cvat_export(f)
        p2 += items
        dropped.update(d)
    names = Counter(img.name for img, _ in p2)
    dupes = [n for n, k in names.items() if k > 1]
    if dupes:
        raise SystemExit(f"{len(dupes)} image names appear in more than one plant-2 export "
                         f"(e.g. {dupes[:3]}) - one copy would overwrite the other. Rename the "
                         "frames in the newer export (or remove frames exported twice) first.")
    p2_train, p2_val = split_by_video(p2, args.val_frac, args.group_regex, args.seed)
    p1 = {"train": [], "val": []}
    if args.plant1:
        p1, d = load_plant1(args.plant1)
        dropped.update(d)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    write_split(out, "train", p2_train, "p2_", args.link)
    write_split(out, "train", p1["train"], "p1_", args.link)
    write_split(out, "val_plant2", p2_val, "p2_", args.link)
    write_split(out, "val", p2_val, "p2_", args.link)
    write_split(out, "val", p1["val"], "p1_", args.link)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    base = f"path: {out.resolve().as_posix()}\ntrain: images/train\n"
    (out / "data.yaml").write_text(base + f"val: images/val\nnames:\n{names}\n")
    (out / "data_plant2val.yaml").write_text(base + f"val: images/val_plant2\nnames:\n{names}\n")

    def count(items):
        return Counter(l[0] for _, ls in items for l in ls)
    c = {k: count(v) for k, v in [("p2 train", p2_train), ("p2 val", p2_val),
                                  ("p1 train", p1["train"]), ("p1 val", p1["val"])]}
    print(f"\n  {'class':22s}" + "".join(f"{k:>10s}" for k in c))
    for i, n in enumerate(CLASSES):
        print(f"  {n:22s}" + "".join(f"{c[k][i]:10d}" for k in c))
    print(f"  {'images':22s}" + "".join(f"{len(v):10d}" for v in
                                        (p2_train, p2_val, p1['train'], p1['val'])))
    if dropped:
        print(f"\n  Ignored labels of other classes: {dict(dropped)}")
    return out


# ---------------------------------------------------------------- train + evaluate

def per_class_report(metrics, title):
    print(f"\n{title}\n{'class':22s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for i, k in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{metrics.names[int(k)]:22s} {p:6.3f} {r:6.3f} {ap50:7.3f} {ap:9.3f}")
    print(f"{'ALL':22s} {metrics.box.mp:6.3f} {metrics.box.mr:6.3f} "
          f"{metrics.box.map50:7.3f} {metrics.box.map:9.3f}")


def evaluate(best, data_dir, args, name):
    from ultralytics import YOLO
    for yml, title in [("data_plant2val.yaml", "PLANT 2 validation (what matters for this plant)"),
                       ("data.yaml", "ALL validation (plant 2 + plant 1)")]:
        m = YOLO(best).val(data=str(data_dir / yml), imgsz=args.imgsz, batch=args.batch,
                           device=args.device, plots=True, project="runs/yolo26_plant2",
                           name=f"{name}_val_{yml.split('.')[0]}", exist_ok=True, verbose=False)
        per_class_report(m, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant2", nargs="+", help="unzipped CVAT Ultralytics YOLO export(s)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="classes to train, by name (order = class ids)")
    ap.add_argument("--plant1", default=None, help="plant-1 YOLO dataset (datasets/ppe/yolo)")
    ap.add_argument("--weights", default="yolo26m.pt", help="plant-1 best.pt, or yolo26m.pt")
    ap.add_argument("--out", default="datasets/plant2")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--group-regex", default=r"^(.+?)[_-]*frame[_-]*\d+$")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2, help="2 for YOLO26-m at 960 on a 6 GB GPU")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--link", action="store_true", help="hard-link images instead of copying")
    ap.add_argument("--resume", default=None, help="last.pt of an interrupted run")
    args = ap.parse_args()

    from ultralytics import YOLO
    CLASSES[:] = args.classes
    name = args.name or f"{Path(args.weights).stem}_ft_{args.imgsz}_s{args.seed}"

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        evaluate(model.trainer.best, Path(args.out), args, name)
        return
    if not args.plant2:
        ap.error("--plant2 is required (or use --resume)")

    data_dir = build_dataset(args)
    model = YOLO(args.weights)
    model.train(
        data=str(data_dir / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs,
        patience=args.patience, batch=args.batch, device=args.device, workers=args.workers,
        seed=args.seed, project="runs/yolo26_plant2", name=name, exist_ok=True, amp=True,
        mixup=0.1, close_mosaic=15, plots=True,
    )
    best = model.trainer.best
    print(f"\nBest weights: {best}")
    evaluate(best, data_dir, args, name)


if __name__ == "__main__":
    main()
