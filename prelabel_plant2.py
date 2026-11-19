"""
Pre-label frames for the new plant with your trained YOLO26 model, for import into CVAT.

Keeps only the classes this plant needs (helmet / no helmet, boots / no boots, plus person,
which the rules engine needs to tie PPE to a worker, and truck, for the outdoor truck gate and
the harness on-truck check). Use a detector that has a 'truck' class (e.g. the plant-1 model
or the new plant-2 model). Class names stay the same as plant 1
(`head` = no helmet, `non_safety_footwear` = no boots), so the two plants' data can be
merged into one training set later.

Memory-safe for small GPUs (6 GB): one image at a time, FP16 on GPU, cache cleared at the
end. Use --device cpu if the GPU is busy.

Usage
  pip install -U ultralytics
  python prelabel_plant2.py --weights best.pt --images plant2_frames/ --out plant2_prelabels.zip

Then in CVAT:
  1. Create a project; under Labels -> Raw, paste the contents of cvat_labels.json
     (written next to the zip).
  2. Create a task in it with the same images (same file names).
  3. Actions -> Upload annotations -> "YOLO 1.1" -> plant2_prelabels.zip
"""

import argparse
import gc
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torchvision.ops import nms
from tqdm import tqdm

# ---------------------------------------------------------------- configuration

# trained-model class name -> label in the new CVAT project (others are ignored)
KEEP = {
    "person": "person",
    "truck": "truck",
    "helmet": "helmet",
    "head": "head",                                # no helmet
    "boots": "boots",
    "non_safety_footwear": "non_safety_footwear",  # no boots
}
# Labels for the CVAT project, in this order. "harness" is included now so it's ready when
# you have harness data (the model gives no harness boxes; just leave it unused for now).
CVAT_LABELS = ["person", "truck", "helmet", "head", "boots", "non_safety_footwear", "harness"]

# Confidence thresholds. A new plant looks different from the training data, so the model
# is less confident there; these are a little lower than you'd use in production.
CONF = {"person": 0.4, "truck": 0.4, "helmet": 0.35, "head": 0.35, "boots": 0.3, "non_safety_footwear": 0.3}

# Classes that can't be on the same object: if their boxes overlap above this IoU,
# keep only the more confident one (helmet vs head on one head, boots vs shoes on one foot).
EXCLUSIVE = [{"helmet", "head"}, {"boots", "non_safety_footwear"}]
EXCLUSIVE_IOU = 0.4

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------- pipeline


def resolve_conflicts(boxes):
    """boxes: list of [x1, y1, x2, y2, conf, label]. Drop contradicting overlaps."""
    for group in EXCLUSIVE:
        idx = [i for i, b in enumerate(boxes) if b[5] in group]
        if len(idx) < 2:
            continue
        t = torch.tensor([boxes[i][:5] for i in idx], dtype=torch.float32)
        keep = set(nms(t[:, :4], t[:, 4], EXCLUSIVE_IOU).tolist())
        drop = {idx[k] for k in range(len(idx)) if k not in keep}
        boxes = [b for i, b in enumerate(boxes) if i not in drop]
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained YOLO26 best.pt")
    ap.add_argument("--images", required=True, help="folder of frames (subfolders included)")
    ap.add_argument("--out", default="plant2_prelabels.zip")
    ap.add_argument("--imgsz", type=int, default=960, help="same as training")
    ap.add_argument("--device", default="0", help="GPU index or 'cpu'")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)

    images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    stems = Counter(p.stem for p in images)
    dupes = [s for s, n in stems.items() if n > 1]
    if dupes:
        raise SystemExit(f"Duplicate file names (CVAT matches by name), e.g. {dupes[:3]}. "
                         "Rename them so every image name is unique.")

    model = YOLO(args.weights)
    names = model.names
    missing = [k for k in KEEP if k not in names.values()]
    if missing:
        raise SystemExit(f"Model has no class {missing}. Its classes: {list(names.values())}")
    keep_ids = [i for i, n in names.items() if n in KEEP]
    min_conf = min(CONF.values())
    on_gpu = args.device != "cpu" and torch.cuda.is_available()
    if not on_gpu and args.device != "cpu":
        print("No CUDA GPU visible - running on CPU.")
    device = args.device if on_gpu else "cpu"
    print(f"{len(images)} images, model {Path(args.weights).name}, imgsz {args.imgsz}, "
          f"device {device}")

    counts, empty = Counter(), 0
    out_zip = Path(args.out)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CVAT_LABELS) + "\n")
        z.writestr("obj.data", f"classes = {len(CVAT_LABELS)}\ntrain = data/train.txt\n"
                               "names = data/obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{p.name}\n" for p in images))

        for p in tqdm(images, unit="img", dynamic_ncols=True, desc="Pre-labelling"):
            kw = {"half": True} if on_gpu else {}
            try:
                r = model.predict(str(p), conf=min_conf, imgsz=args.imgsz, device=device,
                                  classes=keep_ids, verbose=False, **kw)[0]
            except torch.OutOfMemoryError:
                raise SystemExit("GPU out of memory. Close other GPU jobs (check nvidia-smi), "
                                 "or rerun with --imgsz 640 or --device cpu.")
            h, w = r.orig_shape
            boxes = []
            for (x1, y1, x2, y2), c, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                                r.boxes.cls.tolist()):
                label = KEEP[names[int(cls)]]
                if c >= CONF[label]:
                    boxes.append([x1, y1, x2, y2, c, label])
            boxes = resolve_conflicts(boxes)

            lines = []
            for x1, y1, x2, y2, _, label in boxes:
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                lines.append(f"{CVAT_LABELS.index(label)} {(x1 + x2) / 2 / w:.6f} "
                             f"{(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                counts[label] += 1
            empty += not lines
            z.writestr(f"obj_train_data/{p.stem}.txt", "\n".join(lines) + "\n")

    del model
    gc.collect()
    if on_gpu:
        torch.cuda.empty_cache()

    labels_json = out_zip.with_name("cvat_labels.json")
    labels_json.write_text(json.dumps(
        [{"name": n, "type": "rectangle", "attributes": []} for n in CVAT_LABELS], indent=1))

    print(f"\nWrote {out_zip}  ({len(images)} images, {empty} with no boxes)")
    for n in CVAT_LABELS:
        print(f"  {n:22s} {counts[n]}")
    print(f"CVAT labels (paste into Project -> Labels -> Raw): {labels_json}")


if __name__ == "__main__":
    main()
