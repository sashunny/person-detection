"""
Build a YOLO dataset and an RF-DETR (COCO) dataset from CVAT "COCO 1.0" export(s),
with the SAME train/val split for both models.

Why only the COCO export: building both datasets from one source guarantees the two models
train and validate on identical labels and identical images. (The YOLO 1.1 export isn't needed.)

The split is done by VIDEO, not by frame: frames are grouped by the part of the file name
before "frame<number>" (e.g. B_MILL_ROLLER_a_frame000083.jpg -> group "B_MILL_ROLLER_a"),
and whole groups go to train or val. Random frame splits leak near-identical frames into
val and give falsely high scores. Among many random group splits, the one whose per-class
val share is closest to --val-frac is chosen, so rare classes also appear in val.

Usage
  python prepare_datasets.py --coco export_task1/ export_task2/ --images frames/ --out datasets/ppe

  --coco    one or more extracted CVAT COCO exports (folder with annotations/*.json), or .json files
  --images  folder(s) to look for the image files in, if the export doesn't include images

Output
  datasets/ppe/yolo/  images/{train,val}, labels/{train,val}, data.yaml   -> train_yolo26.py
  datasets/ppe/coco/  train/, valid/ (images + _annotations.coco.json)      -> train_rfdetr.py
  datasets/ppe/split.json   which group / image went where
"""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_jsons(paths):
    out = []
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".json":
            out.append(p)
        elif p.is_dir():
            found = sorted(p.rglob("*.json"))
            if not found:
                raise SystemExit(f"No .json annotation file found under {p}")
            out += found
        else:
            raise SystemExit(f"Not found: {p}")
    return out


def index_images(dirs):
    index = {}
    for d in dirs:
        for f in Path(d).rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                index.setdefault(f.name, f)
    return index


def load_coco(json_files, search_dirs):
    """Merge COCO files. Returns (class_names, images) where images =
    {file_name: {"path", "w", "h", "anns": [(class_idx, x, y, w, h)]}}."""
    class_names, images = [], {}
    for jf in json_files:
        data = json.loads(jf.read_text())
        cats = sorted(data["categories"], key=lambda c: c["id"])
        cat_to_idx = {}
        for c in cats:
            if c["name"] not in class_names:
                class_names.append(c["name"])
            cat_to_idx[c["id"]] = class_names.index(c["name"])

        img_index = index_images([jf.parent.parent, jf.parent, *search_dirs])
        id_to_name = {}
        for im in data["images"]:
            name = Path(im["file_name"]).name
            if name in images:
                raise SystemExit(f"Image name {name} appears in more than one export; names "
                                 "must be unique across tasks.")
            if name not in img_index:
                raise SystemExit(f"Image file {name} (from {jf}) not found. Export with images, "
                                 "or pass the frames folder with --images.")
            images[name] = {"path": img_index[name], "w": im["width"], "h": im["height"],
                            "anns": []}
            id_to_name[im["id"]] = name

        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            images[id_to_name[a["image_id"]]]["anns"].append(
                (cat_to_idx[a["category_id"]], x, y, w, h))
        n_ann = sum(len(images[n]["anns"]) for n in id_to_name.values())
        print(f"Loaded {jf}: {len(data['images'])} images, {n_ann} boxes")
    return class_names, images


def group_of(name, pattern):
    stem = Path(name).stem
    m = re.match(pattern, stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return re.sub(r"[_-]*\d+$", "", stem) or stem   # fallback: strip trailing number


def choose_split(images, groups, n_classes, val_frac, trials, seed):
    """Pick the group split whose per-class val share is closest to val_frac."""
    group_imgs = defaultdict(list)
    for name, g in groups.items():
        group_imgs[g].append(name)
    group_cls = {g: Counter(a[0] for n in names for a in images[n]["anns"])
                 for g, names in group_imgs.items()}
    total_cls = Counter()
    for c in group_cls.values():
        total_cls.update(c)
    n_imgs = len(images)
    # classes found in >= 2 groups can (and should) appear in both train and val
    splittable = [k for k in range(n_classes)
                  if sum(1 for c in group_cls.values() if c[k]) >= 2]

    rng = random.Random(seed)
    gnames = sorted(group_imgs)
    best, best_score = None, float("inf")
    for _ in range(trials):
        rng.shuffle(gnames)
        val, n_val = set(), 0
        for g in gnames:
            if n_val >= val_frac * n_imgs:
                break
            val.add(g)
            n_val += len(group_imgs[g])
        if len(val) == len(gnames):          # never put everything in val
            continue
        val_cls = Counter()
        for g in val:
            val_cls.update(group_cls[g])
        score = abs(n_val / n_imgs - val_frac) * 2
        for k in splittable:
            share = val_cls[k] / total_cls[k]
            score += abs(share - val_frac)
            if share == 0 or share == 1:
                score += 1.0                 # class missing from val or from train
        if score < best_score:
            best, best_score = set(val), score
    return best, group_imgs


def write_yolo(out, split_names, images, class_names, link):
    for split, names in split_names.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for n in names:
            im = images[n]
            place(im["path"], out / "images" / split / n, link)
            lines = [f"{c} {(x + w / 2) / im['w']:.6f} {(y + h / 2) / im['h']:.6f} "
                     f"{w / im['w']:.6f} {h / im['h']:.6f}" for c, x, y, w, h in im["anns"]]
            (out / "labels" / split / f"{Path(n).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val",
            "names:"] + [f"  {i}: {n}" for i, n in enumerate(class_names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")


def write_coco(out, split_names, images, class_names, link):
    folder = {"train": "train", "val": "valid"}
    for split, names in split_names.items():
        d = out / folder[split]
        d.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [],
                "categories": [{"id": i + 1, "name": n, "supercategory": "none"}
                               for i, n in enumerate(class_names)]}
        for img_id, n in enumerate(names, 1):
            im = images[n]
            place(im["path"], d / n, link)
            coco["images"].append({"id": img_id, "file_name": n, "width": im["w"],
                                   "height": im["h"]})
            for c, x, y, w, h in im["anns"]:
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": img_id,
                    "category_id": c + 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        (d / "_annotations.coco.json").write_text(json.dumps(coco))


def place(src, dst, link):
    if dst.exists():
        dst.unlink()
    if link:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", nargs="+", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--out", default="datasets/ppe")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--group-regex", default=r"^(.+?)[_-]*frame[_-]*\d+$",
                    help="regex on the file stem; group(1) = video/group id")
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true",
                    help="hard-link images instead of copying (saves disk; same drive only)")
    args = ap.parse_args()

    class_names, images = load_coco(find_jsons(args.coco), args.images)
    groups = {n: group_of(n, args.group_regex) for n in images}
    n_groups = len(set(groups.values()))
    print(f"\n{len(images)} images in {n_groups} video groups, {len(class_names)} classes")
    for g, c in Counter(groups.values()).most_common(8):
        print(f"  {g}: {c} frames")
    if n_groups < 5:
        raise SystemExit("Fewer than 5 groups found - check --group-regex against your file names.")

    val_groups, group_imgs = choose_split(images, groups, len(class_names), args.val_frac,
                                          args.trials, args.seed)
    split_names = {"train": sorted(n for n in images if groups[n] not in val_groups),
                   "val": sorted(n for n in images if groups[n] in val_groups)}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    write_yolo(out / "yolo", split_names, images, class_names, args.link)
    write_coco(out / "coco", split_names, images, class_names, args.link)
    (out / "split.json").write_text(json.dumps(
        {"val_groups": sorted(val_groups),
         "train_groups": sorted(set(group_imgs) - val_groups),
         "class_names": class_names, **split_names}, indent=1))

    # report
    cnt = {s: Counter(a[0] for n in ns for a in images[n]["anns"]) for s, ns in split_names.items()}
    print(f"\nSplit: train {len(split_names['train'])} images "
          f"({len(set(group_imgs) - val_groups)} groups), val {len(split_names['val'])} images "
          f"({len(val_groups)} groups)\n")
    print(f"  {'class':24s} {'train':>7s} {'val':>6s}  val%")
    for i, c in enumerate(class_names):
        t, v = cnt["train"][i], cnt["val"][i]
        flag = ""
        if t + v == 0:
            flag = "  (no labels)"
        elif v == 0:
            flag = "  <- NOT IN VAL (only in one video group?)"
        elif t + v < 200:
            flag = "  <- few instances"
        pct = f"{100 * v / (t + v):4.0f}%" if t + v else "   -"
        print(f"  {c:24s} {t:7d} {v:6d}  {pct}{flag}")
    print(f"\nWrote {out / 'yolo'} and {out / 'coco'}")


if __name__ == "__main__":
    main()
