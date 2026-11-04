"""
Pre-label images with the existing single-use-case models, for import into CVAT.

What it does
  1. Runs every old DETECTION model over a folder of images.
  2. Maps each model's class names to the new unified taxonomy.
  3. Removes duplicates across models (per-class NMS, e.g. "person" from several models).
  4. Crops every detected person and runs the UNIFORM CLASSIFIER (EfficientNet-B2) on it:
       uniform      -> `uniform` box (drawn on the person box)
       no_uniform   -> `non_compliant_clothing` box (drawn on the person box)
       not_a_person -> nothing (optionally removes the person box too)
     People already wearing a vest are skipped (vest rule: label only `vest`).
  5. Writes a CVAT "YOLO 1.1" annotation zip whose file names match your images.

Usage
  pip install ultralytics timm
  python prelabel.py --images frames/ --out prelabels.zip

Then in CVAT: create a task with the same images, then
  Actions -> Upload annotations -> format "YOLO 1.1" -> prelabels.zip

Before running, edit the CONFIGURATION section:
  * MODELS: your detector files and their class-name mapping. Check a detector's class names with
        python -c "from ultralytics import YOLO; print(YOLO('x.pt').names)"
  * UNIFORM_CLASSIFIER: classifier file, class order, input size and normalisation.
    These MUST match how the classifier was trained (check your training script).
  * Check the classifier on its own first:
        python prelabel.py --check-classifier some_person_crop.jpg
"""

import argparse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

# ================================================================ CONFIGURATION

CLASSES = [
    "person", "truck", "helmet", "vest", "uniform", "non_compliant_clothing",
    "harness", "boots", "mask", "mask_improper", "face", "head",
    "non_safety_footwear",
]
CID = {c: i for i, c in enumerate(CLASSES)}

# Detection models (Ultralytics YOLO .pt): model file -> {old class name: new class name}.
# The uniform model is NOT here: it is a classifier, configured in UNIFORM_CLASSIFIER below.
# One of these models must output `person`, because the classifier runs on person boxes.
MODELS = {
    "../pretrained_models/person_model.pt": {"person": "person"},
    # "../pretrained_models/truck.pt":      {"truck": "truck"},          # outdoor zones only
    "../pretrained_models/hardhat_v2.pt":   {"hardhat": "helmet", "no_hardhat": "head"},
    "../pretrained_models/vest.pt":         {"vest": "vest", "person": "person",
                                             "hat": "helmet", "nohat": "head"},
    "../pretrained_models/harness.pt":      {"Harness": "harness"},
    "../pretrained_models/Boots_model.pt":  {"boot": "boots", "no boot": "non_safety_footwear"},
    "../pretrained_models/mask.pt":         {"Improper Mask": "mask_improper", "Mask": "mask",
                                             "No Mask": "face"},
}

UNIFORM_CLASSIFIER = {
    "weights": "pretrained_models/uniform_classifier.pt",   # set to None to skip the classifier

    # Optional metadata saved with the model. If found, class names, input size, mean/std and
    # arch are read from it and OVERRIDE the values below (the script prints what it used).
    "meta": "pretrained_models/uniform_classifier_meta.json",

    # How the .pt was saved:
    #   "auto"        - try everything below (recommended)
    #   "full"        - torch.save(model)              (needs the model's class importable)
    #   "timm"        - state_dict of timm.create_model("efficientnet_b2", num_classes=N)
    #   "torchvision" - state_dict of torchvision.models.efficientnet_b2(num_classes=N)
    "loader": "auto",
    "arch": "efficientnet_b0",

    # Output index order of the classifier (from the meta file). MUST match training.
    "class_names": ["no_uniform", "uniform", "not_a_person"],

    # Classifier class -> new class (None = add no clothing box).
    # Matching ignores case, spaces, hyphens and underscores ("No Uniform" == "no_uniform").
    "map": {"uniform": "uniform", "no_uniform": "non_compliant_clothing", "not_a_person": None},

    # Preprocessing - copy these from your training/inference transforms.
    "input_size": (224, 224),             # (height, width)
    "resize": "letterbox",                # "letterbox" = keep aspect, pad with pad_color
                                          # "squash" = resize to input_size ignoring aspect
    "pad_color": (114, 114, 114),
    "mean": (0.485, 0.456, 0.406),        # ImageNet defaults
    "std": (0.229, 0.224, 0.225),
    "crop_pad": 0.10,                     # expand the person box by 10% per side before
                                          # cropping (meta: crop_padding_note)

    "min_conf": 0.5,          # below this, add no clothing box (annotator decides)
    "min_person_px": 40,      # skip people shorter than this (too small to judge clothing)
    "skip_if_vest": True,     # vest rule: no clothing label on vest wearers
    "drop_person_if_not_person": False,  # True = also delete the person box when the
                                         # classifier says "not_a_person" (confident only)
    "batch_size": 64,
}

# Low thresholds favour recall: deleting a wrong box is faster than drawing a missing one.
DEFAULT_CONF = 0.35
CONF = {"harness": 0.25, "boots": 0.25, "mask": 0.25, "mask_improper": 0.25}

NMS_IOU = 0.6           # merge same-class boxes from different models above this IoU
VEST_IN_BOX = 0.6       # a vest "belongs" to a box if this share of the vest lies inside it

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ================================================================ detection


def conf_for(cls_name: str) -> float:
    return CONF.get(cls_name, DEFAULT_CONF)


def run_detectors(images, imgsz, device):
    """Return {image name: list of [x1, y1, x2, y2, conf, class_id]} plus image sizes."""
    from ultralytics import YOLO

    dets = defaultdict(list)
    sizes = {}
    min_conf = min([DEFAULT_CONF, *CONF.values()])

    for weights, mapping in MODELS.items():
        model = YOLO(weights)
        names = model.names
        unknown = set(mapping) - set(names.values())
        if unknown:
            print(f"WARNING {weights}: classes not in model, check MODELS: {sorted(unknown)}")
        print(f"Running {weights} ...")

        for r in model.predict(source=[str(p) for p in images], stream=True, conf=min_conf,
                               imgsz=imgsz, device=device, half=device != "cpu", verbose=False):
            name = Path(r.path).name
            sizes[name] = r.orig_shape  # (h, w)
            for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                      r.boxes.cls.tolist()):
                new = mapping.get(names[int(cls)])
                if new and conf >= conf_for(new):
                    dets[name].append([*box, conf, CID[new]])
    return dets, sizes


def share_inside(outer, inner):
    """For box tensors outer (N,4) and inner (M,4): share of each inner box inside each outer box."""
    x1 = torch.max(outer[:, None, 0], inner[None, :, 0])
    y1 = torch.max(outer[:, None, 1], inner[None, :, 1])
    x2 = torch.min(outer[:, None, 2], inner[None, :, 2])
    y2 = torch.min(outer[:, None, 3], inner[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = ((inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])).clamp(min=1e-6)
    return inter / area[None, :]  # (N, M)


def dedupe(boxes):
    """Per-class NMS across models."""
    if not boxes:
        return []
    t = torch.tensor(boxes, dtype=torch.float32)
    kept = []
    for c in t[:, 5].unique():
        sub = t[t[:, 5] == c]
        kept.append(sub[nms(sub[:, :4], sub[:, 4], NMS_IOU)])
    return torch.cat(kept).tolist()


def has_vest(boxes, person_boxes):
    """Bool per person: does a vest box lie (mostly) inside it?"""
    vests = [b[:4] for b in boxes if int(b[5]) == CID["vest"]]
    if not vests or not person_boxes:
        return [False] * len(person_boxes)
    s = share_inside(torch.tensor(person_boxes, dtype=torch.float32),
                     torch.tensor(vests, dtype=torch.float32))
    return (s > VEST_IN_BOX).any(dim=1).tolist()


def apply_vest_rule(boxes):
    """Drop any uniform / non_compliant_clothing box that contains a vest."""
    clothing = {CID["uniform"], CID["non_compliant_clothing"]}
    cloth_idx = [i for i, b in enumerate(boxes) if int(b[5]) in clothing]
    if not cloth_idx:
        return boxes
    flags = has_vest(boxes, [boxes[i][:4] for i in cloth_idx])
    drop = {i for i, f in zip(cloth_idx, flags) if f}
    return [b for i, b in enumerate(boxes) if i not in drop]

# ================================================================ uniform classifier


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _flatten(d, out=None):
    """Flatten nested dicts into one {key: value} dict (inner keys win only if new)."""
    out = {} if out is None else out
    for k, v in d.items():
        if isinstance(v, dict) and _norm(k) not in ("idxtoclass", "classtoidx", "classes",
                                                     "classnames", "labels", "names"):
            _flatten(v, out)
        else:
            out.setdefault(_norm(k), v)
    return out


def _class_names_from(v):
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v
    if isinstance(v, dict) and v:
        keys, vals = list(v.keys()), list(v.values())
        if all(str(k).lstrip("-").isdigit() for k in keys):       # {"0": "no_uniform", ...}
            return [v[k] for k in sorted(keys, key=lambda k: int(k))]
        if all(isinstance(x, int) for x in vals):                  # {"no_uniform": 0, ...}
            return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    return None


def apply_meta(cfg):
    """Read the classifier's meta JSON (if any) and override cfg values found in it."""
    import json
    path = cfg.get("meta")
    if not path or not Path(path).exists():
        if path:
            print(f"Note: meta file {path} not found; using UNIFORM_CLASSIFIER values.")
        return cfg
    flat = _flatten(json.loads(Path(path).read_text()))
    cfg, used, consumed = dict(cfg), [], set()

    for key in ("idxtoclass", "classnames", "classes", "labels", "names", "classtoidx"):
        names = _class_names_from(flat.get(key))
        if names:
            cfg["class_names"], _ = names, used.append(f"class_names={names}")
            consumed.add(key)
            break
    else:
        print(f"WARNING: no class names found in {path}; using {cfg['class_names']} - verify!")

    for key in ("inputsize", "imgsize", "imagesize", "imgsz", "resolution", "size",
                "inputshape", "inputdim"):
        v = flat.get(key)
        if isinstance(v, int):
            cfg["input_size"] = (v, v)
        elif isinstance(v, list) and len(v) in (2, 3) and all(isinstance(x, int) for x in v):
            hw = [x for x in v if x != 3] if len(v) == 3 else v       # drop channel dim
            cfg["input_size"] = (hw[0], hw[-1])
        else:
            continue
        used.append(f"input_size={cfg['input_size']}")
        consumed.add(key)
        break

    for key in ("mean", "std"):
        v = flat.get(key) or flat.get(f"normalize{key}") or flat.get(f"norm{key}")
        if isinstance(v, list) and len(v) == 3:
            cfg[key] = tuple(float(x) for x in v)
            used.append(f"{key}={cfg[key]}")

    pad = flat.get("padcolor")
    if isinstance(pad, list) and len(pad) == 3:
        cfg["pad_color"], cfg["resize"] = tuple(int(x) for x in pad), "letterbox"
        used.append(f"letterbox, pad_color={cfg['pad_color']}")
        consumed.add("padcolor")

    decision = flat.get("decision")
    if decision is not None:
        consumed.add("decision")
        if str(decision).lower() != "argmax":
            print(f"WARNING: meta decision={decision!r}; this script uses argmax + min_conf.")

    for key in ("arch", "architecture", "modelname", "backbone", "model"):
        v = flat.get(key)
        if isinstance(v, str):
            cfg["arch"] = v.replace("-", "_").lower()
            used.append(f"arch={cfg['arch']}")
            consumed.add(key)
            break

    print(f"From {path}: " + ("; ".join(used) if used else "nothing usable"))
    consumed |= {"mean", "std", "normalizemean", "normalizestd", "normmean", "normstd"}
    other = sorted(k for k in flat if k not in consumed)
    if other:
        print(f"  other keys in meta (check if preprocessing-related): {other}")

    norm_map = {_norm(k) for k in cfg["map"]}
    missing = [c for c in cfg["class_names"] if _norm(c) not in norm_map]
    if missing:
        print(f"WARNING: classifier classes {missing} are not in UNIFORM_CLASSIFIER['map'] - "
              "they will add no clothing box. Add them to 'map'.")
    return cfg


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "ema"):
            if key in obj:
                return _unwrap_state_dict(obj[key])
        return {k.removeprefix("module.").removeprefix("model."): v for k, v in obj.items()
                if torch.is_tensor(v)}
    return obj


def load_classifier(cfg, device):
    n = len(cfg["class_names"])
    obj = torch.load(cfg["weights"], map_location="cpu", weights_only=False)
    loader = cfg["loader"]

    if isinstance(obj, torch.nn.Module) or (isinstance(obj, dict) and
                                             isinstance(obj.get("model"), torch.nn.Module)):
        if loader not in ("auto", "full"):
            print(f"Note: {cfg['weights']} contains a full model; ignoring loader={loader}")
        model = obj if isinstance(obj, torch.nn.Module) else obj["model"]
        how = "full model"
    else:
        sd = _unwrap_state_dict(obj)
        builders = []
        if loader in ("auto", "timm"):
            def timm_build():
                import timm
                return timm.create_model(cfg["arch"], pretrained=False, num_classes=n)
            builders.append(("timm", timm_build))
        if loader in ("auto", "torchvision"):
            def tv_build():
                import torchvision
                return getattr(torchvision.models, cfg["arch"])(weights=None, num_classes=n)
            builders.append(("torchvision", tv_build))

        model, errors = None, []
        for name, build in builders:
            try:
                m = build()
                m.load_state_dict(sd, strict=True)
                model, how = m, f"{name} {cfg['arch']} state_dict"
                break
            except Exception as e:  # try the next builder
                errors.append(f"  {name}: {str(e).splitlines()[0][:300]}")
        if model is None:
            raise SystemExit("Could not load the uniform classifier. Check 'loader', 'arch' and "
                             "the number of class_names.\n" + "\n".join(errors))

    out = _num_outputs(model, cfg)
    if out != n:
        raise SystemExit(f"Classifier has {out} outputs but class_names lists {n}.")
    print(f"Loaded uniform classifier ({how}), classes: {cfg['class_names']}")
    return model.to(device).eval()


@torch.no_grad()
def _num_outputs(model, cfg):
    h, w = cfg["input_size"]
    return model.eval()(torch.zeros(1, 3, h, w)).shape[-1]


def _preprocess(crop, cfg):
    h, w = cfg["input_size"]
    if cfg["resize"] == "letterbox":
        scale = min(w / crop.width, h / crop.height)
        nw, nh = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
        canvas = Image.new("RGB", (w, h), tuple(cfg.get("pad_color", (114, 114, 114))))
        canvas.paste(crop.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        crop = canvas
    else:
        crop = crop.resize((w, h), Image.BILINEAR)
    t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8)
    t = t.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def classify_crops(model, crops, cfg, device):
    """Return list of (class_name, prob) for a list of PIL RGB crops."""
    results = []
    bs = cfg["batch_size"]
    for i in range(0, len(crops), bs):
        x = torch.stack([_preprocess(c, cfg) for c in crops[i:i + bs]]).to(device)
        probs = F.softmax(model(x).float(), dim=1).cpu()
        conf, idx = probs.max(dim=1)
        results += [(cfg["class_names"][j], p) for j, p in zip(idx.tolist(), conf.tolist())]
    return results


def add_uniform_labels(model, image_path, boxes, cfg, device, stats):
    """Classify each person box in one image and add clothing boxes."""
    persons = [i for i, b in enumerate(boxes) if int(b[5]) == CID["person"]]
    if not persons:
        return boxes
    vest_flags = has_vest(boxes, [boxes[i][:4] for i in persons])

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    todo, crops = [], []
    for i, wears_vest in zip(persons, vest_flags):
        x1, y1, x2, y2 = boxes[i][:4]
        if cfg["skip_if_vest"] and wears_vest:
            stats["skipped: vest"] += 1
            continue
        if y2 - y1 < cfg["min_person_px"]:
            stats["skipped: too small"] += 1
            continue
        px, py = (x2 - x1) * cfg["crop_pad"], (y2 - y1) * cfg["crop_pad"]
        crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))
        todo.append(i)
        crops.append(crop)

    drop = set()
    for i, (cls_name, p) in zip(todo, classify_crops(model, crops, cfg, device)):
        stats[f"{cls_name} ({'kept' if p >= cfg['min_conf'] else 'low conf'})"] += 1
        if p < cfg["min_conf"]:
            continue
        new = {_norm(k): v for k, v in cfg["map"].items()}.get(_norm(cls_name))
        if new:
            boxes.append([*boxes[i][:4], p, CID[new]])
        elif cfg["drop_person_if_not_person"]:
            drop.add(i)
    return [b for i, b in enumerate(boxes) if i not in drop]

# ================================================================ output


def write_cvat_yolo(all_boxes, sizes, out_zip):
    counts = Counter()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASSES)}\ntrain = data/train.txt\n"
                               "names = data/obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{n}\n" for n in sizes))
        for name, (h, w) in sizes.items():
            lines = []
            for x1, y1, x2, y2, _, c in all_boxes.get(name, []):
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts[CLASSES[int(c)]] += 1
            z.writestr(f"obj_train_data/{Path(name).stem}.txt", "\n".join(lines) + "\n")
    return counts


def torch_device(device):
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{device}" if device.isdigit() else device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="flat folder of frames")
    ap.add_argument("--out", default="prelabels.zip")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0", help="GPU index or 'cpu'")
    ap.add_argument("--check-classifier", nargs="+", metavar="CROP",
                    help="only classify these person-crop images and print the result")
    args = ap.parse_args()
    cfg = UNIFORM_CLASSIFIER
    if cfg["weights"]:
        cfg = apply_meta(cfg)
    tdev = torch_device(args.device)

    if args.check_classifier:
        model = load_classifier(cfg, tdev)
        crops = [Image.open(p).convert("RGB") for p in args.check_classifier]
        for p, (c, conf) in zip(args.check_classifier, classify_crops(model, crops, cfg, tdev)):
            print(f"{p}: {c} ({conf:.2f})")
        return

    if not args.images:
        ap.error("--images is required")
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    names = [p.name for p in images]
    if len(set(Path(n).stem for n in names)) != len(names):
        raise SystemExit("Duplicate file stems found; rename so each image stem is unique.")
    if cfg["weights"] and not any("person" in m.values() for m in MODELS.values()):
        raise SystemExit("The uniform classifier needs a detector in MODELS that outputs 'person'.")
    print(f"{len(images)} images")

    classifier = load_classifier(cfg, tdev) if cfg["weights"] else None
    dets, sizes = run_detectors(images, args.imgsz, args.device)

    stats = Counter()
    by_name = {p.name: p for p in images}
    cleaned = {}
    for name in sizes:
        boxes = dedupe(dets.get(name, []))
        if classifier is not None:
            boxes = add_uniform_labels(classifier, by_name[name], boxes, cfg, tdev, stats)
        cleaned[name] = apply_vest_rule(boxes)

    counts = write_cvat_yolo(cleaned, sizes, args.out)

    if classifier is not None:
        print("\nUniform classifier, per person:")
        for k, v in sorted(stats.items()):
            print(f"  {k:28s} {v}")
    print(f"\nWrote {args.out}. Pre-label counts per class:")
    for c in CLASSES:
        print(f"  {c:24s} {counts[c]}")


if __name__ == "__main__":
    main()
