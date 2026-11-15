"""
Train a harness classifier on person crops (EfficientNet, same setup as your uniform
classifier: letterbox to 224 px with grey (114) padding, ImageNet normalisation).

Input folder (one sub-folder per class; nested sub-folders are fine, e.g. harness/unclipped/):
  crops/
    harness/      no_harness/      unsure/

Crops must keep the names from extract_person_crops.py (<video>_f<frame>_p<n>.jpg):
the train/val split is done BY VIDEO, so near-identical crops of the same worker never
end up in both train and val.

Usage (dev server GPU)
  pip install -U timm
  python train_harness_classifier.py --data crops --out harness_model

  --ignore unsure      train only harness vs no_harness
  --max-per-video 30   keep at most 30 crops per class per video (stops one video of the
                       same few workers from dominating; also cuts near-duplicates)
  --val-videos A B     choose the val videos yourself (best: videos with BOTH classes)
  --split time         keep every video in train; val = separate time blocks inside each
                       video, with a gap around them so near-duplicates can't leak
                       (--block-seconds, --gap-seconds, --video-fps)
  --arch efficientnet_b2 --img-size 260    bigger model, if B0 isn't enough

Output (harness_model/)
  harness_classifier.pt            best weights (state_dict, timm)
  harness_classifier_meta.json     classes, preprocessing - same format as the uniform model
  val_predictions.csv              every val crop with its prediction (sort by 'wrong' to
                                   review mistakes - often these are labelling errors)
"""

import argparse
import csv
import json
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN, STD, PAD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (114, 114, 114)


# ---------------------------------------------------------------- data

def video_of(path):
    m = re.match(r"^(.*)_f\d+_p\d+$", path.stem)
    return m.group(1) if m else re.sub(r"[_-]*\d+$", "", path.stem)


def load_items(root, ignore):
    classes = sorted(d.name for d in Path(root).iterdir() if d.is_dir() and d.name not in ignore)
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 class folders in {root}, found {classes}")
    items = [(p, i) for i, c in enumerate(classes) for p in sorted((Path(root) / c).rglob("*"))
             if p.suffix.lower() in IMG_EXTS]
    return classes, items


def cap_per_video(items, cap, seed):
    """Keep at most `cap` crops per (video, class), spread evenly over the video."""
    by = defaultdict(list)
    for it in items:
        by[(video_of(it[0]), it[1])].append(it)
    out = []
    for key, its in sorted(by.items()):
        its = sorted(its, key=lambda it: it[0].name)
        if len(its) > cap:                               # evenly spaced over time
            idx = np.linspace(0, len(its) - 1, cap).round().astype(int)
            its = [its[i] for i in sorted(set(idx))]
        out += its
    return out


def frame_of(path):
    m = re.search(r"_f(\d+)_p\d+$", path.stem)
    return int(m.group(1)) if m else None


def split_by_time(items, n_classes, val_frac, block_s, gap_s, fps):
    """Inside every video: cut the timeline into blocks, send every k-th block to val and
    drop train crops within gap_s of a val block (they'd be near-duplicates)."""
    if any(frame_of(p) is None for p, _ in items):
        raise SystemExit("--split time needs crop names like <video>_f<frame>_p<n>.jpg")
    k = max(2, round(1 / val_frac))
    block, gap = block_s * fps, gap_s * fps
    total = Counter(c for _, c in items)
    best = None
    for offset in range(k):                              # pick the offset with the best balance
        train, val, dropped = [], [], 0
        by_video = defaultdict(list)
        for it in items:
            by_video[video_of(it[0])].append(it)
        for v, its in by_video.items():
            val_blocks = {b for b in range(int(max(frame_of(p) for p, _ in its) // block) + 1)
                          if (b + offset + zlib.crc32(v.encode()) % k) % k == 0}
            for it in its:
                f = frame_of(it[0])
                b = int(f // block)
                if b in val_blocks:
                    val.append(it)
                elif any(abs(f - (vb * block)) < gap or abs(f - (vb + 1) * block) < gap
                         for vb in val_blocks):
                    dropped += 1                         # too close to a val block
                else:
                    train.append(it)
        vc = Counter(c for _, c in val)
        score = sum(abs(vc[c] / max(total[c], 1) - val_frac) + (1 if vc[c] == 0 else 0)
                    for c in range(n_classes))
        if best is None or score < best[0]:
            best = (score, train, val, dropped)
    _, train, val, dropped = best
    print(f"Time split: {len(train)} train, {len(val)} val, {dropped} dropped as near-duplicates "
          f"(blocks {block_s}s, gap {gap_s}s)")
    return train, val, ["(time blocks inside every video)"], len({video_of(p) for p, _ in items})


def split_fixed(items, val_videos):
    names = {video_of(p) for p, _ in items}
    missing = [v for v in val_videos if v not in names]
    if missing:
        raise SystemExit(f"--val-videos not found: {missing}. Videos: {sorted(names)}")
    val = [it for it in items if video_of(it[0]) in val_videos]
    train = [it for it in items if video_of(it[0]) not in val_videos]
    return train, val, sorted(val_videos), len(names)


def split_by_video(items, n_classes, val_frac, seed, trials=5000):
    groups = defaultdict(list)
    for it in items:
        groups[video_of(it[0])].append(it)
    if len(groups) < 3:
        raise SystemExit(f"Only {len(groups)} videos found in crop names - the split needs "
                         "crops from several videos (names like <video>_f<frame>_p<n>.jpg).")
    gc = {g: Counter(c for _, c in its) for g, its in groups.items()}
    total = Counter(c for _, c in items)
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
            vc.update(gc[g])
        score = abs(n / len(items) - val_frac) * 2
        for k in range(n_classes):
            share = vc[k] / max(total[k], 1)
            score += abs(share - val_frac) + (1.0 if share in (0, 1) else 0)
        if score < best_score:
            best, best_score = set(val), score
    train = [it for g in groups if g not in best for it in groups[g]]
    val = [it for g in best for it in groups[g]]
    return train, val, sorted(best), len(groups)


def letterbox(img, size):
    s = min(size / img.width, size / img.height)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (size, size), PAD)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_tensor(img):
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class Crops(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def augment(self, img):
        w, h = img.size
        # detector boxes jitter: random crop of 85-100% of the padded crop, slight shift
        s = random.uniform(0.85, 1.0)
        cw, ch = int(w * random.uniform(s, 1.0)), int(h * random.uniform(s, 1.0))
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.25:                       # night / IR cameras are grayscale
            img = ImageOps.grayscale(img).convert("RGB")
        if random.random() < 0.2:                        # CCTV blur / compression
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        a = np.asarray(img, dtype=np.float32)            # brightness / contrast
        a = (a - 128) * random.uniform(0.7, 1.3) + 128 + random.uniform(-30, 30)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.train:
            img = self.augment(img)
        return to_tensor(letterbox(img, self.size)), label


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, probs, labels = [], [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = model(x.to(device, non_blocking=True)).float().softmax(1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        labels.append(y)
    probs, preds, labels = torch.cat(probs), torch.cat(preds), torch.cat(labels)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    recall = cm.diag().float() / cm.sum(1).clamp(min=1)
    precision = cm.diag().float() / cm.sum(0).clamp(min=1)
    present = cm.sum(1) > 0
    return dict(bal_acc=recall[present].mean().item(), acc=(preds == labels).float().mean().item(),
                recall=recall, precision=precision, cm=cm, probs=probs, preds=preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with harness/ no_harness/ unsure/")
    ap.add_argument("--out", default="harness_model")
    ap.add_argument("--ignore", nargs="*", default=[], help="class folders to leave out")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="stop after N epochs w/o gain")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split", choices=["video", "time"], default="video",
                    help="video: whole videos go to val; time: val = time blocks in every video")
    ap.add_argument("--block-seconds", type=float, default=30)
    ap.add_argument("--gap-seconds", type=float, default=10)
    ap.add_argument("--video-fps", type=float, default=25, help="fps of the source videos")
    ap.add_argument("--val-videos", nargs="*", default=None,
                    help="video names to use as val (instead of an automatic split)")
    ap.add_argument("--max-per-video", type=int, default=None,
                    help="max crops per class per video (applied to train only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only)")
    args = ap.parse_args()

    import timm
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU visible - training on CPU will be slow.")

    classes, items = load_items(args.data, set(args.ignore))
    if args.split == "time":
        train, val, val_videos, n_videos = split_by_time(
            items, len(classes), args.val_frac, args.block_seconds, args.gap_seconds,
            args.video_fps)
    elif args.val_videos:
        train, val, val_videos, n_videos = split_fixed(items, set(args.val_videos))
    else:
        train, val, val_videos, n_videos = split_by_video(items, len(classes), args.val_frac,
                                                          args.seed)
    if args.max_per_video:
        before = len(train)
        train = cap_per_video(train, args.max_per_video, args.seed)
        print(f"--max-per-video {args.max_per_video}: train {before} -> {len(train)} crops")
    print("Val videos: " + ", ".join(val_videos))
    tc, vc = Counter(c for _, c in train), Counter(c for _, c in val)
    where = "time blocks in every video" if args.split == "time" else f"{len(val_videos)} videos"
    print(f"{len(items)} crops from {n_videos} videos; val = {where}\n")
    print(f"  {'class':12s} {'train':>6s} {'val':>5s}")
    for i, c in enumerate(classes):
        flag = "  <- NOT IN VAL" if vc[i] == 0 else ""
        print(f"  {c:12s} {tc[i]:6d} {vc[i]:5d}{flag}")

    # class-balanced sampling, so the smaller classes (e.g. unsure) aren't ignored
    weights = [1.0 / tc[c] for _, c in train]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    pin = device.type == "cuda"
    train_dl = DataLoader(Crops(train, args.img_size, True), batch_size=args.batch,
                          sampler=sampler, num_workers=args.workers, pin_memory=pin,
                          drop_last=len(train) > args.batch)
    val_dl = DataLoader(Crops(val, args.img_size, False), batch_size=args.batch * 2,
                        num_workers=args.workers, pin_memory=pin)

    model = timm.create_model(args.arch, pretrained=not args.no_pretrained,
                              num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, since_best = -1.0, 0, 0
    print(f"\nTraining {args.arch} @ {args.img_size}px on {device}, {args.epochs} epochs max\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        r = evaluate(model, val_dl, device, len(classes))
        per = "  ".join(f"{c} R={r['recall'][i]:.2f}" for i, c in enumerate(classes))
        mark = ""
        if r["bal_acc"] > best:
            best, best_epoch, since_best, mark = r["bal_acc"], epoch, 0, "  * saved"
            torch.save(model.state_dict(), out / "harness_classifier.pt")
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {tot / max(n, 1):.3f}  val bal_acc {r['bal_acc']:.3f}  "
              f"acc {r['acc']:.3f}  {per}  ({time.time() - t0:.0f}s){mark}")
        if since_best >= args.patience:
            print(f"No improvement for {args.patience} epochs - stopping.")
            break

    # final report on the best weights
    model.load_state_dict(torch.load(out / "harness_classifier.pt", map_location=device))
    r = evaluate(model, val_dl, device, len(classes))
    print(f"\nBest epoch {best_epoch}: val balanced accuracy {r['bal_acc']:.3f}\n")
    print(f"  {'class':12s} {'precision':>9s} {'recall':>7s}")
    for i, c in enumerate(classes):
        print(f"  {c:12s} {r['precision'][i]:9.3f} {r['recall'][i]:7.3f}")
    print("\n  Confusion matrix (rows = true, columns = predicted)")
    print("  " + " " * 12 + "".join(f"{c[:10]:>11s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:12s}" + "".join(f"{v:11d}" for v in r["cm"][i].tolist()))
    if "harness" in classes and "no_harness" in classes:
        h, nh = classes.index("harness"), classes.index("no_harness")
        print(f"\n  no_harness predicted as harness (missed violations): {r['cm'][nh, h].item()}")
        print(f"  harness predicted as no_harness (false alarms):     {r['cm'][h, nh].item()}")

    with open(out / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "true", "predicted", "confidence", "wrong"] +
                   [f"p_{c}" for c in classes])
        for (path, y), p, pr in zip(val, r["preds"].tolist(), r["probs"].tolist()):
            w.writerow([str(path), classes[y], classes[p], f"{max(pr):.3f}", int(p != y)] +
                       [f"{v:.3f}" for v in pr])

    meta = {"model_name": args.arch, "classes": classes, "img_size": args.img_size,
            "pad_color": list(PAD), "mean": list(MEAN), "std": list(STD), "decision": "argmax",
            "best_epoch": best_epoch, "val_bal_acc": r["bal_acc"],
            "crop_padding_note": "expand detector box by 10% per side before classifying",
            "framework": "timm", "val_videos": val_videos}
    (out / "harness_classifier_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {out / 'harness_classifier.pt'}, {out / 'harness_classifier_meta.json'}, "
          f"{out / 'val_predictions.csv'}")


if __name__ == "__main__":
    main()
