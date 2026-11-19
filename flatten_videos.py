"""
Move (cut) every video from a multi-level folder tree into one flat destination folder.

Alert clips in different folders often share a name (clip.mp4, 0001.mp4 ...), so each file
is renamed after its folder path: source/cam07/2026-11-18/clip.mp4 -> cam07_2026-11-18_clip.mp4.
That keeps the camera and date in the name (useful later: the train/val split groups frames
by video name) and nothing ever overwrites anything.

Usage
  python flatten_videos.py --src alert_clips --dst plant2_clips --dry-run   # preview first
  python flatten_videos.py --src alert_clips --dst plant2_clips             # move
  python flatten_videos.py --src alert_clips --dst plant2_clips --copy      # copy, keep originals

Options
  --copy           copy instead of move (originals stay where they are)
  --name-only      keep the original file name (adds _1, _2 ... only when a name is taken)
  --remove-empty   after moving, delete folders left empty in the source tree
Writes <dst>/moved_files.csv: original path -> new path, for every file.
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg",
              ".webm", ".h264", ".dav"}


def clean(part):
    """Make one path part safe and readable in a file name."""
    part = re.sub(r"[^\w.-]+", "_", part).strip("._")
    return part or "x"


def unique(path, taken):
    if path not in taken and not path.exists():
        return path
    i = 1
    while True:
        p = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if p not in taken and not p.exists():
            return p
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="top folder of the downloaded clips")
    ap.add_argument("--dst", required=True, help="destination folder (created if missing)")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--name-only", action="store_true", help="keep original names (no folder prefix)")
    ap.add_argument("--remove-empty", action="store_true", help="delete emptied source folders")
    ap.add_argument("--dry-run", action="store_true", help="only show what would happen")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    if src == dst:
        raise SystemExit("--src and --dst must be different folders.")

    videos = sorted(p for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                    and dst not in p.parents)                      # skip files already in dst
    if not videos:
        raise SystemExit(f"No video files found under {src}")

    plan, taken = [], set()
    for v in videos:
        if args.name_only:
            name = clean(v.stem) + v.suffix.lower()
        else:
            parts = [clean(p) for p in v.relative_to(src).parent.parts] + [clean(v.stem)]
            name = "_".join(parts) + v.suffix.lower()
        target = unique(dst / name, taken)
        taken.add(target)
        plan.append((v, target))

    verb = "Copy" if args.copy else "Move"
    total_mb = sum(v.stat().st_size for v, _ in plan) / 1e6
    print(f"{len(plan)} videos ({total_mb:,.0f} MB) under {src}")
    for v, t in plan[:10]:
        print(f"  {v.relative_to(src)}  ->  {t.name}")
    if len(plan) > 10:
        print(f"  ... and {len(plan) - 10} more")
    if args.dry_run:
        print(f"\nDry run: nothing was changed. Run again without --dry-run to {verb.lower()}.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    log_path = dst / "moved_files.csv"
    new_log = not log_path.exists()
    done = failed = 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["original", "new", "action"])
        for i, (v, t) in enumerate(plan, 1):
            try:
                if args.copy:
                    shutil.copy2(v, t)
                else:
                    shutil.move(str(v), str(t))       # works across drives too
                w.writerow([str(v), str(t), verb.lower()])
                done += 1
            except OSError as e:
                print(f"  FAILED {v}: {e}")
                failed += 1
            if i % 50 == 0 or i == len(plan):
                print(f"  {i}/{len(plan)} done", end="\r")
    print()

    removed = 0
    if args.remove_empty and not args.copy:
        for d in sorted((p for p in src.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
            try:
                d.rmdir()                              # only succeeds if the folder is empty
                removed += 1
            except OSError:
                pass

    summary = f"{'Copied' if args.copy else 'Moved'} {done} videos to {dst}"
    if failed:
        summary += f", {failed} failed"
    if removed:
        summary += f"; removed {removed} empty folders"
    print(summary)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
