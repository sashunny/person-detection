"""
Local CPU test: YOLO26n person detection -> uniform / no-uniform classifier on each crop.

Draws boxes + labels, shows the video in a resizable window, and saves the annotated output.

Install (once):
    pip install ultralytics opencv-python pillow torch torchvision onnxruntime

Usage:
    python test_uniform_video.py --video test.mp4
    python test_uniform_video.py --video test.mp4 --cls uniform_classifier.onnx --meta uniform_classifier_meta.json
    python test_uniform_video.py --video test.mp4 --window-width 1600 --out result.mp4

Keys while playing:  q / Esc = quit,  space = pause/resume
The window can also be resized freely with the mouse.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

GREEN = (0, 200, 0)      # uniform        (BGR)
RED = (0, 0, 230)        # no_uniform
GRAY = (160, 160, 160)   # too small to classify


# ------------------------- preprocessing (must match training) -------------------------
def letterbox_pil(img, size, fill):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(fill))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(pil_crops, meta):
    """List of PIL RGB crops -> float32 NCHW array, normalized like training."""
    mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    batch = []
    for c in pil_crops:
        a = np.asarray(letterbox_pil(c, meta["img_size"], meta["pad_color"]), dtype=np.float32) / 255.0
        a = (a - mean) / std
        batch.append(a.transpose(2, 0, 1))
    return np.stack(batch).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------- classifier backends -------------------------------
class TorchClassifier:
    def __init__(self, ckpt_path, threads=None):
        import torch
        from torchvision import models
        import torch.nn as nn

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.meta = ckpt["meta"]
        name, n_cls = self.meta["model_name"], len(self.meta["classes"])
        if name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
        elif name == "mobilenet_v3_large":
            m = models.mobilenet_v3_large(weights=None)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
        else:
            raise ValueError(f"Unknown model_name in checkpoint: {name}")
        m.load_state_dict(ckpt["state_dict"])
        self.model = m.eval()

    def __call__(self, pil_crops):
        x = self.torch.from_numpy(preprocess(pil_crops, self.meta))
        with self.torch.inference_mode():
            logits = self.model(x).numpy()
        return softmax(logits)[:, 1]  # P(uniform)


class OnnxClassifier:
    def __init__(self, onnx_path, meta_path, threads=None):
        import onnxruntime as ort

        with open(meta_path) as f:
            self.meta = json.load(f)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, pil_crops):
        logits = self.sess.run(None, {self.inp: preprocess(pil_crops, self.meta)})[0]
        return softmax(logits)[:, 1]


# ------------------------------------ drawing ------------------------------------
def draw_label(img, x1, y1, text, color):
    scale = max(0.5, min(img.shape[:2]) / 1000)
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y_top = max(0, y1 - th - base - 4)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(img, text, (x1 + 3, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(img, fps, n_uni, n_no):
    lines = [f"FPS: {fps:.1f}", f"Uniform: {n_uni}", f"No uniform: {n_no}"]
    scale = max(0.6, min(img.shape[:2]) / 900)
    thick = max(1, round(scale * 2))
    y = int(30 * scale)
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += int(32 * scale)


# -------------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Uniform detection test on a video (CPU)")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--det", default="yolo26n.pt", help="YOLO person detector weights")
    ap.add_argument("--cls", default="uniform_classifier.pt", help="classifier .pt or .onnx")
    ap.add_argument("--meta", default="uniform_classifier_meta.json", help="meta json (needed for .onnx)")
    ap.add_argument("--out", default=None, help="output video path (default: <input>_annotated.mp4)")
    ap.add_argument("--conf", type=float, default=0.4, help="person detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="detector input size")
    ap.add_argument("--pad", type=float, default=0.10, help="crop padding per side (same as training)")
    ap.add_argument("--min-h", type=int, default=60, help="skip classifying persons shorter than this (px)")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold from meta")
    ap.add_argument("--window-width", type=int, default=1280, help="initial display window width")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for the classifier")
    ap.add_argument("--no-show", action="store_true", help="don't display, only save")
    args = ap.parse_args()

    # --- models ---
    detector = YOLO(args.det)
    if args.cls.lower().endswith(".onnx"):
        classifier = OnnxClassifier(args.cls, args.meta, args.threads)
    else:
        classifier = TorchClassifier(args.cls, args.threads)
    thr = args.threshold if args.threshold is not None else classifier.meta.get("threshold", 0.5)
    print(f"Classifier threshold: {thr:.2f}")

    # --- video io ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.out or str(Path(args.video).with_name(Path(args.video).stem + "_annotated.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    win = "Uniform detection"
    if not args.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)          # resizable window
        disp_w = min(args.window_width, W)
        cv2.resizeWindow(win, disp_w, int(H * disp_w / W))  # keep aspect ratio

    frame_idx, fps_avg, paused = 0, 0.0, False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t0 = time.time()

            # 1) detect persons (class 0) — boxes come back in original frame coords
            res = detector.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                   device="cpu", verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else []

            # 2) crop with padding (from the full-res frame), BGR -> RGB
            crops, keep, small = [], [], []
            for (x1, y1, x2, y2) in boxes:
                bw, bh = x2 - x1, y2 - y1
                if bh < args.min_h:
                    small.append((x1, y1, x2, y2))
                    continue
                px, py = int(bw * args.pad), int(bh * args.pad)
                cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                keep.append((x1, y1, x2, y2))

            # 3) classify all crops of this frame in one batch
            probs = classifier(crops) if crops else []

            # 4) draw
            n_uni = n_no = 0
            for (x1, y1, x2, y2), p in zip(keep, probs):
                is_uni = p >= thr
                color = GREEN if is_uni else RED
                label = f"uniform {p:.2f}" if is_uni else f"no uniform {1 - p:.2f}"
                n_uni += int(is_uni)
                n_no += int(not is_uni)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, x1, y1, label, color)
            for (x1, y1, x2, y2) in small:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, 1)

            dt = time.time() - t0
            fps_avg = 0.9 * fps_avg + 0.1 * (1 / dt) if fps_avg else 1 / dt
            draw_hud(frame, fps_avg, n_uni, n_no)
            writer.write(frame)

            if frame_idx % 50 == 0:
                print(f"frame {frame_idx}/{total}  ~{fps_avg:.1f} FPS")

        if not args.no_show:
            cv2.imshow(win, frame)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            # stop if the user closed the window
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video -> {out_path}")


if __name__ == "__main__":
    main()
