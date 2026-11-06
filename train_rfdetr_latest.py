"""
Fine-tune RF-DETR (transformer detector, DINOv2 backbone, Apache-2.0 for N/S/M/L)
on the PPE dataset made by prepare_datasets.py.

Usage
  pip install -U "rfdetr[train,loggers]"
  python train_rfdetr.py --dataset datasets/ppe/coco --size medium --resolution 960

  Resolution must be a multiple of 32 for nano/small/medium (e.g. 640, 768, 896, 960).
  RTX 3070 (8 GB): --batch 2 --accum 8 (effective batch 16). If you run out of memory,
  use --resolution 768 or --size small.

Output
  runs/rfdetr/<name>/checkpoint_best_total.pth   best checkpoint (by val mAP)
  runs/rfdetr/<name>/                           logs, per-class metrics (tensorboard)
"""

import argparse

import rfdetr

SIZES = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
         "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/ppe/coco")
    ap.add_argument("--size", default="medium", choices=list(SIZES))
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="early stop after N epochs w/o gain")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.size != "large" and args.resolution % 32:
        raise SystemExit("--resolution must be a multiple of 32 (e.g. 768, 896, 960).")

    name = args.name or f"rfdetr_{args.size}_{args.resolution}_s{args.seed}"
    out_dir = f"runs/rfdetr/{name}"
    print(f"Effective batch size: {args.batch * args.accum}")

    model = SIZES[args.size](resolution=args.resolution)   # COCO-pretrained, auto-downloaded
    model.train(
        dataset_dir=args.dataset,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        lr=args.lr,
        num_workers=args.workers,
        seed=args.seed,
        early_stopping=True,
        early_stopping_patience=args.patience,
        log_per_class_metrics=True,
        run_test=False,               # no test split; test on your videos instead
        progress_bar="tqdm",
    )
    print(f"\nDone. Best checkpoint: {out_dir}/checkpoint_best_total.pth")
    print("Compare with YOLO26 on the same val set using compare_models.py.")


if __name__ == "__main__":
    main()
