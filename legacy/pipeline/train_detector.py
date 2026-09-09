"""Train a YOLO detector on the human-labeled Final_Dataset (boxes only).

Detection is trained on clean human boxes; segmentation is handled separately
at inference time by prompting SAM with each predicted box (see detect_sam.py).
This avoids ever training on noisy auto-masks.

Uses workers=0 + cache=ram to stay robust on this host's 64MB /dev/shm.

Usage:
    python train_detector.py --model yolo26m.pt --epochs 80 --batch 32
"""

from __future__ import annotations
import os, argparse, logging
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_detector")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Final_Dataset/data_local.yaml")
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="pole_det")
    p.add_argument("--project", default="runs/detect")
    args = p.parse_args()

    from ultralytics import YOLO, RTDETR
    is_detr = "rtdetr" in args.model.lower()
    Model = RTDETR if is_detr else YOLO
    model = Model(args.model)
    logger.info(f"Training {args.model} on {args.data} ({args.epochs} ep, batch {args.batch})")

    # DETR diverges under YOLO's mosaic + high LR: use a low-LR, mosaic-off recipe.
    hp = dict(lr0=0.002, mosaic=1.0, close_mosaic=10)
    if is_detr:
        hp = dict(lr0=1e-4, mosaic=0.0, close_mosaic=0)

    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, project=args.project, name=args.name,
        exist_ok=True, patience=args.patience, optimizer="AdamW",
        lrf=0.01, cos_lr=True, warmup_epochs=3.0, weight_decay=0.0005,
        degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, mixup=0.0,
        cache="ram", plots=True, verbose=True, seed=0, **hp,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info(f"Done. Best: {best}")


if __name__ == "__main__":
    main()
