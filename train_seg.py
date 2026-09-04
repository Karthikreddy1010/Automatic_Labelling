"""Train YOLO-seg on the auto-annotated utility-pole dataset.

This is a clean, self-contained trainer that fixes the two problems that broke
the previous run:

1. Labels: uses the open-vocabulary dataset produced by ``src.auto_annotate``
   (real pole masks) instead of the COCO-mislabeled boxes.
2. /dev/shm crash: this box has only 64MB of shared memory, which makes
   PyTorch's default DataLoader worker sharing crash. We force the
   ``file_system`` sharing strategy and default to ``workers=0`` so training is
   robust regardless of /dev/shm size.

Usage:
    python train_seg.py --data data/processed/yolo_dataset/dataset.yaml \
        --model yolo11s-seg.pt --epochs 100 --batch 16 --workers 0
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path

# Robust multiprocessing sharing for tiny /dev/shm before torch spins up loaders.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
torch.multiprocessing.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_seg")


def count_split(data_yaml: str):
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(cfg.get("path", Path(data_yaml).parent))
    out = {}
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            continue
        img_dir = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_inst = 0
        if lbl_dir.exists():
            for f in lbl_dir.glob("*.txt"):
                n_inst += sum(1 for ln in f.read_text().splitlines() if ln.strip())
        out[split] = (n_img, n_inst)
    return out


def main():
    p = argparse.ArgumentParser(description="Train YOLO-seg pole detector")
    p.add_argument("--data", default="data/processed/yolo_dataset/dataset.yaml")
    p.add_argument("--model", default="yolo26x-seg.pt",
                   help="pretrained seg checkpoint (already have yolo26x-seg.pt locally)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers. 0 avoids /dev/shm crashes.")
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--name", default="pole_seg")
    p.add_argument("--project", default="runs/pole")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    counts = count_split(args.data)
    logger.info(f"Dataset: {counts}")
    if counts.get("train", (0, 0))[0] < 10:
        raise SystemExit("Not enough training images. Run src.auto_annotate first.")

    model = YOLO(args.model)
    logger.info(f"Training {args.model} for {args.epochs} epochs "
                f"(batch={args.batch}, workers={args.workers}, device={args.device})")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        # Augmentation tuned for upright poles in street imagery.
        degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, mosaic=1.0, close_mosaic=10, mixup=0.0,
        cache=False, plots=True, verbose=True, seed=0,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best}")
    try:
        m = getattr(results, "results_dict", {})
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/mAP50(M)", "metrics/mAP50-95(M)",
                  "metrics/precision(B)", "metrics/recall(B)"):
            if k in m:
                logger.info(f"  {k}: {m[k]:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
