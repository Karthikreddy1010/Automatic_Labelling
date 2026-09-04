"""Run the trained pole segmentation model on images and save visual QA.

Usage:
    python predict.py --weights runs/.../weights/best.pt --images Dataset --n 12 \
        --out output/predictions
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict")


def find_best_weights(project="runs/segment") -> str | None:
    cands = sorted(Path(project).rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="path to best.pt (auto-detect if omitted)")
    p.add_argument("--images", default="Dataset")
    p.add_argument("--n", type=int, default=12, help="number of images to sample (0 = all)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="output/predictions")
    p.add_argument("--montage", action="store_true", help="also build a montage grid")
    args = p.parse_args()

    from ultralytics import YOLO
    import glob

    weights = args.weights or find_best_weights()
    if not weights or not Path(weights).exists():
        raise SystemExit("No weights found. Train first (train_seg.py).")
    logger.info(f"Loading {weights}")
    model = YOLO(weights)

    files = sorted(glob.glob(str(Path(args.images) / "*.jpg")))
    if args.n:
        # sample evenly across the dataset for variety
        step = max(1, len(files) // args.n)
        files = files[::step][:args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_det = 0
    saved = []
    for fp in files:
        r = model(fp, conf=args.conf, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        n_det += n
        vis = r.plot()  # BGR ndarray with masks + boxes
        import cv2
        dst = out / Path(fp).name
        cv2.imwrite(str(dst), vis)
        saved.append(str(dst))
    logger.info(f"Predicted {len(files)} images, {n_det} pole detections -> {out}")

    if args.montage and saved:
        from PIL import Image
        import math
        ims = [Image.open(s).convert("RGB").resize((320, 320)) for s in saved[:16]]
        cols = 4
        rows = math.ceil(len(ims) / cols)
        m = Image.new("RGB", (cols * 320, rows * 320), (20, 20, 20))
        for i, im in enumerate(ims):
            m.paste(im, ((i % cols) * 320, (i // cols) * 320))
        mp = out / "montage.jpg"
        m.save(mp)
        logger.info(f"Montage saved -> {mp}")


if __name__ == "__main__":
    main()
