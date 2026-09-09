"""Predict poles with SAM 3 (text-promptable, single model) and save visual QA.

SAM 3 detects + segments all poles from the text prompt "utility pole" in one
pass, so no separate detector is needed. Benchmarks (see detector_eval_report)
showed SAM 3 gives perfect vertical coverage, zero box leakage, and ~2.6x the
pole recall of the YOLO+SAM2.1 pipeline.

Usage:
    python predict_sam3.py --images Dataset --n 16 --threshold 0.5 --montage
"""
from __future__ import annotations
import pole_env  # noqa: F401  (forces HF offline; robust to token issues)
import os, argparse, glob, csv, math
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import cv2
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="Dataset")
    p.add_argument("--n", type=int, default=16, help="0 = all")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--prompt", default="utility pole")
    p.add_argument("--out", default="output/sam3_pred")
    p.add_argument("--montage", action="store_true")
    p.add_argument("--csv", default=None, help="optional path to write a detections CSV")
    args = p.parse_args()

    from PIL import Image
    from sam3_poles import Sam3Poles
    eng = Sam3Poles(prompt=args.prompt)

    files = sorted(glob.glob(str(Path(args.images) / "*.jpg")))
    if args.n:
        step = max(1, len(files) // args.n)
        files = files[::step][:args.n]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = []
    total = 0
    for fp in files:
        im = Image.open(fp).convert("RGB")
        boxes, scores, masks = eng.detect_segment(im, threshold=args.threshold)
        total += len(boxes)
        img = cv2.imread(fp)
        for i, (b, s, m) in enumerate(zip(boxes, scores, masks)):
            ov = img.copy(); ov[m.astype(bool)] = (0, 255, 0)
            img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img, f"{s:.2f}", (int(b[0]) + 2, max(0, int(b[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            rows.append([Path(fp).name, i, round(s, 4),
                         int(b[0]), int(b[1]), int(b[2]), int(b[3]), int((m > 0).sum())])
        dst = out / Path(fp).name
        cv2.imwrite(str(dst), img)
        saved.append(str(dst))
    print(f"{len(files)} images, {total} poles -> {out}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "pole_id", "score", "x0", "y0", "x1", "y1", "mask_area_px"])
            w.writerows(rows)
        print(f"CSV -> {args.csv}")

    if args.montage and saved:
        from PIL import Image as PImage
        ims = [PImage.open(s).convert("RGB").resize((320, 320)) for s in saved[:16]]
        cols = 4; rows_n = math.ceil(len(ims) / cols)
        mnt = PImage.new("RGB", (cols * 320, rows_n * 320), (20, 20, 20))
        for i, im in enumerate(ims):
            mnt.paste(im, ((i % cols) * 320, (i // cols) * 320))
        mnt.save(out / "montage.jpg")
        print(f"montage -> {out / 'montage.jpg'}")


if __name__ == "__main__":
    main()
