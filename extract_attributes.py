"""Extract per-pole attributes and write a CSV.

Default engine is SAM 3 (text-promptable, single model): it detects + segments
every pole from the prompt "utility pole" in one pass. Pass --engine detsam to
use the YOLO-detector + SAM2.1 pipeline instead.

Honest about what is measurable from a single GSV crop with no camera metadata:

Reliable (pure image geometry):
  - pole pixel height / width / area, aspect ratio (slenderness)
  - tilt from vertical (deg), via PCA on mask pixels (robust for vertical poles)
  - box + centroid location in the frame

Approximate (needs scale we do not have):
  - height_m_rough: rough meters via a fixed assumed distance. Low-confidence.

Not available here:
  - latitude / longitude / heading (no GSV metadata shipped with these images).

Usage:
    python extract_attributes.py --images Dataset --n 0 --out output/results/poles.csv
    python extract_attributes.py --engine detsam --weights models/pole_detector_best.pt
"""

from __future__ import annotations
import pole_env  # noqa: F401  (forces HF offline; robust to token issues)
import os, csv, argparse, glob
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np

ASSUMED_DISTANCE_M = 15.0
GSV_FOV_DEG = 90.0


def tilt_and_geometry(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 0, 0, 0, 0.0, 0.0, 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pole_w = int(x1 - x0 + 1)
    pole_h = int(y1 - y0 + 1)
    area = int((mask > 0).sum())
    cx, cy = float(xs.mean()), float(ys.mean())
    pts = np.stack([xs - cx, ys - cy], axis=1).astype(np.float64)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    tilt = np.degrees(np.arctan2(abs(major[0]), abs(major[1])))
    return pole_h, pole_w, area, round(float(tilt), 2), round(cx, 1), round(cy, 1)


def rough_height_m(pole_h_px: int, img_w: int) -> float:
    f_px = (img_w / 2.0) / np.tan(np.radians(GSV_FOV_DEG / 2.0))
    if f_px <= 0:
        return 0.0
    return round(float(np.clip(pole_h_px * ASSUMED_DISTANCE_M / f_px, 1.0, 40.0)), 2)


def iter_detections(engine_kind, engine, fp, conf):
    """Yield (box, score, mask, img_w, source) per pole for the chosen engine."""
    from PIL import Image
    im = Image.open(fp).convert("RGB")
    if engine_kind == "sam3":
        boxes, scores, masks = engine.detect_segment(im, threshold=conf)
        for b, s, m in zip(boxes, scores, masks):
            if m is not None:
                yield b, s, m, im.size[0], "sam3"
    elif engine_kind == "hybrid":
        for box, sc, mask, src in engine.run(fp, sam3_thr=conf):
            if mask is not None:
                yield box, sc, mask, im.size[0], src
    else:  # detsam
        boxes, scores, masks = engine.run_image(fp, conf=conf)
        for b, s, m in zip(boxes, scores, masks):
            if m is not None:
                yield b, s, m, im.size[0], "detsam"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="sam3", choices=["sam3", "detsam", "hybrid"])
    p.add_argument("--weights", default="models/pole_detector_v2.pt",
                   help="detector weights (detsam/hybrid engines)")
    p.add_argument("--images", default="Dataset")
    p.add_argument("--n", type=int, default=0, help="0 = all images")
    p.add_argument("--conf", type=float, default=0.5,
                   help="SAM3 score threshold (or detector conf for detsam)")
    p.add_argument("--out", default="output/results/poles.csv")
    args = p.parse_args()

    if args.engine == "sam3":
        from sam3_poles import Sam3Poles
        engine = Sam3Poles()
    elif args.engine == "hybrid":
        from hybrid_poles import HybridPoles
        engine = HybridPoles(weights=args.weights)
    else:
        from detect_sam import DetectSAM, find_best
        engine = DetectSAM(args.weights or find_best())

    files = sorted(glob.glob(str(Path(args.images) / "*.jpg")))
    if args.n:
        step = max(1, len(files) // args.n)
        files = files[::step][:args.n]

    cols = ["image", "pole_id", "source", "confidence",
            "box_x0", "box_y0", "box_x1", "box_y1",
            "pole_height_px", "pole_width_px", "mask_area_px", "aspect_ratio",
            "tilt_deg_from_vertical", "centroid_x", "centroid_y",
            "height_m_rough", "height_confidence",
            "latitude", "longitude", "heading"]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_img = n_pole = 0
    with open(out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(cols)
        for fp in files:
            n_img += 1
            pid = 0
            for box, sc, mask, img_w, src in iter_detections(args.engine, engine, fp, args.conf):
                ph, pw, area, tilt, cx, cy = tilt_and_geometry(mask)
                if ph == 0:
                    continue
                aspect = round(ph / pw, 2) if pw else 0.0
                wr.writerow([
                    Path(fp).name, pid, src, round(sc, 4),
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                    ph, pw, area, aspect, tilt, cx, cy,
                    rough_height_m(ph, img_w), "low",
                    "", "", "",
                ])
                pid += 1; n_pole += 1
            if n_img % 200 == 0:
                print(f"  {n_img}/{len(files)} images, {n_pole} poles")
    print(f"DONE [{args.engine}]: {n_img} images, {n_pole} poles -> {out}")


if __name__ == "__main__":
    main()
