"""Hybrid pole pipeline: YOLO detector + SAM 3.

Combines the strengths measured on the human test split:
  - YOLO26m (human-supervised): high-precision, reliably nails the primary
    foreground pole (F1 0.94 on the human test set).
  - SAM 3 (text-prompt): best masks and much higher recall - it also finds the
    distant / secondary poles the human labels omit.

Fusion:
  1. Run both. Match SAM3 poles to YOLO boxes by IoU.
  2. Every kept pole gets a SAM 3 mask (best segmenter). For a YOLO-only pole
     with no SAM3 match, fall back to a box-prompted SAM 3 mask.
  3. Tag each pole's source: 'both' (YOLO+SAM3 agree, highest confidence),
     'yolo' (YOLO only), or 'sam3' (SAM3 only, usually a secondary pole).

This yields YOLO's precision on the main pole, SAM 3's recall on the rest, and
SAM 3 masks throughout. Use --primary-only to keep just the YOLO-anchored poles.
"""
from __future__ import annotations
import pole_env  # noqa: F401  (forces HF offline; robust to token issues)
import os, argparse, glob, csv, math
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import cv2
from pathlib import Path


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-9)


class HybridPoles:
    def __init__(self, weights="models/pole_detector_refined.pt",
                 sam3_id="facebook/sam3", device=None):
        from ultralytics import YOLO
        from sam3_poles import Sam3Poles
        from pole_sam import PoleSAM
        self.det = YOLO(weights)
        self.sam3 = Sam3Poles(model_id=sam3_id, device=device)
        # box-prompted SAM (2.1) to mask YOLO-only poles SAM3 missed
        self._boxsam = None
        self._PoleSAM = PoleSAM

    def _box_masker(self):
        if self._boxsam is None:
            self._boxsam = self._PoleSAM(backend="sam2")
        return self._boxsam

    def run(self, image_path, yolo_conf=0.25, sam3_thr=0.5,
            match_iou=0.5, primary_only=False):
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        # YOLO
        r = self.det(image_path, conf=yolo_conf, verbose=False)[0]
        yb = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])]
        ys = [float(b.conf[0]) for b in (r.boxes or [])]
        # SAM3
        sb, ss, sm = self.sam3.detect_segment(im, threshold=sam3_thr)

        used_sam3 = [False] * len(sb)
        poles = []  # (box, score, mask, source)

        # 1) YOLO-anchored poles: attach best-matching SAM3 mask if any
        for i, yboxi in enumerate(yb):
            best, bi = 0.0, -1
            for j, sboxj in enumerate(sb):
                if used_sam3[j]:
                    continue
                v = iou(yboxi, sboxj)
                if v > best:
                    best, bi = v, j
            if best >= match_iou and bi >= 0:
                used_sam3[bi] = True
                poles.append((sb[bi], max(ys[i], ss[bi]), sm[bi], "both"))
            else:
                # YOLO-only pole: make a SAM2.1 mask from its box
                m = self._box_masker().segment(im, [yboxi])
                mask = m[0] if m else None
                poles.append((yboxi, ys[i], mask, "yolo"))

        # 2) SAM3-only poles (secondary/distant), unless primary_only
        if not primary_only:
            for j in range(len(sb)):
                if not used_sam3[j]:
                    poles.append((sb[j], ss[j], sm[j], "sam3"))
        return poles


def overlay(image_path, poles, out_path):
    img = cv2.imread(image_path)
    color = {"both": (0, 255, 0), "yolo": (255, 128, 0), "sam3": (0, 200, 255)}
    for box, sc, mask, src in poles:
        c = color.get(src, (0, 255, 0))
        if mask is not None:
            ov = img.copy(); ov[mask.astype(bool)] = c
            img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        cv2.putText(img, f"{src} {sc:.2f}", (x0 + 2, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/pole_detector_refined.pt")
    ap.add_argument("--images", default="Dataset")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--sam3-thr", type=float, default=0.5)
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--out", default="output/hybrid")
    ap.add_argument("--montage", action="store_true")
    args = ap.parse_args()

    eng = HybridPoles(weights=args.weights)
    files = sorted(glob.glob(str(Path(args.images) / "*.jpg")))
    if args.n:
        step = max(1, len(files) // args.n)
        files = files[::step][:args.n]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    saved, counts = [], {"both": 0, "yolo": 0, "sam3": 0}
    for fp in files:
        poles = eng.run(fp, yolo_conf=args.yolo_conf, sam3_thr=args.sam3_thr,
                        primary_only=args.primary_only)
        for _, _, _, src in poles:
            counts[src] = counts.get(src, 0) + 1
        dst = out / Path(fp).name
        overlay(fp, poles, str(dst))
        saved.append(str(dst))
    print(f"{len(files)} images -> poles by source: {counts} -> {out}")

    if args.montage and saved:
        from PIL import Image
        ims = [Image.open(s).convert("RGB").resize((320, 320)) for s in saved[:16]]
        cols = 4; rows = math.ceil(len(ims) / cols)
        m = Image.new("RGB", (cols * 320, rows * 320), (20, 20, 20))
        for i, im in enumerate(ims):
            m.paste(im, ((i % cols) * 320, (i // cols) * 320))
        m.save(out / "montage.jpg")
        print(f"montage -> {out / 'montage.jpg'}")


if __name__ == "__main__":
    main()
