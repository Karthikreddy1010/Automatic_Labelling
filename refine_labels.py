"""Refine Final_Dataset labels using YOLO + SAM 3 consensus.

Error analysis showed the human labels omit many real poles (35% of the
detector's "false positives" and 51% of its "misses" are confirmed by SAM 3).
This tool cleans the labels by ADDING only high-confidence poles that BOTH
models independently agree on but that are missing from the human labels.

Consensus rule for a NEW pole box:
  - YOLO detects it (conf >= yolo_conf), AND
  - SAM 3 detects it (score >= sam3_thr) at IoU >= agree_iou, AND
  - it does not already match a human label (IoU < dup_iou to every GT box).

Existing human boxes are always kept. Output is a new dataset dir with the same
images and augmented label files, plus a report of how many were added.

Usage:
  python refine_labels.py --split train --out Final_Dataset_refined
  python refine_labels.py --split test  --out Final_Dataset_refined
"""
from __future__ import annotations
import os, glob, shutil, argparse, json
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
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


def read_labels(lp, w, h):
    boxes = []
    if not Path(lp).exists():
        return boxes
    for line in open(lp):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = [float(x) for x in p[1:5]]
        boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
    return boxes


def to_yolo_line(box, w, h):
    x0, y0, x1, y1 = box
    cx = (x0+x1)/2/w; cy = (y0+y1)/2/h
    bw = (x1-x0)/w; bh = (y1-y0)/h
    cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
    bw = min(max(bw, 1e-3), 1); bh = min(max(bh, 1e-3), 1)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--src", default="Final_Dataset")
    ap.add_argument("--out", default="Final_Dataset_refined")
    ap.add_argument("--mode", default="sam3", choices=["sam3", "consensus"],
                    help="sam3: add SAM3 high-conf new poles; consensus: require YOLO+SAM3 agreement")
    ap.add_argument("--weights", default="models/pole_detector_v2.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.40)
    ap.add_argument("--sam3-thr", type=float, default=0.60)
    ap.add_argument("--agree-iou", type=float, default=0.50)
    ap.add_argument("--dup-iou", type=float, default=0.40)
    ap.add_argument("--nms-iou", type=float, default=0.30,
                    help="drop an added box overlapping an already-added one (dedup crossarm splits)")
    ap.add_argument("--min-aspect", type=float, default=2.5,
                    help="drop SAM3 boxes that are not tall/thin (likely crossarm/blob)")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO
    from sam3_poles import Sam3Poles
    det = YOLO(args.weights) if args.mode == "consensus" else None
    sam3 = Sam3Poles()

    src_img = Path(args.src) / args.split / "images"
    src_lbl = Path(args.src) / args.split / "labels"
    out_img = Path(args.out) / args.split / "images"
    out_lbl = Path(args.out) / args.split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src_img / "*.jpg")))
    if args.n:
        files = files[:args.n]

    n_added = n_human = 0
    for fp in files:
        stem = Path(fp).stem
        im = Image.open(fp).convert("RGB"); w, h = im.size
        human = read_labels(src_lbl / (stem + ".txt"), w, h)
        n_human += len(human)
        sb, ss, _ = sam3.detect_segment(im, threshold=args.sam3_thr)
        # optional YOLO for consensus mode
        yb = []
        if args.mode == "consensus":
            r = det(fp, conf=args.yolo_conf, verbose=False)[0]
            yb = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])]
        added = []
        for i, sbox in enumerate(sb):
            bw = sbox[2] - sbox[0]; bh = sbox[3] - sbox[1]
            if bh / max(bw, 1) < args.min_aspect:      # keep tall/thin poles only
                continue
            if args.mode == "consensus" and not any(iou(sbox, ybox) >= args.agree_iou for ybox in yb):
                continue
            if any(iou(sbox, hb) >= args.dup_iou for hb in human):
                continue
            # NMS against already-added boxes (avoid stacked crossarm duplicates)
            if any(iou(sbox, ab) >= args.nms_iou for ab in added):
                continue
            added.append(sbox)
        n_added += len(added)
        shutil.copy2(fp, out_img / (stem + ".jpg"))
        lines = [to_yolo_line(b, w, h) for b in human] + \
                [to_yolo_line(b, w, h) for b in added]
        (out_lbl / (stem + ".txt")).write_text("\n".join(lines))

    print(f"[{args.split}] images={len(files)} human_boxes={n_human} "
          f"consensus_added={n_added} (+{100*n_added/max(n_human,1):.1f}%)")
    rep = Path(args.out) / "refine_report.jsonl"
    with open(rep, "a") as f:
        f.write(json.dumps({"split": args.split, "images": len(files),
                            "human": n_human, "added": n_added}) + "\n")


if __name__ == "__main__":
    main()
