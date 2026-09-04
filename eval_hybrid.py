"""Evaluate YOLO, SAM3, and fusions on the human-labeled Final_Dataset test split.

All scored against the SAME human ground-truth boxes (502 poles / 478 imgs),
so precision/recall/F1 are directly comparable across methods:

  yolo         : YOLO26m detector only
  sam3         : SAM 3 text-prompt only
  union        : YOLO boxes + SAM3 boxes, deduplicated by NMS
  intersection : only poles both models agree on (IoU match) -> high precision

This is the honest test for "does combining help".
"""
from __future__ import annotations
import os, glob, json, argparse
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


def nms(boxes, scores, thr=0.5):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < thr]
    return keep


def match(gt, pred, thr=0.5):
    """pred = list of [x0,y0,x1,y1,score]. Greedy IoU match -> TP,FP,FN."""
    used = [False] * len(gt); tp = 0
    for pb in sorted(pred, key=lambda x: -x[4]):
        best, bi = 0, -1
        for i, gb in enumerate(gt):
            if used[i]:
                continue
            v = iou(pb[:4], gb)
            if v > best:
                best, bi = v, i
        if best >= thr and bi >= 0:
            used[bi] = True; tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def load_gt(split_dir):
    """Read YOLO-format human labels -> {image_path: [xyxy,...]} in abs px."""
    from PIL import Image
    gt = {}
    for lp in glob.glob(str(Path(split_dir) / "labels" / "*.txt")):
        stem = Path(lp).stem
        ip = Path(split_dir) / "images" / (stem + ".jpg")
        if not ip.exists():
            continue
        w, h = Image.open(ip).size
        boxes = []
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = [float(x) for x in p[1:5]]
            boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
        gt[str(ip)] = boxes
    return gt


def prf(TP, FP, FN):
    P = TP/(TP+FP) if TP+FP else 0
    R = TP/(TP+FN) if TP+FN else 0
    F = 2*P*R/(P+R) if P+R else 0
    return round(P, 4), round(R, 4), round(F, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Final_Dataset/test")
    ap.add_argument("--weights", default="models/pole_detector_v2.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--sam3-thr", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO
    from sam3_poles import Sam3Poles

    gt = load_gt(args.split)
    files = sorted(gt.keys())
    if args.n:
        files = files[:args.n]
    det = YOLO(args.weights)
    sam3 = Sam3Poles()

    acc = {k: [0, 0, 0] for k in ("yolo", "sam3", "union", "intersection")}
    for fp in files:
        g = gt[fp]
        # YOLO
        r = det(fp, conf=args.yolo_conf, verbose=False)[0]
        yb = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])]
        ys = [float(b.conf[0]) for b in (r.boxes or [])]
        # SAM3
        im = Image.open(fp).convert("RGB")
        sb, ss, _ = sam3.detect_segment(im, threshold=args.sam3_thr)

        preds = {
            "yolo": [yb[i] + [ys[i]] for i in range(len(yb))],
            "sam3": [sb[i] + [ss[i]] for i in range(len(sb))],
        }
        # union (dedup by NMS across both)
        ub = yb + sb; us = ys + ss
        keep = nms(ub, us, 0.5)
        preds["union"] = [ub[i] + [us[i]] for i in keep]
        # intersection: YOLO boxes that also have a SAM3 match (IoU>=0.5)
        inter = []
        for i, b in enumerate(yb):
            if any(iou(b, s) >= 0.5 for s in sb):
                inter.append(b + [ys[i]])
        preds["intersection"] = inter

        for k, pr in preds.items():
            tp, fp, fn = match(g, pr, 0.5)
            acc[k][0] += tp; acc[k][1] += fp; acc[k][2] += fn

    print(f"Test split: {len(files)} images, {sum(len(v) for v in gt.values())} human poles\n")
    print(f"{'method':13s} {'P':>7s} {'R':>7s} {'F1':>7s}   TP/FP/FN")
    results = {}
    for k, (TP, FP, FN) in acc.items():
        P, R, F = prf(TP, FP, FN)
        results[k] = {"P": P, "R": R, "F1": F, "TP": TP, "FP": FP, "FN": FN}
        print(f"{k:13s} {P:7.3f} {R:7.3f} {F:7.3f}   {TP}/{FP}/{FN}")
    Path("output/results").mkdir(parents=True, exist_ok=True)
    json.dump(results, open("output/results/hybrid_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
