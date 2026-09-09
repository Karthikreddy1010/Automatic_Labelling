"""Gold-set scorer for the pole detector.

The gold ground-truth boxes are the human-verified detections plus manually
added misses (see build_gold_gt.py). This lets us auto-score any model/config
with IoU matching -> precision / recall / F1, so "improvement" is measured
against human judgment, not against the Grounding DINO pseudo-labels.

Usage:
    python score_gold.py --weights models/pole_seg_best.pt --conf 0.25 --imgsz 640
"""

from __future__ import annotations
import os, json, argparse, glob
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def match(gt_boxes, pred_boxes, iou_thr=0.5):
    """Greedy one-to-one matching by IoU. Returns TP, FP, FN."""
    used = [False] * len(gt_boxes)
    tp = 0
    for pb in sorted(pred_boxes, key=lambda x: -x[4]):  # by conf desc
        best, bi = 0.0, -1
        for i, gb in enumerate(gt_boxes):
            if used[i]:
                continue
            v = iou(pb[:4], gb)
            if v > best:
                best, bi = v, i
        if best >= iou_thr and bi >= 0:
            used[bi] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/pole_seg_best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou-nms", type=float, default=0.5, help="prediction NMS IoU")
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--iou-match", type=float, default=0.5, help="TP match IoU")
    p.add_argument("--gt", default="output/gold/gold_gt.json")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    from ultralytics import YOLO
    gt = json.load(open(args.gt))
    model = YOLO(args.weights)

    TP = FP = FN = 0
    per_img = {}
    for name, gt_boxes in gt.items():
        fp_path = f"Dataset/{name}"
        r = model(fp_path, conf=args.conf, imgsz=args.imgsz, iou=args.iou_nms,
                  agnostic_nms=args.agnostic_nms, verbose=False)[0]
        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
                preds.append([x0, y0, x1, y1, float(b.conf[0])])
        tp, fp, fn = match(gt_boxes, preds, args.iou_match)
        TP += tp; FP += fp; FN += fn
        per_img[name] = {"gt": len(gt_boxes), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn}

    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    out = {"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
           "iou_nms": args.iou_nms, "agnostic_nms": args.agnostic_nms,
           "TP": TP, "FP": FP, "FN": FN,
           "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    tag = args.tag or f"conf{args.conf}_sz{args.imgsz}_nms{args.iou_nms}{'_agn' if args.agnostic_nms else ''}"
    print(f"[{tag}] TP={TP} FP={FP} FN={FN}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    log_path = Path("output/gold/scores.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"tag": tag, **out}) + "\n")
    return out


if __name__ == "__main__":
    main()
