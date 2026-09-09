"""Honest full-test scorer + failure dump for a YOLO pole detector.

Scores a detector against human GT (YOLO-format labels) on the 478-img test
split: precision / recall / F1 at IoU>=0.5, and writes the exact FN (missed)
and FP (spurious) boxes per image so a retrain can target real failure modes
instead of guessing at hyperparameters.

Usage:
    python eval_detector.py --weights models/pole_detector_refined.pt --conf 0.25
"""
from __future__ import annotations
import os, json, argparse, glob
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path
from PIL import Image


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-9)


def load_gt(label_path, W, H):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for ln in open(label_path):
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        boxes.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
    return boxes


def match(gt, pred, thr=0.5):
    """Greedy IoU match. pred sorted by conf desc. Returns matched-gt idx set,
    matched-pred idx set."""
    used_g, used_p = set(), set()
    order = sorted(range(len(pred)), key=lambda i: -pred[i][4])
    for pi in order:
        best, bi = thr, -1
        for gi, gb in enumerate(gt):
            if gi in used_g:
                continue
            v = iou(pred[pi][:4], gb)
            if v >= best:
                best, bi = v, gi
        if bi >= 0:
            used_g.add(bi); used_p.add(pi)
    return used_g, used_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/pole_detector_v2.pt")
    ap.add_argument("--images", default="Final_Dataset_refined/test/images")
    ap.add_argument("--labels", default="Final_Dataset_refined/test/labels")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--out", default="output/gold/detector_eval.json")
    ap.add_argument("--stems", default=None, help="file of stems; score only these")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                  glob.glob(os.path.join(args.images, "*.png")))
    if args.stems:
        keep = set(l.strip() for l in open(args.stems) if l.strip())
        imgs = [p for p in imgs if Path(p).stem in keep]
    TP = FP = FN = 0
    fn_imgs, fp_imgs = [], []
    for ip in imgs:
        stem = Path(ip).stem
        W, H = Image.open(ip).size
        gt = load_gt(os.path.join(args.labels, stem + ".txt"), W, H)
        r = model(ip, conf=args.conf, imgsz=args.imgsz, iou=args.iou_nms, verbose=False)[0]
        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
                preds.append([x0, y0, x1, y1, float(b.conf[0])])
        mg, mp = match(gt, preds, args.iou_match)
        tp = len(mg); fp = len(preds) - len(mp); fn = len(gt) - len(mg)
        TP += tp; FP += fp; FN += fn
        if fn:
            fn_imgs.append({"img": stem, "missed": fn, "gt": len(gt), "pred": len(preds)})
        if fp:
            fp_imgs.append({"img": stem, "extra": fp, "gt": len(gt), "pred": len(preds),
                            "confs": sorted([round(preds[i][4], 3) for i in range(len(preds)) if i not in mp], reverse=True)})

    P = TP/(TP+FP) if TP+FP else 0
    R = TP/(TP+FN) if TP+FN else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    res = {"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
           "TP": TP, "FP": FP, "FN": FN,
           "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
           "fn_imgs": fn_imgs, "fp_imgs": fp_imgs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"{args.weights}  conf{args.conf} sz{args.imgsz}")
    print(f"  TP={TP} FP={FP} FN={FN}  P={P:.3f} R={R:.3f} F1={F1:.3f}")
    print(f"  {len(fn_imgs)} imgs w/ misses, {len(fp_imgs)} imgs w/ extras -> {args.out}")


if __name__ == "__main__":
    main()
