"""Mask-quality benchmark for the detect->SAM pole pipeline.

No pixel ground-truth exists, so we score masks with proxy metrics that a good
pole mask should satisfy:

  vertical_coverage : mask vertical extent / box height   (want high, ~1.0)
  box_leakage       : fraction of mask pixels outside the box (want ~0)
  straightness      : 1 - (mean horizontal deviation of the mask centerline /
                      box width). A straight vertical shaft scores ~1.0; a
                      blobby mask that wanders scores low.
  fill_ratio        : mask area / box area (poles are thin: want moderate, not
                      near-1 which means the whole box got filled)
  empty_rate        : fraction of detections that produced no usable mask

Aggregated over a fixed sample so configs can be compared apples-to-apples.
"""
from __future__ import annotations
import os, argparse, glob, json
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
from pathlib import Path


def mask_metrics(mask, box):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    x0, y0, x1, y1 = box
    bh = max(y1 - y0, 1); bw = max(x1 - x0, 1)
    v_cov = min((ys.max() - ys.min() + 1) / bh, 1.0)
    inb = ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).sum()
    leak = 1 - inb / len(xs)
    # straightness: per-row centroid x, deviation from its mean, normalized by box width
    rows = {}
    for x, y in zip(xs, ys):
        rows.setdefault(y, []).append(x)
    cxs = np.array([np.mean(v) for v in rows.values()])
    straight = max(0.0, 1.0 - (np.std(cxs) / bw))
    fill = (mask > 0).sum() / (bw * bh)
    return dict(v_cov=v_cov, leak=leak, straight=straight, fill=fill)


def run(engine, files, det, conf=0.25):
    agg = {k: [] for k in ("v_cov", "leak", "straight", "fill")}
    n_box = n_empty = 0
    from PIL import Image
    for fp in files:
        r = det(fp, conf=conf, verbose=False)[0]
        boxes = [[float(v) for v in b.xyxy[0].tolist()] for b in (r.boxes or [])]
        if not boxes:
            continue
        pil = Image.open(fp).convert("RGB")
        masks = engine.segment(pil, boxes)
        for m, b in zip(masks, boxes):
            n_box += 1
            met = mask_metrics(m, b) if m is not None else None
            if met is None:
                n_empty += 1
                continue
            for k in agg:
                agg[k].append(met[k])
    out = {k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in agg.items()}
    out["empty_rate"] = round(n_empty / max(n_box, 1), 4)
    out["n_boxes"] = n_box
    # single composite: reward coverage+straightness, penalize leakage
    out["score"] = round(0.4 * out["v_cov"] + 0.4 * out["straight"]
                         + 0.2 * (1 - out["leak"]) - 0.3 * out["empty_rate"], 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/pole_detector_v2.pt")
    p.add_argument("--backend", default="sam2", choices=["sam2", "sam"])
    p.add_argument("--sam2-id", default="facebook/sam2.1-hiera-large")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    from ultralytics import YOLO
    from pole_sam import PoleSAM
    det = YOLO(args.weights)
    eng = PoleSAM(backend=args.backend, sam2_id=args.sam2_id)

    files = sorted(glob.glob("Dataset/*.jpg"))
    step = max(1, len(files) // args.n)
    files = files[::step][:args.n]

    res = run(eng, files, det, conf=args.conf)
    tag = args.tag or f"{args.backend}:{args.sam2_id.split('/')[-1] if args.backend=='sam2' else 'vit-h'}"
    res["tag"] = tag
    print(f"[{tag}] score={res['score']}  v_cov={res['v_cov']} straight={res['straight']} "
          f"leak={res['leak']} fill={res['fill']} empty={res['empty_rate']} n={res['n_boxes']}")
    with open("output/results/mask_bench.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
