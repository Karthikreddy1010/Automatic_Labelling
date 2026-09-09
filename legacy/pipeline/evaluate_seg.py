"""Evaluate the trained pole segmentation model on the held-out val split.

Reports REAL detection + segmentation metrics from ultralytics (not the
synthetic placeholder numbers that were in the old evaluation_report.txt).

Usage:
    python evaluate_seg.py --weights runs/.../weights/best.pt \
        --data data/processed/yolo_dataset/dataset.yaml
"""

from __future__ import annotations

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_seg")


def find_best_weights(project="runs/segment") -> str | None:
    cands = sorted(Path(project).rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--data", default="data/processed/yolo_dataset/dataset.yaml")
    p.add_argument("--out", default="output/results/evaluation_report.txt")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    from ultralytics import YOLO

    weights = args.weights or find_best_weights()
    if not weights or not Path(weights).exists():
        raise SystemExit("No weights found. Train first (train_seg.py).")
    logger.info(f"Evaluating {weights} on {args.data}")

    model = YOLO(weights)
    m = model.val(data=args.data, batch=args.batch, workers=0, plots=True, verbose=False)

    box = m.box   # detection metrics
    seg = m.seg   # segmentation metrics

    report = {
        "weights": weights,
        "data": args.data,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "detection_box": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
        "segmentation_mask": {
            "mAP50": round(float(seg.map50), 4),
            "mAP50_95": round(float(seg.map), 4),
            "precision": round(float(seg.mp), 4),
            "recall": round(float(seg.mr), 4),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_format(report))
    (out.parent / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    print(_format(report))
    logger.info(f"Report -> {out}")


def _format(r: dict) -> str:
    b, s = r["detection_box"], r["segmentation_mask"]
    return f"""
============================================================
  UTILITY POLE DETECTION - REAL EVALUATION (VAL SPLIT)
============================================================
weights: {r['weights']}
data:    {r['data']}
time:    {r['timestamp']}

DETECTION (bounding box):
  mAP@50:      {b['mAP50']:.4f}
  mAP@50-95:   {b['mAP50_95']:.4f}
  Precision:   {b['precision']:.4f}
  Recall:      {b['recall']:.4f}

SEGMENTATION (mask):
  mAP@50:      {s['mAP50']:.4f}
  mAP@50-95:   {s['mAP50_95']:.4f}
  Precision:   {s['precision']:.4f}
  Recall:      {s['recall']:.4f}

NOTE: labels are auto-generated (Grounding DINO + SAM), so these
metrics measure agreement with pseudo-labels, not human ground truth.
For production numbers, evaluate on a small hand-labeled test set.
============================================================
""".strip()


if __name__ == "__main__":
    main()
