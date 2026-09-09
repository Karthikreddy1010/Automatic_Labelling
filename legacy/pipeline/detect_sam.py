"""Detect poles with a YOLO detector, then segment each detection with SAM.

Pipeline:
    YOLO detector (trained on clean human boxes)  ->  pole bounding boxes
    SAM (ViT-H, box-prompted)                     ->  pixel-accurate mask per box

This keeps detection supervised by clean human labels while getting
high-quality masks from a dedicated segmenter, without ever training on
noisy auto-masks.

Usage:
    python detect_sam.py --weights runs/detect/.../best.pt --images Dataset --n 12 \
        --out output/detsam --montage
"""

from __future__ import annotations
import pole_env  # noqa: F401  (forces HF offline; robust to token issues)
import os, json, argparse, glob
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import cv2


def find_best(project="runs/detect"):
    c = sorted(Path(project).rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(c[-1]) if c else None


class DetectSAM:
    def __init__(self, weights, sam_id="facebook/sam-vit-huge", device=None):
        import torch
        from ultralytics import YOLO
        from pole_sam import PoleSAM
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.det = YOLO(weights)
        self.pole_sam = PoleSAM(sam_id=sam_id, device=str(self.device))

    def detect(self, image_path, conf=0.25, imgsz=640, iou=0.5):
        r = self.det(image_path, conf=conf, imgsz=imgsz, iou=iou, verbose=False)[0]
        boxes, scores = [], []
        if r.boxes is not None:
            for b in r.boxes:
                boxes.append([float(v) for v in b.xyxy[0].tolist()])
                scores.append(float(b.conf[0]))
        return boxes, scores

    def segment(self, pil_image, boxes):
        """One clean pole mask per box (improved PoleSAM: points + gating + cleanup)."""
        return self.pole_sam.segment(pil_image, boxes)

    def run_image(self, image_path, conf=0.25, imgsz=640):
        from PIL import Image
        boxes, scores = self.detect(image_path, conf=conf, imgsz=imgsz)
        pil = Image.open(image_path).convert("RGB")
        masks = self.segment(pil, boxes) if boxes else []
        return boxes, scores, masks

    @staticmethod
    def overlay(image_path, boxes, scores, masks, out_path):
        img = cv2.imread(image_path)
        colors = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            if i < len(masks) and masks[i] is not None:
                m = masks[i].astype(bool)
                overlay = img.copy()
                overlay[m] = color
                img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, f"{scores[i]:.2f}", (x0 + 2, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(out_path, img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--images", default="Dataset")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="output/detsam")
    p.add_argument("--montage", action="store_true")
    args = p.parse_args()

    weights = args.weights or find_best()
    if not weights:
        raise SystemExit("No detector weights found. Train train_detector.py first.")
    print(f"Detector: {weights}")
    engine = DetectSAM(weights)

    files = sorted(glob.glob(str(Path(args.images) / "*.jpg")))
    if args.n:
        step = max(1, len(files) // args.n)
        files = files[::step][:args.n]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    saved, total = [], 0
    for fp in files:
        boxes, scores, masks = engine.run_image(fp, conf=args.conf, imgsz=args.imgsz)
        total += len(boxes)
        dst = out / Path(fp).name
        engine.overlay(fp, boxes, scores, masks, str(dst))
        saved.append(str(dst))
    print(f"{len(files)} images, {total} poles detected+segmented -> {out}")

    if args.montage and saved:
        from PIL import Image
        import math
        ims = [Image.open(s).convert("RGB").resize((320, 320)) for s in saved[:16]]
        cols = 4; rows = math.ceil(len(ims) / cols)
        m = Image.new("RGB", (cols * 320, rows * 320), (20, 20, 20))
        for i, im in enumerate(ims):
            m.paste(im, ((i % cols) * 320, (i // cols) * 320))
        m.save(out / "montage.jpg")
        print(f"montage -> {out / 'montage.jpg'}")


if __name__ == "__main__":
    main()
