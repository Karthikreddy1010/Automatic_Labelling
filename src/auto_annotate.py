"""Open-vocabulary auto-annotation for utility poles.

Replaces the previous COCO-based labeling (which mislabeled cars/trucks as
"utility_pole") with a genuine open-vocabulary pipeline:

    Grounding DINO  ->  pole bounding boxes  (open-vocab, no training needed)
    SAM (ViT-H)     ->  pixel-accurate masks ->  YOLO-seg polygons

Outputs a YOLO segmentation dataset (images + polygon label files) plus a
manifest JSON and QA overlays. The step is resumable: images whose label file
already exists are skipped.

Usage:
    python -m src.auto_annotate --images Dataset --out data/processed/yolo_dataset \
        --box-threshold 0.30 --limit 0
"""

from __future__ import annotations

import os
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import cv2
import numpy as np

logger = logging.getLogger("auto_annotate")

POLE_PROMPT = "utility pole. power pole. light pole. telephone pole. street light pole."
CLASS_NAME = "utility_pole"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes: List[np.ndarray], scores: List[float], iou_thr: float = 0.5) -> List[int]:
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou_xyxy(boxes[i], boxes[j]) < iou_thr]
    return keep


def mask_to_polygons(mask: np.ndarray, img_w: int, img_h: int,
                     min_area_frac: float = 0.002) -> List[List[float]]:
    """Convert a binary mask to normalized YOLO-seg polygons (largest contours)."""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    min_area = min_area_frac * img_w * img_h
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        poly = []
        for x, y in approx:
            poly.append(float(np.clip(x / img_w, 0.0, 1.0)))
            poly.append(float(np.clip(y / img_h, 0.0, 1.0)))
        polys.append(poly)
    return polys


class OpenVocabAnnotator:
    def __init__(self, device: Optional[str] = None,
                 dino_id: str = "IDEA-Research/grounding-dino-base",
                 sam_id: str = "facebook/sam-vit-huge",
                 use_sam: bool = True):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_sam = use_sam
        logger.info(f"Loading Grounding DINO ({dino_id}) on {self.device}")
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).eval()

        self.sam = None
        if use_sam:
            from transformers import SamModel
            logger.info(f"Loading SAM ({sam_id}) on {self.device}")
            self.sam_proc = AutoProcessor.from_pretrained(sam_id)
            self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    def detect(self, image, box_threshold: float, text_threshold: float):
        from PIL import Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        inp = self.dino_proc(images=image, text=POLE_PROMPT, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.dino(**inp)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
        boxes = [np.array(b, dtype=np.float32) for b in res["boxes"].tolist()]
        scores = [float(s) for s in res["scores"].tolist()]
        return image, boxes, scores

    def segment(self, image, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Return one binary mask per box using SAM box prompts."""
        if not boxes:
            return []
        input_boxes = [[[float(v) for v in b.tolist()] for b in boxes]]
        inp = self.sam_proc(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.sam(**inp)
        masks = self.sam_proc.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0]  # (num_boxes, C, H, W)
        iou_scores = out.iou_scores.cpu().numpy()[0]  # (num_boxes, C)
        result = []
        arr = masks.numpy()
        for i in range(arr.shape[0]):
            best_c = int(np.argmax(iou_scores[i]))
            result.append((arr[i, best_c] > 0).astype(np.uint8))
        return result

    def annotate_image(self, img_path: str, box_threshold: float,
                       text_threshold: float, nms_iou: float):
        image, boxes, scores = self.detect(img_path, box_threshold, text_threshold)
        w, h = image.size
        keep = nms(boxes, scores, nms_iou) if boxes else []
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]

        label_lines: List[str] = []
        records = []
        if boxes and self.use_sam:
            masks = self.segment(image, boxes)
            for box, score, mask in zip(boxes, scores, masks):
                polys = mask_to_polygons(mask, w, h)
                if not polys:
                    polys = [self._box_polygon(box, w, h)]
                # keep largest polygon only per instance
                poly = max(polys, key=lambda p: len(p))
                label_lines.append("0 " + " ".join(f"{v:.6f}" for v in poly))
                records.append({"box_xyxy": [round(float(v), 1) for v in box.tolist()],
                                "score": round(score, 4), "n_poly_pts": len(poly) // 2})
        else:
            for box, score in zip(boxes, scores):
                poly = self._box_polygon(box, w, h)
                label_lines.append("0 " + " ".join(f"{v:.6f}" for v in poly))
                records.append({"box_xyxy": [round(float(v), 1) for v in box.tolist()],
                                "score": round(score, 4), "n_poly_pts": 4})
        return image, label_lines, records

    @staticmethod
    def _box_polygon(box: np.ndarray, w: int, h: int) -> List[float]:
        x0, y0, x1, y1 = box
        x0, x1 = np.clip([x0, x1], 0, w) / w
        y0, y1 = np.clip([y0, y1], 0, h) / h
        return [float(x0), float(y0), float(x1), float(y0),
                float(x1), float(y1), float(x0), float(y1)]


def run(images_dir: str, out_dir: str, box_threshold: float = 0.30,
        text_threshold: float = 0.25, nms_iou: float = 0.5, val_frac: float = 0.15,
        limit: int = 0, use_sam: bool = True, seed: int = 42, qa_every: int = 200):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(out_dir)
    files = sorted(glob.glob(str(Path(images_dir) / "*.jpg")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .jpg images in {images_dir}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(val_frac * len(files))
    val_set = set(idx[:n_val].tolist())
    split_of = {i: ("val" if i in val_set else "train") for i in range(len(files))}

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    qa_dir = out / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    ann = OpenVocabAnnotator(use_sam=use_sam)
    manifest = []
    n_boxes = 0
    n_empty = 0
    for k, fp in enumerate(files):
        split = split_of[k]
        stem = Path(fp).stem
        img_out = out / split / "images" / (stem + ".jpg")
        lbl_out = out / split / "labels" / (stem + ".txt")
        if lbl_out.exists() and img_out.exists():
            continue
        try:
            image, lines, records = ann.annotate_image(fp, box_threshold, text_threshold, nms_iou)
        except Exception as e:
            logger.warning(f"Failed {stem}: {e}")
            continue
        # copy image + write label (empty file => background image, valid for YOLO)
        image.save(img_out, quality=95)
        lbl_out.write_text("\n".join(lines))
        n_boxes += len(lines)
        if not lines:
            n_empty += 1
        manifest.append({"image": stem + ".jpg", "split": split, "n_instances": len(lines),
                         "instances": records})
        if qa_every and k % qa_every == 0:
            _save_qa(image, records, qa_dir / f"{stem}.jpg")
        if (k + 1) % 100 == 0:
            logger.info(f"[{k+1}/{len(files)}] boxes={n_boxes} empty={n_empty}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_yaml(out)
    logger.info(f"DONE. images={len(manifest)} total_instances={n_boxes} "
                f"empty_images={n_empty} -> {out}")


def _save_qa(image, records, path):
    from PIL import ImageDraw
    im = image.copy()
    d = ImageDraw.Draw(im)
    for r in records:
        x0, y0, x1, y1 = r["box_xyxy"]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        d.text((x0 + 2, y0 + 2), f"{r['score']:.2f}", fill=(255, 255, 0))
    im.save(path)


def _write_yaml(out: Path):
    import yaml
    cfg = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: CLASS_NAME},
        "nc": 1,
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    p = argparse.ArgumentParser(description="Open-vocab pole auto-annotation (DINO + SAM)")
    p.add_argument("--images", default="Dataset")
    p.add_argument("--out", default="data/processed/yolo_dataset")
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-sam", action="store_true", help="boxes only, skip SAM masks")
    args = p.parse_args()
    run(args.images, args.out, args.box_threshold, args.text_threshold,
        args.nms_iou, args.val_frac, args.limit, use_sam=not args.no_sam)


if __name__ == "__main__":
    main()
