"""SAM 3 text-promptable pole segmentation.

SAM 3 detects AND segments instances from an open-vocabulary text prompt in a
single model, so it can replace the whole YOLO-detector + SAM-mask pipeline:

    image + "utility pole"  ->  [ (box, mask, score), ... ]

Provides:
  Sam3Poles.detect_segment(image, threshold) -> boxes, scores, masks

Post-processing reuses the pole cleanup (largest vertical component + wire
suppression) from pole_sam for consistency with the other backend.
"""
from __future__ import annotations
import pole_env  # noqa: F401  (forces HF offline; robust to token issues)
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np

from pole_sam import _clean

DEFAULT_PROMPT = "utility pole"


class Sam3Poles:
    def __init__(self, model_id="facebook/sam3", device=None, prompt=DEFAULT_PROMPT):
        import torch
        from transformers import Sam3Processor, Sam3Model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.prompt = prompt

    def detect_segment(self, pil_image, threshold=0.4, mask_threshold=0.5,
                       min_area=60, clean=True):
        inp = self.proc(images=pil_image, text=self.prompt,
                        return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_instance_segmentation(
            out, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=[pil_image.size[::-1]])[0]
        masks = res.get("masks")
        scores = res.get("scores")
        boxes_out = res.get("boxes")
        out_boxes, out_scores, out_masks = [], [], []
        if masks is None or len(masks) == 0:
            return out_boxes, out_scores, out_masks
        for i in range(len(masks)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            m = (m > 0).astype(np.uint8)
            if m.sum() < min_area:
                continue
            ys, xs = np.where(m > 0)
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            if clean:
                m = _clean(m, box)
                if m.sum() < min_area:
                    continue
                ys, xs = np.where(m > 0)
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            sc = float(scores[i]) if scores is not None else 1.0
            out_boxes.append(box)
            out_scores.append(sc)
            out_masks.append(m)
        return out_boxes, out_scores, out_masks
