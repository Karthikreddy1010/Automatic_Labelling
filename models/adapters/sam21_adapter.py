"""
models/adapters/sam21_adapter.py - SAM 2.1 Pole Segmentation Adapter
====================================================================

Wraps SAM 2.1 (facebook/sam2.1-hiera-base-plus / facebook/sam2.1-hiera-large)
with utility pole morphology cleaning and vertical centerline point prompts.
Reuses domain logic from pole_sam.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

from models.adapters.base import BaseSegmenter, ModelInfo, detect_hardware
from src.geometry_obb import mask_to_obb_corners

try:
    from pole_sam import _clean, _pole_score
except ImportError:
    # Fallback morphology helpers if pole_sam isn't in root
    import cv2
    def _clean(mask: np.ndarray, box=None) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, vert, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, vert, iterations=1)
        return m

    def _pole_score(mask: np.ndarray, box) -> float:
        return float(mask.sum())


class SAM21Adapter(BaseSegmenter):
    """
    Adapter for SAM 2.1 box-prompted segmentation.
    """

    def __init__(
        self,
        sam2_id: str = "facebook/sam2.1-hiera-base-plus",
        n_points: int = 5,
        use_negatives: bool = False
    ):
        self.sam2_id = sam2_id
        self.n_points = n_points
        self.use_negatives = use_negatives
        self.device = "cpu"
        self.proc = None
        self.model = None
        self._status: str = "not_loaded"
        self._error_msg: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

    def load(self, device: str = "AUTO") -> bool:
        try:
            import torch
            from transformers import Sam2Processor, Sam2Model

            resolved_dev, _ = detect_hardware(device)
            self.device = resolved_dev

            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            self.proc = Sam2Processor.from_pretrained(self.sam2_id)
            self.model = Sam2Model.from_pretrained(
                self.sam2_id
            ).to(torch.device(self.device)).eval()

            self._status = "ready"
            self._error_msg = None
            return True
        except Exception as e:
            self._status = "unavailable"
            self._error_msg = f"Failed to load SAM 2.1 ({self.sam2_id}): {str(e)}"
            return False

    def unload(self) -> None:
        self.proc = None
        self.model = None
        self._status = "not_loaded"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _box_gate(self, hw: Tuple[int, int], box: Tuple[float, float, float, float], pad_frac: float = 0.15) -> np.ndarray:
        h, w = hw
        x0, y0, x1, y1 = box
        pw = pad_frac * (x1 - x0)
        ph = pad_frac * (y1 - y0)
        gx0 = int(max(0, x0 - pw))
        gy0 = int(max(0, y0 - ph))
        gx1 = int(min(w, x1 + pw))
        gy1 = int(min(h, y1 + ph))
        g = np.zeros((h, w), dtype=np.uint8)
        g[gy0:gy1, gx0:gx1] = 1
        return g

    def segment_box(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """
        Segment a single bounding box prompt. Returns cleaned 2D uint8 mask (0 or 255).
        """
        if self.model is None or self.proc is None:
            ok = self.load(self.device)
            if not ok:
                return None

        import torch

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            return None

        input_box = [float(v) for v in box_xyxy]
        try:
            inp = self.proc(
                images=pil_img,
                input_boxes=[[input_box]],
                return_tensors="pt"
            ).to(torch.device(self.device))

            with torch.no_grad():
                out = self.model(**inp)

            masks = self.proc.post_process_masks(
                out.pred_masks.cpu(), inp["original_sizes"].cpu()
            )[0]
            arr = masks.numpy()  # shape (1, C, H, W) or (C, H, W)
            if arr.ndim == 3:
                arr = arr[None, :]

            gate = self._box_gate(arr.shape[-2:], input_box, pad_frac=0.15)
            cands = []
            for c in range(arr.shape[1]):
                m = (arr[0, c] > 0).astype(np.uint8) * gate
                score = _pole_score(m, input_box)
                cands.append((score, m))

            _, best_m = max(cands, key=lambda t: t[0])
            cleaned_m = _clean(best_m, input_box)
            return (cleaned_m * 255).astype(np.uint8)
        except Exception as e:
            self._error_msg = f"SAM 2.1 segmentation error: {str(e)}"
            return None

    def segment_and_generate_obb(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box_xyxy: Tuple[float, float, float, float]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segment box and convert mask directly into canonical 4-corner OBB.
        Returns: (mask_uint8, canonical_corners)
        """
        mask = self.segment_box(image, box_xyxy)
        if mask is None or mask.sum() == 0:
            return None, None
        corners = mask_to_obb_corners(mask)
        return mask, corners

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="sam21_segmenter",
            name="SAM 2.1 Pole Segmenter",
            model_type="segmenter",
            status=self._status,
            weights_path=self.sam2_id,
            device=self.device,
            error_message=self._error_msg,
            installation_guide=(
                f"Requires Hugging Face model '{self.sam2_id}'. "
                "Requires transformers >= 4.45.0. Set POLE_ALLOW_ONLINE=1 to download weights."
            ),
        )
