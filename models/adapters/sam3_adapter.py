"""
models/adapters/sam3_adapter.py - SAM 3 Text-Promptable Segmentation Adapter
============================================================================

Wraps SAM 3 (facebook/sam3) for joint open-vocabulary detection and segmentation.
Reuses domain logic from sam3_poles.py and pole_sam.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

from models.adapters.base import BaseSegmenter, DetectionBox, ModelInfo, detect_hardware
from src.geometry_obb import mask_to_obb_corners, obb_corners_to_xyxy

try:
    from pole_sam import _clean
except ImportError:
    import cv2
    def _clean(mask: np.ndarray, box=None) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, vert, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, vert, iterations=1)
        return m


class SAM3Adapter(BaseSegmenter):
    """
    Adapter for SAM 3 text-promptable instance segmentation & detection.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        text_prompt: str = "utility pole",
        threshold: float = 0.40,
        mask_threshold: float = 0.50
    ):
        self.model_id = model_id
        self.text_prompt = text_prompt
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = "cpu"
        self.proc = None
        self.model = None
        self._status: str = "not_loaded"
        self._error_msg: Optional[str] = None

    def is_available(self) -> bool:
        try:
            from transformers import Sam3Processor, Sam3Model
            return True
        except Exception:
            return False

    def load(self, device: str = "AUTO") -> bool:
        try:
            import torch
            from transformers import Sam3Processor, Sam3Model

            resolved_dev, _ = detect_hardware(device)
            self.device = resolved_dev

            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            self.proc = Sam3Processor.from_pretrained(self.model_id)
            self.model = Sam3Model.from_pretrained(
                self.model_id
            ).to(torch.device(self.device)).eval()

            self._status = "ready"
            self._error_msg = None
            return True
        except Exception as e:
            self._status = "unavailable"
            self._error_msg = f"Failed to load SAM 3 ({self.model_id}): {str(e)}"
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

    def detect_and_segment(
        self,
        image: Union[str, np.ndarray, Image.Image],
        threshold: Optional[float] = None,
        min_area: float = 60.0
    ) -> List[DetectionBox]:
        """
        Run open-vocabulary SAM 3 prompt detection and convert resulting masks to canonical OBBs.
        """
        if self.model is None or self.proc is None:
            ok = self.load(self.device)
            if not ok:
                return []

        import torch

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            return []

        thr = threshold if threshold is not None else self.threshold

        try:
            inp = self.proc(images=pil_img, text=self.text_prompt, return_tensors="pt").to(
                torch.device(self.device)
            )
            with torch.no_grad():
                out = self.model(**inp)

            res = self.proc.post_process_instance_segmentation(
                out,
                threshold=thr,
                mask_threshold=self.mask_threshold,
                target_sizes=[pil_img.size[::-1]]
            )[0]

            masks = res.get("masks")
            scores = res.get("scores")
            if masks is None or len(masks) == 0:
                return []

            detections: List[DetectionBox] = []
            for i in range(len(masks)):
                m = masks[i].cpu().numpy() if hasattr(masks[i], "cpu") else np.asarray(masks[i])
                binary_m = (m > 0).astype(np.uint8)
                if binary_m.sum() < min_area:
                    continue

                ys, xs = np.where(binary_m > 0)
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                cleaned_m = _clean(binary_m, box)
                if cleaned_m.sum() < min_area:
                    continue

                # Generate canonical OBB
                corners = mask_to_obb_corners(cleaned_m)
                if corners is None:
                    continue

                score = float(scores[i]) if scores is not None and i < len(scores) else 0.5
                needs_review = score < 0.5
                reasons = ["Low confidence SAM3 prediction"] if needs_review else []

                det = DetectionBox(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    corners=corners.tolist(),
                    confidence=score,
                    class_id=0,
                    class_name="utility_pole",
                    model_source="SAM3",
                    needs_review=needs_review,
                    review_reasons=reasons,
                    mask=cleaned_m,
                )
                detections.append(det)

            return detections
        except Exception as e:
            self._error_msg = f"SAM 3 inference error: {str(e)}"
            return []

    def segment_box(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """
        Part 1: SAM3 should refine/segment the CANDIDATE REGION a proposer (DINO)
        already found, not just trust a text prompt in isolation. Tries native
        box-prompted segmentation first (image + input_boxes, same calling
        convention as the SAM2 family this repo already uses in sam21_adapter.py)
        -- falls back to the existing crop-and-text-prompt approach if the
        installed Sam3Processor doesn't accept input_boxes (API differences
        across transformers versions), so this never hard-depends on an exact
        signature this session couldn't verify against a live HAWK install.

        Public BaseSegmenter-compatible signature (mask only) -- see
        segment_box_with_score() for the mask+confidence variant used by
        services/sam3_service.py.
        """
        mask, _score = self.segment_box_with_score(image, box_xyxy)
        return mask

    def segment_box_with_score(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box_xyxy: Tuple[float, float, float, float]
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Same box-prompted segmentation as segment_box(), but also returns a
        confidence score when the native box-prompt path produces one (SAM's
        mask decoder typically emits an IoU/quality score per predicted mask,
        e.g. `out.iou_scores`). Best-effort: this session could not verify
        the exact field name against a live SAM3 install, so a missing/
        differently-named field just yields score=None rather than raising --
        the mask itself is unaffected either way. The crop-based fallback
        path has no native score to report (score=None), consistent with
        segment_box()'s existing behavior there.
        """
        if self.model is None or self.proc is None:
            ok = self.load(self.device)
            if not ok:
                return None, None

        import torch

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            return None, None

        try:
            inp = self.proc(
                images=pil_img,
                input_boxes=[[list(box_xyxy)]],
                return_tensors="pt",
            ).to(torch.device(self.device))
            with torch.no_grad():
                out = self.model(**inp)
            masks = self.proc.post_process_masks(
                out.pred_masks, inp.get("original_sizes"), inp.get("reshaped_input_sizes"),
            )
            if masks and len(masks) > 0 and masks[0].numel() > 0:
                m = masks[0][0]
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                if m.ndim == 3:
                    m = m[0]
                cleaned = _clean((m > 0).astype(np.uint8), box_xyxy)

                score: Optional[float] = None
                iou_scores = getattr(out, "iou_scores", None)
                if iou_scores is not None:
                    try:
                        s = iou_scores[0][0]
                        s = s.max() if hasattr(s, "max") else s
                        score = float(s.cpu().item()) if hasattr(s, "cpu") else float(s)
                    except Exception:
                        score = None
                return cleaned, score
        except Exception:
            pass  # native box-prompting unsupported/failed -- fall back below

        # --- Fallback: crop to the box (padded) and re-run text-prompted detection ---
        w, h = pil_img.size
        x0, y0, x1, y1 = box_xyxy
        pad_w = 0.1 * (x1 - x0)
        pad_h = 0.1 * (y1 - y0)
        crop_box = (
            max(0, int(x0 - pad_w)),
            max(0, int(y0 - pad_h)),
            min(w, int(x1 + pad_w)),
            min(h, int(y1 + pad_h))
        )
        cropped = pil_img.crop(crop_box)
        dets = self.detect_and_segment(cropped)
        if not dets:
            return None, None

        # Return best match placed back onto full image canvas
        full_mask = np.zeros((h, w), dtype=np.uint8)
        # return mask of highest score detection
        best_det = max(dets, key=lambda d: d.confidence)
        if best_det.corners is not None:
            # Shift corners back to full image
            corners = np.array(best_det.corners)
            corners[:, 0] += crop_box[0]
            corners[:, 1] += crop_box[1]
            import cv2
            cv2.fillPoly(full_mask, [corners.astype(np.int32)], 255)
            return full_mask, best_det.confidence
        return None, None

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="sam3_segmenter",
            name="SAM 3 Open-Vocabulary Segmenter",
            model_type="segmenter",
            status=self._status,
            weights_path=self.model_id,
            device=self.device,
            backend="sam3_inprocess",
            error_message=self._error_msg,
            installation_guide=(
                f"Requires Hugging Face model '{self.model_id}' with transformers>=5.9 "
                "(Sam3Processor/Sam3Model). If this app's environment can't upgrade "
                "transformers safely, run services/sam3_service.py in a separate "
                "environment instead and set verification_pipeline.sam3.backend=http. "
                "Set POLE_ALLOW_ONLINE=1 to download."
            ),
        )
