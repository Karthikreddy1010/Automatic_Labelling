"""
models/adapters/dino_adapter.py - Grounding DINO Zero-Shot Detector Adapter
===========================================================================

Wraps Hugging Face Grounding DINO for open-vocabulary utility pole detection.
Reuses patterns from src/auto_annotate.py. Provides graceful missing-weights
handling and actionable offline guidance.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

from models.adapters.base import BaseDetector, DetectionBox, ModelInfo, detect_hardware
from src.geometry_obb import xyxy_to_obb_corners

DEFAULT_POLE_PROMPT = "utility pole. power pole. light pole. telephone pole."


class GroundingDINOAdapter(BaseDetector):
    """
    Adapter for Grounding DINO open-vocabulary zero-shot detector.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        text_prompt: str = DEFAULT_POLE_PROMPT,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25
    ):
        self.model_id = model_id
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = "cpu"
        self.proc = None
        self.model = None
        self._status: str = "not_loaded"
        self._error_msg: Optional[str] = None

    def set_prompts(self, prompts: Union[str, List[str]]) -> None:
        """Update Grounding DINO text prompt list."""
        if isinstance(prompts, list):
            cleaned = [p.strip().rstrip(".") for p in prompts if p.strip()]
            self.text_prompt = ". ".join(cleaned) + ("." if cleaned else "")
        else:
            self.text_prompt = prompts.strip()

    def is_available(self) -> bool:
        try:
            import transformers
            import torch
            # Check if cached or loadable
            return True
        except ImportError:
            return False

    def load(self, device: str = "AUTO") -> bool:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            resolved_dev, _ = detect_hardware(device)
            self.device = resolved_dev

            # Ensure HF progress bar doesn't mess up logs
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(torch.device(self.device)).eval()

            self._status = "ready"
            self._error_msg = None
            return True
        except Exception as e:
            self._status = "unavailable"
            self._error_msg = f"Failed to load Grounding DINO ({self.model_id}): {str(e)}"
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

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.5,
        imgsz: int = 1280
    ) -> List[DetectionBox]:
        if self.model is None or self.proc is None:
            ok = self.load(self.device)
            if not ok:
                return []

        import torch

        # Prepare PIL image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            # assume BGR from cv2
            rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            return []

        effective_box_thr = max(conf_threshold, self.box_threshold)

        try:
            inp = self.proc(images=pil_img, text=self.text_prompt, return_tensors="pt").to(
                torch.device(self.device)
            )
            with torch.no_grad():
                out = self.model(**inp)

            res = self.proc.post_process_grounded_object_detection(
                out,
                inp["input_ids"],
                threshold=effective_box_thr,
                text_threshold=self.text_threshold,
                target_sizes=[pil_img.size[::-1]]
            )[0]

            boxes_list = res.get("boxes", [])
            scores_list = res.get("scores", [])
            labels_list = res.get("text_labels") or res.get("labels", [])

            dets: List[DetectionBox] = []
            for i in range(len(scores_list)):
                box_vals = boxes_list[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = [float(v) for v in box_vals]
                score = float(scores_list[i])
                label = str(labels_list[i]) if i < len(labels_list) else "utility_pole"

                canonical_corners = xyxy_to_obb_corners(x1, y1, x2, y2)
                needs_review = score < 0.45
                reasons = ["Low confidence open-vocab detection (<0.45)"] if needs_review else []

                det = DetectionBox(
                    xyxy=(x1, y1, x2, y2),
                    corners=canonical_corners.tolist(),
                    confidence=score,
                    class_id=0,
                    class_name=label,
                    model_source="DINO",
                    needs_review=needs_review,
                    review_reasons=reasons,
                )
                dets.append(det)

            return dets
        except Exception as e:
            self._error_msg = f"Inference error in DINO: {str(e)}"
            return []

    def detect_candidate_boxes(
        self,
        image: Union[str, np.ndarray, Image.Image],
        conf_threshold: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Produce candidate bounding boxes (xyxy, confidence, label) without synthetic OBB.
        Used as box prompts for SAM segmentation.
        """
        dets = self.predict(image, conf_threshold=conf_threshold)
        candidates = []
        for d in dets:
            candidates.append({
                "xyxy": d.xyxy,
                "confidence": d.confidence,
                "label": d.class_name,
            })
        return candidates

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="grounding_dino",
            name="Grounding DINO (Open-Vocabulary)",
            model_type="detector",
            status=self._status,
            weights_path=self.model_id,
            device=self.device,
            error_message=self._error_msg,
            installation_guide=(
                f"Requires Hugging Face model '{self.model_id}'. "
                "Ensure internet access or pre-downloaded cache. Set POLE_ALLOW_ONLINE=1 if needed."
            ),
        )
