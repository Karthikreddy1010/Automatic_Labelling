"""
models/adapters/yolo_adapter.py - YOLO & YOLO-OBB Detector Adapter
==================================================================

Wraps Ultralytics YOLO / YOLO-OBB models. Handles dynamic weight loading,
automatic CPU/CUDA device management, axis-aligned and oriented bounding box
prediction, and canonical OBB corner serialization.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

from models.adapters.base import BaseDetector, DetectionBox, ModelInfo, detect_hardware
from src.geometry_obb import (
    xyxy_to_obb_corners,
    order_corners_canonical,
    obb_corners_to_xyxy,
    signed_shoelace_area,
)


class YOLOAdapter(BaseDetector):
    """
    Adapter for Ultralytics YOLO & YOLO-OBB models.
    """

    DEFAULT_CANDIDATE_PATHS = [
        "models/yolo11n-obb.pt",
        "models/yolo11n.pt",
        "models/yolov8n-obb.pt",
        "models/yolov8n.pt",
        "models/best.pt",
        "runs/detect/train/weights/best.pt",
        "yolo11n.pt",
        "yolov8n.pt",
    ]

    def __init__(self, weights_path: Optional[str] = None, model_name: str = "YOLO Detector"):
        self.weights_path = weights_path
        self.model_name = model_name
        self.model = None
        self.device = "cpu"
        self._resolved_weights: Optional[str] = None
        self._is_obb: bool = False
        self._status: str = "not_loaded"
        self._error_msg: Optional[str] = None
        
        self._check_weights()

    def _check_weights(self) -> None:
        """Locate weights or determine missing status."""
        if self.weights_path:
            if os.path.exists(self.weights_path):
                self._resolved_weights = os.path.abspath(self.weights_path)
                self._status = "not_loaded"
                self._is_obb = "obb" in Path(self._resolved_weights).stem.lower()
            else:
                self._status = "missing_weights"
                self._resolved_weights = None
            return

        for cand in self.DEFAULT_CANDIDATE_PATHS:
            if os.path.exists(cand):
                self._resolved_weights = os.path.abspath(cand)
                self._status = "not_loaded"
                self._is_obb = "obb" in Path(self._resolved_weights).stem.lower()
                return

        self._status = "missing_weights"
        self._resolved_weights = None

    def is_available(self) -> bool:
        try:
            import ultralytics
            self._check_weights()
            return self._resolved_weights is not None
        except ImportError:
            return False

    def load(self, device: str = "AUTO") -> bool:
        self._check_weights()
        if not self._resolved_weights:
            self._status = "missing_weights"
            self._error_msg = (
                f"YOLO weights not found. Place a model in 'models/yolo11n.pt' or "
                f"'models/yolo11n-obb.pt', or pass a valid weights_path."
            )
            return False

        try:
            from ultralytics import YOLO
            resolved_dev, _ = detect_hardware(device)
            self.device = resolved_dev
            self.model = YOLO(self._resolved_weights)
            # The loaded model's own task is authoritative; the filename-based
            # guess in _check_weights() can be wrong for a trained model that
            # doesn't happen to have "obb" in its filename (e.g. "best.pt").
            self._is_obb = getattr(self.model, "task", None) == "obb"
            self._status = "ready"
            self._error_msg = None
            return True
        except Exception as e:
            self._status = "unavailable"
            self._error_msg = f"Failed to load YOLO model: {str(e)}"
            return False

    def unload(self) -> None:
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
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: int = 1280
    ) -> List[DetectionBox]:
        if self.model is None:
            ok = self.load(self.device)
            if not ok:
                return []

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        if not results:
            return []

        res = results[0]
        boxes: List[DetectionBox] = []

        # Check if YOLO-OBB results are present
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            obb_data = res.obb
            # xyxyxyxy shape: (N, 4, 2)
            corners_all = obb_data.xyxyxyxy.cpu().numpy()
            confs = obb_data.conf.cpu().numpy()
            classes = obb_data.cls.cpu().numpy()

            for i in range(len(confs)):
                raw_corners = corners_all[i]
                canonical_corners = order_corners_canonical(raw_corners)
                min_x, min_y, max_x, max_y = obb_corners_to_xyxy(canonical_corners)
                conf = float(confs[i])
                cls_id = int(classes[i])
                cls_name = res.names.get(cls_id, "utility_pole") if hasattr(res, "names") else "utility_pole"

                # Review flagging: confidence < 0.5 flagged as needs_review
                needs_review = conf < 0.5
                review_reasons = ["Low confidence (<0.5)"] if needs_review else []

                box = DetectionBox(
                    xyxy=(min_x, min_y, max_x, max_y),
                    corners=canonical_corners.tolist(),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    model_source="YOLO-OBB",
                    needs_review=needs_review,
                    review_reasons=review_reasons,
                )
                boxes.append(box)

        elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            raw_boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()

            for i in range(len(confs)):
                x1, y1, x2, y2 = [float(v) for v in raw_boxes[i]]
                canonical_corners = xyxy_to_obb_corners(x1, y1, x2, y2)
                conf = float(confs[i])
                cls_id = int(classes[i])
                cls_name = res.names.get(cls_id, "utility_pole") if hasattr(res, "names") else "utility_pole"

                needs_review = conf < 0.5
                review_reasons = ["Low confidence (<0.5)"] if needs_review else []

                box = DetectionBox(
                    xyxy=(x1, y1, x2, y2),
                    corners=canonical_corners.tolist(),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    model_source="YOLO",
                    needs_review=needs_review,
                    review_reasons=review_reasons,
                )
                boxes.append(box)

        return boxes

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="yolo_detector",
            name=f"{self.model_name} ({'OBB' if self._is_obb else 'Standard'})",
            model_type="detector",
            status=self._status,
            weights_path=self._resolved_weights,
            device=self.device,
            error_message=self._error_msg,
            installation_guide=(
                "Download YOLO / YOLO-OBB model weights (e.g., yolo11n.pt or yolo11n-obb.pt) "
                "and place in 'models/' directory or specify weights path in configuration."
            ),
        )
