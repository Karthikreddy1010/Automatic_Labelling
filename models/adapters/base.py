"""
models/adapters/base.py - Base Model Interfaces & Data Classes
==============================================================

Provides unified abstractions, strict hardware inspection, and data structures
for YOLO, Grounding DINO, SAM 2.1, SAM 3, and VLM verifiers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch


@dataclass
class HardwareInfo:
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
        }


def detect_hardware(requested_device: str = "AUTO") -> Tuple[str, HardwareInfo]:
    """
    Inspect host hardware and resolve effective execution device.
    Strictly verifies CUDA without fake claims. If CUDA is requested but not
    available, falls back cleanly to CPU with accurate telemetry.
    """
    cuda_avail = torch.cuda.is_available()
    gpu_name = None
    vram_total = None
    vram_free = None

    if cuda_avail:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info()
            vram_total = int(total_b / (1024 * 1024))
            vram_free = int(free_b / (1024 * 1024))
        except Exception:
            gpu_name = "CUDA Device"

    req = requested_device.upper().strip()
    if req == "CUDA":
        resolved = "cuda" if cuda_avail else "cpu"
    elif req == "CPU":
        resolved = "cpu"
    else:  # AUTO
        resolved = "cuda" if cuda_avail else "cpu"

    info = HardwareInfo(
        device=resolved,
        cuda_available=cuda_avail,
        gpu_name=gpu_name if cuda_avail else None,
        vram_total_mb=vram_total if cuda_avail else None,
        vram_free_mb=vram_free if cuda_avail else None,
    )
    return resolved, info


@dataclass
class DetectionBox:
    """
    Standardized bounding representation across all model adapters and canvas UI.
    """
    xyxy: Tuple[float, float, float, float]
    corners: Optional[List[List[float]]] = None  # 4 canonical corners: [[x, y], ...]
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "utility_pole"
    model_source: str = "YOLO"  # "YOLO", "DINO", "SAM", "YOLO+DINO", "HUMAN"
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        d = {
            "xyxy": [float(v) for v in self.xyxy],
            "corners": [[float(pt[0]), float(pt[1])] for pt in self.corners] if self.corners is not None else None,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "model_source": self.model_source,
            "needs_review": bool(self.needs_review),
            "review_reasons": list(self.review_reasons),
            "attributes": dict(self.attributes),
        }
        if include_mask and self.mask is not None:
            # Mask is stored as binary base64 or RLE if requested, or omitted in light payloads
            d["has_mask"] = True
        else:
            d["has_mask"] = self.mask is not None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionBox:
        xyxy = tuple(float(v) for v in data["xyxy"])
        corners = data.get("corners")
        if corners is not None:
            corners = [[float(p[0]), float(p[1])] for p in corners]
        return cls(
            xyxy=xyxy,
            corners=corners,
            confidence=float(data.get("confidence", 1.0)),
            class_id=int(data.get("class_id", 0)),
            class_name=str(data.get("class_name", "utility_pole")),
            model_source=str(data.get("model_source", "HUMAN")),
            needs_review=bool(data.get("needs_review", False)),
            review_reasons=list(data.get("review_reasons", [])),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class ModelInfo:
    id: str
    name: str
    model_type: str  # "detector", "segmenter", "vlm", "ensemble"
    status: str      # "ready", "configured", "loading", "not_loaded", "unavailable", "error", "missing_weights"
    weights_path: Optional[str] = None
    device: str = "cpu"
    error_message: Optional[str] = None
    installation_guide: Optional[str] = None
    backend: Optional[str] = None  # e.g. "transformers", "sam3_http", "ollama:qwen3-vl:2b"
    dtype: Optional[str] = None  # e.g. "bfloat16", "float16", "float32"
    load_time_s: Optional[float] = None  # model load time in seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "status": self.status,
            "weights_path": self.weights_path,
            "device": self.device,
            "error_message": self.error_message,
            "installation_guide": self.installation_guide,
            "backend": self.backend,
            "dtype": self.dtype,
            "load_time_s": self.load_time_s,
        }


class BaseDetector(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model dependencies and weights are present."""
        pass

    @abstractmethod
    def load(self, device: str = "AUTO") -> bool:
        """Load model into memory on specified device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free GPU/CPU memory."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: int = 1280
    ) -> List[DetectionBox]:
        """Perform object detection returning standardized DetectionBox objects."""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return metadata, current status, and installation instructions."""
        pass


class BaseSegmenter(ABC):
    """Abstract interface for instance segmentation models."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model dependencies and weights are present."""
        pass

    @abstractmethod
    def load(self, device: str = "AUTO") -> bool:
        """Load segmenter into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload segmenter and free memory."""
        pass

    @abstractmethod
    def segment_box(
        self,
        image: Union[str, np.ndarray],
        box_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Generate binary mask for the given bounding box prompt."""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return segmenter status and info."""
        pass


class BaseVerifier(ABC):
    """Abstract interface for AI verification / VLM judge."""

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def verify(
        self,
        image: Union[str, np.ndarray],
        detection: DetectionBox
    ) -> Dict[str, Any]:
        """
        Verify detection accuracy. Returns dict with:
        {"is_pole": bool, "confidence": float, "comment": str}
        """
        pass
