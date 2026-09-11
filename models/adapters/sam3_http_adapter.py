"""
models/adapters/sam3_http_adapter.py - SAM3 HTTP Client Adapter
=====================================================================

Talks to services/sam3_service.py running in a separate SAM3-specific
environment (Part 2). Implements the exact same BaseSegmenter interface as
the in-process SAM3Adapter, so backend/app.py's module-level `sam3_adapter`
can point at either one -- every downstream call site (run_ai_pipeline,
_sam_refine_candidates, etc.) is unaffected by which backend is active.

Never crashes the main app when the service is down/unreachable: every
method degrades to is_available()=False / [] / None, matching the in-process
adapter's own graceful-failure contract.
"""

from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image

from models.adapters.base import BaseSegmenter, DetectionBox, ModelInfo


def _to_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        return Image.fromarray(rgb)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class SAM3HttpAdapter(BaseSegmenter):
    def __init__(self, service_url: str, timeout_s: float = 30.0):
        self.service_url = service_url.rstrip("/")
        self.timeout_s = timeout_s
        self._last_error: Optional[str] = None

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            self._last_error = data.get("error")
            return bool(data.get("available"))
        except Exception as e:
            self._last_error = f"SAM3 service unreachable at {self.service_url}: {e}"
            return False

    def load(self, device: str = "AUTO") -> bool:
        # The remote service loads its own model at its own startup; this
        # client has nothing local to load -- just confirm reachability.
        return self.is_available()

    def unload(self) -> None:
        pass  # nothing local to free; the remote process owns its own memory

    def detect_and_segment(self, image: Union[str, np.ndarray, Image.Image]) -> List[DetectionBox]:
        try:
            pil_img = _to_pil(image)
            resp = requests.post(
                f"{self.service_url}/detect_and_segment",
                json={"image_b64": _pil_to_b64(pil_img)},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            results = []
            for entry in resp.json().get("detections", []):
                mask_b64 = entry.pop("mask_b64", None)
                det = DetectionBox.from_dict(entry)
                if mask_b64:
                    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    det.mask = (np.array(mask_img) > 0).astype(np.uint8)
                results.append(det)
            return results
        except Exception as e:
            self._last_error = f"SAM3 service detect_and_segment failed: {e}"
            return []

    def segment_box(
        self, image: Union[str, np.ndarray, Image.Image], box_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        try:
            pil_img = _to_pil(image)
            resp = requests.post(
                f"{self.service_url}/segment_box",
                json={"image_b64": _pil_to_b64(pil_img), "box_xyxy": list(box_xyxy)},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            mask_b64 = resp.json().get("mask_b64")
            if not mask_b64:
                return None
            mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
            return (np.array(mask_img) > 0).astype(np.uint8)
        except Exception as e:
            self._last_error = f"SAM3 service segment_box failed: {e}"
            return None

    def get_info(self) -> ModelInfo:
        available = self.is_available()
        return ModelInfo(
            id="sam3_segmenter",
            name="SAM 3 Open-Vocabulary Segmenter",
            model_type="segmenter",
            status="ready" if available else "unavailable",
            weights_path=self.service_url,
            device="remote",
            backend="sam3_http",
            error_message=None if available else self._last_error,
            installation_guide=(
                f"Start the isolated SAM3 service in its own environment: "
                f"`uvicorn services.sam3_service:app --port 8801`, then confirm "
                f"verification_pipeline.sam3.service_url points at it (currently {self.service_url})."
            ),
        )
