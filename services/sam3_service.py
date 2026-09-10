"""
services/sam3_service.py - Standalone SAM3 microservice
============================================================

Part 2: if SAM3's dependencies (transformers>=5.9) can't safely coexist with
the main app's torch_env_v2 (this repo's own DINO/Qwen/SAM2.1 environment),
run SAM3 in its own Python environment as this small FastAPI service instead,
and point the main app at it via verification_pipeline.sam3.backend=http +
sam3.service_url (see models/adapters/sam3_http_adapter.py, the client that
talks to this service).

Run in the SAM3-specific environment:
    uvicorn services.sam3_service:app --host 0.0.0.0 --port 8801

This process loads SAM3Adapter exactly once at startup and keeps it resident
in memory -- identical loading discipline to the in-process path, just in a
separate OS process/environment.
"""

from __future__ import annotations
import base64
import io
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from models.adapters.sam3_adapter import SAM3Adapter

app = FastAPI(title="SAM3 Isolated Segmentation Service")
sam3_adapter = SAM3Adapter()


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _mask_to_b64_png(mask: np.ndarray) -> str:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray((mask > 0).astype(np.uint8) * 255).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class SegmentBoxRequest(BaseModel):
    image_b64: str
    box_xyxy: Tuple[float, float, float, float]


class DetectRequest(BaseModel):
    image_b64: str
    text_prompt: Optional[str] = None


@app.get("/health")
def health():
    info = sam3_adapter.get_info()
    return {"available": info.status == "ready", "status": info.status, "error": info.error_message}


@app.post("/segment_box")
def segment_box(req: SegmentBoxRequest):
    pil_img = _b64_to_pil(req.image_b64)
    mask = sam3_adapter.segment_box(pil_img, req.box_xyxy)
    if mask is None:
        return {"mask_b64": None}
    return {"mask_b64": _mask_to_b64_png(mask)}


@app.post("/detect_and_segment")
def detect_and_segment(req: DetectRequest):
    pil_img = _b64_to_pil(req.image_b64)
    if req.text_prompt:
        sam3_adapter.text_prompt = req.text_prompt
    detections = sam3_adapter.detect_and_segment(pil_img)
    return {"detections": [d.to_dict() for d in detections]}


@app.on_event("startup")
def _load_on_startup():
    sam3_adapter.load()
