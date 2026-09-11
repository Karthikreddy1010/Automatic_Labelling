"""
services/sam3_service.py - Standalone SAM3 microservice
============================================================

Part 2: if SAM3's dependencies (transformers>=5.9) can't safely coexist with
the main app's torch_env_v2 (this repo's own DINO/Qwen/SAM2.1 environment),
run SAM3 in its own Python environment as this small FastAPI service instead,
and point the main app at it via verification_pipeline.sam3.backend=http +
sam3.service_url (see models/adapters/sam3_http_adapter.py, the client that
talks to this service).

Run in the SAM3-specific environment (e.g. the HAWK Singularity sandbox):
    uvicorn services.sam3_service:app --host 0.0.0.0 --port 8801

This process loads SAM3Adapter exactly once at startup and keeps it resident
in memory -- identical loading discipline to the in-process path, just in a
separate OS process/environment. If that startup load fails (or hasn't
happened yet), every endpoint returns an explicit HTTP 503 instead of
silently retrying the load on each request or returning an empty/null
result that looks like "no detections" rather than "model unavailable".
"""

from __future__ import annotations
import base64
import io
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from models.adapters.sam3_adapter import SAM3Adapter

sam3_adapter = SAM3Adapter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    sam3_adapter.load()
    yield


app = FastAPI(title="SAM3 Isolated Segmentation Service", lifespan=lifespan)


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _mask_to_b64_png(mask: np.ndarray) -> str:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray((mask > 0).astype(np.uint8) * 255).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _require_ready() -> None:
    """
    Raise an explicit 503 (never a silent empty result) when SAM3 isn't
    loaded -- and never trigger a reload attempt from within a request:
    SAM3Adapter.detect_and_segment()/segment_box() both try to (re)load on
    every call if their model is None, which would otherwise turn every
    request against a down/broken service into a full model-load retry.
    """
    info = sam3_adapter.get_info()
    if info.status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"SAM3 not available: {info.error_message or 'model not loaded'}",
        )


class SegmentBoxRequest(BaseModel):
    image_b64: str
    box_xyxy: Tuple[float, float, float, float]


class DetectRequest(BaseModel):
    image_b64: str
    text_prompt: Optional[str] = None


@app.get("/health")
def health():
    info = sam3_adapter.get_info()
    return {
        "available": info.status == "ready",
        "status": info.status,
        "device": sam3_adapter.device,
        "model_id": sam3_adapter.model_id,
        "error": info.error_message,
    }


@app.post("/segment_box")
def segment_box(req: SegmentBoxRequest):
    _require_ready()
    pil_img = _b64_to_pil(req.image_b64)
    mask, score = sam3_adapter.segment_box_with_score(pil_img, req.box_xyxy)
    if mask is None:
        raise HTTPException(status_code=422, detail="SAM3 produced no mask for this box.")
    return {"mask_b64": _mask_to_b64_png(mask), "score": score}


@app.post("/detect_and_segment")
def detect_and_segment(req: DetectRequest):
    _require_ready()
    pil_img = _b64_to_pil(req.image_b64)
    if req.text_prompt:
        sam3_adapter.text_prompt = req.text_prompt
    detections = sam3_adapter.detect_and_segment(pil_img)

    results = []
    for d in detections:
        entry = d.to_dict()
        entry["mask_b64"] = _mask_to_b64_png(d.mask) if d.mask is not None else None
        results.append(entry)
    return {"detections": results}
