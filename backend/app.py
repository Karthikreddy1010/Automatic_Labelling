"""
backend/app.py - FastAPI Application & Auto-Labeling REST Server
================================================================

Provides REST APIs for:
- Hardware inspection & device switching
- Model registry (YOLO, Grounding DINO, SAM 2.1, SAM 3, Gemini)
- Dataset management & image serving
- AI Label (production waterfall) & Run All (experimental comparison)
- Interactive box-prompted SAM segmentation -> canonical OBB
- Verified annotations & differential history tracking
- Async batch auto-labeling engine with pause/resume/cancel
- YOLO-OBB dataset export
"""

from __future__ import annotations
import os
import time
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2
from PIL import Image

from backend.storage import DatasetManager, DEFAULT_DATA_DIR
from models.adapters.base import (
    DetectionBox,
    ModelInfo,
    HardwareInfo,
    detect_hardware,
)
from models.adapters.yolo_adapter import YOLOAdapter
from models.adapters.dino_adapter import GroundingDINOAdapter
from models.adapters.sam21_adapter import SAM21Adapter
from models.adapters.sam3_adapter import SAM3Adapter
from models.adapters.qwen_adapter import QwenVerifier
from models.adapters.gemini_adapter import GeminiVerifier
from models.adapters.reconciliation import reconcile_candidates, get_confidence_category
from models.adapters.quality import dedup_by_iou, score_pole_quality, DEDUP_IOU_THRESHOLD
from models.adapters.geometry_qa import run_geometry_qa, config_from_dict as geometry_config_from_dict
from models.adapters.decision_engine import (
    evaluate_candidate,
    compute_segmentation_score,
    config_from_dict as decision_config_from_dict,
)
from models.adapters.obb_generator import generate_and_validate_obb
from src.geometry_obb import (
    xyxy_to_obb_corners,
    mask_to_obb_corners,
    order_corners_canonical,
    obb_corners_to_xyxy,
)

# Global Manager Instances
storage_mgr = DatasetManager()
active_device = "AUTO"


def _load_verification_pipeline_config() -> Dict[str, Any]:
    """
    Load configs/config.yaml's `verification_pipeline` section (geometry QA
    thresholds, decision weights, Qwen gating default). Never crashes on a
    missing file or missing section -- geometry_qa.py/decision_engine.py's
    dataclasses already default every field, so a partial or absent config
    just falls back to those documented starting values.
    """
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        return full_cfg.get("verification_pipeline", {}) or {}
    except Exception:
        return {}


_verification_cfg = _load_verification_pipeline_config()
GEOMETRY_QA_CONFIG = geometry_config_from_dict(_verification_cfg.get("geometry_qa"))
DECISION_CONFIG = decision_config_from_dict(_verification_cfg.get("decision"))
QWEN_GATING_DEFAULT = (_verification_cfg.get("qwen") or {}).get("gating", "gated")
# Production default: DINO alone proposes candidates; SAM3 only refines via
# box-prompt segmentation. True only enables the experimental dual-proposer
# comparison path (see configs/config.yaml verification_pipeline.sam3 docs).
SAM3_ENABLE_CANDIDATE_PROPOSAL = bool((_verification_cfg.get("sam3") or {}).get("enable_candidate_proposal", False))


def _build_sam3_adapter():
    """
    Part 2: pick the in-process SAM3Adapter or a remote SAM3HttpAdapter based
    on verification_pipeline.sam3.backend. "auto" prefers in-process (works
    fine if transformers>=5.9 is installed in this same environment) and
    only falls back to the HTTP client when a service_url is actually
    configured -- so a plain in-process setup (most deployments) needs zero
    extra config.
    """
    sam3_cfg = _verification_cfg.get("sam3", {}) or {}
    backend_pref = os.environ.get("SAM3_BACKEND", sam3_cfg.get("backend", "auto"))
    service_url = os.environ.get("SAM3_SERVICE_URL", sam3_cfg.get("service_url"))

    if backend_pref == "http" or (backend_pref == "auto" and service_url):
        from models.adapters.sam3_http_adapter import SAM3HttpAdapter
        return SAM3HttpAdapter(service_url=service_url or "http://127.0.0.1:8801")
    return SAM3Adapter()


def _build_qwen_verifier() -> QwenVerifier:
    """
    Reads verification_pipeline.qwen.* (Task 1's config keys) so the
    Transformers/Ollama/DashScope backend selection, local checkpoint path,
    dtype, device_map, and context-crop expansion are actually driven by
    configs/config.yaml rather than requiring every value to be set via
    environment variables. Any individual key can still be overridden by its
    existing env var (QWEN_MODEL_PATH, QWEN_OLLAMA_MODEL, etc.) -- the
    QwenVerifier constructor already prefers an explicit constructor arg over
    the env var, and here the config value is only used when the env var
    itself is unset.
    """
    qwen_cfg = _verification_cfg.get("qwen", {}) or {}
    tvl_cfg = qwen_cfg.get("transformers", {}) or {}
    return QwenVerifier(
        local_model_path=os.environ.get("QWEN_MODEL_PATH") or tvl_cfg.get("model_path"),
        context_crop_expand_pct=float(qwen_cfg.get("context_crop_expand_pct", 0.30)),
        backend=os.environ.get("QWEN_BACKEND", qwen_cfg.get("backend", "auto")),
        transformers_dtype=tvl_cfg.get("dtype", "bfloat16"),
        transformers_device_map=tvl_cfg.get("device_map", "auto"),
    )


# Model Adapters
# Prefer the trained pole-specific OBB detector over the generic candidate
# weights in YOLOAdapter.DEFAULT_CANDIDATE_PATHS when it's present.
_TRAINED_YOLO_OBB_WEIGHTS = "models/best.pt"
yolo_adapter = YOLOAdapter(
    weights_path=_TRAINED_YOLO_OBB_WEIGHTS if os.path.exists(_TRAINED_YOLO_OBB_WEIGHTS) else None
)
dino_adapter = GroundingDINOAdapter()
sam21_adapter = SAM21Adapter()
sam3_adapter = _build_sam3_adapter()
qwen_verifier = _build_qwen_verifier()
gemini_verifier = GeminiVerifier()


# --- BATCH ENGINE STATE ---
class BatchJobState:
    def __init__(self):
        self.lock = threading.Lock()
        self.job_id: Optional[str] = None
        self.dataset_id: Optional[str] = None
        self.status: str = "idle"  # idle, running, paused, cancelled, completed
        self.total_images: int = 0
        self.processed_count: int = 0
        self.accepted_count: int = 0
        self.needs_review_count: int = 0
        self.failed_count: int = 0
        self.confidences: List[float] = []
        self.current_filename: Optional[str] = None
        self.start_time: Optional[float] = None
        self.stop_requested: bool = False
        self.pause_requested: bool = False
        self.error_message: Optional[str] = None

    def reset(self):
        with self.lock:
            self.job_id = None
            self.dataset_id = None
            self.status = "idle"
            self.total_images = 0
            self.processed_count = 0
            self.accepted_count = 0
            self.needs_review_count = 0
            self.failed_count = 0
            self.confidences = []
            self.current_filename = None
            self.start_time = None
            self.stop_requested = False
            self.pause_requested = False
            self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = round(time.time() - self.start_time, 1) if self.start_time else 0.0
            avg_conf = round(float(np.mean(self.confidences)), 3) if self.confidences else 0.0
            return {
                "job_id": self.job_id,
                "dataset_id": self.dataset_id,
                "status": self.status,
                "total_images": self.total_images,
                "processed_count": self.processed_count,
                "accepted_count": self.accepted_count,
                "needs_review_count": self.needs_review_count,
                "failed_count": self.failed_count,
                "avg_confidence": avg_conf,
                "current_filename": self.current_filename,
                "elapsed_seconds": elapsed,
                "error_message": self.error_message,
            }


batch_job = BatchJobState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all available models on startup so they show 'ready' immediately."""
    import logging
    log = logging.getLogger("startup")

    adapters = [
        ("YOLO", yolo_adapter),
        ("Grounding DINO", dino_adapter),
        ("SAM 2.1", sam21_adapter),
        ("SAM 3", sam3_adapter),
    ]

    for name, adapter in adapters:
        t0 = time.time()
        if adapter.is_available():
            ok = adapter.load()
            elapsed = round(time.time() - t0, 1)
            if ok:
                log.info(f"  ✓ {name} loaded in {elapsed}s")
                print(f"  ✓ {name} loaded in {elapsed}s")
            else:
                log.warning(f"  ✗ {name} failed to load ({elapsed}s): {getattr(adapter, '_error_msg', 'unknown')}")
                print(f"  ✗ {name} failed to load ({elapsed}s): {getattr(adapter, '_error_msg', 'unknown')}")
        else:
            # Mark unavailable so UI shows correct status instead of misleading 'standby'
            adapter._status = "unavailable"
            adapter._error_msg = f"{name} dependencies not installed or not found."
            log.info(f"  – {name} not available (dependencies missing)")
            print(f"  – {name} not available (dependencies missing)")

    yield


app = FastAPI(
    title="Electric Utility Pole OBB AI Annotation Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REQUEST & RESPONSE SCHEMAS ---

class DeviceRequest(BaseModel):
    device: str = Field(..., description="'AUTO', 'CUDA', or 'CPU'")


class CreateDatasetRequest(BaseModel):
    dataset_id: str
    name: Optional[str] = None
    classes: Optional[List[str]] = None


class ImportImagesRequest(BaseModel):
    source_dir: str


class DetectionRequest(BaseModel):
    dataset_id: str
    filename: str
    mode: str = Field("AI_LABEL", description="'AI_LABEL', 'RUN_ALL', 'YOLO_ONLY', 'DINO_ONLY', 'SAM_ONLY'")
    conf_threshold: float = Field(0.25, description="DINO box_threshold (also used as YOLO conf in benchmark modes)")
    iou_threshold: float = 0.50
    use_sam_refinement: bool = True
    text_threshold: Optional[float] = Field(None, description="DINO text_threshold override; defaults to the adapter's configured value")
    max_candidates: Optional[int] = Field(None, description="Cap on raw DINO candidates before dedup; defaults to DINO_MAX_RAW_CANDIDATES")
    enable_geometry_qa: bool = Field(True, description="Run geometry_qa.py on each SAM-refined candidate (disable for baseline A/B comparison)")
    qwen_gating: Optional[str] = Field(None, description="'off' | 'gated' | 'always' -- overrides configs/config.yaml verification_pipeline.qwen.gating for this request")


class SegmentBoxRequest(BaseModel):
    dataset_id: str
    filename: str
    box_xyxy: List[float]


class SaveAnnotationRequest(BaseModel):
    boxes: List[Dict[str, Any]]
    image_width: int
    image_height: int
    is_human: bool = True
    action: Optional[str] = Field(None, description="'accepted', 'human_corrected', 'rejected'")
    al_categories: Optional[List[str]] = None
    failure_type: Optional[str] = None
    negative_category: Optional[str] = None
    is_negative: bool = False


class RejectRequest(BaseModel):
    failure_type: str = "false_positive"
    negative_category: Optional[str] = None
    al_categories: Optional[List[str]] = None


class ActiveLearningWeightsRequest(BaseModel):
    weights: Dict[str, int]
    duplicate_phash_threshold: int = 8
    max_priority: int = 100


class BatchStartRequest(BaseModel):
    dataset_id: str
    mode: str = "AI_LABEL"
    conf_threshold: float = 0.25
    only_unlabeled: bool = True
    use_sam_refinement: bool = True
    enable_geometry_qa: bool = True
    qwen_gating: Optional[str] = None
    # Restrict the run to exactly these filenames instead of the whole
    # dataset -- for safe, controlled testing on a small explicitly-selected
    # image set. Filenames not present in the dataset are dropped (not
    # silently treated as processed); None (the default) means "the whole
    # dataset", unchanged from before this field existed.
    filenames: Optional[List[str]] = None


# --- HARDWARE & SYSTEM ENDPOINTS ---

@app.get("/api/system/hardware")
def get_hardware():
    resolved_dev, info = detect_hardware(active_device)
    return {
        "active_device_setting": active_device,
        "resolved_device": resolved_dev,
        "hardware": info.to_dict(),
    }


@app.post("/api/system/device")
def set_device(req: DeviceRequest):
    global active_device
    req_dev = req.device.upper().strip()
    if req_dev not in ("AUTO", "CUDA", "CPU"):
        raise HTTPException(status_code=400, detail="Invalid device. Must be 'AUTO', 'CUDA', or 'CPU'.")
    active_device = req_dev
    resolved, info = detect_hardware(active_device)
    return {"message": f"Device updated to {active_device}", "resolved": resolved, "hardware": info.to_dict()}


@app.get("/api/models/status")
def get_models_status():
    return {
        "models": [
            yolo_adapter.get_info().to_dict(),
            dino_adapter.get_info().to_dict(),
            sam21_adapter.get_info().to_dict(),
            sam3_adapter.get_info().to_dict(),
            qwen_verifier.get_info().to_dict(),
            gemini_verifier.get_info().to_dict(),
        ]
    }


@app.post("/api/models/load")
def load_model_endpoint(model_id: str = Query(...)):
    if "yolo" in model_id:
        ok = yolo_adapter.load()
    elif "dino" in model_id:
        ok = dino_adapter.load()
    elif "sam21" in model_id:
        ok = sam21_adapter.load()
    elif "sam3" in model_id:
        ok = sam3_adapter.load()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")
    return {"success": ok, "models": get_models_status()["models"]}


# --- DATASET MANAGEMENT ENDPOINTS ---

@app.get("/api/datasets")
def list_datasets():
    return {"datasets": storage_mgr.list_datasets()}


@app.post("/api/datasets")
def create_dataset(req: CreateDatasetRequest):
    meta = storage_mgr.create_dataset(req.dataset_id, req.name, req.classes)
    return {"dataset": meta}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = storage_mgr.get_dataset(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"dataset": meta}


@app.get("/api/datasets/{dataset_id}/images")
def get_dataset_images(dataset_id: str, status: Optional[str] = None):
    imgs = storage_mgr.get_image_list(dataset_id, filter_status=status)
    return {"images": imgs, "count": len(imgs)}


@app.post("/api/datasets/{dataset_id}/import")
def import_images_endpoint(dataset_id: str, req: ImportImagesRequest):
    p = Path(req.source_dir)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Source path {req.source_dir} does not exist.")
    imported = storage_mgr.import_images(dataset_id, [p])
    return {"imported_count": len(imported), "imported_images": imported}


@app.post("/api/datasets/{dataset_id}/import_upload")
async def import_uploaded_images_endpoint(dataset_id: str, files: List[UploadFile] = File(...)):
    payload = [(f.filename or "", await f.read()) for f in files]
    imported = storage_mgr.import_uploaded_files(dataset_id, payload)
    return {"imported_count": len(imported), "imported_images": imported}


@app.get("/api/datasets/{dataset_id}/images/{filename}")
def serve_image(dataset_id: str, filename: str):
    img_path = storage_mgr.get_image_path(dataset_id, filename)
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(img_path))


# --- ANNOTATIONS & DIFFERENTIAL TRACKING ---

@app.get("/api/datasets/{dataset_id}/images/{filename}/annotations")
def get_image_annotations(dataset_id: str, filename: str):
    ann = storage_mgr.get_annotation(dataset_id, filename)
    if ann:
        return {"type": "verified_annotation", "data": ann}
    preds = storage_mgr.get_predictions(dataset_id, filename)
    if preds:
        return {"type": "ai_prediction", "data": preds}
    return {"type": "none", "data": None}


@app.post("/api/datasets/{dataset_id}/images/{filename}/annotations")
def save_image_annotations(dataset_id: str, filename: str, req: SaveAnnotationRequest):
    boxes = [DetectionBox.from_dict(b) for b in req.boxes]
    saved = storage_mgr.save_annotation(
        dataset_id,
        filename,
        boxes,
        img_width=req.image_width,
        img_height=req.image_height,
        is_human=req.is_human,
        action=req.action,
        al_categories=req.al_categories,
        failure_type=req.failure_type,
        negative_category=req.negative_category,
        is_negative=req.is_negative,
    )
    return {"message": "Annotation saved successfully", "annotation": saved}


@app.post("/api/datasets/{dataset_id}/images/{filename}/reject")
def reject_image_annotation(
    dataset_id: str,
    filename: str,
    req: Optional[RejectRequest] = None
):
    ft = req.failure_type if req and req.failure_type else "false_positive"
    nc = req.negative_category if req else None
    al_cats = req.al_categories if req else None
    saved = storage_mgr.reject_with_metadata(
        dataset_id,
        filename,
        failure_type=ft,
        negative_category=nc,
        al_categories=al_cats
    )
    return {"message": "Image marked as rejected", "annotation": saved}


@app.post("/api/datasets/{dataset_id}/images/{filename}/skip")
def skip_image_endpoint(dataset_id: str, filename: str):
    storage_mgr.mark_skipped(dataset_id, filename)
    return {"status": "skipped", "filename": filename}


@app.get("/api/datasets/{dataset_id}/images/{filename}/predictions")
def get_raw_predictions(dataset_id: str, filename: str):
    preds = storage_mgr.get_predictions(dataset_id, filename)
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found for this image.")
    return {"predictions": preds}


@app.get("/api/datasets/{dataset_id}/history")
def get_differential_history(dataset_id: str, filename: Optional[str] = None):
    hist = storage_mgr.get_history(dataset_id, filename)
    return {"history": hist, "count": len(hist)}


@app.post("/api/datasets/{dataset_id}/export")
def export_dataset_endpoint(dataset_id: str):
    try:
        res = storage_mgr.export_dataset(dataset_id)
        return {"message": "Export completed successfully", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ACTIVE LEARNING & MASK ENDPOINTS ---

@app.get("/api/datasets/{dataset_id}/active_learning")
def get_active_learning_queue_endpoint(dataset_id: str, filter_reason: Optional[str] = Query(None)):
    queue = storage_mgr.get_active_learning_queue(dataset_id, filter_reason=filter_reason)
    return {"queue": queue, "count": len(queue)}


@app.get("/api/datasets/{dataset_id}/next_difficult")
def get_next_difficult_endpoint(dataset_id: str):
    item = storage_mgr.get_next_difficult(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="No images found in queue.")
    return {"next_item": item}


@app.post("/api/datasets/{dataset_id}/export_hard_cases")
def export_hard_cases_endpoint(dataset_id: str, min_priority: int = Query(60)):
    try:
        res = storage_mgr.export_hard_cases(dataset_id, min_priority=min_priority)
        return {"message": "Hard cases export completed", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{dataset_id}/composition")
def get_dataset_composition_endpoint(dataset_id: str):
    comp = storage_mgr.get_dataset_composition(dataset_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return comp


@app.post("/api/datasets/{dataset_id}/images/{filename}/test_set")
def toggle_image_test_set(dataset_id: str, filename: str):
    try:
        new_state = storage_mgr.toggle_test_set(dataset_id, filename)
        return {"filename": filename, "is_test_set": new_state}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/datasets/{dataset_id}/test_set")
def get_test_set_endpoint(dataset_id: str):
    test_set = storage_mgr.get_test_set(dataset_id)
    return {"test_set": test_set, "count": len(test_set)}


@app.post("/api/datasets/{dataset_id}/export_test_set")
def export_test_set_endpoint(dataset_id: str):
    try:
        res = storage_mgr.export_test_set(dataset_id)
        return {"message": "Test set exported successfully", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{dataset_id}/duplicates")
def get_near_duplicates_endpoint(dataset_id: str, threshold: Optional[int] = Query(None)):
    duplicates = storage_mgr.find_near_duplicates(dataset_id, threshold=threshold)
    return {"duplicates": duplicates, "count": len(duplicates)}


@app.get("/api/config/active_learning_weights")
def get_al_weights_endpoint():
    cfg = storage_mgr._load_al_weights()
    return cfg


@app.put("/api/config/active_learning_weights")
def update_al_weights_endpoint(req: ActiveLearningWeightsRequest):
    data = {
        "weights": req.weights,
        "duplicate_phash_threshold": req.duplicate_phash_threshold,
        "max_priority": req.max_priority,
    }
    saved = storage_mgr._save_al_weights(data)
    return {"message": "Weights updated", "config": saved}


@app.get("/api/config/verification_pipeline")
def get_verification_pipeline_config():
    """Read-only view of the loaded geometry QA / decision engine / Qwen gating
    configuration (configs/config.yaml verification_pipeline section), for the
    frontend's provenance panel and for the benchmark harness to log exact
    settings alongside its results."""
    return {
        "qwen_gating_default": QWEN_GATING_DEFAULT,
        "geometry_qa": dict(GEOMETRY_QA_CONFIG.__dict__),
        "decision": {
            "weights": DECISION_CONFIG.weights.as_dict(),
            "accept_threshold": DECISION_CONFIG.accept_threshold,
            "review_threshold": DECISION_CONFIG.review_threshold,
            "semantic_reject_confidence": DECISION_CONFIG.semantic_reject_confidence,
        },
    }


@app.get("/api/datasets/{dataset_id}/masks/{mask_filename}")
def serve_cached_mask(dataset_id: str, mask_filename: str):
    p = storage_mgr.get_cached_mask_file_path(dataset_id, mask_filename)
    if not p:
        raise HTTPException(status_code=404, detail="Mask not found.")
    return FileResponse(str(p), media_type="image/png")


# --- AI INFERENCE ENGINE ---

YOLO_CONFIDENT_THRESHOLD = 0.55  # used by YOLO_FAST/YOLO_ONLY/RUN_ALL benchmarking only

# --- Production pipeline knobs (single source of truth; do not hard-code
# these elsewhere -- pass overrides through DetectionRequest/BatchStartRequest). ---
DINO_MAX_RAW_CANDIDATES = 12   # raw DINO detections considered before dedup
SAM_REFINE_TOP_N = 5           # deduped candidates actually sent through SAM


def _sam_refine_candidates(
    candidates: List[DetectionBox],
    img_path: Path,
    dataset_id: Optional[str],
    filename: Optional[str],
    img_width: int,
    img_height: int,
    model_source: str = "DINO+SAM",
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[List[DetectionBox], int, int]:
    """
    Refine up to SAM_REFINE_TOP_N deduped candidates (highest confidence
    first) with SAM into a canonical rotated OBB, score each with
    quality.score_pole_quality(), and drop REJECT-bucketed candidates
    (obvious false positives) rather than returning them to the reviewer.

    The resulting OBB is generated and structurally validated via
    obb_generator.py (all corners in-bounds, non-self-intersecting, positive
    area) -- a candidate whose geometry fails that validation is dropped
    rather than silently handed a degenerate label. Each kept candidate keeps
    its raw SAM mask on DetectionBox.mask (never serialized -- see
    DetectionBox.to_dict) so the caller can run geometry QA without
    re-running SAM.

    Returns (kept_boxes, n_quality_rejected, n_invalid_obb). If `timings` is
    given, accumulates "sam3_refine_ms" (time spent in segment_box calls) and
    "obb_ms" (time spent generating/validating OBBs from masks) into it.
    """
    result_boxes: List[DetectionBox] = []
    n_rejected = 0
    n_invalid_obb = 0
    sam_ms_total = 0.0
    obb_ms_total = 0.0
    sorted_candidates = sorted(candidates, key=lambda b: b.confidence, reverse=True)[:SAM_REFINE_TOP_N]
    for idx, det in enumerate(sorted_candidates):
        mask = None
        _t_sam = time.perf_counter()
        try:
            # SAM3 is preferred (per the spec's SAM3-upgrade intent); SAM 2.1
            # is only used when SAM3 itself is unavailable.
            if sam3_adapter.is_available():
                mask = sam3_adapter.segment_box(str(img_path), det.xyxy)
            elif sam21_adapter.is_available():
                mask = sam21_adapter.segment_box(str(img_path), det.xyxy)
        except Exception:
            pass
        sam_ms_total += (time.perf_counter() - _t_sam) * 1000

        quality = score_pole_quality(mask, det.xyxy, det.confidence)
        if quality["category"] == "REJECT":
            n_rejected += 1
            continue

        _t_obb = time.perf_counter()
        obb_result = generate_and_validate_obb(mask, det.xyxy, img_width, img_height)
        obb_ms_total += (time.perf_counter() - _t_obb) * 1000
        if not obb_result["valid"]:
            n_invalid_obb += 1
            continue
        new_corners = obb_result["corners"]

        mask_rel_path = None
        if mask is not None and dataset_id and filename:
            try:
                mask_rel_path = storage_mgr.save_mask(dataset_id, filename, idx, mask)
            except Exception:
                pass

        if obb_result["source"] == "xyxy_fallback":
            sam_quality = "failed"
            mask_area = 0
        else:
            mask_area = int((mask > 0).sum()) if mask is not None else 0
            sam_quality = "good" if mask_area > 50 else "poor"

        enclosing_xyxy = obb_corners_to_xyxy(new_corners)
        result_boxes.append(DetectionBox(
            xyxy=enclosing_xyxy,
            corners=new_corners.tolist(),
            confidence=round(det.confidence, 3),
            class_id=det.class_id,
            class_name=det.class_name or "utility_pole",
            model_source=model_source,
            needs_review=(quality["category"] != "HIGH_QUALITY"),
            review_reasons=quality["reasons"],
            mask=mask,
            attributes={
                "sam_quality": sam_quality,
                "mask_area": mask_area,
                "mask_path": mask_rel_path,
                "quality_score": quality["quality_score"],
                "category": quality["category"],
                "obb_source": obb_result["source"],
            }
        ))
    if timings is not None:
        timings["sam3_refine_ms"] = round(sam_ms_total, 1)
        timings["obb_ms"] = round(obb_ms_total, 1)
    return result_boxes, n_rejected, n_invalid_obb


def _apply_verification(
    candidates: List[DetectionBox],
    img_path: Path,
    img_width: int,
    img_height: int,
    enable_geometry_qa: bool = True,
    qwen_gating: str = "gated",
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[List[DetectionBox], Dict[str, int]]:
    """
    Stage 4 of the production pipeline (after SAM refinement): geometry QA +
    Qwen3-VL semantic verification + the multi-signal decision engine.

    Geometry QA (models/adapters/geometry_qa.py) reuses each candidate's
    already-computed SAM mask (DetectionBox.mask, set by _sam_refine_candidates).
    Qwen (models/adapters/qwen_adapter.py) is gated per `qwen_gating`:
      "off"    -- never called (baseline/geometry-only comparison configs).
      "gated"  -- skipped only for candidates geometry QA already hard-fails
                  (saves a real VLM call on an already-doomed candidate);
                  called otherwise.
      "always" -- called on every surviving candidate (evaluation experiments).
    Neither Qwen nor geometry QA ever produces OBB coordinates -- geometry is
    already finalized by _sam_refine_candidates/obb_generator before this runs.

    A REJECT decision from the combined decision engine drops the candidate
    here (mirrors the existing SAM-quality REJECT drop) rather than showing
    an AI-confirmed non-pole to the human reviewer. ACCEPT/REVIEW candidates
    are kept and flagged via needs_review + attributes.decision.

    Returns (kept_boxes, counts) where counts has n_semantic_rejected,
    n_qwen_calls, n_disagreements.
    """
    counts = {
        "n_semantic_rejected": 0, "n_qwen_calls": 0, "n_disagreements": 0,
        # Distinguishes "gating requested Qwen but it never ran" reasons: a
        # benchmark comparing configs must not read n_qwen_calls==0 as "Qwen
        # rejected everything" when it's actually "Qwen wasn't reachable."
        "qwen_available": qwen_verifier.is_available(),
    }
    kept: List[DetectionBox] = []
    geometry_ms_total = 0.0
    qwen_ms_total = 0.0

    for det in candidates:
        geometry_result = None
        if enable_geometry_qa:
            _t_geo = time.perf_counter()
            geometry_result = run_geometry_qa(det.mask, np.array(det.corners), img_width, img_height, GEOMETRY_QA_CONFIG)
            geometry_ms_total += (time.perf_counter() - _t_geo) * 1000

        geometry_hard_fail = bool(geometry_result and geometry_result["geometry_status"] == "fail")

        should_call_qwen = (
            qwen_gating == "always"
            or (qwen_gating == "gated" and not geometry_hard_fail)
        ) and qwen_gating != "off"

        semantic_result = None
        if should_call_qwen and qwen_verifier.is_available():
            _t_qwen = time.perf_counter()
            semantic_result = qwen_verifier.verify(str(img_path), det)
            qwen_ms_total += (time.perf_counter() - _t_qwen) * 1000
            counts["n_qwen_calls"] += 1

        segmentation_score = compute_segmentation_score(det.mask, det.xyxy)

        decision_result = evaluate_candidate(
            detection_score=det.confidence,
            segmentation_score=segmentation_score,
            geometry_result=geometry_result,
            semantic_result=semantic_result,
            obb_valid=True,  # already validated in _sam_refine_candidates
            config=DECISION_CONFIG,
        )

        if decision_result["disagreements"]:
            counts["n_disagreements"] += 1

        det.attributes["segmentation_score"] = decision_result["segmentation_score"]
        det.attributes["final_score"] = decision_result["final_score"]
        det.attributes["decision"] = decision_result["decision"]
        det.attributes["disagreements"] = decision_result["disagreements"]
        if geometry_result:
            det.attributes["geometry_score"] = geometry_result["geometry_score"]
            det.attributes["geometry_status"] = geometry_result["geometry_status"]
            det.attributes["geometry_flags"] = geometry_result["geometry_flags"]
        if semantic_result:
            det.attributes["qwen_class"] = decision_result["qwen_class"]
            det.attributes["qwen_material"] = decision_result["qwen_material"]
            det.attributes["qwen_visibility"] = decision_result["qwen_visibility"]
            det.attributes["qwen_orientation"] = decision_result["qwen_orientation"]
            det.attributes["qwen_semantic_confidence"] = semantic_result.get("semantic_confidence")
            det.attributes["qwen_reason"] = decision_result["qwen_reason"]
            det.attributes["qwen_ran"] = True
        else:
            det.attributes["qwen_ran"] = False

        det.review_reasons = list(det.review_reasons) + decision_result["reasons"]
        det.needs_review = decision_result["decision"] != "ACCEPT"
        det.mask = None  # never needed past this point; keep DetectionBox light

        if decision_result["decision"] == "REJECT":
            counts["n_semantic_rejected"] += 1
            continue
        kept.append(det)

    if timings is not None:
        timings["geometry_ms"] = round(geometry_ms_total, 1)
        timings["qwen_ms"] = round(qwen_ms_total, 1)
    return kept, counts


def run_ai_pipeline(
    img_path: Path,
    mode: str = "AI_LABEL",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    use_sam_refinement: bool = True,
    device: str = "AUTO",
    dataset_id: Optional[str] = None,
    filename: Optional[str] = None,
    text_threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    enable_geometry_qa: bool = True,
    qwen_gating: Optional[str] = None,
) -> Tuple[List[DetectionBox], Dict[str, Any]]:
    """
    Executes detection pipeline:
    - AI_LABEL / DINO_SAM (production pipeline; intentionally does NOT call
      YOLO/best.pt/A_S.pt):
        1. Grounding DINO proposes candidate boxes -- WHERE the pole might
           be (box_threshold via conf_threshold, text_threshold,
           max_candidates all configurable per-request -- see
           DINO_MAX_RAW_CANDIDATES for the raw cap). SAM3's own
           detect_and_segment() candidate-proposal is NOT called here by
           default (SAM3_ENABLE_CANDIDATE_PROPOSAL=False) -- it would be
           redundant production inference, since every SAM3-proposed
           candidate gets re-segmented via segment_box() in step 3 anyway,
           discarding that first mask. Set verification_pipeline.sam3.
           enable_candidate_proposal=true only for the experimental
           "DINO+SAM3 dual proposer" comparison path.
        2. Candidates deduped by IoU (dedup_by_iou) so one physical pole
           doesn't produce multiple overlapping labels.
        3. Each of the top SAM_REFINE_TOP_N deduped candidates is segmented
           with SAM 3 (or SAM 2.1 fallback) -- WHICH PIXELS belong to the
           candidate -- scored with score_pole_quality(), and
           REJECT-bucketed candidates are dropped. Survivors are flagged
           needs_review unless HIGH_QUALITY. The resulting OBB is generated
           from the mask and structurally validated (obb_generator.py).
        4. Geometry QA (geometry_qa.py) + Qwen3-VL semantic verification
           (qwen_adapter.py, gated by `qwen_gating`) + the multi-signal
           decision engine (decision_engine.py) combine detection/
           segmentation/geometry/semantic evidence into a final ACCEPT/
           REVIEW/REJECT call per candidate (_apply_verification). Neither
           stage ever touches the OBB geometry itself -- see
           decision_engine.py's module docstring for why Qwen/geometry QA
           are additive verification layers, not replacements for DINO/SAM3.
      DINO_ONLY skips steps 3-4 entirely (raw deduped DINO boxes only).
      If DINO or SAM/SAM3 is unavailable, returns raw_outputs["error"]
      instead of silently falling back to any YOLO detector.
    - YOLO_FAST / YOLO_ONLY: benchmark-only accelerator using best.pt, no
      DINO fallback. Never used by AI_LABEL.
    - RUN_ALL (Model comparison / benchmarking): Executes YOLO + DINO + SAM,
      preserves raw outputs separately, and reconciles.
    """
    raw_outputs: Dict[str, Any] = {}
    yolo_boxes: List[DetectionBox] = []
    dino_boxes: List[DetectionBox] = []
    sam_boxes: List[DetectionBox] = []

    # 0. PRODUCTION PIPELINE: Grounding DINO + SAM3 (candidate proposers) ->
    # dedup -> SAM 2.1 refinement -> quality-scored OBB.
    # YOLO (best.pt / A_S.pt) is intentionally never called here.
    if mode in ("AI_LABEL", "DINO_SAM", "DINO_ONLY"):
        timings: Dict[str, float] = {}
        _t0 = time.perf_counter()

        if not dino_adapter.is_available():
            raw_outputs["error"] = "DINO unavailable"
            return [], raw_outputs

        _t_dino = time.perf_counter()
        dino_boxes = dino_adapter.predict(
            str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            text_threshold=text_threshold, max_candidates=max_candidates or DINO_MAX_RAW_CANDIDATES,
        )
        timings["dino_ms"] = round((time.perf_counter() - _t_dino) * 1000, 1)
        raw_outputs["dino_raw_count"] = len(dino_boxes)

        # Production default (SAM3_ENABLE_CANDIDATE_PROPOSAL=False): DINO
        # alone proposes candidates (WHERE); SAM3 is reserved for box-prompt
        # refinement (PIXELS) below, keeping the two models' responsibilities
        # separated. Calling SAM3's own open-vocabulary detect_and_segment()
        # here too would be redundant production inference -- every
        # SAM3-proposed candidate gets re-segmented via segment_box() moments
        # later in _sam_refine_candidates() regardless of origin, discarding
        # this call's mask entirely. Set verification_pipeline.sam3.
        # enable_candidate_proposal=true only for the experimental
        # "DINO+SAM3 as dual proposers" comparison path (e.g.
        # benchmark_dino_sam.py) -- never in production.
        sam3_candidates: List[DetectionBox] = []
        _t_sam3_propose = time.perf_counter()
        if SAM3_ENABLE_CANDIDATE_PROPOSAL and sam3_adapter.is_available():
            try:
                sam3_candidates = sam3_adapter.detect_and_segment(str(img_path))
            except Exception as e:
                # is_available() already confirmed the backend is loaded, so
                # a failure here is a genuine runtime error (CUDA OOM, a
                # malformed image, an HTTP timeout against the SAM3 service,
                # etc.) -- log it rather than silently swallowing it, even
                # though the experimental proposal step degrades gracefully
                # to "DINO's candidates only" either way.
                import logging
                logging.getLogger("sam3").warning(
                    "SAM3 experimental candidate-proposal failed for %s: %s", img_path.name, e
                )
                sam3_candidates = []
        timings["sam3_propose_ms"] = round((time.perf_counter() - _t_sam3_propose) * 1000, 1)
        raw_outputs["sam3_raw_count"] = len(sam3_candidates)

        combined_candidates = dedup_by_iou(dino_boxes + sam3_candidates, iou_threshold=DEDUP_IOU_THRESHOLD)
        raw_outputs["dino"] = [b.to_dict() for b in dino_boxes]
        raw_outputs["sam3_candidates"] = [b.to_dict() for b in sam3_candidates]
        raw_outputs["combined_deduped_count"] = len(combined_candidates)

        if mode == "DINO_ONLY" or not use_sam_refinement:
            timings["total_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
            if (_verification_cfg.get("logging", {}) or {}).get("log_stage_timings", True):
                raw_outputs["timings"] = timings
            return combined_candidates, raw_outputs

        if not (sam21_adapter.is_available() or sam3_adapter.is_available()):
            raw_outputs["error"] = "SAM 2.1/SAM 3 unavailable"
            return [], raw_outputs

        with Image.open(str(img_path)) as _im:
            img_width, img_height = _im.size

        result_boxes, n_rejected, n_invalid_obb = _sam_refine_candidates(
            combined_candidates, img_path, dataset_id, filename, img_width, img_height,
            model_source="DINO+SAM3+SAM", timings=timings,
        )
        raw_outputs["n_sam_rejected"] = n_rejected
        raw_outputs["n_invalid_obb"] = n_invalid_obb

        gating = qwen_gating if qwen_gating is not None else QWEN_GATING_DEFAULT
        result_boxes, verify_counts = _apply_verification(
            result_boxes, img_path, img_width, img_height,
            enable_geometry_qa=enable_geometry_qa, qwen_gating=gating, timings=timings,
        )
        raw_outputs["verification"] = {
            "geometry_qa_enabled": enable_geometry_qa,
            "qwen_gating": gating,
            **verify_counts,
        }

        timings["total_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
        if (_verification_cfg.get("logging", {}) or {}).get("log_stage_timings", True):
            raw_outputs["timings"] = timings
            print(f"[timing] {filename or img_path.name}: {timings}")
        return result_boxes, raw_outputs

    # 2. OPTIONAL ACCELERATOR: YOLO / YOLO-OBB
    elif mode in ("YOLO_FAST", "YOLO_ONLY"):
        if yolo_adapter.is_available():
            yolo_boxes = yolo_adapter.predict(str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            raw_outputs["yolo"] = [b.to_dict() for b in yolo_boxes]
        if use_sam_refinement and (sam21_adapter.is_available() or sam3_adapter.is_available()):
            for det in yolo_boxes[:3]:
                try:
                    # SAM3 is preferred; SAM 2.1 is only used when SAM3 itself
                    # is unavailable.
                    if sam3_adapter.is_available():
                        mask = sam3_adapter.segment_box(str(img_path), det.xyxy)
                        new_corners = mask_to_obb_corners(mask) if mask is not None else None
                    else:
                        _, new_corners = sam21_adapter.segment_and_generate_obb(str(img_path), det.xyxy)
                    if new_corners is not None:
                        det.corners = new_corners.tolist()
                        det.attributes["refined_by_sam"] = True
                except Exception:
                    pass
        return yolo_boxes, raw_outputs

    # 3. EXPERIMENTAL COMPARISON: RUN ALL MODELS
    elif mode == "RUN_ALL":
        if yolo_adapter.is_available():
            yolo_boxes = yolo_adapter.predict(str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            raw_outputs["yolo"] = [b.to_dict() for b in yolo_boxes]

        if dino_adapter.is_available():
            dino_boxes = dino_adapter.predict(str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            raw_outputs["dino"] = [b.to_dict() for b in dino_boxes]

        if sam3_adapter.is_available():
            sam_boxes = sam3_adapter.detect_and_segment(str(img_path))
            raw_outputs["sam3"] = [b.to_dict() for b in sam_boxes]

        reconciled = reconcile_candidates(
            yolo_boxes=yolo_boxes,
            dino_boxes=dino_boxes,
            sam_boxes=sam_boxes,
            iou_threshold=iou_threshold,
            min_confidence=conf_threshold
        )
        return reconciled, raw_outputs

    else:
        if dino_adapter.is_available():
            dino_boxes = dino_adapter.predict(str(img_path), conf_threshold=conf_threshold)
            raw_outputs["dino"] = [b.to_dict() for b in dino_boxes]
        return dino_boxes, raw_outputs


@app.post("/api/inference/detect")
def detect_single_image(req: DetectionRequest):
    img_path = storage_mgr.get_image_path(req.dataset_id, req.filename)
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found in dataset.")

    resolved_dev, _ = detect_hardware(active_device)
    reconciled, raw_outputs = run_ai_pipeline(
        img_path=img_path,
        mode=req.mode,
        conf_threshold=req.conf_threshold,
        iou_threshold=req.iou_threshold,
        use_sam_refinement=req.use_sam_refinement,
        device=resolved_dev,
        dataset_id=req.dataset_id,
        filename=req.filename,
        text_threshold=req.text_threshold,
        max_candidates=req.max_candidates,
        enable_geometry_qa=req.enable_geometry_qa,
        qwen_gating=req.qwen_gating,
    )

    if raw_outputs.get("error"):
        # A required model is unavailable -- surface it clearly rather than
        # silently returning an empty "no poles found" result.
        raise HTTPException(status_code=503, detail=raw_outputs["error"])

    # Persist untouched raw predictions in predictions/
    storage_mgr.save_predictions(req.dataset_id, req.filename, reconciled, raw_outputs)

    return {
        "filename": req.filename,
        "boxes": [b.to_dict() for b in reconciled],
        "count": len(reconciled),
        "raw_outputs": raw_outputs,
    }


@app.post("/api/inference/segment_box")
def segment_box_endpoint(req: SegmentBoxRequest):
    img_path = storage_mgr.get_image_path(req.dataset_id, req.filename)
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found.")

    if len(req.box_xyxy) != 4:
        raise HTTPException(status_code=400, detail="box_xyxy must contain [x1, y1, x2, y2].")

    box_tuple = (req.box_xyxy[0], req.box_xyxy[1], req.box_xyxy[2], req.box_xyxy[3])
    mask = None
    corners = None

    # SAM3 is preferred; SAM 2.1 is only used when SAM3 itself is unavailable.
    if sam3_adapter.is_available():
        mask = sam3_adapter.segment_box(str(img_path), box_tuple)
        corners = mask_to_obb_corners(mask) if mask is not None else None
    elif sam21_adapter.is_available():
        mask, corners = sam21_adapter.segment_and_generate_obb(str(img_path), box_tuple)

    if corners is None:
        # Fallback to canonical box
        corners = xyxy_to_obb_corners(*box_tuple)

    return {
        "corners": corners.tolist(),
        "has_mask": mask is not None,
    }


# --- BATCH AUTO-LABELING WORKER ---

def _batch_worker(
    dataset_id: str,
    image_names: List[str],
    mode: str,
    conf_threshold: float,
    use_sam_refinement: bool,
    enable_geometry_qa: bool = True,
    qwen_gating: Optional[str] = None,
):
    global batch_job
    resolved_dev, _ = detect_hardware(active_device)

    for name in image_names:
        # Check cancellation
        if batch_job.stop_requested:
            batch_job.status = "cancelled"
            return

        # Check pause
        while batch_job.pause_requested:
            batch_job.status = "paused"
            time.sleep(0.5)
            if batch_job.stop_requested:
                batch_job.status = "cancelled"
                return
        batch_job.status = "running"

        batch_job.current_filename = name
        img_path = storage_mgr.get_image_path(dataset_id, name)
        if not img_path:
            batch_job.failed_count += 1
            continue

        try:
            reconciled, raw_outputs = run_ai_pipeline(
                img_path=img_path,
                mode=mode,
                conf_threshold=conf_threshold,
                use_sam_refinement=use_sam_refinement,
                device=resolved_dev,
                dataset_id=dataset_id,
                filename=name,
                enable_geometry_qa=enable_geometry_qa,
                qwen_gating=qwen_gating,
            )

            if raw_outputs.get("error"):
                # A required model is unavailable -- this will recur for
                # every remaining image, so abort the batch now instead of
                # silently saving empty "no poles found" predictions for 485
                # images and burning the whole run on a dead model.
                batch_job.failed_count += 1
                batch_job.status = "failed"
                batch_job.error_message = raw_outputs["error"]
                batch_job.current_filename = None
                return

            # Persist untouched predictions
            storage_mgr.save_predictions(dataset_id, name, reconciled, raw_outputs)

            batch_job.processed_count += 1
            has_review = any(b.needs_review for b in reconciled)
            if has_review or len(reconciled) == 0:
                batch_job.needs_review_count += 1
            else:
                batch_job.accepted_count += 1

            for b in reconciled:
                batch_job.confidences.append(b.confidence)

        except Exception:
            batch_job.failed_count += 1

    batch_job.status = "completed"
    batch_job.current_filename = None


@app.post("/api/batch/start")
def start_batch_job(req: BatchStartRequest, background_tasks: BackgroundTasks):
    global batch_job
    if batch_job.status == "running":
        raise HTTPException(status_code=400, detail="A batch job is already in progress.")

    meta = storage_mgr.get_dataset(req.dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    imgs = storage_mgr.get_image_list(req.dataset_id)
    if req.only_unlabeled:
        imgs = [img for img in imgs if img.get("status") == "unlabeled"]

    image_names = [img["filename"] for img in imgs]
    if req.filenames is not None:
        eligible = set(image_names)
        image_names = [f for f in req.filenames if f in eligible]
    if not image_names:
        return {"message": "No eligible images found to process.", "total": 0}

    batch_job.reset()
    batch_job.job_id = f"job_{int(time.time())}"
    batch_job.dataset_id = req.dataset_id
    batch_job.total_images = len(image_names)
    batch_job.status = "running"
    batch_job.start_time = time.time()

    background_tasks.add_task(
        _batch_worker,
        req.dataset_id,
        image_names,
        req.mode,
        req.conf_threshold,
        req.use_sam_refinement,
        req.enable_geometry_qa,
        req.qwen_gating,
    )

    return {"message": "Batch auto-labeling job initiated", "job": batch_job.to_dict()}


@app.get("/api/batch/status")
def get_batch_status():
    return {"job": batch_job.to_dict()}


@app.post("/api/batch/pause")
def pause_batch():
    if batch_job.status == "running":
        batch_job.pause_requested = True
    return {"status": batch_job.status}


@app.post("/api/batch/resume")
def resume_batch():
    if batch_job.status == "paused":
        batch_job.pause_requested = False
        batch_job.status = "running"
    return {"status": batch_job.status}


@app.post("/api/batch/cancel")
def cancel_batch():
    batch_job.stop_requested = True
    return {"status": "cancelling"}


# --- FRONTEND STATIC MOUNT (IF FRONTEND DIR EXISTS) ---
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
