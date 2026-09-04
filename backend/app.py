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
from src.geometry_obb import (
    xyxy_to_obb_corners,
    mask_to_obb_corners,
    order_corners_canonical,
    obb_corners_to_xyxy,
)

# Global Manager Instances
storage_mgr = DatasetManager()
active_device = "AUTO"

# Model Adapters
yolo_adapter = YOLOAdapter()
dino_adapter = GroundingDINOAdapter()
sam21_adapter = SAM21Adapter()
sam3_adapter = SAM3Adapter()
qwen_verifier = QwenVerifier()
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
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    use_sam_refinement: bool = True


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


@app.get("/api/datasets/{dataset_id}/masks/{mask_filename}")
def serve_cached_mask(dataset_id: str, mask_filename: str):
    p = storage_mgr.base_dir / dataset_id / "masks" / mask_filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Mask not found.")
    return FileResponse(str(p), media_type="image/png")


# --- AI INFERENCE ENGINE ---

def run_ai_pipeline(
    img_path: Path,
    mode: str = "AI_LABEL",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    use_sam_refinement: bool = True,
    device: str = "AUTO",
    dataset_id: Optional[str] = None,
    filename: Optional[str] = None
) -> Tuple[List[DetectionBox], Dict[str, Any]]:
    """
    Executes detection pipeline:
    - AI_LABEL / DINO_SAM (Primary dataset generation):
        1. Grounding DINO detects candidate bounding boxes.
        2. SAM (SAM 2.1 or SAM 3) segments candidate box -> domain cleanup (_clean).
        3. mask_to_obb_corners computes true canonical rotated OBB.
        4. Mask saved to masks/ directory.
        5. Separate scores (dino_confidence, sam_quality, mask_area) preserved.
        6. Status starts as ai_suggested (needs_review=True).
    - YOLO_FAST / YOLO_ONLY:
        Optional accelerator using trained YOLOv8-OBB.
    - RUN_ALL (Model comparison):
        Executes YOLO + DINO + SAM, preserves raw outputs separately, and reconciles.
    """
    raw_outputs: Dict[str, Any] = {}
    yolo_boxes: List[DetectionBox] = []
    dino_boxes: List[DetectionBox] = []
    sam_boxes: List[DetectionBox] = []

    # 1. PRIMARY WORKFLOW: Grounding DINO -> SAM -> Mask -> True OBB
    if mode in ("AI_LABEL", "DINO_SAM", "DINO_ONLY"):
        if dino_adapter.is_available():
            dino_boxes = dino_adapter.predict(str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            raw_outputs["dino"] = [b.to_dict() for b in dino_boxes]

        result_boxes: List[DetectionBox] = []
        if mode != "DINO_ONLY" and use_sam_refinement and (sam21_adapter.is_available() or sam3_adapter.is_available()):
            # Run SAM on each DINO candidate box (limit to top 5 candidates on CPU)
            sorted_dino = sorted(dino_boxes, key=lambda b: b.confidence, reverse=True)[:5]
            for idx, det in enumerate(sorted_dino):
                mask = None
                new_corners = None
                try:
                    if sam21_adapter.is_available():
                        mask, new_corners = sam21_adapter.segment_and_generate_obb(str(img_path), det.xyxy)
                    elif sam3_adapter.is_available():
                        mask = sam3_adapter.segment_box(str(img_path), det.xyxy)
                        new_corners = mask_to_obb_corners(mask) if mask is not None else None
                except Exception:
                    pass

                mask_rel_path = None
                if mask is not None and dataset_id and filename:
                    try:
                        mask_rel_path = storage_mgr.save_mask(dataset_id, filename, idx, mask)
                    except Exception:
                        pass

                if new_corners is None:
                    new_corners = xyxy_to_obb_corners(*det.xyxy)
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
                    class_id=0,
                    class_name=det.class_name or "utility_pole",
                    model_source="DINO+SAM",
                    needs_review=True,
                    review_reasons=["AI suggested candidate (DINO+SAM)"],
                    attributes={
                        "dino_confidence": round(det.confidence, 3),
                        "sam_quality": sam_quality,
                        "mask_area": mask_area,
                        "mask_path": mask_rel_path,
                        "model_agreement": False,
                        "category": "YELLOW",
                    }
                ))
            return result_boxes, raw_outputs
        else:
            return dino_boxes, raw_outputs

    # 2. OPTIONAL ACCELERATOR: YOLO / YOLO-OBB
    elif mode in ("YOLO_FAST", "YOLO_ONLY"):
        if yolo_adapter.is_available():
            yolo_boxes = yolo_adapter.predict(str(img_path), conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            raw_outputs["yolo"] = [b.to_dict() for b in yolo_boxes]
        if use_sam_refinement and (sam21_adapter.is_available() or sam3_adapter.is_available()):
            for det in yolo_boxes[:3]:
                try:
                    if sam21_adapter.is_available():
                        _, new_corners = sam21_adapter.segment_and_generate_obb(str(img_path), det.xyxy)
                    else:
                        mask = sam3_adapter.segment_box(str(img_path), det.xyxy)
                        new_corners = mask_to_obb_corners(mask) if mask is not None else None
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
        filename=req.filename
    )

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

    if sam21_adapter.is_available():
        mask, corners = sam21_adapter.segment_and_generate_obb(str(img_path), box_tuple)
    elif sam3_adapter.is_available():
        mask = sam3_adapter.segment_box(str(img_path), box_tuple)
        corners = mask_to_obb_corners(mask) if mask is not None else None

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
    use_sam_refinement: bool
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
                device=resolved_dev
            )

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
        req.use_sam_refinement
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
