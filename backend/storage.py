"""
backend/storage.py - Dataset Management & Differential Learning Storage
========================================================================

Handles local filesystem datasets, raw AI prediction preservation, verified human
annotations, binary mask caching, differential learning history tracking, and
YOLO-OBB dataset exports.

Directory Structure:
data/
  datasets/
    <dataset_id>/
      images/           # Raw or imported images
      annotations/      # Verified labels (YOLO-OBB .txt, JSON) - updated only on human save/accept
      predictions/      # Raw untouched AI outputs (YOLO/DINO/SAM) - NEVER overwritten by human edits
      masks/            # Cached binary PNG masks
      history/          # Differential learning records: {image, ai_prediction, human_correction, final_annotation, diff_metrics}
      metadata.json     # Dataset config, image counts, image status index
"""

from __future__ import annotations
import os
import json
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2

from models.adapters.base import DetectionBox
from src.geometry_obb import (
    obb_corners_to_yolo_obb_line,
    yolo_obb_line_to_corners,
    signed_shoelace_area,
    obb_iou,
    xyxy_to_obb_corners,
    order_corners_canonical,
)


def _safe_component(value: str) -> Optional[str]:
    """
    Reduce a user-supplied dataset_id/filename to a bare path segment,
    rejecting anything that could escape the intended directory (path
    separators, '..', or an empty value). Returns None if unsafe.
    """
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        return None
    return value

DEFAULT_DATA_DIR = os.environ.get("OBB_DATA_DIR", "data/datasets")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PRIMARY_OBJECTIVE = (
    "The system is being built to create a high-quality training dataset for a "
    "future single-class YOLOv8-OBB model where 0 = utility_pole. DINO + SAM are used "
    "to generate candidate annotations; human review establishes ground truth; active learning "
    "identifies the most informative positive, negative, and failure examples. The system must "
    "distinguish confirmed negatives from missed positives and must never treat an AI rejection "
    "as evidence that an image is negative without human verification."
)

POSITIVE_CATEGORIES = [
    "clear_positive",
    "small_positive",
    "distant_positive",
    "tilted_positive",
    "occluded_positive",
    "edge_positive",
    "unusual_view_positive",
    "multiple_poles",
    "difficult_positive",
]

NEGATIVE_CATEGORIES = [
    "confirmed_negative",
    "tree",
    "sign_post",
    "street_light",
    "flag_pole",
    "fence_post",
    "building",
    "tower",
    "other_pole_like_object",
]

FAILURE_TYPES = [
    "false_positive",
    "missed_pole",
    "bad_obb",
    "bad_segmentation",
    "ambiguous",
]

DEFAULT_AL_WEIGHTS_PATH = Path("data/config/active_learning_weights.json")

DEFAULT_AL_CONFIG = {
    "weights": {
        "missed_pole": 30,
        "false_positive": 30,
        "dino_sam_disagreement": 25,
        "qwen_disagreement": 25,
        "large_human_correction": 25,
        "poor_sam_mask": 20,
        "bad_obb": 20,
        "hard_negative": 20,
        "small_distant_pole": 15,
        "heavy_occlusion": 15,
        "unusual_viewpoint": 10,
        "unusual_aspect_ratio": 10,
        "image_boundary_pole": 10,
        "confirmed_negative": 0,
        "near_duplicate": -25,
    },
    "duplicate_phash_threshold": 8,
    "max_priority": 100,
}


def compute_phash(img_path: Union[str, Path]) -> str:
    """Compute 64-bit DCT perceptual hash (pHash) for an image."""
    try:
        p = Path(img_path)
        if not p.exists():
            return ""
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        med = float(np.median(dct_low[1:, 1:])) if dct_low.size > 1 else float(np.median(dct_low))
        bits = (dct_low > med).flatten()
        h_val = 0
        for bit in bits:
            h_val = (h_val << 1) | int(bit)
        return f"{h_val:016x}"
    except Exception:
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two 16-character hex hash strings."""
    if not h1 or not h2 or len(h1) != 16 or len(h2) != 16:
        return 64
    try:
        x = int(h1, 16) ^ int(h2, 16)
        return bin(x).count("1")
    except Exception:
        return 64


def compute_obb_correction_metrics(
    ai_box: Dict[str, Any],
    human_box: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute geometrical diff metrics between AI OBB and Human OBB."""
    ai_corners = ai_box.get("corners")
    if not ai_corners and "xyxy" in ai_box:
        ai_corners = xyxy_to_obb_corners(*ai_box["xyxy"]).tolist()

    human_corners = human_box.get("corners")
    if not human_corners and "xyxy" in human_box:
        human_corners = xyxy_to_obb_corners(*human_box["xyxy"]).tolist()

    if not ai_corners or not human_corners:
        return {
            "angle_diff": 0.0,
            "area_diff": 1.0,
            "corner_displacement": 0.0,
            "correction_iou": 1.0,
        }

    pts_ai = order_corners_canonical(np.array(ai_corners, dtype=np.float32))
    pts_human = order_corners_canonical(np.array(human_corners, dtype=np.float32))

    area_ai = abs(float(signed_shoelace_area(pts_ai)))
    area_human = abs(float(signed_shoelace_area(pts_human)))
    area_diff = round(area_human / (area_ai + 1e-6), 4)

    dx_ai, dy_ai = pts_ai[1][0] - pts_ai[0][0], pts_ai[1][1] - pts_ai[0][1]
    dx_h, dy_h = pts_human[1][0] - pts_human[0][0], pts_human[1][1] - pts_human[0][1]
    angle_ai = np.degrees(np.arctan2(dy_ai, dx_ai)) % 180
    angle_h = np.degrees(np.arctan2(dy_h, dx_h)) % 180
    raw_angle_diff = abs(angle_h - angle_ai)
    angle_diff = round(min(raw_angle_diff, 180 - raw_angle_diff), 2)

    corner_disp = round(float(np.mean(np.linalg.norm(pts_human - pts_ai, axis=1))), 2)
    iou_val = round(float(obb_iou(pts_human.tolist(), pts_ai.tolist())), 4)

    return {
        "angle_diff": angle_diff,
        "area_diff": area_diff,
        "corner_displacement": corner_disp,
        "correction_iou": iou_val,
    }


class DatasetManager:
    """
    Manages local datasets, annotations, untouched predictions, and differential learning history.
    """

    def __init__(self, base_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Per-dataset locks serializing metadata.json read-modify-write cycles
        # (batch worker and interactive requests can run concurrently against
        # the same dataset from different threads).
        self._metadata_locks: Dict[str, threading.RLock] = {}
        self._metadata_locks_guard = threading.Lock()

    def _lock_for(self, dataset_id: str) -> threading.RLock:
        """Return the RLock guarding metadata.json read-modify-write for one dataset."""
        with self._metadata_locks_guard:
            lock = self._metadata_locks.get(dataset_id)
            if lock is None:
                lock = threading.RLock()
                self._metadata_locks[dataset_id] = lock
            return lock

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets available in base directory."""
        datasets = []
        for p in self.base_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
                    datasets.append(meta)
                except Exception:
                    continue
        return sorted(datasets, key=lambda d: d.get("updated_at", ""), reverse=True)

    def create_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Initialize a new dataset directory and metadata."""
        clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in dataset_id)
        ds_path = self.base_dir / clean_id
        for sub in ["images", "annotations", "predictions", "masks", "history"]:
            (ds_path / sub).mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "dataset_id": clean_id,
            "name": name or clean_id.replace("_", " ").title(),
            "classes": classes or ["utility_pole"],
            "created_at": now,
            "updated_at": now,
            "image_count": 0,
            "verified_count": 0,
            "ai_suggested_count": 0,
            "unlabeled_count": 0,
            "needs_review_count": 0,
            "images": {}
        }
        (ds_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata."""
        meta_file = self.base_dir / dataset_id / "metadata.json"
        if not meta_file.exists():
            return None
        return json.loads(meta_file.read_text(encoding="utf-8"))

    def _update_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        ds_dir = self.base_dir / dataset_id
        meta_file = ds_dir / "metadata.json"
        # Write atomically (temp file + rename) so a crash mid-write can't
        # leave metadata.json truncated/corrupt for the next reader.
        tmp_file = ds_dir / f".metadata.json.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(tmp_file, meta_file)

    def import_images(self, dataset_id: str, source_paths: List[Union[str, Path]]) -> List[str]:
        """
        Import image files into dataset images/ directory and update metadata.
        """
        ds_path = self.base_dir / dataset_id
        img_dir = ds_path / "images"
        imported: List[str] = []

        for p in source_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            if path_obj.is_dir():
                for f in path_obj.iterdir():
                    if f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        target = img_dir / f.name
                        if not target.exists():
                            shutil.copy2(f, target)
                        imported.append(f.name)
            elif path_obj.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                target = img_dir / path_obj.name
                if not target.exists():
                    shutil.copy2(path_obj, target)
                imported.append(path_obj.name)

        self._register_imported_images(dataset_id, imported)
        return imported

    def import_uploaded_files(self, dataset_id: str, files: List[Tuple[str, bytes]]) -> List[str]:
        """
        Import image files uploaded via the browser (filename, raw bytes) into
        the dataset images/ directory and update metadata. Used by the
        folder/file-picker upload flow, as an alternative to import_images()'s
        server-local directory path.
        """
        ds_path = self.base_dir / dataset_id
        img_dir = ds_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        imported: List[str] = []

        for raw_name, content in files:
            # Browser folder uploads may send a relative path (e.g. "sub/img.jpg");
            # flatten to a bare, sanitized filename before touching the filesystem.
            name = _safe_component(Path(raw_name).name)
            if not name or Path(name).suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            target = img_dir / name
            if not target.exists():
                target.write_bytes(content)
            imported.append(name)

        self._register_imported_images(dataset_id, imported)
        return imported

    def _register_imported_images(self, dataset_id: str, imported: List[str]) -> None:
        ds_path = self.base_dir / dataset_id
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if not meta:
                meta = self.create_dataset(dataset_id)

            # Refresh images in metadata
            images_dict = meta.setdefault("images", {})
            for name in imported:
                if name not in images_dict:
                    stem = Path(name).stem
                    # Check if annotation already exists
                    ann_path = ds_path / "annotations" / f"{stem}.json"
                    pred_path = ds_path / "predictions" / f"{stem}.json"
                    status = "unlabeled"
                    if ann_path.exists():
                        status = "verified"
                    elif pred_path.exists():
                        status = "ai_suggested"

                    images_dict[name] = {
                        "filename": name,
                        "stem": stem,
                        "status": status,
                        "needs_review": False,
                        "confidence": None,
                        "annotation_count": 0,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }

            self._recount_stats(meta)
            self._update_metadata(dataset_id, meta)

    def _recount_stats(self, meta: Dict[str, Any]) -> None:
        images = meta.get("images", {})
        meta["image_count"] = len(images)
        meta["accepted_count"] = sum(1 for v in images.values() if v.get("status") == "accepted")
        meta["human_corrected_count"] = sum(1 for v in images.values() if v.get("status") == "human_corrected")
        meta["verified_count"] = (
            meta["accepted_count"] + 
            meta["human_corrected_count"] + 
            sum(1 for v in images.values() if v.get("status") == "verified")
        )
        meta["rejected_count"] = sum(1 for v in images.values() if v.get("status") == "rejected")
        meta["auto_pass_count"] = sum(1 for v in images.values() if v.get("status") == "auto_pass")
        meta["ai_suggested_count"] = sum(1 for v in images.values() if v.get("status") == "ai_suggested")
        meta["unlabeled_count"] = sum(1 for v in images.values() if v.get("status") == "unlabeled")
        meta["skipped_count"] = sum(1 for v in images.values() if v.get("status") == "skipped")
        meta["needs_review_count"] = sum(1 for v in images.values() if v.get("needs_review"))

    def mark_skipped(self, dataset_id: str, filename: str) -> None:
        """Part 13: mark an image as skipped (uncertain, deferred for later
        review) without touching its predictions/annotations -- a distinct
        status from unlabeled/verified/rejected so the review queue can filter
        on it separately."""
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if not meta or filename not in meta.get("images", {}):
                return
            meta["images"][filename]["status"] = "skipped"
            meta["images"][filename]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._recount_stats(meta)
            self._update_metadata(dataset_id, meta)

    def get_image_path(self, dataset_id: str, filename: str) -> Optional[Path]:
        safe_id = _safe_component(dataset_id)
        safe_name = _safe_component(filename)
        if not safe_id or not safe_name:
            return None
        p = self.base_dir / safe_id / "images" / safe_name
        return p if p.exists() else None

    def get_cached_mask_file_path(self, dataset_id: str, mask_filename: str) -> Optional[Path]:
        """Resolve a bare mask filename (from the mask-serving URL) to its file under masks/."""
        safe_id = _safe_component(dataset_id)
        safe_name = _safe_component(mask_filename)
        if not safe_id or not safe_name:
            return None
        p = self.base_dir / safe_id / "masks" / safe_name
        return p if p.exists() else None

    def get_image_list(
        self,
        dataset_id: str,
        filter_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        meta = self.get_dataset(dataset_id)
        if not meta:
            return []
        imgs = list(meta.get("images", {}).values())
        if filter_status:
            imgs = [img for img in imgs if img.get("status") == filter_status]
        return sorted(imgs, key=lambda x: x.get("filename", ""))

    # --- RAW PREDICTIONS (NEVER OVERWRITTEN BY HUMAN EDITS) ---

    def _build_spec_summary(self, filename: str, boxes: List[DetectionBox], raw_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Part 12: a flat, spec-shaped view of the same information already in
        `boxes`/`raw_outputs` -- derived at save time, not a second source of
        truth. `boxes` (with per-candidate .attributes) remains authoritative;
        this is purely a convenience projection for consumers that expect the
        literal dino/sam3/geometry/qwen/obb/final_status shape from the spec.
        """
        geometry_entries, qwen_entries, obb_entries = [], [], []
        any_needs_review = False
        for b in boxes:
            attrs = b.attributes or {}
            any_needs_review = any_needs_review or b.needs_review
            geometry_entries.append({
                "geometry_score": attrs.get("geometry_score"),
                "geometry_status": attrs.get("geometry_status"),
                "geometry_flags": attrs.get("geometry_flags", []),
            })
            qwen_entries.append({
                "ran": attrs.get("qwen_ran", False),
                "class": attrs.get("qwen_class"),
                "material": attrs.get("qwen_material"),
                "visibility": attrs.get("qwen_visibility"),
                "semantic_confidence": attrs.get("qwen_semantic_confidence"),
                "reason": attrs.get("qwen_reason"),
            })
            obb_entries.append({
                "corners": b.corners,
                "source": attrs.get("obb_source"),
            })

        if not boxes:
            final_status = "no_candidates"
        elif any_needs_review:
            final_status = "needs_review"
        else:
            final_status = "reviewed_clean"

        return {
            "image": filename,
            "dino": raw_outputs.get("dino", []),
            "sam3": raw_outputs.get("sam3_candidates", raw_outputs.get("sam3", [])),
            "geometry": geometry_entries,
            "qwen": qwen_entries,
            "obb": obb_entries,
            "final_status": final_status,
        }

    def save_predictions(
        self,
        dataset_id: str,
        filename: str,
        boxes: List[DetectionBox],
        raw_outputs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Persist untouched raw model outputs in predictions/.
        Never modified by human edits.
        """
        stem = Path(filename).stem
        pred_path = self.base_dir / dataset_id / "predictions" / f"{stem}.json"

        record = {
            "dataset_id": dataset_id,
            "filename": filename,
            "stem": stem,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "boxes": [b.to_dict() for b in boxes],
            "raw_outputs": raw_outputs or {},
            "summary": self._build_spec_summary(filename, boxes, raw_outputs or {}),
        }
        pred_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        # Update metadata to ai_suggested if currently unlabeled
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if meta and filename in meta.get("images", {}):
                entry = meta["images"][filename]
                if entry.get("status") != "verified":
                    entry["status"] = "ai_suggested"
                entry["needs_review"] = any(b.needs_review for b in boxes)
                confs = [b.confidence for b in boxes]
                entry["confidence"] = round(float(np.mean(confs)), 3) if confs else 0.0
                entry["annotation_count"] = len(boxes)
                self._recount_stats(meta)
                self._update_metadata(dataset_id, meta)

    def get_predictions(self, dataset_id: str, filename: str) -> Optional[Dict[str, Any]]:
        stem = Path(filename).stem
        pred_path = self.base_dir / dataset_id / "predictions" / f"{stem}.json"
        if not pred_path.exists():
            return None
        return json.loads(pred_path.read_text(encoding="utf-8"))

    # --- MASK CACHING ---

    def save_mask(
        self,
        dataset_id: str,
        filename: str,
        box_index: int,
        mask: np.ndarray
    ) -> str:
        """
        Save binary segmentation mask PNG to masks/ directory.
        Returns relative path to mask.
        """
        stem = Path(filename).stem
        mask_dir = self.base_dir / dataset_id / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{stem}_{box_index}.png"
        mask_path = mask_dir / mask_name
        m_uint8 = (mask > 0).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
        cv2.imwrite(str(mask_path), m_uint8)
        return f"masks/{mask_name}"

    def get_mask_path(self, dataset_id: str, mask_rel_path: str) -> Optional[Path]:
        p = self.base_dir / dataset_id / mask_rel_path
        return p if p.exists() else None

    # --- VERIFIED ANNOTATIONS & DIFFERENTIAL LEARNING ---

    def save_annotation(
        self,
        dataset_id: str,
        filename: str,
        boxes: List[DetectionBox],
        img_width: int,
        img_height: int,
        is_human: bool = True,
        action: Optional[str] = None,
        al_categories: Optional[List[str]] = None,
        failure_type: Optional[str] = None,
        negative_category: Optional[str] = None,
        is_negative: bool = False,
        obb_correction_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save verified labels to annotations/. Computes differential against raw predictions
        and archives changes into history/ for active/differential learning.
        Supports rich active learning categorization, negative class tracking, and test set separation.
        """
        stem = Path(filename).stem
        ds_path = self.base_dir / dataset_id
        ann_dir = ds_path / "annotations"
        ann_json = ann_dir / f"{stem}.json"
        ann_txt = ann_dir / f"{stem}.txt"

        # 1. Read existing raw AI predictions if present
        ai_preds = self.get_predictions(dataset_id, filename)

        # 2. Compute differential metrics
        diff_record = self._compute_diff(ai_preds, boxes, filename, action)
        if diff_record:
            hist_dir = ds_path / "history"
            timestamp_str = int(time.time() * 1000)
            hist_file = hist_dir / f"{stem}_{timestamp_str}.json"
            hist_file.write_text(json.dumps(diff_record, indent=2), encoding="utf-8")

        # Determine target status: accepted, human_corrected, rejected, or auto_pass
        if action == "rejected":
            target_status = "rejected"
        elif action in ("accepted", "human_corrected"):
            target_status = action
        elif not is_human:
            target_status = "auto_pass"
        else:
            has_edits = diff_record and (
                diff_record.get("modified_count", 0) > 0 or 
                diff_record.get("added_count", 0) > 0 or 
                diff_record.get("deleted_count", 0) > 0
            )
            target_status = "human_corrected" if has_edits else "accepted"

        # 3. Save verified JSON annotation
        record = {
            "dataset_id": dataset_id,
            "filename": filename,
            "stem": stem,
            "image_width": img_width,
            "image_height": img_height,
            "status": target_status,
            "verified_by": "human" if is_human else "auto",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "boxes": [b.to_dict() for b in boxes],
            "al_categories": al_categories or [],
            "failure_type": failure_type,
            "negative_category": negative_category,
            "is_negative": is_negative,
            "obb_correction_metrics": obb_correction_metrics,
        }
        ann_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

        # 4. Save standard YOLO-OBB .txt annotation
        # For confirmed negatives (is_negative=True with 0 boxes), save empty .txt for YOLO background training
        if target_status != "rejected" or is_negative:
            if len(boxes) == 0 and is_negative:
                ann_txt.write_text("", encoding="utf-8")
            elif len(boxes) > 0:
                lines = []
                for b in boxes:
                    corners = b.corners
                    if corners is None:
                        corners = xyxy_to_obb_corners(*b.xyxy).tolist()
                    line = obb_corners_to_yolo_obb_line(
                        corners, img_width, img_height, class_id=b.class_id
                    )
                    lines.append(line)
                ann_txt.write_text("\n".join(lines), encoding="utf-8")
            elif ann_txt.exists():
                ann_txt.unlink()
        elif ann_txt.exists():
            ann_txt.unlink()

        # 5. Update metadata status
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if meta and filename in meta.get("images", {}):
                entry = meta["images"][filename]
                entry["status"] = target_status
                entry["needs_review"] = False
                entry["annotation_count"] = len(boxes) if target_status != "rejected" else 0

                # Store AL metadata
                if al_categories is not None:
                    existing_cats = set(entry.get("al_categories", []))
                    existing_cats.update(al_categories)
                    entry["al_categories"] = sorted(list(existing_cats))
                if failure_type is not None:
                    entry["failure_type"] = failure_type
                if negative_category is not None:
                    entry["negative_category"] = negative_category
                if is_negative:
                    entry["is_negative"] = True
                if obb_correction_metrics is not None:
                    entry["obb_correction_metrics"] = obb_correction_metrics

                if diff_record and "difficulty" in diff_record:
                    entry["difficulty_score"] = diff_record["difficulty"].get("score", 0.1)
                    entry["difficulty_reasons"] = diff_record["difficulty"].get("reasons", [])
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._recount_stats(meta)
                self._update_metadata(dataset_id, meta)

        return record

    def get_annotation(self, dataset_id: str, filename: str) -> Optional[Dict[str, Any]]:
        stem = Path(filename).stem
        ann_json = self.base_dir / dataset_id / "annotations" / f"{stem}.json"
        if not ann_json.exists():
            return None
        return json.loads(ann_json.read_text(encoding="utf-8"))

    def _compute_diff(
        self,
        ai_preds: Optional[Dict[str, Any]],
        human_boxes: List[DetectionBox],
        filename: str,
        action: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Compute differences between untouched AI suggestions and human annotations."""
        if not ai_preds or "boxes" not in ai_preds:
            return None

        ai_box_list = [DetectionBox.from_dict(b) for b in ai_preds["boxes"]]
        matched_ai = set()
        matched_human = set()
        modifications = []

        for h_idx, h_box in enumerate(human_boxes):
            best_iou = 0.0
            best_ai_idx = -1
            for a_idx, a_box in enumerate(ai_box_list):
                if a_idx in matched_ai:
                    continue
                if h_box.corners and a_box.corners:
                    iou_val = obb_iou(h_box.corners, a_box.corners)
                else:
                    iou_val = obb_iou(
                        xyxy_to_obb_corners(*h_box.xyxy),
                        xyxy_to_obb_corners(*a_box.xyxy)
                    )
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_ai_idx = a_idx

            if best_iou >= 0.20 and best_ai_idx >= 0:
                matched_human.add(h_idx)
                matched_ai.add(best_ai_idx)
                ai_dict = ai_box_list[best_ai_idx].to_dict()
                h_dict = h_box.to_dict()
                geom_metrics = compute_obb_correction_metrics(ai_dict, h_dict)
                modifications.append({
                    "ai_box": ai_dict,
                    "human_box": h_dict,
                    "iou": round(float(best_iou), 4),
                    "modified": best_iou < 0.95,
                    "metrics": geom_metrics,
                })

        added_by_human = [
            human_boxes[i].to_dict() for i in range(len(human_boxes)) if i not in matched_human
        ]
        deleted_ai = [
            ai_box_list[i].to_dict() for i in range(len(ai_box_list)) if i not in matched_ai
        ]

        # Calculate active learning difficulty score & reasons
        ious = [m["iou"] for m in modifications]
        avg_iou = float(np.mean(ious)) if ious else 1.0
        min_iou = float(min(ious)) if ious else 1.0

        difficulty_reasons = []
        difficulty_score = 0.10

        if action == "rejected":
            difficulty_reasons.append("false_positive")
            difficulty_reasons.append("human_rejected_all")
            difficulty_score += 0.60
        if deleted_ai:
            difficulty_reasons.append("false_positive")
            difficulty_reasons.append("false_positive_ai_prediction")
            difficulty_score += 0.35
        if added_by_human:
            difficulty_reasons.append("missed_pole")
            difficulty_score += 0.35
        if modifications and any(m["modified"] for m in modifications):
            difficulty_reasons.append("large_human_correction")
            difficulty_score += round(0.40 * (1.0 - min_iou), 2)
            # Detect bad OBB orientation or significant shape discrepancy
            for m in modifications:
                met = m.get("metrics", {})
                if met.get("angle_diff", 0) > 15.0 or met.get("correction_iou", 1.0) < 0.70:
                    difficulty_reasons.append("bad_obb")
                    break

        difficulty_reasons = list(dict.fromkeys(difficulty_reasons))
        difficulty_score = min(0.99, max(0.05, round(difficulty_score, 2)))

        return {
            "image": filename,
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ai": {
                "detector": "grounding-dino",
                "dino_box": [b.xyxy for b in ai_box_list],
                "sam_model": "sam2.1",
                "mask_path": [b.attributes.get("mask_path") for b in ai_box_list if b.attributes.get("mask_path")],
                "ai_obb": [b.corners for b in ai_box_list if b.corners],
            },
            "verification": {
                "qwen_decision": "review" if any(b.needs_review for b in ai_box_list) else "accept"
            },
            "human": {
                "action": action or ("corrected" if any(m["modified"] for m in modifications) else "accepted"),
                "final_obb": [b.corners for b in human_boxes if b.corners],
            },
            "metrics": {
                "correction_iou": round(min_iou, 4) if modifications else 1.0,
                "avg_iou": round(avg_iou, 4) if modifications else 1.0,
            },
            "difficulty": {
                "score": difficulty_score,
                "reasons": difficulty_reasons,
            },
            "ai_box_count": len(ai_box_list),
            "human_box_count": len(human_boxes),
            "added_count": len(added_by_human),
            "deleted_count": len(deleted_ai),
            "modified_count": sum(1 for m in modifications if m["modified"]),
            "modifications": modifications,
            "added_by_human": added_by_human,
            "deleted_ai_suggestions": deleted_ai,
        }

    def get_history(self, dataset_id: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve differential history records."""
        hist_dir = self.base_dir / dataset_id / "history"
        if not hist_dir.exists():
            return []
        records = []
        stem = Path(filename).stem if filename else None
        for p in hist_dir.iterdir():
            if p.suffix == ".json":
                if stem and not p.name.startswith(stem):
                    continue
                try:
                    records.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)

    # --- ACTIVE LEARNING & DATASET COMPOSITION ENGINE ---

    def _load_al_weights(self) -> Dict[str, Any]:
        """Load active learning weights configuration or return defaults."""
        if DEFAULT_AL_WEIGHTS_PATH.exists():
            try:
                return json.loads(DEFAULT_AL_WEIGHTS_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return DEFAULT_AL_CONFIG

    def _save_al_weights(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save active learning weights configuration with deep merge."""
        existing = dict(self._load_al_weights())
        if "weights" in config_data and isinstance(config_data["weights"], dict):
            merged_w = dict(existing.get("weights", DEFAULT_AL_CONFIG["weights"]))
            merged_w.update(config_data["weights"])
            existing["weights"] = merged_w
        if config_data.get("duplicate_phash_threshold") is not None:
            existing["duplicate_phash_threshold"] = config_data["duplicate_phash_threshold"]
        if config_data.get("max_priority") is not None:
            existing["max_priority"] = config_data["max_priority"]

        DEFAULT_AL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_AL_WEIGHTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return existing

    def compute_dataset_phashes(self, dataset_id: str) -> Dict[str, str]:
        """Compute or retrieve perceptual hash (pHash) for all images in dataset."""
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if not meta:
                return {}
            images_dict = meta.get("images", {})
            ds_img_dir = self.base_dir / dataset_id / "images"
            hashes = {}
            updated = False

            for filename, entry in images_dict.items():
                h = entry.get("phash")
                if not h:
                    img_p = ds_img_dir / filename
                    if img_p.exists():
                        h = compute_phash(img_p)
                        if h:
                            entry["phash"] = h
                            updated = True
                if h:
                    hashes[filename] = h

            if updated:
                self._update_metadata(dataset_id, meta)

            return hashes

    def find_near_duplicates(
        self,
        dataset_id: str,
        threshold: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify near-duplicate images using pHash Hamming distance.
        Flags duplicates with a priority penalty, preserving all images (no deletion).
        """
        cfg = self._load_al_weights()
        thresh = threshold if threshold is not None else cfg.get("duplicate_phash_threshold", 8)

        with self._lock_for(dataset_id):
            hashes = self.compute_dataset_phashes(dataset_id)
            filenames = list(hashes.keys())
            meta = self.get_dataset(dataset_id)
            if not meta:
                return []

            images_dict = meta.get("images", {})
            duplicates = []
            flagged_as_duplicate = set()

            for i in range(len(filenames)):
                f1 = filenames[i]
                h1 = hashes[f1]
                for j in range(i + 1, len(filenames)):
                    f2 = filenames[j]
                    h2 = hashes[f2]
                    dist = hamming_distance(h1, h2)
                    if dist <= thresh:
                        duplicates.append({
                            "primary": f1,
                            "duplicate": f2,
                            "distance": dist,
                        })
                        flagged_as_duplicate.add(f2)
                        if f2 in images_dict:
                            images_dict[f2]["is_duplicate"] = True
                            images_dict[f2]["duplicate_of"] = f1
                            images_dict[f2]["duplicate_distance"] = dist

            # Reset flag for non-duplicates
            for fn, entry in images_dict.items():
                if fn not in flagged_as_duplicate and entry.get("is_duplicate"):
                    entry["is_duplicate"] = False
                    entry.pop("duplicate_of", None)
                    entry.pop("duplicate_distance", None)

            self._update_metadata(dataset_id, meta)
            return duplicates

    def reject_with_metadata(
        self,
        dataset_id: str,
        filename: str,
        failure_type: str = "false_positive",
        negative_category: Optional[str] = None,
        al_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Reject an image annotation while storing failure type and negative class metadata.
        Preserves untouched raw predictions in predictions/ for false-positive active learning.
        """
        if negative_category == "confirmed_negative" and failure_type == "false_positive":
            failure_type = None

        al_cats = list(al_categories or [])
        if failure_type and failure_type not in al_cats:
            al_cats.append(failure_type)
        if negative_category and negative_category not in al_cats:
            al_cats.append(negative_category)

        is_neg = (negative_category is not None) or (failure_type == "false_positive")

        return self.save_annotation(
            dataset_id=dataset_id,
            filename=filename,
            boxes=[],
            img_width=0,
            img_height=0,
            is_human=True,
            action="rejected",
            al_categories=al_cats,
            failure_type=failure_type,
            negative_category=negative_category,
            is_negative=is_neg,
        )

    def save_missed_pole(
        self,
        dataset_id: str,
        filename: str,
        boxes: List[DetectionBox],
        img_width: int,
        img_height: int,
        al_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Save an annotation where AI missed utility pole(s) and human added them.
        Tags failure_type='missed_pole' for high-priority active learning.
        """
        cats = list(al_categories or [])
        if "missed_pole" not in cats:
            cats.append("missed_pole")
        return self.save_annotation(
            dataset_id=dataset_id,
            filename=filename,
            boxes=boxes,
            img_width=img_width,
            img_height=img_height,
            is_human=True,
            action="human_corrected",
            al_categories=cats,
            failure_type="missed_pole",
        )

    def toggle_test_set(self, dataset_id: str, filename: str) -> bool:
        """Toggle test set status for an image. Isolated from active learning exports."""
        with self._lock_for(dataset_id):
            meta = self.get_dataset(dataset_id)
            if not meta or filename not in meta.get("images", {}):
                raise ValueError(f"Image {filename} not found in dataset {dataset_id}")
            entry = meta["images"][filename]
            current = entry.get("is_test_set", False)
            entry["is_test_set"] = not current
            self._update_metadata(dataset_id, meta)
            return entry["is_test_set"]

    def get_test_set(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Return list of images marked as belonging to the fixed test set."""
        meta = self.get_dataset(dataset_id)
        if not meta:
            return []
        test_images = []
        for fn, entry in meta.get("images", {}).items():
            if entry.get("is_test_set"):
                test_images.append({
                    "filename": fn,
                    "stem": entry.get("stem", Path(fn).stem),
                    "status": entry.get("status", "unlabeled"),
                    "al_categories": entry.get("al_categories", []),
                    "annotation_count": entry.get("annotation_count", 0),
                })
        return test_images

    def get_coverage_warnings(self, comp: Dict[str, Any]) -> List[str]:
        """Generate dataset balance and coverage warnings based on verified data distributions."""
        warnings = []
        total = comp.get("total_images", 0)
        positives = comp.get("positive_poles", 0)
        negatives = comp.get("confirmed_negatives", 0) + comp.get("hard_negatives", 0)
        by_cat = comp.get("by_category", {})
        by_neg = comp.get("by_negative", {})

        # 1. Occlusion coverage warning
        if positives > 0:
            occluded = by_cat.get("occluded_positive", 0)
            occluded_pct = (occluded / positives) * 100.0
            if occluded_pct < 10.0:
                warnings.append(f"⚠ Warning: Only {occluded_pct:.1f}% of positives are heavily occluded (recommended >= 10%).")

            # 2. Tilted / edge pole warning
            tilted_edge = by_cat.get("tilted_positive", 0) + by_cat.get("edge_positive", 0)
            tilted_pct = (tilted_edge / positives) * 100.0
            if tilted_pct < 8.0:
                warnings.append(f"⚠ Warning: Tilted and edge poles are underrepresented ({tilted_pct:.1f}%, recommended >= 8%).")

        # 3. Hard negative representations
        if total >= 10:
            sl_count = by_neg.get("street_light", 0)
            if sl_count < 2:
                warnings.append("⚠ Warning: Street-light hard negatives are underrepresented.")

            tree_count = by_neg.get("tree", 0)
            if tree_count < 2:
                warnings.append("⚠ Warning: Tree trunk hard negatives are underrepresented.")

            sign_count = by_neg.get("sign_post", 0)
            if sign_count < 2:
                warnings.append("⚠ Warning: Sign post hard negatives are underrepresented.")

        # 4. Redundant images / duplicates warning
        dups = comp.get("near_duplicates", 0)
        if total > 0:
            dup_pct = (dups / total) * 100.0
            if dup_pct > 15.0:
                warnings.append(f"⚠ Warning: Too many near-duplicate images from the same scene ({dup_pct:.1f}% redundant frames).")

        # 5. Dataset balance warning (negatives vs positives)
        if total >= 15:
            neg_ratio = (negatives / total) * 100.0
            if neg_ratio < 10.0:
                warnings.append(f"⚠ Warning: Negatives constitute only {neg_ratio:.1f}% of dataset (recommend 15-25% background/hard negatives).")
            elif neg_ratio > 50.0:
                warnings.append(f"⚠ Warning: Negatives constitute {neg_ratio:.1f}% of dataset (imbalance towards non-pole background).")

        return warnings

    def get_dataset_composition(self, dataset_id: str) -> Dict[str, Any]:
        """
        Analyze dataset balance and coverage across positive classes, hard negatives,
        failure types, and test set separation.
        """
        meta = self.get_dataset(dataset_id)
        if not meta:
            return {}

        images_dict = meta.get("images", {})
        total = len(images_dict)
        positives = 0
        hard_positives = 0
        confirmed_negatives = 0
        hard_negatives = 0
        false_positives = 0
        missed_poles = 0
        bad_obb_cases = 0
        test_set_count = 0
        near_duplicates = 0
        unreviewed = 0

        by_cat: Dict[str, int] = {}
        by_fail: Dict[str, int] = {}
        by_neg: Dict[str, int] = {}

        hard_pos_tags = {
            "small_positive", "distant_positive", "tilted_positive",
            "occluded_positive", "edge_positive", "unusual_view_positive",
            "multiple_poles", "difficult_positive"
        }
        hard_neg_tags = {
            "tree", "sign_post", "street_light", "flag_pole",
            "fence_post", "building", "tower", "other_pole_like_object"
        }

        for fn, item in images_dict.items():
            status = item.get("status", "unlabeled")
            al_cats = item.get("al_categories", [])
            ft = item.get("failure_type")
            nc = item.get("negative_category")
            is_neg = item.get("is_negative", False)
            ann_count = item.get("annotation_count", 0)

            if item.get("is_test_set"):
                test_set_count += 1
            if item.get("is_duplicate"):
                near_duplicates += 1

            if status in ("unlabeled", "needs_review", "ai_suggested"):
                unreviewed += 1

            for c in al_cats:
                by_cat[c] = by_cat.get(c, 0) + 1

            if ft:
                by_fail[ft] = by_fail.get(ft, 0) + 1
            if nc:
                by_neg[nc] = by_neg.get(nc, 0) + 1

            if ft == "false_positive":
                false_positives += 1
            elif ft == "missed_pole":
                missed_poles += 1
            elif ft == "bad_obb":
                bad_obb_cases += 1

            if nc in hard_neg_tags:
                hard_negatives += 1
            elif nc == "confirmed_negative" or (is_neg and ann_count == 0):
                confirmed_negatives += 1

            if ann_count > 0 and status in ("verified", "accepted", "human_corrected"):
                positives += 1
                if any(t in al_cats for t in hard_pos_tags):
                    hard_positives += 1

        comp = {
            "dataset_id": dataset_id,
            "total_images": total,
            "positive_poles": positives,
            "hard_positives": hard_positives,
            "confirmed_negatives": confirmed_negatives,
            "hard_negatives": hard_negatives,
            "false_positives": false_positives,
            "missed_poles": missed_poles,
            "bad_obb_cases": bad_obb_cases,
            "test_set_count": test_set_count,
            "near_duplicates": near_duplicates,
            "unreviewed_count": unreviewed,
            "by_category": by_cat,
            "by_failure": by_fail,
            "by_negative": by_neg,
        }
        comp["warnings"] = self.get_coverage_warnings(comp)
        return comp

    def get_active_learning_queue(
        self,
        dataset_id: str,
        filter_reason: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank images by active learning priority score using configurable weights.
        - confirmed_negative = baseline data (0 priority boost)
        - hard_negative = active-learning data (boost +20)
        - false_positive = high-priority active-learning data (boost +30)
        - missed_pole = high-priority active-learning data (boost +30)
        - duplicate penalty = -25
        """
        meta = self.get_dataset(dataset_id)
        if not meta:
            return []

        cfg = self._load_al_weights()
        weights = cfg.get("weights", DEFAULT_AL_CONFIG["weights"])
        max_priority = cfg.get("max_priority", 100)

        images_dict = meta.get("images", {})
        queue = []

        for filename, item in images_dict.items():
            stem = Path(filename).stem
            status = item.get("status", "unlabeled")
            al_cats = item.get("al_categories", [])
            ft = item.get("failure_type")
            nc = item.get("negative_category")
            is_dup = item.get("is_duplicate", False)
            is_neg = item.get("is_negative", False)
            diff_reasons = list(item.get("difficulty_reasons", []))

            reasons = list(set(diff_reasons + al_cats))
            if ft and ft not in reasons:
                reasons.append(ft)
            if nc and nc not in reasons:
                reasons.append(nc)

            base_score = 0
            if (ft == "false_positive" or "false_positive" in reasons) and nc != "confirmed_negative":
                base_score += weights.get("false_positive", 30)
            if ft == "missed_pole" or "missed_pole" in reasons:
                base_score += weights.get("missed_pole", 30)
            if ft == "bad_obb" or "bad_obb" in reasons:
                base_score += weights.get("bad_obb", 20)
            if nc in NEGATIVE_CATEGORIES and nc != "confirmed_negative":
                base_score += weights.get("hard_negative", 20)
            if nc == "confirmed_negative" or (is_neg and not ft and not nc):
                base_score += weights.get("confirmed_negative", 0)

            if "occluded_positive" in al_cats:
                base_score += weights.get("heavy_occlusion", 15)
            if "small_positive" in al_cats or "distant_positive" in al_cats:
                base_score += weights.get("small_distant_pole", 15)
            if "unusual_view_positive" in al_cats:
                base_score += weights.get("unusual_viewpoint", 10)
            if "edge_positive" in al_cats:
                base_score += weights.get("image_boundary_pole", 10)

            if status in ("unlabeled", "needs_review", "ai_suggested"):
                preds = self.get_predictions(dataset_id, filename)
                if preds and "boxes" in preds:
                    boxes = preds["boxes"]
                    confs = [b.get("confidence", 0.5) for b in boxes]
                    avg_conf = float(np.mean(confs)) if confs else 0.0
                    if any(b.get("needs_review") for b in boxes):
                        base_score += weights.get("dino_sam_disagreement", 25)
                        reasons.append("dino_sam_disagreement")
                    if any(b.get("attributes", {}).get("sam_quality") == "poor" for b in boxes):
                        base_score += weights.get("poor_sam_mask", 20)
                        reasons.append("poor_sam_mask")
                    if avg_conf < 0.45:
                        base_score += 15
                        reasons.append("low_candidate_confidence")
                else:
                    if status == "unlabeled":
                        base_score += 20
                        reasons.append("unlabeled")

            if is_dup:
                base_score += weights.get("near_duplicate", -25)
                reasons.append("near_duplicate")

            priority = int(min(max_priority, max(0, base_score)))
            calc_diff_score = round(priority / 100.0, 2)

            if filter_reason and filter_reason.lower() != "all":
                norm_filter = filter_reason.lower().replace(" ", "_")
                if norm_filter == "high_priority":
                    if priority < 60:
                        continue
                elif norm_filter == "hard_negatives":
                    if nc not in NEGATIVE_CATEGORIES or nc == "confirmed_negative":
                        continue
                elif norm_filter == "false_positives":
                    if ft != "false_positive" and "false_positive" not in reasons:
                        continue
                elif norm_filter == "missed_poles":
                    if ft != "missed_pole" and "missed_pole" not in reasons:
                        continue
                elif norm_filter == "bad_obb":
                    if ft != "bad_obb" and "bad_obb" not in reasons:
                        continue
                elif norm_filter == "duplicates":
                    if not is_dup:
                        continue
                elif not any(norm_filter in r.lower() for r in reasons):
                    continue

            queue.append({
                "filename": filename,
                "stem": stem,
                "status": status,
                "priority": priority,
                "difficulty_score": calc_diff_score,
                "reasons": sorted(list(set(reasons))),
                "annotation_count": item.get("annotation_count", 0),
                "is_test_set": item.get("is_test_set", False),
                "is_duplicate": is_dup,
                "failure_type": ft,
                "negative_category": nc,
            })

        return sorted(queue, key=lambda x: x["priority"], reverse=True)

    def get_next_difficult(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Find highest-priority image needing human review."""
        q = self.get_active_learning_queue(dataset_id)
        unreviewed = [img for img in q if img["status"] in ("needs_review", "ai_suggested", "unlabeled")]
        if unreviewed:
            return unreviewed[0]
        return q[0] if q else None

    def export_hard_cases(
        self,
        dataset_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        min_priority: int = 60
    ) -> Dict[str, Any]:
        """Export high-difficulty subset for active learning evaluation."""
        q = self.get_active_learning_queue(dataset_id)
        hard_items = [item for item in q if item["priority"] >= min_priority]
        out_root = Path(output_dir or (self.base_dir / dataset_id / "export" / "hard_cases")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)

        ds_path = self.base_dir / dataset_id
        exported = 0
        for item in hard_items:
            fname = item["filename"]
            stem = item["stem"]
            src_img = ds_path / "images" / fname
            src_txt = ds_path / "annotations" / f"{stem}.txt"
            if src_img.exists():
                shutil.copy2(src_img, out_root / "images" / fname)
                if src_txt.exists():
                    shutil.copy2(src_txt, out_root / "labels" / f"{stem}.txt")
                exported += 1

        manifest = {
            "dataset_id": dataset_id,
            "min_priority": min_priority,
            "count": exported,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "items": hard_items,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"export_path": str(out_root), "count": exported}

    # --- DATASET EXPORT ENGINE ---

    def export_dataset(
        self,
        dataset_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        train_val_split: float = 0.8
    ) -> Dict[str, Any]:
        """
        Export verified annotations to standard YOLO-OBB dataset format with data.yaml.
        - Excludes fixed test set images to prevent contamination.
        - Writes empty .txt for confirmed negative images (standard YOLO background images).
        """
        meta = self.get_dataset(dataset_id)
        if not meta:
            raise ValueError(f"Dataset {dataset_id} does not exist.")

        out_root = Path(output_dir or (self.base_dir / dataset_id / "export")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val"]:
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        verified_images = [
            k for k, v in meta.get("images", {}).items() 
            if v.get("status") in ("verified", "accepted", "human_corrected")
            and not v.get("is_test_set", False)  # CRITICAL: Exclude fixed test set!
        ]

        # Also include confirmed negative images that were verified
        for k, v in meta.get("images", {}).items():
            if v.get("is_negative") and v.get("status") in ("verified", "accepted", "rejected") and not v.get("is_test_set", False):
                if k not in verified_images:
                    verified_images.append(k)

        np.random.seed(42)
        shuffled = list(verified_images)
        np.random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_val_split)
        train_set = set(shuffled[:split_idx])

        exported_count = 0
        validation_errors = []

        ds_path = self.base_dir / dataset_id
        for name in verified_images:
            split = "train" if name in train_set else "val"
            stem = Path(name).stem
            src_img = ds_path / "images" / name
            src_txt = ds_path / "annotations" / f"{stem}.txt"
            img_entry = meta.get("images", {}).get(name, {})

            if not src_img.exists():
                continue

            is_negative_img = img_entry.get("is_negative", False) or (img_entry.get("negative_category") == "confirmed_negative")

            # For confirmed negative images with no txt or empty txt
            if is_negative_img and (not src_txt.exists() or src_txt.read_text(encoding="utf-8").strip() == ""):
                shutil.copy2(src_img, out_root / "images" / split / name)
                (out_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
                exported_count += 1
                continue

            if not src_txt.exists():
                continue

            content = src_txt.read_text(encoding="utf-8").strip()
            if not content:
                # Valid empty background label
                shutil.copy2(src_img, out_root / "images" / split / name)
                (out_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
                exported_count += 1
                continue

            valid_lines = []
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    validation_errors.append(f"{name}: invalid token count {len(parts)}")
                    continue
                coords = [float(v) for v in parts[1:]]
                if any(c < 0.0 or c > 1.0 for c in coords):
                    validation_errors.append(f"{name}: coordinate out of [0.0, 1.0] range")
                    continue

                pts = np.array(coords).reshape((4, 2))
                area = signed_shoelace_area(pts)
                if area <= 0:
                    pts = order_corners_canonical(pts)
                    coords = [float(c) for pt in pts for c in pt]
                    line = f"{parts[0]} " + " ".join(f"{c:.6f}" for c in coords)
                valid_lines.append(line)

            if valid_lines:
                shutil.copy2(src_img, out_root / "images" / split / name)
                (out_root / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(valid_lines), encoding="utf-8"
                )
                exported_count += 1

        classes = meta.get("classes", ["utility_pole"])
        yaml_content = (
            f"# YOLO-OBB Dataset Export: {dataset_id}\n"
            f"path: {out_root.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n"
        )
        for idx, cls_name in enumerate(classes):
            yaml_content += f"  {idx}: {cls_name}\n"

        (out_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

        return {
            "export_path": str(out_root),
            "exported_images": exported_count,
            "train_count": len([x for x in verified_images if x in train_set]),
            "val_count": len([x for x in verified_images if x not in train_set]),
            "validation_errors": validation_errors,
        }

    def export_test_set(
        self,
        dataset_id: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Export dedicated FIXED TEST set partition."""
        meta = self.get_dataset(dataset_id)
        if not meta:
            raise ValueError(f"Dataset {dataset_id} does not exist.")

        out_root = Path(output_dir or (self.base_dir / dataset_id / "export" / "test_set")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)

        test_images = [
            k for k, v in meta.get("images", {}).items()
            if v.get("is_test_set", False)
        ]

        ds_path = self.base_dir / dataset_id
        exported = 0
        for name in test_images:
            stem = Path(name).stem
            src_img = ds_path / "images" / name
            src_txt = ds_path / "annotations" / f"{stem}.txt"
            if src_img.exists():
                shutil.copy2(src_img, out_root / "images" / name)
                if src_txt.exists():
                    shutil.copy2(src_txt, out_root / "labels" / f"{stem}.txt")
                else:
                    (out_root / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
                exported += 1

        manifest = {
            "dataset_id": dataset_id,
            "count": exported,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "images": test_images,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"export_path": str(out_root), "count": exported}

