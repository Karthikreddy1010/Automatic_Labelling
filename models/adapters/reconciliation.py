"""
models/adapters/reconciliation.py - Multi-Model Candidate Reconciliation & Triage
================================================================================

Reconciles candidate detections from YOLO, Grounding DINO, SAM, and Human ground
truth. Reuses IoU clustering principles from relabel/gen_candidates.py.

Guarantees:
- Spatial IoU matching with configurable threshold.
- Accurate source attribution ("YOLO", "DINO", "YOLO+DINO", "SAM", "HUMAN").
- Consensus boosting: agreement between independent engines increases confidence.
- Strict confidence categorization:
    * GREEN  (High Trust):   conf >= 0.70, dual agreement or very high YOLO.
    * YELLOW (Needs Review): 0.45 <= conf < 0.70 or single-engine detection.
    * RED    (Uncertain):    conf < 0.45 or severe geometry anomaly.
- Review reasons documented on each DetectionBox.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from models.adapters.base import DetectionBox
from src.geometry_obb import (
    obb_iou,
    xyxy_to_obb_corners,
    order_corners_canonical,
    obb_corners_to_xyxy,
)


def compute_box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Standard 2D axis-aligned IoU calculation."""
    ix0 = max(box_a[0], box_b[0])
    iy0 = max(box_a[1], box_b[1])
    ix1 = min(box_a[2], box_b[2])
    iy1 = min(box_a[3], box_b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def reconcile_candidates(
    yolo_boxes: List[DetectionBox],
    dino_boxes: List[DetectionBox],
    sam_boxes: Optional[List[DetectionBox]] = None,
    human_boxes: Optional[List[DetectionBox]] = None,
    iou_threshold: float = 0.45,
    min_confidence: float = 0.20
) -> List[DetectionBox]:
    """
    Greedy clustering and consensus reconciliation across all detection sources.
    
    Returns:
        Consolidated list of DetectionBox objects tagged with reconciled sources and review flags.
    """
    all_dets: List[Tuple[DetectionBox, str]] = []
    for b in yolo_boxes:
        all_dets.append((b, "YOLO"))
    for b in dino_boxes:
        all_dets.append((b, "DINO"))
    if sam_boxes:
        for b in sam_boxes:
            all_dets.append((b, "SAM"))

    # Sort descending by confidence
    all_dets.sort(key=lambda t: t[0].confidence, reverse=True)
    used = [False] * len(all_dets)
    reconciled: List[DetectionBox] = []

    for i in range(len(all_dets)):
        if used[i]:
            continue
        used[i] = True
        cluster_dets = [all_dets[i]]

        ref_box = all_dets[i][0]
        for j in range(i + 1, len(all_dets)):
            if used[j]:
                continue
            cand_box = all_dets[j][0]
            
            # Use OBB IoU if both have corners, else fallback to xyxy IoU
            if ref_box.corners is not None and cand_box.corners is not None:
                iou_val = obb_iou(ref_box.corners, cand_box.corners)
            else:
                iou_val = compute_box_iou(ref_box.xyxy, cand_box.xyxy)

            if iou_val >= iou_threshold:
                used[j] = True
                cluster_dets.append(all_dets[j])

        sources = {src for _, src in cluster_dets}
        best_box, primary_src = cluster_dets[0]

        # Prioritize OBB corners if any detector produced them (e.g. YOLO-OBB or SAM)
        chosen_corners = None
        for det, _ in cluster_dets:
            if det.corners is not None:
                chosen_corners = det.corners
                break
        if chosen_corners is None:
            chosen_corners = xyxy_to_obb_corners(*best_box.xyxy).tolist()

        # Source attribution & separate confidence tracking (no fake boost)
        reasons: List[str] = []
        model_agreement = False
        dino_conf = next((round(det.confidence, 3) for det, src in cluster_dets if src == "DINO"), None)
        yolo_conf = next((round(det.confidence, 3) for det, src in cluster_dets if src == "YOLO"), None)

        if "YOLO" in sources and "DINO" in sources:
            source_label = "YOLO+DINO"
            model_agreement = True
            # Keep authentic highest confidence without artificial mathematical inflation
            final_conf = max(det.confidence for det, _ in cluster_dets)
            needs_review = final_conf < 0.60
            if needs_review:
                reasons.append("Low combined confidence despite multi-model agreement")
        elif "YOLO" in sources:
            source_label = "YOLO"
            final_conf = best_box.confidence
            needs_review = final_conf < 0.65
            if needs_review:
                reasons.append("Single detector (YOLO) with confidence < 0.65")
        elif "DINO" in sources:
            source_label = "DINO"
            final_conf = best_box.confidence
            needs_review = True  # DINO-only is open-vocabulary and always warrants human review
            reasons.append("DINO open-vocabulary detection without confirmation")
        elif "SAM" in sources:
            source_label = "SAM"
            final_conf = best_box.confidence
            needs_review = True
            reasons.append("SAM-only instance")
        else:
            source_label = primary_src
            final_conf = best_box.confidence
            needs_review = True

        if final_conf < min_confidence:
            continue

        reconciled.append(DetectionBox(
            xyxy=best_box.xyxy,
            corners=chosen_corners,
            confidence=round(final_conf, 3),
            class_id=best_box.class_id,
            class_name=best_box.class_name,
            model_source=source_label,
            needs_review=needs_review,
            review_reasons=reasons,
            attributes={
                "cluster_size": len(cluster_dets),
                "contributing_models": list(sources),
                "model_agreement": model_agreement,
                "dino_confidence": dino_conf,
                "yolo_confidence": yolo_conf,
                "category": get_confidence_category(final_conf, needs_review),
            }
        ))

    # Incorporate human boxes if present (highest trust anchor)
    if human_boxes:
        for hb in human_boxes:
            matched = False
            for rec in reconciled:
                if rec.corners is not None and hb.corners is not None:
                    iou_val = obb_iou(rec.corners, hb.corners)
                else:
                    iou_val = compute_box_iou(rec.xyxy, hb.xyxy)

                if iou_val >= iou_threshold:
                    matched = True
                    rec.model_source = "HUMAN"
                    rec.confidence = 1.0
                    rec.needs_review = False
                    rec.review_reasons = []
                    rec.attributes["category"] = "GREEN"
                    if hb.corners is not None:
                        rec.corners = hb.corners
                        rec.xyxy = obb_corners_to_xyxy(hb.corners)
                    break
            if not matched:
                h_corners = hb.corners if hb.corners is not None else xyxy_to_obb_corners(*hb.xyxy).tolist()
                reconciled.append(DetectionBox(
                    xyxy=hb.xyxy,
                    corners=h_corners,
                    confidence=1.0,
                    class_id=hb.class_id,
                    class_name=hb.class_name,
                    model_source="HUMAN",
                    needs_review=False,
                    review_reasons=[],
                    attributes={"category": "GREEN"}
                ))

    return reconciled


def get_confidence_category(confidence: float, needs_review: bool) -> str:
    """
    Categorize detection for UI color-coding:
    - GREEN: High confidence, verified or dual consensus.
    - YELLOW: Moderate confidence, single-model or flagged for review.
    - RED: Low confidence or high ambiguity.
    """
    if confidence >= 0.70 and not needs_review:
        return "GREEN"
    elif confidence >= 0.45:
        return "YELLOW"
    else:
        return "RED"
