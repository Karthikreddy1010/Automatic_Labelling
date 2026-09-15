"""
models/adapters/recovery_dedup.py - OBB Recovery Duplicate Suppression
==========================================================================

The OBB recovery stage (models/adapters/obb_recovery.py) must never create a
second label for a pole DINO/SAM3 already found. This module compares each
recovery candidate against the primary pipeline's existing DetectionBoxes
using rotated (Shapely-polygon) IoU -- src/geometry_obb.py::obb_iou -- never
axis-aligned IoU, since two OBBs at different orientations can have heavily
overlapping axis-aligned bounding boxes while barely overlapping as actual
rotated shapes (or vice versa).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from models.adapters.base import DetectionBox
from src.geometry_obb import obb_iou, xyxy_to_obb_corners


def _corners_of(det: DetectionBox) -> np.ndarray:
    if det.corners is not None:
        return np.asarray(det.corners, dtype=np.float64)
    return xyxy_to_obb_corners(*det.xyxy)


def dedup_recovery_against_existing(
    recovery_candidates: List[DetectionBox],
    existing_boxes: List[DetectionBox],
    iou_threshold: float,
) -> Tuple[List[DetectionBox], List[Dict[str, Any]]]:
    """
    Returns (kept, discarded). A recovery candidate is discarded when its
    rotated IoU against ANY existing (DINO/SAM3-origin) box meets or exceeds
    iou_threshold -- the existing detection already covers that physical
    pole, so the recovery candidate would be a duplicate label.

    `discarded` entries are {"candidate": DetectionBox, "reason": str,
    "matched_existing_index": int, "iou": float} -- kept structured (not
    just dropped) so the caller can log/store them for active-learning
    review rather than silently losing the information about why a
    candidate was suppressed.
    """
    kept: List[DetectionBox] = []
    discarded: List[Dict[str, Any]] = []

    existing_corners = [_corners_of(b) for b in existing_boxes]

    for candidate in recovery_candidates:
        cand_corners = _corners_of(candidate)
        best_iou = 0.0
        best_idx = -1
        for idx, other_corners in enumerate(existing_corners):
            try:
                iou = obb_iou(cand_corners, other_corners)
            except Exception:
                continue
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            discarded.append({
                "candidate": candidate,
                "reason": "duplicate_of_existing",
                "matched_existing_index": best_idx,
                "iou": round(best_iou, 4),
            })
        else:
            kept.append(candidate)

    return kept, discarded
