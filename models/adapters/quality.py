"""
models/adapters/quality.py - Pole Candidate Deduplication & Quality Scoring
============================================================================

Two responsibilities, kept separate from detection (DINO) and segmentation
(SAM) per the pipeline's separation-of-concerns:

1. dedup_by_iou(): merge candidate boxes from the same detector pass that
   overlap enough to be the same physical pole, keeping the highest-confidence
   one. Genuinely separate poles (IoU ~0 between them) are never merged.
2. score_pole_quality(): a transparent, multi-signal quality_score (NOT a
   calibrated probability) for each SAM-refined candidate, bucketed into
   HIGH_QUALITY / REVIEW / REJECT for the human reviewer.

Reuses pole_sam._pole_score (verticality/aspect, fill fraction, centerline
alignment, vertical coverage) and reconciliation.compute_box_iou rather than
reimplementing candidate geometry scoring or IoU math a third time.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from models.adapters.base import DetectionBox
from models.adapters.reconciliation import compute_box_iou

try:
    from pole_sam import _pole_score
except ImportError:
    def _pole_score(mask, box):
        return 0.0


# --- Deduplication ---------------------------------------------------------

DEDUP_IOU_THRESHOLD = 0.5  # candidates overlapping >= this are the same pole


def dedup_by_iou(
    candidates: List[DetectionBox],
    iou_threshold: float = DEDUP_IOU_THRESHOLD,
) -> List[DetectionBox]:
    """
    Greedy IoU clustering over a single detector's raw candidates: boxes
    overlapping another box by >= iou_threshold are collapsed to the
    highest-confidence one in that cluster. Two boxes with low/zero IoU are
    always kept independently, so two genuinely different poles standing
    near each other are never suppressed just for being nearby.
    """
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].confidence, reverse=True)
    used = [False] * len(candidates)
    kept: List[DetectionBox] = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        cluster_size = 1
        for j in order:
            if used[j]:
                continue
            if compute_box_iou(candidates[i].xyxy, candidates[j].xyxy) >= iou_threshold:
                used[j] = True
                cluster_size += 1
        best = candidates[i]
        if cluster_size > 1:
            best.attributes["dedup_cluster_size"] = cluster_size
        kept.append(best)
    return kept


# --- Quality scoring ---------------------------------------------------------

# Heuristic thresholds -- NOT calibrated probabilities. Configurable; revisit
# once the 30-image benchmark gives real precision/recall data to tune against.
HIGH_QUALITY_THRESHOLD = 0.62
REJECT_THRESHOLD = 0.30

# _pole_score()'s theoretical max: aspect/12<=1.0 + 0.5*fill_pen<=0.5
# + 0.7*center_bonus<=0.7 + 0.6*v_cover<=0.6 = 2.8. Used to normalize into [0,1].
_POLE_SCORE_MAX = 2.8


def _mask_touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    m = margin if margin < h and margin < w else max(1, min(h, w) - 1)
    return bool(
        mask[:m, :].any() or mask[-m:, :].any()
        or mask[:, :m].any() or mask[:, -m:].any()
    )


def _mask_box_overlap_fraction(mask: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> float:
    """
    Fraction of the mask's foreground pixels that fall within the original
    DINO box (padded 15%, matching sam21_adapter's own box-gate). Low overlap
    means SAM's mask wandered onto a neighboring object instead of the
    candidate pole.
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    pw, ph = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    gx0, gy0 = max(0, int(x0 - pw)), max(0, int(y0 - ph))
    gx1, gy1 = min(w, int(x1 + pw)), min(h, int(y1 + ph))
    inside = mask[gy0:gy1, gx0:gx1].sum()
    return float(inside) / float(mask.sum())


def score_pole_quality(
    mask: Optional[np.ndarray],
    box_xyxy: Tuple[float, float, float, float],
    dino_confidence: float,
) -> Dict[str, Any]:
    """
    Blend DINO confidence with SAM-mask geometry into one transparent
    quality_score in [0, 1], bucketed into HIGH_QUALITY / REVIEW / REJECT.

    This is a heuristic signal for prioritizing human review and filtering
    obvious false positives (wires, tree branches, buildings) -- not a
    calibrated probability of correctness. A tilted, partially-occluded, or
    frame-cut-off pole is a normal case and should land in REVIEW, not be
    hard-rejected, so no single signal here can reject on its own; only the
    blended score does.
    """
    if mask is None or mask.sum() == 0:
        return {
            "quality_score": 0.0,
            "category": "REJECT",
            "reasons": ["SAM produced no usable mask for this candidate"],
        }

    geom_raw = _pole_score(mask, box_xyxy)
    geom_norm = max(0.0, min(1.0, geom_raw / _POLE_SCORE_MAX))

    overlap = _mask_box_overlap_fraction(mask, box_xyxy)
    reasons: List[str] = []
    if overlap < 0.5:
        reasons.append(f"mask mostly outside DINO box (overlap={overlap:.2f})")

    touches_border = _mask_touches_border(mask)
    if touches_border:
        reasons.append("mask touches image border (pole may be cut off/occluded)")

    # Weighted blend: DINO confidence (was this really a pole?), mask
    # geometry (does it look like a pole shaft -- tall, thin, centered,
    # vertically covering the box?), and overlap (did SAM segment the
    # candidate itself, not a neighboring object?).
    quality_score = (
        0.35 * float(dino_confidence)
        + 0.40 * geom_norm
        + 0.25 * overlap
    )
    if touches_border:
        # A real, common case (spec: poles may be "partially outside the
        # frame") -- cap below HIGH_QUALITY so a human double-checks the
        # cut-off edge, but don't reject outright.
        quality_score = min(quality_score, HIGH_QUALITY_THRESHOLD - 0.01)

    quality_score = round(max(0.0, min(1.0, quality_score)), 4)

    if quality_score >= HIGH_QUALITY_THRESHOLD:
        category = "HIGH_QUALITY"
    elif quality_score >= REJECT_THRESHOLD:
        category = "REVIEW"
    else:
        category = "REJECT"
        if not reasons:
            reasons.append("low combined DINO-confidence/mask-geometry quality score")

    return {"quality_score": quality_score, "category": category, "reasons": reasons}
