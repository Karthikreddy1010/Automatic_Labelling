"""
models/adapters/decision_engine.py - Multi-Signal ACCEPT/REVIEW/REJECT Decision
==================================================================================

Combines four INDEPENDENT evidence signals into one final_score and a
three-way decision, per the spec:
    detection_score    -- Grounding DINO / SAM3 candidate confidence
    segmentation_score -- SAM mask quality (geometry+overlap, NOT DINO conf)
    geometry_score      -- models/adapters/geometry_qa.py output
    semantic_score      -- Qwen semantic_confidence (models/adapters/qwen_adapter.py)

Deliberately does NOT just multiply the four scores together (multiplying
means any single low/absent score can zero out an otherwise-solid candidate,
and there's no way to reason about *why* -- see spec section 8). Uses a
configurable weighted sum instead, with weights normalized over whichever
signals actually ran (Qwen is often gated off for cheap candidates -- an
un-run semantic check must not silently count as a 0).

segmentation_score is deliberately NOT models.adapters.quality.score_pole_quality()
-- that function already blends in DINO confidence (0.35 weight), which would
double-count detection_score here. Instead this reuses the same underlying
mask-geometry (_pole_score) and box-overlap (_mask_box_overlap_fraction)
sub-signals from quality.py, kept independent of DINO confidence.

Guarantees (spec section 19):
- Never lets Qwen (or any single signal) unilaterally decide; REJECT-by-
  semantics requires a confident, non-uncertain Qwen call AND is still just
  one input to the weighted score plus an explicit rule below.
- severely_occluded / too_distant candidates are capped at REVIEW, never
  auto-ACCEPTed, regardless of final_score (spec section 3).
- geometry_status == "fail" or an invalid OBB forces REJECT.
- Disagreement between signals is recorded explicitly, never silently averaged away.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.adapters.quality import _pole_score, _mask_box_overlap_fraction, _POLE_SCORE_MAX

NON_POLE_CLASSES = {
    "non_utility_pole", "tree", "building_or_structure",
    "street_light_or_lamp_post", "other_object",
}


@dataclass
class DecisionWeights:
    detection: float = 0.20
    semantic: float = 0.35
    geometry: float = 0.25
    segmentation: float = 0.20

    def as_dict(self) -> Dict[str, float]:
        return {"detection": self.detection, "semantic": self.semantic,
                "geometry": self.geometry, "segmentation": self.segmentation}


@dataclass
class DecisionConfig:
    weights: DecisionWeights = field(default_factory=DecisionWeights)
    accept_threshold: float = 0.65
    review_threshold: float = 0.35
    # Qwen must be at least this confident (and NOT "uncertain") to drive a
    # REJECT purely on semantics -- a low-confidence "probably a tree" lands
    # in REVIEW instead, never an automatic drop.
    semantic_reject_confidence: float = 0.60


DEFAULT_DECISION_CONFIG = DecisionConfig()


def config_from_dict(d: Optional[Dict[str, Any]]) -> DecisionConfig:
    """Build a DecisionConfig from a plain dict (e.g. parsed config.yaml section),
    tolerating missing/partial/unknown keys -- never crashes on absent config."""
    if not d:
        return DecisionConfig()
    weights_d = d.get("weights") or {}
    weights = DecisionWeights(
        detection=float(weights_d.get("detection", DecisionWeights.detection)),
        semantic=float(weights_d.get("semantic", DecisionWeights.semantic)),
        geometry=float(weights_d.get("geometry", DecisionWeights.geometry)),
        segmentation=float(weights_d.get("segmentation", DecisionWeights.segmentation)),
    )
    return DecisionConfig(
        weights=weights,
        accept_threshold=float(d.get("accept_threshold", DecisionConfig.accept_threshold)),
        review_threshold=float(d.get("review_threshold", DecisionConfig.review_threshold)),
        semantic_reject_confidence=float(d.get("semantic_reject_confidence", DecisionConfig.semantic_reject_confidence)),
    )


def compute_segmentation_score(mask: Optional[np.ndarray], box_xyxy: Tuple[float, float, float, float]) -> float:
    """
    Pure SAM-mask-quality signal, independent of DINO confidence: does the
    mask look pole-shaped (tall/thin/centered/vertically-covering), and did
    SAM actually segment the candidate region (not a neighboring object)?
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    geom_raw = _pole_score(mask, box_xyxy)
    geom_norm = max(0.0, min(1.0, geom_raw / _POLE_SCORE_MAX))
    overlap = _mask_box_overlap_fraction(mask, box_xyxy)
    return round(0.5 * geom_norm + 0.5 * overlap, 4)


def detect_disagreement(
    detection_score: float,
    segmentation_score: float,
    geometry_result: Optional[Dict[str, Any]],
    semantic_result: Optional[Dict[str, Any]],
) -> List[str]:
    """Explicitly named disagreement cases between independent signals (spec section 10)."""
    flags: List[str] = []
    qwen_class = semantic_result.get("class") if semantic_result else None
    qwen_conf = semantic_result.get("semantic_confidence") if semantic_result else None

    if semantic_result and qwen_class in NON_POLE_CLASSES and qwen_conf is not None:
        if detection_score >= 0.6 and qwen_conf >= 0.5:
            flags.append(f"DINO confident ({detection_score:.2f}) but Qwen says '{qwen_class}' ({qwen_conf:.2f})")

    if semantic_result and qwen_class == "electric_utility_pole" and segmentation_score < 0.30:
        flags.append(f"Qwen says pole but SAM segmentation_score is low ({segmentation_score:.2f})")

    if (
        semantic_result and geometry_result and qwen_class == "electric_utility_pole"
        and segmentation_score >= 0.5
        and geometry_result.get("geometry_status") in ("warning", "fail")
    ):
        flags.append(f"Qwen+SAM agree it's a pole but geometry_status={geometry_result.get('geometry_status')}")

    return flags


def evaluate_candidate(
    detection_score: float,
    segmentation_score: float,
    geometry_result: Optional[Dict[str, Any]],
    semantic_result: Optional[Dict[str, Any]],
    obb_valid: bool = True,
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> Dict[str, Any]:
    """
    Combine the (up to) four independent signals into one final_score and
    ACCEPT/REVIEW/REJECT decision. semantic_result is None when Qwen was
    gated off/unavailable for this candidate; geometry_result is None when
    geometry QA was disabled for this experiment run (spec section 13's
    config A/B comparisons) -- in both cases weights are renormalized over
    the signals that actually ran rather than treating a skipped check as a
    zero score.
    """
    geometry_ran = geometry_result is not None
    geometry_score = float(geometry_result.get("geometry_score", 0.0)) if geometry_ran else None
    geometry_status = geometry_result.get("geometry_status") if geometry_ran else "not_run"

    semantic_ran = bool(
        semantic_result
        and semantic_result.get("status") not in ("unavailable", "error")
        and semantic_result.get("semantic_confidence") is not None
    )
    semantic_score = float(semantic_result["semantic_confidence"]) if semantic_ran else None

    active_scores: Dict[str, float] = {
        "detection": detection_score,
        "segmentation": segmentation_score,
    }
    active_weights: Dict[str, float] = {
        "detection": config.weights.detection,
        "segmentation": config.weights.segmentation,
    }
    if geometry_ran:
        active_scores["geometry"] = geometry_score
        active_weights["geometry"] = config.weights.geometry
    if semantic_ran:
        active_scores["semantic"] = semantic_score
        active_weights["semantic"] = config.weights.semantic

    weight_sum = sum(active_weights.values())
    final_score = (
        sum(active_scores[k] * active_weights[k] for k in active_scores) / weight_sum
        if weight_sum > 0 else 0.0
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    disagreements = detect_disagreement(detection_score, segmentation_score, geometry_result, semantic_result)

    reasons: List[str] = []
    if geometry_ran:
        reasons.extend(f"geometry: {f}" for f in geometry_result.get("geometry_flags", []))
    reasons.extend(disagreements)

    qwen_class = semantic_result.get("class") if semantic_result else None
    qwen_conf = semantic_result.get("semantic_confidence") if semantic_result else None
    qwen_visibility = semantic_result.get("visibility") if semantic_result else None

    # --- Hard REJECT conditions -------------------------------------------------
    if not obb_valid:
        decision = "REJECT"
        reasons.append("OBB failed structural validation")
    elif geometry_status == "fail":
        decision = "REJECT"
        reasons.append("geometry QA hard failure")
    elif (
        semantic_ran and qwen_class in NON_POLE_CLASSES
        and qwen_conf is not None and qwen_conf >= config.semantic_reject_confidence
    ):
        decision = "REJECT"
        reasons.append(f"Qwen confidently identified non-pole class '{qwen_class}' ({qwen_conf:.2f})")
    else:
        # --- Uncertain visibility caps at REVIEW, never auto-ACCEPT (section 3) ---
        capped_review = qwen_visibility in ("severely_occluded", "too_distant")
        geometry_ok = geometry_status in ("pass", "not_run")

        if disagreements:
            decision = "REVIEW"
        elif final_score >= config.accept_threshold and geometry_ok and not capped_review:
            decision = "ACCEPT"
        elif final_score < config.review_threshold:
            decision = "REJECT"
            reasons.append(f"final_score {final_score:.2f} below review threshold {config.review_threshold:.2f}")
        else:
            decision = "REVIEW"
            if capped_review:
                reasons.append(f"Qwen visibility='{qwen_visibility}' -- capped at REVIEW, not auto-accepted")

    return {
        "detection_score": round(detection_score, 4),
        "segmentation_score": round(segmentation_score, 4),
        "geometry_score": geometry_score,
        "geometry_status": geometry_status,
        "semantic_score": round(semantic_score, 4) if semantic_score is not None else None,
        "semantic_ran": semantic_ran,
        "final_score": final_score,
        "weights_used": active_weights,
        "decision": decision,
        "disagreements": disagreements,
        "reasons": reasons,
        "qwen_class": qwen_class,
        "qwen_material": semantic_result.get("material") if semantic_result else None,
        "qwen_visibility": qwen_visibility,
        "qwen_orientation": semantic_result.get("orientation") if semantic_result else None,
        "qwen_reason": semantic_result.get("reason") if semantic_result else None,
    }
