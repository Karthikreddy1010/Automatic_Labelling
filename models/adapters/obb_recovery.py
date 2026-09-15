"""
models/adapters/obb_recovery.py - OBB Pole Recovery (fallback detection stage)
=================================================================================

A geometry-driven fallback that runs ONLY when the primary DINO + SAM3
pipeline (see backend/app.py::run_ai_pipeline, models/adapters/quality.py,
models/adapters/decision_engine.py) missed a pole entirely, returned only a
weak/low-confidence candidate, or (via reconstruct_partial_pole) segmented
only part of one. It never replaces DINO/SAM3 and never runs when they
already produced a confident result -- see should_run_recovery().

Two independent capabilities, both geometry-first (never "every vertical
thing is a pole"):

1. generate_recovery_candidates(): finds candidate pole-shaped OBBs directly
   from image edges/structure when DINO+SAM3 returned nothing usable, using
   a real multi-signal pole_score (verticality, elongation, line continuity,
   edge consistency, width consistency, plus DINO/SAM/position agreement
   when available) -- never a bare "is it vertical" check.
2. reconstruct_partial_pole(): given a SAM mask that covers only PART of a
   pole, extends the OBB along the observed centerline/orientation/width by
   following continuing edge evidence in the image, bounded by
   RecoveryConfig.max_extension_factor -- never blindly extending to the
   image border.

Every candidate this module produces is validated via
models/adapters/obb_generator.py::validate_obb before being returned, so a
malformed OBB can never reach the label writer. Every candidate is also
explicitly marked needs_review unless its pole_score clears
RecoveryConfig.review_threshold (recovery never silently auto-accepts on its
own -- decision_engine.py's existing ACCEPT/REVIEW/REJECT call, and human
review, are still the only paths to a final accepted label).

Duplicate suppression against existing DINO/SAM3 candidates lives in the
sibling module models/adapters/recovery_dedup.py (rotated/Shapely IoU, never
axis-aligned) and is applied by the caller (backend/app.py) after this
module returns its candidates -- kept separate so recovery-candidate
*generation* and *deduplication against other sources* stay independently
testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from models.adapters.base import DetectionBox
from models.adapters.obb_generator import validate_obb
from src.geometry_obb import order_corners_canonical


# --- Configuration -----------------------------------------------------------

@dataclass
class RecoveryWeights:
    verticality: float = 0.15
    elongation: float = 0.15
    line_continuity: float = 0.15
    edge_consistency: float = 0.10
    width_consistency: float = 0.10
    segmentation_agreement: float = 0.15
    dino_agreement: float = 0.10
    position_context: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "verticality": self.verticality, "elongation": self.elongation,
            "line_continuity": self.line_continuity, "edge_consistency": self.edge_consistency,
            "width_consistency": self.width_consistency, "segmentation_agreement": self.segmentation_agreement,
            "dino_agreement": self.dino_agreement, "position_context": self.position_context,
        }


@dataclass
class RecoveryConfig:
    enabled: bool = True
    recovery_activation_threshold: float = 0.55
    min_pole_score: float = 0.25
    review_threshold: float = 0.55
    max_candidates_per_image: int = 6
    min_aspect_ratio: float = 2.5
    max_aspect_ratio: float = 40.0
    max_angle_from_vertical_deg: float = 25.0
    min_width_px: float = 4.0
    max_width_px: float = 120.0
    duplicate_iou_threshold: float = 0.35
    max_extension_factor: float = 2.0
    weights: RecoveryWeights = field(default_factory=RecoveryWeights)


def config_from_dict(d: Optional[Dict[str, Any]]) -> RecoveryConfig:
    """Build a RecoveryConfig from a plain dict (e.g. parsed config.yaml's
    verification_pipeline.obb_recovery section), tolerating missing/partial/
    unknown keys -- never crashes on absent config, mirrors
    decision_engine.py::config_from_dict's pattern."""
    if not d:
        return RecoveryConfig()
    weights_d = d.get("weights") or {}
    defaults = RecoveryWeights()
    weights = RecoveryWeights(
        verticality=float(weights_d.get("verticality", defaults.verticality)),
        elongation=float(weights_d.get("elongation", defaults.elongation)),
        line_continuity=float(weights_d.get("line_continuity", defaults.line_continuity)),
        edge_consistency=float(weights_d.get("edge_consistency", defaults.edge_consistency)),
        width_consistency=float(weights_d.get("width_consistency", defaults.width_consistency)),
        segmentation_agreement=float(weights_d.get("segmentation_agreement", defaults.segmentation_agreement)),
        dino_agreement=float(weights_d.get("dino_agreement", defaults.dino_agreement)),
        position_context=float(weights_d.get("position_context", defaults.position_context)),
    )
    dflt = RecoveryConfig()
    return RecoveryConfig(
        enabled=bool(d.get("enabled", dflt.enabled)),
        recovery_activation_threshold=float(d.get("recovery_activation_threshold", dflt.recovery_activation_threshold)),
        min_pole_score=float(d.get("min_pole_score", dflt.min_pole_score)),
        review_threshold=float(d.get("review_threshold", dflt.review_threshold)),
        max_candidates_per_image=int(d.get("max_candidates_per_image", dflt.max_candidates_per_image)),
        min_aspect_ratio=float(d.get("min_aspect_ratio", dflt.min_aspect_ratio)),
        max_aspect_ratio=float(d.get("max_aspect_ratio", dflt.max_aspect_ratio)),
        max_angle_from_vertical_deg=float(d.get("max_angle_from_vertical_deg", dflt.max_angle_from_vertical_deg)),
        min_width_px=float(d.get("min_width_px", dflt.min_width_px)),
        max_width_px=float(d.get("max_width_px", dflt.max_width_px)),
        duplicate_iou_threshold=float(d.get("duplicate_iou_threshold", dflt.duplicate_iou_threshold)),
        max_extension_factor=float(d.get("max_extension_factor", dflt.max_extension_factor)),
        weights=weights,
    )


# --- Activation gating ---------------------------------------------------------

def should_run_recovery(existing_boxes: List[DetectionBox], config: RecoveryConfig) -> bool:
    """
    Recovery only runs when the primary DINO+SAM3 pipeline missed the pole
    entirely (no surviving candidates) or only produced weak ones (best
    confidence below recovery_activation_threshold) -- never as a blanket
    second-guess of a confident result (spec: 'If DINO/SAM confidently
    detects a pole -> use the normal result').
    """
    if not config.enabled:
        return False
    if not existing_boxes:
        return True
    best_confidence = max(b.confidence for b in existing_boxes)
    return best_confidence < config.recovery_activation_threshold


# --- Geometry helpers ----------------------------------------------------------

def _long_axis(corners: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Given 4 OBB corners (in perimeter order), return (unit_direction,
    half_length, half_width, center) of the box's long axis. unit_direction
    always points in the direction of the longer pair of opposite edges;
    robust to any of the four possible starting-corner rotations.
    """
    c = np.asarray(corners, dtype=np.float64)
    edges = [c[(i + 1) % 4] - c[i] for i in range(4)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    # Opposite edge pairs: (0,2) and (1,3).
    pair_a_len = (lens[0] + lens[2]) / 2.0
    pair_b_len = (lens[1] + lens[3]) / 2.0
    if pair_b_len >= pair_a_len:
        long_len, short_len = pair_b_len, pair_a_len
        direction = edges[1]
    else:
        long_len, short_len = pair_a_len, pair_b_len
        direction = edges[0]
    norm = np.linalg.norm(direction)
    unit = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
    center = c.mean(axis=0)
    return unit, long_len / 2.0, short_len / 2.0, center


def score_verticality(corners: np.ndarray) -> float:
    """1.0 for a perfectly vertical long axis, 0.0 for perfectly horizontal --
    cosine of the angle between the long axis and true vertical (0, 1)."""
    unit, _, _, _ = _long_axis(corners)
    angle_deg = float(np.degrees(np.arctan2(abs(unit[0]), abs(unit[1]) + 1e-12)))
    return float(max(0.0, np.cos(np.radians(angle_deg))))


def score_elongation(corners: np.ndarray, config: RecoveryConfig) -> float:
    """
    Aspect ratio (long/short) normalized against config.min_aspect_ratio: a
    ratio of 1.0 (square) scores 0, a ratio of 2*min_aspect_ratio or more
    scores 1.0, linear in between -- so the score's shape follows the same
    configurable threshold used to hard-filter candidates elsewhere, rather
    than an unrelated magic constant.
    """
    _, half_len, half_width, _ = _long_axis(corners)
    if half_width <= 1e-9:
        return 1.0 if half_len > 1e-9 else 0.0
    ratio = half_len / half_width
    target = max(config.min_aspect_ratio * 2.0, 1.0 + 1e-6)
    return float(max(0.0, min(1.0, (ratio - 1.0) / (target - 1.0))))


def score_width_consistency(corners: np.ndarray, width_samples: Optional[List[float]]) -> float:
    """
    1.0 minus the coefficient of variation of width measured at several
    points along the shaft (a pole has near-constant width along its
    height; a false positive formed from unrelated edge fragments usually
    doesn't). When no per-position samples are available (width_samples is
    None -- e.g. scoring a fitted OBB directly with no raw measurement
    trail), the fitted rectangle is width-consistent by construction, so
    this returns 1.0 rather than penalizing for missing data the caller
    never had.
    """
    if not width_samples or len(width_samples) < 2:
        return 1.0
    arr = np.asarray(width_samples, dtype=np.float64)
    mean = arr.mean()
    if mean <= 1e-9:
        return 0.0
    cv = arr.std() / mean
    return float(max(0.0, min(1.0, 1.0 - cv)))


def score_line_continuity(fill_fraction: Optional[float]) -> Optional[float]:
    """Fraction of the candidate's own bounding rectangle actually covered by
    edge/structure evidence -- a solid, continuous shaft scores near 1.0; a
    fragmented/dotted line of edge pixels scores low. None (no evidence) when
    the caller has no fill-fraction data to offer, so it's excluded from the
    blend rather than counted as a hard 0."""
    if fill_fraction is None:
        return None
    return float(max(0.0, min(1.0, fill_fraction)))


def score_edge_consistency(edge_support_fraction: Optional[float]) -> Optional[float]:
    """Fraction of the candidate OBB's two long edges that have real Canny
    edge pixels running along them -- distinguishes an edge-supported
    rectangle from one that only happens to sit over a vaguely elongated
    region. None when unavailable."""
    if edge_support_fraction is None:
        return None
    return float(max(0.0, min(1.0, edge_support_fraction)))


def compute_pole_score(
    corners: np.ndarray,
    config: RecoveryConfig,
    dino_confidence: Optional[float] = None,
    sam_mask_agreement: Optional[float] = None,
    position_context: Optional[float] = None,
    fill_fraction: Optional[float] = None,
    edge_support_fraction: Optional[float] = None,
    width_samples: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Blend the available pole-likelihood signals into one pole_score in
    [0, 1], renormalized over whichever signals actually have evidence for
    this candidate -- mirrors decision_engine.py::evaluate_candidate's
    renormalization so an unavailable signal (e.g. no DINO box anywhere near
    this candidate) is never silently treated as a 0.
    """
    w = config.weights
    components: Dict[str, float] = {
        "verticality": score_verticality(corners),
        "elongation": score_elongation(corners, config),
        "width_consistency": score_width_consistency(corners, width_samples),
    }
    active_weights: Dict[str, float] = {
        "verticality": w.verticality, "elongation": w.elongation, "width_consistency": w.width_consistency,
    }

    line_continuity = score_line_continuity(fill_fraction)
    if line_continuity is not None:
        components["line_continuity"] = line_continuity
        active_weights["line_continuity"] = w.line_continuity

    edge_consistency = score_edge_consistency(edge_support_fraction)
    if edge_consistency is not None:
        components["edge_consistency"] = edge_consistency
        active_weights["edge_consistency"] = w.edge_consistency

    if sam_mask_agreement is not None:
        components["segmentation_agreement"] = float(max(0.0, min(1.0, sam_mask_agreement)))
        active_weights["segmentation_agreement"] = w.segmentation_agreement

    if dino_confidence is not None:
        components["dino_agreement"] = float(max(0.0, min(1.0, dino_confidence)))
        active_weights["dino_agreement"] = w.dino_agreement

    if position_context is not None:
        components["position_context"] = float(max(0.0, min(1.0, position_context)))
        active_weights["position_context"] = w.position_context

    weight_sum = sum(active_weights.values())
    pole_score = (
        sum(components[k] * active_weights[k] for k in components) / weight_sum
        if weight_sum > 0 else 0.0
    )
    return {
        "pole_score": round(float(max(0.0, min(1.0, pole_score))), 4),
        "components": {k: round(v, 4) for k, v in components.items()},
    }


# --- Image loading -------------------------------------------------------------

def _to_gray(image: Union[str, "Path", np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image: {image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image type for OBB recovery: {type(image)}")


# --- Candidate generation --------------------------------------------------------

def _rect_corners_from_contour(contour: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_corners_canonical(np.asarray(box, dtype=np.float64))


def _fill_fraction(contour: np.ndarray, corners: np.ndarray) -> float:
    """Contour area vs. its fitted rectangle's area -- how solid/continuous
    the detected structure is along its own footprint."""
    _, half_len, half_width, _ = _long_axis(corners)
    rect_area = (2 * half_len) * (2 * half_width)
    if rect_area <= 1e-9:
        return 0.0
    contour_area = float(cv2.contourArea(contour))
    return float(max(0.0, min(1.0, contour_area / rect_area)))


def _edge_support_fraction(corners: np.ndarray, edges: np.ndarray, samples: int = 12) -> float:
    """Fraction of sample points along the candidate's two long edges that
    land on (or within 2px of) a real Canny edge pixel."""
    c = np.asarray(corners, dtype=np.float64)
    h, w = edges.shape[:2]
    pairs = [(c[0], c[3]), (c[1], c[2])]  # the two long edges given canonical [TL,TR,BR,BL]-style ordering
    hits = 0
    total = 0
    for p0, p1 in pairs:
        for t in np.linspace(0.0, 1.0, samples):
            x, y = p0 + t * (p1 - p0)
            xi, yi = int(round(x)), int(round(y))
            total += 1
            y0, y1 = max(0, yi - 2), min(h, yi + 3)
            x0, x1 = max(0, xi - 2), min(w, xi + 3)
            if y1 > y0 and x1 > x0 and edges[y0:y1, x0:x1].any():
                hits += 1
    return hits / total if total else 0.0


def _best_dino_overlap_confidence(corners: np.ndarray, dino_boxes: Optional[List[DetectionBox]]) -> Optional[float]:
    """The confidence of whichever DINO candidate (if any) most overlaps this
    recovery candidate -- a *weak* DINO box near a recovered shaft is
    corroborating evidence even though it wasn't strong enough on its own."""
    if not dino_boxes:
        return None
    from src.geometry_obb import xyxy_to_obb_corners, obb_iou
    best = None
    for det in dino_boxes:
        try:
            other_corners = np.asarray(det.corners) if det.corners else xyxy_to_obb_corners(*det.xyxy)
            iou = obb_iou(corners, other_corners)
        except Exception:
            continue
        if iou > 0.05 and (best is None or det.confidence > best):
            best = det.confidence
    return best


def generate_recovery_candidates(
    image: Union[str, "Path", np.ndarray, Image.Image],
    existing_boxes: List[DetectionBox],
    config: Optional[RecoveryConfig] = None,
    dino_boxes: Optional[List[DetectionBox]] = None,
) -> List[DetectionBox]:
    """
    Find candidate pole-shaped OBBs directly from image structure, for use
    when the primary DINO+SAM3 pipeline missed the pole (see
    should_run_recovery). Returns candidates sorted best-first, capped at
    config.max_candidates_per_image, each already structurally validated
    (obb_generator.validate_obb) and score-filtered (>= min_pole_score).

    `existing_boxes` is accepted for signature symmetry with the pipeline's
    other stages and future position-context scoring, but duplicate removal
    against them is NOT done here -- see recovery_dedup.py, applied by the
    caller after this returns, keeping generation and dedup independently
    testable.
    """
    if config is None:
        config = RecoveryConfig()
    if not config.enabled:
        return []

    gray = _to_gray(image)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 50, 150)
    if not edges.any():
        return []

    # Connect fragmented edge pixels of an elongated vertical structure into
    # one solid blob per candidate shaft, without also connecting unrelated
    # nearby edges together (a narrow, tall kernel only bridges gaps along
    # roughly the vertical direction).
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    connected = cv2.dilate(edges, vertical_kernel, iterations=2)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[DetectionBox] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4.0:
            continue
        corners = _rect_corners_from_contour(contour)
        errors = validate_obb(corners, w, h)
        if errors:
            continue

        unit, half_len, half_width, _ = _long_axis(corners)
        width_px = 2.0 * half_width
        if not (config.min_width_px <= width_px <= config.max_width_px):
            continue
        angle_deg = float(np.degrees(np.arctan2(abs(unit[0]), abs(unit[1]) + 1e-12)))
        if angle_deg > config.max_angle_from_vertical_deg:
            continue
        aspect_ratio = half_len / max(half_width, 1e-9)
        if aspect_ratio < config.min_aspect_ratio:
            continue

        fill_fraction = _fill_fraction(contour, corners)
        edge_support = _edge_support_fraction(corners, edges)
        dino_conf = _best_dino_overlap_confidence(corners, dino_boxes)

        score_result = compute_pole_score(
            corners, config,
            dino_confidence=dino_conf,
            fill_fraction=fill_fraction,
            edge_support_fraction=edge_support,
        )
        if score_result["pole_score"] < config.min_pole_score:
            continue

        needs_review = score_result["pole_score"] < config.review_threshold
        det = DetectionBox(
            xyxy=(
                float(corners[:, 0].min()), float(corners[:, 1].min()),
                float(corners[:, 0].max()), float(corners[:, 1].max()),
            ),
            corners=corners.tolist(),
            confidence=score_result["pole_score"],
            class_name="utility_pole",
            model_source="OBB_RECOVERY",
            needs_review=True,  # recovery never silently auto-accepts; see module docstring
            review_reasons=(["REVIEW_REQUIRED: recovery pole_score below review_threshold"] if needs_review else []),
            attributes={
                "recovery_score": score_result,
                "recovery_status": "REVIEW_REQUIRED" if needs_review else "RECOVERED_CANDIDATE",
                "recovery_geometry": {
                    "width_px": round(width_px, 2),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "angle_from_vertical_deg": round(angle_deg, 2),
                },
            },
        )
        candidates.append(det)

    candidates.sort(key=lambda c: c.attributes["recovery_score"]["pole_score"], reverse=True)
    return candidates[: config.max_candidates_per_image]


# --- Partial pole reconstruction -------------------------------------------------

def reconstruct_partial_pole(
    image: Union[str, "Path", np.ndarray, Image.Image],
    partial_mask: np.ndarray,
    config: Optional[RecoveryConfig] = None,
) -> Optional[Dict[str, Any]]:
    """
    Given a mask covering only PART of a pole (e.g. SAM3 segmented the lower
    section but missed the upper one), estimate the pole's centerline,
    orientation, and width from the observed portion, then search along that
    direction -- in the image, not blindly -- for continuing structure
    evidence, extending the OBB only as far as that evidence (and the
    observed segment's own length x config.max_extension_factor) supports.

    Returns None if the mask has no usable content. Returns
    {"corners": ..., "extension_up_px": ..., "extension_down_px": ...,
    "observed_length_px": ...} otherwise.
    """
    if config is None:
        config = RecoveryConfig()
    if partial_mask is None or partial_mask.sum() == 0:
        return None

    gray = _to_gray(image)
    h, w = gray.shape[:2]

    mask_u8 = (partial_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1.0:
        return None

    corners = _rect_corners_from_contour(contour)
    unit, half_len, half_width, center = _long_axis(corners)
    observed_length = 2.0 * half_len
    if observed_length < 1.0:
        return None

    # Two candidate extension directions: along +unit and -unit. Normalize so
    # unit always points toward increasing y ("downward") for one end and
    # -unit toward decreasing y ("upward") for the other, regardless of
    # _long_axis's arbitrary sign choice.
    down_dir = unit if unit[1] >= 0 else -unit
    up_dir = -down_dir

    # Reference appearance of the OBSERVED segment (what we know is really
    # the pole), sampled once, to compare candidate extension points against.
    ref_strip = _sample_strip(gray, center, unit, half_width, t=0.0, strip_half_len=max(2.0, half_len * 0.5))
    ref_mean = float(ref_strip.mean()) if ref_strip.size else 0.0
    ref_std = float(ref_strip.std()) if ref_strip.size else 1.0
    tolerance = max(15.0, ref_std * 2.5)

    top_end = center - half_len * down_dir  # the observed segment's own "up" end
    bottom_end = center + half_len * down_dir  # the observed segment's own "down" end

    max_extend = observed_length * config.max_extension_factor
    extension_up = _search_extension(gray, top_end, up_dir, half_width, ref_mean, tolerance, max_extend, w, h)
    extension_down = _search_extension(gray, bottom_end, down_dir, half_width, ref_mean, tolerance, max_extend, w, h)

    new_top = top_end + extension_up * up_dir
    new_bottom = bottom_end + extension_down * down_dir
    width_dir = np.array([-down_dir[1], down_dir[0]])  # perpendicular
    new_corners = order_corners_canonical(np.array([
        new_top + half_width * width_dir, new_top - half_width * width_dir,
        new_bottom - half_width * width_dir, new_bottom + half_width * width_dir,
    ]))

    errors = validate_obb(new_corners, w, h)
    if errors:
        # Extension pushed out of bounds/degenerate -- fall back to the
        # observed-only OBB rather than returning an invalid label.
        new_corners = corners
        extension_up = extension_down = 0.0
        errors = validate_obb(new_corners, w, h)
        if errors:
            return None

    return {
        "corners": new_corners.tolist(),
        "extension_up_px": round(float(extension_up), 2),
        "extension_down_px": round(float(extension_down), 2),
        "observed_length_px": round(float(observed_length), 2),
    }


def _sample_strip(gray: np.ndarray, point: np.ndarray, direction: np.ndarray, half_width: float,
                   t: float, strip_half_len: float) -> np.ndarray:
    """Grayscale values in a small strip centered at point + t*direction,
    spanning +/- strip_half_len along `direction` and +/- half_width
    perpendicular to it."""
    h, w = gray.shape[:2]
    width_dir = np.array([-direction[1], direction[0]])
    samples = []
    for dl in np.linspace(-strip_half_len, strip_half_len, max(3, int(strip_half_len))):
        for dw in np.linspace(-half_width, half_width, max(2, int(half_width))):
            p = point + t * direction + dl * direction + dw * width_dir
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < w and 0 <= yi < h:
                samples.append(gray[yi, xi])
    return np.array(samples, dtype=np.float64)


def _search_extension(gray: np.ndarray, start: np.ndarray, direction: np.ndarray, half_width: float,
                       ref_mean: float, tolerance: float, max_extend: float,
                       img_w: int, img_h: int, step: float = 2.0) -> float:
    """Step outward from `start` along `direction`, comparing each small
    strip's mean intensity to the observed segment's reference appearance.
    Stops -- and reports the extension distance up to that point -- as soon
    as the evidence no longer matches, or the image boundary / max_extend
    cap is reached. Never extends blindly past either of those."""
    extended = 0.0
    while extended + step <= max_extend:
        candidate = start + (extended + step) * direction
        if not (0 <= candidate[0] < img_w and 0 <= candidate[1] < img_h):
            break
        strip = _sample_strip(gray, start, direction, half_width, t=extended + step, strip_half_len=step)
        if strip.size == 0:
            break
        if abs(float(strip.mean()) - ref_mean) > tolerance:
            break
        extended += step
    return extended
