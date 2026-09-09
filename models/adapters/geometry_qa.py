"""
models/adapters/geometry_qa.py - Geometric Quality Assurance for SAM Masks/OBBs
=================================================================================

Runs AFTER SAM mask cleanup and OBB generation, BEFORE the final decision. Its
job is to catch masks/OBBs that are geometrically implausible for a utility
pole -- not to enforce a single rigid "poles must look like X" rule.

Mask cleanup itself (largest connected component, vertical morphological
open/close to drop thin horizontal wires) is already done upstream by
pole_sam._clean (used by both sam21_adapter and sam3_adapter) -- this module
does NOT re-clean the mask, it scores what cleanup already produced.

Thresholds are intentionally NOT single global hard-coded constants: every
threshold lives in GeometryQAConfig (loadable from configs/config.yaml,
section `verification_pipeline.geometry_qa`) so they can be tuned per-dataset
without editing code. Defaults here are reasonable starting points, not
claimed-optimal values -- see config.yaml for the same caveat.

Guarantees:
- Never hard-rejects purely on tilt: a tilted pole is a normal case (existing
  app-wide policy -- tilt is never forced to zero). Orientation only feeds a
  soft warning signal.
- Never hard-rejects purely on partial occlusion/boundary-touching: flagged
  as a warning signal, same as models/adapters/quality.py's existing policy.
- Only geometry that is actively implausible for ANY elongated pole (near
  full-image OBB, extremely squat/wide box, degenerate/empty mask, badly
  fragmented mask) drives geometry_status toward "fail".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from src.geometry_obb import signed_shoelace_area


@dataclass
class GeometryQAConfig:
    # Candidate covering more than this fraction of the image is suspicious
    # (a real pole crop should never be a huge fraction of a street-level photo).
    max_image_area_fraction: float = 0.35

    # OBB long-axis length below this many pixels is too short to be a
    # meaningfully-labeled pole instance.
    min_obb_length_px: float = 15.0

    # long-axis / short-axis of the rotated rect. Real poles are elongated;
    # below this the "pole" is about as wide as it is long (probably not a pole).
    min_length_to_thickness: float = 1.6

    # Orientation angle (degrees from vertical, 0=vertical, 90=horizontal).
    # Poles CAN be tilted (see module docstring) -- these only add warnings/
    # a fail past the second, much more extreme threshold.
    warn_tilt_deg: float = 35.0
    fail_tilt_deg: float = 65.0

    # mask pixel count / OBB polygon area. A tight-fitting rotated rect around
    # a real pole silhouette should still contain a reasonable fraction of
    # its own area as foreground (unlike a rect drawn around a scattered/
    # fragmented mask).
    min_mask_to_obb_ratio: float = 0.20

    # convex-hull solidity (contour_area / hull_area). Low solidity means a
    # ragged/fragmented/non-pole-shaped silhouette.
    min_solidity: float = 0.30

    min_mask_area_px: float = 40.0

    # Bucketing thresholds for the blended geometry_score in [0, 1].
    fail_score_below: float = 0.30
    warn_score_below: float = 0.60


DEFAULT_GEOMETRY_QA_CONFIG = GeometryQAConfig()


def _polygon_edge_lengths(corners: np.ndarray) -> Tuple[float, float]:
    """Canonical corners are TL,TR,BR,BL clockwise -- edge0=TL->TR, edge1=TR->BR."""
    e0 = float(np.linalg.norm(corners[1] - corners[0]))
    e1 = float(np.linalg.norm(corners[2] - corners[1]))
    return e0, e1


def _orientation_deg_from_vertical(corners: np.ndarray, long_edge_idx: int) -> float:
    """
    Angle (0-90) between the OBB's long axis and vertical (screen y-axis).
    0 = perfectly vertical pole, 90 = perfectly horizontal.
    """
    p0, p1 = (corners[0], corners[1]) if long_edge_idx == 0 else (corners[1], corners[2])
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    if dx == 0 and dy == 0:
        return 0.0
    angle_from_horizontal = np.degrees(np.arctan2(abs(dy), abs(dx)))
    return float(90.0 - angle_from_horizontal)


def _mask_touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    m = margin if margin < h and margin < w else max(1, min(h, w) - 1)
    return bool(
        mask[:m, :].any() or mask[-m:, :].any()
        or mask[:, :m].any() or mask[:, -m:].any()
    )


def compute_geometry_metrics(
    mask: Optional[np.ndarray],
    corners: np.ndarray,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    Compute the raw geometric measurements listed in the spec: areas, ratios,
    orientation, compactness/convexity, boundary distance, etc. Pure
    measurement -- no pass/fail judgement (see score_geometry for that).
    """
    corners = np.asarray(corners, dtype=np.float64)
    obb_area = abs(signed_shoelace_area(corners))
    bbox_x0, bbox_y0 = corners[:, 0].min(), corners[:, 1].min()
    bbox_x1, bbox_y1 = corners[:, 0].max(), corners[:, 1].max()
    bbox_area = float((bbox_x1 - bbox_x0) * (bbox_y1 - bbox_y0))

    e0, e1 = _polygon_edge_lengths(corners)
    long_edge_idx = 0 if e0 >= e1 else 1
    obb_length = max(e0, e1)
    obb_thickness = max(min(e0, e1), 1e-6)
    orientation_angle_deg = _orientation_deg_from_vertical(corners, long_edge_idx)

    img_area = float(max(image_width * image_height, 1))
    pct_image_area = obb_area / img_area

    dist_left = bbox_x0
    dist_top = bbox_y0
    dist_right = image_width - bbox_x1
    dist_bottom = image_height - bbox_y1
    boundary_distance_px = float(min(dist_left, dist_top, dist_right, dist_bottom))

    mask_area = 0.0
    mask_to_obb_ratio = 0.0
    compactness = 0.0
    convexity = 0.0
    touches_boundary = False
    num_components = 0
    largest_component_fraction = 0.0

    if mask is not None and mask.size > 0:
        binary = (mask > 0).astype(np.uint8)
        mask_area = float(binary.sum())
        touches_boundary = _mask_touches_border(binary)

        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_components = max(0, n_cc - 1)
        if num_components > 0:
            largest_component_fraction = float(stats[1:, cv2.CC_STAT_AREA].max()) / max(mask_area, 1.0)

        if obb_area > 1e-6:
            mask_to_obb_ratio = mask_area / obb_area

        contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 1e-6:
                compactness = float(4.0 * np.pi * contour_area / (perimeter ** 2))
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 1e-6:
                convexity = float(contour_area / hull_area)

    return {
        "mask_area": mask_area,
        "bbox_area": bbox_area,
        "obb_area": obb_area,
        "mask_to_obb_ratio": round(mask_to_obb_ratio, 4),
        "aspect_ratio": round(obb_length / obb_thickness, 3),
        "obb_width": round(obb_thickness, 2),
        "obb_height": round(obb_length, 2),
        "orientation_angle_deg": round(orientation_angle_deg, 2),
        "mask_compactness": round(compactness, 4),
        "mask_convexity": round(convexity, 4),
        "pct_image_area": round(pct_image_area, 5),
        "boundary_distance_px": round(boundary_distance_px, 2),
        "touches_image_boundary": touches_boundary,
        "num_mask_components": num_components,
        "largest_component_fraction": round(largest_component_fraction, 4),
    }


def score_geometry(
    metrics: Dict[str, Any],
    config: GeometryQAConfig = DEFAULT_GEOMETRY_QA_CONFIG,
) -> Dict[str, Any]:
    """
    Turn raw geometry metrics into a transparent geometry_score in [0, 1] and
    a geometry_status bucket (pass / warning / fail), with human-readable
    flags explaining any deduction. No single flag other than the
    hard-implausibility ones (near-empty mask, near-full-image OBB, extreme
    tilt) can push straight to "fail" -- everything else only nudges the
    blended score down into "warning" territory, consistent with "do not
    blindly reject unusual poles."
    """
    flags: List[str] = []
    hard_fail = False

    if metrics["mask_area"] < config.min_mask_area_px:
        flags.append(f"mask too small ({metrics['mask_area']:.0f}px)")
        hard_fail = True

    if metrics["pct_image_area"] > config.max_image_area_fraction:
        flags.append(f"OBB covers {metrics['pct_image_area']*100:.1f}% of image (excessively large)")
        hard_fail = True

    if metrics["obb_height"] < config.min_obb_length_px:
        flags.append(f"OBB long axis only {metrics['obb_height']:.1f}px (excessively short)")
        hard_fail = True

    if metrics["aspect_ratio"] < config.min_length_to_thickness:
        flags.append(f"aspect ratio {metrics['aspect_ratio']:.2f} too low (excessively wide/square)")

    if metrics["orientation_angle_deg"] > config.fail_tilt_deg:
        flags.append(f"orientation {metrics['orientation_angle_deg']:.1f} deg from vertical (implausible)")
        hard_fail = True
    elif metrics["orientation_angle_deg"] > config.warn_tilt_deg:
        flags.append(f"orientation {metrics['orientation_angle_deg']:.1f} deg from vertical (strongly tilted)")

    if metrics["mask_to_obb_ratio"] < config.min_mask_to_obb_ratio:
        flags.append(f"mask/OBB fill ratio {metrics['mask_to_obb_ratio']:.2f} low (noisy/disconnected mask)")

    if metrics["mask_convexity"] > 0 and metrics["mask_convexity"] < config.min_solidity:
        flags.append(f"mask solidity {metrics['mask_convexity']:.2f} low (fragmented/irregular silhouette)")

    if metrics["num_mask_components"] > 1 and metrics["largest_component_fraction"] < 0.6:
        flags.append(f"mask fragmented across {metrics['num_mask_components']} components")

    if metrics["touches_image_boundary"]:
        flags.append("mask/OBB touches image boundary (may be cut off)")

    # Weighted blend of the continuous signals (each already 0-1-ish or
    # normalized below). Fragmentation/solidity/fill-ratio are the strongest
    # signals for "this mask leaked onto a building/tree/sky", so they carry
    # the most weight; orientation is intentionally the lightest since tilt
    # alone is not disqualifying.
    fill_component = min(1.0, metrics["mask_to_obb_ratio"] / max(config.min_mask_to_obb_ratio * 2, 1e-6))
    solidity_component = min(1.0, metrics["mask_convexity"] / max(config.min_solidity * 2, 1e-6)) if metrics["mask_convexity"] > 0 else 0.5
    aspect_component = min(1.0, metrics["aspect_ratio"] / max(config.min_length_to_thickness * 2, 1e-6))
    size_component = 1.0 - min(1.0, metrics["pct_image_area"] / max(config.max_image_area_fraction, 1e-6))
    tilt_component = 1.0 - min(1.0, metrics["orientation_angle_deg"] / max(config.fail_tilt_deg, 1e-6))

    geometry_score = (
        0.30 * fill_component
        + 0.25 * solidity_component
        + 0.20 * aspect_component
        + 0.15 * size_component
        + 0.10 * tilt_component
    )
    geometry_score = round(max(0.0, min(1.0, geometry_score)), 4)

    if hard_fail or geometry_score < config.fail_score_below:
        status = "fail"
    elif geometry_score < config.warn_score_below or flags:
        status = "warning" if not hard_fail else "fail"
    else:
        status = "pass"

    return {"geometry_score": geometry_score, "geometry_status": status, "geometry_flags": flags}


def run_geometry_qa(
    mask: Optional[np.ndarray],
    corners: np.ndarray,
    image_width: int,
    image_height: int,
    config: GeometryQAConfig = DEFAULT_GEOMETRY_QA_CONFIG,
) -> Dict[str, Any]:
    """Convenience wrapper: compute_geometry_metrics() + score_geometry() in one call."""
    metrics = compute_geometry_metrics(mask, corners, image_width, image_height)
    result = score_geometry(metrics, config)
    result["metrics"] = metrics
    return result


def config_from_dict(d: Optional[Dict[str, Any]]) -> GeometryQAConfig:
    """Build a GeometryQAConfig from a plain dict (e.g. parsed from config.yaml),
    falling back to defaults for any missing/unknown keys -- never crashes on
    a partially-specified or absent config section."""
    if not d:
        return GeometryQAConfig()
    valid_fields = {f for f in GeometryQAConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return GeometryQAConfig(**kwargs)
