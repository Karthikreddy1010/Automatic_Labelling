"""
models/adapters/obb_generator.py - OBB Generation & Structural Validation
===========================================================================

Thin wrapper around src/geometry_obb.py's existing mask->OBB conversion and
canonical ordering. Does NOT reimplement geometry math -- it exists to give
the mask-to-label step a single, testable entry point that both generates the
OBB AND validates it structurally before it's allowed downstream, per the
spec's OBB-generation requirements:
  - all coordinates inside the image
  - exactly four valid corners
  - polygon ordering is valid (positive shoelace area / clockwise)
  - the polygon does not self-intersect
  - width/height are positive, area is non-zero

The OBB geometry always originates from the SAM segmentation mask (or, if SAM
produced no usable mask, the axis-aligned detector box as an explicit,
labeled fallback) -- never from Qwen or any other verifier.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon

from src.geometry_obb import (
    mask_to_obb_corners,
    xyxy_to_obb_corners,
    signed_shoelace_area,
    order_corners_canonical,
)

_COORD_TOLERANCE_PX = 1.0  # allow a hair of float slop at the image edge


def validate_obb(
    corners: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[str]:
    """Return a list of human-readable validation errors; empty list means valid."""
    errors: List[str] = []
    arr = np.asarray(corners, dtype=np.float64)

    if arr.shape != (4, 2):
        errors.append(f"expected 4 corners, got shape {arr.shape}")
        return errors

    if not np.all(np.isfinite(arr)):
        errors.append("non-finite coordinate in OBB")
        return errors

    xs, ys = arr[:, 0], arr[:, 1]
    if xs.min() < -_COORD_TOLERANCE_PX or xs.max() > image_width + _COORD_TOLERANCE_PX:
        errors.append(f"x coordinate outside image bounds [0, {image_width}]")
    if ys.min() < -_COORD_TOLERANCE_PX or ys.max() > image_height + _COORD_TOLERANCE_PX:
        errors.append(f"y coordinate outside image bounds [0, {image_height}]")

    area = signed_shoelace_area(arr)
    if area <= 1e-6:
        errors.append("non-positive polygon area (degenerate or invalid winding)")

    poly = Polygon(arr)
    if not poly.is_valid:
        errors.append("polygon is self-intersecting or otherwise invalid")

    e0 = float(np.linalg.norm(arr[1] - arr[0]))
    e1 = float(np.linalg.norm(arr[2] - arr[1]))
    if e0 <= 1e-6 or e1 <= 1e-6:
        errors.append("OBB has zero width or height")

    return errors


def generate_and_validate_obb(
    mask: Optional[np.ndarray],
    fallback_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    min_mask_area: float = 10.0,
) -> Dict[str, Any]:
    """
    Generate a canonical 4-corner OBB from a segmentation mask, falling back
    to the axis-aligned detector box (explicitly labeled as such) if the mask
    is missing or too small to produce a usable contour. Always validates the
    result before returning it.

    Returns:
        {
          "corners": np.ndarray(4,2) or None (None only if fallback itself is invalid),
          "source": "mask" | "xyxy_fallback",
          "errors": [str, ...],   # empty if valid
          "valid": bool,
        }
    """
    corners: Optional[np.ndarray] = None
    source = "mask"

    if mask is not None and mask.size > 0:
        corners = mask_to_obb_corners(mask, min_area=min_mask_area)

    if corners is None:
        corners = xyxy_to_obb_corners(*fallback_xyxy)
        source = "xyxy_fallback"
    else:
        corners = order_corners_canonical(corners)

    errors = validate_obb(corners, image_width, image_height)
    return {
        "corners": corners if not errors else None,
        "raw_corners": corners,
        "source": source,
        "errors": errors,
        "valid": len(errors) == 0,
    }
