"""
src/geometry_obb.py - Deterministic Oriented Bounding Box (OBB) Geometry
========================================================================

Canonical representation and deterministic ordering of 4-corner Oriented Bounding
Boxes in image/screen coordinates (x right, y down).

Mathematical guarantees:
1. Corner 0 is canonical Top-Left vertex (minimizing x + y, breaking ties by min y then min x).
2. Corners 1 -> 2 -> 3 proceed strictly CLOCKWISE in screen coordinates.
3. Polygon winding verified using the signed shoelace area:
      sum_{i=0}^3 (x_i * y_{i+1} - x_{i+1} * y_i) > 0  (where y points down).
4. Permutation invariant: regardless of input vertex order, output ordering is identical.
5. Invariance guarantee: save -> reload -> export produces identical sequence and positive signed area.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
import cv2
from shapely.geometry import Polygon


def signed_shoelace_area(pts: Union[np.ndarray, List[List[float]], List[Tuple[float, float]]]) -> float:
    """
    Calculate the signed shoelace area of a polygon in screen coordinates (x right, y down).
    
    Formula:
        2 * A = sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i) with (x_n, y_n) = (x_0, y_0)
        
    In screen coordinates (+y points down):
        A > 0 indicates CLOCKWISE winding.
        A < 0 indicates COUNTER-CLOCKWISE winding.
        A == 0 indicates collinear or degenerate points.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    if len(pts_arr) < 3:
        return 0.0
    n = len(pts_arr)
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts_arr[i]
        x2, y2 = pts_arr[(i + 1) % n]
        area2 += (x1 * y2 - x2 * y1)
    return area2 / 2.0


def is_clockwise_screen(pts: Union[np.ndarray, List]) -> bool:
    """Return True if vertices follow a strict clockwise path in screen coordinates."""
    return signed_shoelace_area(pts) > 1e-7


def order_corners_canonical(pts: Union[np.ndarray, List]) -> np.ndarray:
    """
    Order 4 corners of an oriented bounding box into canonical representation.
    
    Guarantees:
    - Index 0 is the canonical top-left vertex: minimizes (x + y).
      In case of tie (within 1e-5), selects point with minimum y, then minimum x.
    - Points proceed strictly CLOCKWISE: 0 (TL) -> 1 (TR) -> 2 (BR) -> 3 (BL).
    - Signed shoelace area in screen space is strictly positive (> 0).
    - Input order invariant: any permutation of the same 4 points produces identical output.
    
    Args:
        pts: (4, 2) array or list of (x, y) coordinates.
        
    Returns:
        np.ndarray of shape (4, 2) with canonical clockwise ordering.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    if pts_arr.shape != (4, 2):
        raise ValueError(f"Expected array of shape (4, 2), got {pts_arr.shape}")
        
    # 1. Compute centroid
    cx, cy = np.mean(pts_arr, axis=0)
    
    # 2. Sort by polar angle around centroid in screen coordinates (-pi to pi)
    # In screen coordinates (x right, y down), atan2(dy, dx) increases clockwise:
    # Top (-pi/2) -> Right (0) -> Bottom (+pi/2) -> Left (+/-pi)
    angles = np.arctan2(pts_arr[:, 1] - cy, pts_arr[:, 0] - cx)
    sort_idx = np.argsort(angles)
    sorted_pts = pts_arr[sort_idx]
    
    # 3. Mathematically verify winding via signed shoelace area
    area = signed_shoelace_area(sorted_pts)
    if area < 0:
        # If counter-clockwise, reverse sequence
        sorted_pts = sorted_pts[::-1]
        
    # 4. Find canonical top-left vertex
    # Top-left is defined as minimizing (x + y)
    scores = sorted_pts[:, 0] + sorted_pts[:, 1]
    min_score = np.min(scores)
    ties = np.where(np.abs(scores - min_score) < 1e-5)[0]
    if len(ties) > 1:
        # Tie break on min y then min x
        best_local_idx = ties[np.lexsort((sorted_pts[ties, 0], sorted_pts[ties, 1]))[0]]
    else:
        best_local_idx = ties[0]
        
    # 5. Roll array so canonical top-left is at index 0
    ordered = np.roll(sorted_pts, -best_local_idx, axis=0)
    
    # Sanity check shoelace area remains positive
    final_area = signed_shoelace_area(ordered)
    if final_area <= 0:
        raise ValueError(f"Ordering error: non-positive shoelace area {final_area}")
        
    return ordered


def xyxy_to_obb_corners(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """
    Convert axis-aligned bounding box (x1, y1, x2, y2) to 4 canonical OBB corners.
    
    Returns:
        (4, 2) array: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] (canonical TL, TR, BR, BL).
    """
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype=np.float64)
    return order_corners_canonical(corners)


def obb_corners_to_xyxy(corners: Union[np.ndarray, List]) -> Tuple[float, float, float, float]:
    """
    Convert 4 OBB corners to an enclosing axis-aligned bounding box (min_x, min_y, max_x, max_y).
    """
    pts = np.asarray(corners, dtype=np.float64)
    min_x = float(np.min(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_x = float(np.max(pts[:, 0]))
    max_y = float(np.max(pts[:, 1]))
    return (min_x, min_y, max_x, max_y)


def mask_to_obb_corners(mask: np.ndarray, min_area: float = 10.0) -> Optional[np.ndarray]:
    """
    Compute deterministic 4-corner OBB from a binary segmentation mask.
    
    Args:
        mask: 2D uint8 numpy array where 255/1 indicates foreground.
        min_area: Minimum contour area in pixels to consider.
        
    Returns:
        (4, 2) numpy array of canonical corners, or None if no valid contour.
    """
    if mask is None or mask.size == 0:
        return None
        
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get largest contour by area
    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) < min_area:
        return None
        
    # cv2.minAreaRect gives ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(largest_cnt)
    raw_box = cv2.boxPoints(rect)  # 4 points
    
    return order_corners_canonical(raw_box)


def normalize_corners(corners: Union[np.ndarray, List], img_width: int, img_height: int) -> np.ndarray:
    """
    Normalize pixel coordinates into [0.0, 1.0] range relative to image dimensions.
    """
    pts = np.asarray(corners, dtype=np.float64).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(img_width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(img_height), 0.0, 1.0)
    return pts


def denormalize_corners(corners_norm: Union[np.ndarray, List], img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized [0.0, 1.0] coordinates to absolute pixel coordinates.
    """
    pts = np.asarray(corners_norm, dtype=np.float64).copy()
    pts[:, 0] = pts[:, 0] * float(img_width)
    pts[:, 1] = pts[:, 1] * float(img_height)
    return pts


def obb_corners_to_yolo_obb_line(
    corners: Union[np.ndarray, List],
    img_width: int,
    img_height: int,
    class_id: int = 0
) -> str:
    """
    Format 4 corners into standard YOLO-OBB text representation:
    "class_id x1 y1 x2 y2 x3 y3 x4 y4" with coordinates normalized to [0.0, 1.0].
    
    Guarantees:
    - Input corners are canonicalized before serialization.
    """
    ordered = order_corners_canonical(corners)
    norm = normalize_corners(ordered, img_width, img_height)
    coords_str = " ".join(f"{coord:.6f}" for pt in norm for coord in pt)
    return f"{int(class_id)} {coords_str}"


def yolo_obb_line_to_corners(
    line: str,
    img_width: int,
    img_height: int
) -> Tuple[int, np.ndarray]:
    """
    Parse a standard YOLO-OBB text representation into (class_id, canonical_corners_px).
    
    Input line format:
    "class_id x1 y1 x2 y2 x3 y3 x4 y4"
    """
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Invalid YOLO-OBB line (expected 9 tokens, got {len(parts)}): '{line}'")
        
    class_id = int(parts[0])
    raw_coords = [float(p) for p in parts[1:]]
    pts_norm = np.array(raw_coords, dtype=np.float64).reshape((4, 2))
    pts_px = denormalize_corners(pts_norm, img_width, img_height)
    canonical = order_corners_canonical(pts_px)
    return class_id, canonical


def obb_iou(corners1: Union[np.ndarray, List], corners2: Union[np.ndarray, List]) -> float:
    """
    Calculate exact geometric Intersection over Union (IoU) between two 4-corner OBBs.
    Uses Shapely polygon intersection.
    """
    p1 = Polygon(np.asarray(corners1, dtype=np.float64))
    p2 = Polygon(np.asarray(corners2, dtype=np.float64))
    
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
        
    if not p1.is_valid or not p2.is_valid:
        return 0.0
        
    if p1.area <= 1e-7 or p2.area <= 1e-7:
        return 0.0
        
    intersection_area = p1.intersection(p2).area
    union_area = p1.area + p2.area - intersection_area
    if union_area <= 1e-7:
        return 0.0
    return float(intersection_area / union_area)
