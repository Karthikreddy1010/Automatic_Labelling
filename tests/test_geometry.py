"""
tests/test_geometry.py - Comprehensive Unit Tests for Deterministic OBB Geometry
================================================================================

Validates:
- Horizontal rectangles (0 deg)
- Vertical rectangles (90 deg)
- Positive rotations (+15, +30, +45, +60 deg)
- Negative rotations (-15, -30, -45, -60 deg)
- Near-90 deg rotations (89.9, 90.1 deg)
- Near-180 deg rotations (179.9, 180.1 deg)
- Clockwise winding in screen coordinates (signed shoelace area > 0)
- Permutation invariance (any initial order produces identical canonical sequence)
- YOLO-OBB serialization roundtrip invariance (corners -> line -> corners -> line)
- Mask to OBB extraction
- OBB IoU calculation
"""

import math
import unittest
import numpy as np
import cv2

from src.geometry_obb import (
    signed_shoelace_area,
    is_clockwise_screen,
    order_corners_canonical,
    xyxy_to_obb_corners,
    obb_corners_to_xyxy,
    mask_to_obb_corners,
    obb_corners_to_yolo_obb_line,
    yolo_obb_line_to_corners,
    obb_iou,
)


def make_rotated_box(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Generate 4 corners of a rotated rectangle in image coordinates (x right, y down)."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    # Base canonical rectangle relative to center (TL, TR, BR, BL)
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh]
    ], dtype=np.float64)
    # Screen coordinate rotation matrix (x right, y down)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    rotated = np.dot(corners, R.T)
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


class TestDeterministicGeometry(unittest.TestCase):

    def test_signed_shoelace_screen_coordinates(self):
        """Verify that clockwise screen-coordinate vertices yield positive shoelace area."""
        # Screen coordinates: (0,0) is top-left, x right, y down
        # Clockwise polygon: (10, 10) -> (50, 10) -> (50, 80) -> (10, 80)
        cw_box = np.array([[10, 10], [50, 10], [50, 80], [10, 80]], dtype=np.float64)
        area_cw = signed_shoelace_area(cw_box)
        self.assertGreater(area_cw, 0.0, "Clockwise screen box must have signed shoelace area > 0")
        self.assertAlmostEqual(area_cw, 40 * 70, places=4)
        self.assertTrue(is_clockwise_screen(cw_box))

        # Counter-clockwise polygon: reverse order
        ccw_box = cw_box[::-1]
        area_ccw = signed_shoelace_area(ccw_box)
        self.assertLess(area_ccw, 0.0, "Counter-clockwise screen box must have signed shoelace area < 0")
        self.assertFalse(is_clockwise_screen(ccw_box))

    def test_rotations_shoelace_and_clockwise(self):
        """Test comprehensive rotations: 0, 90, pos, neg, near-90, near-180."""
        test_angles = [
            0.0, 15.0, 30.0, 45.0, 60.0,
            89.9, 90.0, 90.1,
            120.0, 135.0,
            179.9, 180.0, 180.1,
            -15.0, -30.0, -45.0, -60.0,
            -89.9, -90.0
        ]
        cx, cy, w, h = 200.0, 300.0, 30.0, 150.0
        for angle in test_angles:
            with self.subTest(angle=angle):
                raw_box = make_rotated_box(cx, cy, w, h, angle)
                # Shuffle input corners to guarantee order-independence
                shuffled = raw_box[[2, 0, 3, 1]]
                ordered = order_corners_canonical(shuffled)
                
                # Check area is positive (clockwise)
                area = signed_shoelace_area(ordered)
                self.assertGreater(area, 0.0, f"Area for angle {angle} must be positive, got {area}")
                self.assertAlmostEqual(area, w * h, places=2, msg=f"Area mismatch for angle {angle}")

                # Check that ordered[0] is top-left:
                # scores = x + y for all vertices
                scores = ordered[:, 0] + ordered[:, 1]
                self.assertAlmostEqual(scores[0], np.min(scores), places=4,
                                       msg=f"Corner 0 must minimize (x+y) for angle {angle}")

    def test_permutation_invariance(self):
        """Regardless of which corner is listed first or winding, canonical output is identical."""
        box = make_rotated_box(150, 150, 40, 120, 35.0)
        canonical_ref = order_corners_canonical(box)
        
        # Test all 24 possible permutations of 4 vertices
        from itertools import permutations
        for perm in permutations([0, 1, 2, 3]):
            shuffled = box[list(perm)]
            result = order_corners_canonical(shuffled)
            np.testing.assert_allclose(
                result, canonical_ref, atol=1e-5,
                err_msg=f"Permutation {perm} did not match canonical reference"
            )

    def test_xyxy_conversion(self):
        """Test axis-aligned box to canonical OBB and back."""
        corners = xyxy_to_obb_corners(50, 60, 120, 200)
        self.assertEqual(corners.shape, (4, 2))
        self.assertGreater(signed_shoelace_area(corners), 0)
        
        # Check TL, TR, BR, BL order
        np.testing.assert_allclose(corners[0], [50, 60])
        np.testing.assert_allclose(corners[1], [120, 60])
        np.testing.assert_allclose(corners[2], [120, 200])
        np.testing.assert_allclose(corners[3], [50, 200])
        
        # Convert back
        x1, y1, x2, y2 = obb_corners_to_xyxy(corners)
        self.assertAlmostEqual(x1, 50.0)
        self.assertAlmostEqual(y1, 60.0)
        self.assertAlmostEqual(x2, 120.0)
        self.assertAlmostEqual(y2, 200.0)

    def test_yolo_obb_roundtrip_invariance(self):
        """Test save -> reload -> export invariance: line -> corners -> line -> corners."""
        img_w, img_h = 1920, 1080
        test_angles = [0.0, 25.0, 89.9, 90.0, 179.9, -45.0]
        
        for angle in test_angles:
            with self.subTest(angle=angle):
                orig_box = make_rotated_box(500, 400, 60, 250, angle)
                canonical_orig = order_corners_canonical(orig_box)
                
                # Serialize to YOLO-OBB line
                line1 = obb_corners_to_yolo_obb_line(canonical_orig, img_w, img_h, class_id=0)
                
                # Parse back
                cls_id, parsed_corners1 = yolo_obb_line_to_corners(line1, img_w, img_h)
                self.assertEqual(cls_id, 0)
                
                # Re-serialize
                line2 = obb_corners_to_yolo_obb_line(parsed_corners1, img_w, img_h, class_id=cls_id)
                
                # Check lines match exactly (to 5 decimal places)
                coords1 = [float(x) for x in line1.split()[1:]]
                coords2 = [float(x) for x in line2.split()[1:]]
                np.testing.assert_allclose(coords1, coords2, atol=1e-5)
                
                # Check pixel corners match closely
                np.testing.assert_allclose(canonical_orig, parsed_corners1, atol=1e-2)
                
                # Area must remain strictly positive
                self.assertGreater(signed_shoelace_area(parsed_corners1), 0.0)

    def test_mask_to_obb_extraction(self):
        """Test extracting canonical OBB from a synthetic binary mask."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        # Draw a rotated rectangle on the mask
        center = (150, 150)
        size = (40, 120)
        angle = 20.0
        rect = (center, size, angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 255)
        
        extracted_obb = mask_to_obb_corners(mask)
        self.assertIsNotNone(extracted_obb)
        self.assertEqual(extracted_obb.shape, (4, 2))
        self.assertGreater(signed_shoelace_area(extracted_obb), 0.0)
        
        # Verify Corner 0 is top-left
        scores = extracted_obb[:, 0] + extracted_obb[:, 1]
        self.assertAlmostEqual(scores[0], np.min(scores), places=3)
        
        # Verify IoU between extracted OBB and ground truth is high (> 0.95)
        gt_obb = order_corners_canonical(cv2.boxPoints(rect))
        iou = obb_iou(extracted_obb, gt_obb)
        self.assertGreater(iou, 0.95, f"IoU too low: {iou}")

    def test_obb_iou_calculation(self):
        """Test geometric IoU between overlapping and non-overlapping OBBs."""
        box1 = xyxy_to_obb_corners(10, 10, 50, 50)
        # Identical box -> IoU = 1.0
        self.assertAlmostEqual(obb_iou(box1, box1), 1.0, places=4)
        
        # Disjoint box -> IoU = 0.0
        box2 = xyxy_to_obb_corners(100, 100, 150, 150)
        self.assertAlmostEqual(obb_iou(box1, box2), 0.0, places=4)
        
        # 50% overlap box
        box3 = xyxy_to_obb_corners(30, 10, 50, 50)  # half width overlap
        iou = obb_iou(box1, box3)
        # box1 area = 40 * 40 = 1600. box3 area = 20 * 40 = 800.
        # intersection = 20 * 40 = 800. union = 1600 + 800 - 800 = 1600. IoU = 800/1600 = 0.5
        self.assertAlmostEqual(iou, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
