"""
tests/test_obb_generator.py - OBB Generation & Validation Unit Tests
========================================================================

Pure unit tests -- no models, no network. Validates
models/adapters/obb_generator.py's structural checks (in-bounds corners,
valid winding/non-self-intersection, positive area/width/height) and the
mask -> OBB -> validated-corners pipeline, per the spec's explicit
requirement to test OBB coordinate normalization/validation.
"""

import unittest
import numpy as np

from models.adapters.obb_generator import validate_obb, generate_and_validate_obb
from src.geometry_obb import (
    xyxy_to_obb_corners,
    obb_corners_to_yolo_obb_line,
    yolo_obb_line_to_corners,
    mask_to_obb_corners,
)


class TestValidateOBB(unittest.TestCase):

    def test_valid_axis_aligned_box_passes(self):
        corners = xyxy_to_obb_corners(10, 10, 50, 200)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertEqual(errors, [])

    def test_out_of_bounds_x_fails(self):
        corners = np.array([[10, 10], [700, 10], [700, 200], [10, 200]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(any("outside image bounds" in e for e in errors))

    def test_out_of_bounds_negative_y_fails(self):
        corners = np.array([[10, -50], [60, -50], [60, 200], [10, 200]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(any("outside image bounds" in e for e in errors))

    def test_degenerate_zero_area_fails(self):
        corners = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(len(errors) > 0)

    def test_self_intersecting_bowtie_fails(self):
        # Bowtie: corners cross over each other -- shapely should flag invalid.
        corners = np.array([[10, 10], [100, 100], [10, 100], [100, 10]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(len(errors) > 0)

    def test_wrong_corner_count_fails(self):
        corners = np.array([[10, 10], [50, 10], [50, 50]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(any("expected 4 corners" in e for e in errors))

    def test_non_finite_coordinate_fails(self):
        corners = np.array([[10, 10], [float("nan"), 10], [50, 50], [10, 50]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(any("non-finite" in e for e in errors))

    def test_zero_width_fails(self):
        corners = np.array([[10, 10], [10, 10], [10, 200], [10, 200]], dtype=np.float64)
        errors = validate_obb(corners, image_width=640, image_height=640)
        self.assertTrue(len(errors) > 0)


class TestGenerateAndValidateOBB(unittest.TestCase):

    def test_valid_mask_produces_mask_sourced_obb(self):
        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[40:400, 195:215] = 255
        result = generate_and_validate_obb(mask, fallback_xyxy=(190, 35, 220, 405), image_width=640, image_height=640)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source"], "mask")
        self.assertIsNotNone(result["corners"])
        self.assertEqual(np.asarray(result["corners"]).shape, (4, 2))

    def test_none_mask_falls_back_to_xyxy(self):
        result = generate_and_validate_obb(None, fallback_xyxy=(10, 10, 50, 200), image_width=640, image_height=640)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source"], "xyxy_fallback")

    def test_empty_mask_falls_back_to_xyxy(self):
        mask = np.zeros((640, 640), dtype=np.uint8)
        result = generate_and_validate_obb(mask, fallback_xyxy=(10, 10, 50, 200), image_width=640, image_height=640)
        self.assertEqual(result["source"], "xyxy_fallback")
        self.assertTrue(result["valid"])

    def test_fallback_box_outside_image_is_invalid(self):
        result = generate_and_validate_obb(None, fallback_xyxy=(10, 10, 900, 200), image_width=640, image_height=640)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["corners"])
        self.assertTrue(len(result["errors"]) > 0)


class TestYoloNormalizationRoundTrip(unittest.TestCase):
    """Confirms the OBB -> YOLO-OBB text -> OBB round trip stays exact
    (obb_generator's validated corners feed directly into this existing,
    already-tested serialization -- re-verified here for the new candidates
    that now flow through obb_generator first)."""

    def test_round_trip_preserves_geometry(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:350, 300:330] = 255
        result = generate_and_validate_obb(mask, fallback_xyxy=(295, 95, 335, 355), image_width=640, image_height=480)
        self.assertTrue(result["valid"])
        corners = result["corners"]

        line = obb_corners_to_yolo_obb_line(corners, 640, 480, class_id=0)
        parts = line.split()
        self.assertEqual(len(parts), 9)
        coords = [float(p) for p in parts[1:]]
        self.assertTrue(all(0.0 <= c <= 1.0 for c in coords))

        _, round_tripped = yolo_obb_line_to_corners(line, 640, 480)
        np.testing.assert_allclose(round_tripped, corners, atol=0.5)


if __name__ == "__main__":
    unittest.main()
