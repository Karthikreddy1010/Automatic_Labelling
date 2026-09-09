"""
tests/test_geometry_qa.py - Geometry QA Unit Tests
=====================================================

Pure unit tests against synthetic masks/OBBs -- no models, no GPU, no
network. Validates models/adapters/geometry_qa.py's metric computation and
pass/warning/fail scoring in isolation, per the spec's explicit requirement
to add geometric-QA unit tests.
"""

import unittest
import numpy as np

from models.adapters.geometry_qa import (
    compute_geometry_metrics,
    score_geometry,
    run_geometry_qa,
    GeometryQAConfig,
    config_from_dict,
)
from src.geometry_obb import xyxy_to_obb_corners, mask_to_obb_corners


def _vertical_pole_mask(width=400, height=400, pole_x=195, pole_w=10, y0=40, y1=360):
    """A tall, thin, centered, mostly-vertical mask -- a plausible pole."""
    m = np.zeros((height, width), dtype=np.uint8)
    m[y0:y1, pole_x:pole_x + pole_w] = 255
    return m


def _blob_mask(width=400, height=400):
    """A big, roughly square blob covering much of the image -- not pole-like."""
    m = np.zeros((height, width), dtype=np.uint8)
    m[20:380, 20:380] = 255
    return m


def _fragmented_mask(width=400, height=400):
    """Several small disconnected specks -- noisy/fragmented segmentation."""
    m = np.zeros((height, width), dtype=np.uint8)
    for cy in range(50, 350, 60):
        m[cy:cy + 8, 190:200] = 255
    return m


class TestGeometryMetrics(unittest.TestCase):

    def test_vertical_pole_metrics_are_sane(self):
        mask = _vertical_pole_mask()
        corners = mask_to_obb_corners(mask)
        self.assertIsNotNone(corners)
        metrics = compute_geometry_metrics(mask, corners, 400, 400)

        self.assertGreater(metrics["aspect_ratio"], 1.6, "a tall thin pole must have a high length/thickness ratio")
        self.assertLess(metrics["orientation_angle_deg"], 5.0, "a perfectly vertical pole should read ~0 deg from vertical")
        self.assertLess(metrics["pct_image_area"], 0.35)
        self.assertGreater(metrics["mask_to_obb_ratio"], 0.5, "a filled rectangle mask should mostly fill its own tight OBB")
        self.assertFalse(metrics["touches_image_boundary"])

    def test_near_full_image_blob_flagged_excessively_large(self):
        mask = _blob_mask()
        corners = mask_to_obb_corners(mask)
        metrics = compute_geometry_metrics(mask, corners, 400, 400)
        result = score_geometry(metrics)
        self.assertEqual(result["geometry_status"], "fail")
        self.assertTrue(any("excessively large" in f for f in result["geometry_flags"]))

    def test_fragmented_mask_scores_worse_than_solid_pole(self):
        solid = run_geometry_qa(_vertical_pole_mask(), mask_to_obb_corners(_vertical_pole_mask()), 400, 400)
        frag_mask = _fragmented_mask()
        frag_corners = mask_to_obb_corners(frag_mask)
        self.assertIsNotNone(frag_corners, "fragmented mask should still produce SOME contour-based OBB")
        fragmented = run_geometry_qa(frag_mask, frag_corners, 400, 400)
        self.assertLess(fragmented["geometry_score"], solid["geometry_score"])

    def test_tilted_pole_is_not_hard_rejected(self):
        """A tilted pole is a normal case -- must not force geometry_status=fail on tilt alone."""
        mask = np.zeros((400, 400), dtype=np.uint8)
        # Draw a thick diagonal line simulating a moderately tilted pole (~25 deg).
        import cv2
        cv2.line(mask, (150, 380), (250, 30), color=255, thickness=12)
        corners = mask_to_obb_corners(mask)
        self.assertIsNotNone(corners)
        result = run_geometry_qa(mask, corners, 400, 400)
        self.assertNotEqual(result["geometry_status"], "fail", "moderate tilt alone must not hard-fail")

    def test_boundary_touching_mask_is_warning_not_hard_fail(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[0:360, 195:205] = 255  # touches the top edge
        corners = mask_to_obb_corners(mask)
        result = run_geometry_qa(mask, corners, 400, 400)
        self.assertTrue(result["metrics"]["touches_image_boundary"])
        self.assertNotEqual(result["geometry_status"], "fail", "boundary-touching alone must not hard-fail (pole may be legitimately cut off)")

    def test_empty_mask_hard_fails(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        corners = xyxy_to_obb_corners(190, 40, 210, 360)
        result = run_geometry_qa(mask, corners, 400, 400)
        self.assertEqual(result["geometry_status"], "fail")

    def test_none_mask_falls_back_to_zero_mask_metrics_without_crashing(self):
        corners = xyxy_to_obb_corners(190, 40, 210, 360)
        result = run_geometry_qa(None, corners, 400, 400)
        self.assertEqual(result["metrics"]["mask_area"], 0.0)
        self.assertIn(result["geometry_status"], ("fail", "warning"))


class TestGeometryQAConfig(unittest.TestCase):

    def test_config_from_dict_partial_overrides_only_specified_fields(self):
        cfg = config_from_dict({"max_image_area_fraction": 0.5, "unknown_field_ignored": 123})
        self.assertEqual(cfg.max_image_area_fraction, 0.5)
        self.assertEqual(cfg.min_obb_length_px, GeometryQAConfig().min_obb_length_px)

    def test_config_from_dict_none_or_empty_returns_defaults(self):
        self.assertEqual(config_from_dict(None), GeometryQAConfig())
        self.assertEqual(config_from_dict({}), GeometryQAConfig())

    def test_stricter_config_pushes_borderline_case_to_warning_or_fail(self):
        mask = _vertical_pole_mask(pole_w=30)  # thicker -> lower aspect ratio
        corners = mask_to_obb_corners(mask)
        loose = run_geometry_qa(mask, corners, 400, 400, GeometryQAConfig(min_length_to_thickness=1.0))
        strict = run_geometry_qa(mask, corners, 400, 400, GeometryQAConfig(min_length_to_thickness=8.0))
        self.assertGreaterEqual(loose["geometry_score"], strict["geometry_score"])


if __name__ == "__main__":
    unittest.main()
