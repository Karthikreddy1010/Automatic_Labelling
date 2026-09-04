"""
tests/test_phase5_sam.py - Phase 5 SAM 2.1 Box-Prompted Segmentation & OBB Verification
========================================================================================

Tests:
- Real SAM 2.1 loading from local cache
- Real box-prompted segmentation on a utility pole region
- Morphology cleaning (vertical filter, wire suppression)
- Mask -> canonical 4-corner OBB conversion
- Signed shoelace area (> 0) and canonical corner 0 anchor
- Latency measurement
"""

import time
import unittest
import numpy as np
import cv2

from models.adapters.sam21_adapter import SAM21Adapter
from src.geometry_obb import signed_shoelace_area, is_clockwise_screen


class TestPhase5SAM(unittest.TestCase):

    def test_real_sam_box_segmentation(self):
        """Execute real SAM 2.1 box-prompted segmentation and OBB generation."""
        adapter = SAM21Adapter()
        ok = adapter.load(device="CPU")
        self.assertTrue(ok, f"SAM 2.1 must load successfully: {adapter._error_msg}")

        # Utility pole box coordinates in montage.jpg (from predictions.json pole_1006_0: [218.0, 1.4, 340.8, 625.5])
        pole_box = (218.0, 1.4, 340.8, 625.5)

        t0 = time.time()
        mask, corners = adapter.segment_and_generate_obb("eval_upload/montage.jpg", pole_box)
        seg_time_ms = (time.time() - t0) * 1000

        print(f"\n[PHASE 5 REAL SEGMENTATION] SAM 2.1 on montage.jpg:")
        print(f"  - Segmentation + OBB latency: {seg_time_ms:.1f} ms")
        print(f"  - Mask foreground pixels: {np.sum(mask > 0) if mask is not None else 0}")
        print(f"  - Generated canonical corners: {corners}")

        self.assertIsNotNone(mask, "Mask must not be None")
        self.assertGreater(np.sum(mask > 0), 100, "Mask must contain foreground pole pixels")
        self.assertIsNotNone(corners, "OBB corners must be generated from mask")
        self.assertEqual(corners.shape, (4, 2))

        # Winding check: shoelace area in screen coords must be strictly positive
        area = signed_shoelace_area(corners)
        self.assertGreater(area, 0.0, f"OBB shoelace area must be positive, got {area}")
        self.assertTrue(is_clockwise_screen(corners))

        # Canonical corner 0 anchor: must minimize (x+y)
        scores = corners[:, 0] + corners[:, 1]
        self.assertAlmostEqual(scores[0], np.min(scores), places=2)


if __name__ == "__main__":
    unittest.main()
