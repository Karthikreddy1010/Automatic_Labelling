"""
tests/test_phase4_yolo.py - Phase 4 YOLO & YOLO-OBB Real Inference Verification
=================================================================================

Tests:
- Real YOLO-OBB loading from models/yolo11n-obb.pt
- Real inference execution on eval_upload/montage.jpg
- Canonical 4-corner verification on real detections
- Inference timing measurement
- Device verification (CPU)
"""

import time
import unittest
import numpy as np
import cv2

from models.adapters.yolo_adapter import YOLOAdapter
from src.geometry_obb import signed_shoelace_area, is_clockwise_screen


class TestPhase4YOLO(unittest.TestCase):

    def test_real_yolo_obb_inference(self):
        """Execute real YOLO-OBB model on sample montage image."""
        adapter = YOLOAdapter(weights_path="models/yolo11n-obb.pt")
        self.assertTrue(adapter.is_available(), "YOLO-OBB model must be available")

        # Load model on CPU
        ok = adapter.load(device="CPU")
        self.assertTrue(ok, "YOLO-OBB must load successfully")
        self.assertEqual(adapter.device, "cpu")

        # Run real inference on montage.jpg
        t0 = time.time()
        dets = adapter.predict("eval_upload/montage.jpg", conf_threshold=0.15)
        inference_time_ms = (time.time() - t0) * 1000

        print(f"\n[PHASE 4 REAL INFERENCE] YOLO-OBB on montage.jpg:")
        print(f"  - Inference latency: {inference_time_ms:.1f} ms")
        print(f"  - Detections found: {len(dets)}")

        self.assertGreater(len(dets), 0, "Should detect objects in montage image")

        # Verify geometric validity of all returned detections
        for i, det in enumerate(dets):
            self.assertEqual(det.model_source, "YOLO-OBB")
            self.assertIsNotNone(det.corners)
            self.assertEqual(len(det.corners), 4)

            # Winding test: shoelace area in screen coords must be strictly positive
            area = signed_shoelace_area(det.corners)
            self.assertGreater(area, 0.0, f"Detection {i} must have positive shoelace area (clockwise)")

            # Canonical top-left anchor test: Corner 0 must minimize (x+y)
            scores = [p[0] + p[1] for p in det.corners]
            self.assertAlmostEqual(scores[0], min(scores), places=3,
                                   msg=f"Detection {i} corner 0 must be canonical top-left anchor")


if __name__ == "__main__":
    unittest.main()
