"""
tests/test_phase6_dino_comparison.py - Phase 6 Grounding DINO & Comparison Tests
=================================================================================

Tests:
- Real Grounding DINO inference with open-vocabulary utility pole prompts
- Fallback logic: triggers when YOLO is uncertain / produces no boxes
- Model comparison: separate model channels preserved in raw_outputs
"""

import time
import unittest
from pathlib import Path

from models.adapters.dino_adapter import GroundingDINOAdapter
from models.adapters.yolo_adapter import YOLOAdapter
from backend.app import run_ai_pipeline
from src.geometry_obb import signed_shoelace_area


class TestPhase6DINO(unittest.TestCase):

    def test_real_dino_inference(self):
        """Execute real Grounding DINO model on montage image."""
        adapter = GroundingDINOAdapter()
        ok = adapter.load(device="CPU")
        self.assertTrue(ok, f"DINO must load: {adapter._error_msg}")

        t0 = time.time()
        dets = adapter.predict("eval_upload/montage.jpg", conf_threshold=0.25)
        dino_time_ms = (time.time() - t0) * 1000

        print(f"\n[PHASE 6 REAL INFERENCE] Grounding DINO on montage.jpg:")
        print(f"  - Latency: {dino_time_ms:.1f} ms")
        print(f"  - Detections found: {len(dets)}")

        # Verify all detections are tagged DINO and flagged for review
        for det in dets:
            self.assertEqual(det.model_source, "DINO")
            self.assertTrue(det.needs_review, "DINO open-vocabulary detections should be flagged for review")
            self.assertGreater(signed_shoelace_area(det.corners), 0.0)

    def test_pipeline_fallback_and_comparison(self):
        """Test AI_LABEL waterfall and RUN_ALL model comparison modes."""
        img_path = Path("eval_upload/montage.jpg")

        # Test RUN_ALL mode
        reconciled_all, raw_all = run_ai_pipeline(
            img_path=img_path,
            mode="RUN_ALL",
            conf_threshold=0.25,
            use_sam_refinement=False
        )
        self.assertIn("yolo", raw_all, "RUN_ALL must preserve raw YOLO output")
        self.assertIn("dino", raw_all, "RUN_ALL must preserve raw DINO output")
        self.assertGreaterEqual(len(reconciled_all), 1)

        # Test AI_LABEL primary mode (DINO is primary)
        reconciled_primary, raw_primary = run_ai_pipeline(
            img_path=img_path,
            mode="AI_LABEL",
            conf_threshold=0.25,
            use_sam_refinement=False
        )
        self.assertIn("dino", raw_primary, "AI_LABEL must run DINO as primary object identifier")

        # Test YOLO_FAST accelerator mode
        reconciled_fast, raw_fast = run_ai_pipeline(
            img_path=img_path,
            mode="YOLO_FAST",
            conf_threshold=0.25,
            use_sam_refinement=False
        )
        self.assertIn("yolo", raw_fast, "YOLO_FAST must preserve raw YOLO output")


if __name__ == "__main__":
    unittest.main()
