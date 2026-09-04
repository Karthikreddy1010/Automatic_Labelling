"""
tests/test_adapters.py - Unit Tests for Adapters & Reconciliation
==================================================================

Validates:
- Hardware detection (CPU vs CUDA auto-resolution without fake claims)
- DetectionBox data class serialization
- YOLO adapter missing-weights handling & ModelInfo
- Reconciliation logic (YOLO+DINO consensus, YOLO-only, DINO-only, HUMAN override)
- Confidence categorization (GREEN, YELLOW, RED)
"""

import unittest
import numpy as np

from models.adapters.base import (
    DetectionBox,
    detect_hardware,
    ModelInfo,
)
from models.adapters.yolo_adapter import YOLOAdapter
from models.adapters.reconciliation import (
    reconcile_candidates,
    get_confidence_category,
    compute_box_iou,
)
from src.geometry_obb import xyxy_to_obb_corners


class TestAdaptersAndReconciliation(unittest.TestCase):

    def test_hardware_detection(self):
        """Hardware detection resolves accurately without fake claims."""
        dev_auto, info_auto = detect_hardware("AUTO")
        self.assertIn(dev_auto, ["cpu", "cuda"])
        self.assertEqual(info_auto.device, dev_auto)

        dev_cpu, info_cpu = detect_hardware("CPU")
        self.assertEqual(dev_cpu, "cpu")

        # If system has no CUDA, requesting CUDA should cleanly fall back to cpu
        dev_cuda, info_cuda = detect_hardware("CUDA")
        if not info_cuda.cuda_available:
            self.assertEqual(dev_cuda, "cpu")
        else:
            self.assertEqual(dev_cuda, "cuda")

    def test_detection_box_serialization(self):
        """DetectionBox round-trips through dictionary accurately."""
        corners = xyxy_to_obb_corners(10, 20, 30, 80).tolist()
        box = DetectionBox(
            xyxy=(10.0, 20.0, 30.0, 80.0),
            corners=corners,
            confidence=0.88,
            class_id=0,
            class_name="utility_pole",
            model_source="YOLO",
            needs_review=False,
            review_reasons=[],
            attributes={"tilt_deg": 2.5}
        )
        d = box.to_dict()
        self.assertEqual(d["confidence"], 0.88)
        self.assertEqual(d["model_source"], "YOLO")
        self.assertEqual(len(d["corners"]), 4)

        restored = DetectionBox.from_dict(d)
        self.assertEqual(restored.xyxy, (10.0, 20.0, 30.0, 80.0))
        self.assertEqual(restored.confidence, 0.88)
        self.assertEqual(restored.model_source, "YOLO")
        self.assertEqual(restored.attributes["tilt_deg"], 2.5)

    def test_yolo_adapter_missing_weights_handling(self):
        """YOLO adapter cleanly reports missing_weights without throwing exceptions."""
        adapter = YOLOAdapter(weights_path="non_existent_path.pt")
        self.assertFalse(adapter.is_available())
        info = adapter.get_info()
        self.assertEqual(info.status, "missing_weights")
        self.assertIsNotNone(info.installation_guide)

        # Attempting predict should return empty list gracefully
        dets = adapter.predict(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(dets, [])

    def test_reconciliation_consensus_boost(self):
        """Overlapping YOLO + DINO detections merge into YOLO+DINO with model_agreement tracked."""
        yolo_box = DetectionBox(
            xyxy=(100.0, 50.0, 140.0, 350.0),
            corners=xyxy_to_obb_corners(100, 50, 140, 350).tolist(),
            confidence=0.75,
            model_source="YOLO",
        )
        # Slightly shifted DINO box (high IoU)
        dino_box = DetectionBox(
            xyxy=(102.0, 48.0, 138.0, 352.0),
            corners=xyxy_to_obb_corners(102, 48, 138, 352).tolist(),
            confidence=0.70,
            model_source="DINO",
        )

        reconciled = reconcile_candidates(
            yolo_boxes=[yolo_box],
            dino_boxes=[dino_box],
            iou_threshold=0.5
        )

        self.assertEqual(len(reconciled), 1, "Should merge into 1 candidate")
        merged = reconciled[0]
        self.assertEqual(merged.model_source, "YOLO+DINO")
        self.assertEqual(merged.confidence, 0.75, "Confidence preserved without fake boost")
        self.assertTrue(merged.attributes["model_agreement"])
        self.assertEqual(merged.attributes["yolo_confidence"], 0.75)
        self.assertEqual(merged.attributes["dino_confidence"], 0.70)
        self.assertEqual(merged.attributes["category"], "GREEN")
        self.assertFalse(merged.needs_review)

    def test_reconciliation_dino_only_flagged_for_review(self):
        """DINO-only detection is tagged DINO and flagged for review."""
        dino_box = DetectionBox(
            xyxy=(300.0, 50.0, 330.0, 300.0),
            corners=xyxy_to_obb_corners(300, 50, 330, 300).tolist(),
            confidence=0.60,
            model_source="DINO",
        )
        reconciled = reconcile_candidates(yolo_boxes=[], dino_boxes=[dino_box])
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0].model_source, "DINO")
        self.assertTrue(reconciled[0].needs_review)
        self.assertEqual(reconciled[0].attributes["category"], "YELLOW")

    def test_reconciliation_human_override(self):
        """Human annotations act as ground truth anchors with 1.0 confidence."""
        yolo_box = DetectionBox(
            xyxy=(100.0, 50.0, 140.0, 350.0),
            corners=xyxy_to_obb_corners(100, 50, 140, 350).tolist(),
            confidence=0.65,
            model_source="YOLO",
            needs_review=True
        )
        human_box = DetectionBox(
            xyxy=(100.0, 50.0, 140.0, 350.0),
            corners=xyxy_to_obb_corners(100, 50, 140, 350).tolist(),
            confidence=1.0,
            model_source="HUMAN"
        )
        reconciled = reconcile_candidates(
            yolo_boxes=[yolo_box],
            dino_boxes=[],
            human_boxes=[human_box]
        )
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0].model_source, "HUMAN")
        self.assertEqual(reconciled[0].confidence, 1.0)
        self.assertFalse(reconciled[0].needs_review)
        self.assertEqual(reconciled[0].attributes["category"], "GREEN")


if __name__ == "__main__":
    unittest.main()
