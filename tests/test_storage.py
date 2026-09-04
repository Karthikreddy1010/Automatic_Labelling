"""
tests/test_storage.py - Unit Tests for Dataset Storage & Differential Learning
==============================================================================

Validates:
- Dataset creation and metadata integrity
- Raw AI predictions are preserved in predictions/ and NEVER overwritten by human edits
- Verified annotations are saved in annotations/ (.json and .txt)
- Human edits trigger differential learning records in history/
- YOLO-OBB export validates [0.0, 1.0] range and positive shoelace area
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path

from backend.storage import DatasetManager
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners


class TestStorageAndDifferential(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = DatasetManager(base_dir=self.temp_dir)
        self.ds_id = "test_pole_dataset"
        self.mgr.create_dataset(self.ds_id, name="Test Pole Dataset")

        # Create a synthetic image in the dataset
        img_dir = Path(self.temp_dir) / self.ds_id / "images"
        import cv2
        dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
        self.img_name = "pole_001.jpg"
        cv2.imwrite(str(img_dir / self.img_name), dummy_img)
        self.mgr.import_images(self.ds_id, [img_dir / self.img_name])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_raw_prediction_preservation_and_human_differential(self):
        """Human edits create a differential record in history/ and leave predictions/ untouched."""
        # 1. Simulate AI detector producing 2 candidate boxes
        ai_box1 = DetectionBox(
            xyxy=(50.0, 50.0, 100.0, 300.0),
            corners=xyxy_to_obb_corners(50, 50, 100, 300).tolist(),
            confidence=0.82,
            model_source="YOLO",
        )
        ai_box2 = DetectionBox(
            xyxy=(200.0, 60.0, 240.0, 320.0),
            corners=xyxy_to_obb_corners(200, 60, 240, 320).tolist(),
            confidence=0.48,
            model_source="DINO",
            needs_review=True,
        )
        self.mgr.save_predictions(self.ds_id, self.img_name, [ai_box1, ai_box2])

        # Verify predictions/ saved
        preds = self.mgr.get_predictions(self.ds_id, self.img_name)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds["boxes"]), 2)

        # 2. Human reviewer verifies:
        # - Accepts box1 with slight adjustment
        # - Deletes box2 (false positive)
        # - Adds a new box3 (missed pole)
        adjusted_box1 = DetectionBox(
            xyxy=(52.0, 48.0, 98.0, 302.0),
            corners=xyxy_to_obb_corners(52, 48, 98, 302).tolist(),
            confidence=1.0,
            model_source="HUMAN",
        )
        new_box3 = DetectionBox(
            xyxy=(300.0, 100.0, 340.0, 350.0),
            corners=xyxy_to_obb_corners(300, 100, 340, 350).tolist(),
            confidence=1.0,
            model_source="HUMAN",
        )
        human_boxes = [adjusted_box1, new_box3]

        self.mgr.save_annotation(
            self.ds_id, self.img_name, human_boxes, img_width=400, img_height=400, is_human=True
        )

        # 3. CRITICAL: Raw predictions MUST REMAIN EXACTLY AS THEY WERE
        preserved_preds = self.mgr.get_predictions(self.ds_id, self.img_name)
        self.assertEqual(len(preserved_preds["boxes"]), 2, "predictions/ must NEVER be overwritten")
        self.assertEqual(preserved_preds["boxes"][0]["confidence"], 0.82)
        self.assertEqual(preserved_preds["boxes"][1]["model_source"], "DINO")

        # 4. Verified annotations must reflect human edits
        ann = self.mgr.get_annotation(self.ds_id, self.img_name)
        self.assertEqual(len(ann["boxes"]), 2)
        self.assertEqual(ann["verified_by"], "human")

        # 5. Check differential history record
        history = self.mgr.get_history(self.ds_id, self.img_name)
        self.assertEqual(len(history), 1)
        record = history[0]
        self.assertEqual(record["ai_box_count"], 2)
        self.assertEqual(record["human_box_count"], 2)
        self.assertEqual(record["added_count"], 1)  # new_box3
        self.assertEqual(record["deleted_count"], 1)  # deleted ai_box2
        self.assertEqual(len(record["modifications"]), 1)  # adjusted_box1

        # 6. Verify image status updated to 'human_corrected' and difficulty score computed
        meta = self.mgr.get_dataset(self.ds_id)
        self.assertEqual(meta["images"][self.img_name]["status"], "human_corrected")
        self.assertEqual(meta["verified_count"], 1)
        self.assertGreater(meta["images"][self.img_name]["difficulty_score"], 0.50)
        self.assertIn("difficulty", record)
        self.assertGreaterEqual(record["difficulty"]["score"], 0.50)
        self.assertIn("false_positive_ai_prediction", record["difficulty"]["reasons"])
        self.assertIn("missed_pole", record["difficulty"]["reasons"])

    def test_yolo_obb_export(self):
        """Export generates data.yaml, images, and labels with valid OBB lines."""
        box = DetectionBox(
            xyxy=(40.0, 40.0, 120.0, 360.0),
            corners=xyxy_to_obb_corners(40, 40, 120, 360).tolist(),
            confidence=1.0,
            model_source="HUMAN",
        )
        self.mgr.save_annotation(self.ds_id, self.img_name, [box], img_width=400, img_height=400)

        export_res = self.mgr.export_dataset(self.ds_id)
        self.assertEqual(export_res["exported_images"], 1)
        self.assertEqual(len(export_res["validation_errors"]), 0)

        export_path = Path(export_res["export_path"])
        self.assertTrue((export_path / "data.yaml").exists())

    def test_human_accepted_and_rejected_workflow(self):
        """Test accepting without edits and rejecting candidates."""
        # 1. Image with AI box accepted without changes
        box = DetectionBox(
            xyxy=(50.0, 50.0, 100.0, 300.0),
            corners=xyxy_to_obb_corners(50, 50, 100, 300).tolist(),
            confidence=0.88,
            model_source="DINO+SAM",
        )
        self.mgr.save_predictions(self.ds_id, self.img_name, [box])
        
        # Human accepts without changes
        self.mgr.save_annotation(
            self.ds_id, self.img_name, [box], img_width=400, img_height=400, is_human=True, action="accepted"
        )
        meta = self.mgr.get_dataset(self.ds_id)
        self.assertEqual(meta["images"][self.img_name]["status"], "accepted")
        self.assertEqual(meta["accepted_count"], 1)

        # Raw prediction still untouched
        preds = self.mgr.get_predictions(self.ds_id, self.img_name)
        self.assertEqual(preds["boxes"][0]["confidence"], 0.88)

        # 2. Reject annotation
        self.mgr.save_annotation(
            self.ds_id, self.img_name, [], img_width=400, img_height=400, is_human=True, action="rejected"
        )
        meta = self.mgr.get_dataset(self.ds_id)
        self.assertEqual(meta["images"][self.img_name]["status"], "rejected")
        self.assertEqual(meta["rejected_count"], 1)
        # Predictions still untouched after rejection
        preds2 = self.mgr.get_predictions(self.ds_id, self.img_name)
        self.assertEqual(len(preds2["boxes"]), 1)


if __name__ == "__main__":
    unittest.main()
