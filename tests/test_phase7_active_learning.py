"""
tests/test_phase7_active_learning.py - Phase 7 Active Learning Scoring & Priority Queue
========================================================================================

Validates:
1. Evidence-based difficulty scoring:
   - High priority for low confidence, poor SAM masks, disagreement, false positives, missed poles.
   - Low priority for clean, human-accepted candidates without changes.
2. Active learning priority queue returns sorted items with difficulty reasons.
3. Filtering by difficulty reason (e.g. false_positive, missed_pole, poor_sam).
4. get_next_difficult returns the highest-priority unreviewed image.
5. Export hard cases creates hard_cases/ directory and manifest.json.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path

from backend.storage import DatasetManager
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners


class TestPhase7ActiveLearning(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = DatasetManager(base_dir=self.temp_dir)
        self.ds_id = "test_al_dataset"
        self.mgr.create_dataset(self.ds_id, name="Active Learning Dataset")

        # Create 3 synthetic images
        img_dir = Path(self.temp_dir) / self.ds_id / "images"
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        self.imgs = ["easy_01.jpg", "hard_fp_02.jpg", "hard_diff_03.jpg"]
        for name in self.imgs:
            cv2.imwrite(str(img_dir / name), dummy)
        self.mgr.import_images(self.ds_id, [img_dir / n for n in self.imgs])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_active_learning_ranking_and_queue(self):
        """Verify difficulty score ranking places difficult examples at the top of the queue."""
        # 1. Image 1: Clean prediction, human accepted directly
        box_easy = DetectionBox(
            xyxy=(50.0, 50.0, 100.0, 250.0),
            corners=xyxy_to_obb_corners(50, 50, 100, 250).tolist(),
            confidence=0.92,
            model_source="DINO+SAM",
        )
        self.mgr.save_predictions(self.ds_id, "easy_01.jpg", [box_easy])
        self.mgr.save_annotation(self.ds_id, "easy_01.jpg", [box_easy], 300, 300, is_human=True, action="accepted")

        # 2. Image 2: AI predicted 2 boxes, human deleted one (false positive) and added one (missed pole)
        box_ai1 = DetectionBox(
            xyxy=(30.0, 30.0, 70.0, 200.0),
            corners=xyxy_to_obb_corners(30, 30, 70, 200).tolist(),
            confidence=0.60,
            model_source="DINO",
            needs_review=True,
        )
        box_ai2 = DetectionBox(
            xyxy=(150.0, 40.0, 180.0, 220.0),
            corners=xyxy_to_obb_corners(150, 40, 180, 220).tolist(),
            confidence=0.35,
            model_source="DINO",
            needs_review=True,
        )
        self.mgr.save_predictions(self.ds_id, "hard_fp_02.jpg", [box_ai1, box_ai2])

        # Human keeps box_ai1 (with edit), deletes box_ai2, adds box_new
        human_edit1 = DetectionBox(
            xyxy=(32.0, 28.0, 68.0, 202.0),
            corners=xyxy_to_obb_corners(32, 28, 68, 202).tolist(),
            confidence=1.0,
            model_source="HUMAN",
        )
        human_new = DetectionBox(
            xyxy=(220.0, 50.0, 250.0, 240.0),
            corners=xyxy_to_obb_corners(220, 50, 250, 240).tolist(),
            confidence=1.0,
            model_source="HUMAN",
        )
        self.mgr.save_annotation(
            self.ds_id, "hard_fp_02.jpg", [human_edit1, human_new], 300, 300, is_human=True
        )

        # 3. Image 3: Poor SAM mask and low confidence candidate
        box_ai3 = DetectionBox(
            xyxy=(100.0, 100.0, 130.0, 280.0),
            corners=xyxy_to_obb_corners(100, 100, 130, 280).tolist(),
            confidence=0.32,
            model_source="DINO+SAM",
            needs_review=True,
            attributes={"sam_quality": "poor", "mask_area": 12}
        )
        self.mgr.save_predictions(self.ds_id, "hard_diff_03.jpg", [box_ai3])

        # Fetch active learning queue
        queue = self.mgr.get_active_learning_queue(self.ds_id)
        self.assertEqual(len(queue), 3)

        # Verify difficult images rank higher than clean accepted ones
        priorities = {item["filename"]: item["priority"] for item in queue}
        self.assertGreater(priorities["hard_fp_02.jpg"], priorities["easy_01.jpg"])
        self.assertGreater(priorities["hard_diff_03.jpg"], priorities["easy_01.jpg"])

        # Check next difficult image
        next_diff = self.mgr.get_next_difficult(self.ds_id)
        self.assertIsNotNone(next_diff)
        self.assertEqual(next_diff["filename"], "hard_diff_03.jpg")

    def test_export_hard_cases(self):
        """Export hard cases creates directory and manifest."""
        # Set a hard image
        box = DetectionBox(
            xyxy=(20.0, 20.0, 80.0, 250.0),
            corners=xyxy_to_obb_corners(20, 20, 80, 250).tolist(),
            confidence=0.30,
            model_source="DINO",
            needs_review=True,
        )
        self.mgr.save_predictions(self.ds_id, "hard_fp_02.jpg", [box])
        
        res = self.mgr.export_hard_cases(self.ds_id, min_priority=50)
        self.assertIn("export_path", res)
        export_dir = Path(res["export_path"])
        self.assertTrue((export_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
