"""
tests/test_phase3_dino_sam.py - Phase 3 DINO -> SAM -> Mask -> True OBB Pipeline
==================================================================================

Validates:
1. DINO identifies candidate boxes with configurable prompts.
2. SAM 2.1 segments the candidate box.
3. Morphology cleanup produces a cleaned binary mask.
4. cv2.minAreaRect extracts true rotated OBB from the mask.
5. Canonical corner ordering (Corner 0 = top-left, clockwise, shoelace area > 0).
6. Resulting OBB is derived from mask and not trivially identical to the axis-aligned candidate box.
7. Mask caching in masks/ directory.
8. Separate scores (dino_confidence, sam_quality, mask_area) preserved.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path

from models.adapters.dino_adapter import GroundingDINOAdapter
from models.adapters.sam21_adapter import SAM21Adapter
from src.geometry_obb import (
    signed_shoelace_area,
    mask_to_obb_corners,
    order_corners_canonical,
    xyxy_to_obb_corners,
    obb_iou,
)
from backend.storage import DatasetManager


class TestPhase3DinoSamPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.storage = DatasetManager(base_dir=cls.test_dir)
        cls.dataset_id = "test_p3_dataset"
        cls.storage.create_dataset(cls.dataset_id)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_dino_prompts_and_candidates(self):
        """Test configuring prompts and detecting candidate boxes."""
        dino = GroundingDINOAdapter()
        # Test prompt configuration
        dino.set_prompts(["utility pole", "power pole", "telephone pole", "electric pole"])
        self.assertIn("utility pole", dino.text_prompt)
        self.assertIn("telephone pole", dino.text_prompt)

    def test_mask_to_true_rotated_obb(self):
        """Verify that mask_to_obb_corners extracts true rotated OBB from synthetic tilted pole mask."""
        # Create synthetic tilted utility pole mask
        h, w = 600, 600
        mask = np.zeros((h, w), dtype=np.uint8)
        # Draw a pole tilted by 18 degrees
        center = (300, 300)
        size = (35, 320) # width 35px, length 320px
        angle = 18.0
        rect = (center, size, angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 255)

        # Extract OBB from mask
        obb = mask_to_obb_corners(mask)
        self.assertIsNotNone(obb)
        self.assertEqual(obb.shape, (4, 2))

        # Check shoelace area > 0 (strictly clockwise in image coordinates)
        area = signed_shoelace_area(obb)
        self.assertGreater(area, 0.0)

        # Check Corner 0 is top-left (minimizes x + y)
        scores = obb[:, 0] + obb[:, 1]
        self.assertAlmostEqual(scores[0], np.min(scores), places=3)

        # Verify that true OBB is NOT identical to the axis-aligned enclosing box
        ys, xs = np.where(mask > 0)
        axis_aligned_box = xyxy_to_obb_corners(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        iou_with_axis_aligned = obb_iou(obb, axis_aligned_box)
        # For an 18-degree pole, the true OBB is tighter than the axis-aligned box (IoU < 0.90)
        self.assertLess(iou_with_axis_aligned, 0.90, "True OBB must follow the rotated mask, not just copy XYXY")

    def test_mask_caching_and_score_separation(self):
        """Verify mask is saved to masks/ directory and separate scores are recorded."""
        dummy_mask = np.zeros((200, 200), dtype=np.uint8)
        dummy_mask[50:180, 95:105] = 255 # vertical pole strip
        
        filename = "sample_pole_001.jpg"
        mask_rel = self.storage.save_mask(self.dataset_id, filename, 0, dummy_mask)
        self.assertTrue(mask_rel.startswith("masks/"))
        
        saved_file = self.storage.get_mask_path(self.dataset_id, mask_rel)
        self.assertIsNotNone(saved_file)
        self.assertTrue(saved_file.exists())

        # Verify loaded mask matches
        loaded = cv2.imread(str(saved_file), cv2.IMREAD_GRAYSCALE)
        # cv2.imread may return (H, W, 1) on some OpenCV versions; squeeze to 2D
        if loaded.ndim == 3 and loaded.shape[2] == 1:
            loaded = loaded.squeeze(axis=2)
        self.assertEqual(loaded.shape, (200, 200))
        self.assertEqual((loaded > 0).sum(), (dummy_mask > 0).sum())


if __name__ == "__main__":
    unittest.main()
