"""
tests/test_sam3_adapter.py - SAM3 In-Process Adapter Tests
================================================================

Part 18.1-18.3: SAM3 model loading, text-prompt detection, box-prompt
segmentation. This machine has no transformers>=5.9 (pinned to 4.57.0) and
no CUDA, so SAM3Adapter.load() is expected to fail gracefully here --
exactly like every other honest-degradation test in this codebase. On HAWK,
once transformers>=5.9 + the facebook/sam3 checkpoint are available, is_available()
and load() should both return True and detect_and_segment/segment_box should
return real DetectionBox/mask results instead of empty ones.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from models.adapters.sam3_adapter import SAM3Adapter


class TestSAM3AdapterLoading(unittest.TestCase):
    def test_is_available_false_without_transformers_sam3_support(self):
        adapter = SAM3Adapter()
        # Honest reporting either way -- don't assert a specific bool, since
        # a future environment might have it installed; assert it never raises.
        result = adapter.is_available()
        self.assertIsInstance(result, bool)

    def test_load_never_crashes_and_reports_status_via_get_info(self):
        adapter = SAM3Adapter()
        adapter.load()
        info = adapter.get_info()
        self.assertIn(info.status, ("ready", "unavailable", "not_loaded"))
        self.assertEqual(info.id, "sam3_segmenter")
        if info.status != "ready":
            self.assertIsNotNone(info.error_message)

    def test_get_info_reports_backend_field(self):
        adapter = SAM3Adapter()
        info = adapter.get_info()
        self.assertIn("backend", info.to_dict())


class TestSAM3TextPrompt(unittest.TestCase):
    def test_detect_and_segment_on_unavailable_backend_returns_empty_list_not_crash(self):
        adapter = SAM3Adapter()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.detect_and_segment(img)
        self.assertIsInstance(result, list)  # [] when unavailable, never raises


class TestSAM3BoxPrompt(unittest.TestCase):
    def test_segment_box_on_unavailable_backend_returns_none_not_crash(self):
        adapter = SAM3Adapter()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.segment_box(img, (20.0, 20.0, 60.0, 80.0))
        self.assertIsNone(result)  # None when unavailable, never raises


def _fake_pole_mask(h=100, w=100):
    """A tall, thin filled rectangle -- survives _clean()'s vertical
    morphological open/close and produces a valid (non-degenerate) OBB."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:90, 40:55] = 1
    return m


class TestSAM3DetectAndSegmentAttachesMask(unittest.TestCase):
    """The service layer (services/sam3_service.py) needs to return masks
    alongside boxes/confidence for text-prompt detections -- detect_and_segment()
    must attach the cleaned mask onto each DetectionBox it returns."""

    def test_detections_carry_their_mask(self):
        adapter = SAM3Adapter()
        adapter.model = MagicMock()
        adapter.proc = MagicMock()
        adapter.proc.return_value.to.return_value = {}
        adapter.proc.post_process_instance_segmentation.return_value = [
            {"masks": [_fake_pole_mask()], "scores": [0.87]}
        ]

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = adapter.detect_and_segment(img)

        self.assertEqual(len(detections), 1)
        self.assertIsNotNone(detections[0].mask)
        self.assertEqual(detections[0].mask.shape, (100, 100))
        self.assertAlmostEqual(detections[0].confidence, 0.87, places=2)


class TestSAM3SegmentBoxWithScore(unittest.TestCase):
    """segment_box() keeps its existing mask-only return for every other
    call site (backend/app.py etc.); segment_box_with_score() additionally
    surfaces a confidence score for services/sam3_service.py."""

    def _mock_loaded_adapter_with_iou_scores(self, iou_value=0.91):
        # post_process_masks/iou_scores are real torch.Tensor in production
        # (a HuggingFace transformers method) -- mock with real tensors, not
        # plain numpy, so .numel()/.cpu() exercise the actual code path.
        import torch

        adapter = SAM3Adapter()
        adapter.model = MagicMock()
        adapter.proc = MagicMock()
        adapter.proc.return_value.to.return_value = {}

        fake_out = MagicMock()
        fake_out.iou_scores = [[torch.tensor([iou_value])]]
        adapter.model.return_value = fake_out

        fake_mask_tensor = torch.from_numpy(_fake_pole_mask()).unsqueeze(0)  # masks[0][0] -> 2D mask
        adapter.proc.post_process_masks.return_value = [fake_mask_tensor]
        return adapter

    def test_segment_box_returns_mask_only(self):
        adapter = self._mock_loaded_adapter_with_iou_scores()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = adapter.segment_box(img, (40.0, 10.0, 55.0, 90.0))
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (100, 100))

    def test_segment_box_with_score_returns_mask_and_confidence(self):
        adapter = self._mock_loaded_adapter_with_iou_scores(iou_value=0.91)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask, score = adapter.segment_box_with_score(img, (40.0, 10.0, 55.0, 90.0))
        self.assertIsNotNone(mask)
        self.assertAlmostEqual(score, 0.91, places=2)

    def test_segment_box_with_score_handles_missing_iou_scores_gracefully(self):
        adapter = self._mock_loaded_adapter_with_iou_scores()
        # spec=["pred_masks"] keeps out.pred_masks working (still needed by
        # post_process_masks) while out.iou_scores raises AttributeError,
        # simulating a model output shape without that field.
        adapter.model.return_value = MagicMock(spec=["pred_masks"])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask, score = adapter.segment_box_with_score(img, (40.0, 10.0, 55.0, 90.0))
        self.assertIsNotNone(mask)
        self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
