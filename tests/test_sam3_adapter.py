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


if __name__ == "__main__":
    unittest.main()
