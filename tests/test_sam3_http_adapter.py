"""
tests/test_sam3_http_adapter.py - SAM3 HTTP Client Adapter Tests
======================================================================

Part 2: SAM3 as an isolated HTTP service, for when SAM3's dependencies
can't coexist with the main app's torch_env_v2. SAM3HttpAdapter implements
the exact same BaseSegmenter interface as the in-process SAM3Adapter, so
backend/app.py can swap between them via one config value.
"""
import unittest
from models.adapters.base import BaseSegmenter
from models.adapters.sam3_http_adapter import SAM3HttpAdapter


class TestSAM3HttpAdapter(unittest.TestCase):
    def test_implements_base_segmenter_interface(self):
        self.assertTrue(issubclass(SAM3HttpAdapter, BaseSegmenter))

    def test_unreachable_service_reports_unavailable_not_crash(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")  # nothing listens on port 1
        self.assertFalse(adapter.is_available())
        info = adapter.get_info()
        self.assertEqual(info.status, "unavailable")
        self.assertIn("backend", info.to_dict())
        self.assertEqual(info.to_dict()["backend"], "sam3_http")

    def test_segment_box_on_unreachable_service_returns_none(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.segment_box(img, (10.0, 10.0, 50.0, 50.0))
        self.assertIsNone(result)

    def test_detect_and_segment_on_unreachable_service_returns_empty_list(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.detect_and_segment(img)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
