"""
tests/test_sam3_http_adapter.py - SAM3 HTTP Client Adapter Tests
======================================================================

Part 2: SAM3 as an isolated HTTP service, for when SAM3's dependencies
can't coexist with the main app's torch_env_v2. SAM3HttpAdapter implements
the exact same BaseSegmenter interface as the in-process SAM3Adapter, so
backend/app.py can swap between them via one config value.
"""
import base64
import io
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from models.adapters.base import BaseSegmenter
from models.adapters.sam3_http_adapter import SAM3HttpAdapter


def _mask_b64(fill=1):
    buf = io.BytesIO()
    Image.fromarray((np.ones((20, 20), dtype=np.uint8) * fill * 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


class TestSAM3HttpAdapterParsesLiveResponses(unittest.TestCase):
    """Interface parity with the in-process SAM3Adapter: when the service
    returns a mask per detection, the client reconstructs it onto
    DetectionBox.mask so callers see identical behavior regardless of
    which SAM3 backend is active."""

    def test_detect_and_segment_reconstructs_mask_from_response(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:8801")
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "detections": [{
                "xyxy": [1.0, 1.0, 10.0, 15.0],
                "corners": [[1, 1], [10, 1], [10, 15], [1, 15]],
                "confidence": 0.81,
                "class_id": 0,
                "class_name": "utility_pole",
                "model_source": "SAM3",
                "needs_review": False,
                "review_reasons": [],
                "attributes": {},
                "mask_b64": _mask_b64(),
            }]
        }
        fake_response.raise_for_status = lambda: None

        with patch("models.adapters.sam3_http_adapter.requests.post", return_value=fake_response):
            img = np.zeros((20, 20, 3), dtype=np.uint8)
            detections = adapter.detect_and_segment(img)

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.81, places=2)
        self.assertIsNotNone(detections[0].mask)
        self.assertEqual(detections[0].mask.shape, (20, 20))

    def test_detect_and_segment_handles_detection_without_mask(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:8801")
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "detections": [{
                "xyxy": [1.0, 1.0, 10.0, 15.0], "corners": None, "confidence": 0.5,
                "class_id": 0, "class_name": "utility_pole", "model_source": "SAM3",
                "needs_review": False, "review_reasons": [], "attributes": {},
                "mask_b64": None,
            }]
        }
        fake_response.raise_for_status = lambda: None

        with patch("models.adapters.sam3_http_adapter.requests.post", return_value=fake_response):
            img = np.zeros((20, 20, 3), dtype=np.uint8)
            detections = adapter.detect_and_segment(img)

        self.assertEqual(len(detections), 1)
        self.assertIsNone(detections[0].mask)

    def test_segment_box_parses_score_from_response(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:8801")
        fake_response = MagicMock()
        fake_response.json.return_value = {"mask_b64": _mask_b64(), "score": 0.77}
        fake_response.raise_for_status = lambda: None

        with patch("models.adapters.sam3_http_adapter.requests.post", return_value=fake_response):
            img = np.zeros((20, 20, 3), dtype=np.uint8)
            mask = adapter.segment_box(img, (1.0, 1.0, 10.0, 10.0))

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (20, 20))


if __name__ == "__main__":
    unittest.main()
