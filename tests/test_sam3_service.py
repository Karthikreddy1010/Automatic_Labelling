"""
tests/test_sam3_service.py - SAM3 Isolated Microservice Endpoint Tests
============================================================================

Exercises services/sam3_service.py's FastAPI endpoints directly (via
TestClient), mocking SAM3Adapter's methods so no real SAM3 weights/GPU are
needed here. Verifies the HAWK-deployment contract: explicit 503 when not
ready (never a silent empty result, never a reload-per-request), and masks/
boxes/confidence all present in successful responses.
"""
import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from services.sam3_service import app, sam3_adapter
from models.adapters.base import DetectionBox, ModelInfo


def _b64_png(w=20, h=20):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestSAM3ServiceHealth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_reports_unavailable_honestly_when_not_loaded(self):
        with patch.object(sam3_adapter, "get_info", return_value=ModelInfo(
            id="sam3_segmenter", name="SAM 3", model_type="segmenter",
            status="unavailable", error_message="no transformers>=5.9",
        )):
            res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["available"])
        self.assertEqual(data["status"], "unavailable")
        self.assertIn("device", data)
        self.assertIn("model_id", data)

    def test_health_reports_ready_when_loaded(self):
        with patch.object(sam3_adapter, "get_info", return_value=ModelInfo(
            id="sam3_segmenter", name="SAM 3", model_type="segmenter", status="ready",
        )):
            res = self.client.get("/health")
        self.assertTrue(res.json()["available"])


class TestSAM3ServiceReadinessGuard(unittest.TestCase):
    """Never a silent empty result, never a reload-per-request -- an
    unavailable model must produce an explicit 503 before segment_box/
    detect_and_segment are even called."""

    def setUp(self):
        self.client = TestClient(app)

    def test_segment_box_returns_503_when_not_ready(self):
        with patch.object(sam3_adapter, "get_info", return_value=ModelInfo(
            id="sam3_segmenter", name="SAM 3", model_type="segmenter",
            status="unavailable", error_message="no GPU",
        )), patch.object(sam3_adapter, "segment_box_with_score") as mock_seg:
            res = self.client.post("/segment_box", json={
                "image_b64": _b64_png(), "box_xyxy": [1, 1, 10, 10],
            })
        self.assertEqual(res.status_code, 503)
        self.assertIn("no GPU", res.json()["detail"])
        mock_seg.assert_not_called()  # never attempts inference on an unready model

    def test_detect_and_segment_returns_503_when_not_ready(self):
        with patch.object(sam3_adapter, "get_info", return_value=ModelInfo(
            id="sam3_segmenter", name="SAM 3", model_type="segmenter",
            status="not_loaded",
        )), patch.object(sam3_adapter, "detect_and_segment") as mock_det:
            res = self.client.post("/detect_and_segment", json={"image_b64": _b64_png()})
        self.assertEqual(res.status_code, 503)
        mock_det.assert_not_called()


class TestSAM3ServiceSuccessResponses(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.ready_info = ModelInfo(
            id="sam3_segmenter", name="SAM 3", model_type="segmenter", status="ready",
        )

    def test_segment_box_returns_mask_and_score(self):
        fake_mask = np.ones((20, 20), dtype=np.uint8)
        with patch.object(sam3_adapter, "get_info", return_value=self.ready_info), \
             patch.object(sam3_adapter, "segment_box_with_score", return_value=(fake_mask, 0.93)):
            res = self.client.post("/segment_box", json={
                "image_b64": _b64_png(), "box_xyxy": [1, 1, 10, 10],
            })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNotNone(data["mask_b64"])
        self.assertAlmostEqual(data["score"], 0.93, places=2)

    def test_segment_box_no_mask_found_returns_422_not_silent_null(self):
        with patch.object(sam3_adapter, "get_info", return_value=self.ready_info), \
             patch.object(sam3_adapter, "segment_box_with_score", return_value=(None, None)):
            res = self.client.post("/segment_box", json={
                "image_b64": _b64_png(), "box_xyxy": [1, 1, 10, 10],
            })
        self.assertEqual(res.status_code, 422)

    def test_detect_and_segment_returns_boxes_confidence_and_masks(self):
        det = DetectionBox(
            xyxy=(1.0, 1.0, 10.0, 15.0), corners=[[1, 1], [10, 1], [10, 15], [1, 15]],
            confidence=0.81, class_name="utility_pole", model_source="SAM3",
            mask=np.ones((20, 20), dtype=np.uint8),
        )
        with patch.object(sam3_adapter, "get_info", return_value=self.ready_info), \
             patch.object(sam3_adapter, "detect_and_segment", return_value=[det]):
            res = self.client.post("/detect_and_segment", json={
                "image_b64": _b64_png(), "text_prompt": "utility pole",
            })
        self.assertEqual(res.status_code, 200)
        detections = res.json()["detections"]
        self.assertEqual(len(detections), 1)
        entry = detections[0]
        self.assertIn("xyxy", entry)
        self.assertAlmostEqual(entry["confidence"], 0.81, places=2)
        self.assertIsNotNone(entry["mask_b64"])

    def test_detect_and_segment_handles_detection_without_mask(self):
        det = DetectionBox(
            xyxy=(1.0, 1.0, 10.0, 15.0), corners=[[1, 1], [10, 1], [10, 15], [1, 15]],
            confidence=0.6, model_source="SAM3",
        )
        with patch.object(sam3_adapter, "get_info", return_value=self.ready_info), \
             patch.object(sam3_adapter, "detect_and_segment", return_value=[det]):
            res = self.client.post("/detect_and_segment", json={"image_b64": _b64_png()})
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["detections"][0]["mask_b64"])


if __name__ == "__main__":
    unittest.main()
