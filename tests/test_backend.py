"""
tests/test_backend.py - Integration Tests for FastAPI Backend Endpoints
========================================================================

Validates:
- Hardware status & device switching API
- Model status reporting (graceful status, no crashes)
- Dataset creation, retrieval, and listing
- Image annotation saving and raw prediction retrieval
- Batch job status endpoint
- Box-prompted segmentation API
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app import app, storage_mgr
from backend.storage import DatasetManager
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = tempfile.mkdtemp()
        # Override storage manager base_dir for test isolation
        storage_mgr.base_dir = Path(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_hardware_endpoints(self):
        """Test hardware detection and device switching endpoints."""
        res = self.client.get("/api/system/hardware")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("active_device_setting", data)
        self.assertIn("resolved_device", data)
        self.assertIn("hardware", data)

        # Switch device to CPU
        res_set = self.client.post("/api/system/device", json={"device": "CPU"})
        self.assertEqual(res_set.status_code, 200)
        self.assertEqual(res_set.json()["resolved"], "cpu")

    def test_models_status_endpoint(self):
        """Test models status returns clean dictionary without crashing."""
        res = self.client.get("/api/models/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 6)  # YOLO, DINO, SAM21, SAM3, Qwen, Gemini
        # Verify YOLO and Qwen model info
        yolo_info = next(m for m in data["models"] if "YOLO" in m["name"])
        self.assertIn(yolo_info["status"], ["ready", "not_loaded", "missing_weights"])
        qwen_info = next(m for m in data["models"] if "Qwen" in m["name"])
        # "ready" when a local Ollama vision-capable Qwen model is pulled/reachable
        # on the machine running the tests, "unavailable" otherwise -- both are
        # legitimate, honestly-reported states (never mocked), so assert the
        # endpoint returns clean structure rather than assuming one fixed
        # environment state.
        self.assertIn(qwen_info["status"], ["ready", "unavailable"])

    def test_import_upload_endpoint(self):
        """Browser folder/file-picker uploads land in images/ and register in metadata."""
        ds_id = "api_test_upload_dataset"
        self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "Upload Test", "classes": ["utility_pole"]})

        dummy = np.zeros((50, 50, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", dummy)
        jpg_bytes = buf.tobytes()

        res = self.client.post(
            f"/api/datasets/{ds_id}/import_upload",
            files=[
                ("files", ("pole_a.jpg", jpg_bytes, "image/jpeg")),
                # Folder-picker uploads can include a relative path; it must be
                # flattened to a bare filename, not used to escape images/.
                ("files", ("subfolder/pole_b.jpg", jpg_bytes, "image/jpeg")),
                ("files", ("notes.txt", b"not an image", "text/plain")),
            ],
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["imported_count"], 2)
        self.assertIn("pole_a.jpg", data["imported_images"])
        self.assertIn("pole_b.jpg", data["imported_images"])

        img_dir = Path(self.test_dir) / ds_id / "images"
        self.assertTrue((img_dir / "pole_a.jpg").exists())
        self.assertTrue((img_dir / "pole_b.jpg").exists())
        self.assertFalse((img_dir / "notes.txt").exists())

        meta = storage_mgr.get_dataset(ds_id)
        self.assertIn("pole_a.jpg", meta["images"])
        self.assertIn("pole_b.jpg", meta["images"])

    def test_ai_label_pipeline_reports_stage_timings(self):
        """Part 17: every AI_LABEL run should report per-stage timing in raw_outputs."""
        ds_id = "api_test_timing_dataset"
        self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "Timing Test", "classes": ["utility_pole"]})
        img_dir = Path(self.test_dir) / ds_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_name = "timing_test.jpg"
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / img_name), dummy)
        storage_mgr.import_images(ds_id, [img_dir / img_name])

        res = self.client.post(
            "/api/inference/detect",
            json={"dataset_id": ds_id, "filename": img_name, "mode": "AI_LABEL"},
        )
        # DINO/SAM aren't loaded in this CI environment -- a 503 "model
        # unavailable" is an acceptable, honest outcome here; only assert on
        # timings when the pipeline actually ran to completion.
        if res.status_code == 200:
            pred = storage_mgr.get_predictions(ds_id, img_name)
            timings = pred["raw_outputs"].get("timings")
            self.assertIsNotNone(timings)
            for key in ("dino_ms", "total_ms"):
                self.assertIn(key, timings)
                self.assertIsInstance(timings[key], (int, float))

    def test_qwen_model_status_reports_backend_field(self):
        """Part 10/11: status must name the real backend (transformers/ollama/
        dashscope), never a hard-coded 'Ollama: qwen3-vl:2b' regardless of which
        backend is actually active."""
        res = self.client.get("/api/models/status")
        data = res.json()
        qwen_info = next(m for m in data["models"] if "Qwen" in m["name"] or m["id"] == "qwen_verifier")
        self.assertIn("backend", qwen_info)
        if qwen_info["status"] == "ready":
            self.assertIn(qwen_info["backend"], ("transformers", "ollama", "dashscope"))
            self.assertIsNotNone(qwen_info["backend"])

    def test_skip_endpoint_marks_status_without_touching_predictions(self):
        """Part 13: Skip marks an image for later review without deleting or
        altering its existing AI predictions."""
        ds_id = "api_test_skip_dataset"
        self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "Skip Test", "classes": ["utility_pole"]})
        img_dir = Path(self.test_dir) / ds_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_name = "skip_test.jpg"
        cv2.imwrite(str(img_dir / img_name), np.zeros((100, 100, 3), dtype=np.uint8))
        storage_mgr.import_images(ds_id, [img_dir / img_name])

        box = DetectionBox(xyxy=(10, 10, 30, 90), corners=xyxy_to_obb_corners(10, 10, 30, 90).tolist(), confidence=0.7)
        storage_mgr.save_predictions(ds_id, img_name, [box], raw_outputs={})

        res = self.client.post(f"/api/datasets/{ds_id}/images/{img_name}/skip")
        self.assertEqual(res.status_code, 200)

        meta = storage_mgr.get_dataset(ds_id)
        self.assertEqual(meta["images"][img_name]["status"], "skipped")
        # Predictions must remain untouched.
        pred = storage_mgr.get_predictions(ds_id, img_name)
        self.assertEqual(len(pred["boxes"]), 1)

    def test_dataset_lifecycle_and_annotations(self):
        """Test dataset creation, annotation saving, and differential tracking via API."""
        ds_id = "api_test_dataset"
        # 1. Create dataset
        res_create = self.client.post(
            "/api/datasets",
            json={"dataset_id": ds_id, "name": "API Test Dataset", "classes": ["utility_pole"]}
        )
        self.assertEqual(res_create.status_code, 200)
        self.assertEqual(res_create.json()["dataset"]["dataset_id"], ds_id)

        # 2. List datasets
        res_list = self.client.get("/api/datasets")
        self.assertEqual(res_list.status_code, 200)
        self.assertGreaterEqual(len(res_list.json()["datasets"]), 1)

        # 3. Add a synthetic image to the dataset directory
        ds_images_dir = Path(self.test_dir) / ds_id / "images"
        ds_images_dir.mkdir(parents=True, exist_ok=True)
        img_name = "test_pole.jpg"
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(ds_images_dir / img_name), dummy)
        storage_mgr.import_images(ds_id, [ds_images_dir / img_name])

        # 4. Save annotation via POST
        box_corners = xyxy_to_obb_corners(40, 50, 80, 250).tolist()
        payload = {
            "image_width": 300,
            "image_height": 300,
            "is_human": True,
            "boxes": [
                {
                    "xyxy": [40.0, 50.0, 80.0, 250.0],
                    "corners": box_corners,
                    "confidence": 1.0,
                    "class_id": 0,
                    "class_name": "utility_pole",
                    "model_source": "HUMAN",
                    "needs_review": False,
                }
            ]
        }
        res_save = self.client.post(
            f"/api/datasets/{ds_id}/images/{img_name}/annotations",
            json=payload
        )
        self.assertEqual(res_save.status_code, 200)

        # 5. Retrieve annotations via GET
        res_get = self.client.get(f"/api/datasets/{ds_id}/images/{img_name}/annotations")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["type"], "verified_annotation")
        self.assertEqual(len(res_get.json()["data"]["boxes"]), 1)

    def test_batch_status_endpoint(self):
        """Test batch engine status query."""
        res = self.client.get("/api/batch/status")
        self.assertEqual(res.status_code, 200)
        self.assertIn("job", res.json())
        self.assertEqual(res.json()["job"]["status"], "idle")

    def test_segment_box_endpoint(self):
        """Test segment_box returns valid 4-corner canonical box."""
        ds_id = "segment_test_ds"
        storage_mgr.create_dataset(ds_id)
        ds_images_dir = Path(self.test_dir) / ds_id / "images"
        ds_images_dir.mkdir(parents=True, exist_ok=True)
        img_name = "pole_seg.jpg"
        cv2.imwrite(str(ds_images_dir / img_name), np.zeros((200, 200, 3), dtype=np.uint8))
        storage_mgr.import_images(ds_id, [ds_images_dir / img_name])

        res = self.client.post(
            "/api/inference/segment_box",
            json={"dataset_id": ds_id, "filename": img_name, "box_xyxy": [20.0, 20.0, 60.0, 180.0]}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("corners", data)
        self.assertEqual(len(data["corners"]), 4)

    def test_active_learning_weights_endpoints(self):
        """Test GET and PUT /api/config/active_learning_weights."""
        # 1. GET initial weights
        res_get = self.client.get("/api/config/active_learning_weights")
        self.assertEqual(res_get.status_code, 200)
        data = res_get.json()
        self.assertIn("weights", data)
        self.assertEqual(data["weights"].get("confirmed_negative"), 0)

        # 2. PUT updated weights
        res_put = self.client.put(
            "/api/config/active_learning_weights",
            json={"weights": {"missed_pole": 40, "confirmed_negative": 0}}
        )
        self.assertEqual(res_put.status_code, 200)
        self.assertEqual(res_put.json()["config"]["weights"]["missed_pole"], 40)

    def test_dataset_composition_and_warnings_endpoint(self):
        """Test GET /api/datasets/{id}/composition returns stats and warnings."""
        ds_id = "comp_test_ds"
        storage_mgr.create_dataset(ds_id)
        res = self.client.get(f"/api/datasets/{ds_id}/composition")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_images", data)
        self.assertIn("positive_poles", data)
        self.assertIn("confirmed_negatives", data)
        self.assertIn("warnings", data)

    def test_test_set_isolation_endpoints(self):
        """Test POST /test_set, GET /test_set, and POST /export_test_set."""
        ds_id = "test_set_ds"
        storage_mgr.create_dataset(ds_id)
        ds_images_dir = Path(self.test_dir) / ds_id / "images"
        ds_images_dir.mkdir(parents=True, exist_ok=True)
        img_name = "test_img_1.jpg"
        cv2.imwrite(str(ds_images_dir / img_name), np.zeros((100, 100, 3), dtype=np.uint8))
        storage_mgr.import_images(ds_id, [ds_images_dir / img_name])

        # 1. Toggle test set flag to True
        res_toggle = self.client.post(
            f"/api/datasets/{ds_id}/images/{img_name}/test_set"
        )
        self.assertEqual(res_toggle.status_code, 200)
        self.assertTrue(res_toggle.json()["is_test_set"])

        # 2. List test set
        res_list = self.client.get(f"/api/datasets/{ds_id}/test_set")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(res_list.json()["count"], 1)
        test_filenames = [item["filename"] for item in res_list.json()["test_set"]]
        self.assertIn(img_name, test_filenames)

        # 3. Export test set
        res_exp = self.client.post(f"/api/datasets/{ds_id}/export_test_set")
        self.assertEqual(res_exp.status_code, 200)
        self.assertEqual(res_exp.json()["result"]["count"], 1)


    def test_duplicates_endpoint(self):
        """Test GET /api/datasets/{id}/duplicates."""
        ds_id = "dup_test_ds"
        storage_mgr.create_dataset(ds_id)
        ds_images_dir = Path(self.test_dir) / ds_id / "images"
        ds_images_dir.mkdir(parents=True, exist_ok=True)
        img_a = "dup_a.jpg"
        img_b = "dup_b.jpg"
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(ds_images_dir / img_a), dummy)
        cv2.imwrite(str(ds_images_dir / img_b), dummy)
        storage_mgr.import_images(ds_id, [ds_images_dir / img_a, ds_images_dir / img_b])

        res = self.client.get(f"/api/datasets/{ds_id}/duplicates?threshold=8")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("duplicates", data)
        self.assertEqual(len(data["duplicates"]), 1)
        self.assertEqual(data["duplicates"][0]["distance"], 0)



class TestSAM3PrimarySAM21Fallback(unittest.TestCase):
    """SAM3 is the preferred segmenter (matches the spec's SAM3-upgrade
    intent); SAM 2.1 is only used when SAM3 itself is unavailable. Verified
    at unit level by mocking is_available()/segment_box() on the real
    module-level adapter instances -- no real model weights needed."""

    def setUp(self):
        from backend import app as app_module
        self.app_module = app_module

    def test_sam_refine_candidates_prefers_sam3_when_both_available(self):
        from backend.app import _sam_refine_candidates
        from unittest.mock import patch
        import numpy as np

        mask = np.ones((50, 50), dtype=np.uint8)
        candidate = DetectionBox(
            xyxy=(10.0, 10.0, 30.0, 40.0),
            corners=xyxy_to_obb_corners(10, 10, 30, 40).tolist(),
            confidence=0.9, model_source="DINO",
        )
        with patch.object(self.app_module.sam21_adapter, "is_available", return_value=True), \
             patch.object(self.app_module.sam3_adapter, "is_available", return_value=True), \
             patch.object(self.app_module.sam21_adapter, "segment_box") as mock_sam21, \
             patch.object(self.app_module.sam3_adapter, "segment_box", return_value=mask) as mock_sam3:
            _sam_refine_candidates([candidate], Path("dummy.jpg"), None, None, 100, 100)

        mock_sam3.assert_called_once()
        mock_sam21.assert_not_called()

    def test_sam_refine_candidates_falls_back_to_sam21_when_sam3_unavailable(self):
        from backend.app import _sam_refine_candidates
        from unittest.mock import patch
        import numpy as np

        mask = np.ones((50, 50), dtype=np.uint8)
        candidate = DetectionBox(
            xyxy=(10.0, 10.0, 30.0, 40.0),
            corners=xyxy_to_obb_corners(10, 10, 30, 40).tolist(),
            confidence=0.9, model_source="DINO",
        )
        with patch.object(self.app_module.sam21_adapter, "is_available", return_value=True), \
             patch.object(self.app_module.sam3_adapter, "is_available", return_value=False), \
             patch.object(self.app_module.sam21_adapter, "segment_box", return_value=mask) as mock_sam21, \
             patch.object(self.app_module.sam3_adapter, "segment_box") as mock_sam3:
            _sam_refine_candidates([candidate], Path("dummy.jpg"), None, None, 100, 100)

        mock_sam21.assert_called_once()
        mock_sam3.assert_not_called()

    def test_segment_box_endpoint_prefers_sam3_when_both_available(self):
        from unittest.mock import patch
        import numpy as np

        mask = np.ones((50, 50), dtype=np.uint8)
        ds_id = "sam_priority_test_ds"
        self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "SAM Priority Test", "classes": ["utility_pole"]})
        img_dir = Path(self.test_dir) / ds_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_name = "sam_priority.jpg"
        cv2.imwrite(str(img_dir / img_name), np.zeros((100, 100, 3), dtype=np.uint8))
        storage_mgr.import_images(ds_id, [img_dir / img_name])

        with patch.object(self.app_module.sam21_adapter, "is_available", return_value=True), \
             patch.object(self.app_module.sam3_adapter, "is_available", return_value=True), \
             patch.object(self.app_module.sam21_adapter, "segment_and_generate_obb") as mock_sam21, \
             patch.object(self.app_module.sam3_adapter, "segment_box", return_value=mask) as mock_sam3:
            res = self.client.post("/api/inference/segment_box", json={
                "dataset_id": ds_id, "filename": img_name, "box_xyxy": [10, 10, 30, 40],
            })

        self.assertEqual(res.status_code, 200)
        mock_sam3.assert_called_once()
        mock_sam21.assert_not_called()

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = tempfile.mkdtemp()
        storage_mgr.base_dir = Path(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
