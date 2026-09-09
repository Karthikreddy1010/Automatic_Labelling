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



if __name__ == "__main__":
    unittest.main()
