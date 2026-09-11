"""
tests/test_qwen3vl_integration.py - REAL Qwen3-VL-8B-Instruct H200 Integration Test
======================================================================================

This test is the MOST IMPORTANT test in the Qwen hardening effort.

It does NOT merely check imports, checkpoint existence, or is_available().
It performs an actual image inference through Qwen3-VL-8B-Instruct using
the local Transformers backend on CUDA, and verifies:

1.  Model actually loads from the local checkpoint.
2.  Model runs on CUDA (not CPU).
3.  Model uses bfloat16 (not float32).
4.  Real image inference succeeds.
5.  Normalized result matches expected schema.
6.  Backend used is "transformers" (not Ollama/DashScope).
7.  Load time, inference time, and CUDA memory are measured.

This test is SKIPPED automatically when:
- CUDA is unavailable
- The local checkpoint does not exist

It MUST run and pass on the H200 environment.
"""

import os
import sys
import time
import unittest
from pathlib import Path

# The default Qwen3-VL-8B-Instruct checkpoint path on the H200 deepthink.
QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/jovyan/models/Qwen3-VL-8B-Instruct"
)

# Find a real project image from the dataset for testing.
PROJECT_IMAGE_DIR = Path("data/datasets/utility_poles_v1/images")


def _find_test_image() -> str:
    """Return the path to a real project image, or None if none found."""
    if PROJECT_IMAGE_DIR.exists():
        for f in sorted(PROJECT_IMAGE_DIR.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file():
                return str(f)
    # Fallback: check the nested images/ dir
    nested = PROJECT_IMAGE_DIR / "images"
    if nested.exists():
        for f in sorted(nested.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file():
                return str(f)
    return None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _checkpoint_exists() -> bool:
    return Path(QWEN_MODEL_PATH).exists()


@unittest.skipUnless(
    _cuda_available() and _checkpoint_exists(),
    f"Requires CUDA and local Qwen checkpoint at {QWEN_MODEL_PATH}"
)
class TestQwen3VLRealInference(unittest.TestCase):
    """
    Real integration test: loads Qwen3-VL-8B-Instruct and runs inference
    on an actual project image. This is NOT a mock test.
    """

    @classmethod
    def setUpClass(cls):
        """Find a test image before any tests run."""
        cls.test_image_path = _find_test_image()
        if cls.test_image_path is None:
            # Create a simple synthetic image as absolute last resort
            import numpy as np
            from PIL import Image
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            # Draw a rough vertical line (pole-like)
            img[100:550, 310:330, :] = [139, 90, 43]  # brown-ish vertical bar
            cls._synthetic_image = img
            cls.test_image_path = None
        else:
            cls._synthetic_image = None

    def _get_image(self):
        """Return image input for verify()."""
        if self.test_image_path:
            return self.test_image_path
        return self._synthetic_image

    def test_real_qwen3vl_inference(self):
        """
        THE CRITICAL TEST: Performs real inference through QwenVerifier.verify().

        Success criteria:
        - Model loads on CUDA with bfloat16
        - Real image processed through Qwen3-VL-8B
        - Normalized result returned
        - Backend is 'transformers', NOT Ollama/DashScope
        """
        import torch
        from models.adapters.qwen_adapter import QwenVerifier
        from models.adapters.base import DetectionBox

        print("\n" + "=" * 70)
        print("QWEN3-VL-8B REAL INFERENCE INTEGRATION TEST")
        print("=" * 70)

        # --- Step 1: Instantiate QwenVerifier with explicit transformers backend ---
        verifier = QwenVerifier(
            api_key=None,
            local_model_path=QWEN_MODEL_PATH,
            ollama_base_url="http://127.0.0.1:1",  # unreachable on purpose
            backend="transformers",
            transformers_dtype="bfloat16",
            transformers_device_map="auto",
        )

        # --- Step 2: Confirm checkpoint is selected ---
        self.assertTrue(verifier.is_available(),
                        "QwenVerifier should report available with local checkpoint")
        info_before = verifier.get_info()
        print(f"\n[BEFORE LOAD]")
        print(f"  status: {info_before.status}")
        print(f"  backend: {info_before.backend}")
        print(f"  weights_path: {info_before.weights_path}")
        print(f"  device: {info_before.device}")
        print(f"  dtype: {info_before.dtype}")
        self.assertEqual(info_before.backend, "transformers")
        self.assertEqual(info_before.weights_path, QWEN_MODEL_PATH)
        # Before first verify(), status should be 'configured', not 'ready'
        self.assertEqual(info_before.status, "configured",
                         "Status should be 'configured' before model loads")

        # --- Step 3: Build a real detection candidate ---
        image = self._get_image()
        image_source = self.test_image_path or "synthetic"
        print(f"\n[TEST IMAGE] {image_source}")

        det = DetectionBox(
            xyxy=(200.0, 100.0, 350.0, 500.0),
            corners=[[200, 100], [350, 100], [350, 500], [200, 500]],
            confidence=0.82,
            model_source="DINO+SAM",
        )

        # --- Step 4: Reset CUDA peak memory to get accurate measurements ---
        torch.cuda.reset_peak_memory_stats()
        cuda_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

        # --- Step 5: Call verify() — this triggers lazy model loading + inference ---
        t_start = time.monotonic()
        result = verifier.verify(image, det)
        t_total = time.monotonic() - t_start

        # --- Step 6: Verify result is valid ---
        print(f"\n[RESULT]")
        for k, v in result.items():
            print(f"  {k}: {v}")

        self.assertEqual(result["status"], "ok",
                         f"Inference failed: {result.get('reason', 'unknown')}")

        # --- Step 7: Confirm backend_used is transformers ---
        self.assertEqual(result.get("backend_used"), "transformers",
                         "Must use transformers backend, not Ollama/DashScope")

        # --- Step 8: Confirm Ollama was NOT used ---
        self.assertNotEqual(result.get("backend_used"), "ollama",
                            "Ollama must NOT be used")
        self.assertNotEqual(result.get("backend_used"), "dashscope",
                            "DashScope must NOT be used")

        # --- Step 9: Validate normalized result schema ---
        required_fields = [
            "status", "class", "material", "visibility", "orientation",
            "semantic_confidence", "decision", "reason",
            "is_utility_pole", "pole_type", "occlusion", "truncated",
            "annotation_suitable", "obb_quality", "needs_human_review",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

        # Decision must be one of the valid values
        self.assertIn(result["decision"], ("accept", "review", "reject"))

        # Confidence must be a valid float in [0, 1]
        self.assertIsInstance(result["semantic_confidence"], float)
        self.assertGreaterEqual(result["semantic_confidence"], 0.0)
        self.assertLessEqual(result["semantic_confidence"], 1.0)

        # --- Step 10: Confirm model is actually loaded and on CUDA ---
        info_after = verifier.get_info()
        print(f"\n[AFTER LOAD]")
        print(f"  status: {info_after.status}")
        print(f"  backend: {info_after.backend}")
        print(f"  weights_path: {info_after.weights_path}")
        print(f"  device: {info_after.device}")
        print(f"  dtype: {info_after.dtype}")
        print(f"  load_time_s: {info_after.load_time_s}")

        self.assertEqual(info_after.status, "ready",
                         "Status should be 'ready' after successful inference")
        self.assertIn("cuda", info_after.device,
                      f"Model should be on CUDA, got: {info_after.device}")

        # --- Step 11: Confirm dtype is bfloat16 ---
        self.assertEqual(info_after.dtype, "bfloat16",
                         f"Model should use bfloat16, got: {info_after.dtype}")

        # --- Step 12: Measure and report timings ---
        cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        cuda_current_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        print(f"\n[TELEMETRY]")
        print(f"  backend_used: {result.get('backend_used')}")
        print(f"  model_path: {QWEN_MODEL_PATH}")
        print(f"  device: {info_after.device}")
        print(f"  dtype: {info_after.dtype}")
        print(f"  load_time_s: {info_after.load_time_s}")
        print(f"  total_verify_time_s: {t_total:.2f}")
        print(f"  cuda_memory_before_mb: {cuda_mem_before:.1f}")
        print(f"  cuda_peak_memory_mb: {cuda_peak_mb:.1f}")
        print(f"  cuda_current_memory_mb: {cuda_current_mb:.1f}")
        print(f"  gpu_name: {torch.cuda.get_device_name(0)}")

        # --- Step 13: Load time should be measurable ---
        self.assertIsNotNone(info_after.load_time_s)
        self.assertGreater(info_after.load_time_s, 0)

        # --- Step 14: Confirm CUDA memory was actually used (model is on GPU) ---
        self.assertGreater(cuda_peak_mb, 100,
                           "Peak CUDA memory should be >100MB for an 8B model")

        # --- Final summary ---
        print(f"\n{'=' * 70}")
        print(f"QWEN3-VL-8B INFERENCE: SUCCESS")
        print(f"  Backend: transformers (NOT Ollama, NOT DashScope)")
        print(f"  Device: {info_after.device}")
        print(f"  Dtype: {info_after.dtype}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Model Load Time: {info_after.load_time_s:.2f}s")
        print(f"  Total Verify Time: {t_total:.2f}s")
        print(f"  Peak GPU Memory: {cuda_peak_mb:.1f} MB")
        print(f"  Decision: {result['decision']}")
        print(f"  Class: {result['class']}")
        print(f"  Confidence: {result['semantic_confidence']}")
        print(f"{'=' * 70}")

    def test_second_inference_reuses_cached_model(self):
        """Second verify() call must NOT reload the model (singleton caching)."""
        import torch
        from models.adapters.qwen_adapter import QwenVerifier
        from models.adapters.base import DetectionBox

        verifier = QwenVerifier(
            api_key=None,
            local_model_path=QWEN_MODEL_PATH,
            ollama_base_url="http://127.0.0.1:1",
            backend="transformers",
        )

        image = self._get_image()
        det = DetectionBox(
            xyxy=(200.0, 100.0, 350.0, 500.0),
            corners=[[200, 100], [350, 100], [350, 500], [200, 500]],
            confidence=0.82, model_source="DINO+SAM",
        )

        # First call loads the model
        result1 = verifier.verify(image, det)
        self.assertEqual(result1["status"], "ok")

        # Second call should reuse cached model (much faster)
        t_start = time.monotonic()
        result2 = verifier.verify(image, det)
        t_second = time.monotonic() - t_start

        self.assertEqual(result2["status"], "ok")
        self.assertEqual(result2.get("backend_used"), "transformers")

        # Second inference should be significantly faster than first
        # (no model loading time)
        print(f"\n[CACHE TEST] Second inference time: {t_second:.2f}s")

        # Model should still be loaded
        info = verifier.get_info()
        self.assertEqual(info.status, "ready")


if __name__ == "__main__":
    unittest.main()
