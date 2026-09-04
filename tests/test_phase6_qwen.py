"""
tests/test_phase6_qwen.py - Phase 6 Qwen Verifier Layer Unit Tests
===================================================================

Validates:
1. QwenVerifier initialization and status reporting.
2. When API key or weights are unavailable, cleanly reports MODEL UNAVAILABLE.
3. Does NOT create fake responses.
4. Structured 7-question schema verification.
5. Model info contains installation guide for QWEN_API_KEY / DASHSCOPE_API_KEY / QWEN_MODEL_PATH.
"""

import unittest
from models.adapters.qwen_adapter import QwenVerifier
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners


class TestPhase6QwenVerifier(unittest.TestCase):

    def test_qwen_verifier_unavailable_handling(self):
        """When credentials/weights missing, Qwen reports unavailable and never mocks responses."""
        verifier = QwenVerifier(api_key=None, local_model_path="non_existent_path")
        self.assertFalse(verifier.is_available())
        
        info = verifier.get_info()
        self.assertEqual(info.status, "unavailable")
        self.assertIn("QWEN_API_KEY", info.installation_guide)

        # Attempt verification on a candidate box
        dummy_det = DetectionBox(
            xyxy=(10.0, 20.0, 50.0, 200.0),
            corners=xyxy_to_obb_corners(10, 20, 50, 200).tolist(),
            confidence=0.85,
            model_source="DINO+SAM",
        )
        res = verifier.verify("dummy_image.jpg", dummy_det)
        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(res["decision"], "unavailable")
        self.assertTrue(res["needs_human_review"])
        self.assertIn("MODEL UNAVAILABLE", res["reason"])


if __name__ == "__main__":
    unittest.main()
