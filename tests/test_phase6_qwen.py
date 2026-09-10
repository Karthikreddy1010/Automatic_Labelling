"""
tests/test_phase6_qwen.py - Phase 6 Qwen Verifier Layer Unit Tests
===================================================================

Validates:
1. QwenVerifier initialization and status reporting.
2. When no backend (Ollama, API key, or local weights) is available, cleanly
   reports MODEL UNAVAILABLE.
3. Does NOT create fake responses.
4. Structured 7-question schema verification.
5. Model info contains installation guide for Ollama / QWEN_API_KEY / DASHSCOPE_API_KEY / QWEN_MODEL_PATH.
6. A local Ollama server with a Qwen model pulled is auto-discovered and
   preferred over the cloud API (skipped if no such server is reachable).
"""

import unittest
from unittest.mock import patch
from models.adapters.qwen_adapter import QwenVerifier, _discover_ollama_qwen_model, _normalize_qwen_response
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners

# Deliberately unreachable so tests of the "nothing configured" path are not
# accidentally satisfied by an Ollama server that happens to be running on
# the machine running the tests.
UNREACHABLE_OLLAMA_URL = "http://127.0.0.1:1"


class TestPhase6QwenVerifier(unittest.TestCase):

    def test_qwen_verifier_unavailable_handling(self):
        """When no backend is configured/reachable, Qwen reports unavailable and never mocks responses."""
        verifier = QwenVerifier(api_key=None, local_model_path="non_existent_path", ollama_base_url=UNREACHABLE_OLLAMA_URL)
        self.assertFalse(verifier.is_available())

        info = verifier.get_info()
        self.assertEqual(info.status, "unavailable")
        self.assertIn("Ollama", info.installation_guide)
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

    def test_ollama_qwen_discovery_and_priority(self):
        """If a local Ollama server has a Qwen model pulled, it's discovered and preferred over a cloud API key."""
        model = _discover_ollama_qwen_model("http://localhost:11434")
        if not model:
            self.skipTest("No local Ollama server with a Qwen model reachable on this machine")

        verifier = QwenVerifier(api_key="dummy-cloud-key-should-not-be-used")
        self.assertTrue(verifier.is_available())
        self.assertEqual(verifier.ollama_model, model)

        info = verifier.get_info()
        self.assertEqual(info.status, "ready")
        self.assertIn("Ollama", info.name)
        self.assertEqual(info.device, "local_ollama")


class TestNormalizeQwenResponse(unittest.TestCase):
    """A malformed/partial Qwen JSON response must never crash the pipeline --
    unknown enum values and missing keys are normalized to safe defaults."""

    def test_well_formed_response_passes_through(self):
        norm = _normalize_qwen_response({
            "class": "electric_utility_pole", "material": "wood", "visibility": "mostly_visible",
            "orientation": "slightly_tilted", "semantic_confidence": 0.94, "decision": "accept",
            "reason": "Clear pole.",
        })
        self.assertEqual(norm["class"], "electric_utility_pole")
        self.assertEqual(norm["semantic_confidence"], 0.94)

    def test_missing_keys_fall_back_to_safe_defaults(self):
        norm = _normalize_qwen_response({})
        self.assertEqual(norm["class"], "uncertain")
        self.assertEqual(norm["material"], "unknown")
        self.assertEqual(norm["decision"], "review")
        self.assertEqual(norm["semantic_confidence"], 0.5)
        self.assertIn("No reason provided", norm["reason"])

    def test_unrecognized_enum_values_normalized_not_raised(self):
        norm = _normalize_qwen_response({
            "class": "definitely a pole trust me", "material": "unobtainium",
            "visibility": "somewhat visible i guess", "orientation": "diagonal-ish",
            "decision": "maybe",
        })
        self.assertEqual(norm["class"], "uncertain")
        self.assertEqual(norm["material"], "unknown")
        self.assertEqual(norm["visibility"], "mostly_visible")
        self.assertEqual(norm["orientation"], "vertical")
        self.assertEqual(norm["decision"], "review")

    def test_non_numeric_confidence_falls_back_without_raising(self):
        norm = _normalize_qwen_response({"semantic_confidence": "very confident"})
        self.assertEqual(norm["semantic_confidence"], 0.5)

    def test_out_of_range_confidence_is_clamped(self):
        self.assertEqual(_normalize_qwen_response({"semantic_confidence": 5.0})["semantic_confidence"], 1.0)
        self.assertEqual(_normalize_qwen_response({"semantic_confidence": -3.0})["semantic_confidence"], 0.0)

    def test_overlong_reason_is_truncated(self):
        norm = _normalize_qwen_response({"reason": "x" * 10000})
        self.assertLessEqual(len(norm["reason"]), 500)

    def test_wrong_type_class_field_does_not_raise(self):
        norm = _normalize_qwen_response({"class": 12345})
        self.assertEqual(norm["class"], "uncertain")


class TestVerifyHandlesMalformedBackendResponse(unittest.TestCase):
    """End-to-end through verify(): a backend call that returns non-dict JSON,
    or raises outright, must be handled without crashing the pipeline."""

    def _dummy_det(self):
        return DetectionBox(
            xyxy=(10.0, 20.0, 50.0, 200.0),
            corners=xyxy_to_obb_corners(10, 20, 50, 200).tolist(),
            confidence=0.85,
            model_source="DINO+SAM",
        )

    def test_backend_returning_non_dict_json_is_normalized_safely(self):
        verifier = QwenVerifier(api_key=None, ollama_base_url=UNREACHABLE_OLLAMA_URL)
        verifier._status = "ready"
        verifier.ollama_model = "qwen2.5vl:7b"  # pretend a model was discovered
        with patch.object(verifier, "_build_crops", return_value=("fakebase64tight", "fakebase64context")), \
             patch.object(verifier, "_verify_via_ollama", return_value=["not", "a", "dict"]):
            res = verifier.verify("dummy.jpg", self._dummy_det())
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["class"], "uncertain")
        self.assertTrue(res["needs_human_review"])

    def test_backend_raising_is_caught_as_error_status(self):
        verifier = QwenVerifier(api_key=None, ollama_base_url=UNREACHABLE_OLLAMA_URL)
        verifier._status = "ready"
        verifier.ollama_model = "qwen2.5vl:7b"
        with patch.object(verifier, "_build_crops", side_effect=RuntimeError("corrupt image")):
            res = verifier.verify("dummy.jpg", self._dummy_det())
        self.assertEqual(res["status"], "error")
        self.assertTrue(res["needs_human_review"])
        self.assertIsNone(res["semantic_confidence"])


class TestQwenExtendedSchema(unittest.TestCase):
    """Part 4/5: Qwen's normalized response carries the spec's richer field
    set (is_utility_pole/pole_type/occlusion/truncated/annotation_suitable)
    in addition to -- not instead of -- the existing class/decision schema
    that decision_engine.py already depends on."""

    def test_normalize_adds_new_fields_with_safe_defaults_on_empty_input(self):
        norm = _normalize_qwen_response({})
        # Existing fields (decision_engine.py contract) must still be present.
        for key in ("class", "material", "visibility", "orientation",
                    "semantic_confidence", "decision", "reason"):
            self.assertIn(key, norm)
        # New spec fields.
        self.assertIn("is_utility_pole", norm)
        self.assertIn("pole_type", norm)
        self.assertIn("occlusion", norm)
        self.assertIn("truncated", norm)
        self.assertIn("annotation_suitable", norm)
        self.assertFalse(norm["is_utility_pole"])
        self.assertFalse(norm["annotation_suitable"])
        self.assertIn(norm["occlusion"], ("none", "low", "medium", "high"))

    def test_normalize_maps_new_schema_response_correctly(self):
        raw = {
            "is_utility_pole": True,
            "pole_type": "utility_pole",
            "material": "wood",
            "visibility": "high",
            "occlusion": "low",
            "truncated": False,
            "annotation_suitable": True,
            "reason": "Vertical wooden pole carrying utility wires.",
            "confidence": 0.91,
        }
        norm = _normalize_qwen_response(raw)
        self.assertTrue(norm["is_utility_pole"])
        self.assertEqual(norm["pole_type"], "utility_pole")
        self.assertEqual(norm["occlusion"], "low")
        self.assertFalse(norm["truncated"])
        self.assertTrue(norm["annotation_suitable"])
        self.assertAlmostEqual(norm["semantic_confidence"], 0.91)  # "confidence" aliases semantic_confidence
        # Existing schema still derivable/compatible: a confident is_utility_pole
        # candidate should not be silently dropped from the old `class` contract.
        self.assertEqual(norm["class"], "electric_utility_pole")
        self.assertEqual(norm["decision"], "accept")

    def test_normalize_uncertain_new_schema_example_from_spec(self):
        raw = {
            "is_utility_pole": False,
            "pole_type": "uncertain",
            "material": "unknown",
            "visibility": "low",
            "occlusion": "high",
            "truncated": True,
            "annotation_suitable": False,
            "reason": "Candidate is mostly hidden by vegetation.",
            "confidence": 0.42,
        }
        norm = _normalize_qwen_response(raw)
        self.assertFalse(norm["is_utility_pole"])
        self.assertFalse(norm["annotation_suitable"])
        self.assertEqual(norm["decision"], "reject")

    def test_build_crops_returns_two_distinct_images(self):
        import numpy as np

        verifier = QwenVerifier(ollama_base_url="http://127.0.0.1:1")  # unreachable, fine for this test
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :100] = 255  # left half white, right half black -- crops will differ visibly
        det = DetectionBox(xyxy=(80.0, 50.0, 120.0, 150.0), corners=None, confidence=0.9)

        tight_b64, context_b64 = verifier._build_crops(img, det)
        self.assertIsInstance(tight_b64, str)
        self.assertIsInstance(context_b64, str)
        self.assertGreater(len(tight_b64), 0)
        self.assertGreater(len(context_b64), 0)
        # Context crop (expanded bbox) must decode to a larger image than the tight crop.
        import base64, io
        from PIL import Image
        tight_img = Image.open(io.BytesIO(base64.b64decode(tight_b64)))
        context_img = Image.open(io.BytesIO(base64.b64decode(context_b64)))
        self.assertGreaterEqual(context_img.size[0] * context_img.size[1],
                                 tight_img.size[0] * tight_img.size[1])


if __name__ == "__main__":
    unittest.main()
