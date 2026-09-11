"""
tests/test_config_verification.py - Verification Pipeline Config Keys
========================================================================

Verify the new SAM3/Qwen-backend config keys are present with safe defaults.
"""
import unittest
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


class TestVerificationConfigKeys(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.vp = self.cfg["verification_pipeline"]

    def test_qwen_backend_selection_keys_present(self):
        self.assertIn("backend", self.vp["qwen"])
        self.assertIn(self.vp["qwen"]["backend"], ("auto", "transformers", "ollama", "dashscope"))
        tcfg = self.vp["qwen"]["transformers"]
        self.assertIn("model_path", tcfg)
        self.assertIn("dtype", tcfg)
        self.assertIn("device_map", tcfg)
        self.assertEqual(tcfg["device_map"], "auto")

    def test_context_crop_expand_pct_in_spec_range(self):
        pct = self.vp["qwen"]["context_crop_expand_pct"]
        self.assertGreaterEqual(pct, 0.20)
        self.assertLessEqual(pct, 0.40)

    def test_sam3_backend_selection_keys_present(self):
        self.assertIn("backend", self.vp["sam3"])
        self.assertIn(self.vp["sam3"]["backend"], ("auto", "inprocess", "http"))
        self.assertIn("service_url", self.vp["sam3"])

    def test_sam3_candidate_proposal_off_by_default(self):
        # DINO alone proposes candidates in production; SAM3's own
        # detect_and_segment() candidate-proposal is redundant production
        # inference and must default to disabled.
        self.assertIn("enable_candidate_proposal", self.vp["sam3"])
        self.assertFalse(self.vp["sam3"]["enable_candidate_proposal"])

    def test_logging_section_present(self):
        self.assertIn("log_stage_timings", self.vp["logging"])


if __name__ == "__main__":
    unittest.main()
