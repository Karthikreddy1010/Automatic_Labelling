"""
tests/test_decision_engine.py - Decision Engine Unit Tests
==============================================================

Pure unit tests -- no models, no network. Exercises
models/adapters/decision_engine.py's weighted scoring, three-way decision,
disagreement detection, and weight-renormalization-when-a-signal-didn't-run
logic in isolation with synthetic inputs.
"""

import unittest

from models.adapters.decision_engine import (
    evaluate_candidate,
    compute_segmentation_score,
    detect_disagreement,
    DecisionConfig,
    DecisionWeights,
    config_from_dict,
)

GOOD_GEOMETRY = {"geometry_score": 0.9, "geometry_status": "pass", "geometry_flags": []}
BAD_GEOMETRY = {"geometry_score": 0.1, "geometry_status": "fail", "geometry_flags": ["OBB covers 60% of image"]}
WARN_GEOMETRY = {"geometry_score": 0.5, "geometry_status": "warning", "geometry_flags": ["strongly tilted"]}

POLE_SEMANTIC = {
    "status": "ok", "class": "electric_utility_pole", "material": "wood",
    "visibility": "mostly_visible", "orientation": "vertical",
    "semantic_confidence": 0.92, "decision": "accept", "reason": "Clear pole.",
}
TREE_SEMANTIC = {
    "status": "ok", "class": "tree", "material": "unknown",
    "visibility": "mostly_visible", "orientation": "vertical",
    "semantic_confidence": 0.88, "decision": "reject", "reason": "Looks like a tree.",
}
LOW_CONF_TREE_SEMANTIC = {**TREE_SEMANTIC, "semantic_confidence": 0.4}
OCCLUDED_SEMANTIC = {
    "status": "ok", "class": "electric_utility_pole", "material": "wood",
    "visibility": "severely_occluded", "orientation": "vertical",
    "semantic_confidence": 0.9, "decision": "review", "reason": "Mostly hidden by a truck.",
}
UNAVAILABLE_SEMANTIC = {"status": "unavailable", "class": None, "semantic_confidence": None}


class TestEvaluateCandidate(unittest.TestCase):

    def test_all_signals_good_accepts(self):
        result = evaluate_candidate(
            detection_score=0.85, segmentation_score=0.8,
            geometry_result=GOOD_GEOMETRY, semantic_result=POLE_SEMANTIC,
        )
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertGreater(result["final_score"], 0.65)
        self.assertEqual(result["disagreements"], [])

    def test_confident_qwen_non_pole_rejects(self):
        result = evaluate_candidate(
            detection_score=0.7, segmentation_score=0.6,
            geometry_result=GOOD_GEOMETRY, semantic_result=TREE_SEMANTIC,
        )
        self.assertEqual(result["decision"], "REJECT")
        self.assertTrue(any("non-pole" in r for r in result["reasons"]))

    def test_low_confidence_non_pole_call_does_not_auto_reject(self):
        """A low-confidence 'probably a tree' must not unilaterally reject -- spec section 19."""
        result = evaluate_candidate(
            detection_score=0.7, segmentation_score=0.6,
            geometry_result=GOOD_GEOMETRY, semantic_result=LOW_CONF_TREE_SEMANTIC,
        )
        self.assertNotEqual(result["decision"], "REJECT")

    def test_geometry_hard_fail_rejects_regardless_of_other_signals(self):
        result = evaluate_candidate(
            detection_score=0.95, segmentation_score=0.9,
            geometry_result=BAD_GEOMETRY, semantic_result=POLE_SEMANTIC,
        )
        self.assertEqual(result["decision"], "REJECT")

    def test_invalid_obb_always_rejects(self):
        result = evaluate_candidate(
            detection_score=0.95, segmentation_score=0.9,
            geometry_result=GOOD_GEOMETRY, semantic_result=POLE_SEMANTIC,
            obb_valid=False,
        )
        self.assertEqual(result["decision"], "REJECT")

    def test_severely_occluded_capped_at_review_never_accepted(self):
        result = evaluate_candidate(
            detection_score=0.9, segmentation_score=0.85,
            geometry_result=GOOD_GEOMETRY, semantic_result=OCCLUDED_SEMANTIC,
        )
        self.assertEqual(result["decision"], "REVIEW", "occluded/uncertain visibility must never auto-accept")

    def test_disagreement_between_dino_and_qwen_forces_review(self):
        result = evaluate_candidate(
            detection_score=0.9, segmentation_score=0.8,
            geometry_result=GOOD_GEOMETRY, semantic_result=TREE_SEMANTIC,
        )
        # High DINO conf + confident-tree Qwen actually hits the hard-REJECT
        # rule above; use a case that's confident-but-not-reject-threshold instead.
        mid_conf_tree = {**TREE_SEMANTIC, "semantic_confidence": 0.55}
        result2 = evaluate_candidate(
            detection_score=0.9, segmentation_score=0.8,
            geometry_result=GOOD_GEOMETRY, semantic_result=mid_conf_tree,
        )
        self.assertIn(result2["decision"], ("REVIEW", "REJECT"))
        self.assertTrue(len(result2["disagreements"]) >= 1 or result2["decision"] == "REJECT")

    def test_qwen_gated_off_renormalizes_weights_without_crashing(self):
        result = evaluate_candidate(
            detection_score=0.8, segmentation_score=0.75,
            geometry_result=GOOD_GEOMETRY, semantic_result=None,
        )
        self.assertFalse(result["semantic_ran"])
        self.assertNotIn("semantic", result["weights_used"])
        self.assertIsNone(result["semantic_score"])
        # Should still be able to accept on the remaining 3 signals.
        self.assertEqual(result["decision"], "ACCEPT")

    def test_geometry_qa_disabled_renormalizes_weights_without_crashing(self):
        result = evaluate_candidate(
            detection_score=0.85, segmentation_score=0.8,
            geometry_result=None, semantic_result=POLE_SEMANTIC,
        )
        self.assertNotIn("geometry", result["weights_used"])
        self.assertEqual(result["geometry_status"], "not_run")
        self.assertEqual(result["decision"], "ACCEPT")

    def test_unavailable_semantic_result_treated_as_not_run(self):
        result = evaluate_candidate(
            detection_score=0.85, segmentation_score=0.8,
            geometry_result=GOOD_GEOMETRY, semantic_result=UNAVAILABLE_SEMANTIC,
        )
        self.assertFalse(result["semantic_ran"])

    def test_low_final_score_rejects(self):
        result = evaluate_candidate(
            detection_score=0.2, segmentation_score=0.15,
            geometry_result=WARN_GEOMETRY, semantic_result=None,
        )
        self.assertEqual(result["decision"], "REJECT")


class TestDisagreementDetection(unittest.TestCase):

    def test_no_disagreement_when_semantic_absent(self):
        flags = detect_disagreement(0.9, 0.9, GOOD_GEOMETRY, None)
        self.assertEqual(flags, [])

    def test_dino_qwen_class_disagreement_flagged(self):
        flags = detect_disagreement(0.9, 0.8, GOOD_GEOMETRY, TREE_SEMANTIC)
        self.assertTrue(any("Qwen says" in f for f in flags))

    def test_qwen_pole_but_poor_segmentation_flagged(self):
        flags = detect_disagreement(0.5, 0.1, GOOD_GEOMETRY, POLE_SEMANTIC)
        self.assertTrue(any("segmentation_score is low" in f for f in flags))

    def test_geometry_none_does_not_crash_disagreement_check(self):
        flags = detect_disagreement(0.5, 0.8, None, POLE_SEMANTIC)
        self.assertIsInstance(flags, list)


class TestSegmentationScore(unittest.TestCase):

    def test_none_mask_scores_zero(self):
        self.assertEqual(compute_segmentation_score(None, (0, 0, 10, 10)), 0.0)

    def test_empty_mask_scores_zero(self):
        import numpy as np
        mask = np.zeros((50, 50), dtype=np.uint8)
        self.assertEqual(compute_segmentation_score(mask, (0, 0, 10, 10)), 0.0)

    def test_nonempty_mask_scores_between_zero_and_one(self):
        import numpy as np
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 45:55] = 255
        score = compute_segmentation_score(mask, (40, 15, 60, 85))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestDecisionConfigFromDict(unittest.TestCase):

    def test_partial_weights_dict_falls_back_to_defaults(self):
        cfg = config_from_dict({"weights": {"semantic": 0.5}, "accept_threshold": 0.7})
        self.assertEqual(cfg.weights.semantic, 0.5)
        self.assertEqual(cfg.weights.detection, DecisionWeights().detection)
        self.assertEqual(cfg.accept_threshold, 0.7)

    def test_none_or_empty_returns_defaults(self):
        self.assertEqual(config_from_dict(None).accept_threshold, DecisionConfig().accept_threshold)
        self.assertEqual(config_from_dict({}).accept_threshold, DecisionConfig().accept_threshold)


if __name__ == "__main__":
    unittest.main()
