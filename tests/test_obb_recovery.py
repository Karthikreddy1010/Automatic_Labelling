"""
tests/test_obb_recovery.py - OBB Pole Recovery Stage
=======================================================

Tests models/adapters/obb_recovery.py: the geometry-driven fallback that
finds candidate poles when DINO+SAM3 miss them, misses part of one, or
returns only a weak candidate. Never replaces DINO/SAM3 -- see
obb_recovery.py's module docstring.

Split into three tiers, cheapest/most-deterministic first:
1. Config parsing (RecoveryConfig/config_from_dict) -- pure data.
2. Scoring functions given synthetic OBB corners directly -- pure geometry,
   no image processing, fully deterministic.
3. Candidate generation from synthetic images (a drawn vertical bar) --
   exercises the real OpenCV edge/line pipeline end-to-end.
"""
import unittest
import numpy as np
import cv2

from models.adapters.obb_recovery import (
    RecoveryConfig,
    config_from_dict,
    should_run_recovery,
    score_verticality,
    score_elongation,
    score_width_consistency,
    compute_pole_score,
    generate_recovery_candidates,
    reconstruct_partial_pole,
)
from models.adapters.recovery_dedup import dedup_recovery_against_existing
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners, obb_iou


def _vertical_obb(cx=100.0, top=20.0, bottom=180.0, half_width=6.0):
    """A perfectly vertical OBB centered at cx, spanning [top, bottom]."""
    return np.array([
        [cx - half_width, top],
        [cx + half_width, top],
        [cx + half_width, bottom],
        [cx - half_width, bottom],
    ], dtype=np.float64)


def _horizontal_obb(cy=100.0, left=20.0, right=180.0, half_height=6.0):
    """A perfectly horizontal OBB -- should score low on verticality/elongation
    for a *pole* (poles are tall, not wide)."""
    return np.array([
        [left, cy - half_height],
        [right, cy - half_height],
        [right, cy + half_height],
        [left, cy + half_height],
    ], dtype=np.float64)


def _tilted_obb(angle_deg, cx=100.0, cy=100.0, half_length=80.0, half_width=6.0):
    """A rectangle whose long axis is exactly angle_deg off true vertical, by
    construction (rotation), so tests assert against a known-exact tilt
    rather than a hand-picked coordinate set that might not match intent."""
    theta = np.radians(angle_deg)
    length_dir = np.array([np.sin(theta), np.cos(theta)])
    width_dir = np.array([np.cos(theta), -np.sin(theta)])
    center = np.array([cx, cy])
    top_c = center - half_length * length_dir
    bot_c = center + half_length * length_dir
    return np.array([
        top_c + half_width * width_dir,
        top_c - half_width * width_dir,
        bot_c - half_width * width_dir,
        bot_c + half_width * width_dir,
    ])


class TestRecoveryConfig(unittest.TestCase):

    def test_defaults_are_sane(self):
        cfg = RecoveryConfig()
        self.assertGreater(cfg.min_aspect_ratio, 1.0)
        self.assertGreater(cfg.max_aspect_ratio, cfg.min_aspect_ratio)
        self.assertGreater(cfg.review_threshold, cfg.min_pole_score)
        self.assertGreater(cfg.duplicate_iou_threshold, 0.0)

    def test_config_from_dict_reads_all_keys(self):
        raw = {
            "enabled": False,
            "recovery_activation_threshold": 0.7,
            "min_pole_score": 0.4,
            "review_threshold": 0.6,
            "max_candidates_per_image": 3,
            "min_aspect_ratio": 3.0,
            "max_aspect_ratio": 20.0,
            "max_angle_from_vertical_deg": 15.0,
            "min_width_px": 5.0,
            "max_width_px": 80.0,
            "duplicate_iou_threshold": 0.4,
            "max_extension_factor": 1.5,
            "weights": {
                "verticality": 1.0, "elongation": 1.0, "line_continuity": 1.0,
                "edge_consistency": 1.0, "width_consistency": 1.0,
                "segmentation_agreement": 1.0, "dino_agreement": 1.0,
                "position_context": 1.0,
            },
        }
        cfg = config_from_dict(raw)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.recovery_activation_threshold, 0.7)
        self.assertEqual(cfg.min_pole_score, 0.4)
        self.assertEqual(cfg.max_candidates_per_image, 3)
        self.assertEqual(cfg.weights.verticality, 1.0)

    def test_config_from_dict_tolerates_missing_keys(self):
        cfg = config_from_dict({"min_pole_score": 0.5})
        self.assertEqual(cfg.min_pole_score, 0.5)
        # Everything else falls back to the dataclass default, not a crash.
        self.assertEqual(cfg.max_candidates_per_image, RecoveryConfig().max_candidates_per_image)

    def test_config_from_dict_tolerates_none(self):
        cfg = config_from_dict(None)
        self.assertEqual(cfg, RecoveryConfig())


class TestActivationGating(unittest.TestCase):
    """Recovery must only run when the primary pipeline missed or produced a
    weak result -- never as a blanket second-guess of a confident DINO/SAM
    result (spec: 'If DINO/SAM confidently detects a pole -> use the normal
    result')."""

    def test_no_existing_candidates_triggers_recovery(self):
        cfg = RecoveryConfig()
        self.assertTrue(should_run_recovery([], cfg))

    def test_confident_existing_candidate_skips_recovery(self):
        cfg = RecoveryConfig(recovery_activation_threshold=0.55)
        strong = DetectionBox(xyxy=(10, 10, 50, 200), confidence=0.9, model_source="DINO+SAM3+SAM")
        self.assertFalse(should_run_recovery([strong], cfg))

    def test_weak_existing_candidate_triggers_recovery(self):
        cfg = RecoveryConfig(recovery_activation_threshold=0.55)
        weak = DetectionBox(xyxy=(10, 10, 50, 200), confidence=0.30, model_source="DINO+SAM3+SAM")
        self.assertTrue(should_run_recovery([weak], cfg))

    def test_disabled_config_never_triggers_recovery(self):
        cfg = RecoveryConfig(enabled=False)
        self.assertFalse(should_run_recovery([], cfg))


class TestGeometryScoring(unittest.TestCase):
    """Pure-geometry signals given OBB corners directly -- no image I/O, so
    these are exact and deterministic."""

    def test_vertical_obb_scores_high_verticality(self):
        corners = _vertical_obb()
        self.assertGreater(score_verticality(corners), 0.9)

    def test_horizontal_obb_scores_low_verticality(self):
        corners = _horizontal_obb()
        self.assertLess(score_verticality(corners), 0.2)

    def test_tilted_obb_scores_between(self):
        corners = _tilted_obb(30.0)  # exactly 30 degrees off vertical
        v = score_verticality(corners)
        self.assertAlmostEqual(v, np.cos(np.radians(30.0)), places=2)
        self.assertGreater(v, 0.2)
        self.assertLess(v, 0.95)

    def test_tall_thin_obb_scores_high_elongation(self):
        corners = _vertical_obb(half_width=4.0, top=20, bottom=200)  # ~180 tall, 8 wide -> AR=22.5
        self.assertGreater(score_elongation(corners, RecoveryConfig()), 0.7)

    def test_square_obb_scores_low_elongation(self):
        corners = np.array([[50, 50], [100, 50], [100, 100], [50, 100]], dtype=np.float64)
        self.assertLess(score_elongation(corners, RecoveryConfig()), 0.2)

    def test_width_consistency_perfect_for_rectangle(self):
        # A true axis-aligned rectangle has perfectly constant width along its height.
        corners = _vertical_obb()
        self.assertGreater(score_width_consistency(corners, None), 0.95)


class TestComputePoleScore(unittest.TestCase):
    """The blended pole_score must combine signals with configurable weights
    and renormalize over whichever signals actually have evidence, mirroring
    decision_engine.py's existing renormalization pattern."""

    def test_strong_vertical_candidate_scores_higher_than_horizontal(self):
        cfg = RecoveryConfig()
        vertical_result = compute_pole_score(_vertical_obb(), config=cfg)
        horizontal_result = compute_pole_score(_horizontal_obb(), config=cfg)
        self.assertGreater(vertical_result["pole_score"], horizontal_result["pole_score"])

    def test_score_result_reports_all_components(self):
        result = compute_pole_score(_vertical_obb(), config=RecoveryConfig())
        for key in ("verticality", "elongation", "width_consistency"):
            self.assertIn(key, result["components"])
        self.assertIn("pole_score", result)
        self.assertGreaterEqual(result["pole_score"], 0.0)
        self.assertLessEqual(result["pole_score"], 1.0)

    def test_dino_agreement_boosts_score_when_present(self):
        cfg = RecoveryConfig()
        # A moderate (not already-maxed) candidate, so a strong DINO signal
        # has room to pull the blended score up rather than just diluting it.
        corners = _vertical_obb(top=90.0, bottom=130.0)  # AR ~3.3, mid-range elongation
        without = compute_pole_score(corners, config=cfg, dino_confidence=None)
        with_strong_dino = compute_pole_score(corners, config=cfg, dino_confidence=0.9)
        self.assertGreater(with_strong_dino["pole_score"], without["pole_score"])


class TestCandidateGenerationOnSyntheticImages(unittest.TestCase):
    """End-to-end candidate generation via the real OpenCV edge/line pipeline,
    on synthetic images where the ground truth is known by construction."""

    def _blank_image(self, w=300, h=300):
        return np.full((h, w, 3), 40, dtype=np.uint8)  # flat dark gray, no edges at all

    def _image_with_vertical_bar(self, w=300, h=300, x=150, bar_width=10):
        img = self._blank_image(w, h)
        img[20:280, x - bar_width // 2: x + bar_width // 2] = 220  # bright vertical bar
        return img

    def test_blank_image_yields_no_candidates(self):
        img = self._blank_image()
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        self.assertEqual(candidates, [])

    def test_vertical_bar_yields_at_least_one_candidate(self):
        img = self._image_with_vertical_bar()
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        self.assertGreaterEqual(len(candidates), 1)
        # The best candidate should be roughly vertical and tall.
        best = max(candidates, key=lambda c: c.attributes["recovery_score"]["pole_score"])
        self.assertGreater(score_verticality(np.array(best.corners)), 0.7)

    def test_every_generated_candidate_is_structurally_valid(self):
        img = self._image_with_vertical_bar()
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        from models.adapters.obb_generator import validate_obb
        for c in candidates:
            errors = validate_obb(np.array(c.corners), img.shape[1], img.shape[0])
            self.assertEqual(errors, [], f"invalid OBB from recovery: {errors}")

    def test_candidates_are_capped_at_max_candidates_per_image(self):
        img = self._blank_image(600, 600)
        # Several bars -> several candidates, but never more than the cap.
        for x in (60, 150, 250, 350, 450, 550):
            img[20:580, x - 5:x + 5] = 220
        cfg = RecoveryConfig(max_candidates_per_image=3)
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=cfg)
        self.assertLessEqual(len(candidates), 3)

    def test_tree_canopy_shape_does_not_produce_a_confident_pole_candidate(self):
        """False-positive protection (spec): a tree canopy is wide/irregular,
        not tall-thin like a pole shaft -- it must not produce a
        high-confidence candidate purely from being a large bright blob."""
        img = self._blank_image(300, 300)
        cv2.circle(img, (150, 100), 70, (200, 200, 200), -1)  # wide round canopy, no trunk
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        for c in candidates:
            # Either filtered out entirely (aspect ratio too low), or kept
            # but never at high confidence -- must not be silently ACCEPT-able.
            self.assertLess(c.attributes["recovery_score"]["pole_score"], 0.6)

    def test_building_edge_wide_rectangle_is_rejected_by_aspect_ratio(self):
        """A building wall/edge is wide and short relative to a pole shaft --
        must fail the configurable min_aspect_ratio filter, not be treated
        as 'vertical enough' just because it has straight edges."""
        img = self._blank_image(300, 300)
        img[100:180, 20:280] = 220  # wide horizontal band, low aspect ratio either orientation
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        self.assertEqual(candidates, [])

    def test_recovered_candidates_are_marked_with_recovery_model_source(self):
        img = self._image_with_vertical_bar()
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=RecoveryConfig())
        for c in candidates:
            self.assertIn("RECOVERY", c.model_source)

    def test_low_score_candidate_marked_review_required_never_silently_accepted(self):
        # A short, borderline-aspect-ratio bar should still get flagged for
        # review rather than silently treated as a confident accept.
        img = self._blank_image()
        img[130:170, 145:155] = 150  # short, low-contrast, near-square smudge
        cfg = RecoveryConfig(min_pole_score=0.05)  # let weak candidates through so we can inspect them
        candidates = generate_recovery_candidates(img, existing_boxes=[], config=cfg)
        for c in candidates:
            if c.attributes["recovery_score"]["pole_score"] < cfg.review_threshold:
                self.assertTrue(c.needs_review)
                self.assertIn("REVIEW_REQUIRED", c.attributes.get("recovery_status", ""))


class TestPartialPoleReconstruction(unittest.TestCase):
    """Spec: SAM detects only the lower section of a pole -- reconstruct
    toward the missing section using centerline/orientation/width evidence,
    never blindly extending to the image border."""

    def _image_with_vertical_bar(self, w=300, h=300, x=150, bar_width=10, top=20, bottom=280):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        img[top:bottom, x - bar_width // 2: x + bar_width // 2] = 220
        return img

    def test_reconstructs_toward_missing_upper_section(self):
        img = self._image_with_vertical_bar(top=20, bottom=280)
        # Partial mask covers only the lower third (bottom=280 up to y=200).
        partial_mask = np.zeros((300, 300), dtype=np.uint8)
        partial_mask[200:280, 145:155] = 1
        result = reconstruct_partial_pole(img, partial_mask, config=RecoveryConfig())
        self.assertIsNotNone(result)
        recon_corners = np.array(result["corners"])
        # Reconstructed top should move upward (smaller y) from the partial
        # mask's own top (200), but not blindly jump to the image border (0).
        recon_top_y = recon_corners[:, 1].min()
        self.assertLess(recon_top_y, 200)
        self.assertGreater(recon_top_y, 0)

    def test_does_not_extend_past_max_extension_factor(self):
        img = self._image_with_vertical_bar(top=20, bottom=280)
        partial_mask = np.zeros((300, 300), dtype=np.uint8)
        partial_mask[250:280, 145:155] = 1  # only a 30px sliver observed
        cfg = RecoveryConfig(max_extension_factor=1.5)  # extend at most 1.5x the observed 30px
        result = reconstruct_partial_pole(img, partial_mask, config=cfg)
        self.assertIsNotNone(result)
        recon_corners = np.array(result["corners"])
        recon_top_y = recon_corners[:, 1].min()
        # Observed segment spans y in [250, 280] (30px); extension capped at
        # 1.5x that (45px) above y=250, i.e. must not go above y=205.
        self.assertGreaterEqual(recon_top_y, 250 - 30 * 1.5 - 1.0)

    def test_empty_mask_returns_none(self):
        img = self._image_with_vertical_bar()
        empty_mask = np.zeros((300, 300), dtype=np.uint8)
        result = reconstruct_partial_pole(img, empty_mask, config=RecoveryConfig())
        self.assertIsNone(result)


class TestDuplicateHandling(unittest.TestCase):
    """The recovery stage must not create duplicate labels when DINO/SAM
    already detected the same pole -- rotated (Shapely) IoU, never
    axis-aligned."""

    def test_heavily_overlapping_recovery_candidate_is_discarded(self):
        existing = DetectionBox(
            xyxy=(90, 20, 110, 180),
            corners=_vertical_obb().tolist(),
            confidence=0.8,
            model_source="DINO+SAM3+SAM",
        )
        # Same physical pole, near-identical OBB.
        duplicate = DetectionBox(
            xyxy=(91, 21, 109, 179),
            corners=_vertical_obb(cx=100.5).tolist(),
            confidence=0.5,
            model_source="OBB_RECOVERY",
            attributes={"recovery_score": {"pole_score": 0.6}},
        )
        kept, discarded = dedup_recovery_against_existing(
            [duplicate], [existing], iou_threshold=0.35,
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0]["reason"], "duplicate_of_existing")

    def test_genuinely_separate_pole_is_kept(self):
        existing = DetectionBox(
            xyxy=(90, 20, 110, 180), corners=_vertical_obb().tolist(),
            confidence=0.8, model_source="DINO+SAM3+SAM",
        )
        far_away = DetectionBox(
            xyxy=(490, 20, 510, 180), corners=_vertical_obb(cx=500).tolist(),
            confidence=0.5, model_source="OBB_RECOVERY",
            attributes={"recovery_score": {"pole_score": 0.6}},
        )
        kept, discarded = dedup_recovery_against_existing(
            [far_away], [existing], iou_threshold=0.35,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(discarded, [])

    def test_axis_aligned_false_positive_is_correctly_kept_by_rotated_iou(self):
        # Two OBBs whose axis-aligned bounding boxes overlap heavily (so a
        # naive axis-aligned IoU would wrongly call them duplicates) but whose
        # actual rotated polygons barely touch -- e.g. two poles crossing
        # near a diagonal wire, at very different angles.
        existing_corners = np.array([[95, 20], [105, 20], [105, 180], [95, 180]])  # vertical, x in [95,105]
        recovery_corners = np.array([[20, 95], [180, 95], [180, 105], [20, 105]])  # horizontal, y in [95,105]
        existing = DetectionBox(xyxy=(95, 20, 105, 180), corners=existing_corners.tolist(),
                                 confidence=0.8, model_source="DINO+SAM3+SAM")
        recovery = DetectionBox(xyxy=(20, 95, 180, 105), corners=recovery_corners.tolist(),
                                 confidence=0.5, model_source="OBB_RECOVERY",
                                 attributes={"recovery_score": {"pole_score": 0.6}})
        rotated_iou = obb_iou(existing_corners, recovery_corners)
        self.assertLess(rotated_iou, 0.35)  # they only cross in a small square, not a real duplicate
        kept, discarded = dedup_recovery_against_existing([recovery], [existing], iou_threshold=0.35)
        self.assertEqual(len(kept), 1)


class TestPipelineIntegration(unittest.TestCase):
    """backend.app._apply_obb_recovery -- the actual integration point where
    the recovery stage is wired into run_ai_pipeline(), after
    _apply_verification() (DINO+SAM3+geometry+Qwen+decision_engine)."""

    def _image_with_vertical_bar_path(self, tmp_path, w=300, h=300, x=150, bar_width=10):
        import cv2
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        img[20:280, x - bar_width // 2: x + bar_width // 2] = 220
        p = tmp_path / "bar.jpg"
        cv2.imwrite(str(p), img)
        return p

    def test_confident_existing_result_is_not_touched_by_recovery(self):
        import tempfile
        from pathlib import Path as P
        from backend.app import _apply_obb_recovery

        with tempfile.TemporaryDirectory() as td:
            img_path = self._image_with_vertical_bar_path(P(td))
            confident = DetectionBox(
                xyxy=(140, 20, 160, 280), corners=_vertical_obb(cx=150, top=20, bottom=280, half_width=10).tolist(),
                confidence=0.95, model_source="DINO+SAM3+SAM", attributes={"decision": "ACCEPT"},
            )
            result_boxes, counts = _apply_obb_recovery([confident], img_path, dino_boxes=[], filename="bar.jpg")
            self.assertFalse(counts["recovery_ran"])
            self.assertEqual(result_boxes, [confident])  # untouched, not even re-ordered

    def test_missed_pole_triggers_recovery_and_adds_candidate(self):
        import tempfile
        from pathlib import Path as P
        from backend.app import _apply_obb_recovery

        with tempfile.TemporaryDirectory() as td:
            img_path = self._image_with_vertical_bar_path(P(td))
            result_boxes, counts = _apply_obb_recovery([], img_path, dino_boxes=[], filename="bar.jpg")
            self.assertTrue(counts["recovery_ran"])
            self.assertGreaterEqual(len(result_boxes), 1)
            for b in result_boxes:
                self.assertEqual(b.model_source, "OBB_RECOVERY")
                self.assertTrue(b.needs_review)

    def test_enable_override_false_forces_recovery_off_regardless_of_config(self):
        import tempfile
        from pathlib import Path as P
        from backend.app import _apply_obb_recovery

        with tempfile.TemporaryDirectory() as td:
            img_path = self._image_with_vertical_bar_path(P(td))
            # No existing boxes -> config.yaml's default would normally
            # trigger recovery -- enable_override=False (the A-vs-B
            # evaluation harness's "A: DINO+SAM only" config) must suppress
            # it entirely regardless.
            result_boxes, counts = _apply_obb_recovery(
                [], img_path, dino_boxes=[], filename="bar.jpg", enable_override=False,
            )
            self.assertFalse(counts["recovery_ran"])
            self.assertEqual(result_boxes, [])

    def test_recovery_does_not_duplicate_an_existing_weak_candidate_of_the_same_pole(self):
        import tempfile
        from pathlib import Path as P
        from backend.app import _apply_obb_recovery

        with tempfile.TemporaryDirectory() as td:
            img_path = self._image_with_vertical_bar_path(P(td))
            # A weak existing candidate that's actually the SAME bar (same
            # location), just below the activation threshold -- recovery
            # should run (weak result) but must not add a second label for
            # the same physical pole.
            weak_same_pole = DetectionBox(
                xyxy=(140, 20, 160, 280),
                corners=_vertical_obb(cx=150, top=20, bottom=280, half_width=10).tolist(),
                confidence=0.30, model_source="DINO+SAM3+SAM",
            )
            result_boxes, counts = _apply_obb_recovery(
                [weak_same_pole], img_path, dino_boxes=[], filename="bar.jpg",
            )
            self.assertTrue(counts["recovery_ran"])
            # Still exactly one box for this one physical pole -- the
            # recovered candidate(s) for the same bar must have been
            # deduped away, not appended as a second label.
            self.assertEqual(len(result_boxes), 1)
            self.assertIs(result_boxes[0], weak_same_pole)


if __name__ == "__main__":
    unittest.main()
