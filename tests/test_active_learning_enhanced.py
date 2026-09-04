"""
tests/test_active_learning_enhanced.py - Tests for Active Learning Dataset Quality System
========================================================================================

Validates:
1. Primary objective declaration and category taxonomies.
2. confirmed_negative = baseline data (0 priority boost).
3. false_positive and missed_pole = high-priority active learning data (+30 boost).
4. Bad OBB detection and geometrical correction metrics (angle_diff, area_diff, corner_disp).
5. pHash duplicate detection (flags & penalizes without deleting).
6. Fixed test set isolation from train/val exports and dedicated test set export.
7. Empty .txt label generation for confirmed negative exports.
8. Dataset balance & coverage warning engine.
9. Configurable active learning weights file.
"""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import cv2

from backend.storage import (
    DatasetManager,
    PRIMARY_OBJECTIVE,
    POSITIVE_CATEGORIES,
    NEGATIVE_CATEGORIES,
    FAILURE_TYPES,
    compute_phash,
    hamming_distance,
    compute_obb_correction_metrics,
)
from models.adapters.base import DetectionBox


@pytest.fixture
def temp_dataset_dir():
    temp_dir = tempfile.mkdtemp(prefix="obb_al_test_")
    mgr = DatasetManager(base_dir=temp_dir)
    ds_id = "test_al_ds"
    mgr.create_dataset(ds_id, name="Active Learning Test DS")
    yield mgr, ds_id, Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _create_dummy_image(path: Path, color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((100, 100, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_primary_objective_and_category_taxonomy():
    """Verify primary objective is unambiguously defined and single-class pole focused."""
    assert "0 = utility_pole" in PRIMARY_OBJECTIVE
    assert "DINO + SAM" in PRIMARY_OBJECTIVE
    assert "human review establishes ground truth" in PRIMARY_OBJECTIVE
    assert "never treat an AI rejection as evidence that an image is negative" in PRIMARY_OBJECTIVE

    # Positive category taxonomy
    assert "clear_positive" in POSITIVE_CATEGORIES
    assert "occluded_positive" in POSITIVE_CATEGORIES
    assert "tilted_positive" in POSITIVE_CATEGORIES
    assert "small_positive" in POSITIVE_CATEGORIES

    # Negative category taxonomy
    assert "confirmed_negative" in NEGATIVE_CATEGORIES
    assert "tree" in NEGATIVE_CATEGORIES
    assert "street_light" in NEGATIVE_CATEGORIES
    assert "sign_post" in NEGATIVE_CATEGORIES

    # Failure types
    assert "false_positive" in FAILURE_TYPES
    assert "missed_pole" in FAILURE_TYPES
    assert "bad_obb" in FAILURE_TYPES
    assert "bad_segmentation" in FAILURE_TYPES


def test_confirmed_negative_zero_priority_boost(temp_dataset_dir):
    """
    confirmed_negative = baseline data (0 priority boost).
    An ordinary empty street image must NOT outrank a difficult false positive.
    """
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"

    # Image 1: Confirmed negative (empty baseline)
    fn_neg = "empty_street.jpg"
    _create_dummy_image(img_dir / fn_neg)
    mgr.import_images(ds_id, [img_dir / fn_neg])
    mgr.save_annotation(
        ds_id, fn_neg, boxes=[], img_width=100, img_height=100,
        action="accepted", is_negative=True, negative_category="confirmed_negative",
        al_categories=["confirmed_negative"]
    )

    # Image 2: Difficult false positive (tree trunk repeatedly mistaken by DINO)
    fn_fp = "tree_mistaken.jpg"
    _create_dummy_image(img_dir / fn_fp)
    mgr.import_images(ds_id, [img_dir / fn_fp])
    mgr.reject_with_metadata(
        ds_id, fn_fp, failure_type="false_positive", negative_category="tree"
    )

    # Image 3: Hard negative (street light)
    fn_hn = "street_light.jpg"
    _create_dummy_image(img_dir / fn_hn)
    mgr.import_images(ds_id, [img_dir / fn_hn])
    mgr.reject_with_metadata(
        ds_id, fn_hn, failure_type=None, negative_category="street_light"
    )

    queue = mgr.get_active_learning_queue(ds_id)
    q_map = {item["filename"]: item for item in queue}

    # Baseline confirmed negative should have priority 0
    assert q_map[fn_neg]["priority"] == 0

    # False positive (tree) should have high priority (>= 30)
    assert q_map[fn_fp]["priority"] >= 30

    # Hard negative (street light) should have moderate priority (>= 20)
    assert q_map[fn_hn]["priority"] >= 20

    # False positive must outrank confirmed negative!
    assert q_map[fn_fp]["priority"] > q_map[fn_neg]["priority"]


def test_reject_with_false_positive_preserves_predictions(temp_dataset_dir):
    """
    Rejecting an image records false_positive failure_type and negative_category,
    and preserves untouched AI predictions in predictions/ for active learning.
    """
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"
    fn = "sign_pole.jpg"
    _create_dummy_image(img_dir / fn)
    mgr.import_images(ds_id, [img_dir / fn])

    # Save mock DINO prediction
    ai_box = DetectionBox(
        xyxy=(10.0, 10.0, 30.0, 80.0),
        confidence=0.88,
        model_source="grounding-dino",
        corners=[[10, 10], [30, 10], [30, 80], [10, 80]]
    )
    mgr.save_predictions(ds_id, fn, [ai_box], {"dino": [ai_box.to_dict()]})

    # Human rejects as false positive on sign_post
    res = mgr.reject_with_metadata(
        ds_id,
        fn,
        failure_type="false_positive",
        negative_category="sign_post",
        al_categories=["sign_post", "false_positive"]
    )

    assert res["status"] == "rejected"
    assert res["failure_type"] == "false_positive"
    assert res["negative_category"] == "sign_post"

    # Verify untouched raw predictions still exist
    preds = mgr.get_predictions(ds_id, fn)
    assert preds is not None
    assert len(preds["boxes"]) == 1
    assert preds["boxes"][0]["model_source"] == "grounding-dino"


def test_missed_pole_recording(temp_dataset_dir):
    """Human-created OBB where AI found nothing records missed_pole with high priority."""
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"
    fn = "missed_pole_img.jpg"
    _create_dummy_image(img_dir / fn)
    mgr.import_images(ds_id, [img_dir / fn])

    # AI found 0 boxes
    mgr.save_predictions(ds_id, fn, [], {"dino": []})

    # Human adds a pole
    human_box = DetectionBox(
        xyxy=(20.0, 5.0, 40.0, 95.0),
        confidence=1.0,
        model_source="human",
        corners=[[20, 5], [40, 5], [40, 95], [20, 95]]
    )
    res = mgr.save_missed_pole(ds_id, fn, [human_box], 100, 100)

    assert res["status"] == "human_corrected"
    assert res["failure_type"] == "missed_pole"

    queue = mgr.get_active_learning_queue(ds_id)
    item = [x for x in queue if x["filename"] == fn][0]
    assert item["priority"] >= 30
    assert "missed_pole" in item["reasons"]


def test_bad_obb_correction_metrics():
    """Geometrical OBB difference metrics (angle, area, corner displacement) are accurately computed."""
    ai_box = {
        "corners": [[10.0, 10.0], [30.0, 10.0], [30.0, 80.0], [10.0, 80.0]]
    }
    # Tilted human box
    human_box = {
        "corners": [[15.0, 10.0], [35.0, 15.0], [25.0, 85.0], [5.0, 80.0]]
    }
    metrics = compute_obb_correction_metrics(ai_box, human_box)
    assert "angle_diff" in metrics
    assert "area_diff" in metrics
    assert "corner_displacement" in metrics
    assert "correction_iou" in metrics
    assert metrics["angle_diff"] > 0
    assert metrics["corner_displacement"] > 0


def test_phash_near_duplicate_detection_and_penalty(temp_dataset_dir):
    """Near-duplicate images are flagged (not deleted) and receive priority penalty."""
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"

    # Create two identical images
    fn1 = "scene_frame_01.jpg"
    fn2 = "scene_frame_02.jpg"
    _create_dummy_image(img_dir / fn1, color=(200, 100, 50))
    _create_dummy_image(img_dir / fn2, color=(200, 100, 50))

    mgr.import_images(ds_id, [img_dir / fn1, img_dir / fn2])

    # Compute pHash directly
    h1 = compute_phash(img_dir / fn1)
    h2 = compute_phash(img_dir / fn2)
    assert len(h1) == 16
    assert hamming_distance(h1, h2) == 0  # Identical images

    # Detect duplicates
    duplicates = mgr.find_near_duplicates(ds_id, threshold=8)
    assert len(duplicates) >= 1
    assert duplicates[0]["primary"] == fn1
    assert duplicates[0]["duplicate"] == fn2

    # Check that frame 2 is flagged as duplicate and penalised in AL queue
    queue = mgr.get_active_learning_queue(ds_id)
    q_map = {item["filename"]: item for item in queue}
    assert q_map[fn2]["is_duplicate"] is True
    assert "near_duplicate" in q_map[fn2]["reasons"]
    # Penalty ensures duplicate has lower priority than primary
    assert q_map[fn2]["priority"] < q_map[fn1]["priority"]


def test_fixed_test_set_isolation(temp_dataset_dir):
    """Test set images are strictly excluded from train/val exports, and exported separately."""
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"

    fn_train = "train_pole.jpg"
    fn_test = "benchmark_pole.jpg"
    _create_dummy_image(img_dir / fn_train)
    _create_dummy_image(img_dir / fn_test)
    mgr.import_images(ds_id, [img_dir / fn_train, img_dir / fn_test])

    box = DetectionBox(
        xyxy=(10.0, 10.0, 30.0, 80.0),
        confidence=1.0,
        model_source="human",
        corners=[[10, 10], [30, 10], [30, 80], [10, 80]]
    )

    mgr.save_annotation(ds_id, fn_train, [box], 100, 100, action="accepted")
    mgr.save_annotation(ds_id, fn_test, [box], 100, 100, action="accepted")

    # Mark benchmark_pole.jpg as FIXED TEST SET
    mgr.toggle_test_set(ds_id, fn_test)

    # Normal export (train/val)
    export_res = mgr.export_dataset(ds_id)
    exp_dir = Path(export_res["export_path"])
    exported_train_val = [p.name for p in (exp_dir / "images").glob("**/*.*")]

    # FIXED TEST SET image must NEVER be in train/val export!
    assert fn_test not in exported_train_val
    assert fn_train in exported_train_val

    # Export dedicated test set
    test_res = mgr.export_test_set(ds_id)
    test_exp_dir = Path(test_res["export_path"])
    test_exported = [p.name for p in (test_exp_dir / "images").glob("*.*")]
    assert fn_test in test_exported
    assert fn_train not in test_exported


def test_empty_label_for_confirmed_negative_export(temp_dataset_dir):
    """Verified negative images export with an empty .txt label file for YOLO background training."""
    mgr, ds_id, root = temp_dataset_dir
    img_dir = root / ds_id / "images"
    fn = "pure_background.jpg"
    _create_dummy_image(img_dir / fn)
    mgr.import_images(ds_id, [img_dir / fn])

    mgr.save_annotation(
        ds_id,
        fn,
        boxes=[],
        img_width=100,
        img_height=100,
        action="accepted",
        is_negative=True,
        negative_category="confirmed_negative"
    )

    export_res = mgr.export_dataset(ds_id)
    exp_dir = Path(export_res["export_path"])

    # Image is exported
    exported_imgs = [p.name for p in (exp_dir / "images").glob("**/*.*")]
    assert fn in exported_imgs

    # Label file exists and is empty
    label_files = list((exp_dir / "labels").glob("**/pure_background.txt"))
    assert len(label_files) == 1
    assert label_files[0].read_text(encoding="utf-8").strip() == ""


def test_dataset_coverage_warnings(temp_dataset_dir):
    """Rule-based warnings trigger when dataset is imbalanced or missing critical variations."""
    mgr, ds_id, root = temp_dataset_dir
    comp = {
        "total_images": 25,
        "positive_poles": 20,
        "hard_positives": 1,
        "confirmed_negatives": 1,
        "hard_negatives": 0,
        "false_positives": 0,
        "missed_poles": 0,
        "near_duplicates": 5,
        "by_category": {
            "clear_positive": 19,
            "occluded_positive": 0,  # 0% occluded -> should warn!
            "tilted_positive": 0,
            "edge_positive": 0,      # 0% tilted/edge -> should warn!
        },
        "by_negative": {
            "street_light": 0,       # Underrepresented -> should warn!
            "tree": 0,               # Underrepresented -> should warn!
        }
    }
    warnings = mgr.get_coverage_warnings(comp)
    assert any("heavily occluded" in w for w in warnings)
    assert any("Street-light" in w for w in warnings)
    assert any("Tree trunk" in w for w in warnings)
    assert any("near-duplicate" in w for w in warnings)
    assert any("Negatives constitute" in w for w in warnings)


def test_configurable_weights_persistence(temp_dataset_dir):
    """Active learning weights can be saved and loaded dynamically."""
    mgr, ds_id, root = temp_dataset_dir
    cfg = mgr._load_al_weights()
    assert "weights" in cfg
    assert cfg["weights"]["confirmed_negative"] == 0
    assert cfg["weights"]["false_positive"] == 30

    # Modify and save
    cfg["weights"]["false_positive"] = 45
    mgr._save_al_weights(cfg)

    reloaded = mgr._load_al_weights()
    assert reloaded["weights"]["false_positive"] == 45

    # Restore to 30
    cfg["weights"]["false_positive"] = 30
    mgr._save_al_weights(cfg)
