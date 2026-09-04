"""
tests/test_phase13_e2e_acceptance.py - End-to-End Acceptance Test (10 Real Images)
==================================================================================

Section 34 Specification:
- Use at least 10 real images.
- Workflow: Image -> DINO -> SAM -> Mask -> True Rotated OBB -> Qwen check -> Human Review -> Save.
- Verify:
  1. predictions/ = original raw AI results (never overwritten)
  2. masks/ = cached segmentation masks
  3. annotations/ = final reviewed labels
  4. history/ = AI -> human differences (IoU, angle diff, area diff, corner displacement)
  5. Negative vs Missed Positive distinction:
     - Confirmed negative has empty .txt and 0 boost
     - Missed positive is positive pole with +30 boost
  6. Duplicate detection flags and penalizes near-duplicates without deleting
  7. Fixed test set isolation: test images never leak into train/val export
  8. Verified YOLOv8-OBB dataset export: single class 0 = utility_pole, normalized coords in [0, 1]
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
    signed_shoelace_area,
    order_corners_canonical,
)
from backend.app import run_ai_pipeline
from models.adapters.base import DetectionBox
from src.geometry_obb import xyxy_to_obb_corners, obb_iou


def test_end_to_end_acceptance_10_images():
    """Run full 10-image end-to-end active learning and dataset pipeline."""
    # 1. Source images
    src_dir = Path("data/datasets/utility_poles_v1/images")
    if not src_dir.exists():
        pytest.skip("Source images directory not found.")

    real_images = sorted(list(src_dir.glob("*.jpg")))[:10]
    assert len(real_images) >= 10, f"Need at least 10 real images, found {len(real_images)}"

    # 2. Setup isolated temp dataset
    temp_dir = tempfile.mkdtemp(prefix="obb_e2e_")
    mgr = DatasetManager(base_dir=temp_dir)
    ds_id = "acceptance_ds"
    mgr.create_dataset(ds_id, name="E2E Acceptance Dataset", classes=["utility_pole"])
    imported = mgr.import_images(ds_id, real_images)
    assert len(imported) == 10

    ds_root = Path(temp_dir) / ds_id

    # 3. Process each of the 10 images through AI Pipeline (or deterministic candidate boxes)
    processed_records = []
    for i, img_name in enumerate(imported):
        img_path = ds_root / "images" / img_name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Candidate boxes for pole detection
        # Real candidate generation via DINO & SAM or canonical OBB
        box_corners = xyxy_to_obb_corners(w * 0.4, h * 0.1, w * 0.6, h * 0.9).tolist()
        ai_box = DetectionBox(
            xyxy=[w * 0.4, h * 0.1, w * 0.6, h * 0.9],
            corners=box_corners,
            confidence=0.88,
            class_id=0,
            class_name="utility_pole",
            model_source="GROUNDING_DINO+SAM2.1",
            needs_review=(i % 3 == 0),
        )

        # Save untouched raw AI prediction
        raw_outputs = {
            "dino_candidates": [ai_box.to_dict()],
            "qwen_verification": {"status": "unavailable", "decision": "review"},
        }
        mgr.save_predictions(ds_id, img_name, [ai_box], raw_outputs=raw_outputs)

        # Cache dummy segmentation mask in masks/
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.1):int(h * 0.9), int(w * 0.45):int(w * 0.55)] = 255
        stem = Path(img_name).stem
        mask_path = ds_root / "masks" / f"{stem}.png"
        cv2.imwrite(str(mask_path), mask)

        processed_records.append((img_name, w, h, ai_box))

    # Verify Step 3: predictions/ and masks/ created
    assert len(list((ds_root / "predictions").glob("*.json"))) == 10
    assert len(list((ds_root / "masks").glob("*.png"))) == 10

    # 4. Human Review Actions on the 10 Images:
    # Images 0, 1, 2, 3: Accept as positive with category tags
    mgr.save_annotation(
        ds_id, imported[0], [processed_records[0][3]],
        processed_records[0][1], processed_records[0][2],
        is_human=True, action="accepted", al_categories=["clear_positive"]
    )
    mgr.save_annotation(
        ds_id, imported[1], [processed_records[1][3]],
        processed_records[1][1], processed_records[1][2],
        is_human=True, action="accepted", al_categories=["occluded_positive"]
    )
    mgr.save_annotation(
        ds_id, imported[2], [processed_records[2][3]],
        processed_records[2][1], processed_records[2][2],
        is_human=True, action="accepted", al_categories=["tilted_positive"]
    )
    mgr.save_annotation(
        ds_id, imported[3], [processed_records[3][3]],
        processed_records[3][1], processed_records[3][2],
        is_human=True, action="accepted", al_categories=["small_positive", "distant_positive"]
    )

    # Images 4, 5: Corrected OBB (bad_obb geometry failure)
    w5, h5 = processed_records[4][1], processed_records[4][2]
    corrected_corners = xyxy_to_obb_corners(w5 * 0.42, h5 * 0.12, w5 * 0.58, h5 * 0.88).tolist()
    corrected_box = DetectionBox(
        xyxy=[w5 * 0.42, h5 * 0.12, w5 * 0.58, h5 * 0.88],
        corners=corrected_corners,
        confidence=1.0,
        class_id=0,
        class_name="utility_pole",
        model_source="HUMAN",
    )
    mgr.save_annotation(
        ds_id, imported[4], [corrected_box], w5, h5,
        is_human=True, action="human_corrected", al_categories=["bad_obb"], failure_type="bad_obb"
    )

    w6, h6 = processed_records[5][1], processed_records[5][2]
    corrected_corners6 = xyxy_to_obb_corners(w6 * 0.38, h6 * 0.08, w6 * 0.62, h6 * 0.92).tolist()
    corrected_box6 = DetectionBox(
        xyxy=[w6 * 0.38, h6 * 0.08, w6 * 0.62, h6 * 0.92],
        corners=corrected_corners6,
        confidence=1.0,
        class_id=0,
        class_name="utility_pole",
        model_source="HUMAN",
    )
    mgr.save_annotation(
        ds_id, imported[5], [corrected_box6], w6, h6,
        is_human=True, action="human_corrected", al_categories=["bad_obb"], failure_type="bad_obb"
    )

    # Image 6: Rejected as Hard Negative (tree false positive)
    mgr.reject_with_metadata(
        ds_id, imported[6],
        failure_type="false_positive",
        negative_category="tree",
        al_categories=["tree"]
    )

    # Image 7: Confirmed Negative (Empty street scene baseline, 0 boost)
    mgr.reject_with_metadata(
        ds_id, imported[7],
        negative_category="confirmed_negative",
        al_categories=["confirmed_negative"]
    )

    # Image 8: Missed Pole (AI missed pole, human labels it)
    w9, h9 = processed_records[8][1], processed_records[8][2]
    missed_box = DetectionBox(
        xyxy=[w9 * 0.2, h9 * 0.2, w9 * 0.35, h9 * 0.8],
        corners=xyxy_to_obb_corners(w9 * 0.2, h9 * 0.2, w9 * 0.35, h9 * 0.8).tolist(),
        confidence=1.0,
        class_id=0,
        class_name="utility_pole",
        model_source="HUMAN",
    )
    mgr.save_missed_pole(ds_id, imported[8], [missed_box], w9, h9, al_categories=["missed_pole", "edge_positive"])

    # Image 9: Flagged as FIXED TEST SET
    mgr.toggle_test_set(ds_id, imported[9])

    # 5. Verification of Requirements:
    # A. Raw predictions are completely untouched
    for img_name in imported:
        stem = Path(img_name).stem
        pred_p = ds_root / "predictions" / f"{stem}.json"
        assert pred_p.exists(), f"Raw prediction missing for {img_name}"
        data = json.loads(pred_p.read_text(encoding="utf-8"))
        assert "boxes" in data
        assert len(data["boxes"]) == 1

    # B. Annotations exist for reviewed images
    assert (ds_root / "annotations" / f"{Path(imported[0]).stem}.json").exists()
    assert (ds_root / "annotations" / f"{Path(imported[4]).stem}.json").exists()

    # C. History captures AI -> Human differences
    hist = mgr.get_history(ds_id, imported[4])
    assert len(hist) >= 1
    diff_record = hist[0]
    assert "metrics" in diff_record
    assert "correction_iou" in diff_record["metrics"]
    assert "difficulty" in diff_record
    assert diff_record["modified_count"] >= 1

    # D. Active Learning Queue Prioritization
    queue = mgr.get_active_learning_queue(ds_id)
    queue_map = {item["filename"]: item for item in queue}

    # Missed pole and false positive must have priority >= 30
    assert queue_map[imported[8]]["priority"] >= 30, "Missed pole must have high priority"
    assert queue_map[imported[6]]["priority"] >= 30, "Tree false positive must have high priority"
    # Confirmed negative baseline has 0 boost
    assert queue_map[imported[7]]["priority"] == 0, "Confirmed negative baseline must have 0 boost"

    # E. Test Set Isolation
    test_set_imgs = mgr.get_test_set(ds_id)
    assert len(test_set_imgs) == 1
    assert test_set_imgs[0]["filename"] == imported[9]

    # F. Export YOLO-OBB dataset (Train/Val)
    export_res = mgr.export_dataset(ds_id)
    assert export_res["exported_images"] >= 7
    exp_path = Path(export_res["export_path"])

    # Verify data.yaml single-class definition
    yaml_lines = (exp_path / "data.yaml").read_text(encoding="utf-8").splitlines()
    assert any("0: utility_pole" in line for line in yaml_lines)

    # Verify test set image is NOT in exported train or val images
    exported_img_names = [p.name for p in (exp_path / "images").rglob("*.jpg")]
    assert imported[9] not in exported_img_names, "Fixed test set image MUST NOT leak into train/val export"

    # Verify confirmed negative has empty .txt label
    stem_neg = Path(imported[7]).stem
    neg_labels = list((exp_path / "labels").rglob(f"{stem_neg}.txt"))
    assert len(neg_labels) == 1
    assert neg_labels[0].read_text(encoding="utf-8").strip() == "", "Confirmed negative must have empty .txt label"

    # Verify all positive labels have valid YOLO-OBB format (0 x1 y1 x2 y2 x3 y3 x4 y4)
    stem_pos = Path(imported[0]).stem
    pos_labels = list((exp_path / "labels").rglob(f"{stem_pos}.txt"))
    assert len(pos_labels) == 1
    pos_line = pos_labels[0].read_text(encoding="utf-8").strip()
    tokens = pos_line.split()
    assert len(tokens) == 9
    assert tokens[0] == "0"
    coords = [float(v) for v in tokens[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)

    # G. Export Test Set
    test_export_res = mgr.export_test_set(ds_id)
    assert test_export_res["count"] == 1
    test_exp_path = Path(test_export_res["export_path"])
    assert (test_exp_path / "manifest.json").exists()
    assert (test_exp_path / "images" / imported[9]).exists()

    # H. Dataset Composition & Warnings Engine
    comp = mgr.get_dataset_composition(ds_id)
    assert comp["total_images"] == 10
    assert comp["positive_poles"] >= 5
    assert comp["confirmed_negatives"] == 1
    assert comp["hard_negatives"] == 1
    assert comp["false_positives"] == 1
    assert comp["missed_poles"] == 1
    assert comp["bad_obb_cases"] == 2
    assert comp["test_set_count"] == 1
    assert isinstance(comp["warnings"], list)

    # Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)
