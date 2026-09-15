"""
benchmark_dino_sam.py - Small-image-set benchmark harness for the production
Grounding DINO -> SAM3/SAM 2.1 -> Geometry QA -> Qwen3-VL -> Decision Engine
pipeline (backend.app.run_ai_pipeline, mode="AI_LABEL").

Calls the exact same function the live API uses -- no duplicated pipeline
logic -- draws each image's OBB candidates color-coded by decision, and
writes a structured report.

Two modes:
1. Single-config (default): run one pipeline configuration on the image set,
   draw overlays, report per-image and aggregate counts.
2. --compare-configs: run all four configurations from the spec's section 13
   on the SAME images and report them side by side:
     A: DINO+SAM3 only            (geometry_qa=off, qwen=off)
     B: DINO+SAM3+Qwen            (geometry_qa=off, qwen=always)
     C: DINO+SAM3+GeometryQA      (geometry_qa=on,  qwen=off)
     D: DINO+SAM3+GeometryQA+Qwen (geometry_qa=on,  qwen=always)

This tool does NOT auto-compute true precision/recall unless you pass
--ground-truth-dir pointing at human-verified YOLO-OBB .txt files (same
stem as each image, format "class x1 y1 x2 y2 x3 y3 x4 y4" normalized
[0,1] -- exactly what backend/storage.py writes to annotations/). Without
ground truth, the report only counts candidates and decision buckets --
a report with zero REJECTs is not evidence the pipeline is accurate, it
only means nothing was rejected on these particular images. Open the
overlay images and judge each one by eye, or supply real ground truth.

Usage:
    # Sample N images from a directory, single config (defaults match the
    # live app's AI_LABEL button: geometry QA on, Qwen gated)
    python benchmark_dino_sam.py --images data/datasets/utility_poles_v1/images --n 30 --out output/benchmark_30

    # Explicit hand-picked file list
    python benchmark_dino_sam.py --files img1.jpg img2.jpg ... --out output/benchmark_30

    # Compare all 4 verification-pipeline configurations on the same images
    python benchmark_dino_sam.py --files ... --compare-configs --out output/compare

    # With real ground truth for actual precision/recall
    python benchmark_dino_sam.py --files ... --ground-truth-dir data/datasets/utility_poles_v1/annotations --out output/eval

    # With category labels for the stratified breakdown, and a YOLO(best.pt) comparison overlay
    python benchmark_dino_sam.py --files ... --manifest categories.json --compare-yolo --out output/benchmark_30

Where categories.json is {"img1.jpg": "tilted", "img2.jpg": "small_distant", ...}.
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from backend.app import run_ai_pipeline
from src.evaluation import DetectionEvaluator
from src.geometry_obb import yolo_obb_line_to_corners, obb_corners_to_xyxy

DECISION_COLOR = {
    "ACCEPT": (0, 200, 0),     # green (BGR)
    "REVIEW": (0, 200, 255),   # amber
    "REJECT": (0, 0, 255),     # red (only ever drawn if a caller inspects a
                                # pre-filter candidate; the pipeline itself
                                # drops REJECTs before returning)
}
# Fallback for candidates that never reached the decision engine (e.g.
# geometry QA / Qwen disabled entirely and quality.py's own category is all
# that's available).
CATEGORY_COLOR = {"HIGH_QUALITY": (0, 200, 0), "REVIEW": (0, 200, 255), "REJECT": (0, 0, 255)}
YOLO_COMPARE_COLOR = (255, 128, 0)  # orange, benchmark-only overlay
# Matches frontend/app.js::drawOBB's recovery colors (BGR here, RGB there)
# so a reviewer sees the same visual language in both the live UI and these
# offline overlays.
RECOVERY_COLOR = {"RECOVERED_CANDIDATE": (207, 197, 57), "REVIEW_REQUIRED": (186, 120, 247)}
DINO_RAW_COLOR = (128, 128, 128)  # gray, --full-debug only

CONFIG_PRESETS = {
    "A": {"enable_geometry_qa": False, "qwen_gating": "off"},
    "B": {"enable_geometry_qa": False, "qwen_gating": "always"},
    "C": {"enable_geometry_qa": True, "qwen_gating": "off"},
    "D": {"enable_geometry_qa": True, "qwen_gating": "always"},
}
CONFIG_LABELS = {
    "A": "DINO+SAM3 only",
    "B": "DINO+SAM3+Qwen",
    "C": "DINO+SAM3+GeometryQA",
    "D": "DINO+SAM3+GeometryQA+Qwen",
}


def draw_obb(img: np.ndarray, corners, color, label: str) -> None:
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    x, y = pts[0][0]
    cv2.putText(img, label, (int(x) + 2, max(12, int(y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _load_ground_truth(gt_dir: str, name: str, img_w: int, img_h: int):
    """Load a human-verified YOLO-OBB .txt (same stem as the image) and
    return a list of {'bbox': xyxy} dicts for DetectionEvaluator, or None if
    no ground-truth file exists for this image."""
    stem = Path(name).stem
    txt_path = Path(gt_dir) / f"{stem}.txt"
    if not txt_path.exists():
        return None
    content = txt_path.read_text(encoding="utf-8").strip()
    if not content:
        return []  # confirmed-negative image: zero ground-truth poles
    gts = []
    for line in content.splitlines():
        try:
            _, corners = yolo_obb_line_to_corners(line, img_w, img_h)
            gts.append({"bbox": obb_corners_to_xyxy(corners)})
        except Exception:
            continue
    return gts


def _run_one_config(
    img_path: Path, name: str, dataset_id: str,
    enable_geometry_qa: bool, qwen_gating: str,
    enable_obb_recovery: bool | None = None,
):
    boxes, raw = run_ai_pipeline(
        img_path=img_path, mode="AI_LABEL", use_sam_refinement=True,
        dataset_id=dataset_id, filename=name,
        enable_geometry_qa=enable_geometry_qa, qwen_gating=qwen_gating,
        enable_obb_recovery=enable_obb_recovery,
    )
    decision_counts: dict = defaultdict(int)
    for b in boxes:
        decision = b.attributes.get("decision") or b.attributes.get("category", "REVIEW")
        decision_counts[decision] += 1
    verification = raw.get("verification", {})
    recovery = raw.get("obb_recovery", {})
    return {
        "boxes": boxes,
        "raw": raw,
        "decision_counts": dict(decision_counts),
        "dino_raw_count": raw.get("dino_raw_count"),
        "sam3_raw_count": raw.get("sam3_raw_count"),
        "combined_deduped_count": raw.get("combined_deduped_count"),
        "n_sam_rejected": raw.get("n_sam_rejected"),
        "n_invalid_obb": raw.get("n_invalid_obb"),
        "n_semantic_rejected": verification.get("n_semantic_rejected"),
        "n_qwen_calls": verification.get("n_qwen_calls"),
        "n_disagreements": verification.get("n_disagreements"),
        "qwen_available": verification.get("qwen_available"),
        "final_count": len(boxes),
        "error": raw.get("error"),
        "recovery_ran": recovery.get("recovery_ran", False),
        "n_recovery_candidates": recovery.get("n_recovery_candidates", 0),
        "n_recovery_kept": recovery.get("n_recovery_kept", 0),
        "n_recovery_duplicates": recovery.get("n_recovery_duplicates", 0),
    }


def run_benchmark(
    image_paths: list[str],
    out_dir: str,
    dataset_id: str,
    compare_yolo: bool = False,
    categories: dict[str, str] | None = None,
    enable_geometry_qa: bool = True,
    qwen_gating: str = "gated",
    ground_truth_dir: str | None = None,
    enable_obb_recovery: bool | None = None,
    full_debug: bool = False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    categories = categories or {}

    report: dict = {
        "config": {"enable_geometry_qa": enable_geometry_qa, "qwen_gating": qwen_gating},
        "images": {}, "totals": defaultdict(int),
        "by_category_label": defaultdict(lambda: defaultdict(int)),
    }
    per_image_pr = []  # (precision, recall) when ground truth is available

    for img_path in image_paths:
        name = Path(img_path).name
        cat_label = categories.get(name, "uncategorized")

        result = _run_one_config(
            Path(img_path), name, dataset_id, enable_geometry_qa, qwen_gating,
            enable_obb_recovery=enable_obb_recovery,
        )
        boxes, raw = result["boxes"], result["raw"]

        img = cv2.imread(str(img_path))
        if img is None:
            report["images"][name] = {"error": f"could not read image at {img_path}"}
            continue
        img_h, img_w = img.shape[:2]

        if full_debug:
            # Debug visualization mode (spec): show raw DINO detections
            # underneath the final overlay, so it's visible by eye whether a
            # kept box came from a real DINO candidate or was only found by
            # recovery -- draw these first so final boxes render on top.
            for d in raw.get("dino", []):
                x0, y0, x1, y1 = [int(v) for v in d["xyxy"]]
                cv2.rectangle(img, (x0, y0), (x1, y1), DINO_RAW_COLOR, 1)
                cv2.putText(img, f"DINO raw {d['confidence']:.2f}", (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, DINO_RAW_COLOR, 1, cv2.LINE_AA)

        for b in boxes:
            if b.model_source == "OBB_RECOVERY":
                status = b.attributes.get("recovery_status", "RECOVERED_CANDIDATE")
                report["totals"][f"recovery_{status}"] += 1
                report["by_category_label"][cat_label][f"recovery_{status}"] += 1
                color = RECOVERY_COLOR.get(status, (255, 255, 255))
                pole_score = b.attributes.get("recovery_score", {}).get("pole_score", 0.0)
                draw_obb(img, b.corners, color, f"RECOVERY:{status} score={pole_score:.2f}")
                continue
            decision = b.attributes.get("decision") or b.attributes.get("category", "REVIEW")
            report["totals"][decision] += 1
            report["by_category_label"][cat_label][decision] += 1
            color = DECISION_COLOR.get(decision, CATEGORY_COLOR.get(decision, (255, 255, 255)))
            score = b.attributes.get("final_score", b.attributes.get("quality_score", 0))
            draw_obb(img, b.corners, color, f"{decision} s={score:.2f} c={b.confidence:.2f}")

        gt_precision = gt_recall = None
        if ground_truth_dir:
            gts = _load_ground_truth(ground_truth_dir, name, img_w, img_h)
            if gts is not None:
                preds = [{"bbox": b.xyxy, "confidence": b.confidence} for b in boxes]
                if gts:
                    gt_precision, gt_recall = DetectionEvaluator.compute_precision_recall(preds, gts)
                elif not preds:
                    gt_precision, gt_recall = 1.0, 1.0  # correctly predicted nothing on a negative image
                else:
                    gt_precision, gt_recall = 0.0, 1.0  # all predictions are false positives
                per_image_pr.append((gt_precision, gt_recall))
                for i, box_gt in enumerate(gts):
                    x0, y0, x1, y1 = [int(v) for v in box_gt["bbox"]]
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

        yolo_summary = None
        if compare_yolo:
            yolo_boxes, _ = run_ai_pipeline(img_path=Path(img_path), mode="YOLO_FAST", use_sam_refinement=False)
            yolo_summary = {"count": len(yolo_boxes), "confidences": [round(b.confidence, 3) for b in yolo_boxes]}
            for b in yolo_boxes:
                x0, y0, x1, y1 = [int(v) for v in b.xyxy]
                cv2.rectangle(img, (x0, y0), (x1, y1), YOLO_COMPARE_COLOR, 1)
                cv2.putText(img, f"YOLO(best.pt) {b.confidence:.2f}", (x0, max(12, y0 - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, YOLO_COMPARE_COLOR, 1, cv2.LINE_AA)

        cv2.imwrite(str(out / name), img)

        img_report = {
            "category_label": cat_label,
            "dino_raw_count": result["dino_raw_count"],
            "sam3_raw_count": result["sam3_raw_count"],
            "combined_deduped_count": result["combined_deduped_count"],
            "n_sam_rejected": result["n_sam_rejected"],
            "n_invalid_obb": result["n_invalid_obb"],
            "n_semantic_rejected": result["n_semantic_rejected"],
            "n_qwen_calls": result["n_qwen_calls"],
            "qwen_available": result["qwen_available"],
            "n_disagreements": result["n_disagreements"],
            "final_count": result["final_count"],
            "by_decision": result["decision_counts"],
            "ground_truth_precision": gt_precision,
            "ground_truth_recall": gt_recall,
            "recovery_ran": result["recovery_ran"],
            "n_recovery_candidates": result["n_recovery_candidates"],
            "n_recovery_kept": result["n_recovery_kept"],
            "n_recovery_duplicates": result["n_recovery_duplicates"],
            "error": raw.get("error"),
        }
        if yolo_summary is not None:
            img_report["yolo_benchmark_only"] = yolo_summary
        report["images"][name] = img_report

        print(
            f"{name} [{cat_label}]: dino_raw={result['dino_raw_count']} sam3_raw={result['sam3_raw_count']} "
            f"deduped={result['combined_deduped_count']} sam_rejected={result['n_sam_rejected']} "
            f"invalid_obb={result['n_invalid_obb']} semantic_rejected={result['n_semantic_rejected']} "
            f"final={result['final_count']} by_decision={result['decision_counts']}"
            + (f" P={gt_precision} R={gt_recall}" if gt_precision is not None else "")
            + (f" yolo_count={yolo_summary['count']}" if yolo_summary else "")
        )

    report["totals"] = dict(report["totals"])
    report["by_category_label"] = {k: dict(v) for k, v in report["by_category_label"].items()}
    if per_image_pr:
        report["ground_truth_summary"] = {
            "n_images_with_ground_truth": len(per_image_pr),
            "macro_precision": round(float(np.mean([p for p, _ in per_image_pr])), 4),
            "macro_recall": round(float(np.mean([r for _, r in per_image_pr])), 4),
            "note": "Macro-averaged per-image precision/recall at IoU>=0.5 (axis-aligned bbox), via src.evaluation.DetectionEvaluator.",
        }
    else:
        report["ground_truth_summary"] = {
            "note": "Not evaluated -- no --ground-truth-dir supplied. Counts above are candidate/decision "
                    "buckets only, not correctness against real ground truth."
        }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Totals across {len(image_paths)} images: {report['totals']}")
    if categories:
        print(f"By category label: {json.dumps(report['by_category_label'], indent=2)}")
    print(f"Ground truth: {report['ground_truth_summary']}")
    print(f"Overlays + report.json written to {out}/")
    print(
        "\nIMPORTANT: without --ground-truth-dir, this report counts candidates and\n"
        "decision buckets only. It does not know which candidates are true poles vs\n"
        "false positives, or which real poles were missed -- open the overlay images\n"
        "in this directory and judge each one by eye, or supply ground truth."
    )
    print(f"{'=' * 60}")
    return report


def run_compare_configs(
    image_paths: list[str], out_dir: str, dataset_id: str,
    categories: dict[str, str] | None = None, ground_truth_dir: str | None = None,
) -> dict:
    """Run all 4 spec-section-13 configurations on the same image set and
    report them side by side. Does NOT assume config D is better -- reports
    the same measured counts for all four so the caller can judge."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    categories = categories or {}

    comparison: dict = {"configs": CONFIG_LABELS, "per_config": {}, "per_image": {}}

    for cfg_key, preset in CONFIG_PRESETS.items():
        print(f"\n--- Config {cfg_key}: {CONFIG_LABELS[cfg_key]} ---")
        agg = defaultdict(int)
        pr_list = []
        for img_path in image_paths:
            name = Path(img_path).name
            result = _run_one_config(Path(img_path), name, f"{dataset_id}_{cfg_key}", **preset)
            for decision, n in result["decision_counts"].items():
                agg[decision] += n
            agg["n_invalid_obb"] += result["n_invalid_obb"] or 0
            agg["n_semantic_rejected"] += result["n_semantic_rejected"] or 0
            agg["n_qwen_calls"] += result["n_qwen_calls"] or 0
            agg["n_disagreements"] += result["n_disagreements"] or 0
            agg["n_errors"] += 1 if result["error"] else 0

            comparison["per_image"].setdefault(name, {})[cfg_key] = {
                "final_count": result["final_count"], "by_decision": result["decision_counts"],
                "n_disagreements": result["n_disagreements"], "error": result["error"],
            }

            if ground_truth_dir:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    gts = _load_ground_truth(ground_truth_dir, name, w, h)
                    if gts is not None:
                        preds = [{"bbox": b.xyxy, "confidence": b.confidence} for b in result["boxes"]]
                        if gts:
                            p, r = DetectionEvaluator.compute_precision_recall(preds, gts)
                        else:
                            p, r = (1.0, 1.0) if not preds else (0.0, 1.0)
                        pr_list.append((p, r))

        comparison["per_config"][cfg_key] = {
            "label": CONFIG_LABELS[cfg_key],
            "totals": dict(agg),
            "qwen_available": result["qwen_available"],
        }
        if pr_list:
            comparison["per_config"][cfg_key]["ground_truth"] = {
                "macro_precision": round(float(np.mean([p for p, _ in pr_list])), 4),
                "macro_recall": round(float(np.mean([r for _, r in pr_list])), 4),
                "n_images": len(pr_list),
            }
        print(json.dumps(comparison["per_config"][cfg_key], indent=2))

    (out / "compare_report.json").write_text(json.dumps(comparison, indent=2))
    print(f"\n{'=' * 60}\nComparison written to {out}/compare_report.json")
    if not ground_truth_dir:
        print(
            "No --ground-truth-dir was supplied: the table above shows candidate/decision\n"
            "counts only. Do NOT conclude any config is 'more accurate' from counts alone --\n"
            "more ACCEPTs is not evidence of correctness without ground truth to check against."
        )
    print(f"{'=' * 60}")
    return comparison


def run_recovery_comparison(
    image_paths: list[str], out_dir: str, dataset_id: str,
    categories: dict[str, str] | None = None, ground_truth_dir: str | None = None,
) -> dict:
    """
    Spec's dedicated OBB-recovery evaluation mode: A (DINO+SAM only,
    recovery off) vs B (DINO+SAM+OBB Recovery, recovery on) -- a DIFFERENT
    comparison axis than run_compare_configs' geometry_qa/qwen A/B/C/D
    presets above. Both A and B here hold geometry_qa=on, qwen=gated (the
    live app's production defaults) so the ONLY thing that differs is
    whether models/adapters/obb_recovery.py ran.

    Reports (spec section 'Evaluation'): pole recall, precision,
    false-positive rate, number of recovered true poles, number of
    recovered false positives, duplicate rate, OBB IoU against ground-truth
    labels, percentage of candidates requiring human review.

    Recall/precision/recovered-true/recovered-false and OBB-IoU are ONLY
    computed when --ground-truth-dir is supplied -- without it, only
    candidate/duplicate/review counts are reported, since (spec) 'the
    primary objective is to increase pole recall without causing a large
    increase in false positives' and neither can be measured without real
    ground truth to check against.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    categories = categories or {}

    presets = {"A_no_recovery": False, "B_with_recovery": True}
    labels = {"A_no_recovery": "DINO+SAM only", "B_with_recovery": "DINO+SAM+OBB Recovery"}

    comparison: dict = {"configs": labels, "per_config": {}, "per_image": {}}

    for cfg_key, recovery_on in presets.items():
        print(f"\n--- Recovery Config {cfg_key}: {labels[cfg_key]} ---")
        agg: dict = defaultdict(int)
        pr_list = []
        iou_list = []
        n_review_required = 0
        n_total_boxes = 0
        last_result = None

        for img_path in image_paths:
            name = Path(img_path).name
            result = _run_one_config(
                Path(img_path), name, f"{dataset_id}_{cfg_key}",
                enable_geometry_qa=True, qwen_gating="gated", enable_obb_recovery=recovery_on,
            )
            last_result = result
            boxes = result["boxes"]
            n_total_boxes += len(boxes)
            n_review_required += sum(1 for b in boxes if b.needs_review)
            agg["n_recovery_candidates"] += result["n_recovery_candidates"]
            agg["n_recovery_kept"] += result["n_recovery_kept"]
            agg["n_recovery_duplicates"] += result["n_recovery_duplicates"]
            agg["n_errors"] += 1 if result["error"] else 0

            recovered_true = recovered_false = None
            if ground_truth_dir:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    gts = _load_ground_truth(ground_truth_dir, name, w, h)
                    if gts is not None:
                        preds = [{"bbox": b.xyxy, "confidence": b.confidence} for b in boxes]
                        if gts:
                            p, r = DetectionEvaluator.compute_precision_recall(preds, gts)
                        else:
                            p, r = (1.0, 1.0) if not preds else (0.0, 1.0)
                        pr_list.append((p, r))

                        # Classify each RECOVERY-origin box as a true/false
                        # positive by greedy IoU>=0.5 matching against
                        # not-yet-matched ground-truth boxes.
                        recovery_boxes = [b for b in boxes if b.model_source == "OBB_RECOVERY"]
                        if recovery_boxes:
                            recovered_true, recovered_false = 0, 0
                            used_gt = [False] * len(gts)
                            for b in recovery_boxes:
                                best_iou, best_idx = 0.0, -1
                                for gi, g in enumerate(gts):
                                    if used_gt[gi]:
                                        continue
                                    iou = DetectionEvaluator.compute_iou(b.xyxy, g["bbox"])
                                    if iou > best_iou:
                                        best_iou, best_idx = iou, gi
                                if best_iou >= 0.5:
                                    recovered_true += 1
                                    used_gt[best_idx] = True
                                    iou_list.append(best_iou)
                                else:
                                    recovered_false += 1
                            agg["n_recovered_true"] += recovered_true
                            agg["n_recovered_false"] += recovered_false

            comparison["per_image"].setdefault(name, {})[cfg_key] = {
                "final_count": len(boxes),
                "n_recovery_kept": result["n_recovery_kept"],
                "n_recovery_duplicates": result["n_recovery_duplicates"],
                "recovered_true": recovered_true, "recovered_false": recovered_false,
            }

        n_kept = agg["n_recovery_kept"]
        n_dup = agg["n_recovery_duplicates"]
        duplicate_rate = (n_dup / (n_dup + n_kept)) if (n_dup + n_kept) > 0 else None
        review_pct = (n_review_required / n_total_boxes) if n_total_boxes else None

        comparison["per_config"][cfg_key] = {
            "label": labels[cfg_key],
            "totals": dict(agg),
            "duplicate_rate": round(duplicate_rate, 4) if duplicate_rate is not None else None,
            "pct_candidates_needing_review": round(review_pct, 4) if review_pct is not None else None,
            "qwen_available": last_result["qwen_available"] if last_result else None,
        }
        if pr_list:
            mean_p = float(np.mean([p for p, _ in pr_list]))
            comparison["per_config"][cfg_key]["ground_truth"] = {
                "macro_precision": round(mean_p, 4),
                "macro_recall": round(float(np.mean([r for _, r in pr_list])), 4),
                "false_positive_rate": round(1.0 - mean_p, 4),
                "n_images": len(pr_list),
            }
        if iou_list:
            comparison["per_config"][cfg_key]["recovered_obb_iou_vs_ground_truth"] = {
                "mean": round(float(np.mean(iou_list)), 4), "n_matched": len(iou_list),
            }
        print(json.dumps(comparison["per_config"][cfg_key], indent=2))

    (out / "recovery_compare_report.json").write_text(json.dumps(comparison, indent=2))

    print(f"\n{'=' * 60}\nRecovery comparison written to {out}/recovery_compare_report.json")
    a_gt = comparison["per_config"]["A_no_recovery"].get("ground_truth")
    b_gt = comparison["per_config"]["B_with_recovery"].get("ground_truth")
    if a_gt and b_gt:
        delta = round(b_gt["macro_recall"] - a_gt["macro_recall"], 4)
        verdict = "IMPROVED" if delta > 0 else ("NO CHANGE" if delta == 0 else "REGRESSED")
        print(f"Recall with recovery: {a_gt['macro_recall']} -> {b_gt['macro_recall']} (delta={delta:+.4f}, {verdict})")
        print(
            f"False-positive rate: {a_gt['false_positive_rate']} -> {b_gt['false_positive_rate']}\n"
            "Objective (spec): increase pole recall without causing a large increase in false "
            "positives -- judge BOTH numbers together, not recall alone."
        )
    else:
        print(
            "No --ground-truth-dir was supplied: candidate/duplicate/review counts only.\n"
            "Per the spec: 'Do not claim that the recovery stage improves performance until\n"
            "the evaluation demonstrates it.' Recall/precision/recovered-true-vs-false cannot\n"
            "be computed without real ground truth to check against."
        )
    print(f"{'=' * 60}")
    return comparison


def main():
    ap = argparse.ArgumentParser(description="Benchmark the DINO+SAM3->GeometryQA->Qwen->Decision production pipeline on a small image set")
    ap.add_argument("--images", help="Directory to sample images from")
    ap.add_argument("--files", nargs="+", help="Explicit image paths (overrides --images sampling)")
    ap.add_argument("--n", type=int, default=30, help="How many images to sample from --images (ignored with --files)")
    ap.add_argument("--manifest", help="Optional JSON {filename: category_label} for the stratified breakdown")
    ap.add_argument("--dataset-id", default="benchmark_run", help="Scratch dataset id used for mask storage under data/datasets/")
    ap.add_argument("--out", default="output/benchmark_30")
    ap.add_argument("--compare-yolo", action="store_true",
                     help="Also run YOLO_FAST (models/best.pt) on the same images as a benchmark-only overlay")
    ap.add_argument("--compare-configs", action="store_true",
                     help="Run all 4 verification-pipeline configs (A/B/C/D, spec section 13) and report side by side")
    ap.add_argument("--compare-recovery", action="store_true",
                     help="Run the OBB-recovery A/B evaluation: DINO+SAM only vs DINO+SAM+OBB Recovery, holding "
                          "geometry_qa/qwen at production defaults -- reports recall/precision/false-positive-rate/"
                          "recovered-true-vs-false/duplicate-rate/review-pct (best with --ground-truth-dir)")
    ap.add_argument("--enable-geometry-qa", dest="enable_geometry_qa", action="store_true", default=True)
    ap.add_argument("--disable-geometry-qa", dest="enable_geometry_qa", action="store_false")
    ap.add_argument("--qwen-gating", choices=["off", "gated", "always"], default="gated")
    ap.add_argument("--enable-obb-recovery", dest="enable_obb_recovery", action="store_true", default=None,
                     help="Force OBB recovery on for this single-config run, overriding configs/config.yaml")
    ap.add_argument("--disable-obb-recovery", dest="enable_obb_recovery", action="store_false",
                     help="Force OBB recovery off for this single-config run, overriding configs/config.yaml")
    ap.add_argument("--full-debug", action="store_true",
                     help="Debug visualization mode: also draw raw (pre-dedup) DINO boxes on the overlay, "
                          "alongside SAM-refined/recovered/final boxes, so it's visible by eye whether the "
                          "recovery stage is finding genuinely-missed poles or just creating false positives")
    ap.add_argument("--ground-truth-dir", help="Directory of human-verified YOLO-OBB .txt files (same stem as each image) for real precision/recall")
    args = ap.parse_args()

    if args.files:
        image_paths = args.files
    elif args.images:
        exts = ("*.jpg", "*.jpeg", "*.png")
        files: list[Path] = []
        for ext in exts:
            files.extend(sorted(Path(args.images).glob(ext)))
        files = sorted(set(files))
        step = max(1, len(files) // args.n) if args.n else 1
        image_paths = [str(p) for p in files[::step][:args.n]]
    else:
        raise SystemExit("Provide --images DIR (to sample from) or --files a.jpg b.jpg ...")

    if not image_paths:
        raise SystemExit("No images found to benchmark.")

    categories = None
    if args.manifest:
        categories = json.loads(Path(args.manifest).read_text())

    if args.compare_configs:
        run_compare_configs(image_paths, args.out, args.dataset_id, categories=categories, ground_truth_dir=args.ground_truth_dir)
    elif args.compare_recovery:
        run_recovery_comparison(image_paths, args.out, args.dataset_id, categories=categories, ground_truth_dir=args.ground_truth_dir)
    else:
        run_benchmark(
            image_paths, args.out, args.dataset_id, compare_yolo=args.compare_yolo, categories=categories,
            enable_geometry_qa=args.enable_geometry_qa, qwen_gating=args.qwen_gating,
            ground_truth_dir=args.ground_truth_dir,
            enable_obb_recovery=args.enable_obb_recovery, full_debug=args.full_debug,
        )


if __name__ == "__main__":
    main()
