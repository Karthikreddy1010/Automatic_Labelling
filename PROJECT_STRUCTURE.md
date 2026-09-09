# Project Structure

Companion to [REQUIREMENTS.md](REQUIREMENTS.md) (what the system must do)
and [ARCHITECTURE.md](ARCHITECTURE.md) (how the backend/frontend fit
together). This document maps the actual repository layout.

The repo holds **two generations of work**: the active PoleAnnotator AI app
(sections 1–5 below), and an earlier CLI-based detection/relabeling pipeline
that PROJECT_SUMMARY.md describes and that predates it (section 6). They
share `src/`, `models/`, and some root-level scripts, so both are documented
here rather than pretending the older one doesn't exist.

## 1. PoleAnnotator AI application (active)

```
backend/
├── __init__.py
├── app.py              # FastAPI app: all REST endpoints, run_ai_pipeline()
│                        # (the production DINO+SAM3+GeometryQA+Qwen+Decision
│                        # pipeline), batch worker, model adapter instances
└── storage.py           # DatasetManager: dataset/image/annotation/prediction/
                         # mask/history CRUD, atomic+lock-guarded metadata,
                         # YOLO-OBB export

frontend/
├── index.html            # Single-page app shell
├── app.js                # All client logic: canvas rendering, box editing,
│                          # undo/redo, API calls, active learning UI, batch modal
└── style.css              # Roboflow-style dark theme

models/
├── best.pt, A_S.pt, yolo11n*.pt   # Trained/pretrained weights (benchmark-
│                                    # only detector paths -- never used by
│                                    # the production AI_LABEL pipeline)
└── adapters/
    ├── base.py               # DetectionBox/ModelInfo/HardwareInfo + abstract
    │                          # BaseDetector/BaseSegmenter/BaseVerifier
    ├── dino_adapter.py        # Grounding DINO (candidate proposer)
    ├── sam3_adapter.py        # SAM 3 (candidate proposer + segmenter;
    │                          # requires transformers>=5.9, not currently met)
    ├── sam21_adapter.py       # SAM 2.1 (box-prompted mask refinement)
    ├── yolo_adapter.py        # YOLOv8-OBB (benchmark-only detector)
    ├── qwen_adapter.py        # Qwen3-VL semantic verifier (local Ollama,
    │                          # preferred, or DashScope cloud API)
    ├── gemini_adapter.py      # Gemini VLM verifier (present, not wired into
    │                          # run_ai_pipeline -- unused today)
    ├── quality.py             # IoU dedup + heuristic quality_score
    ├── geometry_qa.py         # Mask/OBB shape-plausibility scoring
    ├── obb_generator.py       # OBB generation + structural validation
    ├── decision_engine.py     # Multi-signal ACCEPT/REVIEW/REJECT decision
    └── reconciliation.py      # Multi-model consensus (used by RUN_ALL mode)

configs/
└── config.yaml            # Project config incl. verification_pipeline
                            # section (geometry QA thresholds, decision
                            # weights, Qwen gating mode)

src/geometry_obb.py         # Deterministic canonical 4-corner OBB geometry:
                             # ordering, shoelace area, IoU, YOLO-OBB
                             # serialization -- shared by backend + benchmark

data/
├── config/
│   └── active_learning_weights.json    # Tunable AL priority weights
└── datasets/
    └── <dataset_id>/
        ├── metadata.json     # Per-image status/counters (lock-guarded)
        ├── images/           # Source images (never modified by the app)
        ├── predictions/      # Raw AI output per image (immutable)
        ├── annotations/      # Human-verified labels (.txt YOLO-OBB + .json)
        ├── masks/            # Cached SAM segmentation masks (.png)
        ├── history/          # AI-vs-human diff records (active learning)
        └── export/           # Generated YOLO-OBB train/val split + data.yaml
    (utility_poles_v1 is the only dataset currently present, and it's empty
    -- deliberately wiped mid-session pending new source images.)

tests/
├── test_backend.py, test_storage.py, test_frontend_mount.py     # Core API/storage
├── test_geometry.py, test_obb_generator.py                       # OBB geometry
├── test_phase3_dino_sam.py, test_phase4_yolo.py, test_phase5_sam.py,
│   test_phase6_dino_comparison.py, test_phase6_qwen.py             # Per-model adapters
├── test_geometry_qa.py, test_decision_engine.py                   # New verification pipeline
├── test_active_learning_enhanced.py, test_phase7_active_learning.py  # Active learning
├── test_adapters.py                                               # Adapter base contracts
└── test_phase13_e2e_acceptance.py                                # 10-image E2E (needs real images)

benchmark_dino_sam.py       # Small-image-set harness for the production
                             # pipeline: single-config runs, --compare-configs
                             # (the spec's A/B/C/D verification configs side
                             # by side), --ground-truth-dir for real P/R
run_backend.py               # One-command launcher (FastAPI + static frontend)
run_app.bat, run_app.ps1     # Windows launch shortcuts for run_backend.py
```

## 2. Documentation

```
README.md              # Quick start + key workflows (STALE: still describes
                        # the pre-this-session YOLO-first/DINO-fallback
                        # architecture; see REQUIREMENTS.md's header)
ARCHITECTURE.md         # System topology + OBB geometry guarantees (partially
                        # stale: predates SAM3/GeometryQA/Qwen/DecisionEngine)
REQUIREMENTS.md          # Functional/non-functional requirements (current)
PROJECT_STRUCTURE.md     # This file
PROJECT_SUMMARY.md       # Describes the OLDER pipeline (section 6) --
                         # self-flagged at the top as predating this app
ONBOARDING.md            # (not reviewed as part of this pass)
LICENSE
wiki/                    # Home.md, Roadmap.md, Model-History.md,
                         # Evaluation-Methodology.md, Active-Learning.md,
                         # Wire-Detection.md, Review-Tool.md -- documentation
                         # for the older pipeline (section 6), not the app
```

## 3. Configuration & dependency files

```
requirements.txt    # Pinned to versions actually installed/verified in the
                     # current dev environment (see REQUIREMENTS.md section 5)
.gitignore           # Note: contains a blanket **/*.md rule -- new markdown
                      # files (like this one) need `git add -f` to be tracked
```

## 4. Evaluation / QA artifacts

```
eval_upload/
├── montage.jpg, montage_hightilt.jpg   # The two composite test images used
│                                        # throughout this session's live
│                                        # testing (no other local images
│                                        # currently exist to test against)
├── labels/                              # 2789 auto-labeled GSV images
│                                        # (older pipeline output)
├── predictions.json
├── compute.py, score_and_post.py        # Older pipeline's Fly.io upload/
                                          # scoring scripts (review_tool/)

output/
├── results/    # Evaluation reports, dashboards, CSVs -- mixed current/older
└── (benchmark_dino_sam.py writes here by default: output/benchmark_30/,
    output/verify_compare/, etc. -- not currently present, created on run)
```

## 5. Root-level app entry points

| File | Purpose |
|---|---|
| `run_backend.py` | Starts the FastAPI server + opens the browser |
| `pipeline.py` | CLI pipeline runner (`run_evaluation()` etc., shared with the older pipeline) |

## 6. Older pipeline (pre-dates PoleAnnotator AI, still present)

Per [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): a separate, earlier GSV →
detect → segment → measure → CSV pipeline (YOLO26x + SAM 2.1 for attribute
extraction, hosted review tool on Fly.io). Shares `src/` and `models/` with
the app above but is otherwise independent — none of it is imported by
`backend/app.py`.

```
_gen_candidates.py, detect_sam.py, draw_disputed.py, eval_detector.py,
eval_hybrid.py, evaluate_seg.py, extract_attributes.py, hybrid_poles.py,
mask_bench.py, pole_env.py, pole_sam.py, predict.py, predict_sam3.py,
refine_labels.py, sam3_poles.py, score_gold.py, train.py, train_detector.py,
train_seg.py, run_det_smoke.sh, run_train.sh
    # Standalone CLI scripts for the older detect/train/eval workflow.
    # pole_sam.py's _clean()/_pole_score() ARE reused by the current app's
    # models/adapters/sam21_adapter.py, sam3_adapter.py, and quality.py --
    # not fully independent of section 1.

relabel/     # Auto-relabeling pipeline (candidate generation, LLM-judge
             # calibration, versioned review batches v6/v7) for the older
             # dataset -- gen_candidates.py, judge.py, judge_calibrate*.py

review_tool/  # Standalone Flask/Fly.io app for human review of the older
              # pipeline's auto-labels -- separate deployment from
              # PoleAnnotator AI's own FastAPI app

Final_Dataset_allpole/, Final_Dataset_refined/   # Training data splits
    # (train/val/test with labels + labels_qwen3 variants) for the older
    # detector, plus their data.yaml files

src/  (the non-geometry_obb.py modules)
├── attribute_extraction.py    # Height/tilt/width from mask + camera geometry
├── auto_annotate.py
├── data_preprocessing.py
├── evaluation.py               # DetectionEvaluator -- reused by
│                                # benchmark_dino_sam.py's --ground-truth-dir
├── gsv_collector.py            # Google Street View image collection
├── huggingface_models.py
└── model_comparison.py
```
