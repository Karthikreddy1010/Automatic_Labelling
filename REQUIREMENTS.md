# PoleAnnotator AI — Requirements Specification

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (how the system is built) and
> [README.md](README.md) (how to run it). This document states what the system
> is required to do and the constraints it must operate under. It describes
> the **current, actual** implementation — not an aspirational target — so it
> should be updated whenever a requirement changes, not treated as frozen.
>
> **Note:** [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
> have been updated to match the pipeline described in section 2.2 below.
> [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) describes a separate, older
> pipeline (see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) section 6) and
> is intentionally left as-is.

## 1. Purpose

PoleAnnotator AI is a local, single-machine annotation platform for building
an oriented-bounding-box (OBB) training dataset of electric utility poles
from street-level imagery. It combines AI-assisted candidate generation with
mandatory human review, and exports a standard YOLO-OBB dataset for training
a detector. It is explicitly **not** a multi-tenant SaaS platform, a model
training service, or a dataset versioning/deployment product — those are out
of scope (see section 6).

## 2. Functional Requirements

### 2.1 Dataset Management
- Create, list, and retrieve datasets, each with its own class list (default:
  `["utility_pole"]`).
- Import images into a dataset from a local file or directory path.
- List a dataset's images, filterable by status (`unlabeled`, `ai_suggested`,
  `needs_review`, `verified`, etc.).
- Track per-image status, confidence, annotation count, and `needs_review`
  in dataset metadata (`backend/storage.py::DatasetManager`).
- Mark/unmark an image as part of a fixed, held-out test set, excluded from
  training export.
- Detect near-duplicate images (perceptual hash) for dataset hygiene.

### 2.2 AI-Assisted Detection & Verification Pipeline (production)
Given an image, the system MUST produce candidate OBBs via the following
pipeline (`backend/app.py::run_ai_pipeline`, modes `AI_LABEL`/`DINO_SAM`):

1. **Candidate proposal**: Grounding DINO (open-vocabulary text-prompted
   detection) and, if available, SAM 3 (joint open-vocabulary detect+segment)
   each propose candidate boxes independently.
2. **Deduplication**: candidates are merged by IoU clustering
   (`models/adapters/quality.py::dedup_by_iou`) so one physical pole does not
   produce multiple overlapping candidates.
3. **Segmentation refinement**: up to `SAM_REFINE_TOP_N` (5) highest-confidence
   deduped candidates are segmented with SAM 2.1 (or SAM 3 as fallback) into a
   precise mask, from which a canonical 4-corner OBB is generated.
4. **OBB structural validation** (`models/adapters/obb_generator.py`): every
   generated OBB MUST have exactly 4 corners, all in-bounds, non-self-
   intersecting, positive area — invalid OBBs are dropped, never persisted.
5. **Quality scoring** (`models/adapters/quality.py::score_pole_quality`):
   a heuristic 0–1 `quality_score` blending DINO confidence, mask geometry,
   and box/mask overlap; candidates scoring in the REJECT bucket are dropped
   before reaching the reviewer.
6. **Geometry QA** (`models/adapters/geometry_qa.py`): independently scores
   mask/OBB shape plausibility (aspect ratio, orientation, fill ratio,
   fragmentation, image-area fraction) into `geometry_score` +
   `geometry_status` (`pass`/`warning`/`fail`). Configurable thresholds
   (`configs/config.yaml`, `verification_pipeline.geometry_qa`) — no single
   hard-coded global threshold.
7. **Semantic verification** (`models/adapters/qwen_adapter.py`): a Qwen
   vision-language model (local Ollama, preferred, or DashScope cloud API)
   classifies the candidate crop (`class`, `material`, `visibility`,
   `orientation`, `semantic_confidence`) and is gated (`off`/`gated`/`always`,
   configurable per-request or via `verification_pipeline.qwen.gating`) to
   avoid calling it on candidates an earlier stage already confidently
   rejects.
8. **Decision engine** (`models/adapters/decision_engine.py`): combines
   detection/segmentation/geometry/semantic scores (configurable weights,
   `verification_pipeline.decision.weights`) into one `final_score` and an
   `ACCEPT` / `REVIEW` / `REJECT` decision, with explicit cross-signal
   disagreement detection. `REJECT` candidates are dropped; `ACCEPT`/`REVIEW`
   are persisted to `predictions/` and shown to the reviewer, flagged
   `needs_review` unless `ACCEPT`.

`DINO_ONLY` mode exposes step 1–2 output without refinement, for inspection.

### 2.3 Benchmark-Only Detection Modes
- `YOLO_FAST` / `YOLO_ONLY`: the trained YOLOv8-OBB detector
  (`models/best.pt`), for benchmarking/comparison only.
- `RUN_ALL`: runs YOLO + DINO + SAM and reconciles all three
  (`models/adapters/reconciliation.py`) for side-by-side model comparison.
- These modes MUST NOT be reachable from the production `AI_LABEL`/`DINO_SAM`
  code path (see constraint 3.3).

### 2.4 Human Review & Annotation Editing
- Interactive canvas: pan/zoom, 4-corner OBB drag (move/resize per-corner/
  rotate), draw-new-box, delete, undo/redo (full-snapshot history, one step
  per committed action).
- Per-candidate inspector showing source, confidence, decision, final/
  geometry/segmentation scores, Qwen classification (when it ran), and
  review reasons.
- Accept (verify as-is), Save Edits (persist human corrections), Reject
  (mark false positive, with failure-type/negative-category), Remove All
  Labels (current image only — clears locally, requires explicit Save to
  persist).
- Client-side validation before every save: reject degenerate/malformed
  boxes, clamp out-of-frame corners to image bounds rather than silently
  rejecting a legitimately frame-cut-off pole.
- Unsaved-changes protection: navigating away or closing the tab with
  pending edits prompts for confirmation.
- Coordinates are stored and edited in original-image pixel space throughout
  (never canvas/display pixels), normalized only at YOLO-OBB export/save
  time.

### 2.5 Active Learning & Review Prioritization
- A configurable-weight priority queue surfaces the most valuable images for
  human review (disagreement, low confidence, poor geometry, unusual
  aspect/occlusion/orientation, multiple/overlapping poles, near-boundary
  poles, etc.).
- "Review Next Difficult" jumps directly to the highest-priority unreviewed
  image.
- Hard cases exportable to a separate review directory.

### 2.6 Export
- Verified (`verified`/`accepted`/`human_corrected`) non-test-set images
  export to a standard YOLO-OBB dataset (`images/{train,val}`,
  `labels/{train,val}`, `data.yaml`), with an 80/20 train/val split
  (seeded, reproducible).
- Confirmed-negative images export as empty label files (standard YOLO
  background-image convention).
- Export-time validation re-checks coordinate range and polygon winding;
  invalid lines are logged, not silently included.
- The held-out test set is always excluded from export.

### 2.7 Batch Auto-Labeling
- Start/pause/resume/cancel an async batch job over some or all unlabeled
  images in a dataset, with live progress (processed/accepted/needs-review/
  failed counts, average confidence).
- If a required model becomes unavailable mid-batch, the job MUST abort
  immediately with a clear `error_message` — not silently save empty
  "no poles found" predictions for every remaining image.
- Same `enable_geometry_qa`/`qwen_gating`/`use_sam_refinement` controls as
  single-image detection.

### 2.8 System & Model Status
- Hardware inspection (CPU/CUDA, device switching AUTO/CUDA/CPU) without
  fabricating GPU telemetry when CUDA is unavailable.
- Per-model status (`ready`/`not_loaded`/`unavailable`/`missing_weights`)
  with a human-readable installation guide when unavailable — never a mocked
  "ready" state.
- Read-only view of the loaded verification-pipeline configuration
  (`/api/config/verification_pipeline`).

## 3. Non-Functional Requirements & Constraints

### 3.1 Data Safety & Integrity
- Raw AI predictions (`predictions/`) are immutable once written — human
  edits only ever produce a separate `annotations/` record.
- No endpoint or batch operation deletes source images or trained model
  weights.
- No destructive dataset-wide operation (e.g. bulk-clearing all labels
  across a dataset) exists without an explicit, separate human confirmation
  step — a per-image "Remove All Labels" is the only bulk-clear implemented
  today (dataset-wide clearing was explicitly scoped out).
- Concurrent metadata writes are serialized per-dataset (`threading.RLock`)
  and written atomically (temp file + `os.replace`).
- All user-supplied path components are sanitized before filesystem use
  (path traversal prevention).

### 3.2 No Silent Fallback
- If a required model for the requested mode is unavailable (e.g. DINO or
  SAM down for `AI_LABEL`), the pipeline MUST return a clear
  `raw_outputs["error"]` (surfaced as HTTP 503) — never silently substitute
  another detector or return an empty "no poles found" result.
- A malformed or unparseable Qwen response is normalized to safe defaults
  (`class: "uncertain"`, etc.) and never crashes the pipeline, but is never
  treated as a confident judgment either.

### 3.3 Model Usage Constraints
- `models/best.pt` / `A_S.pt` (the trained YOLO-OBB detector) MUST NOT be
  used by the production `AI_LABEL`/`DINO_SAM`/`DINO_ONLY` pipeline. They are
  wired only into the explicitly benchmark-only modes (`YOLO_FAST`,
  `YOLO_ONLY`, `RUN_ALL`).
- No single verification signal (Qwen, geometry, DINO confidence) may
  unilaterally REJECT a candidate — Qwen requires both a confident
  (≥ configurable threshold) *and* non-"uncertain" classification to drive a
  REJECT; geometry only hard-fails on genuinely implausible geometry
  (near-empty mask, near-full-image OBB, extreme tilt), never on tilt or
  boundary-touching alone.
- `severely_occluded`/`too_distant` Qwen visibility calls are capped at
  `REVIEW` and can never auto-`ACCEPT`, regardless of the blended score.
- The final OBB geometry MUST always originate from the SAM segmentation
  mask (or, failing that, an explicitly-labeled axis-aligned fallback) —
  never from Qwen or any other verifier.
- Batch runs must not process the full image set without being explicitly
  invoked to do so; day-to-day development/testing must use small,
  explicitly-scoped image sets.

### 3.4 Configurability
- Geometry QA thresholds, decision-engine weights/thresholds, and Qwen
  gating mode live in `configs/config.yaml` under `verification_pipeline`,
  not hard-coded — loaded with safe dataclass defaults if the file/section is
  missing or partial.
- Decision weights are documented as starting values, not claimed-optimal;
  they should be retuned once real precision/recall data exists.

### 3.5 Transparency of Heuristic Signals
- `quality_score`, `geometry_score`, `segmentation_score`, and
  `semantic_confidence` MUST be labeled and treated as heuristic signals for
  prioritizing review — never presented as calibrated probabilities.
- Evaluation reports (`src/evaluation.py`, `pipeline.py`,
  `benchmark_dino_sam.py`) MUST print "Not evaluated — no ground truth
  supplied" rather than fabricate precision/recall/height/tilt metrics when
  no ground truth is available.

### 3.6 Deployment Compatibility
- The frontend MUST resolve its own API base from the script's own URL
  (`new URL('.', document.currentScript.src)`), not a hard-coded absolute
  `/api/...` path, so it continues to work when served from a sub-path (a
  JupyterHub `/proxy/` or VS Code Server `/vscode/proxy/` prefix), not only
  at domain root.
- No change may require installing new Python packages, upgrading a pinned
  major dependency version (e.g. `transformers` 4.x → 5.x), reinstalling
  PyTorch, or creating a new environment without explicit user approval
  first — see section 6 for the current SAM3/`transformers` gap this
  constraint is blocking.

### 3.7 Testing
- New pipeline logic (geometry QA, decision engine, OBB validation, Qwen
  response normalization) MUST have unit tests that don't require GPU/model
  weights, alongside integration tests that exercise real model inference
  where available.
- A test must not encode a fixed assumption about optional-model
  availability (e.g. "Qwen is always unavailable in this environment") when
  that state is legitimately environment-dependent — assert on valid states,
  not one fixed expected state.

## 4. Data & Storage Requirements

Per dataset, under `data/datasets/<dataset_id>/`:

| Path | Contents | Mutability |
|---|---|---|
| `images/` | Source images | Never modified by the app |
| `predictions/<stem>.json` | Raw AI pipeline output, all candidates + `raw_outputs` | Immutable once written |
| `annotations/<stem>.txt` | Verified YOLO-OBB labels (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized [0,1]) | Human-write only |
| `annotations/<stem>.json` | Verified annotations + metadata | Human-write only |
| `masks/<stem>_<idx>.png` | Cached SAM segmentation masks | Written by pipeline, read-only after |
| `history/<stem>_<timestamp>.json` | Differential record (AI→human diff) | Append-only |
| `export/` | Generated YOLO-OBB train/val dataset + `data.yaml` | Regenerated on export |
| `metadata.json` | Per-image status/counters | Lock-guarded, atomic writes |

YOLO-OBB label format (`class_id x1 y1 x2 y2 x3 y3 x4 y4`): 4 corners,
canonically ordered clockwise from top-left, normalized to `[0, 1]` —
guaranteed by `src/geometry_obb.py`'s deterministic ordering (permutation-
invariant, positive signed shoelace area).

## 5. External Dependencies & Environment Requirements

- Python 3.13, PyTorch, `transformers` (pinned `==4.57.0`; see section 6 for
  the SAM3 gap this creates), FastAPI/uvicorn, OpenCV, Shapely (used by
  `src/geometry_obb.py` and `models/adapters/obb_generator.py`), PyYAML.
  `requirements.txt` pins exact versions verified working in the current
  dev environment rather than loose lower bounds; `tensorflow`,
  `pycocotools`, `gsvmaps`, and `label-studio-converter` were removed from it
  as confirmed-unused leftovers from an earlier pipeline.
- Grounding DINO, SAM 2.1: require `transformers` and, for first download,
  `POLE_ALLOW_ONLINE=1`.
- SAM 3: requires `transformers>=5.9` (`Sam3Model`/`Sam3Processor`) — **not
  met by the currently pinned/installed version**; the adapter degrades
  gracefully (`is_available()` returns `False`, reported honestly via
  `/api/models/status`) rather than crashing.
- Qwen verification: either a local Ollama server (preferred — auto-
  discovers the smallest vision-capable `qwen*vl*` model pulled, or a
  `QWEN_OLLAMA_MODEL`-pinned one) or `QWEN_API_KEY`/`DASHSCOPE_API_KEY` for
  the DashScope cloud API. CPU-only inference on an 8B+ VLM is impractically
  slow (empirically 5+ minutes per crop) — a GPU (e.g. the DeepThink H200
  target) or a smaller local model (2B/4B) is required for practical
  interactive use; `QWEN_OLLAMA_TIMEOUT_S` (default 300s) is tunable per
  environment.
- No hardware GPU is required for the app to run — every stage degrades to
  CPU or reports `unavailable`/`error` rather than requiring CUDA.

## 6. Out of Scope / Known Gaps

- **Dataset-wide "clear all labels"**: explicitly not built (scoped out by
  the user); only per-image clearing exists.
- **True precision/recall**: not computable without human-verified ground
  truth for the images in question. `benchmark_dino_sam.py --ground-truth-
  dir` is implemented and ready, but no such ground-truth set currently
  exists locally.
- **SAM3 on this machine**: blocked on a `transformers` major-version
  upgrade (4.x → 5.x) that has not been approved (constraint 3.6). Until
  then, SAM3 contributes zero candidates and DINO alone proposes them.
- **Qwen semantic-verification quality**: integration verified working
  end-to-end against a real local Ollama model, but the smallest (2B) local
  model's actual judgment quality is unverified/suspect (observed near-
  identical templated responses across different crops in manual testing) —
  not yet validated as a trustworthy signal on its own.
- **Multi-tenancy, dataset versioning, model training/deployment UI**: not
  part of this system; it produces a YOLO-OBB dataset for training
  elsewhere.
- **Gemini verifier** (`models/adapters/gemini_adapter.py`): present in the
  codebase and exposed via `/api/models/status`, but not wired into
  `run_ai_pipeline` — optional/unused today.

## 7. Glossary

| Term | Meaning |
|---|---|
| `quality_score` | Heuristic 0–1 blend of DINO confidence + SAM mask geometry + box/mask overlap (`quality.py`). Pre-filters obvious false positives before the decision engine runs. |
| `geometry_score` / `geometry_status` | Independent mask/OBB shape-plausibility signal (`geometry_qa.py`). `pass`/`warning`/`fail`. |
| `segmentation_score` | SAM-mask-quality signal used by the decision engine, deliberately independent of DINO confidence (reuses `quality.py`'s geometry/overlap sub-signals, not its DINO-blended `quality_score`). |
| `semantic_confidence` | Qwen's self-reported confidence in its classification of a candidate crop. |
| `final_score` | Decision engine's weighted combination of the above (weights renormalized over whichever signals actually ran). |
| `decision` | `ACCEPT` / `REVIEW` / `REJECT` — the decision engine's output; advisory, always human-overridable via the normal edit/Accept/Reject flow. |
| `needs_review` | Per-candidate flag; `True` unless `decision == ACCEPT` (or, for pre-decision-engine modes, unless quality/confidence clears their own bar). |
