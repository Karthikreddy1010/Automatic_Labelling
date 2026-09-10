# Architecture Overview — PoleAnnotator AI

> See also [REQUIREMENTS.md](REQUIREMENTS.md) for what each stage is
> required to do and why, and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
> for the full repo layout.

## 1. System Topology

```
+-------------------------------------------------------------------+
|                        FRONTEND SPA (HTML5/Canvas)                |
|  - High-DPI Interactive Canvas (Pan, Zoom, 4-Corner OBB Drag)     |
|  - Model Comparison Layer Toggles (YOLO, Grounding DINO, SAM)     |
|  - Provenance panel: decision/scores/Qwen classification/         |
|    disagreements per candidate; Labels list w/ decision filters   |
|  - Differential History & Review Triage Inspector                 |
|  - Batch Auto-Labeling Engine Modal (geometry QA / Qwen gating)   |
+---------------------------------+---------------------------------+
                                  | REST / Static Mount
+---------------------------------v---------------------------------+
|                        FASTAPI BACKEND (backend/app.py)           |
|  - /api/system/hardware         | /api/models/status              |
|  - /api/datasets                | /api/datasets/{id}/images       |
|  - /api/inference/detect        | /api/inference/segment_box      |
|  - /api/batch/start|pause|resume|cancel                           |
|  - /api/datasets/{id}/export    | /api/config/verification_pipeline|
+---------------------------------+---------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
+--------v-------------------------------+  +---------------v-----------------+
|          MODELS & ADAPTERS             |  |     STORAGE & REPOSITORY        |
| Candidate proposers:                   |  | storage.py (DatasetManager)     |
|  - dino_adapter.py (Grounding DINO)    |  | - images/                       |
|  - sam3_adapter.py (SAM 3, also        |  | - annotations/ (verified)       |
|    proposes; requires transformers>=5.9)| | - predictions/ (raw untouched)  |
| Segmentation refinement:               |  | - masks/ (cached SAM masks)     |
|  - sam21_adapter.py (SAM 2.1)          |  | - history/ (differential diffs) |
| Verification (post-segmentation):      |  | - export/ (YOLO-OBB dataset)    |
|  - geometry_qa.py, obb_generator.py    |  +----------------------------------+
|  - qwen_adapter.py (Qwen3-VL, local
|    Ollama preferred / DashScope cloud)
|  - decision_engine.py (ACCEPT/REVIEW/
|    REJECT)
| Candidate reduction:
|  - quality.py (IoU dedup, quality_score)
| Benchmark-only:
|  - yolo_adapter.py (best.pt/A_S.pt),
|    reconciliation.py (RUN_ALL consensus)
| Unused (present, not wired into
| run_ai_pipeline):
|  - gemini_adapter.py
| Shared geometry:
|  - src/geometry_obb.py
+-----------------------------------------+
```

## 2. Production AI Pipeline (`backend/app.py::run_ai_pipeline`, mode `AI_LABEL`/`DINO_SAM`)

`best.pt`/`A_S.pt` (YOLOv8-OBB) are **never called** on this path — they are
wired only into the explicitly benchmark-only modes (`YOLO_FAST`,
`YOLO_ONLY`, `RUN_ALL`). This replaced an earlier YOLO-first-with-DINO-
fallback design.

```
image
  │
  ├─► Grounding DINO ──┐
  │  (open-vocab, text- │
  │   prompted candidates)
  │                     ├─► dedup_by_iou() ──► up to SAM_REFINE_TOP_N (5)
  ├─► SAM 3             │    (quality.py)      highest-confidence candidates
  │  (open-vocab detect+┘
  │   segment; also
  │   proposes boxes)
  │
  ▼
SAM 3 (or SAM 2.1 fallback) segments each ──► score_pole_quality()
  candidate into a precise mask               (quality.py) — REJECT-bucketed
  (SAM 3 preferred; SAM 2.1 used only          candidates dropped here
   when SAM 3 itself is unavailable)
  ▼
mask → canonical 4-corner OBB ──► obb_generator.validate_obb()
  (src/geometry_obb.py)             (in-bounds, non-self-intersecting,
  │                                  positive area — invalid OBBs dropped)
  ▼
run_geometry_qa() (geometry_qa.py) ──► geometry_score, geometry_status
  configurable thresholds               (pass/warning/fail)
  (configs/config.yaml
   verification_pipeline.geometry_qa)
  │
  ▼
Qwen gating (off/gated/always) ──► qwen_adapter.verify() when it runs
  skips the call when geometry         (class, material, visibility,
  already hard-failed (unless           orientation, semantic_confidence)
  "always")
  │
  ▼
decision_engine.evaluate_candidate() ──► final_score (weighted combination,
  (configs/config.yaml                    renormalized over whichever
   verification_pipeline.decision)        signals actually ran) + decision
  │                                        (ACCEPT/REVIEW/REJECT) +
  │                                        disagreements[]
  ▼
REJECT dropped · ACCEPT/REVIEW persisted to predictions/, shown to reviewer
```

Two independent signals feed the decision engine's `segmentation_score` and
`detection_score` deliberately without double-counting: `segmentation_score`
reuses `quality.py`'s underlying mask-geometry/overlap sub-signals (not its
already-DINO-blended `quality_score`), so DINO confidence isn't counted
twice.

### Configuration (`configs/config.yaml`, section `verification_pipeline`)
- `qwen.gating`: `off` | `gated` (default) | `always`.
- `geometry_qa.*`: aspect ratio, tilt, fill-ratio, solidity, image-area-
  fraction thresholds — all configurable, no single hard-coded global value.
- `decision.weights`: `detection`/`semantic`/`geometry`/`segmentation`
  (default `0.20`/`0.35`/`0.25`/`0.20`), `accept_threshold` (`0.65`),
  `review_threshold` (`0.35`) — documented as starting values, not claimed-
  optimal; retune once real precision/recall data exists.
- Loaded once at backend startup with dataclass defaults if the file/section
  is missing or partial — never crashes on absent config.

## 3. Deterministic 4-Corner OBB Geometry (`src/geometry_obb.py`)

- **Image Coordinates**: $(0, 0)$ top-left, $+X$ right, $+Y$ down.
- **Signed Shoelace Area**:
  $$2 A = \sum_{i=0}^3 (x_i y_{i+1} - x_{i+1} y_i) > 0$$
  Guarantees strictly **clockwise winding** in screen coordinates.
- **Canonical Top-Left Anchor (Corner 0)**:
  Vertex minimizing $x + y$ (tie-breaking on $\min y$ then $\min x$).
- **Permutation Invariance**:
  Any input order of the 4 vertices yields the exact same canonical sequence.
- **Serialization Invariance**:
  `corners -> line -> corners -> line` produces identical coordinates.

## 4. Storage Hierarchy & Differential Active Learning

- `predictions/<stem>.json`: Preserves raw model outputs (including each
  candidate's `attributes`: `decision`, `final_score`, `geometry_score`,
  `geometry_status`, `qwen_class`/`qwen_material`/`qwen_visibility`/
  `qwen_semantic_confidence` when Qwen ran, `disagreements`); **never
  overwritten** by human edits.
- `annotations/<stem>.txt`: Verified annotations in standard YOLO-OBB format.
- `annotations/<stem>.json`: Verified annotations in JSON format with metadata.
- `masks/<stem>_<idx>.png`: Cached SAM segmentation masks, referenced from
  `predictions/<stem>.json`'s `attributes.mask_path`.
- `history/<stem>_<timestamp>.json`: Differential learning records logging adjustments between AI suggestions and human edits (IoU, modified, added, deleted).
- `export/`: Generated YOLO-OBB `images/{train,val}` + `labels/{train,val}` +
  `data.yaml`, regenerated on each export from `annotations/` only (never
  from `predictions/` directly — a human must have verified it first).

All metadata reads/writes are serialized per-dataset (`threading.RLock`) and
written atomically (temp file + `os.replace`); user-supplied path components
are sanitized before any filesystem use (`backend/storage.py`).
