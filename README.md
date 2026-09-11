# PoleAnnotator AI: Local Roboflow-Style OBB Annotation Platform

A local, high-performance AI annotation and auto-labeling platform specifically engineered for **electric utility pole detection and Oriented Bounding Box (OBB) labeling**.

---

## Quick Start

### 1. One-Command Launch

In your terminal or PowerShell inside the repository directory:

```bash
python run_backend.py
```

Or double-click:
```cmd
run_app.bat
```

The application will start the FastAPI server and automatically open your default browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

*(Interactive REST API documentation is available at `http://127.0.0.1:8000/docs`)*

---

## Key Workflows

### 1. Importing Images
1. Click the **Import Images** button (arrow-up icon next to the dataset dropdown).
2. Enter the local folder path containing your utility pole images (e.g. `eval_upload` or `C:\Users\dukar\OneDrive\Desktop\auto_annotation\img1`).
3. Click **Import Images**. Images are indexed into the active dataset.

### 2. Single-Image AI Labeling
- **AI Label** (production pipeline) — YOLO/`best.pt` is **never** used here:
  1. **Grounding DINO** proposes candidate boxes (open-vocabulary, text-prompted) — WHERE the pole might be. SAM 3's own candidate-proposal is available but off by default (`verification_pipeline.sam3.enable_candidate_proposal`) to avoid a redundant second SAM 3 call per candidate — see "Note on SAM 3" below.
  2. Candidates are deduplicated by IoU so one physical pole doesn't produce overlapping labels.
  3. The top candidates are segmented with **SAM 3** (falling back to SAM 2.1 only when SAM 3 itself is unavailable) into a precise mask, and a canonical 4-corner OBB is generated and structurally validated from it.
  4. **Geometry QA** scores mask/OBB shape plausibility (aspect ratio, orientation, fragmentation) — configurable thresholds, never hard-rejects on tilt or a frame-cut-off pole alone.
  5. **Qwen3-VL-8B-Instruct** (direct local Hugging Face Transformers, preferred; local Ollama or a cloud API key as fallbacks — see `verification_pipeline.qwen.backend`) semantically classifies the crop — is this actually a utility pole, what material/visibility/orientation — gated to skip candidates geometry QA already rejects.
  6. A **decision engine** combines all four signals (detection/segmentation/geometry/semantic) into one `final_score` and an `ACCEPT`/`REVIEW`/`REJECT` call, with disagreement between signals flagged explicitly. Only `ACCEPT`/`REVIEW` candidates reach you; `REJECT` is dropped before it's shown.
  See [REQUIREMENTS.md](REQUIREMENTS.md) section 2.2 and [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline.
- **Run All** (Model Comparison, benchmark-only): Runs YOLO (`best.pt`), Grounding DINO, and SAM concurrently and reconciles them, so you can toggle individual layers (`YOLO`, `DINO`, `SAM`) in the top floating bar. `best.pt`/`A_S.pt` are wired into this mode and `YOLO_FAST`/`YOLO_ONLY` **only** — never into AI Label.
- **Refine SAM (`S`)**: Select any bounding box and press `S` (or click *Refine SAM*) to tighten the box to the pole's vertical shaft using SAM 3 (or SAM 2.1 if SAM 3 is unavailable).
- Every candidate's evidence (decision, scores, Qwen classification, disagreements) is visible in the **Provenance & Evidence** panel and the **Labels** list once you select a box — filterable by Accept/Review/Reject/Disagreement/Geometry Warning.

### 3. Interactive Canvas & 4-Corner OBB Editing
- **Select / Move (`V`)**:
  - Drag inside box: Moves entire box.
  - Drag corners 0, 1, 2, 3: Adjusts individual corners (automatically maintains canonical top-left anchor and clockwise winding).
  - Drag top antenna handle: Rotates the OBB smoothly around its centroid.
- **Draw Box (`W`)**: Click and drag to create a new 4-corner OBB.
- **Zoom & Pan**: Mouse wheel zooms centered on your cursor; hold `Space` or middle-click and drag to pan; press `F` to fit image.
- **Undo / Redo** (`Ctrl+Z` / `Ctrl+Shift+Z` or `Ctrl+Y`): one step per committed action (add/delete/move/resize/rotate/corner-drag/clear-all) — a single drag is a single undo step, not one per pixel moved.
- On-canvas box color reflects the decision engine's call (green `ACCEPT`, yellow `REVIEW`, red `REJECT`) when available, or human-edited (blue).

### 4. Human Verification & Differential Learning
- Press **`Space`** or **`Enter`** (or click *Verify & Save*):
  - Saves verified label to `annotations/<stem>.txt` (standard YOLO-OBB format) and `annotations/<stem>.json`.
  - **Preserves raw AI predictions**: Untouched model outputs remain in `predictions/` and are **never** overwritten.
  - **Differential Diff**: Calculates adjustments between AI suggestions and your edits (added, modified, deleted) and archives them in `history/` for active learning.
  - Automatically advances to the next image!
- **Remove All Labels** clears every box on the *current image only* (not the dataset) — still requires **Save Edits** to persist. Unsaved changes prompt for confirmation before you navigate away or close the tab.

### 5. Batch Auto-Labeling Engine
1. Click **Batch Engine** in the top navigation.
2. Choose pipeline mode (*AI Label*, *Run All*, or *YOLO Only*).
3. For *AI Label*, optionally toggle **Run geometry QA** and set **Qwen semantic verification** (`Gated`/`Always`/`Off`) — `Gated` (default) skips Qwen only on candidates geometry QA already hard-rejects; `Always` calls it on every candidate (useful for A/B comparing configs, see `benchmark_dino_sam.py --compare-configs`).
4. Click **Start Job**.
5. The background async worker processes the dataset with real-time statistics (Processed, Accepted, Needs Review, Failed, and progress bar). If a required model becomes unavailable mid-run, the job aborts with a clear error instead of silently saving empty results for every remaining image.
6. Pause, Resume, or Cancel anytime.

### 6. Exporting YOLO-OBB Dataset
- Click **Export YOLO-OBB**:
  - Validates coordinate ranges $[0.0, 1.0]$.
  - Verifies signed shoelace area $> 0$ (clockwise).
  - Generates standard YOLO-OBB directory structure (`images/train`, `images/val`, `labels/train`, `labels/val`) and `data.yaml`.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| <kbd>Space</kbd> / <kbd>Enter</kbd> | Save verified annotation and advance to next image |
| <kbd>A</kbd> | Accept active detection |
| <kbd>R</kbd> | Reject active detection (false positive) |
| <kbd>S</kbd> | Refine active box with SAM segmentation |
| <kbd>D</kbd> / <kbd>Delete</kbd> / <kbd>Backspace</kbd> | Delete active box |
| <kbd>W</kbd> | Switch to Draw Box tool |
| <kbd>V</kbd> / <kbd>Esc</kbd> | Switch to Select / Edit tool |
| <kbd>N</kbd> / <kbd>&rarr;</kbd> | Next image |
| <kbd>P</kbd> / <kbd>&larr;</kbd> | Previous image |
| <kbd>Q</kbd> | Jump to next highest-priority difficult image |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> | Undo |
| <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd> | Redo |
| <kbd>F</kbd> | Fit zoom to screen |
| <kbd>+</kbd> / <kbd>-</kbd> | Zoom in / Zoom out |
| <kbd>?</kbd> | Open Keyboard Shortcuts Cheat Sheet |

---

## Hardware Support
- Real GPU/CPU auto-detection via `torch.cuda.is_available()`.
- Switch hardware at any time using the device dropdown in the top navigation bar (`AUTO`, `CUDA`, `CPU`).
- Clean fallback to CPU with zero crashes.

---

## Optional: Enabling Qwen Semantic Verification
The AI Label pipeline works fully without Qwen (geometry QA + the decision
engine still run) — Qwen adds an extra semantic-classification signal.
Three backends are tried in order via `verification_pipeline.qwen.backend`
(default `"auto"`), first one available wins:
1. **Direct Hugging Face Transformers** — Qwen3-VL-8B-Instruct loaded once
   from a local checkpoint (no Ollama/network needed). Set
   `verification_pipeline.qwen.transformers.model_path` (or `QWEN_MODEL_PATH`)
   to either a local checkpoint directory or a Hugging Face Hub id already
   cached locally (e.g. `Qwen/Qwen3-VL-8B-Instruct`). Requires a GPU for
   practical per-candidate latency.
2. **Local [Ollama](https://ollama.com)**:
   ```bash
   ollama pull qwen3-vl:2b   # or qwen3-vl:4b/8b -- a bigger model needs a GPU
   ```
   The app auto-discovers the smallest vision-capable `qwen*vl*` model
   you've pulled (pin one with `QWEN_OLLAMA_MODEL`). Without a GPU, an 8B+
   model can take 5+ minutes per candidate — start with `2b`/`4b` for local
   testing.
3. **DashScope cloud API** — set `QWEN_API_KEY`/`DASHSCOPE_API_KEY`.

Check `/api/models/status` (or the app's Model Status panel) to see which
backend, if any, was found — it now names the actual backend (`transformers`
/ `ollama` / `dashscope`) rather than a hard-coded label.

## Note on SAM 3
SAM 3 is the **preferred segmenter** (mask refinement), not a second
candidate proposer — DINO alone decides WHERE candidates are in production;
SAM 3 decides WHICH PIXELS belong to each one, with SAM 2.1 used only as a
fallback when SAM 3 itself isn't available. SAM 3's own open-vocabulary
detection *can* also propose candidates alongside DINO, but this is off by
default (`verification_pipeline.sam3.enable_candidate_proposal: false`) —
enabling it calls SAM 3 twice per SAM-3-proposed candidate (propose+segment,
then segment again during refinement), which is redundant inference; it
exists only for the experimental "DINO+SAM3 dual proposer" comparison path.
SAM 3 requires
`transformers>=5.9`; this repo currently pins `4.57.0` (see
[REQUIREMENTS.md](REQUIREMENTS.md)), so on an unmodified install SAM 3
reports `unavailable` and SAM 2.1 handles refinement instead — the pipeline
keeps working either way. If SAM 3's dependency needs conflict with this
app's own environment, it can run as an isolated HTTP service instead (see
`services/sam3_service.py`) via `verification_pipeline.sam3.backend: "http"`.
`/api/models/status` reports SAM 3's real status honestly rather than
failing silently.
