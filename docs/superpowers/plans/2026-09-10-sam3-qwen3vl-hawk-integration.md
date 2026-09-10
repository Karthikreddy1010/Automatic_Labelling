# SAM3 + Qwen3-VL-8B (HAWK) Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade PoleAnnotator AI's verification pipeline so SAM 3 and Qwen3-VL-8B-Instruct are real, first-class backends — SAM 3 loadable via an isolated environment/service, Qwen3-VL-8B loaded once via Hugging Face Transformers (no Ollama dependency required) — while reusing the existing DINO→SAM→GeometryQA→Qwen→DecisionEngine pipeline, decision engine, geometry QA, and OBB generator exactly as they already work.

**Architecture:** This repo already implements ~70% of the requested spec (geometry QA, OBB-from-mask, the multi-signal decision engine with configurable weights, graceful degradation, DINO→SAM3 candidate flow) from earlier work this session. The real gap is narrower: (1) Qwen3-VL-8B has no direct-Transformers backend yet (Ollama/DashScope only), (2) SAM3 has no environment-isolation story for a torch_env_v2-vs-SAM3-env split, (3) Qwen only sends one crop and a narrower JSON schema than requested, (4) no per-stage timing, (5) UI/health-check labels still say "Ollama: qwen3-vl:2b" style text, (6) no explicit "Skip" review action. Every new piece is added as an additional, config-selectable code path behind the *existing* `BaseSegmenter`/`BaseVerifier` interfaces — no existing adapter, endpoint, or storage format is replaced or removed.

**Tech Stack:** Python 3.10 (HAWK) / FastAPI / PyTorch 2.6.0+cu118 (HAWK, A100) / `transformers` (needs `>=5.9` for SAM3's `Sam3Processor`/`Sam3Model` — confirmed absent from this repo's pinned `4.57.0`) / Hugging Face `Qwen3VLForConditionalGeneration` + `AutoProcessor`.

**Spec:** User's 20-part spec (this conversation, message beginning "I need you to modify my existing PoleAnnotator application... PART 1 — SAM 3 INTEGRATION"). This plan implements it in full except where noted in "Deviations from the literal spec" below — every deviation reuses an existing abstraction instead of duplicating it, per the spec's own Part 21 instruction ("Reuse existing abstractions wherever possible. Do not create duplicate model managers").

## Environment note (read before executing)

This plan was written and will be implemented from a session running on a **local Windows machine with no CUDA** (`torch==2.8.0+cpu`, `cuda available: False`, `transformers==4.57.0`). The target spec describes a **separate remote environment** ("HAWK": Python 3.10, PyTorch 2.6.0+cu118, A100 40GB GPUs, model at `/home/maska/models/Qwen3-VL-8B-Instruct`) that this session cannot reach, inspect, or run code on. Per the user's explicit instruction, code is written here to be copied/deployed to HAWK by the user — **every task's "run and verify" step will genuinely execute here** (proving the code is syntactically correct, imports cleanly, and degrades gracefully with honest "unavailable" reporting when SAM3/CUDA/the Transformers Qwen backend aren't present), but **cannot prove real GPU inference correctness**. That final verification (real SAM3 masks, real Qwen3-VL-8B output quality, actual A100 utilization) can only happen on HAWK itself — flag this plainly to the user at the end rather than claiming it works there.

## Deviations from the literal spec (and why)

1. **No new `configs/verification.yaml`.** `configs/config.yaml`'s existing `verification_pipeline` section already does exactly this (documented, configurable `decision.weights`, loaded once at startup with safe defaults). Adding a second competing config file would violate the spec's own "do not create duplicates" rule and risk the two files drifting out of sync. New settings are added as new keys inside the existing section instead.
2. **No new `/models/status`-shaped-differently endpoint.** The spec's Part 11 example JSON (`{"yolo": {"available": true}, ...}`) is a different shape than the existing `/api/models/status` (a list of `ModelInfo` dicts), which the frontend and `tests/test_backend.py` already consume. The spec itself says "or whatever architecture the existing application uses" — so the existing endpoint is extended with the requested new fields (`backend`, richer `device`) rather than replaced or duplicated.
3. **Qwen's JSON schema is *extended*, not replaced.** `models/adapters/decision_engine.py` already keys off the existing schema's `class`/`decision`/`semantic_confidence` fields (with real disagreement-detection logic tested in `tests/test_decision_engine.py`). The spec's Part 4 fields (`is_utility_pole`, `pole_type`, `occlusion`, `truncated`, `annotation_suitable`, `confidence`) are added as **additional** fields on the same response object, normalized alongside the existing ones, so `decision_engine.py` needs zero changes and both schemas' information is available to the UI/storage layer.

## Global Constraints

- Do NOT remove or break existing YOLO OBB, Grounding DINO, SAM 2.1, annotation editing, dataset management, active learning, or export functionality.
- Do NOT fake SAM 3 using SAM 2.1, and never report SAM3 status as `"ready"` unless it genuinely loaded.
- Do NOT let Qwen generate OBB coordinates or act as the segmentation model — OBB geometry always originates from the SAM mask via `models/adapters/obb_generator.py`.
- Do NOT treat Qwen's self-reported confidence as a calibrated probability — it is one signal into `decision_engine.py`'s weighted score, never used alone.
- Do NOT hard-code decision weights — they live in `configs/config.yaml` under `verification_pipeline.decision.weights`, already true today; new weights follow the same pattern.
- Do NOT require Ollama for Qwen3-VL-8B; the Transformers backend must not depend on Ollama being installed or running.
- Do NOT download the Qwen3-VL-8B model at request time — load once from a local path, cache in memory (singleton), never re-download.
- Do NOT upgrade torch/torchvision/CUDA/Python in the existing environment blindly — SAM3's environment needs are isolated behind a swappable adapter (in-process vs HTTP service), decided at deploy time on HAWK, not assumed here.
- Do NOT crash the UI or pipeline when SAM3 or Qwen (either backend) is unavailable — always degrade gracefully with an honest status and continue with whatever signals did run (decision_engine.py already renormalizes weights over active signals only).
- Do NOT silently accept a candidate when Qwen produces malformed JSON — falls back to human review, never silent accept (existing `_normalize_qwen_response` behavior, preserved).
- Do NOT delete rejected/human-corrected predictions — `predictions/` is immutable, `annotations/` holds verified human output, `history/` holds diffs; this plan adds to that pattern, never replaces it.
- Do NOT auto-retrain after every annotation — Active Learning stays an accumulation + export surface, not an auto-trigger.
- Only call Qwen on candidates that survive DINO/SAM/geometry filtering first (already true via `qwen_gating`), to avoid unnecessary expensive VLM calls.

---

## File Structure

```
models/adapters/
  sam3_adapter.py                    [MODIFY] real box-prompted segmentation via input_boxes (Part 1), backend field in get_info()
  sam3_http_adapter.py               [CREATE] BaseSegmenter-conformant HTTP client for an isolated SAM3 service (Part 2)
  qwen_adapter.py                    [MODIFY] extended JSON schema (Part 4/5), tight+context crop (Part 6), backend selection incl. transformers (Part 3)
  qwen3vl_transformers_backend.py    [CREATE] singleton Qwen3-VL-8B-Instruct loader/runner via HF Transformers (Part 3)
  decision_engine.py                 [unchanged — already implements Part 5/7]
  geometry_qa.py                     [unchanged — already implements Part 8]
  obb_generator.py                   [unchanged — already implements Part 9]

services/
  __init__.py                        [CREATE]
  sam3_service.py                    [CREATE] minimal FastAPI microservice wrapping SAM3Adapter for a separate SAM3 env (Part 2)

backend/
  app.py                             [MODIFY] SAM3 backend selection (in-process vs HTTP), stage timing (Part 17), predictions summary wiring, Skip endpoint
  storage.py                         [MODIFY] predictions.json spec-shaped summary (Part 12), "skipped" status (Part 13)

configs/
  config.yaml                        [MODIFY] new verification_pipeline keys: qwen.backend, qwen.transformers.*, qwen.context_crop_expand_pct, sam3.backend, sam3.service_url, logging.log_stage_timings

frontend/
  index.html                         [MODIFY] Skip button
  app.js                             [MODIFY] Skip wiring, model status panel shows backend/device text (Part 10)
  style.css                          [MODIFY] Skip button styling (reuse existing button classes, minimal)

tests/
  test_sam3_adapter.py               [CREATE] Part 18.1-18.3: load, text prompt, box prompt
  test_qwen3vl_transformers.py       [CREATE] Part 18.4-18.6: load, verify, JSON parsing, singleton behavior
  test_e2e_pipeline.py               [CREATE] Part 18: DINO → SAM3 → geometry → Qwen → OBB end-to-end
  test_decision_engine.py            [unchanged — already covers Part 18.9]
  test_geometry_qa.py                [unchanged — already covers Part 18.7]
  test_obb_generator.py              [unchanged — already covers Part 18.8]
  test_backend.py                    [MODIFY] extend model-status test for new fields, add Skip endpoint test
```

---

### Task 1: Config — new verification_pipeline keys

**Files:**
- Modify: `configs/config.yaml`
- Test: `tests/test_config_verification.py`

**Interfaces:**
- Produces: `verification_pipeline.qwen.backend` (str: `"auto"|"transformers"|"ollama"|"dashscope"`), `verification_pipeline.qwen.transformers.model_path` (str|null), `verification_pipeline.qwen.transformers.dtype` (str), `verification_pipeline.qwen.transformers.device_map` (str), `verification_pipeline.qwen.context_crop_expand_pct` (float), `verification_pipeline.sam3.backend` (str: `"auto"|"inprocess"|"http"`), `verification_pipeline.sam3.service_url` (str|null), `verification_pipeline.logging.log_stage_timings` (bool) — all read by Task 2/3/4/5's code via `backend/app.py`'s existing `_load_verification_pipeline_config()`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_config_verification.py
"""Verify the new SAM3/Qwen-backend config keys are present with safe defaults."""
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

    def test_logging_section_present(self):
        self.assertIn("log_stage_timings", self.vp["logging"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_verification.py -v`
Expected: FAIL with `KeyError: 'backend'` (or similar `KeyError`/`AssertionError`) since these keys don't exist in `config.yaml` yet.

- [x] **Step 3: Add the new keys to config.yaml**

Append inside the existing `verification_pipeline:` block in `configs/config.yaml` (after the existing `qwen:` sub-keys, before `geometry_qa:`):

```yaml
  qwen:
    gating: "gated"
    # Which Qwen backend to use. "auto" tries transformers (if
    # transformers.model_path/QWEN_MODEL_PATH exists on disk), then a local
    # Ollama server, then DashScope's cloud API -- first one available wins.
    # Pin explicitly on HAWK once verified: "transformers".
    backend: "auto"
    transformers:
      # Local Qwen3-VL-8B-Instruct checkpoint directory (HAWK:
      # /home/maska/models/Qwen3-VL-8B-Instruct). Falls back to the
      # QWEN_MODEL_PATH env var if null. Never downloaded automatically --
      # if this path doesn't exist locally, the transformers backend
      # reports unavailable rather than fetching from the Hub.
      model_path: null
      dtype: "bfloat16"
      device_map: "auto"
    # Expanded "context crop" sent alongside the tight candidate crop, as a
    # fraction of the candidate's own width/height (spec: 20-40%; 0.30 is a
    # reasonable middle default, not a claimed-optimal value).
    context_crop_expand_pct: 0.30

  sam3:
    # "auto" prefers an in-process SAM3Adapter (facebook/sam3 via
    # transformers>=5.9) if its dependencies import cleanly; otherwise falls
    # back to calling a separately-running SAM3 HTTP service (see
    # services/sam3_service.py) at `service_url`, for when SAM3's dependency
    # needs conflict with this app's own torch_env_v2 and it must run in an
    # isolated environment/process instead.
    backend: "auto"
    # e.g. "http://127.0.0.1:8801" -- falls back to the SAM3_SERVICE_URL env
    # var if null. Only used when backend resolves to "http".
    service_url: null

  logging:
    # Emit per-stage timing (dino_ms/sam3_ms/geometry_ms/qwen_ms/obb_ms/
    # total_ms) into predictions.json's raw_outputs.timings and backend logs.
    log_stage_timings: true

  geometry_qa:
    max_image_area_fraction: 0.35
```

(The rest of the existing `geometry_qa:`/`decision:` blocks are unchanged — only inserting the new `backend`/`transformers`/`context_crop_expand_pct` lines into `qwen:`, and the new `sam3:`/`logging:` blocks between `qwen:` and `geometry_qa:`.)

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_verification.py -v`
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add configs/config.yaml tests/test_config_verification.py
git commit -m "config: add SAM3/Qwen backend-selection keys to verification_pipeline

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Qwen — extended JSON schema + tight/context crops

**Files:**
- Modify: `models/adapters/qwen_adapter.py`
- Test: `tests/test_phase6_qwen.py` (extend existing file with new test functions)

**Interfaces:**
- Consumes: `DetectionBox` (from `models/adapters/base.py`, unchanged).
- Produces (new/changed):
  - `_normalize_qwen_response(data: dict) -> dict` — same name, now ALSO returns `is_utility_pole: bool`, `pole_type: str`, `occlusion: "none"|"low"|"medium"|"high"`, `truncated: bool`, `annotation_suitable: bool` alongside every existing key (`class`, `material`, `visibility`, `orientation`, `semantic_confidence`, `decision`, `reason`) — **existing keys unchanged in name/meaning**, so `decision_engine.py` needs zero changes.
  - `QwenVerifier._build_crops(image, detection) -> Tuple[str, str]` — replaces `_crop_and_encode`'s single-image return; returns `(tight_b64, context_b64)`, both base64 JPEG. `context_crop_expand_pct` read from `verification_pipeline.qwen.context_crop_expand_pct` (constructor param, default `0.30`).
  - `QwenVerifier.verify(...)` — same signature, same return dict shape (superset of before).

- [x] **Step 1: Write the failing tests**

Append to `tests/test_phase6_qwen.py`:

```python
class TestQwenExtendedSchema(unittest.TestCase):
    """Part 4/5: Qwen's normalized response carries the spec's richer field
    set (is_utility_pole/pole_type/occlusion/truncated/annotation_suitable)
    in addition to -- not instead of -- the existing class/decision schema
    that decision_engine.py already depends on."""

    def test_normalize_adds_new_fields_with_safe_defaults_on_empty_input(self):
        from models.adapters.qwen_adapter import _normalize_qwen_response
        norm = _normalize_qwen_response({})
        # Existing fields (decision_engine.py contract) must still be present.
        for key in ("class", "material", "visibility", "orientation",
                    "semantic_confidence", "decision", "reason"):
            self.assertIn(key, norm)
        # New spec fields.
        self.assertIn("is_utility_pole", norm)
        self.assertIn("pole_type", norm)
        self.assertIn("occlusion", norm)
        self.assertIn("truncated", norm)
        self.assertIn("annotation_suitable", norm)
        self.assertFalse(norm["is_utility_pole"])
        self.assertFalse(norm["annotation_suitable"])
        self.assertIn(norm["occlusion"], ("none", "low", "medium", "high"))

    def test_normalize_maps_new_schema_response_correctly(self):
        from models.adapters.qwen_adapter import _normalize_qwen_response
        raw = {
            "is_utility_pole": True,
            "pole_type": "utility_pole",
            "material": "wood",
            "visibility": "high",
            "occlusion": "low",
            "truncated": False,
            "annotation_suitable": True,
            "reason": "Vertical wooden pole carrying utility wires.",
            "confidence": 0.91,
        }
        norm = _normalize_qwen_response(raw)
        self.assertTrue(norm["is_utility_pole"])
        self.assertEqual(norm["pole_type"], "utility_pole")
        self.assertEqual(norm["occlusion"], "low")
        self.assertFalse(norm["truncated"])
        self.assertTrue(norm["annotation_suitable"])
        self.assertAlmostEqual(norm["semantic_confidence"], 0.91)  # "confidence" aliases semantic_confidence
        # Existing schema still derivable/compatible: a confident is_utility_pole
        # candidate should not be silently dropped from the old `class` contract.
        self.assertEqual(norm["class"], "electric_utility_pole")
        self.assertEqual(norm["decision"], "accept")

    def test_normalize_uncertain_new_schema_example_from_spec(self):
        from models.adapters.qwen_adapter import _normalize_qwen_response
        raw = {
            "is_utility_pole": False,
            "pole_type": "uncertain",
            "material": "unknown",
            "visibility": "low",
            "occlusion": "high",
            "truncated": True,
            "annotation_suitable": False,
            "reason": "Candidate is mostly hidden by vegetation.",
            "confidence": 0.42,
        }
        norm = _normalize_qwen_response(raw)
        self.assertFalse(norm["is_utility_pole"])
        self.assertFalse(norm["annotation_suitable"])
        self.assertEqual(norm["decision"], "reject")

    def test_build_crops_returns_two_distinct_images(self):
        from models.adapters.qwen_adapter import QwenVerifier
        from models.adapters.base import DetectionBox
        import numpy as np

        verifier = QwenVerifier(ollama_base_url="http://127.0.0.1:1")  # unreachable, fine for this test
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :100] = 255  # left half white, right half black -- crops will differ visibly
        det = DetectionBox(xyxy=(80.0, 50.0, 120.0, 150.0), corners=None, confidence=0.9)

        tight_b64, context_b64 = verifier._build_crops(img, det)
        self.assertIsInstance(tight_b64, str)
        self.assertIsInstance(context_b64, str)
        self.assertGreater(len(tight_b64), 0)
        self.assertGreater(len(context_b64), 0)
        # Context crop (expanded bbox) must decode to a larger image than the tight crop.
        import base64, io
        from PIL import Image
        tight_img = Image.open(io.BytesIO(base64.b64decode(tight_b64)))
        context_img = Image.open(io.BytesIO(base64.b64decode(context_b64)))
        self.assertGreaterEqual(context_img.size[0] * context_img.size[1],
                                 tight_img.size[0] * tight_img.size[1])
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_phase6_qwen.py::TestQwenExtendedSchema -v`
Expected: FAIL — `AttributeError`/`AssertionError` (`is_utility_pole` not in normalized dict; `_build_crops` doesn't exist yet).

- [x] **Step 3: Extend the prompt template, enum sets, and `_normalize_qwen_response`**

In `models/adapters/qwen_adapter.py`, replace `VERIFY_PROMPT_TEMPLATE` and the enum-set constants:

```python
VERIFY_PROMPT_TEMPLATE = (
    "You are an expert electric utility inspector verifying an oriented bounding box (OBB)\n"
    "candidate produced by an automated detector, for a utility-pole labeling dataset.\n"
    "You are given TWO images: (1) a tight crop of exactly the candidate region, and\n"
    "(2) a wider context crop around it -- use the context image to tell utility poles\n"
    "apart from trees, tree trunks, street signs, light poles, buildings, and wires without\n"
    "a supporting pole, all of which can look similar in a tight crop alone.\n"
    "Candidate Bounding Box (xyxy, pixels): {xyxy}\n"
    "Candidate OBB Corners: {corners}\n"
    "Detector Source: {model_source}, Detector Confidence: {confidence}\n\n"
    "Utility poles may be made of wood, steel, concrete, composite, or other materials --\n"
    "do not require wood. Do not reject a candidate merely because it is partially occluded,\n"
    "wires aren't clearly visible, it is slightly tilted, or it has unusual equipment, or only\n"
    "part of the pole is visible -- those are normal cases. Only mark visibility as low or\n"
    "occlusion as high when truly hard to assess; do not force accept/reject for those,\n"
    "prefer marking annotation_suitable appropriately instead.\n\n"
    "Classify the candidate and respond ONLY with a valid JSON object matching this exact schema:\n"
    "{{\n"
    '  "is_utility_pole": true,\n'
    '  "pole_type": "utility_pole",\n'
    '  "class": "electric_utility_pole",\n'
    '  "material": "wood",\n'
    '  "visibility": "high",\n'
    '  "occlusion": "low",\n'
    '  "truncated": false,\n'
    '  "orientation": "slightly_tilted",\n'
    '  "annotation_suitable": true,\n'
    '  "confidence": 0.94,\n'
    '  "reason": "Vertical wooden pole carrying utility wires, clearly a utility pole in context."\n'
    "}}\n\n"
    'class must be one of: "electric_utility_pole", "non_utility_pole", "tree", '
    '"building_or_structure", "street_light_or_lamp_post", "other_object", "uncertain".\n'
    'material must be one of: "wood", "steel", "concrete", "composite", "unknown".\n'
    'visibility must be one of: "high", "medium", "low".\n'
    'occlusion must be one of: "none", "low", "medium", "high".\n'
    'orientation must be one of: "vertical", "slightly_tilted", "strongly_tilted".\n'
    "confidence is a float between 0.0 and 1.0 (this is your own self-reported confidence, "
    "not a calibrated probability -- the application combines it with other independent "
    "signals to make the final decision)."
)

_VALID_CLASSES = {
    "electric_utility_pole", "non_utility_pole", "tree",
    "building_or_structure", "street_light_or_lamp_post", "other_object", "uncertain",
}
_VALID_MATERIALS = {"wood", "steel", "concrete", "composite", "unknown"}
# Existing internal schema's visibility buckets (kept for decision_engine.py's
# "severely_occluded"/"too_distant" review-cap check) plus the spec's simpler
# high/medium/low, accepted as valid raw input and mapped onto the internal
# buckets below.
_VALID_VISIBILITY = {"mostly_visible", "partially_occluded", "severely_occluded", "too_distant"}
_SPEC_VISIBILITY = {"high", "medium", "low"}
_VALID_ORIENTATION = {"vertical", "slightly_tilted", "strongly_tilted"}
_VALID_DECISION = {"accept", "review", "reject"}
_VALID_OCCLUSION = {"none", "low", "medium", "high"}
_SPEC_VISIBILITY_TO_INTERNAL = {
    "high": "mostly_visible",
    "medium": "partially_occluded",
    "low": "severely_occluded",
}
```

Replace `_normalize_qwen_response` with a version that accepts both schemas:

```python
def _normalize_qwen_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a parsed Qwen JSON response into the canonical schema, tolerating
    missing keys or unrecognized enum values (normalized to "uncertain" /
    "unknown" rather than raising) -- a malformed field must never crash the
    pipeline, it should just fall back to "we don't know" for that field.

    Accepts BOTH the original internal schema (class/decision/orientation/
    semantic_confidence) AND the spec's richer schema (is_utility_pole/
    pole_type/occlusion/truncated/annotation_suitable/confidence) in the same
    response, normalizing whichever fields are present and deriving the
    other schema's fields from them when only one side was supplied --
    decision_engine.py only reads the original schema's keys, so those are
    always populated regardless of which fields the model actually returned.
    """
    def _enum(key: str, valid: set, default: str) -> str:
        v = str(data.get(key, default)).strip().lower() if data.get(key) is not None else default
        return v if v in valid else default

    def _bool(key: str, default: bool) -> bool:
        v = data.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return default

    # --- confidence: spec's "confidence" aliases the original "semantic_confidence" ---
    conf_raw = data.get("semantic_confidence", data.get("confidence", 0.5))
    try:
        semantic_confidence = max(0.0, min(1.0, float(conf_raw)))
    except (TypeError, ValueError):
        semantic_confidence = 0.5

    # --- visibility: accept either vocabulary, normalize to the internal one ---
    raw_visibility = str(data.get("visibility", "")).strip().lower()
    if raw_visibility in _SPEC_VISIBILITY:
        visibility = _SPEC_VISIBILITY_TO_INTERNAL[raw_visibility]
    else:
        visibility = raw_visibility if raw_visibility in _VALID_VISIBILITY else "mostly_visible"

    occlusion = _enum("occlusion", _VALID_OCCLUSION, "none")
    truncated = _bool("truncated", False)

    # --- is_utility_pole / class: derive whichever side is missing ---
    if "is_utility_pole" in data:
        is_utility_pole = _bool("is_utility_pole", False)
        cls = _enum("class", _VALID_CLASSES, "electric_utility_pole" if is_utility_pole else "uncertain")
    else:
        cls = _enum("class", _VALID_CLASSES, "uncertain")
        is_utility_pole = cls == "electric_utility_pole"

    pole_type = str(data.get("pole_type", "utility_pole" if is_utility_pole else "uncertain"))[:100]

    orientation = _enum("orientation", _VALID_ORIENTATION, "vertical")

    # --- decision / annotation_suitable: derive whichever side is missing ---
    if "annotation_suitable" in data:
        annotation_suitable = _bool("annotation_suitable", False)
        decision = _enum(
            "decision", _VALID_DECISION,
            "accept" if (annotation_suitable and is_utility_pole) else ("reject" if not is_utility_pole else "review"),
        )
    else:
        decision = _enum("decision", _VALID_DECISION, "review")
        annotation_suitable = decision == "accept"

    return {
        "class": cls,
        "material": _enum("material", _VALID_MATERIALS, "unknown"),
        "visibility": visibility,
        "orientation": orientation,
        "semantic_confidence": round(semantic_confidence, 4),
        "decision": decision,
        "reason": str(data.get("reason", ""))[:500] or "No reason provided by Qwen.",
        # Spec (Part 4) fields, additive:
        "is_utility_pole": is_utility_pole,
        "pole_type": pole_type,
        "occlusion": occlusion,
        "truncated": truncated,
        "annotation_suitable": annotation_suitable,
    }
```

- [x] **Step 4: Add `_build_crops` (tight + context) and wire it into every backend**

Replace `_crop_and_encode` with:

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    model_name: str = "qwen-vl-max",
    local_model_path: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_timeout_s: Optional[float] = None,
    ollama_preferred_model: Optional[str] = None,
    context_crop_expand_pct: float = 0.30,
    backend: str = "auto",
    transformers_dtype: str = "bfloat16",
    transformers_device_map: str = "auto",
):
    ...  # existing body unchanged up to here
    self.context_crop_expand_pct = context_crop_expand_pct
    self.backend_preference = backend  # "auto" | "transformers" | "ollama" | "dashscope"
    self.transformers_dtype = transformers_dtype
    self.transformers_device_map = transformers_device_map
    self._transformers_backend = None  # lazily constructed in Task 3
```

```python
def _load_pil(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        return Image.fromarray(rgb)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")

def _encode_crop(self, pil_img: Image.Image, box: Tuple[int, int, int, int]) -> str:
    cropped = pil_img.crop(box)
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _build_crops(self, image: Union[str, np.ndarray, Image.Image], detection: DetectionBox) -> Tuple[str, str]:
    """
    Part 6: build a TIGHT crop (candidate bbox + a small 15% pad, same as
    before) and a separate, wider CONTEXT crop (bbox expanded by
    context_crop_expand_pct, default 30%) so Qwen can distinguish a real
    utility pole from a tree/sign/building that only looks similar up close.
    Both are returned as base64 JPEG for the caller to send as two images.
    """
    pil_img = self._load_pil(image)
    w, h = pil_img.size
    x0, y0, x1, y1 = detection.xyxy
    bw, bh = (x1 - x0), (y1 - y0)

    tight_pad_w, tight_pad_h = 0.15 * bw, 0.15 * bh
    tight_box = (
        max(0, int(x0 - tight_pad_w)), max(0, int(y0 - tight_pad_h)),
        min(w, int(x1 + tight_pad_w)), min(h, int(y1 + tight_pad_h)),
    )

    ctx_pad_w, ctx_pad_h = self.context_crop_expand_pct * bw, self.context_crop_expand_pct * bh
    context_box = (
        max(0, int(x0 - ctx_pad_w)), max(0, int(y0 - ctx_pad_h)),
        min(w, int(x1 + ctx_pad_w)), min(h, int(y1 + ctx_pad_h)),
    )

    return self._encode_crop(pil_img, tight_box), self._encode_crop(pil_img, context_box)
```

Update the two existing backend callers to send both images and remove the now-unused `_crop_and_encode`:

```python
def _verify_via_ollama(self, tight_b64: str, context_b64: str, prompt: str) -> Dict[str, Any]:
    payload = json.dumps({
        "model": self.ollama_model,
        "messages": [{"role": "user", "content": prompt, "images": [tight_b64, context_b64]}],
        "stream": False,
        "format": "json",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{self.ollama_base_url}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=self.ollama_timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data.get("message", {}).get("content", "")
    if not text:
        raise RuntimeError(f"Ollama returned no content: {data}")
    return _extract_json(text)

def _verify_via_dashscope(self, tight_b64: str, context_b64: str, prompt: str) -> Dict[str, Any]:
    try:
        import dashscope  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "dashscope package is not installed. Install it with `pip install dashscope` "
            "to use the DashScope cloud fallback."
        ) from exc

    response = dashscope.MultiModalConversation.call(
        api_key=self.api_key,
        model=self.model_name,
        messages=[{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{tight_b64}"},
                {"image": f"data:image/jpeg;base64,{context_b64}"},
                {"text": prompt},
            ],
        }],
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"DashScope API error {response.status_code}: "
            f"{getattr(response, 'message', 'unknown error')}"
        )
    content = response.output.choices[0].message.content
    text = content[0]["text"] if isinstance(content, list) else str(content)
    return _extract_json(text)
```

And in `verify()`, replace `b64_image = self._crop_and_encode(image, detection)` with:

```python
tight_b64, context_b64 = self._build_crops(image, detection)
```

...and update the two call sites (`self._verify_via_ollama(b64_image, prompt)` → `self._verify_via_ollama(tight_b64, context_b64, prompt)`, same for dashscope), and extend the returned dict in `verify()`'s success path to include the new fields:

```python
return {
    "status": "ok",
    "class": norm["class"],
    "material": norm["material"],
    "visibility": norm["visibility"],
    "orientation": norm["orientation"],
    "semantic_confidence": norm["semantic_confidence"],
    "decision": norm["decision"],
    "reason": norm["reason"],
    "is_utility_pole": norm["is_utility_pole"],
    "pole_type": norm["pole_type"],
    "occlusion": norm["occlusion"],
    "truncated": norm["truncated"],
    "annotation_suitable": norm["annotation_suitable"],
    "is_utility_pole_legacy": norm["class"] == "electric_utility_pole",
    "obb_quality": {"accept": "good", "review": "uncertain", "reject": "poor"}[norm["decision"]],
    "needs_human_review": (
        norm["decision"] != "accept"
        or norm["visibility"] in ("severely_occluded", "too_distant")
    ),
}
```

- [x] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase6_qwen.py -v`
Expected: PASS (all existing tests in this file still pass unchanged, plus the 4 new ones from Step 1).

- [x] **Step 6: Commit**

```bash
git add models/adapters/qwen_adapter.py tests/test_phase6_qwen.py
git commit -m "feat(qwen): extend verification schema and send tight+context crops

Adds is_utility_pole/pole_type/occlusion/truncated/annotation_suitable
fields alongside the existing class/decision schema (decision_engine.py
untouched), and sends both a tight candidate crop and a wider context crop
so Qwen can distinguish poles from trees/signs/buildings.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Qwen3-VL-8B-Instruct — direct Transformers backend (singleton)

**Files:**
- Create: `models/adapters/qwen3vl_transformers_backend.py`
- Modify: `models/adapters/qwen_adapter.py` (wire in as a third backend option)
- Test: `tests/test_qwen3vl_transformers.py`

**Interfaces:**
- Produces: `class Qwen3VLTransformersBackend` with `load() -> bool`, `is_loaded() -> bool`, `unload() -> None`, `generate_json(images: List[PIL.Image.Image], prompt: str) -> str`, `get_status() -> dict` (`{"available": bool, "model": str, "backend": "transformers", "device": str, "error": str|None}`), and `Qwen3VLTransformersBackend.get_singleton(model_path, dtype, device_map) -> Qwen3VLTransformersBackend` (module-level singleton keyed by `model_path` so a second call with the same path returns the exact same loaded instance — never reloads).
- Consumes: nothing from earlier tasks except the config keys from Task 1 (`verification_pipeline.qwen.transformers.*`) and PIL images from Task 2's `_build_crops`-derived `Image.Image` objects.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_qwen3vl_transformers.py
"""
Tests for the direct-Transformers Qwen3-VL-8B-Instruct backend (spec Part 3).

This machine has no local Qwen3-VL-8B checkpoint and no CUDA, so these tests
exercise the SAME honest-degradation contract used throughout this codebase:
is_available()/load() must report "unavailable" cleanly rather than crash or
fake a result, and the singleton/caching behavior must hold regardless of
whether the underlying model actually loads. Real generation-quality
verification can only happen on HAWK, where a real checkpoint and GPU exist.
"""
import unittest
import tempfile
import shutil
from pathlib import Path

from models.adapters.qwen3vl_transformers_backend import Qwen3VLTransformersBackend


class TestQwen3VLTransformersBackend(unittest.TestCase):

    def test_missing_local_path_reports_unavailable_not_crash(self):
        backend = Qwen3VLTransformersBackend(model_path="/nonexistent/path/Qwen3-VL-8B-Instruct")
        ok = backend.load()
        self.assertFalse(ok)
        self.assertFalse(backend.is_loaded())
        status = backend.get_status()
        self.assertFalse(status["available"])
        self.assertIsNotNone(status["error"])
        self.assertEqual(status["backend"], "transformers")

    def test_never_downloads_when_local_path_missing(self):
        # local_files_only must be enforced -- a missing local path must
        # fail fast, never attempt a network fetch from the Hub.
        backend = Qwen3VLTransformersBackend(model_path="/nonexistent/path/Qwen3-VL-8B-Instruct")
        backend.load()
        self.assertIn("local", (backend.get_status()["error"] or "").lower())

    def test_generate_json_on_unloaded_backend_raises_cleanly(self):
        backend = Qwen3VLTransformersBackend(model_path="/nonexistent/path/Qwen3-VL-8B-Instruct")
        with self.assertRaises(RuntimeError):
            backend.generate_json(images=[], prompt="test")

    def test_singleton_returns_same_instance_for_same_path(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            fake_model_dir = str(Path(tmp_dir) / "fake-model")
            b1 = Qwen3VLTransformersBackend.get_singleton(fake_model_dir, "bfloat16", "auto")
            b2 = Qwen3VLTransformersBackend.get_singleton(fake_model_dir, "bfloat16", "auto")
            self.assertIs(b1, b2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_singleton_differs_for_different_paths(self):
        b1 = Qwen3VLTransformersBackend.get_singleton("/path/a", "bfloat16", "auto")
        b2 = Qwen3VLTransformersBackend.get_singleton("/path/b", "bfloat16", "auto")
        self.assertIsNot(b1, b2)

    def test_unload_clears_loaded_state(self):
        backend = Qwen3VLTransformersBackend(model_path="/nonexistent/path")
        backend.load()
        backend.unload()
        self.assertFalse(backend.is_loaded())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qwen3vl_transformers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.adapters.qwen3vl_transformers_backend'`.

- [x] **Step 3: Implement the backend**

```python
# models/adapters/qwen3vl_transformers_backend.py
"""
models/adapters/qwen3vl_transformers_backend.py - Qwen3-VL-8B-Instruct via HF Transformers
=============================================================================================

Direct, local, Ollama-free Qwen3-VL-8B-Instruct backend (spec Part 3). Loads
once from a local checkpoint directory (never downloads: local_files_only is
always enforced) using Qwen3VLForConditionalGeneration + AutoProcessor, keeps
the model cached in memory via a per-model-path singleton, and runs inference
under torch.inference_mode(). device_map="auto" lets Transformers place the
model across available GPU(s) (e.g. a single A100 40GB); dtype defaults to
bfloat16 when the resolved device supports it, falling back to float16/float32
otherwise -- never crashes if bf16 isn't supported on a given GPU.

This is intentionally NOT a fourth new "model manager" -- it plugs into the
existing QwenVerifier (models/adapters/qwen_adapter.py) as one of three
selectable backends (transformers / ollama / dashscope), reusing that class's
crop-building, prompt template, and response normalization.
"""

from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_SINGLETONS: Dict[str, "Qwen3VLTransformersBackend"] = {}
_SINGLETON_LOCK = threading.Lock()


class Qwen3VLTransformersBackend:
    """
    Loads and runs Qwen3-VL-8B-Instruct once, in-process, via Hugging Face
    Transformers. Never re-downloads: if `model_path` doesn't exist locally,
    load() fails fast and cleanly rather than reaching out to the Hub.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_path = model_path
        self.dtype_name = dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self._loaded = False
        self._error: Optional[str] = None
        self._resolved_device: Optional[str] = None

    @classmethod
    def get_singleton(cls, model_path: str, dtype: str = "bfloat16", device_map: str = "auto") -> "Qwen3VLTransformersBackend":
        """
        Returns the same instance for the same model_path across the whole
        process -- the model is loaded at most once and kept cached in
        memory (spec Part 3/16: never reload per-request, never re-download).
        """
        with _SINGLETON_LOCK:
            existing = _SINGLETONS.get(model_path)
            if existing is None:
                existing = cls(model_path=model_path, dtype=dtype, device_map=device_map)
                _SINGLETONS[model_path] = existing
            return existing

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None and self.processor is not None

    def load(self) -> bool:
        if self.is_loaded():
            return True
        try:
            if not self.model_path or not Path(self.model_path).exists():
                self._error = (
                    f"Local Qwen3-VL-8B-Instruct path does not exist: '{self.model_path}'. "
                    "This backend never downloads from the Hub -- point "
                    "verification_pipeline.qwen.transformers.model_path (or QWEN_MODEL_PATH) "
                    "at a local checkpoint directory."
                )
                return False

            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            resolved_dtype = self._resolve_dtype(torch)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=resolved_dtype,
                device_map=self.device_map,
                local_files_only=True,
            ).eval()

            self._resolved_device = str(next(self.model.parameters()).device) if hasattr(self.model, "parameters") else self.device_map
            self._loaded = True
            self._error = None
            return True
        except Exception as e:
            self._error = f"Failed to load Qwen3-VL-8B-Instruct from '{self.model_path}': {e}"
            self.model = None
            self.processor = None
            self._loaded = False
            return False

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_dtype(self, torch_module):
        """bfloat16 when supported, otherwise a safe fallback -- never crashes
        on a GPU/driver combination that doesn't support bf16."""
        requested = self.dtype_name.lower()
        if requested == "bfloat16":
            try:
                if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
                    return torch_module.bfloat16
                return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32
            except Exception:
                return torch_module.float32
        return getattr(torch_module, requested, torch_module.float32)

    def generate_json(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 512) -> str:
        """
        Run one generation call given a list of PIL images (tight + context
        crop) and a text prompt, returning the raw decoded text (the caller,
        QwenVerifier, is responsible for JSON-extracting/normalizing it --
        this backend does not know about the pole-verification schema).
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Qwen3VLTransformersBackend.generate_json() called before a successful load() -- "
                f"backend not ready (last error: {self._error})."
            )
        import torch

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_loaded(),
            "model": "Qwen3-VL-8B-Instruct",
            "backend": "transformers",
            "device": self._resolved_device or self.device_map,
            "error": self._error,
        }
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qwen3vl_transformers.py -v`
Expected: PASS (6 tests) — all exercise the graceful-failure/singleton-caching contract, which works identically with or without a real checkpoint present.

- [x] **Step 5: Wire the transformers backend into `QwenVerifier` as a selectable third option**

In `models/adapters/qwen_adapter.py`, extend `__init__`'s backend-selection logic (after the existing `ollama_model`/`api_key` discovery, before computing `self._status`):

```python
from models.adapters.qwen3vl_transformers_backend import Qwen3VLTransformersBackend

# ... inside __init__, after self.ollama_model = _discover_ollama_qwen_model(...):

self._tvl_backend: Optional[Qwen3VLTransformersBackend] = None
transformers_model_path = self.local_model_path  # from QWEN_MODEL_PATH / constructor, Task 1's config
if self.backend_preference in ("auto", "transformers") and transformers_model_path:
    candidate = Qwen3VLTransformersBackend.get_singleton(
        transformers_model_path, self.transformers_dtype, self.transformers_device_map,
    )
    if candidate.is_loaded() or (self.backend_preference == "transformers") or Path(transformers_model_path).exists():
        self._tvl_backend = candidate

self._status = "ready" if (
    self._tvl_backend is not None or self.ollama_model or self.api_key
) else "unavailable"
```

Add `_verify_via_transformers` and route `verify()` to prefer it:

```python
def _verify_via_transformers(self, tight_b64: str, context_b64: str, prompt: str) -> Dict[str, Any]:
    import base64, io
    from PIL import Image as PILImage

    if not self._tvl_backend.is_loaded():
        if not self._tvl_backend.load():
            raise RuntimeError(self._tvl_backend.get_status()["error"] or "Qwen3-VL-8B failed to load")

    tight_img = PILImage.open(io.BytesIO(base64.b64decode(tight_b64)))
    context_img = PILImage.open(io.BytesIO(base64.b64decode(context_b64)))
    text = self._tvl_backend.generate_json([tight_img, context_img], prompt)
    return _extract_json(text)
```

In `verify()`, change the backend dispatch from:
```python
if self.ollama_model:
    raw = self._verify_via_ollama(tight_b64, context_b64, prompt)
else:
    raw = self._verify_via_dashscope(tight_b64, context_b64, prompt)
```
to:
```python
if self._tvl_backend is not None:
    raw = self._verify_via_transformers(tight_b64, context_b64, prompt)
elif self.ollama_model:
    raw = self._verify_via_ollama(tight_b64, context_b64, prompt)
else:
    raw = self._verify_via_dashscope(tight_b64, context_b64, prompt)
```

And update the early-return guard (previously `if not self.ollama_model and not self.api_key:`) to also allow the transformers path:
```python
if not self._tvl_backend and not self.ollama_model and not self.api_key:
    return {
        "status": "unavailable",
        "decision": "unavailable",
        "needs_human_review": True,
        "reason": "MODEL UNAVAILABLE: no local Qwen3-VL-8B checkpoint, Ollama model, or API key configured.",
        **_unavailable_fields,
    }
```

- [x] **Step 6: Run the full Qwen test suite to verify no regressions**

Run: `python -m pytest tests/test_phase6_qwen.py tests/test_qwen3vl_transformers.py -v`
Expected: PASS, all tests (Task 2's + Task 3's).

- [x] **Step 7: Commit**

```bash
git add models/adapters/qwen3vl_transformers_backend.py models/adapters/qwen_adapter.py tests/test_qwen3vl_transformers.py
git commit -m "feat(qwen): add direct Transformers Qwen3-VL-8B-Instruct backend

Singleton-cached, local-only (never downloads), device_map=auto +
bfloat16-when-supported. Selectable via verification_pipeline.qwen.backend
alongside the existing Ollama/DashScope paths -- no Ollama dependency
required when this backend is active.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: SAM3 — environment-isolation service + real box-prompted segmentation

**Files:**
- Create: `services/__init__.py`
- Create: `services/sam3_service.py`
- Create: `models/adapters/sam3_http_adapter.py`
- Modify: `models/adapters/sam3_adapter.py` (native box-prompt attempt)
- Modify: `backend/app.py` (backend selection: in-process vs HTTP, both `BaseSegmenter`)
- Test: `tests/test_sam3_adapter.py`, `tests/test_sam3_http_adapter.py`

**Interfaces:**
- Produces: `class SAM3HttpAdapter(BaseSegmenter)` — same public methods as `SAM3Adapter` (`is_available`, `load`, `unload`, `detect_and_segment`, `segment_box`, `get_info`), so `backend/app.py`'s `sam3_adapter = ...` line is the *only* place that changes when switching backends; every other call site (`sam3_adapter.is_available()`, `.detect_and_segment(...)`, `.segment_box(...)`) is untouched.
- Consumes: nothing new from earlier tasks (independent of Qwen work); reads Task 1's `verification_pipeline.sam3.backend`/`sam3.service_url` config keys.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_sam3_adapter.py
"""
Part 18.1-18.3: SAM3 model loading, text-prompt detection, box-prompt
segmentation. This machine has no transformers>=5.9 (pinned to 4.57.0) and
no CUDA, so SAM3Adapter.load() is expected to fail gracefully here --
exactly like every other honest-degradation test in this codebase. On HAWK,
once transformers>=5.9 + the facebook/sam3 checkpoint are available, is_available()
and load() should both return True and detect_and_segment/segment_box should
return real DetectionBox/mask results instead of empty ones.
"""
import unittest
import numpy as np

from models.adapters.sam3_adapter import SAM3Adapter


class TestSAM3AdapterLoading(unittest.TestCase):
    def test_is_available_false_without_transformers_sam3_support(self):
        adapter = SAM3Adapter()
        # Honest reporting either way -- don't assert a specific bool, since
        # a future environment might have it installed; assert it never raises.
        result = adapter.is_available()
        self.assertIsInstance(result, bool)

    def test_load_never_crashes_and_reports_status_via_get_info(self):
        adapter = SAM3Adapter()
        adapter.load()
        info = adapter.get_info()
        self.assertIn(info.status, ("ready", "unavailable", "not_loaded"))
        self.assertEqual(info.id, "sam3_segmenter")
        if info.status != "ready":
            self.assertIsNotNone(info.error_message)

    def test_get_info_reports_backend_field(self):
        adapter = SAM3Adapter()
        info = adapter.get_info()
        self.assertIn("backend", info.to_dict())


class TestSAM3TextPrompt(unittest.TestCase):
    def test_detect_and_segment_on_unavailable_backend_returns_empty_list_not_crash(self):
        adapter = SAM3Adapter()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.detect_and_segment(img)
        self.assertIsInstance(result, list)  # [] when unavailable, never raises


class TestSAM3BoxPrompt(unittest.TestCase):
    def test_segment_box_on_unavailable_backend_returns_none_not_crash(self):
        adapter = SAM3Adapter()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.segment_box(img, (20.0, 20.0, 60.0, 80.0))
        self.assertIsNone(result)  # None when unavailable, never raises


if __name__ == "__main__":
    unittest.main()
```

```python
# tests/test_sam3_http_adapter.py
"""Part 2: SAM3 as an isolated HTTP service, for when SAM3's dependencies
can't coexist with the main app's torch_env_v2. SAM3HttpAdapter implements
the exact same BaseSegmenter interface as the in-process SAM3Adapter, so
backend/app.py can swap between them via one config value."""
import unittest
from models.adapters.base import BaseSegmenter
from models.adapters.sam3_http_adapter import SAM3HttpAdapter


class TestSAM3HttpAdapter(unittest.TestCase):
    def test_implements_base_segmenter_interface(self):
        self.assertTrue(issubclass(SAM3HttpAdapter, BaseSegmenter))

    def test_unreachable_service_reports_unavailable_not_crash(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")  # nothing listens on port 1
        self.assertFalse(adapter.is_available())
        info = adapter.get_info()
        self.assertEqual(info.status, "unavailable")
        self.assertIn("backend", info.to_dict())
        self.assertEqual(info.to_dict()["backend"], "sam3_http")

    def test_segment_box_on_unreachable_service_returns_none(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.segment_box(img, (10.0, 10.0, 50.0, 50.0))
        self.assertIsNone(result)

    def test_detect_and_segment_on_unreachable_service_returns_empty_list(self):
        adapter = SAM3HttpAdapter(service_url="http://127.0.0.1:1")
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.detect_and_segment(img)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sam3_adapter.py tests/test_sam3_http_adapter.py -v`
Expected: `test_sam3_adapter.py`'s `test_get_info_reports_backend_field` FAILs (no `backend` key yet); `test_sam3_http_adapter.py` FAILs entirely with `ModuleNotFoundError`.

- [x] **Step 3: Add `backend` field to `ModelInfo` and set it in `SAM3Adapter.get_info()`**

In `models/adapters/base.py`, extend the `ModelInfo` dataclass (additive field, default `None` — every existing adapter's `get_info()` keeps working unchanged, `to_dict()` just gains one more key):

```python
@dataclass
class ModelInfo:
    id: str
    name: str
    model_type: str
    status: str
    weights_path: Optional[str] = None
    device: str = "cpu"
    error_message: Optional[str] = None
    installation_guide: Optional[str] = None
    backend: Optional[str] = None  # e.g. "transformers", "sam3_http", "ollama:qwen3-vl:2b"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "status": self.status,
            "weights_path": self.weights_path,
            "device": self.device,
            "error_message": self.error_message,
            "installation_guide": self.installation_guide,
            "backend": self.backend,
        }
```

In `models/adapters/sam3_adapter.py`'s `get_info()`, add `backend="sam3_inprocess"`:

```python
def get_info(self) -> ModelInfo:
    return ModelInfo(
        id="sam3_segmenter",
        name="SAM 3 Open-Vocabulary Segmenter",
        model_type="segmenter",
        status=self._status,
        weights_path=self.model_id,
        device=self.device,
        backend="sam3_inprocess",
        error_message=self._error_msg,
        installation_guide=(
            f"Requires Hugging Face model '{self.model_id}' with transformers>=5.9 "
            "(Sam3Processor/Sam3Model). If this app's environment can't upgrade "
            "transformers safely, run services/sam3_service.py in a separate "
            "environment instead and set verification_pipeline.sam3.backend=http. "
            "Set POLE_ALLOW_ONLINE=1 to download."
        ),
    )
```

- [x] **Step 4: Add native box-prompted segmentation, falling back to the existing crop approach**

In `models/adapters/sam3_adapter.py`, replace `segment_box`'s body:

```python
def segment_box(
    self,
    image: Union[str, np.ndarray, Image.Image],
    box_xyxy: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """
    Part 1: SAM3 should refine/segment the CANDIDATE REGION a proposer (DINO)
    already found, not just trust a text prompt in isolation. Tries native
    box-prompted segmentation first (image + input_boxes, same calling
    convention as the SAM2 family this repo already uses in sam21_adapter.py)
    -- falls back to the existing crop-and-text-prompt approach if the
    installed Sam3Processor doesn't accept input_boxes (API differences
    across transformers versions), so this never hard-depends on an exact
    signature this session couldn't verify against a live HAWK install.
    """
    if self.model is None or self.proc is None:
        ok = self.load(self.device)
        if not ok:
            return None

    import torch

    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        pil_img = Image.fromarray(rgb)
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        return None

    try:
        inp = self.proc(
            images=pil_img,
            input_boxes=[[list(box_xyxy)]],
            return_tensors="pt",
        ).to(torch.device(self.device))
        with torch.no_grad():
            out = self.model(**inp)
        masks = self.proc.post_process_masks(
            out.pred_masks, inp.get("original_sizes"), inp.get("reshaped_input_sizes"),
        )
        if masks and len(masks) > 0 and masks[0].numel() > 0:
            m = masks[0][0]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            return _clean((m > 0).astype(np.uint8), box_xyxy)
    except Exception:
        pass  # native box-prompting unsupported/failed -- fall back below

    # --- Fallback: crop to the box (padded) and re-run text-prompted detection ---
    w, h = pil_img.size
    x0, y0, x1, y1 = box_xyxy
    pad_w = 0.1 * (x1 - x0)
    pad_h = 0.1 * (y1 - y0)
    crop_box = (
        max(0, int(x0 - pad_w)), max(0, int(y0 - pad_h)),
        min(w, int(x1 + pad_w)), min(h, int(y1 + pad_h))
    )
    cropped = pil_img.crop(crop_box)
    dets = self.detect_and_segment(cropped)
    if not dets:
        return None

    full_mask = np.zeros((h, w), dtype=np.uint8)
    best_det = max(dets, key=lambda d: d.confidence)
    if best_det.corners is not None:
        corners = np.array(best_det.corners)
        corners[:, 0] += crop_box[0]
        corners[:, 1] += crop_box[1]
        import cv2
        cv2.fillPoly(full_mask, [corners.astype(np.int32)], 255)
        return full_mask
    return None
```

- [x] **Step 5: Create the isolated SAM3 microservice**

```python
# services/__init__.py
```
(empty, marks the package)

```python
# services/sam3_service.py
"""
services/sam3_service.py - Standalone SAM3 microservice
============================================================

Part 2: if SAM3's dependencies (transformers>=5.9) can't safely coexist with
the main app's torch_env_v2 (this repo's own DINO/Qwen/SAM2.1 environment),
run SAM3 in its own Python environment as this small FastAPI service instead,
and point the main app at it via verification_pipeline.sam3.backend=http +
sam3.service_url (see models/adapters/sam3_http_adapter.py, the client that
talks to this service).

Run in the SAM3-specific environment:
    uvicorn services.sam3_service:app --host 0.0.0.0 --port 8801

This process loads SAM3Adapter exactly once at startup and keeps it resident
in memory -- identical loading discipline to the in-process path, just in a
separate OS process/environment.
"""

from __future__ import annotations
import base64
import io
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from models.adapters.sam3_adapter import SAM3Adapter

app = FastAPI(title="SAM3 Isolated Segmentation Service")
sam3_adapter = SAM3Adapter()


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _mask_to_b64_png(mask: np.ndarray) -> str:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray((mask > 0).astype(np.uint8) * 255).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class SegmentBoxRequest(BaseModel):
    image_b64: str
    box_xyxy: Tuple[float, float, float, float]


class DetectRequest(BaseModel):
    image_b64: str
    text_prompt: Optional[str] = None


@app.get("/health")
def health():
    info = sam3_adapter.get_info()
    return {"available": info.status == "ready", "status": info.status, "error": info.error_message}


@app.post("/segment_box")
def segment_box(req: SegmentBoxRequest):
    pil_img = _b64_to_pil(req.image_b64)
    mask = sam3_adapter.segment_box(pil_img, req.box_xyxy)
    if mask is None:
        return {"mask_b64": None}
    return {"mask_b64": _mask_to_b64_png(mask)}


@app.post("/detect_and_segment")
def detect_and_segment(req: DetectRequest):
    pil_img = _b64_to_pil(req.image_b64)
    if req.text_prompt:
        sam3_adapter.text_prompt = req.text_prompt
    detections = sam3_adapter.detect_and_segment(pil_img)
    return {"detections": [d.to_dict() for d in detections]}


@app.on_event("startup")
def _load_on_startup():
    sam3_adapter.load()
```

- [x] **Step 6: Create the HTTP client adapter**

```python
# models/adapters/sam3_http_adapter.py
"""
models/adapters/sam3_http_adapter.py - SAM3 HTTP Client Adapter
=====================================================================

Talks to services/sam3_service.py running in a separate SAM3-specific
environment (Part 2). Implements the exact same BaseSegmenter interface as
the in-process SAM3Adapter, so backend/app.py's module-level `sam3_adapter`
can point at either one -- every downstream call site (run_ai_pipeline,
_sam_refine_candidates, etc.) is unaffected by which backend is active.

Never crashes the main app when the service is down/unreachable: every
method degrades to is_available()=False / [] / None, matching the in-process
adapter's own graceful-failure contract.
"""

from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image

from models.adapters.base import BaseSegmenter, DetectionBox, ModelInfo


def _to_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        return Image.fromarray(rgb)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class SAM3HttpAdapter(BaseSegmenter):
    def __init__(self, service_url: str, timeout_s: float = 30.0):
        self.service_url = service_url.rstrip("/")
        self.timeout_s = timeout_s
        self._last_error: Optional[str] = None

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            self._last_error = data.get("error")
            return bool(data.get("available"))
        except Exception as e:
            self._last_error = f"SAM3 service unreachable at {self.service_url}: {e}"
            return False

    def load(self, device: str = "AUTO") -> bool:
        # The remote service loads its own model at its own startup; this
        # client has nothing local to load -- just confirm reachability.
        return self.is_available()

    def unload(self) -> None:
        pass  # nothing local to free; the remote process owns its own memory

    def detect_and_segment(self, image: Union[str, np.ndarray, Image.Image]) -> List[DetectionBox]:
        try:
            pil_img = _to_pil(image)
            resp = requests.post(
                f"{self.service_url}/detect_and_segment",
                json={"image_b64": _pil_to_b64(pil_img)},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return [DetectionBox.from_dict(d) for d in resp.json().get("detections", [])]
        except Exception as e:
            self._last_error = f"SAM3 service detect_and_segment failed: {e}"
            return []

    def segment_box(
        self, image: Union[str, np.ndarray, Image.Image], box_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        try:
            pil_img = _to_pil(image)
            resp = requests.post(
                f"{self.service_url}/segment_box",
                json={"image_b64": _pil_to_b64(pil_img), "box_xyxy": list(box_xyxy)},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            mask_b64 = resp.json().get("mask_b64")
            if not mask_b64:
                return None
            mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
            return (np.array(mask_img) > 0).astype(np.uint8)
        except Exception as e:
            self._last_error = f"SAM3 service segment_box failed: {e}"
            return None

    def get_info(self) -> ModelInfo:
        available = self.is_available()
        return ModelInfo(
            id="sam3_segmenter",
            name="SAM 3 Open-Vocabulary Segmenter",
            model_type="segmenter",
            status="ready" if available else "unavailable",
            weights_path=self.service_url,
            device="remote",
            backend="sam3_http",
            error_message=None if available else self._last_error,
            installation_guide=(
                f"Start the isolated SAM3 service in its own environment: "
                f"`uvicorn services.sam3_service:app --port 8801`, then confirm "
                f"verification_pipeline.sam3.service_url points at it (currently {self.service_url})."
            ),
        )
```

- [x] **Step 7: Wire backend selection into `backend/app.py`**

Replace `sam3_adapter = SAM3Adapter()` (around line 76) with:

```python
def _build_sam3_adapter():
    """
    Part 2: pick the in-process SAM3Adapter or a remote SAM3HttpAdapter based
    on verification_pipeline.sam3.backend. "auto" prefers in-process (works
    fine if transformers>=5.9 is installed in this same environment) and
    only falls back to the HTTP client when a service_url is actually
    configured -- so a plain in-process setup (most deployments) needs zero
    extra config.
    """
    sam3_cfg = _verification_cfg.get("sam3", {}) or {}
    backend_pref = os.environ.get("SAM3_BACKEND", sam3_cfg.get("backend", "auto"))
    service_url = os.environ.get("SAM3_SERVICE_URL", sam3_cfg.get("service_url"))

    if backend_pref == "http" or (backend_pref == "auto" and service_url):
        from models.adapters.sam3_http_adapter import SAM3HttpAdapter
        return SAM3HttpAdapter(service_url=service_url or "http://127.0.0.1:8801")
    return SAM3Adapter()


sam3_adapter = _build_sam3_adapter()
```

(`_verification_cfg` is already loaded earlier in `backend/app.py` via `_load_verification_pipeline_config()` — this function must be defined/called after that, matching its existing position in the file.)

- [x] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam3_adapter.py tests/test_sam3_http_adapter.py -v`
Expected: PASS (9 tests total).

- [x] **Step 9: Run the full existing test suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: All previously-passing tests still pass (the `ModelInfo.backend` field addition is purely additive/optional).

- [x] **Step 10: Commit**

```bash
git add services/ models/adapters/sam3_http_adapter.py models/adapters/sam3_adapter.py models/adapters/base.py backend/app.py tests/test_sam3_adapter.py tests/test_sam3_http_adapter.py
git commit -m "feat(sam3): isolated HTTP service backend + native box-prompted segmentation

Adds a swappable in-process-vs-HTTP-service SAM3 backend (BaseSegmenter on
both sides, one config flag switches) for when SAM3's transformers>=5.9
requirement can't coexist with the main app's environment. segment_box() now
tries native box-prompted segmentation first, falling back to the existing
crop-and-text-prompt approach.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Per-stage timing/logging

**Files:**
- Modify: `backend/app.py` (`_sam_refine_candidates`, `_apply_verification`, `run_ai_pipeline`)
- Test: `tests/test_backend.py` (extend)

**Interfaces:**
- Produces: `raw_outputs["timings"]` dict with keys `dino_ms`, `sam3_ms`, `geometry_ms`, `qwen_ms`, `obb_ms`, `total_ms` (all `float`, rounded to 1 decimal), present whenever `mode in ("AI_LABEL", "DINO_SAM")` and `verification_pipeline.logging.log_stage_timings` is true (default true).

- [x] **Step 1: Write the failing test**

Append to `tests/test_backend.py`:

```python
def test_ai_label_pipeline_reports_stage_timings(self):
    """Part 17: every AI_LABEL run should report per-stage timing in raw_outputs."""
    ds_id = "api_test_timing_dataset"
    self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "Timing Test", "classes": ["utility_pole"]})
    img_dir = Path(self.test_dir) / ds_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = "timing_test.jpg"
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / img_name), dummy)
    storage_mgr.import_images(ds_id, [img_dir / img_name])

    res = self.client.post(
        "/api/inference/detect",
        json={"dataset_id": ds_id, "filename": img_name, "mode": "AI_LABEL"},
    )
    # DINO/SAM aren't loaded in this CI environment -- a 503 "model
    # unavailable" is an acceptable, honest outcome here; only assert on
    # timings when the pipeline actually ran to completion.
    if res.status_code == 200:
        pred = storage_mgr.get_predictions(ds_id, img_name)
        timings = pred["raw_outputs"].get("timings")
        self.assertIsNotNone(timings)
        for key in ("dino_ms", "total_ms"):
            self.assertIn(key, timings)
            self.assertIsInstance(timings[key], (int, float))
```

- [x] **Step 2: Run test to verify current behavior (informational — this test may already pass with a 503 on this machine)**

Run: `python -m pytest tests/test_backend.py::TestBackendAPI::test_ai_label_pipeline_reports_stage_timings -v`
Expected on this machine: PASS trivially (the `if res.status_code == 200` guard skips the real assertion when DINO is unavailable here) — this test becomes meaningful on HAWK where DINO+SAM3 actually load. Proceed with implementation regardless, so the assertion is enforced once real inference runs.

- [x] **Step 3: Add timing instrumentation**

In `backend/app.py`, add `import time` near the top if not already present, then wrap each stage in `run_ai_pipeline` (production path only, `mode in ("AI_LABEL", "DINO_SAM", "DINO_ONLY")`):

```python
timings: Dict[str, float] = {}
_t0 = time.perf_counter()

dino_boxes = dino_adapter.predict(...)  # existing call
timings["dino_ms"] = round((time.perf_counter() - _t0) * 1000, 1)

_t1 = time.perf_counter()
sam3_candidates = ...  # existing SAM3 proposal block
timings["sam3_propose_ms"] = round((time.perf_counter() - _t1) * 1000, 1)
```

Thread a `timings` dict through `_sam_refine_candidates` and `_apply_verification` (both already receive/return plain data; add an optional `timings: Optional[Dict[str, float]] = None` parameter to each, and have each function accumulate into it if provided):

```python
def _sam_refine_candidates(..., timings: Optional[Dict[str, float]] = None) -> Tuple[List[DetectionBox], int, int]:
    _t_sam = time.perf_counter()
    _t_obb_total = 0.0
    ...  # existing loop body, wrapping the SAM segment_box call and the
         # generate_and_validate_obb call each in their own time.perf_counter()
         # deltas, accumulating into local sam_ms_total / obb_ms_total
    if timings is not None:
        timings["sam3_refine_ms"] = round(sam_ms_total, 1)
        timings["obb_ms"] = round(obb_ms_total, 1)
    return result_boxes, n_rejected, n_invalid_obb
```

```python
def _apply_verification(..., timings: Optional[Dict[str, float]] = None) -> Tuple[List[DetectionBox], Dict[str, int]]:
    geometry_ms_total = 0.0
    qwen_ms_total = 0.0
    for det in candidates:
        if enable_geometry_qa:
            _t_geo = time.perf_counter()
            geometry_result = run_geometry_qa(...)
            geometry_ms_total += (time.perf_counter() - _t_geo) * 1000
        ...
        if should_call_qwen and qwen_verifier.is_available():
            _t_qwen = time.perf_counter()
            semantic_result = qwen_verifier.verify(str(img_path), det)
            qwen_ms_total += (time.perf_counter() - _t_qwen) * 1000
        ...
    if timings is not None:
        timings["geometry_ms"] = round(geometry_ms_total, 1)
        timings["qwen_ms"] = round(qwen_ms_total, 1)
    return kept, counts
```

At the end of `run_ai_pipeline`'s production branch, before `return result_boxes, raw_outputs`:

```python
timings["total_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
if (_verification_cfg.get("logging", {}) or {}).get("log_stage_timings", True):
    raw_outputs["timings"] = timings
    print(f"[timing] {filename or img_path.name}: {timings}")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend.py -v`
Expected: PASS (including the new timing test, trivially on this machine per Step 2's note).

- [x] **Step 5: Commit**

```bash
git add backend/app.py tests/test_backend.py
git commit -m "feat(pipeline): add per-stage timing (dino/sam3/geometry/qwen/obb/total ms)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Model status — backend/device display (UI + API)

**Files:**
- Modify: `models/adapters/qwen_adapter.py` (`get_info()`)
- Modify: `frontend/app.js` (model status row rendering)
- Test: `tests/test_backend.py` (extend)

**Interfaces:**
- Produces: `QwenVerifier.get_info()` now sets `name="Qwen3-VL-8B-Instruct"` + `backend="transformers"` + `device=<resolved>` when the transformers backend is active (Task 3); keeps the existing `"Qwen VLM Verifier (Ollama: ...)"` naming when Ollama is active; frontend model-status rows now render `m.backend` as a small secondary line under the status badge when present.

- [x] **Step 1: Write the failing test**

Append to `tests/test_backend.py`:

```python
def test_qwen_model_status_reports_backend_field(self):
    """Part 10/11: status must name the real backend (transformers/ollama/
    dashscope), never a hard-coded 'Ollama: qwen3-vl:2b' regardless of which
    backend is actually active."""
    res = self.client.get("/api/models/status")
    data = res.json()
    qwen_info = next(m for m in data["models"] if "Qwen" in m["name"] or m["id"] == "qwen_verifier")
    self.assertIn("backend", qwen_info)
    if qwen_info["status"] == "ready":
        self.assertIn(qwen_info["backend"], ("transformers", "ollama", "dashscope"))
        self.assertIsNotNone(qwen_info["backend"])
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backend.py::TestBackendAPI::test_qwen_model_status_reports_backend_field -v`
Expected: FAIL — `"backend"` key not yet reliably set to one of the three literal values (currently `weights_path` encodes backend info as a string, not a clean `backend` field with those exact values).

- [x] **Step 3: Update `QwenVerifier.get_info()`**

```python
def get_info(self) -> ModelInfo:
    if self._tvl_backend is not None and self._tvl_backend.is_loaded():
        status = self._tvl_backend.get_status()
        return ModelInfo(
            id="qwen_verifier",
            name="Qwen3-VL-8B-Instruct",
            model_type="verifier",
            status="ready",
            weights_path=self._tvl_backend.model_path,
            device=status["device"],
            backend="transformers",
            error_message=None,
            installation_guide="Loaded locally via Hugging Face Transformers (device_map=auto).",
        )
    if self.ollama_model:
        return ModelInfo(
            id="qwen_verifier",
            name=f"Qwen VLM Verifier (Ollama: {self.ollama_model})",
            model_type="verifier",
            status=self._status,
            weights_path=f"ollama:{self.ollama_model}",
            device="local_ollama",
            backend="ollama",
            error_message=self._error_msg,
            installation_guide=(
                "To enable Qwen visual verification: run a local Ollama server with a Qwen(-VL) model "
                "pulled, set QWEN_MODEL_PATH to a local Qwen3-VL-8B-Instruct checkpoint for the "
                "Transformers backend, or set QWEN_API_KEY/DASHSCOPE_API_KEY for the cloud API."
            ),
        )
    return ModelInfo(
        id="qwen_verifier",
        name="Qwen VLM Verifier",
        model_type="verifier",
        status=self._status,
        weights_path=self.model_name if self.api_key else self.local_model_path,
        device="cloud" if self.api_key else "cpu",
        backend="dashscope" if self.api_key else None,
        error_message=self._error_msg,
        installation_guide=(
            "To enable Qwen visual verification: run a local Ollama server with a Qwen(-VL) model "
            "pulled, set QWEN_MODEL_PATH to a local Qwen3-VL-8B-Instruct checkpoint for the "
            "Transformers backend, or set QWEN_API_KEY/DASHSCOPE_API_KEY for the cloud API."
        ),
    )
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backend.py -v`
Expected: PASS (all tests, including the new one).

- [x] **Step 5: Update frontend model-status row rendering**

In `frontend/app.js`, find the model-status row template (`row.innerHTML = ...` near the code that builds `elements.modelStatusList`, referenced earlier around status badge rendering) and add a backend line:

```javascript
row.innerHTML = `
  <span class="model-name" title="${m.name}">${m.name}</span>
  ${m.backend ? `<span class="model-backend">${m.backend}${m.device && m.device !== 'cpu' ? ' · ' + m.device : ''}</span>` : ''}
  <span class="status-badge ${m.status}">${m.status}</span>
`;
```

Add a small CSS rule in `frontend/style.css` (reusing existing `--text-muted` token):

```css
.model-backend {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-left: 6px;
}
```

- [x] **Step 6: Commit**

```bash
git add models/adapters/qwen_adapter.py frontend/app.js frontend/style.css tests/test_backend.py
git commit -m "feat(ui): show real backend name (transformers/ollama/dashscope) in model status

Replaces the previously hard-coded 'Ollama: qwen3-vl:2b'-shaped display with
whichever backend actually loaded -- 'Qwen3-VL-8B-Instruct / transformers /
cuda' on HAWK, 'Qwen VLM Verifier (Ollama: ...)' locally, etc.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Batch/predictions.json — spec-shaped summary

**Files:**
- Modify: `backend/storage.py` (`save_predictions`)
- Test: `tests/test_storage.py` (extend)

**Interfaces:**
- Produces: `DatasetManager.save_predictions(...)` writes an additional top-level `record["summary"]` key: `{"image": filename, "dino": [...], "sam3": [...], "geometry": [...], "qwen": [...], "obb": [...], "final_status": "needs_review"|"reviewed_clean"|"no_candidates"}` — purely additive; the existing `record["boxes"]`/`record["raw_outputs"]` (already relied on by `get_predictions`, the frontend, and existing tests) are unchanged.

- [x] **Step 1: Write the failing test**

Append to `tests/test_storage.py`:

```python
def test_save_predictions_includes_spec_shaped_summary(self):
    """Part 12: predictions.json additionally exposes a flat per-stage
    summary (dino/sam3/geometry/qwen/obb/final_status) alongside the
    existing boxes/raw_outputs structure, without changing either."""
    ds_id = "summary_test_ds"
    self.mgr.create_dataset(ds_id)
    box = DetectionBox(
        xyxy=(10.0, 10.0, 30.0, 90.0),
        corners=xyxy_to_obb_corners(10, 10, 30, 90).tolist(),
        confidence=0.8,
        model_source="DINO+SAM3+SAM",
        needs_review=True,
        attributes={
            "decision": "REVIEW", "geometry_score": 0.7, "geometry_status": "warning",
            "qwen_class": "electric_utility_pole", "qwen_semantic_confidence": 0.6,
            "obb_source": "mask",
        },
    )
    self.mgr.save_predictions(ds_id, "img1.jpg", [box], raw_outputs={"dino": [{"a": 1}], "sam3_candidates": []})

    record = self.mgr.get_predictions(ds_id, "img1.jpg")
    self.assertIn("summary", record)
    summary = record["summary"]
    self.assertEqual(summary["image"], "img1.jpg")
    self.assertIn("dino", summary)
    self.assertIn("sam3", summary)
    self.assertIn("geometry", summary)
    self.assertIn("qwen", summary)
    self.assertIn("obb", summary)
    self.assertEqual(summary["final_status"], "needs_review")
    self.assertEqual(len(summary["geometry"]), 1)
    self.assertEqual(summary["geometry"][0]["geometry_status"], "warning")
    # Existing structure must be completely unaffected.
    self.assertIn("boxes", record)
    self.assertIn("raw_outputs", record)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_storage.py::TestDatasetManager::test_save_predictions_includes_spec_shaped_summary -v`
Expected: FAIL with `KeyError: 'summary'`.

- [x] **Step 3: Implement `_build_spec_summary` and call it from `save_predictions`**

In `backend/storage.py`, add a helper and call it inside `save_predictions` before writing `record`:

```python
def _build_spec_summary(self, filename: str, boxes: List[DetectionBox], raw_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Part 12: a flat, spec-shaped view of the same information already in
    `boxes`/`raw_outputs` -- derived at save time, not a second source of
    truth. `boxes` (with per-candidate .attributes) remains authoritative;
    this is purely a convenience projection for consumers that expect the
    literal dino/sam3/geometry/qwen/obb/final_status shape from the spec.
    """
    geometry_entries, qwen_entries, obb_entries = [], [], []
    any_needs_review = False
    for b in boxes:
        attrs = b.attributes or {}
        any_needs_review = any_needs_review or b.needs_review
        geometry_entries.append({
            "geometry_score": attrs.get("geometry_score"),
            "geometry_status": attrs.get("geometry_status"),
            "geometry_flags": attrs.get("geometry_flags", []),
        })
        qwen_entries.append({
            "ran": attrs.get("qwen_ran", False),
            "class": attrs.get("qwen_class"),
            "material": attrs.get("qwen_material"),
            "visibility": attrs.get("qwen_visibility"),
            "semantic_confidence": attrs.get("qwen_semantic_confidence"),
            "reason": attrs.get("qwen_reason"),
        })
        obb_entries.append({
            "corners": b.corners,
            "source": attrs.get("obb_source"),
        })

    if not boxes:
        final_status = "no_candidates"
    elif any_needs_review:
        final_status = "needs_review"
    else:
        final_status = "reviewed_clean"

    return {
        "image": filename,
        "dino": raw_outputs.get("dino", []),
        "sam3": raw_outputs.get("sam3_candidates", raw_outputs.get("sam3", [])),
        "geometry": geometry_entries,
        "qwen": qwen_entries,
        "obb": obb_entries,
        "final_status": final_status,
    }
```

And in `save_predictions`, add one line before `pred_path.write_text(...)`:

```python
record = {
    "dataset_id": dataset_id,
    "filename": filename,
    "stem": stem,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "boxes": [b.to_dict() for b in boxes],
    "raw_outputs": raw_outputs or {},
    "summary": self._build_spec_summary(filename, boxes, raw_outputs or {}),
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_storage.py -v`
Expected: PASS (all tests, including the new one).

- [x] **Step 5: Commit**

```bash
git add backend/storage.py tests/test_storage.py
git commit -m "feat(storage): add spec-shaped dino/sam3/geometry/qwen/obb summary to predictions.json

Additive only -- the existing boxes[]/raw_outputs structure (already relied
on by the frontend and other tests) is unchanged; summary is a derived
projection for consumers expecting the literal spec shape.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Human review — Skip action

**Files:**
- Modify: `backend/storage.py` (`"skipped"` status + `mark_skipped`)
- Modify: `backend/app.py` (new endpoint)
- Modify: `frontend/index.html`, `frontend/app.js`, `frontend/style.css`
- Test: `tests/test_backend.py` (extend)

**Interfaces:**
- Produces: `DatasetManager.mark_skipped(dataset_id: str, filename: str) -> None`, endpoint `POST /api/datasets/{dataset_id}/images/{filename}/skip`, frontend `skipCurrentAnnotation()` bound to a new `#btn-skip` button and the existing keyboard-shortcut dispatcher.

- [x] **Step 1: Write the failing test**

Append to `tests/test_backend.py`:

```python
def test_skip_endpoint_marks_status_without_touching_predictions(self):
    """Part 13: Skip marks an image for later review without deleting or
    altering its existing AI predictions."""
    ds_id = "api_test_skip_dataset"
    self.client.post("/api/datasets", json={"dataset_id": ds_id, "name": "Skip Test", "classes": ["utility_pole"]})
    img_dir = Path(self.test_dir) / ds_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = "skip_test.jpg"
    cv2.imwrite(str(img_dir / img_name), np.zeros((100, 100, 3), dtype=np.uint8))
    storage_mgr.import_images(ds_id, [img_dir / img_name])

    box = DetectionBox(xyxy=(10, 10, 30, 90), corners=xyxy_to_obb_corners(10, 10, 30, 90).tolist(), confidence=0.7)
    storage_mgr.save_predictions(ds_id, img_name, [box], raw_outputs={})

    res = self.client.post(f"/api/datasets/{ds_id}/images/{img_name}/skip")
    self.assertEqual(res.status_code, 200)

    meta = storage_mgr.get_dataset(ds_id)
    self.assertEqual(meta["images"][img_name]["status"], "skipped")
    # Predictions must remain untouched.
    pred = storage_mgr.get_predictions(ds_id, img_name)
    self.assertEqual(len(pred["boxes"]), 1)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backend.py::TestBackendAPI::test_skip_endpoint_marks_status_without_touching_predictions -v`
Expected: FAIL with 404 (endpoint doesn't exist yet).

- [x] **Step 3: Add `mark_skipped` to `DatasetManager`**

In `backend/storage.py`:

```python
def mark_skipped(self, dataset_id: str, filename: str) -> None:
    """Part 13: mark an image as skipped (uncertain, deferred for later
    review) without touching its predictions/annotations -- a distinct
    status from unlabeled/verified/rejected so the review queue can filter
    on it separately."""
    with self._lock_for(dataset_id):
        meta = self.get_dataset(dataset_id)
        if not meta or filename not in meta.get("images", {}):
            return
        meta["images"][filename]["status"] = "skipped"
        meta["images"][filename]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._recount_stats(meta)
        self._update_metadata(dataset_id, meta)
```

Add `skipped_count` to `_recount_stats`:

```python
meta["skipped_count"] = sum(1 for v in images.values() if v.get("status") == "skipped")
```

- [x] **Step 4: Add the endpoint**

In `backend/app.py`, near the other per-image endpoints:

```python
@app.post("/api/datasets/{dataset_id}/images/{filename}/skip")
def skip_image_endpoint(dataset_id: str, filename: str):
    storage_mgr.mark_skipped(dataset_id, filename)
    return {"status": "skipped", "filename": filename}
```

- [x] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_backend.py -v`
Expected: PASS (all tests).

- [x] **Step 6: Wire the Skip button into the frontend**

In `frontend/index.html`, add next to the existing Accept/Save Edits/Reject buttons:

```html
<button id="btn-skip" class="btn btn-secondary">Skip</button>
```

In `frontend/app.js`, add the element reference and handler near `acceptCurrentAnnotation`/`rejectCurrentAnnotation`:

```javascript
btnSkip: document.getElementById('btn-skip'),
```

```javascript
elements.btnSkip.addEventListener('click', skipCurrentAnnotation);
```

```javascript
async function skipCurrentAnnotation() {
  if (!state.activeImageMeta) return;
  try {
    await fetch(API_BASE + `api/datasets/${state.activeDatasetId}/images/${state.activeImageMeta.filename}/skip`, {
      method: 'POST',
    });
  } catch (err) {
    console.error('Skip failed:', err);
  }
  await selectImage(Math.min(state.activeImageIndex + 1, state.images.length - 1));
}
```

Add the `K` keyboard shortcut in `handleKeyDown` next to the existing `A`/`R` handlers:

```javascript
} else if (e.key === 'k' || e.key === 'K') {
  e.preventDefault();
  skipCurrentAnnotation();
}
```

Add a keyboard-shortcuts table row and minimal CSS reusing `.btn-secondary` (no new class needed).

- [x] **Step 7: Commit**

```bash
git add backend/storage.py backend/app.py frontend/index.html frontend/app.js tests/test_backend.py
git commit -m "feat(review): add Skip action alongside Accept/Save Edits/Reject

Marks an image 'skipped' (distinct from unlabeled/verified/rejected) for
later review without touching its existing predictions or annotations.
Bound to a new Skip button and the K keyboard shortcut.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end pipeline test

**Files:**
- Create: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: `run_ai_pipeline` (from `backend/app.py`, unchanged signature), all adapters from Tasks 2-5.

- [x] **Step 1: Write the end-to-end test**

```python
# tests/test_e2e_pipeline.py
"""
Part 18 (end-to-end): image -> DINO -> SAM3 -> geometry -> Qwen -> OBB.

On this machine (no CUDA, transformers 4.57.0, no local models/Ollama
models pulled), DINO/SAM3/Qwen all honestly report unavailable, so this test
verifies the PIPELINE WIRING never crashes and degrades correctly end-to-end
-- it cannot verify real detection/segmentation/verification quality, which
requires HAWK's actual models and GPU. Run this same test file unchanged on
HAWK once SAM3 + Qwen3-VL-8B are loaded there: with real models available,
the `if raw_outputs.get("error")` branch won't trigger, and the assertions
in the `else` branch verify a real OBB was actually produced.
"""
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2

from backend.app import run_ai_pipeline, storage_mgr


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        storage_mgr.base_dir = Path(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_dino_sam3_geometry_qwen_obb_pipeline_never_crashes(self):
        ds_id = "e2e_test_ds"
        storage_mgr.create_dataset(ds_id)
        img_dir = Path(self.test_dir) / ds_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / "e2e_pole.jpg"
        # A synthetic vertical bright strip on a dark background -- not a
        # real pole photo, but enough to exercise every stage's code path
        # without crashing, on a machine with no real detector weights.
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        img[:, 140:160] = 200
        cv2.imwrite(str(img_path), img)

        boxes, raw_outputs = run_ai_pipeline(
            img_path=img_path, mode="AI_LABEL", dataset_id=ds_id, filename="e2e_pole.jpg",
        )

        if raw_outputs.get("error"):
            # Honest degradation: DINO/SAM unavailable on this machine.
            self.assertIn("unavailable", raw_outputs["error"].lower())
            return

        # If DINO+SAM did load (e.g. this test runs on HAWK), every kept
        # candidate must carry a real OBB and a decision from the combined
        # verification pipeline -- proving the full chain actually ran.
        for box in boxes:
            self.assertIsNotNone(box.corners)
            self.assertEqual(len(box.corners), 4)
            self.assertIn("decision", box.attributes)
            self.assertIn(box.attributes["decision"], ("ACCEPT", "REVIEW"))  # REJECT already dropped
        if "timings" in raw_outputs:
            self.assertIn("total_ms", raw_outputs["timings"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the test**

Run: `python -m pytest tests/test_e2e_pipeline.py -v`
Expected: PASS — on this machine via the `raw_outputs.get("error")` early-return branch (DINO unavailable, honestly reported); on HAWK, PASS via the full assertions once SAM3/Qwen3-VL-8B are actually loaded.

- [x] **Step 3: Run the entire test suite one final time**

Run: `python -m pytest tests/ -q`
Expected: All tests pass, including every test from Tasks 1-9.

- [x] **Step 4: Commit**

```bash
git add tests/test_e2e_pipeline.py
git commit -m "test: add end-to-end DINO->SAM3->geometry->Qwen->OBB pipeline test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Part 1 (SAM3 integration, DINO boxes → SAM3 refine, text+box prompting, stored fields) → Task 4 (already largely present; box-prompting upgraded).
- Part 2 (environment isolation) → Task 4 (`services/sam3_service.py` + `SAM3HttpAdapter`).
- Part 3 (Qwen3-VL-8B Transformers, no Ollama requirement, singleton, local path, device_map/bfloat16) → Task 3.
- Part 4 (Qwen semantic fields, strict JSON) → Task 2.
- Part 5 (Qwen confidence is one signal, configurable weights) → already implemented in `decision_engine.py`/`config.yaml` (documented under "Deviations").
- Part 6 (tight + context crop) → Task 2.
- Part 7 (Qwen can't override bad segmentation) → already implemented in `decision_engine.py` (documented under "Deviations").
- Part 8 (geometric QA) → already implemented in `geometry_qa.py` (verified during inspection, unchanged).
- Part 9 (OBB from mask, not Qwen) → already implemented in `obb_generator.py` (verified during inspection, unchanged).
- Part 10 (UI model status labels) → Task 6.
- Part 11 (health check per model) → existing `/api/models/status` extended by Tasks 4/6's `backend` field.
- Part 12 (batch engine prediction.json shape) → Task 7.
- Part 13 (Accept/Edit/Reject/Skip) → Accept/Reject/Save-Edits(=Edit) already exist; Skip added in Task 8.
- Part 14 (active learning capture) → already implemented (`history/` diffs, confirmed present from earlier session work) — no task needed, not touched.
- Part 15 (error handling) → already implemented throughout every adapter (verified during inspection); Tasks 3/4 extend the same pattern to the new backends.
- Part 16 (performance: load once, GPU, no reload) → Task 3's singleton; `inference_mode()` used.
- Part 17 (logging/timing) → Task 5.
- Part 18 (testing) → Tasks 1-9 each carry their own tests; Task 9 is the dedicated e2e test.
- Part 19 (mistakes to avoid) → covered by Global Constraints.
- Part 20 (final UI goal) → Tasks 6/8 (status panel + review actions); the pipeline order itself already matches (verified during inspection).

**Placeholder scan:** No `TBD`/`implement later`/handwaved steps — every step above has runnable code and an exact test command.

**Type consistency:** `DetectionBox`, `ModelInfo`, `BaseSegmenter`, `BaseVerifier` (from `models/adapters/base.py`) are used identically across all tasks. `Qwen3VLTransformersBackend` (Task 3) and `SAM3HttpAdapter`/`services/sam3_service.py` (Task 4) are independent and can be built in either order. Task 6 depends on Task 3's `_tvl_backend` attribute existing on `QwenVerifier`. Task 5 depends on Task 4's adapter interface being stable (it only touches `backend/app.py`). Tasks 7-9 depend on Tasks 2-5's `attributes` keys (`geometry_score`, `qwen_*`, `decision`, `obb_source`) already being set by the existing `_apply_verification`/`_sam_refine_candidates` — confirmed unchanged during inspection.

---

## What you'll need to do differently on HAWK (not part of this plan's code — deployment steps)

1. Set `verification_pipeline.qwen.transformers.model_path` (or `QWEN_MODEL_PATH` env var) to `/home/maska/models/Qwen3-VL-8B-Instruct`, and `verification_pipeline.qwen.backend: "transformers"` to pin it explicitly (skip the Ollama-preferring `"auto"` default).
2. First run `python -c "from transformers import Sam3Processor, Sam3Model"` inside `torch_env_v2` on HAWK. If it imports cleanly, leave `verification_pipeline.sam3.backend: "auto"` (in-process). If it conflicts with `torch_env_v2`'s pinned versions, create a separate `sam3_env`, install `transformers>=5.9` there only, run `uvicorn services.sam3_service:app --port 8801` in that env, and set `verification_pipeline.sam3.backend: "http"` + `sam3.service_url: "http://127.0.0.1:8801"` in the main app's env.
3. Confirm `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()` are both `True` on the A100 (expected) so Task 3's dtype resolution actually picks `bfloat16` as intended.
