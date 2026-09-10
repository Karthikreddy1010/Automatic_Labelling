"""
models/adapters/qwen_adapter.py - Qwen VLM Visual/Semantic Verifier Adapter
=============================================================================

Wraps Qwen (via a local Ollama server, DashScope's cloud API, or a local
HuggingFace Qwen-VL) to act as an optional verification layer for candidate
utility pole OBB annotations.

Backend priority: a locally running Ollama server with a pulled Qwen(-VL)
model is preferred when present (free, local, no API key/network egress
needed -- matches "use the existing environment, don't add cloud
dependencies" for the H200/deepthink setup). Falls back to DashScope's cloud
API when a QWEN_API_KEY/DASHSCOPE_API_KEY is configured and no local Ollama
Qwen model is found. Reports MODEL UNAVAILABLE, never mocks fake responses,
when neither is available.

Guarantees:
- Qwen is NOT the detector and NEVER generates OBB coordinates; it evaluates
  proposed DINO/SAM candidate labels. Geometry always comes from the SAM mask
  (see models/adapters/obb_generator.py); Qwen only judges "does this crop
  actually look like a utility pole."
- Returns structured JSON classifying the candidate:
    {
      "class": "electric_utility_pole" | "non_utility_pole" | "tree" |
               "building_or_structure" | "street_light_or_lamp_post" |
               "other_object" | "uncertain",
      "material": "wood" | "steel" | "concrete" | "composite" | "unknown",
      "visibility": "mostly_visible" | "partially_occluded" |
                    "severely_occluded" | "too_distant",
      "orientation": "vertical" | "slightly_tilted" | "strongly_tilted",
      "semantic_confidence": 0.0-1.0,
      "decision": "accept" | "review" | "reject",
      "reason": str
    }
  material/visibility/orientation are supporting metadata, NOT separate YOLO
  classes -- the dataset class always stays "utility_pole".
- Does not require wood, full visibility, or perfect verticality to accept a
  pole (material/occlusion/tilt alone never disqualify a candidate) -- see
  the semantic rules encoded in models/adapters/decision_engine.py, which is
  the component that makes the final accept/review/reject call using Qwen's
  output as ONE of several independent signals, never on its own.
- Handles malformed/partial JSON safely: unknown enum values are normalized
  to "uncertain"/"unknown" rather than raised as errors; any other failure
  (network, timeout, non-JSON body) returns status="error" with
  needs_human_review=True. Never crashes the labeling pipeline.
- NEVER overwrites annotations directly. Its output is active-learning evidence.
- If neither Ollama nor an API key/local weights are available, clearly
  reports MODEL UNAVAILABLE. Never mocks fake responses.
"""

from __future__ import annotations
import os
import io
import re
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image

from models.adapters.base import BaseVerifier, DetectionBox, ModelInfo
from models.adapters.qwen3vl_transformers_backend import Qwen3VLTransformersBackend

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


def _parse_param_size_b(tag_name: str) -> float:
    """
    Extract an approximate parameter count in billions from an Ollama tag
    like "qwen3-vl:2b" or "qwen3-vl:8b-instruct-q4" -> 2.0 / 8.0. Returns
    +inf for a tag with no parseable size (e.g. "qwen3-vl:latest") so it
    never gets preferred over an explicitly-sized, known-smaller model.
    """
    m = re.search(r"(\d+(?:\.\d+)?)b", tag_name.lower())
    return float(m.group(1)) if m else float("inf")


def _discover_ollama_qwen_model(
    base_url: str, timeout: float = 1.5, preferred_model: Optional[str] = None
) -> Optional[str]:
    """
    Return the name of a vision-capable Qwen model ("...vl..." tag, e.g.
    "qwen3-vl:2b", "qwen2.5vl:7b") already pulled in a local Ollama server,
    or None if Ollama isn't reachable or has no such model pulled.

    Deliberately does NOT fall back to a text-only Qwen model (e.g.
    "qwen2.5-coder"): this verifier's whole job is judging an image crop, so
    a model that can't see the image must not be reported as available for
    it -- that would silently produce judgments from a model that never
    looked at the picture, which is exactly the "no mocked responses"
    guarantee this adapter exists to uphold.

    Selection when multiple vision-capable Qwen models are pulled:
    - `preferred_model` (exact name, or a substring match) wins if given --
      set via QWEN_OLLAMA_MODEL for an explicit pin.
    - Otherwise prefers the SMALLEST parseable model size. CPU-only
      inference on an 8B+ VLM is impractically slow for per-candidate
      verification (empirically 5+ minutes per crop, timed out twice on a
      16GB CPU-only machine); the smallest available vision-capable Qwen
      model gives usably-fast responses without a GPU. On a GPU box (e.g.
      DeepThink's H200) pin QWEN_OLLAMA_MODEL to a larger model explicitly
      if desired -- bigger is not assumed better here, smaller is just the
      safer default for typical local hardware.

    Fast/short-timeout: this runs at adapter construction time (backend
    startup) and must not hang.
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        vl_matches = [n for n in names if "qwen" in n.lower() and "vl" in n.lower()]
        if not vl_matches:
            return None

        if preferred_model:
            exact = [n for n in vl_matches if n == preferred_model]
            if exact:
                return exact[0]
            substr = [n for n in vl_matches if preferred_model.lower() in n.lower()]
            if substr:
                return substr[0]

        return min(vl_matches, key=_parse_param_size_b)
    except Exception:
        return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Strip an optional ```json ... ``` fence and parse the remaining JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class QwenVerifier(BaseVerifier):
    """
    Adapter for Qwen Vision-Language Model quality verification.
    """

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
        self.context_crop_expand_pct = context_crop_expand_pct
        self.backend_preference = backend  # "auto" | "transformers" | "ollama" | "dashscope"
        self.transformers_dtype = transformers_dtype
        self.transformers_device_map = transformers_device_map
        self.api_key = (
            api_key or
            os.environ.get("QWEN_API_KEY") or
            os.environ.get("DASHSCOPE_API_KEY")
        )
        self.model_name = model_name
        # CPU-only inference on an 8B VLM is genuinely slow (multiple minutes
        # per crop is normal without a GPU) -- default generous enough for
        # that, overridable via QWEN_OLLAMA_TIMEOUT_S for a GPU box (e.g. the
        # DeepThink H200) where a much shorter timeout is more appropriate.
        self.ollama_timeout_s = ollama_timeout_s or float(os.environ.get("QWEN_OLLAMA_TIMEOUT_S", "300"))
        self.local_model_path = local_model_path or os.environ.get("QWEN_MODEL_PATH")

        self.ollama_base_url = (ollama_base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.ollama_preferred_model = ollama_preferred_model or os.environ.get("QWEN_OLLAMA_MODEL")
        self.ollama_model = _discover_ollama_qwen_model(
            self.ollama_base_url, preferred_model=self.ollama_preferred_model
        )

        # Direct-Transformers Qwen3-VL-8B-Instruct backend (Part 3): only
        # constructed (not necessarily loaded yet -- load() is lazy, on first
        # verify() call) when a local checkpoint path is actually configured.
        # Never required to be present -- is_available()/verify() fall back
        # to Ollama/DashScope cleanly when it isn't.
        self._tvl_backend: Optional[Qwen3VLTransformersBackend] = None
        if self.backend_preference in ("auto", "transformers") and self.local_model_path:
            self._tvl_backend = Qwen3VLTransformersBackend.get_singleton(
                self.local_model_path, self.transformers_dtype, self.transformers_device_map,
            )

        local_checkpoint_exists = bool(self.local_model_path and Path(self.local_model_path).exists())
        self._status = "ready" if (
            (self._tvl_backend is not None and local_checkpoint_exists) or self.ollama_model or self.api_key
        ) else "unavailable"
        self._error_msg = None if self._status == "ready" else (
            f"No local Qwen3-VL-8B-Instruct checkpoint (QWEN_MODEL_PATH), vision-capable Qwen model "
            f"(a '...vl...' tag) found in Ollama at {self.ollama_base_url} -- a text-only Qwen model "
            "(e.g. qwen2.5-coder) does not count, since this verifier must look at an image crop -- "
            "and QWEN_API_KEY/DASHSCOPE_API_KEY not set. Run `ollama pull qwen2.5vl:7b` (or similar), "
            "or point QWEN_MODEL_PATH at a local Qwen3-VL-8B-Instruct directory, to enable a local path."
        )

    def is_available(self) -> bool:
        return self._status == "ready"

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

    def _verify_via_transformers(self, tight_b64: str, context_b64: str, prompt: str) -> Dict[str, Any]:
        if not self._tvl_backend.is_loaded():
            if not self._tvl_backend.load():
                raise RuntimeError(self._tvl_backend.get_status()["error"] or "Qwen3-VL-8B failed to load")

        tight_img = Image.open(io.BytesIO(base64.b64decode(tight_b64)))
        context_img = Image.open(io.BytesIO(base64.b64decode(context_b64)))
        text = self._tvl_backend.generate_json([tight_img, context_img], prompt)
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

    def verify(
        self,
        image: Union[str, np.ndarray, Image.Image],
        detection: DetectionBox
    ) -> Dict[str, Any]:
        """
        Verify candidate utility pole annotation.
        Returns structured verification evidence.
        """
        _unavailable_fields = {
            "class": None, "material": None, "visibility": None,
            "orientation": None, "semantic_confidence": None,
            "is_utility_pole": None, "obb_quality": "unknown",
        }
        if not self.is_available():
            return {
                "status": "unavailable",
                "decision": "unavailable",
                "needs_human_review": True,
                "reason": "MODEL UNAVAILABLE: no local Ollama Qwen model, API key, or local weights configured.",
                **_unavailable_fields,
            }

        if self._tvl_backend is None and not self.ollama_model and not self.api_key:
            return {
                "status": "unavailable",
                "decision": "unavailable",
                "needs_human_review": True,
                "reason": (
                    "MODEL UNAVAILABLE: no local Qwen3-VL-8B checkpoint (QWEN_MODEL_PATH), Ollama "
                    "model, or API key configured."
                ),
                **_unavailable_fields,
            }

        try:
            tight_b64, context_b64 = self._build_crops(image, detection)
            prompt = VERIFY_PROMPT_TEMPLATE.format(
                xyxy=detection.xyxy, corners=detection.corners,
                model_source=detection.model_source, confidence=detection.confidence,
            )

            raw = None
            tvl_error: Optional[Exception] = None
            if self._tvl_backend is not None:
                try:
                    raw = self._verify_via_transformers(tight_b64, context_b64, prompt)
                except Exception as e:
                    tvl_error = e
                    if self.backend_preference == "transformers":
                        # Explicitly pinned to the transformers backend -- do
                        # not silently fall back to a different backend the
                        # deployer didn't ask for; surface the real failure.
                        raise
            if raw is None:
                if self.ollama_model:
                    raw = self._verify_via_ollama(tight_b64, context_b64, prompt)
                elif self.api_key:
                    raw = self._verify_via_dashscope(tight_b64, context_b64, prompt)
                elif tvl_error is not None:
                    raise tvl_error
                else:
                    raise RuntimeError("No Qwen backend available to handle this request.")

            # Malformed/partial JSON (missing keys, bad enum values, wrong
            # types) is normalized here rather than raised -- only a hard
            # failure (network error, no JSON found at all) falls through
            # to the except block below.
            norm = _normalize_qwen_response(raw if isinstance(raw, dict) else {})

            return {
                "status": "ok",
                "class": norm["class"],
                "material": norm["material"],
                "visibility": norm["visibility"],
                "orientation": norm["orientation"],
                "semantic_confidence": norm["semantic_confidence"],
                "decision": norm["decision"],
                "reason": norm["reason"],
                # Spec (Part 4) fields, additive:
                "is_utility_pole": norm["is_utility_pole"],
                "pole_type": norm["pole_type"],
                "occlusion": norm["occlusion"],
                "truncated": norm["truncated"],
                "annotation_suitable": norm["annotation_suitable"],
                "obb_quality": {"accept": "good", "review": "uncertain", "reject": "poor"}[norm["decision"]],
                "needs_human_review": (
                    norm["decision"] != "accept"
                    or norm["visibility"] in ("severely_occluded", "too_distant")
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "decision": "review",
                "needs_human_review": True,
                "reason": f"Qwen verification error: {str(e)}",
                **_unavailable_fields,
            }

    def get_info(self) -> ModelInfo:
        install_guide = (
            "To enable Qwen visual verification: run a local Ollama server with a Qwen(-VL) model "
            "pulled, set QWEN_MODEL_PATH to a local Qwen3-VL-8B-Instruct checkpoint for the "
            "Transformers backend, or set QWEN_API_KEY/DASHSCOPE_API_KEY for the cloud API."
        )
        if self._tvl_backend is not None and self._tvl_backend.is_loaded():
            tvl_status = self._tvl_backend.get_status()
            return ModelInfo(
                id="qwen_verifier",
                name="Qwen3-VL-8B-Instruct",
                model_type="verifier",
                status="ready",
                weights_path=self._tvl_backend.model_path,
                device=tvl_status["device"],
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
                installation_guide=install_guide,
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
            installation_guide=install_guide,
        )
