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
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from models.adapters.base import BaseVerifier, DetectionBox, ModelInfo

VERIFY_PROMPT_TEMPLATE = (
    "You are an expert electric utility inspector verifying an oriented bounding box (OBB)\n"
    "candidate produced by an automated detector, for a utility-pole labeling dataset.\n"
    "You are given the original image for context and a focused crop of the candidate.\n"
    "Candidate Bounding Box (xyxy, pixels): {xyxy}\n"
    "Candidate OBB Corners: {corners}\n"
    "Detector Source: {model_source}, Detector Confidence: {confidence}\n\n"
    "Utility poles may be made of wood, steel, concrete, composite, or other materials --\n"
    "do not require wood. Do not reject a candidate merely because it is partially occluded,\n"
    "wires aren't clearly visible, it is slightly tilted, or it has unusual equipment, or only\n"
    "part of the pole is visible -- those are normal cases. Only mark visibility as\n"
    "severely_occluded or too_distant when truly hard to assess; do not force an accept/reject\n"
    "for those, prefer review.\n\n"
    "Classify the candidate crop and respond ONLY with a valid JSON object matching this exact schema:\n"
    "{{\n"
    '  "class": "electric_utility_pole",\n'
    '  "material": "wood",\n'
    '  "visibility": "mostly_visible",\n'
    '  "orientation": "slightly_tilted",\n'
    '  "semantic_confidence": 0.94,\n'
    '  "decision": "accept",\n'
    '  "reason": "Candidate has the characteristic elongated pole structure and visible electrical equipment."\n'
    "}}\n\n"
    'class must be one of: "electric_utility_pole", "non_utility_pole", "tree", '
    '"building_or_structure", "street_light_or_lamp_post", "other_object", "uncertain".\n'
    'material must be one of: "wood", "steel", "concrete", "composite", "unknown".\n'
    'visibility must be one of: "mostly_visible", "partially_occluded", "severely_occluded", "too_distant".\n'
    'orientation must be one of: "vertical", "slightly_tilted", "strongly_tilted".\n'
    'decision must be one of: "accept", "review", "reject".\n'
    "semantic_confidence is a float between 0.0 and 1.0."
)

_VALID_CLASSES = {
    "electric_utility_pole", "non_utility_pole", "tree",
    "building_or_structure", "street_light_or_lamp_post", "other_object", "uncertain",
}
_VALID_MATERIALS = {"wood", "steel", "concrete", "composite", "unknown"}
_VALID_VISIBILITY = {"mostly_visible", "partially_occluded", "severely_occluded", "too_distant"}
_VALID_ORIENTATION = {"vertical", "slightly_tilted", "strongly_tilted"}
_VALID_DECISION = {"accept", "review", "reject"}


def _normalize_qwen_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a parsed Qwen JSON response into the canonical schema, tolerating
    missing keys or unrecognized enum values (normalized to "uncertain" /
    "unknown" rather than raising) -- a malformed field must never crash the
    pipeline, it should just fall back to "we don't know" for that field.
    """
    def _enum(key: str, valid: set, default: str) -> str:
        v = str(data.get(key, default)).strip().lower() if data.get(key) is not None else default
        return v if v in valid else default

    cls = _enum("class", _VALID_CLASSES, "uncertain")
    material = _enum("material", _VALID_MATERIALS, "unknown")
    visibility = _enum("visibility", _VALID_VISIBILITY, "mostly_visible")
    orientation = _enum("orientation", _VALID_ORIENTATION, "vertical")
    decision = _enum("decision", _VALID_DECISION, "review")

    try:
        semantic_confidence = float(data.get("semantic_confidence", 0.5))
        semantic_confidence = max(0.0, min(1.0, semantic_confidence))
    except (TypeError, ValueError):
        semantic_confidence = 0.5

    return {
        "class": cls,
        "material": material,
        "visibility": visibility,
        "orientation": orientation,
        "semantic_confidence": round(semantic_confidence, 4),
        "decision": decision,
        "reason": str(data.get("reason", ""))[:500] or "No reason provided by Qwen.",
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
    ):
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

        self._status = "ready" if (
            self.ollama_model or self.api_key or (self.local_model_path and Path(self.local_model_path).exists())
        ) else "unavailable"
        self._error_msg = None if self._status == "ready" else (
            f"No vision-capable Qwen model (a '...vl...' tag) found in Ollama at {self.ollama_base_url} "
            "-- a text-only Qwen model (e.g. qwen2.5-coder) does not count, since this verifier must "
            "look at an image crop -- and QWEN_API_KEY/DASHSCOPE_API_KEY not set. Run `ollama pull "
            "qwen2.5vl:7b` (or similar) to enable the local path."
        )

    def is_available(self) -> bool:
        return self._status == "ready"

    def _crop_and_encode(self, image: Union[str, np.ndarray, Image.Image], detection: DetectionBox) -> str:
        """Crop to the candidate region (padded) and return a base64 JPEG."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        w, h = pil_img.size
        x0, y0, x1, y1 = detection.xyxy
        pad_w = 0.15 * (x1 - x0)
        pad_h = 0.15 * (y1 - y0)
        crop_box = (
            max(0, int(x0 - pad_w)),
            max(0, int(y0 - pad_h)),
            min(w, int(x1 + pad_w)),
            min(h, int(y1 + pad_h)),
        )
        cropped = pil_img.crop(crop_box)

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _verify_via_ollama(self, b64_image: str, prompt: str) -> Dict[str, Any]:
        payload = json.dumps({
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt, "images": [b64_image]}],
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

    def _verify_via_dashscope(self, b64_image: str, prompt: str) -> Dict[str, Any]:
        import dashscope

        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{b64_image}"},
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

        if not self.ollama_model and not self.api_key:
            # local_model_path was configured, but local HF Qwen-VL inference
            # isn't implemented -- report that plainly instead of faking a result.
            return {
                "status": "unavailable",
                "decision": "unavailable",
                "needs_human_review": True,
                "reason": (
                    "MODEL UNAVAILABLE: local HuggingFace Qwen-VL inference (QWEN_MODEL_PATH) is not "
                    "implemented. Run Ollama with a Qwen(-VL) model pulled, or set QWEN_API_KEY/DASHSCOPE_API_KEY."
                ),
                **_unavailable_fields,
            }

        try:
            b64_image = self._crop_and_encode(image, detection)
            prompt = VERIFY_PROMPT_TEMPLATE.format(
                xyxy=detection.xyxy, corners=detection.corners,
                model_source=detection.model_source, confidence=detection.confidence,
            )

            if self.ollama_model:
                raw = self._verify_via_ollama(b64_image, prompt)
            else:
                raw = self._verify_via_dashscope(b64_image, prompt)

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
                # Legacy/compat fields derived from the new schema.
                "is_utility_pole": norm["class"] == "electric_utility_pole",
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
        backend = f"ollama:{self.ollama_model}" if self.ollama_model else (self.model_name if self.api_key else self.local_model_path)
        return ModelInfo(
            id="qwen_verifier",
            name="Qwen VLM Verifier" + (f" (Ollama: {self.ollama_model})" if self.ollama_model else ""),
            model_type="verifier",
            status=self._status,
            weights_path=backend,
            device="local_ollama" if self.ollama_model else "cloud_or_local",
            error_message=self._error_msg,
            installation_guide=(
                "To enable Qwen visual verification: run a local Ollama server with a Qwen(-VL) model "
                "pulled (preferred, no API key needed), or set the QWEN_API_KEY/DASHSCOPE_API_KEY "
                "environment variable for the cloud API."
            ),
        )
