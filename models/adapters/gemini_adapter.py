"""
models/adapters/gemini_adapter.py - Optional Gemini VLM Verifier & Judge
========================================================================

Provides optional multimodal AI verification for ambiguous detections.
Assesses whether a candidate region is a utility pole and extracts visible
attributes (wood/concrete/metal, transformers, crossarms, estimated tilt).
Gracefully reports unavailable if API key is not configured.
"""

from __future__ import annotations
import os
import io
import json
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image

from models.adapters.base import BaseVerifier, DetectionBox, ModelInfo


class GeminiVerifier(BaseVerifier):
    """
    Optional verifier powered by Gemini 1.5 Flash / Pro.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None
        self._status = "not_loaded"
        self._error_msg = None

        self._check_available()

    def _check_available(self) -> bool:
        if not self.api_key:
            self._status = "unavailable"
            self._error_msg = "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set."
            return False
        self._status = "ready"
        return True

    def is_available(self) -> bool:
        return bool(self.api_key)

    def verify(
        self,
        image: Union[str, Image.Image],
        detection: DetectionBox
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "is_pole": True,
                "confidence": detection.confidence,
                "material": "unknown",
                "verified": False,
                "comment": "Gemini verifier unavailable (missing API key)."
            }

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)

            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image

            # Crop region around detection
            w, h = pil_img.size
            x0, y0, x1, y1 = detection.xyxy
            pad = 10
            crop_box = (
                max(0, int(x0 - pad)),
                max(0, int(y0 - pad)),
                min(w, int(x1 + pad)),
                min(h, int(y1 + pad))
            )
            cropped = pil_img.crop(crop_box)

            prompt = (
                "You are an expert power utility inspector. Analyze this image crop of a suspected utility pole.\n"
                "Return a JSON object with keys:\n"
                "- is_pole: boolean (true if utility/power/light/telephone pole, false if tree/car/building)\n"
                "- material: string ('wood', 'concrete', 'steel', 'composite', or 'unknown')\n"
                "- has_transformer: boolean\n"
                "- has_crossarms: boolean\n"
                "- confidence: float between 0.0 and 1.0\n"
                "- comment: short explanation (under 20 words)\n"
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=[cropped, prompt]
            )

            # Parse JSON from response
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            data["verified"] = True
            return data

        except Exception as e:
            return {
                "is_pole": True,
                "confidence": detection.confidence,
                "verified": False,
                "comment": f"Gemini verification failed: {str(e)}"
            }

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            id="gemini_verifier",
            name=f"Gemini VLM Verifier ({self.model_name})",
            model_type="vlm",
            status=self._status,
            weights_path="cloud_api",
            device="cloud",
            error_message=self._error_msg,
            installation_guide="Set GEMINI_API_KEY environment variable to enable VLM verification.",
        )
