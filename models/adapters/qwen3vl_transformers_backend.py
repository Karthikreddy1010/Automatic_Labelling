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
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_SINGLETONS: Dict[str, "Qwen3VLTransformersBackend"] = {}
_SINGLETON_LOCK = threading.Lock()

_HUB_ID_RE = re.compile(r"[\w.-]+/[\w.-]+")


def _looks_like_hub_id(model_path: str) -> bool:
    """
    True for a bare "Org/Repo"-shaped string (e.g. "Qwen/Qwen3-VL-8B-Instruct")
    that should be resolved through transformers' own local Hugging Face Hub
    cache lookup, rather than treated as a literal filesystem directory.
    False for anything that looks like an actual path: starts with "/" or
    "~" (Unix absolute/home-relative), contains a backslash (Windows), or
    has more than one path segment (e.g. "C:/models/x", "a/b/c").
    """
    if not model_path or model_path.startswith(("/", "~")) or "\\" in model_path:
        return False
    return bool(_HUB_ID_RE.fullmatch(model_path))


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
            if not self.model_path:
                self._error = (
                    "No Qwen3-VL-8B-Instruct model_path configured. Set "
                    "verification_pipeline.qwen.transformers.model_path (or QWEN_MODEL_PATH) to "
                    "either a local checkpoint directory or a Hugging Face Hub id "
                    "(e.g. 'Qwen/Qwen3-VL-8B-Instruct') already cached locally."
                )
                return False

            # A literal filesystem path is pre-checked for existence so a
            # typo'd directory fails fast with a clear message before ever
            # importing transformers. A bare "Org/Repo"-shaped string (e.g.
            # "Qwen/Qwen3-VL-8B-Instruct") is a Hugging Face Hub id, not a
            # real directory relative to the current working directory --
            # Path(...).exists() would incorrectly reject a valid,
            # already-cached Hub id, so that case skips straight to
            # from_pretrained's own local-cache resolution below (still
            # local_files_only=True, so it still never downloads).
            if not _looks_like_hub_id(self.model_path) and not Path(self.model_path).exists():
                self._error = (
                    f"Local Qwen3-VL-8B-Instruct path does not exist: '{self.model_path}'. "
                    "This backend never downloads from the Hub -- point "
                    "verification_pipeline.qwen.transformers.model_path (or QWEN_MODEL_PATH) "
                    "at a local checkpoint directory, or a Hugging Face Hub id already cached locally."
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
