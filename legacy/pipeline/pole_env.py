"""Shared environment setup for the pole pipeline.

Import this first in any entry-point script. It forces HuggingFace offline mode
so cached model weights (SAM, SAM2.1, SAM3, YOLO) load without contacting the
Hub. This makes the pipeline robust to an expired/invalid HF token: once the
models are cached, no authentication is needed.

Set POLE_ALLOW_ONLINE=1 to disable offline mode (e.g. first-time download of a
new model).
"""
import os

if os.environ.get("POLE_ALLOW_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
