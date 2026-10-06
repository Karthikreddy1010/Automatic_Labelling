"""
tests/ml_guard.py - Skip markers for tests that need the optional ML stack
==========================================================================

The app supports two installs (see requirements-core.txt):

  * core      -- annotation workspace only, no torch/ultralytics/transformers
  * full      -- everything, including the AI labeling pipeline

Tests that exercise real model loading can only run on a full install. They
use these markers so that `python -m unittest discover -s tests` is GREEN on a
core install -- reporting them as skipped, which is accurate, instead of
failed, which would hide genuine regressions in the noise.
"""

import importlib.util
import unittest


def module_installed(name: str) -> bool:
    """True when `name` can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


ML_STACK_INSTALLED = module_installed("torch")


def requires(*modules: str):
    """Skip unless every named module is installed."""
    missing = [m for m in modules if not module_installed(m)]
    return unittest.skipIf(
        bool(missing),
        f"needs {', '.join(missing)} -- install with: pip install -r requirements.txt",
    )


requires_ml = requires("torch")
