"""
tests/test_core_install.py - The App Runs Without the ML Stack
==============================================================

The annotation workspace is usable with only the light dependencies in
requirements-core.txt (no torch, ultralytics, transformers). These tests pin
that contract down, because it is easy to break by adding one module-level
`import torch` to something the server imports at startup -- which is exactly
what used to make `python run_backend.py` die before the server came up.

They assert behaviour that must hold in BOTH installs:
  * the server imports and starts;
  * hardware detection degrades to CPU instead of raising;
  * missing models are reported as `unavailable`, never as a crash;
  * an AI endpoint called without its model returns an actionable 503.

Nothing here requires torch to be absent -- with the full stack installed the
same assertions hold, so the suite is valid either way.
"""

import ast
import importlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app, storage_mgr

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages that must never be imported at module scope by anything on the
# server's startup import path.
HEAVY = {"torch", "torchvision", "ultralytics", "transformers", "accelerate"}

# Modules the server imports at startup (backend.app's own import block).
STARTUP_MODULES = [
    REPO_ROOT / "backend" / "app.py",
    REPO_ROOT / "backend" / "storage.py",
    REPO_ROOT / "src" / "geometry_obb.py",
] + sorted((REPO_ROOT / "models" / "adapters").glob("*.py"))


def _module_level_imports(path: Path):
    """Top-level import names in `path`, ignoring imports inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # A guarded `try: import torch / except ImportError:` is fine --
            # that is the supported way to depend on an optional package.
            continue
    return names


class TestRunsWithoutMLStack(unittest.TestCase):

    def test_no_unguarded_heavy_imports_on_startup_path(self):
        """
        No module the server loads at import time may import the ML stack
        unguarded. Adapters import their own heavy dependency lazily inside
        load()/is_available(); that is what keeps core-only installs working.
        """
        offenders = []
        for path in STARTUP_MODULES:
            heavy = _module_level_imports(path) & HEAVY
            if heavy:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(heavy)}")
        self.assertEqual(
            offenders, [],
            "Unguarded heavy import(s) on the startup path -- the server will "
            "refuse to start on a core-only install:\n  " + "\n  ".join(offenders),
        )

    def test_detect_hardware_degrades_to_cpu(self):
        """detect_hardware() must answer, not raise, when torch is absent."""
        base = importlib.import_module("models.adapters.base")
        self.assertTrue(hasattr(base, "TORCH_AVAILABLE"))

        for requested in ("AUTO", "CPU", "CUDA"):
            device, info = base.detect_hardware(requested)
            self.assertIn(device, ("cpu", "cuda"))
            if not base.TORCH_AVAILABLE:
                self.assertEqual(device, "cpu", f"{requested} should fall back to CPU")
                self.assertFalse(info.cuda_available)

    def test_requirements_core_is_consistent_with_full(self):
        """
        requirements.txt must stay a superset of requirements-core.txt at the
        same pins, so installing either (or both) gives one coherent set.
        opencv is the one allowed difference: core takes the headless wheel.
        """
        def pins(path):
            out = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].strip()
                m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==(.+)$", line)
                if m:
                    out[m.group(1).lower()] = m.group(3).strip()
            return out

        core = pins(REPO_ROOT / "requirements-core.txt")
        full = pins(REPO_ROOT / "requirements.txt")
        self.assertTrue(core, "requirements-core.txt parsed to nothing")

        mismatched = []
        for name, version in core.items():
            if name == "opencv-python-headless":
                # Full install uses opencv-python; both satisfy `import cv2`.
                counterpart = full.get("opencv-python")
                if counterpart and counterpart != version:
                    mismatched.append(f"opencv: core={version} full={counterpart}")
                continue
            if name not in full:
                mismatched.append(f"{name} missing from requirements.txt")
            elif full[name] != version:
                mismatched.append(f"{name}: core={version} full={full[name]}")
        self.assertEqual(mismatched, [], "requirements drift:\n  " + "\n  ".join(mismatched))


class TestGracefulDegradationAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = tempfile.mkdtemp()
        cls._saved_base = storage_mgr.base_dir
        storage_mgr.base_dir = Path(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        storage_mgr.base_dir = cls._saved_base
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_hardware_endpoint_answers(self):
        res = self.client.get("/api/system/hardware")
        self.assertEqual(res.status_code, 200)
        self.assertIn(res.json()["resolved_device"], ("cpu", "cuda"))

    def test_model_status_reports_unavailable_not_error(self):
        """A missing model is a reported status, never a 500."""
        res = self.client.get("/api/models/status")
        self.assertEqual(res.status_code, 200)
        models = res.json()["models"]
        self.assertGreaterEqual(len(models), 4)
        for m in models:
            self.assertIn("status", m)
            self.assertNotEqual(m["status"], "error", f"{m['id']} reported as error")

    def test_ai_label_without_models_returns_actionable_503(self):
        """
        Asking for AI labeling with no models installed must explain itself and
        name the fix, rather than returning a bare 500 or a cryptic string.
        """
        from backend.app import dino_adapter

        if dino_adapter.is_available():
            self.skipTest("Grounding DINO is installed; nothing to degrade")

        self.client.post("/api/datasets", json={
            "dataset_id": "degrade_ds", "name": "Degrade", "classes": ["utility_pole"],
        })
        img_dir = Path(self.test_dir) / "degrade_ds" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import cv2
        cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((48, 48, 3), dtype=np.uint8))
        storage_mgr.import_images("degrade_ds", [img_dir])

        res = self.client.post("/api/inference/detect", json={
            "dataset_id": "degrade_ds", "filename": "a.jpg", "mode": "AI_LABEL",
        })
        self.assertEqual(res.status_code, 503, res.text)
        detail = res.json().get("detail", "")
        self.assertIn("pip install -r requirements.txt", detail)
        self.assertIn("not installed", detail)


if __name__ == "__main__":
    unittest.main()
