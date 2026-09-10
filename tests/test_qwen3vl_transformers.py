"""
tests/test_qwen3vl_transformers.py - Qwen3-VL-8B-Instruct Transformers Backend
==================================================================================

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

from models.adapters.qwen3vl_transformers_backend import Qwen3VLTransformersBackend, _looks_like_hub_id


class TestHubIdDetection(unittest.TestCase):
    """Distinguishing a literal filesystem directory from a Hugging Face Hub
    id (e.g. 'Qwen/Qwen3-VL-8B-Instruct', already cached locally under
    ~/.cache/huggingface/hub/) matters because Path(...).exists() would
    incorrectly reject a valid, already-cached Hub id -- it's not a real
    directory relative to the current working directory."""

    def test_hub_id_shaped_strings_are_detected(self):
        self.assertTrue(_looks_like_hub_id("Qwen/Qwen3-VL-8B-Instruct"))
        self.assertTrue(_looks_like_hub_id("meta-llama/Llama-3.1-8B"))

    def test_absolute_unix_path_is_not_a_hub_id(self):
        self.assertFalse(_looks_like_hub_id("/home/maska/models/Qwen3-VL-8B-Instruct"))

    def test_home_relative_path_is_not_a_hub_id(self):
        self.assertFalse(_looks_like_hub_id("~/models/Qwen3-VL-8B-Instruct"))

    def test_multi_segment_path_is_not_a_hub_id(self):
        self.assertFalse(_looks_like_hub_id("C:/models/Qwen3-VL-8B-Instruct"))

    def test_windows_backslash_path_is_not_a_hub_id(self):
        self.assertFalse(_looks_like_hub_id(r"C:\models\Qwen3-VL-8B-Instruct"))

    def test_empty_string_is_not_a_hub_id(self):
        self.assertFalse(_looks_like_hub_id(""))


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

    def test_hub_id_not_cached_here_fails_via_from_pretrained_not_path_precheck(self):
        # A Hub-id-shaped model_path must NOT be rejected by the fast
        # Path.exists() pre-check (it isn't a real directory) -- it should
        # reach transformers' own from_pretrained(local_files_only=True),
        # which fails honestly here since this machine has no local HF cache
        # entry for it. The resulting error must come from that call, not
        # from our own "path does not exist" pre-check message.
        backend = Qwen3VLTransformersBackend(model_path="Qwen/Qwen3-VL-8B-Instruct-not-a-real-cached-repo")
        ok = backend.load()
        self.assertFalse(ok)
        error = backend.get_status()["error"] or ""
        self.assertNotIn("does not exist", error)


if __name__ == "__main__":
    unittest.main()
