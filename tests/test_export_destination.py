"""
tests/test_export_destination.py - Choosing Where an Export Lands
=================================================================

The export endpoints take a destination folder from the caller and create
directories and files in it. That is a write to an arbitrary filesystem path
driven by request input, so these tests spend most of their effort on the
guards: absolute-only, not over an existing file, and -- when OBB_EXPORT_ROOT
is set -- confined to that root.
"""

import importlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import backend.app as app_module
from backend.app import app, storage_mgr
from models.adapters.base import DetectionBox


def _seed(dataset_id, test_dir, names=("a.jpg", "b.jpg")):
    """A dataset with verified annotations, so an export has something to write."""
    storage_mgr.create_dataset(dataset_id, dataset_id, ["utility_pole"])
    src = Path(test_dir) / f"src_{dataset_id}"
    src.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (64, 64), (20, 40, 60)).save(src / n)
    storage_mgr.import_images(dataset_id, [src])
    for n in names:
        box = DetectionBox(
            xyxy=(4, 4, 40, 60),
            corners=[[4, 4], [40, 4], [40, 60], [4, 60]],
            model_source="HUMAN",
        )
        storage_mgr.save_annotation(dataset_id, n, [box], 64, 64,
                                    is_human=True, action="human_corrected")


class TestExportDestination(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = tempfile.mkdtemp()
        cls._saved_base = storage_mgr.base_dir
        storage_mgr.base_dir = Path(cls.test_dir) / "datasets"
        storage_mgr.base_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        storage_mgr.base_dir = cls._saved_base
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        # EXPORT_ROOT is read at import time; make sure one test's override
        # cannot leak into the next.
        self._saved_root = app_module.EXPORT_ROOT
        app_module.EXPORT_ROOT = ""

    def tearDown(self):
        app_module.EXPORT_ROOT = self._saved_root

    def test_default_destination_is_the_dataset_folder(self):
        _seed("exp_default", self.test_dir)
        res = self.client.post("/api/datasets/exp_default/export", json={})
        self.assertEqual(res.status_code, 200, res.text)
        path = Path(res.json()["result"]["export_path"])
        self.assertEqual(path, (storage_mgr.base_dir / "exp_default" / "export").resolve())
        self.assertTrue((path / "data.yaml").exists())

    def test_custom_destination_is_used_and_created(self):
        _seed("exp_custom", self.test_dir)
        target = Path(self.test_dir) / "chosen" / "yolo_obb"
        self.assertFalse(target.exists())

        res = self.client.post("/api/datasets/exp_custom/export",
                               json={"output_dir": str(target)})
        self.assertEqual(res.status_code, 200, res.text)

        self.assertEqual(Path(res.json()["result"]["export_path"]), target.resolve())
        self.assertTrue((target / "data.yaml").exists())
        self.assertTrue((target / "images").is_dir())
        self.assertTrue((target / "labels").is_dir())
        # Nothing was written to the default location.
        self.assertFalse((storage_mgr.base_dir / "exp_custom" / "export" / "data.yaml").exists())

    def test_empty_or_null_destination_falls_back_to_default(self):
        _seed("exp_blank", self.test_dir)
        for payload in ({"output_dir": None}, {"output_dir": ""}, {"output_dir": "   "}, {}):
            with self.subTest(payload=payload):
                res = self.client.post("/api/datasets/exp_blank/export", json=payload)
                self.assertEqual(res.status_code, 200, res.text)
                self.assertEqual(
                    Path(res.json()["result"]["export_path"]),
                    (storage_mgr.base_dir / "exp_blank" / "export").resolve(),
                )

    def test_relative_destination_is_refused(self):
        """
        A relative path resolves against the server's working directory, which
        the person typing it cannot see -- it would land somewhere neither of
        them predicted, so it is rejected rather than guessed at.
        """
        _seed("exp_rel", self.test_dir)
        for bad in ("exports", "./exports", "../exports", "sub/dir"):
            with self.subTest(path=bad):
                res = self.client.post("/api/datasets/exp_rel/export",
                                       json={"output_dir": bad})
                self.assertEqual(res.status_code, 400, f"{bad}: {res.text}")
                self.assertIn("absolute", res.json()["detail"].lower())

    def test_tilde_is_expanded_not_treated_as_relative(self):
        """
        `~/somewhere` is a normal way to name an absolute path, so it is
        allowed -- but the absolute check runs on the raw input, so expansion
        cannot be used to smuggle a relative-looking path through.
        """
        _seed("exp_tilde", self.test_dir)
        root = Path(self.test_dir) / "fake_home_root"
        root.mkdir(parents=True, exist_ok=True)
        app_module.EXPORT_ROOT = str(root)

        # Confined by EXPORT_ROOT, a tilde path outside it is still refused --
        # proving it went through the same resolution as any other path.
        res = self.client.post("/api/datasets/exp_tilde/export",
                               json={"output_dir": "~/poles_export"})
        self.assertEqual(res.status_code, 403, res.text)

        app_module.EXPORT_ROOT = ""
        home_target = Path.home() / "poles_export_test"
        try:
            ok = self.client.post("/api/datasets/exp_tilde/export",
                                  json={"output_dir": "~/poles_export_test"})
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertEqual(Path(ok.json()["result"]["export_path"]), home_target.resolve())
        finally:
            shutil.rmtree(home_target, ignore_errors=True)

    def test_destination_that_is_an_existing_file_is_refused(self):
        _seed("exp_file", self.test_dir)
        a_file = Path(self.test_dir) / "not_a_folder.txt"
        a_file.write_text("do not clobber me")

        res = self.client.post("/api/datasets/exp_file/export",
                               json={"output_dir": str(a_file)})
        self.assertEqual(res.status_code, 400, res.text)
        self.assertEqual(a_file.read_text(), "do not clobber me")

    def test_export_root_confines_the_destination(self):
        """With OBB_EXPORT_ROOT set, a path outside it must be refused."""
        _seed("exp_root", self.test_dir)
        root = Path(self.test_dir) / "allowed"
        root.mkdir(parents=True, exist_ok=True)
        outside = Path(self.test_dir) / "elsewhere"
        app_module.EXPORT_ROOT = str(root)

        ok = self.client.post("/api/datasets/exp_root/export",
                              json={"output_dir": str(root / "run1")})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertTrue((root / "run1" / "data.yaml").exists())

        denied = self.client.post("/api/datasets/exp_root/export",
                                  json={"output_dir": str(outside)})
        self.assertEqual(denied.status_code, 403, denied.text)
        self.assertIn("OBB_EXPORT_ROOT", denied.json()["detail"])
        self.assertFalse(outside.exists(), "a refused export must not create its directory")

    def test_export_root_cannot_be_escaped_with_dot_dot(self):
        _seed("exp_escape", self.test_dir)
        root = Path(self.test_dir) / "allowed2"
        root.mkdir(parents=True, exist_ok=True)
        app_module.EXPORT_ROOT = str(root)

        escape = str(root / ".." / "escaped")
        res = self.client.post("/api/datasets/exp_escape/export",
                               json={"output_dir": escape})
        self.assertEqual(res.status_code, 403, res.text)
        self.assertFalse((Path(self.test_dir) / "escaped").exists())

    def test_test_set_and_hard_cases_accept_a_destination_too(self):
        _seed("exp_other", self.test_dir)
        storage_mgr.toggle_test_set("exp_other", "a.jpg")

        target = Path(self.test_dir) / "other_exports"
        res = self.client.post("/api/datasets/exp_other/export_test_set",
                               json={"output_dir": str(target)})
        self.assertEqual(res.status_code, 200, res.text)

        hard = self.client.post("/api/datasets/exp_other/export_hard_cases",
                                json={"output_dir": str(target)})
        self.assertEqual(hard.status_code, 200, hard.text)


if __name__ == "__main__":
    unittest.main()
