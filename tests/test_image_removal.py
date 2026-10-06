"""
tests/test_image_removal.py - Removing Images From a Dataset
=============================================================

Covers the "I uploaded the wrong folder" path: taking images back out, either
one at a time, as a ticked selection, as a whole upload, or all of them.

The stakes are asymmetric. Leaving a derived file behind is not cosmetic -- a
later import of a file with the same name would silently inherit the previous
image's annotations, masks and history. Deleting one file too many destroys
labelling work outright. So these tests check both directions: everything
belonging to the removed image goes, and nothing belonging to any other image
is touched.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app, storage_mgr
from backend.storage import DatasetManager
from models.adapters.base import DetectionBox


def _write_images(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (64, 64), (10, 20, 30)).save(folder / n)
    return folder


def _box():
    return DetectionBox(
        xyxy=(1, 1, 30, 60),
        corners=[[1, 1], [30, 1], [30, 60], [1, 60]],
        model_source="HUMAN",
    )


class TestDeleteImagesStorage(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.dm = DatasetManager(str(self.dir))
        self.dm.create_dataset("d", "D", ["utility_pole"])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _inventory(self, sub):
        p = self.dir / "d" / sub
        return sorted(f.name for f in p.iterdir()) if p.is_dir() else []

    def test_removes_image_and_every_derived_artifact(self):
        src = _write_images(self.dir / "src", ["a.jpg", "b.jpg"])
        self.dm.import_images("d", [src])
        for name in ("a.jpg", "b.jpg"):
            self.dm.save_mask("d", name, 0, np.ones((64, 64), dtype=np.uint8))
            self.dm.save_annotation("d", name, [_box()], 64, 64,
                                    is_human=True, action="human_corrected")
        (self.dir / "d" / "history" / "a_1759750000000.json").write_text("{}")

        result = self.dm.delete_images("d", ["a.jpg"])

        self.assertEqual(result["deleted"], ["a.jpg"])
        self.assertNotIn("a.jpg", self._inventory("images"))
        self.assertNotIn("a.json", self._inventory("annotations"))
        self.assertNotIn("a.txt", self._inventory("annotations"))
        self.assertNotIn("a_0.png", self._inventory("masks"))
        self.assertNotIn("a_1759750000000.json", self._inventory("history"))

        meta = self.dm.get_dataset("d")
        self.assertNotIn("a.jpg", meta["images"])
        self.assertEqual(meta["image_count"], 1)

        # The other image is untouched.
        self.assertIn("b.jpg", self._inventory("images"))
        self.assertIn("b.json", self._inventory("annotations"))
        self.assertIn("b_0.png", self._inventory("masks"))

    def test_does_not_delete_a_longer_named_neighbours_files(self):
        """
        `pole1` must not take `pole1_closeup`'s files with it. Mask and history
        names are `<stem>_<digits>`, so a `<stem>_*` glob would match the
        neighbour -- the regex is anchored to digits precisely to stop that.
        """
        src = _write_images(self.dir / "src", ["pole1.jpg", "pole1_closeup.jpg"])
        self.dm.import_images("d", [src])
        for name in ("pole1.jpg", "pole1_closeup.jpg"):
            self.dm.save_mask("d", name, 0, np.ones((64, 64), dtype=np.uint8))
            self.dm.save_annotation("d", name, [_box()], 64, 64,
                                    is_human=True, action="human_corrected")
        hist = self.dir / "d" / "history"
        (hist / "pole1_1759750000000.json").write_text("{}")
        (hist / "pole1_closeup_1759750000000.json").write_text("{}")
        (hist / "pole1_notes.json").write_text("{}")  # not a history record

        self.dm.delete_images("d", ["pole1.jpg"])

        self.assertIn("pole1_closeup.jpg", self._inventory("images"))
        self.assertIn("pole1_closeup.json", self._inventory("annotations"))
        self.assertIn("pole1_closeup_0.png", self._inventory("masks"))
        self.assertIn("pole1_closeup_1759750000000.json", self._inventory("history"))
        self.assertIn("pole1_notes.json", self._inventory("history"))
        self.assertNotIn("pole1_1759750000000.json", self._inventory("history"))
        self.assertNotIn("pole1_0.png", self._inventory("masks"))

    def test_unknown_filenames_are_reported_not_raised(self):
        src = _write_images(self.dir / "src", ["a.jpg"])
        self.dm.import_images("d", [src])
        result = self.dm.delete_images("d", ["a.jpg", "ghost.jpg"])
        self.assertEqual(result["deleted"], ["a.jpg"])
        self.assertEqual(result["not_found"], ["ghost.jpg"])

    def test_path_traversal_filename_is_rejected(self):
        src = _write_images(self.dir / "src", ["a.jpg"])
        self.dm.import_images("d", [src])
        result = self.dm.delete_images("d", ["../../etc/passwd"])
        self.assertEqual(result["deleted"], [])
        self.assertIn("a.jpg", self._inventory("images"))

    def test_missing_dataset_raises(self):
        with self.assertRaises(ValueError):
            self.dm.delete_images("nope", ["a.jpg"])


class TestImportBatches(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.dm = DatasetManager(str(self.dir))
        self.dm.create_dataset("d", "D", ["utility_pole"])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_each_import_is_its_own_batch(self):
        a = _write_images(self.dir / "a", ["a1.jpg", "a2.jpg"])
        b = _write_images(self.dir / "b", ["b1.jpg", "b2.jpg", "b3.jpg"])
        self.dm.import_images("d", [a])
        self.dm.import_images("d", [b])

        batches = self.dm.get_import_batches("d")
        self.assertEqual([x["count"] for x in batches], [2, 3])

    def test_shared_batch_id_groups_per_file_uploads(self):
        """
        The browser uploads one file per request. With a shared batch_id they
        must register as ONE undoable upload, not three of one image each.
        """
        for name in ("x1.jpg", "x2.jpg", "x3.jpg"):
            self.dm.import_uploaded_files("d", [(name, b"\xff\xd8\xff")], batch_id="sharedbatch")
        batches = self.dm.get_import_batches("d")
        self.assertEqual(len(batches), 1, batches)
        self.assertEqual(batches[0]["count"], 3)

    def test_delete_import_batch_removes_only_that_upload(self):
        a = _write_images(self.dir / "a", ["a1.jpg", "a2.jpg"])
        b = _write_images(self.dir / "b", ["b1.jpg", "b2.jpg"])
        self.dm.import_images("d", [a])
        self.dm.import_images("d", [b])
        newest = self.dm.get_import_batches("d")[-1]["batch_id"]

        self.dm.delete_import_batch("d", newest)

        remaining = sorted(self.dm.get_dataset("d")["images"])
        self.assertEqual(remaining, ["a1.jpg", "a2.jpg"])
        self.assertEqual(len(self.dm.get_import_batches("d")), 1)

    def test_reimporting_existing_files_records_no_empty_batch(self):
        a = _write_images(self.dir / "a", ["a1.jpg"])
        self.dm.import_images("d", [a])
        self.dm.import_images("d", [a])  # same file again -- nothing new
        self.assertEqual(len(self.dm.get_import_batches("d")), 1)

    def test_clear_removes_everything_but_keeps_the_dataset(self):
        a = _write_images(self.dir / "a", ["a1.jpg", "a2.jpg"])
        self.dm.import_images("d", [a])
        self.dm.save_annotation("d", "a1.jpg", [_box()], 64, 64,
                                is_human=True, action="accepted")

        self.dm.clear_dataset_images("d")

        meta = self.dm.get_dataset("d")
        self.assertEqual(meta["images"], {})
        self.assertEqual(meta["image_count"], 0)
        self.assertEqual(meta.get("import_batches"), [])
        self.assertEqual(meta["classes"], ["utility_pole"])
        self.assertEqual(list((self.dir / "d" / "images").iterdir()), [])


class TestDeleteWholeDataset(unittest.TestCase):
    """
    Deleting a dataset rmtree's a directory built from a user-supplied id, so
    the guards matter more than the happy path: a crafted id must never reach
    outside the data directory.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.base = self.root / "datasets"
        self.base.mkdir()
        self.dm = DatasetManager(str(self.base))
        self.dm.create_dataset("keepme", "Keep Me", ["utility_pole"])
        self.dm.create_dataset("killme", "Kill Me", ["utility_pole"])
        # Something outside the data directory that must survive everything.
        self.outsider = self.root / "precious"
        self.outsider.mkdir()
        (self.outsider / "do_not_delete.txt").write_text("important")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_deletes_the_directory_and_reports_what_went(self):
        src = _write_images(self.root / "src", ["a.jpg", "b.jpg"])
        self.dm.import_images("killme", [src])

        result = self.dm.delete_dataset("killme")

        self.assertTrue(result["deleted"])
        self.assertEqual(result["dataset_id"], "killme")
        self.assertEqual(result["image_count"], 2)
        self.assertFalse((self.base / "killme").exists())
        self.assertIsNone(self.dm.get_dataset("killme"))
        # Other datasets are untouched.
        self.assertTrue((self.base / "keepme").exists())
        self.assertEqual([d["dataset_id"] for d in self.dm.list_datasets()], ["keepme"])

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            self.dm.delete_dataset("never_existed")

    def test_traversal_ids_are_refused_and_delete_nothing(self):
        for bad in ["../precious", "..", ".", "", "../../tmp", "keepme/../precious"]:
            with self.subTest(dataset_id=bad):
                with self.assertRaises(ValueError):
                    self.dm.delete_dataset(bad)
        self.assertTrue((self.outsider / "do_not_delete.txt").exists())
        self.assertTrue((self.base / "keepme").exists())

    def test_a_directory_without_metadata_is_not_a_dataset(self):
        stray = self.base / "not_a_dataset"
        stray.mkdir()
        (stray / "something.txt").write_text("x")
        with self.assertRaises(ValueError):
            self.dm.delete_dataset("not_a_dataset")
        self.assertTrue(stray.exists())

    def test_id_can_be_reused_after_deletion(self):
        self.dm.delete_dataset("killme")
        meta = self.dm.create_dataset("killme", "Kill Me Again", ["utility_pole"])
        self.assertEqual(meta["dataset_id"], "killme")
        self.assertEqual(meta["images"], {})


class TestRemovalAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = tempfile.mkdtemp()
        cls._saved = storage_mgr.base_dir
        storage_mgr.base_dir = Path(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        storage_mgr.base_dir = cls._saved
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _fresh(self, ds_id, names):
        self.client.post("/api/datasets", json={
            "dataset_id": ds_id, "name": ds_id, "classes": ["utility_pole"]})
        src = _write_images(Path(self.test_dir) / f"src_{ds_id}", names)
        storage_mgr.import_images(ds_id, [src])

    def test_delete_one_image(self):
        self._fresh("api_one", ["a.jpg", "b.jpg"])
        res = self.client.delete("/api/datasets/api_one/images/a.jpg")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["deleted"], ["a.jpg"])
        listed = self.client.get("/api/datasets/api_one/images").json()["images"]
        self.assertEqual([i["filename"] for i in listed], ["b.jpg"])

    def test_delete_unknown_image_is_404(self):
        self._fresh("api_404", ["a.jpg"])
        res = self.client.delete("/api/datasets/api_404/images/ghost.jpg")
        self.assertEqual(res.status_code, 404)

    def test_bulk_delete(self):
        self._fresh("api_bulk", ["a.jpg", "b.jpg", "c.jpg"])
        res = self.client.post("/api/datasets/api_bulk/images/delete",
                               json={"filenames": ["a.jpg", "c.jpg"]})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(sorted(res.json()["deleted"]), ["a.jpg", "c.jpg"])
        listed = self.client.get("/api/datasets/api_bulk/images").json()["images"]
        self.assertEqual([i["filename"] for i in listed], ["b.jpg"])

    def test_bulk_delete_with_no_filenames_is_400(self):
        self._fresh("api_empty", ["a.jpg"])
        res = self.client.post("/api/datasets/api_empty/images/delete", json={"filenames": []})
        self.assertEqual(res.status_code, 400)

    def test_batches_listed_and_undoable(self):
        self._fresh("api_batch", ["a.jpg"])
        src2 = _write_images(Path(self.test_dir) / "src_api_batch2", ["z1.jpg", "z2.jpg"])
        storage_mgr.import_images("api_batch", [src2])

        batches = self.client.get("/api/datasets/api_batch/import_batches").json()["batches"]
        self.assertEqual(len(batches), 2)

        newest = batches[-1]["batch_id"]
        res = self.client.delete(f"/api/datasets/api_batch/import_batches/{newest}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(sorted(res.json()["deleted"]), ["z1.jpg", "z2.jpg"])

        listed = self.client.get("/api/datasets/api_batch/images").json()["images"]
        self.assertEqual([i["filename"] for i in listed], ["a.jpg"])

    def test_undoing_an_unknown_batch_is_404(self):
        self._fresh("api_nobatch", ["a.jpg"])
        res = self.client.delete("/api/datasets/api_nobatch/import_batches/deadbeef")
        self.assertEqual(res.status_code, 404)

    def test_clear_images(self):
        self._fresh("api_clear", ["a.jpg", "b.jpg"])
        res = self.client.post("/api/datasets/api_clear/clear_images")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(sorted(res.json()["deleted"]), ["a.jpg", "b.jpg"])
        listed = self.client.get("/api/datasets/api_clear/images").json()["images"]
        self.assertEqual(listed, [])
        # Dataset itself still exists and can take a fresh batch.
        self.assertEqual(self.client.get("/api/datasets/api_clear").status_code, 200)

    def test_delete_dataset_endpoint(self):
        self._fresh("api_doomed", ["a.jpg"])
        self.assertEqual(self.client.get("/api/datasets/api_doomed").status_code, 200)

        res = self.client.delete("/api/datasets/api_doomed")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deleted"])

        self.assertEqual(self.client.get("/api/datasets/api_doomed").status_code, 404)
        listed = [d["dataset_id"] for d in self.client.get("/api/datasets").json()["datasets"]]
        self.assertNotIn("api_doomed", listed)

    def test_delete_unknown_dataset_is_404(self):
        res = self.client.delete("/api/datasets/no_such_dataset")
        self.assertEqual(res.status_code, 404)

    def test_upload_response_reports_its_batch_id(self):
        self._fresh("api_reports", ["a.jpg"])
        batches = self.client.get("/api/datasets/api_reports/import_batches").json()["batches"]
        self.assertTrue(batches and batches[-1]["batch_id"])


if __name__ == "__main__":
    unittest.main()
