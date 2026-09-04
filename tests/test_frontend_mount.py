"""
tests/test_frontend_mount.py - Verify Static Files and Frontend Serving
========================================================================
"""

import unittest
from fastapi.testclient import TestClient
from backend.app import app


class TestFrontendMount(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_frontend_index_served(self):
        """Test GET / serves frontend index.html with new AL elements."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("PoleAnnotator AI", res.text)
        self.assertIn("annotation-canvas", res.text)
        self.assertIn("Dataset Balance & Coverage", res.text)
        self.assertIn("Active Learning Tags:", res.text)
        self.assertIn("coverage-warnings", res.text)
        self.assertIn("reject-modal", res.text)
        self.assertIn("btn-toggle-test-set", res.text)

    def test_frontend_style_served(self):
        """Test GET /style.css serves CSS stylesheet."""
        res = self.client.get("/style.css")
        self.assertEqual(res.status_code, 200)
        self.assertIn("--bg-dark", res.text)
        self.assertIn(".coverage-card", res.text)
        self.assertIn(".modal-card", res.text)

    def test_frontend_js_served(self):
        """Test GET /app.js serves JavaScript application with new AL methods."""
        res = self.client.get("/app.js")
        self.assertEqual(res.status_code, 200)
        self.assertIn("orderCornersCanonical", res.text)
        self.assertIn("loadDatasetComposition", res.text)
        self.assertIn("toggleTestSetCurrent", res.text)
        self.assertIn("confirmRejectCurrent", res.text)


if __name__ == "__main__":
    unittest.main()
