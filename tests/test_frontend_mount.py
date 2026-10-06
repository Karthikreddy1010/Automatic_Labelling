"""
tests/test_frontend_mount.py - Verify Static Files and Frontend Serving
========================================================================
"""

import unittest
from html.parser import HTMLParser
from fastapi.testclient import TestClient
from backend.app import app

# Elements that are implicitly closed by the parser rather than by a tag.
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class _IdAncestry(HTMLParser):
    """Map each id= element to the ids of the elements it is nested inside."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._stack = []
        self.parents = {}

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        el_id = dict(attrs).get("id")
        if el_id:
            self.parents[el_id] = [i for _, i in self._stack if i]
        self._stack.append((tag, el_id))

    def handle_endtag(self, tag):
        # Close the innermost matching element, discarding anything left
        # unclosed inside it -- the same recovery a browser performs.
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                return


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
        # Guided tour entry points (frontend/tour.js)
        self.assertIn("btn-tour", res.text)
        self.assertIn("tour.js", res.text)
        self.assertIn('id="toast"', res.text)

    def test_modals_are_siblings_not_nested(self):
        """
        Every modal overlay must be a top-level element.

        An unclosed <div> once left #reject-modal nested inside
        #shortcuts-modal, so it inherited that modal's `hidden` class and
        rendered 0x0 -- clicking Reject silently did nothing.
        """
        res = self.client.get("/")
        page = _IdAncestry()
        page.feed(res.text)

        modal_ids = ["batch-modal", "import-modal", "shortcuts-modal", "reject-modal"]
        for modal_id in modal_ids:
            self.assertIn(modal_id, page.parents, f"{modal_id} missing from the page")
            nested_in = [p for p in page.parents[modal_id] if p in modal_ids]
            self.assertEqual(
                nested_in, [], f"#{modal_id} is nested inside {nested_in} -- it will inherit `hidden`"
            )

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
        # Review-advance helpers: identity captured before the save request,
        # and the post-reload selection resolved by filename.
        self.assertIn("captureAdvanceTarget", res.text)
        self.assertIn("resolveSelectionIndex", res.text)
        self.assertIn("reviewActionInFlight", res.text)

    def test_frontend_tour_js_served(self):
        """GET /tour.js serves the guided tour module."""
        res = self.client.get("/tour.js")
        self.assertEqual(res.status_code, 200)
        self.assertIn("PoleTour", res.text)
        self.assertIn("maybeAutoStart", res.text)
        self.assertIn("tour-spotlight", res.text)


if __name__ == "__main__":
    unittest.main()
