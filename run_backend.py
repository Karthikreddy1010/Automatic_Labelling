"""
run_backend.py - One-Command Launcher for PoleAnnotator AI Platform
===================================================================

Starts the FastAPI server with CORS, model adapters, and static frontend.
Open http://127.0.0.1:8000 in your browser.

Usage:
    python run_backend.py [--port 8000] [--host 127.0.0.1] [--no-browser]

The implementation now lives in backend/cli.py so that it ships inside the
installed package and the `poleannotator` command behaves identically. This
file stays as the familiar entry point for anyone running from a checkout.
"""

from backend.cli import main

if __name__ == "__main__":
    main()
