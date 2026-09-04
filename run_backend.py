"""
run_backend.py - One-Command Launcher for PoleAnnotator AI Platform
===================================================================

Starts the FastAPI server with CORS, model adapters, and static frontend.
Open http://127.0.0.1:8000 in your browser.

Usage:
    python run_backend.py [--port 8000] [--host 127.0.0.1] [--no-browser]
"""

import sys
import argparse
import webbrowser
import threading
import time
import uvicorn

from models.adapters.base import detect_hardware


def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    print(f"\n-> Opening browser at {url} ...")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="PoleAnnotator AI Platform Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    args = parser.parse_args()

    resolved_dev, hw = detect_hardware("AUTO")
    print("=" * 68)
    print("  POLEANNOTATOR AI - LOCAL ROBOFLOW-STYLE OBB AUTO-LABELING")
    print("=" * 68)
    print(f"  Hardware Device : {resolved_dev.upper()}")
    print(f"  CUDA Available  : {hw.cuda_available}")
    if hw.gpu_name:
        print(f"  GPU Name        : {hw.gpu_name}")
    print(f"  Server URL      : http://{args.host}:{args.port}")
    print(f"  Docs API        : http://{args.host}:{args.port}/docs")
    print("=" * 68 + "\n")

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((args.host, args.port)) == 0:
            print(f"\n[ERROR] Port {args.port} is already in use by an existing server process.")
            print(f"        To run on a different port: python run_backend.py --port {args.port + 1}")
            print(f"        Or close the existing terminal/process running on port {args.port}.\n")
            sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
