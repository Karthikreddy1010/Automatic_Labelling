"""
backend/cli.py - The `poleannotator` command
=============================================

The launcher lives here rather than in run_backend.py at the repo root so it
ships inside the installed package: `pip install poleannotator` then
`poleannotator` works from any directory, with no checkout to cd into.

run_backend.py still works and simply calls main() here, so there is one
implementation rather than two that drift apart.
"""

import argparse
import socket
import sys
import threading
import time
import webbrowser


def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    print(f"\n-> Opening browser at {url} ...")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poleannotator",
        description="PoleAnnotator AI - local OBB annotation workspace.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser automatically")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Where datasets are stored (default: OBB_DATA_DIR, "
                             "else ./data/datasets)")
    parser.add_argument("--version", action="store_true", help="Print the version and exit")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.version:
        print(f"poleannotator {_version()}")
        return

    # Set this BEFORE importing the app: backend.storage reads OBB_DATA_DIR at
    # import time, so a later assignment would be ignored and datasets would
    # quietly land in the default directory instead.
    if args.data_dir:
        import os
        os.environ["OBB_DATA_DIR"] = args.data_dir

    import uvicorn
    from models.adapters.base import detect_hardware

    resolved_dev, hw = detect_hardware("AUTO")
    print("=" * 68)
    print("  POLEANNOTATOR AI - LOCAL ROBOFLOW-STYLE OBB AUTO-LABELING")
    print("=" * 68)
    print(f"  Version         : {_version()}")
    print(f"  Hardware Device : {resolved_dev.upper()}")
    print(f"  CUDA Available  : {hw.cuda_available}")
    if hw.gpu_name:
        print(f"  GPU Name        : {hw.gpu_name}")
    print(f"  Server URL      : http://{args.host}:{args.port}")
    print(f"  Docs API        : http://{args.host}:{args.port}/docs")
    print("=" * 68 + "\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((args.host, args.port)) == 0:
            print(f"\n[ERROR] Port {args.port} is already in use by an existing server process.")
            print(f"        To run on a different port: poleannotator --port {args.port + 1}")
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
        log_level="info",
    )


def _version() -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("poleannotator")
        except PackageNotFoundError:
            return "dev (running from a checkout)"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
