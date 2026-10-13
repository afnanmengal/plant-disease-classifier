"""Standalone desktop launcher for Plant Disease Classifier Streamlit app."""

import multiprocessing
import os
import sys
from pathlib import Path


def main():
    multiprocessing.freeze_support()

    if getattr(sys, "frozen", False):
        base_dir = Path(sys._MEIPASS)
        exe_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent
        exe_dir = base_dir

    src_dir = base_dir / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Also check external src if running next to executable
    external_src = exe_dir / "src"
    if external_src.is_dir() and str(external_src) not in sys.path:
        sys.path.insert(0, str(external_src))

    app_path = base_dir / "app.py"
    if not app_path.is_file():
        app_path = exe_dir / "app.py"

    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.headless=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
