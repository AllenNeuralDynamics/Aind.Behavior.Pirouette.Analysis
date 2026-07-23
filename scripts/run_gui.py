"""Launch the Pirouette visualization GUI (Dash web app).

Parameters come from CLI flags and/or ``.env`` (CLI overrides ``.env``). The app
is served on the host that holds the data; viewers open the URL in a browser
(use ``--host 0.0.0.0`` for the LAN, or put a tunnel like ngrok in front for
remote access).

Usage
-----
    python scripts/run_gui.py
    python scripts/run_gui.py --host 0.0.0.0 --port 8050
    python scripts/run_gui.py --spike-offset-s 113097.0
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from pirouette_data.visualization_gui import run


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="run_gui.py", description="Serve the Pirouette explorer.")
    p.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR") or os.getenv("SAVE_DIR"),
                   help="Directory of dataset files (.parquet/.pkl/.csv).")
    p.add_argument("--units-dir", default=os.getenv("UNITS_DIR"),
                   help="Directory of spike-times .pkl files.")
    p.add_argument("--video-dir", default=os.getenv("VIDEO_DIR"),
                   help="Directory of <source_file>.mp4 tracked videos.")
    p.add_argument("--spike-offset-s", type=float,
                   default=float(os.getenv("SPIKE_OFFSET_S", "0") or 0),
                   help="Offset (s) added to spike times -> experiment reference.")
    p.add_argument("--host", default=os.getenv("GUI_HOST", "0.0.0.0"),
                   help="Bind address; 0.0.0.0 (default) allows LAN access.")
    p.add_argument("--port", type=int, default=int(os.getenv("GUI_PORT", "8050")))
    p.add_argument("--share", action="store_true",
                   default=os.getenv("SHARE", "").lower() in ("1", "true", "yes"),
                   help="Open a public ngrok tunnel (needs pyngrok + NGROK_AUTHTOKEN).")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    missing = [n for n in ("dataset_dir", "units_dir", "video_dir") if not getattr(args, n)]
    if missing:
        raise SystemExit(f"Missing required paths (set via CLI or .env): {', '.join(missing)}")

    run(
        dataset_dir=args.dataset_dir,
        units_dir=args.units_dir,
        video_dir=args.video_dir,
        spike_offset_s=args.spike_offset_s,
        host=args.host,
        port=args.port,
        debug=args.debug,
        share=args.share,
    )


if __name__ == "__main__":
    main()
