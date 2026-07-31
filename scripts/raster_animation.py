"""Render the Powers-of-Ten spike-raster zoom animation.

Reads a manually-curated units file (``good_units.pkl`` with per-unit ``depth``)
and writes an ``.mp4``/``.gif`` that zooms smoothly out from a 10 ms window to the
full ~36 h recording. Parameters come from CLI flags and/or ``.env`` (CLI wins).

Usage
-----
    python scripts/raster_animation.py
    python scripts/raster_animation.py --units-file path/to/good_units.pkl --out raster.mp4
    python scripts/raster_animation.py --duration-s 30 --fps 60
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pirouette_data.animations import make_animation


def _default_units() -> str | None:
    explicit = os.getenv("RASTER_UNITS_FILE")
    if explicit:
        return explicit
    units_dir = os.getenv("UNITS_DIR")
    if units_dir:
        cand = Path(units_dir) / "good_units.pkl"
        if cand.exists():
            return str(cand)
    return None


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="raster_animation.py",
                                description="Powers-of-Ten spike-raster zoom animation.")
    p.add_argument("--units-file", default=_default_units(),
                   help="Units pickle with per-unit 'spike_times' and 'depth' "
                        "(env RASTER_UNITS_FILE, or UNITS_DIR/good_units.pkl).")
    p.add_argument("--out", default=os.getenv("RASTER_OUT", "raster_animation.mp4"),
                   help="Output .mp4 or .gif (env RASTER_OUT).")
    p.add_argument("--center-s", type=float,
                   default=(float(os.getenv("RASTER_CENTER_S"))
                            if os.getenv("RASTER_CENTER_S") else None),
                   help="Time the zoom centres on (default: middle of the recording).")
    p.add_argument("--window-start-s", type=float,
                   default=float(os.getenv("RASTER_WINDOW_START_S", "0.01")),
                   help="Initial window width in seconds (default 0.01 = 10 ms).")
    p.add_argument("--window-end-s", type=float,
                   default=(float(os.getenv("RASTER_WINDOW_END_S"))
                            if os.getenv("RASTER_WINDOW_END_S") else None),
                   help="Final window width (default: full recording span).")
    p.add_argument("--duration-s", type=float,
                   default=float(os.getenv("RASTER_DURATION_S", "20")),
                   help="Zoom-out duration in seconds (default 20).")
    p.add_argument("--hold-s", type=float,
                   default=float(os.getenv("RASTER_HOLD_S", "3")),
                   help="Closing-card hold in seconds (default 3).")
    p.add_argument("--fps", type=int, default=int(os.getenv("RASTER_FPS", "30")))
    p.add_argument("--cap", type=int, default=int(os.getenv("RASTER_CAP", "60000")),
                   help="Max spikes drawn per frame (subsampled; default 60000).")
    p.add_argument("--dpi", type=int, default=int(os.getenv("RASTER_DPI", "120")))
    p.add_argument("--invert-depth", action="store_true",
                   default=os.getenv("RASTER_INVERT_DEPTH", "").lower()
                   in ("1", "true", "yes"),
                   help="Flip depth ordering if 'highest at top' is upside-down.")
    args = p.parse_args(argv)

    if not args.units_file:
        raise SystemExit(
            "No units file. Pass --units-file, or set RASTER_UNITS_FILE / UNITS_DIR "
            "in .env (expects a good_units.pkl with per-unit 'depth')."
        )

    print(f"Rendering raster animation from {args.units_file} -> {args.out} ...")
    out = make_animation(
        args.units_file, args.out,
        center_s=args.center_s,
        window_start_s=args.window_start_s,
        window_end_s=args.window_end_s,
        duration_s=args.duration_s,
        hold_s=args.hold_s,
        fps=args.fps,
        cap=args.cap,
        dpi=args.dpi,
        invert_depth=args.invert_depth,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
