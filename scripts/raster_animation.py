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
    p.add_argument("--save-dir", default=os.getenv("RASTER_SAVE_DIR"),
                   help="Directory to write the animation into (env RASTER_SAVE_DIR). "
                        "Ignored if --out is an absolute path.")
    p.add_argument("--out", default=os.getenv("RASTER_OUT", "raster_animation.mp4"),
                   help="Output file name or path, .mp4/.gif (env RASTER_OUT). "
                        "A bare name is written under --save-dir.")
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
    p.add_argument("--palette", default=os.getenv("RASTER_CMAP", "muted"),
                   help="Colour palette: 'muted' (default), 'rainbow', or any "
                        "Matplotlib colormap name (env RASTER_CMAP).")
    p.add_argument("--band-spread", type=float,
                   default=float(os.getenv("RASTER_BAND_SPREAD", "0.4")),
                   help="Band jitter as a fraction of the median depth gap "
                        "(slight offset for overlapping units; env RASTER_BAND_SPREAD).")
    p.add_argument("--invert-depth", action="store_true",
                   default=os.getenv("RASTER_INVERT_DEPTH", "").lower()
                   in ("1", "true", "yes"),
                   help="Put the deepest units at the TOP (default: bottom).")
    args = p.parse_args(argv)

    if not args.units_file:
        raise SystemExit(
            "No units file. Pass --units-file, or set RASTER_UNITS_FILE / UNITS_DIR "
            "in .env (expects a good_units.pkl with per-unit 'depth')."
        )

    # Resolve output: a bare name goes under --save-dir (if set).
    out = Path(args.out)
    if not out.is_absolute() and args.save_dir:
        out = Path(args.save_dir) / out

    print(f"Rendering raster animation from {args.units_file} -> {out} ...")
    saved = make_animation(
        args.units_file, out,
        center_s=args.center_s,
        window_start_s=args.window_start_s,
        window_end_s=args.window_end_s,
        duration_s=args.duration_s,
        hold_s=args.hold_s,
        fps=args.fps,
        cap=args.cap,
        dpi=args.dpi,
        invert_depth=args.invert_depth,
        palette=args.palette,
        band_spread=args.band_spread,
    )
    print(f"Saved {saved}")


if __name__ == "__main__":
    main()
