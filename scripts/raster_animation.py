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
    # Priority: EPHYS_DATASET (new) > RASTER_UNITS_FILE > UNITS_DIR/good_units.pkl
    for var in ("EPHYS_DATASET", "RASTER_UNITS_FILE"):
        val = os.getenv(var)
        if val:
            return val
    units_dir = os.getenv("UNITS_DIR")
    if units_dir:
        cand = Path(units_dir) / "good_units.pkl"
        if cand.exists():
            return str(cand)
    return None


def _default_center() -> float | None:
    # RASTER_START_TIME_S (new) or legacy RASTER_CENTER_S
    for var in ("RASTER_START_TIME_S", "RASTER_CENTER_S"):
        val = os.getenv(var)
        if val:
            return float(val)
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
                   default=_default_center(),
                   help="Time the zoom centres on (default: middle of the recording). "
                        "Env: RASTER_START_TIME_S or RASTER_CENTER_S.")
    p.add_argument("--window-start-s", type=float,
                   default=float(os.getenv("RASTER_WINDOW_START_S", "0.001")),
                   help="Initial window width in seconds (default 0.001 = 1 ms).")
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
    p.add_argument("--invert-depth", action="store_true",
                   default=os.getenv("RASTER_INVERT_DEPTH", "").lower()
                   in ("1", "true", "yes"),
                   help="Put the deepest units at the TOP (default: bottom).")
    p.add_argument("--axis-unit-ms-to-s", type=float,
                   default=float(os.getenv("RASTER_AXIS_MS_TO_S", "1.0")),
                   help="Window width (s) at which axis labels switch ms→s "
                        "(default 1.0, env RASTER_AXIS_MS_TO_S).")
    p.add_argument("--axis-unit-s-to-min", type=float,
                   default=float(os.getenv("RASTER_AXIS_S_TO_MIN", "100.0")),
                   help="Window width (s) at which axis labels switch s→min "
                        "(default 100, env RASTER_AXIS_S_TO_MIN).")
    p.add_argument("--axis-unit-min-to-hr", type=float,
                   default=float(os.getenv("RASTER_AXIS_MIN_TO_HR", "6000.0")),
                   help="Window width (s) at which axis labels switch min→hours "
                        "(default 6000, env RASTER_AXIS_MIN_TO_HR).")
    p.add_argument("--easing",
                   default=os.getenv("RASTER_EASING", "ease-in-out"),
                   help="Zoom-speed curve: 'ease-in-out' (default), 'ease-in', "
                        "'ease-out', or 'linear' (env RASTER_EASING).")
    p.add_argument("--milestone-dwell-s", type=float,
                   default=float(os.getenv("RASTER_MILESTONE_DWELL_S", "0.5")),
                   help="Seconds to hold at each scale-bar milestone "
                        "(1 ms/100 ms/1 s/…/100 hrs). 0 = no dwell "
                        "(env RASTER_MILESTONE_DWELL_S).")
    p.add_argument("--zoom-mode",
                   default=os.getenv("RASTER_ZOOM_MODE", "anchor-left"),
                   choices=["anchor-left", "center", "center-right"],
                   help="Zoom style: 'anchor-left' pins the left edge; 'center' zooms "
                        "from a fixed centre; 'center-right' centres until the left "
                        "edge reaches t=0 then expands rightward only, allowing the "
                        "window to show blank space beyond the data end "
                        "(env RASTER_ZOOM_MODE).")
    p.add_argument("--depth-lo", type=float,
                   default=(float(os.getenv("RASTER_DEPTH_LO_UM"))
                            if os.getenv("RASTER_DEPTH_LO_UM") else None),
                   help="Fixed lower y-limit for the raster (µm). "
                        "Default: computed from data (env RASTER_DEPTH_LO_UM).")
    p.add_argument("--depth-hi", type=float,
                   default=(float(os.getenv("RASTER_DEPTH_HI_UM"))
                            if os.getenv("RASTER_DEPTH_HI_UM") else None),
                   help="Fixed upper y-limit for the raster (µm). "
                        "Default: computed from data (env RASTER_DEPTH_HI_UM).")
    p.add_argument("--anchor-s", type=float,
                   default=(float(os.getenv("RASTER_ANCHOR_S"))
                            if os.getenv("RASTER_ANCHOR_S") else None),
                   help="Left-edge start time (s) for anchor-left mode. "
                        "Default: 70 s into the recording (env RASTER_ANCHOR_S).")
    p.add_argument("--rate-bin-s", type=float,
                   default=float(os.getenv("RASTER_RATE_BIN_S", "1800")),
                   help="Bin width (s) for the population firing-rate panel. "
                        "1800 = 30 min (default); 3600 = 1 hr (env RASTER_RATE_BIN_S).")
    p.add_argument("--t0-pst-s", type=float,
                   default=float(os.getenv("RASTER_T0_PST_S", "0")),
                   help="PST time-of-day in seconds at t=0 of the recording "
                        "(e.g. 66866 for 18:34 PST). Used for the day/night strip "
                        "(env RASTER_T0_PST_S).")
    p.add_argument("--raster-mode",
                   default=os.getenv("RASTER_MODE", "hybrid"),
                   choices=["hybrid", "charlie"],
                   help="Raster display mode: 'hybrid' (default) — per-unit colour "
                        "ticks blending into a density heatmap at wide zoom; "
                        "'charlie' — dartsort-style amplitude-ordered scatter with "
                        "gaussian depth-smear at wide zoom (env RASTER_MODE).")
    p.add_argument("--charlie-max-spikes", type=int,
                   default=int(os.getenv("CHARLIE_MAX_SPIKES", "500000")),
                   help="Global spike count cap for charlie mode (default 500 000; "
                        "env CHARLIE_MAX_SPIKES).")
    p.add_argument("--charlie-jitter-density",
                   default=os.getenv("CHARLIE_JITTER_DENSITY", "per-unit"),
                   choices=["per-unit", "global", "none"],
                   help="Jitter density mode for charlie scatter: "
                        "'per-unit' (default) — each unit's spread scales by its own "
                        "local firing density so quiet units stay as sharp ticks; "
                        "'global' — spread scales by total population spike density; "
                        "'none' — uniform jitter at full sigma, no density weighting "
                        "(env CHARLIE_JITTER_DENSITY).")
    p.add_argument("--show-probe-map", action="store_true",
                   default=os.getenv("RASTER_SHOW_PROBE_MAP", "").lower()
                   in ("1", "true", "yes"),
                   help="Add a static NP2.0 probe spiking map to the left of the raster, "
                        "sharing the depth axis and using the same unit colours "
                        "(env RASTER_SHOW_PROBE_MAP).")
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
        easing=args.easing,
        milestone_dwell_s=args.milestone_dwell_s,
        axis_unit_ms_to_s=args.axis_unit_ms_to_s,
        axis_unit_s_to_min=args.axis_unit_s_to_min,
        axis_unit_min_to_hr=args.axis_unit_min_to_hr,
        zoom_mode=args.zoom_mode,
        anchor_s=args.anchor_s,
        rate_bin_s=args.rate_bin_s,
        t0_pst_s=args.t0_pst_s,
        raster_mode=args.raster_mode,
        charlie_max_spikes=args.charlie_max_spikes,
        charlie_jitter_density=args.charlie_jitter_density,
        depth_lo_um=args.depth_lo,
        depth_hi_um=args.depth_hi,
        show_probe_map=args.show_probe_map,
    )
    print(f"Saved {saved}")


if __name__ == "__main__":
    main()
