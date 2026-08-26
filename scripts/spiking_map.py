"""Render a Neuropixels 2.0 probe schematic with optional waveform overlay.

Draws one shank (bottom N sites) and overlays the best-amplitude mean waveform
of each unit at its probe depth.  Waveform colours match the raster animation
(same glasbey palette and depth-ordered colour indices).  Parameters come from
``.env`` / environment variables; CLI flags override both.

Usage
-----
    python scripts/spiking_map.py
    python scripts/spiking_map.py --show
    python scripts/spiking_map.py --units-file path/to/good_units.pkl
    python scripts/spiking_map.py --wf-amp 8 --jitter-step 3 --n-jitter 7
    python scripts/spiking_map.py --no-waveforms --out blank_schematic.pdf
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from pirouette_data.vis import ProbeGeometry, make_probe_figure, overlay_waveforms


# ---------------------------------------------------------------------------
# Env parsing helpers (mirrors cli.py)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return default if not val or not val.strip() else float(val)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return default if not val or not val.strip() else int(val)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if not val or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _default_units() -> str | None:
    """Resolve unit file path: SPIKING_MAP_UNITS_FILE > EPHYS_DATASET > UNITS_DIR/good_units.pkl."""
    for var in ("SPIKING_MAP_UNITS_FILE", "EPHYS_DATASET"):
        val = os.getenv(var)
        if val:
            return val
    units_dir = os.getenv("UNITS_DIR")
    if units_dir:
        cand = Path(units_dir) / "good_units.pkl"
        if cand.exists():
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, resolving defaults from ``os.environ``."""
    p = argparse.ArgumentParser(
        prog="spiking_map.py",
        description=(
            "Render a Neuropixels 2.0 probe schematic (1 shank, bottom N sites) "
            "with per-unit mean waveforms overlaid at each unit's probe depth. "
            "Parameters come from .env / env vars; CLI flags override both."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Output ------------------------------------------------------------
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--save-dir",
        default=os.getenv("SPIKING_MAP_SAVE_DIR"),
        help=(
            "Directory to write the figure (env SPIKING_MAP_SAVE_DIR). "
            "Ignored when --out is an absolute path."
        ),
    )
    out_grp.add_argument(
        "--out",
        default=os.getenv("SPIKING_MAP_OUT", "spiking_map.pdf"),
        help="Output filename or path (.pdf / .png / .svg; env SPIKING_MAP_OUT).",
    )
    out_grp.add_argument(
        "--dpi",
        type=int,
        default=_env_int("SPIKING_MAP_DPI", 150),
        help="Figure DPI for raster output (env SPIKING_MAP_DPI).",
    )
    out_grp.add_argument(
        "--show",
        action="store_true",
        help="Open the figure in an interactive window after saving.",
    )

    # ---- Waveform overlay --------------------------------------------------
    wf_grp = p.add_argument_group("Waveform overlay")
    wf_grp.add_argument(
        "--units-file",
        default=_default_units(),
        metavar="PATH",
        help=(
            "Units pickle (.pkl) with per-unit 'spike_times', 'depth', "
            "'mean_waveform', 'waveform_channels', and 'waveform_positions'. "
            "Env: SPIKING_MAP_UNITS_FILE, EPHYS_DATASET, or UNITS_DIR/good_units.pkl."
        ),
    )
    wf_grp.add_argument(
        "--no-waveforms",
        action="store_true",
        help="Skip waveform overlay (blank schematic only).",
    )
    wf_grp.add_argument(
        "--palette",
        default=os.getenv("SPIKING_MAP_PALETTE", os.getenv("RASTER_CMAP", "glasbey")),
        help=(
            "Colour palette for units — must match the raster-animation palette "
            "for identical colours (env SPIKING_MAP_PALETTE or RASTER_CMAP)."
        ),
    )
    wf_grp.add_argument(
        "--invert-depth",
        action=argparse.BooleanOptionalAction,
        default=_env_bool(
            "SPIKING_MAP_INVERT_DEPTH",
            _env_bool("RASTER_INVERT_DEPTH", True),
        ),
        help=(
            "Sort deepest unit first when assigning colour indices, matching "
            "RASTER_INVERT_DEPTH (env SPIKING_MAP_INVERT_DEPTH)."
        ),
    )
    wf_grp.add_argument(
        "--wf-x-span",
        type=float,
        dest="wf_x_span_um",
        default=_env_float("SPIKING_MAP_WF_X_SPAN_UM", 24.0),
        help="Total horizontal waveform width in µm (env SPIKING_MAP_WF_X_SPAN_UM).",
    )
    wf_grp.add_argument(
        "--wf-amp",
        type=float,
        dest="wf_amp_um",
        default=_env_float("SPIKING_MAP_WF_AMP_UM", 10.0),
        help=(
            "Peak-to-trough amplitude in µm.  With --no-relative-amp (default) "
            "every waveform is this height; with --relative-amp this is the height "
            "of the largest-amplitude unit (env SPIKING_MAP_WF_AMP_UM)."
        ),
    )
    wf_grp.add_argument(
        "--relative-amp",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SPIKING_MAP_WF_RELATIVE_AMP", False),
        help=(
            "Scale waveform heights relative to the dataset-wide maximum "
            "peak-to-trough amplitude so true amplitude ratios are visible.  "
            "Default off (uniform height).  (env SPIKING_MAP_WF_RELATIVE_AMP)"
        ),
    )
    wf_grp.add_argument(
        "--jitter-step",
        type=float,
        dest="jitter_step_um",
        default=_env_float("SPIKING_MAP_JITTER_STEP_UM", 4.0),
        help="Horizontal step between jitter positions in µm (env SPIKING_MAP_JITTER_STEP_UM).",
    )
    wf_grp.add_argument(
        "--n-jitter",
        type=int,
        default=_env_int("SPIKING_MAP_N_JITTER", 5),
        help="Number of distinct x jitter positions (env SPIKING_MAP_N_JITTER).",
    )
    wf_grp.add_argument(
        "--upsample",
        type=int,
        default=_env_int("SPIKING_MAP_WF_UPSAMPLE", 4),
        help=(
            "Upsample factor for waveform display (default 4 × 150 samples → 600 "
            "display points for smoother curves; env SPIKING_MAP_WF_UPSAMPLE)."
        ),
    )
    wf_grp.add_argument(
        "--jitter-y",
        type=float,
        dest="jitter_y_um",
        default=_env_float("SPIKING_MAP_JITTER_Y_UM", 2.5),
        help=(
            "Step between y jitter positions in µm — small offset to separate "
            "overlapping baselines (env SPIKING_MAP_JITTER_Y_UM)."
        ),
    )
    wf_grp.add_argument(
        "--n-jitter-y",
        type=int,
        default=_env_int("SPIKING_MAP_N_JITTER_Y", 3),
        help="Number of distinct y jitter positions (env SPIKING_MAP_N_JITTER_Y).",
    )
    wf_grp.add_argument(
        "--wf-lw",
        type=float,
        dest="wf_linewidth",
        default=_env_float("SPIKING_MAP_WF_LW", 0.5),
        help="Waveform line width in pts (env SPIKING_MAP_WF_LW).",
    )
    wf_grp.add_argument(
        "--wf-alpha",
        type=float,
        dest="wf_alpha",
        default=_env_float("SPIKING_MAP_WF_ALPHA", 0.85),
        help="Waveform line opacity 0–1 (env SPIKING_MAP_WF_ALPHA).",
    )
    wf_grp.add_argument(
        "--seed",
        type=int,
        default=_env_int("SPIKING_MAP_SEED", 0),
        help="RNG seed for jitter layout — change for a different random arrangement (env SPIKING_MAP_SEED).",
    )

    # ---- Per-site colour coding (optional; separate from waveform units) ---
    site_grp = p.add_argument_group("Per-site colour coding (optional)")
    site_grp.add_argument(
        "--values-npy",
        default=os.getenv("SPIKING_MAP_VALUES_NPY"),
        metavar="PATH",
        help=(
            "Path to a .npy file with a 1-D array of length n_sites. "
            "Sites are colour-coded by these values (env SPIKING_MAP_VALUES_NPY)."
        ),
    )
    site_grp.add_argument(
        "--cmap",
        default=os.getenv("SPIKING_MAP_CMAP", "viridis"),
        help="Matplotlib colormap for per-site values (env SPIKING_MAP_CMAP).",
    )
    site_grp.add_argument(
        "--vmin", type=float, default=None,
        help="Colormap lower limit (defaults to data minimum).",
    )
    site_grp.add_argument(
        "--vmax", type=float, default=None,
        help="Colormap upper limit (defaults to data maximum).",
    )
    site_grp.add_argument(
        "--colorbar-label",
        default=os.getenv("SPIKING_MAP_COLORBAR_LABEL", "Value"),
        help="Colourbar axis label (env SPIKING_MAP_COLORBAR_LABEL).",
    )

    # ---- Probe geometry ----------------------------------------------------
    geo_grp = p.add_argument_group("Probe geometry (µm)")
    geo_grp.add_argument(
        "--shank-width", type=float,
        default=_env_float("PROBE_SHANK_WIDTH_UM", 70.0),
        help="Shank width in µm (env PROBE_SHANK_WIDTH_UM).",
    )
    geo_grp.add_argument(
        "--tip-length", type=float,
        default=_env_float("PROBE_TIP_LENGTH_UM", 175.0),
        help="Tapered-tip length in µm (env PROBE_TIP_LENGTH_UM).",
    )
    geo_grp.add_argument(
        "--site-width", type=float,
        default=_env_float("PROBE_SITE_WIDTH_UM", 12.0),
        help="Site width in µm (env PROBE_SITE_WIDTH_UM).",
    )
    geo_grp.add_argument(
        "--site-height", type=float,
        default=_env_float("PROBE_SITE_HEIGHT_UM", 12.0),
        help="Site height in µm (env PROBE_SITE_HEIGHT_UM).",
    )
    geo_grp.add_argument(
        "--vertical-pitch", type=float,
        default=_env_float("PROBE_VERTICAL_PITCH_UM", 15.0),
        help="Vertical site pitch front-to-front in µm (env PROBE_VERTICAL_PITCH_UM).",
    )
    geo_grp.add_argument(
        "--horizontal-pitch", type=float,
        default=_env_float("PROBE_HORIZONTAL_PITCH_UM", 32.0),
        help="Horizontal site pitch right-to-right in µm (env PROBE_HORIZONTAL_PITCH_UM).",
    )
    geo_grp.add_argument(
        "--n-sites", type=int,
        default=_env_int("PROBE_N_SITES", 96),
        help="Total number of sites to display (env PROBE_N_SITES).",
    )
    geo_grp.add_argument(
        "--first-site-offset", type=float,
        default=_env_float("PROBE_FIRST_SITE_OFFSET_UM", 20.0),
        help=(
            "Distance (µm) from the tip/body junction to the first site centre "
            "(env PROBE_FIRST_SITE_OFFSET_UM)."
        ),
    )
    geo_grp.add_argument(
        "--no-stagger",
        action="store_true",
        default=not _env_bool("PROBE_STAGGER", False),
        help="Disable the checkerboard column stagger (env PROBE_STAGGER=true to enable).",
    )

    # ---- Figure ------------------------------------------------------------
    fig_grp = p.add_argument_group("Figure")
    fig_grp.add_argument(
        "--fig-width", type=float,
        default=_env_float("SPIKING_MAP_FIG_WIDTH", 2.5),
        help="Figure width in inches (env SPIKING_MAP_FIG_WIDTH).",
    )
    fig_grp.add_argument(
        "--title",
        default=os.getenv(
            "SPIKING_MAP_TITLE",
            "Neuropixels 2.0 — 1 shank, bottom 96 sites",
        ),
        help="Figure title (env SPIKING_MAP_TITLE; pass '' to suppress).",
    )
    fig_grp.add_argument(
        "--no-scale-bar", action="store_true",
        default=_env_bool("SPIKING_MAP_NO_SCALE_BAR", False),
        help="Omit the scale bar (env SPIKING_MAP_NO_SCALE_BAR).",
    )
    fig_grp.add_argument(
        "--scale-bar-um", type=float,
        default=_env_float("SPIKING_MAP_SCALE_BAR_UM", 100.0),
        help="Scale bar length in µm (env SPIKING_MAP_SCALE_BAR_UM).",
    )
    fig_grp.add_argument(
        "--no-colorbar", action="store_true",
        help="Suppress the colourbar even when --values-npy is given.",
    )
    fig_grp.add_argument(
        "--probe-label",
        default=os.getenv("SPIKING_MAP_PROBE_LABEL", "NP 2.0"),
        help="Short italic label printed above the shank (env SPIKING_MAP_PROBE_LABEL).",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901
    load_dotenv()
    args = build_parser().parse_args(argv)

    # --- Build probe geometry -----------------------------------------------
    geometry = ProbeGeometry(
        shank_width=args.shank_width,
        tip_length=args.tip_length,
        site_width=args.site_width,
        site_height=args.site_height,
        vertical_pitch=args.vertical_pitch,
        horizontal_pitch=args.horizontal_pitch,
        n_sites=args.n_sites,
        first_site_offset=args.first_site_offset,
        stagger=not args.no_stagger,
    )

    # --- Optional per-site values -------------------------------------------
    site_values: np.ndarray | None = None
    if args.values_npy:
        npy_path = Path(args.values_npy)
        if not npy_path.exists():
            raise SystemExit(f"Values file not found: {npy_path}")
        site_values = np.load(npy_path).ravel()
        if len(site_values) != geometry.n_sites:
            raise SystemExit(
                f"Values array has {len(site_values)} elements but "
                f"--n-sites is {geometry.n_sites}."
            )
        print(f"Loaded per-site values from {npy_path} "
              f"(min={site_values.min():.3g}, max={site_values.max():.3g})")

    # --- Load units ---------------------------------------------------------
    units: dict | None = None
    if not args.no_waveforms:
        if not args.units_file:
            print(
                "Warning: no units file found (set SPIKING_MAP_UNITS_FILE, "
                "EPHYS_DATASET, or UNITS_DIR in .env, or pass --units-file). "
                "Rendering blank schematic."
            )
        else:
            pkl_path = Path(args.units_file)
            if not pkl_path.exists():
                raise SystemExit(f"Units file not found: {pkl_path}")
            print(f"Loading units from {pkl_path} ...")
            with open(pkl_path, "rb") as f:
                units = pickle.load(f)
            print(f"  {len(units)} units loaded.")

    # --- Resolve output path ------------------------------------------------
    out = Path(args.out)
    if not out.is_absolute() and args.save_dir:
        out = Path(args.save_dir) / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Render probe schematic ---------------------------------------------
    print(f"Rendering probe schematic → {out} ...")
    fig, ax = make_probe_figure(
        geometry,
        site_values=site_values,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        colorbar=(not args.no_colorbar),
        colorbar_label=args.colorbar_label,
        title=args.title,
        fig_width=args.fig_width,
        dpi=args.dpi,
        probe_label=f"{len(units)} units" if units is not None else args.probe_label,
        scale_bar=(not args.no_scale_bar),
        scale_bar_um=args.scale_bar_um,
    )

    # --- Overlay waveforms --------------------------------------------------
    if units is not None:
        print(f"Overlaying waveforms (palette={args.palette}, "
              f"wf_amp={args.wf_amp_um} µm, jitter={args.n_jitter}×{args.jitter_step_um} µm) ...")
        overlay_waveforms(
            ax, units, geometry,
            palette=args.palette,
            invert_depth=args.invert_depth,
            wf_x_span_um=args.wf_x_span_um,
            wf_amp_um=args.wf_amp_um,
            relative_amp=args.relative_amp,
            jitter_step_um=args.jitter_step_um,
            n_jitter=args.n_jitter,
            jitter_y_um=args.jitter_y_um,
            n_jitter_y=args.n_jitter_y,
            upsample=args.upsample,
            linewidth=args.wf_linewidth,
            alpha=args.wf_alpha,
            seed=args.seed,
        )

    fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved {out}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
