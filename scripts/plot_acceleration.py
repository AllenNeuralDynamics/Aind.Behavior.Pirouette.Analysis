"""Plot head-mounted BNO055 acceleration over time.

Four stacked subplots sharing one x axis (time in hours since the first BNO
sample): the 3D acceleration collapsed to 1D (magnitude) on top, then the X, Y and
Z components. Acceleration is in m/s^2 (the BNO055 default unit).

The data are hourly ``OnixBNO055Data_<timestamp>.csv`` files whose ``Seconds``
column is Harp time. Files are streamed one at a time (from disk or anonymously
from S3) and reduced to per-bin statistics, so memory use stays small even for
multi-day recordings.

Usage
-----
    python scripts/plot_acceleration.py DATASET_PATH
    python scripts/plot_acceleration.py DATASET_PATH --timespan 2 14 --bin-size 30
    python scripts/plot_acceleration.py DATASET_PATH --save-dir figures --no-show

``DATASET_PATH`` is the dataset root (local path or ``s3://bucket/prefix`` URI);
the ``OnixBNO055Data_*.csv`` files are read from ``<root>/behavior/OnixBNO055Data/``.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from pirouette_data.ingestion import get_s3_client, harp_to_datetime
from pirouette_data.kinematics import (
    ACCEL_CHANNELS,
    ACCEL_UNIT,
    SECONDS_PER_HOUR,
    bin_bno_chunks,
    finalize_binned_acceleration,
    first_bno_seconds,
    list_bno_chunks,
    rebin_acceleration,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Location of the BNO055 files relative to the dataset root.
BNO_SUBDIR = "behavior/OnixBNO055Data"

# Light/dark cycle, Pacific time (same values as raster_animation.py).
DAWN_S = 6 * 3600  # lights on, 6 AM
DUSK_S = 20 * 3600  # lights off, 8 PM
TRANS_S = 1800  # smooth 30-min dawn/dusk transitions
DAY_RGB = np.array([1.00, 0.88, 0.25])  # sunny yellow
NIGHT_RGB = np.array([0.25, 0.25, 0.25])  # dark gray

PANEL_LABELS = {
    "magnitude": "Absolute",
    "x": "X",
    "y": "Y",
    "z": "Z",
}


def time_of_day_rgb(tod_s: np.ndarray) -> np.ndarray:
    """RGB colour per time-of-day (seconds after midnight, Pacific)."""
    day = np.zeros_like(tod_s)
    m = (tod_s >= DAWN_S) & (tod_s < DAWN_S + TRANS_S)
    day[m] = (tod_s[m] - DAWN_S) / TRANS_S
    day[(tod_s >= DAWN_S + TRANS_S) & (tod_s < DUSK_S)] = 1.0
    m = (tod_s >= DUSK_S) & (tod_s < DUSK_S + TRANS_S)
    day[m] = 1.0 - (tod_s[m] - DUSK_S) / TRANS_S
    return day[:, None] * DAY_RGB + (1 - day[:, None]) * NIGHT_RGB


def pacific_time_of_day_s(harp_seconds: float) -> float:
    """Seconds after local (Pacific) midnight for a Harp timestamp."""
    ts = harp_to_datetime(pd.Series([harp_seconds])).iloc[0]
    return ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        help=(
            "Root of the dataset (local directory or s3:// URI); the BNO files are "
            f"read from <root>/{BNO_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--timespan",
        nargs=2,
        type=float,
        metavar=("START_H", "END_H"),
        default=None,
        help="Span to plot, in hours since the first BNO sample (default: all data).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.1,
        help=(
            "Temporal resolution: width of each bin in HOURS (default: 0.1 = 6 min). "
            "The number of bins follows from the timespan. Minimum 1 s (~0.000278 h)."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=("abs_mean", "mean"),
        default="abs_mean",
        help=(
            "Per-bin summary: mean of |a| (default; movement intensity) or the "
            "signed mean (axes average toward 0 over long bins)."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel file downloads (default: 8)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache" / "acceleration",
        help="Cache of per-file reduced data, reused by later runs (default: %(default)s).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write the cache.")
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (default: 16, i.e. 16:9 with the default height).",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Save the figure here.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()

    if args.bin_size * SECONDS_PER_HOUR < 1:
        parser.error("--bin-size must be at least 1 second (0.000278 hours)")
    if args.timespan is not None and args.timespan[1] <= args.timespan[0]:
        parser.error("--timespan END_H must be greater than START_H")
    return args


def main() -> None:
    args = parse_args()

    root = args.dataset_path.rstrip("/\\")
    bno_dir = f"{root}/{BNO_SUBDIR}"
    client = get_s3_client() if root.startswith("s3://") else None
    chunks = list_bno_chunks(bno_dir, s3_client=client)

    # Time zero is the first sample of the recording. Chunk files start on whole
    # hours of Harp time, so each file's Harp span follows from its filename.
    origin = first_bno_seconds(chunks[0][0], s3_client=client)
    first_chunk_start = math.floor(origin / SECONDS_PER_HOUR) * SECONDS_PER_HOUR

    if args.timespan is None:
        start_s, end_s = None, None
    else:
        start_s = origin + args.timespan[0] * SECONDS_PER_HOUR
        end_s = origin + args.timespan[1] * SECONDS_PER_HOUR

    selected = []
    for source, stamp in chunks:
        chunk_start = first_chunk_start + (stamp - chunks[0][1]).total_seconds()
        chunk_end = chunk_start + SECONDS_PER_HOUR
        if (end_s is None or chunk_start < end_s) and (start_s is None or chunk_end > start_s):
            selected.append(source)
    if not selected:
        raise SystemExit(
            f"No files overlap the requested timespan "
            f"({len(chunks)} files cover ~{len(chunks)} h from the recording start)."
        )
    print(f"Using {len(selected)} of {len(chunks)} hourly files.")

    # Cache key = dataset root, so different datasets never share entries.
    cache_dir = None if args.no_cache else args.cache_dir / Path(root).name
    base = bin_bno_chunks(
        selected,
        s3_client=client,
        cache_dir=cache_dir,
        workers=args.workers,
        progress=partial(tqdm, total=len(selected), desc="files"),
    )

    bin_size_s = args.bin_size * SECONDS_PER_HOUR
    series = finalize_binned_acceleration(
        rebin_acceleration(base, origin, bin_size_s, start_s, end_s), bin_size_s, stat=args.stat
    )

    time_h = series["time_h"].to_numpy()
    valid = time_h[np.isfinite(series["magnitude"].to_numpy())]
    t_lo, t_hi = float(valid.min()), float(valid.max() + args.bin_size)
    span_h = t_hi - t_lo

    width = args.fig_width or 16.0
    fig, axes = plt.subplots(
        len(ACCEL_CHANNELS) + 1,
        1,
        sharex=True,
        figsize=(width, 9),
        gridspec_kw={"height_ratios": [0.2, 1, 1, 1, 1]},
        constrained_layout=True,
    )
    ax_tod, panels = axes[0], axes[1:]

    # Light/dark strip (Pacific time of day), above the data panels.
    tod0 = pacific_time_of_day_s(origin)
    t_fine = np.linspace(t_lo, t_hi, max(2, int(span_h * 60)))  # ~1-min resolution
    tod_rgb = time_of_day_rgb((tod0 + t_fine * SECONDS_PER_HOUR) % 86400.0)
    ax_tod.imshow(
        tod_rgb[np.newaxis, :, :],
        aspect="auto",
        extent=[t_lo, t_hi, 0, 1],
        origin="lower",
        interpolation="bilinear",
    )
    ax_tod.set_yticks([])
    ax_tod.tick_params(bottom=False, labelbottom=False)
    for spine in ax_tod.spines.values():
        spine.set_visible(False)
    ax_tod.set_title(f"BNO055 acceleration ({args.bin_size:g} h bins)")

    for ax, channel in zip(panels, ACCEL_CHANNELS):
        ax.plot(time_h, series[channel].to_numpy(), lw=0.5, alpha=0.75, color="k")
        ax.set_ylabel(f"{PANEL_LABELS[channel]}\n({ACCEL_UNIT})")
        ax.spines[["top", "right"]].set_visible(False)
    panels[-1].set_xlabel("Time (hours)")
    if span_h > 48:
        # Ticks at whole days (one per light/dark cycle), thinned to avoid overlap.
        days_per_tick = max(1, math.ceil(span_h / 24 / 14))
        panels[-1].xaxis.set_major_locator(MultipleLocator(24 * days_per_tick))
    ax_tod.set_xlim(t_lo, t_hi)  # axes are shared, so this sets all panels

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        name = Path(root).name or "acceleration"
        out = args.save_dir / f"acceleration_{name}.png"
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
