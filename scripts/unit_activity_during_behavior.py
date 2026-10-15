"""Plot one unit's activity alongside behavior over a long recording window.

Three stacked panels sharing a time axis (hours), mirroring the visualization
GUI's timeseries view:

1. behaviour classification (rest: gray, movement: salmon),
2. smoothed ear velocity (mm/s),
3. instantaneous firing rate of the chosen unit (Hz, black),
4. time-of-day strip (yellow = light, dark gray = dark; as in raster_animation.py).

Spike times are shifted by ``--spike-offset-s`` onto the dataset's
``time_since_start`` timeline (as in the GUI). The plotted window starts where
the pose data and spikes first overlap (or at ``--start-s``) and lasts
``--duration-hours``. Velocity and behaviour are averaged into
``--plot-bin-s`` bins for plotting; the firing rate is computed at the GUI's
resolution (50 ms bins, 0.2 s Gaussian) and then averaged into the same bins.

With ``--rate-only`` just the firing rate (+ time-of-day strip) is drawn over
the unit's full spike range (~100 h); ``--start-hours`` / ``--duration-hours``
restrict the window.

Usage
-----
    python scripts/unit_activity_during_behavior.py --unit-id 2
    python scripts/unit_activity_during_behavior.py --unit-id 2 \
        --save-dir C:/Users/brandon.pratt/Desktop/figures --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pirouette_data.ephys import instantaneous_firing_rate
from pirouette_data.visualization_gui import (
    COL_BEHAVIOR,
    COL_DATETIME,
    COL_TIME,
    COL_VELOCITY,
    MOVE_COLOR,
    REST_COLOR,
    load_dataset,
    load_units,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(
    r"C:\Users\brandon.pratt\Desktop\data\body-kinematics\datasets"
    r"\854393_2026-06-09_19-34-26_pirouette_dataset_36hr.parquet"
)
UNITS = Path(
    r"C:\Users\brandon.pratt\Desktop\data\body-kinematics\units"
    r"\good_units_100hrs_v2.pkl"
)
SPIKE_OFFSET_S = 111600.0
FR_BIN_S = 0.05
FR_SMOOTH_S = 0.2

# Light/dark cycle, Pacific time (same values as raster_animation.py).
DAWN_S = 6 * 3600  # lights on, 6 AM
DUSK_S = 20 * 3600  # lights off, 8 PM
TRANS_S = 1800  # smooth 30-min dawn/dusk transitions
DAY_RGB = np.array([1.00, 0.88, 0.25])  # sunny yellow
NIGHT_RGB = np.array([0.25, 0.25, 0.25])  # dark gray

# Figure style (matches the population commutator plot): Arial, thick offset spines.
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 18
SPINE_LW = 2.5
SPINE_OFFSET = 10  # points
FIG_WIDTH = 9  # inches (height is fixed at 10)
RATE_ONLY_WIDTH = 12  # inches, for --rate-only (height 5.5)


def bin_mean(t: np.ndarray, x: np.ndarray, t0: float, n_bins: int, bin_s: float):
    """Mean of *x* in consecutive ``bin_s`` bins starting at *t0* (NaN if empty)."""
    idx = ((t - t0) // bin_s).astype("int64")
    ok = (idx >= 0) & (idx < n_bins) & np.isfinite(x)
    counts = np.bincount(idx[ok], minlength=n_bins)
    sums = np.bincount(idx[ok], weights=x[ok], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def firing_rate_binned(spikes, t0, t1, bin_s, n_bins) -> np.ndarray:
    """GUI-resolution firing rate, block-averaged into ``bin_s`` bins."""
    _, rate = instantaneous_firing_rate(spikes, t0, t1, FR_BIN_S, FR_SMOOTH_S)
    factor = max(1, int(round(bin_s / FR_BIN_S)))
    n = min(len(rate) // factor, n_bins)
    out = np.full(n_bins, np.nan)
    out[:n] = rate[: n * factor].reshape(n, factor).mean(axis=1)
    return out


def time_of_day_rgb(tod_s: np.ndarray) -> np.ndarray:
    """RGB colour per time-of-day (seconds after midnight, Pacific)."""
    day = np.zeros_like(tod_s)
    m = (tod_s >= DAWN_S) & (tod_s < DAWN_S + TRANS_S)
    day[m] = (tod_s[m] - DAWN_S) / TRANS_S
    day[(tod_s >= DAWN_S + TRANS_S) & (tod_s < DUSK_S)] = 1.0
    m = (tod_s >= DUSK_S) & (tod_s < DUSK_S + TRANS_S)
    day[m] = 1.0 - (tod_s[m] - DUSK_S) / TRANS_S
    return day[:, None] * DAY_RGB + (1 - day[:, None]) * NIGHT_RGB


def style_axis(ax, ylim=None, yticks=None, show_x=False, show_y=True) -> None:
    """Offset, thick, minimal axes; x axis is drawn only on the bottom panel."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(show_x)
    ax.spines["left"].set_visible(show_y)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_LW)
        ax.spines[side].set_position(("outward", SPINE_OFFSET))
    if show_y and ylim is not None:
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.spines["left"].set_bounds(yticks[0], yticks[-1])
    else:
        ax.set_yticks([])
    ax.tick_params(width=SPINE_LW, length=7, labelsize=TICK_FS)
    if not show_x:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


def nice_ticks(lo: float, hi: float, step: float, include_zero: bool) -> list[float]:
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    ticks = [lo, hi]
    if include_zero and lo < 0 < hi:
        ticks.insert(1, 0.0)
    elif not include_zero:
        ticks.insert(1, (lo + hi) / 2)
    return ticks


def plot_unit(args) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,  # editable text in the PDF
    })
    rate_only = args.rate_only

    cols = [COL_TIME, COL_DATETIME] + ([] if rate_only else [COL_BEHAVIOR, COL_VELOCITY])
    df = load_dataset(args.dataset, columns=cols)
    units = load_units(args.units)
    if args.unit_id not in units:
        raise SystemExit(f"Unit {args.unit_id} not found. Available: {sorted(units)}")
    spikes = np.asarray(units[args.unit_id]["spike_times"], dtype="float64")
    spikes = spikes + args.spike_offset_s

    t = df[COL_TIME].to_numpy(dtype="float64")
    # Window start: where pose and spikes first overlap (behaviour plot) or where
    # the spikes begin (rate-only); --start-hours shifts it, --start-s overrides.
    base_start = spikes.min() if rate_only else max(t[0], spikes.min())
    t_start = (args.start_s if args.start_s is not None
               else base_start + args.start_hours * 3600.0)
    duration_h = args.duration_hours if args.duration_hours is not None else (
        np.inf if rate_only else 36.0)
    t_end = min(t_start + duration_h * 3600.0, spikes.max())
    if not rate_only:
        t_end = min(t_end, t[-1])
    n_bins = int(np.floor((t_end - t_start) / args.plot_bin_s + 1e-6))  # tolerance: float error
    if n_bins < 1:
        raise SystemExit("Empty window: check --start-hours/--start-s/--duration-hours.")
    t_end = t_start + n_bins * args.plot_bin_s
    hours = (np.arange(n_bins) + 0.5) * args.plot_bin_s / 3600.0
    dur_h = (t_end - t_start) / 3600.0

    rate = firing_rate_binned(spikes, t_start, t_end, args.plot_bin_s, n_bins)

    # Pacific clock time, anchored on the nearest dataset row and extended linearly
    # (so the strip also covers spike data beyond the pose dataset).
    i0 = min(int(np.searchsorted(t, t_start)), len(t) - 1)
    dt0 = df[COL_DATETIME].iloc[i0]
    tod0 = dt0.hour * 3600 + dt0.minute * 60 + dt0.second + dt0.microsecond / 1e6
    tod0 += t_start - t[i0]
    tod_s = (tod0 + hours * 3600.0) % 86400.0

    if rate_only:
        fig, (ax_r, ax_t) = plt.subplots(
            2, 1, sharex=True, figsize=(RATE_ONLY_WIDTH, 5.5),
            gridspec_kw={"height_ratios": [1, 0.12], "hspace": 0.25},
        )
    else:
        fig, (ax_b, ax_v, ax_r, ax_t) = plt.subplots(
            4, 1, sharex=True, figsize=(FIG_WIDTH, 10),
            gridspec_kw={"height_ratios": [0.35, 1, 1, 0.12], "hspace": 0.25},
        )
        movement = (df[COL_BEHAVIOR] == "movement").to_numpy(dtype="float64")
        behavior = bin_mean(t, movement, t_start, n_bins, args.plot_bin_s) > 0.5
        velocity = bin_mean(
            t, df[COL_VELOCITY].to_numpy(dtype="float64"), t_start, n_bins,
            args.plot_bin_s,
        )

        # 1. behaviour classification
        ax_b.imshow(
            behavior[None, :].astype(int), aspect="auto", interpolation="nearest",
            cmap=ListedColormap([REST_COLOR, MOVE_COLOR]), vmin=0, vmax=1,
            extent=[0, dur_h, 0, 1],
        )
        style_axis(ax_b, show_y=False)
        ax_b.legend(
            handles=[Patch(color=REST_COLOR, label="Rest"),
                     Patch(color=MOVE_COLOR, label="Movement")],
            loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
            fontsize=LEGEND_FS, handlelength=1.2, borderaxespad=0.2,
        )

        # 2. velocity
        ax_v.plot(hours, velocity, color="black", linewidth=1)
        vt = nice_ticks(np.nanmin(velocity), np.nanmax(velocity), 20, include_zero=True)
        ax_v.set_ylabel("Velocity (mm/s)", fontsize=LABEL_FS)
        style_axis(ax_v, (vt[0], vt[-1]), vt)

    # firing rate
    ax_r.plot(hours, rate, color="black", linewidth=1)
    if args.rate_ylim is not None:
        lo, hi = args.rate_ylim
        ylim_r, yticks_r = (lo, hi), [lo, (lo + hi) / 2, hi]
    else:  # auto: 0 to a rounded maximum
        rt = nice_ticks(0, np.nanmax(rate), 5, include_zero=False)
        ylim_r, yticks_r = (0, rt[-1]), [0, rt[1], rt[-1]]
    ax_r.set_ylabel("Firing rate (Hz)", fontsize=LABEL_FS)
    style_axis(ax_r, ylim_r, yticks_r)

    # time-of-day strip (bottom); carries the shared hours axis
    ax_t.imshow(time_of_day_rgb(tod_s)[None, :, :], aspect="auto",
                interpolation="bilinear", extent=[0, dur_h, 0, 1])
    style_axis(ax_t, show_x=True, show_y=False)
    ax_t.set_xlabel("Time (hours)", fontsize=LABEL_FS)
    # Long windows: tick every 24 h (one per light/dark cycle); else 4 even ticks.
    xt = np.arange(0, dur_h + 1e-9, 24.0) if dur_h > 48 else np.linspace(0, dur_h, 4)
    ax_t.set_xlim(0, dur_h)
    ax_t.set_xticks(xt)
    ax_t.set_xticklabels([f"{x:g}" for x in np.round(xt, 1)])
    ax_t.spines["bottom"].set_bounds(0, dur_h)

    if rate_only:
        fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2)
    else:
        fig.align_ylabels([ax_v, ax_r])
        fig.subplots_adjust(left=0.17, right=0.95, top=0.9, bottom=0.11)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--unit-id", type=int, required=True, help="Unit key in the pickle.")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--units", type=Path, default=UNITS)
    p.add_argument("--spike-offset-s", type=float, default=SPIKE_OFFSET_S)
    p.add_argument("--start-s", type=float, default=None,
                   help="Window start (time_since_start seconds); default: "
                        "first time pose data and spikes overlap.")
    p.add_argument("--start-hours", type=float, default=0.0,
                   help="Shift the window start by this many hours from its default "
                        "start (ignored if --start-s is given).")
    p.add_argument("--duration-hours", type=float, default=None,
                   help="Window length in hours. Default: 36 for the full plot; "
                        "the whole spike range (~100 h) with --rate-only.")
    p.add_argument("--rate-only", action="store_true",
                   help="Plot only the firing rate (+ time-of-day strip) over the "
                        "full spike range instead of behaviour/velocity/rate.")
    p.add_argument("--plot-bin-s", type=float, default=10.0,
                   help="Averaging bin for plotting (seconds).")
    p.add_argument("--rate-ylim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Restrict the firing-rate y-axis to this range in Hz "
                        "(e.g. --rate-ylim 2.5 5.5). Default: auto, 0 to max.")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    fig = plot_unit(args)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        tag = "firing_rate" if args.rate_only else "activity"
        out = (args.save_dir / f"unit_{args.unit_id}_{tag}.{ext}").resolve()
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
