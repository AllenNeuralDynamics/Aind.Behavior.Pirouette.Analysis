"""Plot the distribution of mean firing rates across units.

Mean rate per unit = number of spikes / recording duration, where the duration
is the span of spike times across *all* units in the file (so every unit is
normalised by the same length, even if it was silent at the edges).

Rates are heavily right-skewed, so by default the histogram uses log-spaced bins
on a log x axis; pass ``--linear`` for linear bins/axis.

Usage
-----
    python scripts/firing_rate_distribution.py
    python scripts/firing_rate_distribution.py --linear --bins 40 --no-show
    python scripts/firing_rate_distribution.py --save-dir C:/Users/brandon.pratt/Desktop/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pirouette_data.visualization_gui import load_units

REPO_ROOT = Path(__file__).resolve().parent.parent
UNITS = Path(
    r"C:\Users\brandon.pratt\Desktop\data\body-kinematics\units"
    r"\good_units_100hrs_v2.pkl"
)

# Figure style (matches the other plots): Arial, thick offset spines.
LABEL_FS = 22
TICK_FS = 18
SPINE_LW = 2.5
SPINE_OFFSET = 10  # points
FIG_WIDTH = 6  # inches
FIG_HEIGHT = 8  # inches


def mean_firing_rates(units: dict) -> tuple[np.ndarray, float]:
    """Mean rate (Hz) of every unit and the shared recording duration (s)."""
    spikes = {uid: np.asarray(u["spike_times"], dtype="float64") for uid, u in units.items()}
    t0 = min(s.min() for s in spikes.values() if s.size)
    t1 = max(s.max() for s in spikes.values() if s.size)
    duration = t1 - t0
    return np.array([s.size / duration for s in spikes.values()]), duration


def plot_distribution(rates: np.ndarray, bins: int, log: bool) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,  # editable text in the PDF
    })
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if log:
        pos = rates[rates > 0]
        edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1)
        ax.set_xscale("log")
    else:
        edges = np.linspace(0, rates.max(), bins + 1)
    ax.hist(rates, bins=edges, color="black", edgecolor="white", linewidth=0.5)
    median = np.median(rates)
    ax.axvline(median, color="red", linestyle=":", linewidth=SPINE_LW)

    ax.set_xlabel("Mean firing rate (Hz)", fontsize=LABEL_FS)
    ax.set_ylabel("Units (#)", fontsize=LABEL_FS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_LW)
        ax.spines[side].set_position(("outward", SPINE_OFFSET))
    ax.tick_params(width=SPINE_LW, length=7, labelsize=TICK_FS)
    ax.tick_params(which="minor", width=SPINE_LW * 0.6, length=4)
    ax.text(0.03, 0.95, f"n = {rates.size} units\nmedian = {median:.2f} Hz",
            transform=ax.transAxes, ha="left", va="top", fontsize=TICK_FS)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--units", type=Path, default=UNITS)
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--linear", action="store_true",
                   help="Linear bins and x axis instead of log.")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    rates, duration = mean_firing_rates(load_units(args.units))
    print(f"{rates.size} units over {duration / 3600:.1f} h; "
          f"median {np.median(rates):.2f} Hz, range "
          f"{rates.min():.4f}-{rates.max():.2f} Hz")
    fig = plot_distribution(rates, args.bins, log=not args.linear)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = (args.save_dir / f"firing_rate_distribution.{ext}").resolve()
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
