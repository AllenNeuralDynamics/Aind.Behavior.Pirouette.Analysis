"""Plots comparing spike-sorting runs across subsampling times."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Figure style shared with the other Pirouette plots: Arial, thick offset spines.
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 16
SPINE_LW = 2.5
SPINE_OFFSET = 10  # points


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _style_for(i: int, n: int) -> tuple:
    """(color, marker) for line *i* of *n*: distinct colors, then distinct markers.

    Uses tab10 up to 10 runs, then the 20-color tab20 set (saturated colors
    first); beyond 20 runs the marker shape also changes so no pair repeats.
    """
    if n <= 10:
        return plt.get_cmap("tab10")(i), "o"
    tab20 = plt.get_cmap("tab20")
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return tab20(order[i % 20]), MARKERS[(i // 20) % len(MARKERS)]


def apply_qc_style() -> None:
    """Set the Arial / editable-PDF-text rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })


def plot_metric_vs_subsampling(
    runs: dict[str, pd.DataFrame],
    y_col: str,
    ylabel: str,
    log_x: bool = True,
    figsize: tuple[float, float] = (12, 7),
) -> plt.Figure:
    """One line per dataset: *y_col* against ``subsampling_minutes``.

    Parameters
    ----------
    runs:
        ``{legend_label: DataFrame}`` from :func:`pirouette_data.qc.load_runs`.
    y_col, ylabel:
        Column to plot and its axis label.
    log_x:
        Log2 x axis with ticks at the sampled subsampling times (they double
        each step); ``False`` for a linear axis.
    """
    apply_qc_style()
    fig, ax = plt.subplots(figsize=figsize)
    n = len(runs)
    for i, (label, df) in enumerate(runs.items()):
        color, marker = _style_for(i, n)
        ax.plot(df["subsampling_minutes"], df[y_col], marker=marker, markersize=8,
                linewidth=2.5, color=color, label=label)

    if log_x:
        ticks = np.unique(np.concatenate(
            [df["subsampling_minutes"].to_numpy() for df in runs.values()]))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Subsampling time (min)", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_LW)
        ax.spines[side].set_position(("outward", SPINE_OFFSET))
    ax.tick_params(width=SPINE_LW, length=7, labelsize=TICK_FS)
    # Legend outside the axes (right) so it never covers the data.
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout()
    return fig


def plot_num_clusters(runs: dict[str, pd.DataFrame], **kwargs) -> plt.Figure:
    """Number of clusters (``num_units``) vs subsampling time."""
    return plot_metric_vs_subsampling(
        runs, "num_units", "Number of clusters (#)", **kwargs)


def plot_find_templates_time(runs: dict[str, pd.DataFrame], **kwargs) -> plt.Figure:
    """Find-templates capsule run duration (min) vs subsampling time."""
    runs_min = {k: df.assign(find_templates_min=df["find_templates_s"] / 60.0)
                for k, df in runs.items()}
    return plot_metric_vs_subsampling(
        runs_min, "find_templates_min", "find template run duration (min)", **kwargs)
