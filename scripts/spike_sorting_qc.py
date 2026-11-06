"""Compare spike-sorting runs across subsampling times.

Loads every result folder in ``--root`` (each holds a ``fit_summary.csv``) and
draws two figures, with all datasets overlaid and the legend keyed by the folder
name (underscores removed):

1. number of clusters vs subsampling time,
2. find templates capsule run duration (s) vs subsampling time.

The x axis is log2 (subsampling times double each step); pass ``--linear-x`` for
a linear axis.

Usage
-----
    python scripts/spike_sorting_qc.py
    python scripts/spike_sorting_qc.py --root C:/path/to/results --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from pirouette_data.qc import load_runs, plot_num_clusters, plot_find_templates_time

load_dotenv()  # defaults below come from .env (CLI flags override)
REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.getenv("SPIKE_SORTING_QC_ROOT"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR") or REPO_ROOT / "figures")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Folder containing one result folder per dataset.",
    )
    p.add_argument(
        "--linear-x",
        action="store_true",
        help="Linear instead of log2 subsampling-time axis.",
    )
    p.add_argument("--save-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    runs = load_runs(args.root)
    for label, df in runs.items():
        lo, hi = df["subsampling_minutes"].min(), df["subsampling_minutes"].max()
        print(f"{label}: {len(df)} runs, {lo:g}-{hi:g} min")

    log_x = not args.linear_x
    figs = {
        "spike_sorting_qc_num_clusters": plot_num_clusters(runs, log_x=log_x),
        "spike_sorting_qc_find_templates_time": plot_find_templates_time(runs, log_x=log_x),
    }
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            out = (args.save_dir / f"{name}.{ext}").resolve()
            fig.savefig(out, dpi=200)
            print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
