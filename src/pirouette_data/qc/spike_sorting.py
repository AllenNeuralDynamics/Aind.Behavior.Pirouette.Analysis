"""Load spike-sorting subsampling-test results.

Each result is a folder (e.g. ``10m_base_5_powers_motion``) holding a
``fit_summary.csv`` with one row per sorting run: ``subsampling_minutes``,
``num_units`` (number of clusters), ``fit_time_min`` (total run time), etc.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

SUMMARY_NAME = "fit_summary.csv"
REQUIRED_COLUMNS = ("subsampling_minutes", "num_units", "fit_time_min")


def label_from_folder(folder: str | Path) -> str:
    """Legend label from a result folder: base name with underscores removed.

    ``"10m_base_5_powers_motion"`` -> ``"10m base 5 powers motion"``.
    """
    return Path(folder).name.replace("_", " ")


def find_run_dirs(root: str | Path) -> list[Path]:
    """Sorted sub-folders of *root* that contain a ``fit_summary.csv``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")
    return sorted(p for p in root.iterdir() if (p / SUMMARY_NAME).is_file())


def load_fit_summary(folder: str | Path, only_ok: bool = True) -> pd.DataFrame:
    """Load one folder's ``fit_summary.csv``, sorted by subsampling time.

    Parameters
    ----------
    folder:
        Result folder containing ``fit_summary.csv``.
    only_ok:
        Keep only rows with ``status == "ok"`` (when a status column exists);
        a warning is issued if rows are dropped.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    folder = Path(folder)
    df = pd.read_csv(folder / SUMMARY_NAME)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{folder / SUMMARY_NAME} is missing columns: {missing}")
    if only_ok and "status" in df.columns:
        bad = df["status"] != "ok"
        if bad.any():
            warnings.warn(
                f"{folder.name}: dropping {int(bad.sum())} run(s) with status != 'ok'",
                stacklevel=2,
            )
            df = df[~bad]
    return df.sort_values("subsampling_minutes").reset_index(drop=True)


def load_runs(root: str | Path, only_ok: bool = True) -> dict[str, pd.DataFrame]:
    """Load every result folder under *root*.

    Returns
    -------
    dict
        ``{legend_label: fit_summary DataFrame}`` in folder-name order.
    """
    dirs = find_run_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No folders with {SUMMARY_NAME} under {root}")
    return {label_from_folder(d): load_fit_summary(d, only_ok) for d in dirs}
