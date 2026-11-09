"""Load spike-sorting subsampling-test results.

Each result is a folder (e.g. ``10m_base_5_powers_motion``) holding a
``fit_summary.csv`` with one row per sorting run: ``subsampling_minutes``,
``num_units`` (number of clusters), ``fit_time_min`` (total run time), etc.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

SUMMARY_NAME = "fit_summary.csv"
LOG_NAME = "output"  # Nextflow pipeline log
REQUIRED_COLUMNS = ("subsampling_minutes", "num_units", "fit_time_min")

# Lines in the find-templates capsule log, one pair per fit.
_FIT_START = re.compile(r"Starting DARTsort model fitting.*?(fit\d{4})")
_FIT_DONE = re.compile(r"^Done\. Total time: ([\d.]+)s")
_FIT_ID = re.compile(r"fit\d{4}")


def parse_find_templates_durations(log_path: str | Path) -> dict[str, float]:
    """Run duration (s) of each fit in the ``find_templates`` capsule log.

    The capsule prints ``Starting DARTsort model fitting ... fitNNNN`` and, when
    that fit finishes, ``Done. Total time: <seconds>s``.

    Returns
    -------
    dict
        ``{"fit0000": seconds, ...}``; empty if the log has no such lines.
    """
    durations: dict[str, float] = {}
    current: str | None = None
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _FIT_START.search(line)
            if m:
                current = m.group(1)
                continue
            m = _FIT_DONE.match(line)
            if m and current is not None:
                durations[current] = float(m.group(1))
                current = None
    return durations


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
    df = df.sort_values("subsampling_minutes").reset_index(drop=True)
    df["find_templates_s"] = _find_templates_seconds(folder, df)
    return df


def _find_templates_seconds(folder: Path, df: pd.DataFrame) -> pd.Series:
    """Find-templates capsule duration (s) per row.

    Taken from the pipeline log (``output``); where the log or a fit is missing,
    falls back to ``fit_time_min * 60`` (the DARTsort fit alone, ~90 s shorter)
    with a warning.
    """
    fallback = df["fit_time_min"] * 60.0
    log = folder / LOG_NAME
    durations = parse_find_templates_durations(log) if log.is_file() else {}
    if "run_name" in df.columns:
        ids = df["run_name"].astype(str).str.extract(f"({_FIT_ID.pattern})")[0]
    else:
        ids = pd.Series([None] * len(df), index=df.index)
    from_log = ids.map(durations)
    if from_log.isna().any():
        warnings.warn(
            f"{folder.name}: find-templates duration not found in {LOG_NAME} for "
            f"{int(from_log.isna().sum())} run(s); using fit_time_min * 60",
            stacklevel=3,
        )
    return from_log.fillna(fallback)


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
