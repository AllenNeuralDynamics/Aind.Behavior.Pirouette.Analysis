"""Velocity-based behavior classification.

Labels each frame as ``"rest"`` or ``"movement"`` by thresholding the mouse's
speed (typically the smoothed ear-midpoint velocity from
:mod:`pirouette_data.kinematics`).

The threshold can be supplied explicitly or estimated **unsupervised** from the
speed distribution with Otsu's method (the speed histogram is bimodal: a large
low-speed rest peak and a spread-out movement mode).

Two mechanisms clean up the binary label sequence:

* a **median filter** (window in frames) that removes very short flickers, and
* a **minimum bout duration** (in seconds): movement bouts shorter than
  ``min_bout_s`` are relabelled rest, and brief sub-``bridge_gap_s`` dips within
  movement are bridged. This is what captures *slow but sustained* movement — set
  a sensitive (low) threshold so slow motion clears it, then keep only bouts that
  persist for at least ``min_bout_s`` (e.g. 0.5 s).

Because the slow-movement mode sits close to the rest peak, a plain Otsu split of
the raw speed lands far out on the tail (missing slow movement). Passing
``log=True`` runs Otsu on ``log1p(speed)``, which places the threshold just above
the rest floor and is the recommended setting for capturing slow movement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

#: Default velocity column to classify on (the smoothed ear-midpoint velocity).
DEFAULT_VELOCITY_COLUMN = "ear_velocity_smooth_mm_s"

REST_LABEL = "rest"
MOVEMENT_LABEL = "movement"


# ---------------------------------------------------------------------------
# Unsupervised threshold
# ---------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, nbins: int = 256, log: bool = False) -> float:
    """Otsu's method: the threshold that maximises between-class variance.

    Parameters
    ----------
    values:
        1-D array of speeds (non-finite entries are ignored).
    nbins:
        Number of histogram bins.
    log:
        When ``True``, the threshold is found on ``log1p(values)`` and mapped back
        to the linear scale. Useful when the rest peak dominates near zero.

    Returns
    -------
    float
        The speed threshold separating the low (rest) and high (movement) modes.

    Raises
    ------
    ValueError
        If *values* contains no finite entries.
    """
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to threshold.")
    work = np.log1p(v) if log else v

    hist, edges = np.histogram(work, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight = hist.astype("float64")
    total = weight.sum()
    if total == 0:
        return float(np.median(v))

    w_bg = np.cumsum(weight)
    w_fg = total - w_bg
    sum_total = np.sum(weight * centers)
    sum_bg = np.cumsum(weight * centers)

    # Avoid divide-by-zero / NaN warnings for empty classes.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    between[~np.isfinite(between)] = 0.0

    thr = centers[int(np.argmax(between))]
    return float(np.expm1(thr) if log else thr)


def estimate_velocity_threshold(
    speed: np.ndarray, method: str = "otsu", nbins: int = 256, log: bool = False
) -> float:
    """Estimate a rest/movement speed threshold with an unsupervised method.

    Parameters
    ----------
    speed:
        1-D array of (non-negative) speeds.
    method:
        Currently only ``"otsu"`` (see :func:`otsu_threshold`).
    nbins, log:
        Passed to :func:`otsu_threshold`.

    Returns
    -------
    float
        The estimated threshold.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method != "otsu":
        raise ValueError("Unknown method; only 'otsu' is supported.")
    return otsu_threshold(speed, nbins=nbins, log=log)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _as_odd(size: int) -> int:
    """Return *size* forced to the nearest odd integer >= 1."""
    size = int(size)
    if size < 1:
        return 1
    return size if size % 2 == 1 else size + 1


def _remove_short_runs(binary: np.ndarray, min_len: int, value: int) -> np.ndarray:
    """Flip runs equal to *value* shorter than *min_len* to the other value.

    Used to enforce minimum bout durations: e.g. ``value=1, min_len=30`` relabels
    movement bouts shorter than 30 frames as rest; ``value=0`` bridges short rest
    gaps into movement.
    """
    out = np.asarray(binary).copy()
    n = out.size
    if min_len <= 1 or n == 0:
        return out
    change = np.flatnonzero(np.diff(out)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for s, e in zip(starts, ends):
        if out[s] == value and (e - s) < min_len:
            out[s:e] = 1 - value
    return out


def classify_rest_movement(
    velocity: np.ndarray,
    threshold: float | None = None,
    method: str = "otsu",
    fps: float | None = None,
    min_bout_s: float | None = 0.5,
    bridge_gap_s: float | None = 0.2,
    median_filter_size: int = 1,
    use_abs: bool = True,
    rest_label: str = REST_LABEL,
    movement_label: str = MOVEMENT_LABEL,
    nbins: int = 256,
    log: bool = True,
) -> np.ndarray:
    """Classify each frame as rest or movement from a velocity signal.

    Speed (``abs(velocity)`` by default) is thresholded into a binary sequence,
    which is then cleaned up in this order:

    1. optional **median filter** (``median_filter_size`` frames);
    2. **bridge** rest gaps shorter than ``bridge_gap_s`` (relabel to movement),
       so a slow movement briefly dipping below threshold stays continuous;
    3. **remove** movement bouts shorter than ``min_bout_s`` (relabel to rest),
       keeping only sustained movement.

    Steps 2-3 require *fps*; they are skipped if *fps* is ``None``.

    To capture slow-but-sustained movement, use a sensitive threshold (the default
    ``log=True`` Otsu, which sits just above the rest floor) together with
    ``min_bout_s`` (e.g. 0.5 s).

    Parameters
    ----------
    velocity:
        1-D velocity (signed) or speed array.
    threshold:
        Speed threshold; frames at or above it are movement. When ``None`` it is
        estimated via :func:`estimate_velocity_threshold` using *method*.
    method:
        Unsupervised threshold method used when *threshold* is ``None``.
    fps:
        Sampling rate (frames/second), required for the ``min_bout_s`` /
        ``bridge_gap_s`` steps. When ``None`` those steps are skipped.
    min_bout_s:
        Minimum movement-bout duration (seconds); shorter movement bouts are
        relabelled rest. ``None`` or ``0`` disables.
    bridge_gap_s:
        Rest gaps shorter than this (seconds) inside movement are bridged to
        movement. ``None`` or ``0`` disables.
    median_filter_size:
        Window (frames) of an optional median filter on the binary labels; forced
        to the nearest odd integer. ``<= 1`` disables (the default).
    use_abs:
        When ``True`` (default), classify on ``abs(velocity)`` so the sign
        (forward/backward) does not matter.
    rest_label, movement_label:
        Output label strings.
    nbins, log:
        Passed to the threshold estimator (``log=True`` recommended for slow
        movement).

    Returns
    -------
    numpy.ndarray
        Object array of per-frame labels. Frames with non-finite velocity are
        labelled *rest_label*.
    """
    v = np.asarray(velocity, dtype="float64")
    speed = np.abs(v) if use_abs else v
    finite = np.isfinite(speed)

    if threshold is None:
        threshold = estimate_velocity_threshold(
            speed[finite], method=method, nbins=nbins, log=log
        )

    binary = np.zeros(speed.shape, dtype="int8")
    binary[finite & (speed >= threshold)] = 1

    size = _as_odd(median_filter_size)
    if size > 1:
        binary = median_filter(binary, size=size, mode="nearest")

    if fps is not None and fps > 0:
        if bridge_gap_s:
            binary = _remove_short_runs(binary, int(round(bridge_gap_s * fps)), value=0)
        if min_bout_s:
            binary = _remove_short_runs(binary, int(round(min_bout_s * fps)), value=1)

    labels = np.where(binary == 1, movement_label, rest_label).astype(object)
    return labels


def append_behavior_labels(
    df: pd.DataFrame,
    velocity_column: str = DEFAULT_VELOCITY_COLUMN,
    threshold: float | None = None,
    method: str = "otsu",
    fps: float | None = None,
    min_bout_s: float | None = 0.5,
    bridge_gap_s: float | None = 0.2,
    median_filter_size: int = 1,
    use_abs: bool = True,
    rest_label: str = REST_LABEL,
    movement_label: str = MOVEMENT_LABEL,
    column: str = "behavior",
    time_column: str = "harp_time",
    nbins: int = 256,
    log: bool = True,
) -> pd.DataFrame:
    """Append a rest/movement behavior label column to the DataFrame.

    Wrapper around :func:`classify_rest_movement`. When *fps* is ``None`` it is
    derived from *time_column* (median frame interval), enabling the
    ``min_bout_s`` / ``bridge_gap_s`` steps. The speed threshold used is recorded
    in ``df.attrs["behavior_velocity_threshold"]``.

    The defaults (``log=True`` Otsu + ``min_bout_s=0.5``) are tuned to capture
    slow but sustained movement.

    Parameters
    ----------
    df:
        DataFrame containing *velocity_column* (e.g. from
        :func:`pirouette_data.kinematics.append_ear_velocity`).
    velocity_column:
        Column holding the velocity/speed to classify (default the smoothed
        ear-midpoint velocity).
    fps:
        Sampling rate; when ``None`` it is computed from *time_column*.
    time_column:
        Column of timestamps (seconds) used to derive *fps* (default
        ``"harp_time"``).
    threshold, method, min_bout_s, bridge_gap_s, median_filter_size, use_abs, rest_label, movement_label, nbins, log:
        Passed to :func:`classify_rest_movement`.
    column:
        Name of the appended label column (default ``"behavior"``).

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with the label column added.

    Raises
    ------
    KeyError
        If *velocity_column* is missing.
    """
    if velocity_column not in df.columns:
        raise KeyError(f"df must contain the velocity column {velocity_column!r}.")

    if fps is None and time_column in df.columns:
        dt = float(np.median(np.diff(df[time_column].to_numpy(dtype="float64"))))
        if dt > 0:
            fps = 1.0 / dt

    velocity = df[velocity_column].to_numpy(dtype="float64")
    speed = np.abs(velocity) if use_abs else velocity
    used_threshold = (
        threshold
        if threshold is not None
        else estimate_velocity_threshold(
            speed[np.isfinite(speed)], method=method, nbins=nbins, log=log
        )
    )

    labels = classify_rest_movement(
        velocity,
        threshold=used_threshold,
        method=method,
        fps=fps,
        min_bout_s=min_bout_s,
        bridge_gap_s=bridge_gap_s,
        median_filter_size=median_filter_size,
        use_abs=use_abs,
        rest_label=rest_label,
        movement_label=movement_label,
        nbins=nbins,
        log=log,
    )

    out = df.copy()
    out[column] = labels
    out.attrs["behavior_velocity_threshold"] = float(used_threshold)
    return out
