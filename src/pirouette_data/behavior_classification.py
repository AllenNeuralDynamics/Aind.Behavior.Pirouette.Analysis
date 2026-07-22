"""Velocity-based behavior classification.

Labels each frame as ``"rest"`` or ``"movement"`` by thresholding the mouse's
speed (typically the smoothed ear-midpoint velocity from
:mod:`pirouette_data.kinematics`).

The threshold can be supplied explicitly or estimated **unsupervised** from the
speed distribution with Otsu's method (the speed histogram is bimodal: a large
low-speed rest peak and a spread-out movement mode).

A median filter is then applied to the binary label sequence so that very short
bouts — e.g. a brief flicker of "movement" while the mouse is merely
repositioning during rest — are replaced by the longer sandwiching label.
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


def classify_rest_movement(
    velocity: np.ndarray,
    threshold: float | None = None,
    method: str = "otsu",
    median_filter_size: int = 15,
    use_abs: bool = True,
    rest_label: str = REST_LABEL,
    movement_label: str = MOVEMENT_LABEL,
    nbins: int = 256,
    log: bool = False,
) -> np.ndarray:
    """Classify each frame as rest or movement from a velocity signal.

    Speed (``abs(velocity)`` by default) is thresholded, then the binary sequence
    is median-filtered to remove bouts shorter than ~``median_filter_size / 2``
    frames — replacing them with the surrounding label.

    Parameters
    ----------
    velocity:
        1-D velocity (signed) or speed array.
    threshold:
        Speed threshold; frames at or above it are movement. When ``None`` it is
        estimated via :func:`estimate_velocity_threshold` using *method*.
    method:
        Unsupervised threshold method used when *threshold* is ``None``.
    median_filter_size:
        Window (frames) of the median filter applied to the binary labels; forced
        to the nearest odd integer. Larger removes longer spurious bouts. Pass
        ``<= 1`` to disable filtering.
    use_abs:
        When ``True`` (default), classify on ``abs(velocity)`` so the sign
        (forward/backward) does not matter.
    rest_label, movement_label:
        Output label strings.
    nbins, log:
        Passed to the threshold estimator.

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

    labels = np.where(binary == 1, movement_label, rest_label).astype(object)
    return labels


def append_behavior_labels(
    df: pd.DataFrame,
    velocity_column: str = DEFAULT_VELOCITY_COLUMN,
    threshold: float | None = None,
    method: str = "otsu",
    median_filter_size: int = 15,
    use_abs: bool = True,
    rest_label: str = REST_LABEL,
    movement_label: str = MOVEMENT_LABEL,
    column: str = "behavior",
    nbins: int = 256,
    log: bool = False,
) -> pd.DataFrame:
    """Append a rest/movement behavior label column to the DataFrame.

    Thin wrapper around :func:`classify_rest_movement`. The speed threshold that
    was used is recorded in ``df.attrs["behavior_velocity_threshold"]``.

    Parameters
    ----------
    df:
        DataFrame containing *velocity_column* (e.g. from
        :func:`pirouette_data.kinematics.append_ear_velocity`).
    velocity_column:
        Column holding the velocity/speed to classify (default the smoothed
        ear-midpoint velocity).
    threshold, method, median_filter_size, use_abs, rest_label, movement_label, nbins, log:
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
