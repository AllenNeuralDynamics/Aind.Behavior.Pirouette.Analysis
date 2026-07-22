"""Processing utilities that derive new columns from the pose DataFrame.

The main capability here is spatial calibration: converting tracked keypoints
from pixels to millimetres using the chamber corners as a known-size reference.

The chamber (spelled ``champber`` for two of the corner keypoints because of a
labelling typo) has fixed physical dimensions:

* length = 373 mm  (the horizontal edges: upper-left -> upper-right and
  lower-left -> lower-right)
* width  = 194 mm  (the vertical edges: upper-left -> lower-left and
  upper-right -> lower-right)

For each frame the four edge lengths are measured in pixels; the pooled median of
the two horizontal edges gives the length in pixels and the pooled median of the
two vertical edges gives the width in pixels. Combined with the known mm
dimensions these yield per-axis scale factors (mm/pixel) that are applied to
every tracked keypoint to produce ``<bodypart>_x_mm`` / ``<bodypart>_y_mm``
columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Known chamber dimensions in millimetres.
CHAMBER_LENGTH_MM = 373.0
CHAMBER_WIDTH_MM = 194.0

#: Body-part names of the four chamber corners (note the ``champber`` typo on the
#: upper corners is intentional — it matches the tracked keypoint names).
DEFAULT_CHAMBER_CORNERS: dict[str, str] = {
    "ul": "ul_champber",  # upper-left
    "ur": "ur_champber",  # upper-right
    "lr": "lr_chamber",  # lower-right
    "ll": "ll_chamber",  # lower-left
}


@dataclass(frozen=True)
class ChamberScale:
    """Result of :func:`estimate_chamber_scale`.

    Attributes
    ----------
    length_px:
        Median chamber length in pixels (horizontal edges).
    width_px:
        Median chamber width in pixels (vertical edges).
    mm_per_px_x:
        Millimetres-per-pixel scale for the x-axis (``length_mm / length_px``).
    mm_per_px_y:
        Millimetres-per-pixel scale for the y-axis (``width_mm / width_px``).
    """

    length_px: float
    width_px: float
    mm_per_px_x: float
    mm_per_px_y: float


def _edge_length_px(
    df: pd.DataFrame,
    corner_a: str,
    corner_b: str,
    likelihood_threshold: float,
) -> np.ndarray:
    """Per-frame Euclidean distance (pixels) between two chamber corners.

    Frames where either corner falls below *likelihood_threshold* (when the
    ``*_likelihood`` columns are present) are returned as ``NaN`` so they are
    excluded from the median.
    """
    ax = df[f"{corner_a}_x"].to_numpy(dtype="float64")
    ay = df[f"{corner_a}_y"].to_numpy(dtype="float64")
    bx = df[f"{corner_b}_x"].to_numpy(dtype="float64")
    by = df[f"{corner_b}_y"].to_numpy(dtype="float64")
    dist = np.hypot(ax - bx, ay - by)

    if likelihood_threshold > 0.0:
        for corner in (corner_a, corner_b):
            like_col = f"{corner}_likelihood"
            if like_col in df.columns:
                dist = np.where(
                    df[like_col].to_numpy(dtype="float64") > likelihood_threshold,
                    dist,
                    np.nan,
                )
    return dist


def estimate_chamber_scale(
    df: pd.DataFrame,
    corners: dict[str, str] | None = None,
    length_mm: float = CHAMBER_LENGTH_MM,
    width_mm: float = CHAMBER_WIDTH_MM,
    likelihood_threshold: float = 0.0,
) -> ChamberScale:
    """Estimate pixel->mm scale factors from the chamber corner keypoints.

    The length in pixels is the median of the pooled horizontal-edge distances
    (upper-left -> upper-right and lower-left -> lower-right); the width in pixels
    is the median of the pooled vertical-edge distances (upper-left -> lower-left
    and upper-right -> lower-right). Medians are taken over all frames and both
    edges of each orientation, ignoring ``NaN``.

    Parameters
    ----------
    df:
        Flattened pose DataFrame containing the four chamber-corner
        ``<corner>_x`` / ``<corner>_y`` columns (and optionally
        ``<corner>_likelihood``).
    corners:
        Mapping with keys ``"ul"``, ``"ur"``, ``"lr"``, ``"ll"`` to the corner
        body-part names. Defaults to :data:`DEFAULT_CHAMBER_CORNERS`.
    length_mm, width_mm:
        Known physical chamber dimensions in millimetres.
    likelihood_threshold:
        When > 0 and ``*_likelihood`` columns exist, frames where a corner is
        below this confidence are excluded from the medians.

    Returns
    -------
    ChamberScale
        The estimated pixel lengths and per-axis mm/pixel scale factors.

    Raises
    ------
    KeyError
        If any expected corner coordinate column is missing.
    ValueError
        If no valid frames remain to estimate a median (all ``NaN``).
    """
    corners = corners or DEFAULT_CHAMBER_CORNERS
    required = [f"{corners[k]}_{ax}" for k in ("ul", "ur", "lr", "ll") for ax in "xy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected chamber corner columns: {missing}")

    # Horizontal edges -> length; vertical edges -> width.
    length_edges = np.concatenate(
        [
            _edge_length_px(df, corners["ul"], corners["ur"], likelihood_threshold),
            _edge_length_px(df, corners["ll"], corners["lr"], likelihood_threshold),
        ]
    )
    width_edges = np.concatenate(
        [
            _edge_length_px(df, corners["ul"], corners["ll"], likelihood_threshold),
            _edge_length_px(df, corners["ur"], corners["lr"], likelihood_threshold),
        ]
    )

    if np.all(np.isnan(length_edges)) or np.all(np.isnan(width_edges)):
        raise ValueError(
            "No valid frames to estimate the chamber scale (all distances NaN); "
            "try lowering likelihood_threshold."
        )

    length_px = float(np.nanmedian(length_edges))
    width_px = float(np.nanmedian(width_edges))

    return ChamberScale(
        length_px=length_px,
        width_px=width_px,
        mm_per_px_x=length_mm / length_px,
        mm_per_px_y=width_mm / width_px,
    )


def keypoint_columns(df: pd.DataFrame) -> list[str]:
    """Return the base names of tracked keypoints (those with ``_x`` and ``_y``).

    Parameters
    ----------
    df:
        Flattened pose DataFrame.

    Returns
    -------
    list[str]
        Body-part base names for which both ``<name>_x`` and ``<name>_y`` exist,
        in column order.
    """
    bases = []
    for col in df.columns:
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns:
            bases.append(col[:-2])
    return bases


def append_mm_columns(
    df: pd.DataFrame,
    scale: ChamberScale | None = None,
    corners: dict[str, str] | None = None,
    length_mm: float = CHAMBER_LENGTH_MM,
    width_mm: float = CHAMBER_WIDTH_MM,
    likelihood_threshold: float = 0.0,
    suffix: str = "_mm",
) -> pd.DataFrame:
    """Append millimetre coordinate columns for every tracked keypoint.

    For each body part with ``<name>_x`` / ``<name>_y`` pixel columns, adds
    ``<name>_x<suffix>`` and ``<name>_y<suffix>`` obtained by scaling the pixel
    coordinates with the chamber-derived mm/pixel factors (x uses the length
    scale, y uses the width scale).

    Parameters
    ----------
    df:
        Flattened pose DataFrame.
    scale:
        Pre-computed :class:`ChamberScale`. When ``None``, it is estimated from
        *df* via :func:`estimate_chamber_scale` using the parameters below.
    corners, length_mm, width_mm, likelihood_threshold:
        Passed to :func:`estimate_chamber_scale` when *scale* is ``None``.
    suffix:
        Suffix appended after ``_x`` / ``_y`` for the new columns (default
        ``"_mm"`` -> e.g. ``left_ear_x_mm``).

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with the millimetre columns added.
    """
    if scale is None:
        scale = estimate_chamber_scale(
            df,
            corners=corners,
            length_mm=length_mm,
            width_mm=width_mm,
            likelihood_threshold=likelihood_threshold,
        )

    out = df.copy()
    for base in keypoint_columns(df):
        out[f"{base}_x{suffix}"] = df[f"{base}_x"].to_numpy() * scale.mm_per_px_x
        out[f"{base}_y{suffix}"] = df[f"{base}_y"].to_numpy() * scale.mm_per_px_y
    return out
