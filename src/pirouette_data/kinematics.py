"""Kinematics utilities derived from behavior data streams.

This module estimates the animal's heading angle from the commutator
``AccumulatedCommutatorTurns`` stream and aligns it to the camera/pose timeline.

The commutator records the cumulative number of turns of the tether. Because the
tether rotates with the animal, the change in accumulated turns is a proxy for
the change in heading. The heading estimate is built as follows:

1. Subtract the first accumulated-turns value within the pose-estimate time
   window (the reference orientation).
2. Convert turns to degrees (``* 360``) and wrap the result into ``[0, 360)``.
3. Apply an angular offset so that "facing right" corresponds to 0 degrees
   (standard math quadrant). The offset is calibrated from the pose data — the
   first frame where both ears are tracked defines the animal's true facing
   direction (orthogonal to the inter-aural axis).

The commutator is sampled at ~10 Hz while the camera runs at ~60 Hz, so the
heading is interpolated onto the camera timestamps with
:class:`scipy.interpolate.interp1d`.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pirouette_data.ingestion import (
    _parse_s3_uri,
    get_s3_client,
    parse_camera_and_timestamp,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

#: Default commutator register (CSV filename prefix) used for the heading estimate.
DEFAULT_COMMUTATOR_REGISTER = "Commutator_AccumulatedCommutatorTurns"


# ---------------------------------------------------------------------------
# Core heading transform
# ---------------------------------------------------------------------------
def commutator_heading_estimate(
    accumulated_turns: np.ndarray | pd.Series,
    reference_value: float | None = None,
    offset_deg: float = 0.0,
    direction: int = 1,
) -> np.ndarray:
    """Estimate heading (degrees) from accumulated commutator turns.

    The transform is:

    1. ``net = (accumulated_turns - reference_value) * direction`` — turns
       relative to the reference orientation.
    2. ``deg = (net * 360) % 360`` — turns converted to degrees and wrapped into
       ``[0, 360)``.
    3. ``heading = (deg + offset_deg) % 360`` — rotated so that the desired zero
       (facing right, by convention) reads as 0 degrees.

    Parameters
    ----------
    accumulated_turns:
        The cumulative commutator turn measurements.
    reference_value:
        The accumulated-turns value treated as the reference (0-turn)
        orientation. When ``None``, the first element of *accumulated_turns* is
        used.
    offset_deg:
        Angular offset (degrees) added after wrapping so that "facing right"
        maps to 0 degrees. See :func:`heading_offset_from_ears`.
    direction:
        ``+1`` (default) or ``-1``. Flips the sign of the commutator rotation to
        match the standard counter-clockwise-positive convention if the physical
        wiring rotates the opposite way.

    Returns
    -------
    numpy.ndarray
        Heading in degrees, wrapped to ``[0, 360)``, aligned to the input.
    """
    turns = np.asarray(accumulated_turns, dtype="float64")
    if reference_value is None:
        reference_value = turns[0]
    net = (turns - reference_value) * direction
    deg = (net * 360.0) % 360.0
    return (deg + offset_deg) % 360.0


# ---------------------------------------------------------------------------
# Offset calibration from pose (ear keypoints)
# ---------------------------------------------------------------------------
def first_valid_ear_frame(
    df: pd.DataFrame,
    left_ear: str = "left_ear",
    right_ear: str = "right_ear",
    likelihood_threshold: float = 0.0,
) -> int:
    """Return the row position of the first frame with both ears tracked.

    A frame qualifies when both ear ``x``/``y`` coordinates are non-null and both
    ``likelihood`` values exceed *likelihood_threshold*.

    Parameters
    ----------
    df:
        Flattened pose DataFrame with ``<bodypart>_x``, ``<bodypart>_y`` and
        ``<bodypart>_likelihood`` columns (as produced by
        :func:`pirouette_data.ingestion.build_dataset`).
    left_ear, right_ear:
        Body-part names of the two ears.
    likelihood_threshold:
        Minimum DLC likelihood required for both ears.

    Returns
    -------
    int
        Zero-based row position (usable with ``df.iloc``) of the first qualifying
        frame.

    Raises
    ------
    KeyError
        If the expected ear columns are missing.
    ValueError
        If no frame satisfies the criteria.
    """
    required = [
        f"{left_ear}_x",
        f"{left_ear}_y",
        f"{left_ear}_likelihood",
        f"{right_ear}_x",
        f"{right_ear}_y",
        f"{right_ear}_likelihood",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected ear columns: {missing}")

    valid = (
        df[f"{left_ear}_likelihood"].to_numpy() > likelihood_threshold
    ) & (df[f"{right_ear}_likelihood"].to_numpy() > likelihood_threshold)
    coords = df[
        [f"{left_ear}_x", f"{left_ear}_y", f"{right_ear}_x", f"{right_ear}_y"]
    ].to_numpy()
    valid &= ~np.isnan(coords).any(axis=1)

    positions = np.flatnonzero(valid)
    if positions.size == 0:
        raise ValueError(
            "No frame found where both ears exceed the likelihood threshold."
        )
    return int(positions[0])


def heading_offset_from_ears(
    df: pd.DataFrame,
    left_ear: str = "left_ear",
    right_ear: str = "right_ear",
    likelihood_threshold: float = 0.0,
    forward_sign: int = 1,
) -> float:
    """Calibrate the heading offset (degrees) from the ear keypoints.

    Using the first frame where both ears are tracked, the animal's facing
    direction is taken as the vector orthogonal to the inter-aural
    (left-ear -> right-ear) axis. The returned offset is the angle of that facing
    vector in the standard math quadrant (0 deg = right/+x, 90 deg = up,
    counter-clockwise positive), so that adding it to a relative commutator
    heading anchors "facing right" to 0 degrees.

    Image (pixel) coordinates have the y-axis pointing down; this is converted to
    the standard y-up convention before computing the angle.

    .. note::
       There are two directions orthogonal to the ear axis (nose vs. tail). This
       function picks one by convention (*forward_sign* ``+1`` = rotate the
       left->right ear vector 90 deg counter-clockwise). If the resulting heading
       is 180 deg out, pass ``forward_sign=-1``.

    Parameters
    ----------
    df:
        Flattened pose DataFrame (see :func:`first_valid_ear_frame`).
    left_ear, right_ear:
        Body-part names of the two ears.
    likelihood_threshold:
        Minimum DLC likelihood required for both ears at the calibration frame.
    forward_sign:
        ``+1`` or ``-1``; selects which of the two orthogonal directions is
        treated as "forward" (see note).

    Returns
    -------
    float
        The facing-direction angle in degrees, in ``[0, 360)``.
    """
    pos = first_valid_ear_frame(
        df, left_ear, right_ear, likelihood_threshold=likelihood_threshold
    )
    row = df.iloc[pos]
    lx, ly = float(row[f"{left_ear}_x"]), float(row[f"{left_ear}_y"])
    rx, ry = float(row[f"{right_ear}_x"]), float(row[f"{right_ear}_y"])

    # Inter-aural vector (left -> right) in standard y-up coordinates.
    ear_dx = rx - lx
    ear_dy = -(ry - ly)  # flip image y (down) to standard y (up)

    # Forward = ear vector rotated 90 deg CCW: (x, y) -> (-y, x), with an optional
    # sign flip to choose the nose-ward orthogonal.
    fx = forward_sign * (-ear_dy)
    fy = forward_sign * ear_dx

    return float(np.degrees(np.arctan2(fy, fx)) % 360.0)


# ---------------------------------------------------------------------------
# Commutator loading from S3
# ---------------------------------------------------------------------------
def load_commutator_turns(
    s3_behavior_uri: str,
    timestamps: list[str],
    register: str = DEFAULT_COMMUTATOR_REGISTER,
    s3_client: "S3Client | None" = None,
    anonymous: bool = True,
) -> pd.DataFrame:
    """Load and concatenate commutator accumulated-turns CSVs from S3.

    Parameters
    ----------
    s3_behavior_uri:
        S3 URI of the session's ``behavior`` directory, e.g.
        ``"s3://aind-open-data/854393_2026-06-09_19-34-26/behavior"``. The
        register files are read from ``<behavior>/Commutator/``.
    timestamps:
        Timestamp tokens (e.g. ``["2026-06-11T03-00-00", ...]``) identifying
        which hourly files to load, typically the tokens of the pose files.
    register:
        Commutator register filename prefix (default
        ``"Commutator_AccumulatedCommutatorTurns"``).
    s3_client:
        Optional pre-built S3 client. When ``None``, one is created via
        :func:`pirouette_data.ingestion.get_s3_client`.
    anonymous:
        Passed to :func:`get_s3_client` when *s3_client* is ``None``.

    Returns
    -------
    pandas.DataFrame
        Columns ``harp_time`` (Harp seconds) and ``turns`` (accumulated turns),
        sorted by ``harp_time`` with duplicate timestamps dropped.
    """
    if s3_client is None:
        s3_client = get_s3_client(anonymous=anonymous)
    bucket, prefix = _parse_s3_uri(s3_behavior_uri)

    frames: list[pd.DataFrame] = []
    for ts in timestamps:
        key = f"{prefix.rstrip('/')}/Commutator/{register}_{ts}.csv"
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        part = pd.read_csv(io.BytesIO(obj["Body"].read()))
        frames.append(part.rename(columns={"Seconds": "harp_time", "Value": "turns"}))

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined[["harp_time", "turns"]]
        .drop_duplicates(subset="harp_time")
        .sort_values("harp_time", kind="stable")
        .reset_index(drop=True)
    )
    return combined


# ---------------------------------------------------------------------------
# Append heading to the pose dataframe
# ---------------------------------------------------------------------------
def append_commutator_heading(
    df: pd.DataFrame,
    s3_behavior_uri: str,
    register: str = DEFAULT_COMMUTATOR_REGISTER,
    offset_deg: float | None = None,
    direction: int = 1,
    left_ear: str = "left_ear",
    right_ear: str = "right_ear",
    likelihood_threshold: float = 0.0,
    forward_sign: int = 1,
    anonymous: bool = True,
    column: str = "commutator_heading_deg",
) -> pd.DataFrame:
    """Append a commutator-derived heading column aligned to the camera stream.

    The commutator accumulated-turns stream is loaded from S3, converted to a
    heading in ``[0, 360)`` via :func:`commutator_heading_estimate`, and then
    interpolated (``interp1d``) onto the pose/camera timestamps in
    ``df["harp_time"]``.

    The reference orientation is the first commutator sample within the pose time
    window (``harp_time >= df["harp_time"].min()``). When *offset_deg* is
    ``None``, it is calibrated from the ear keypoints via
    :func:`heading_offset_from_ears`.

    .. note::
       Heading is circular, so linear interpolation across the 360 -> 0 wrap
       introduces brief spikes at each crossing (interpolation is performed on
       the already-wrapped signal, as requested).

    Parameters
    ----------
    df:
        Pose DataFrame from :func:`pirouette_data.ingestion.build_dataset`; must
        contain ``harp_time`` and ``source_file`` columns plus the ear keypoints.
    s3_behavior_uri:
        S3 URI of the session's ``behavior`` directory (see
        :func:`load_commutator_turns`).
    register:
        Commutator register filename prefix.
    offset_deg:
        Heading offset in degrees. When ``None``, it is calibrated from the ears.
    direction:
        Commutator rotation sign (``+1`` or ``-1``); see
        :func:`commutator_heading_estimate`.
    left_ear, right_ear, likelihood_threshold, forward_sign:
        Passed to :func:`heading_offset_from_ears` when calibrating the offset.
    anonymous:
        When ``True`` (default), S3 is accessed with unsigned requests.
    column:
        Name of the appended heading column.

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with the heading column added (degrees, ``[0, 360)``).
    """
    if "harp_time" not in df.columns:
        raise KeyError("df must contain a 'harp_time' column.")
    if "source_file" not in df.columns:
        raise KeyError("df must contain a 'source_file' column.")

    # Timestamp tokens of the pose files -> matching commutator hour files.
    timestamps = sorted(
        {parse_camera_and_timestamp(name)[1] for name in df["source_file"].unique()}
    )

    s3_client = get_s3_client(anonymous=anonymous)
    commutator = load_commutator_turns(
        s3_behavior_uri,
        timestamps,
        register=register,
        s3_client=s3_client,
        anonymous=anonymous,
    )

    comm_time = commutator["harp_time"].to_numpy()
    comm_turns = commutator["turns"].to_numpy()

    # Reference = first commutator sample within the pose time window.
    pose_start = float(df["harp_time"].min())
    ref_idx = int(np.searchsorted(comm_time, pose_start, side="left"))
    ref_idx = min(ref_idx, len(comm_turns) - 1)
    reference_value = float(comm_turns[ref_idx])

    # Offset calibrated from the ears unless supplied.
    if offset_deg is None:
        offset_deg = heading_offset_from_ears(
            df,
            left_ear=left_ear,
            right_ear=right_ear,
            likelihood_threshold=likelihood_threshold,
            forward_sign=forward_sign,
        )

    # Heading on the native commutator timeline (wrap), then interpolate.
    comm_heading = commutator_heading_estimate(
        comm_turns,
        reference_value=reference_value,
        offset_deg=offset_deg,
        direction=direction,
    )

    interpolator = interp1d(
        comm_time,
        comm_heading,
        kind="linear",
        bounds_error=False,
        fill_value=(comm_heading[0], comm_heading[-1]),
        assume_sorted=True,
    )

    out = df.copy()
    out[column] = interpolator(out["harp_time"].to_numpy())
    return out
