"""Ingestion utilities for pose-tracking data.

This module loads DeepLabCut (DLC) pose ``.h5`` files, concatenates them into a
single :class:`pandas.DataFrame` ordered in time, and augments every frame with
Harp timing information pulled from the matching camera ``.csv`` files stored in
an AWS S3 open-data bucket.

The camera frame-index CSVs (e.g. ``TopCamera_2026-06-11T03-00-00.csv``) contain
one row per video frame with a ``Seconds`` column holding the Harp timestamp in
fractional seconds. Each CSV aligns frame-for-frame with the DLC ``.h5`` file of
the same name, so timing is joined positionally.

Three timing columns are appended to the combined DataFrame:

* ``harp_time`` — the raw Harp timestamp in seconds (from the CSV ``Seconds``).
* ``time_since_start`` — ``harp_time`` minus the experiment start time (the first
  Harp timestamp in the earliest CSV of the camera directory).
* ``datetime_pacific`` — the Harp timestamp converted to a timezone-aware Pacific
  wall-clock datetime (DST-aware; PST in winter, PDT in summer).

The Harp-to-datetime conversion is delegated to the Aeon API
(:func:`swc.aeon.io.api.to_datetime`), which decodes Harp seconds relative to the
1904-01-01 UTC reference epoch.
"""

from __future__ import annotations

import io
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from swc.aeon.io import api as aeon_api

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

# Default timezone for the ``datetime_pacific`` column. ``America/Los_Angeles``
# is used rather than a fixed UTC offset so daylight-saving transitions are
# handled automatically (PST = UTC-8, PDT = UTC-7).
DEFAULT_TIMEZONE = "America/Los_Angeles"

# Filename pattern: ``<Camera>_<YYYY-MM-DDTHH-MM-SS>`` (e.g.
# ``TopCamera_2026-06-11T03-00-00``). The camera name may itself contain no
# trailing ``_<timestamp>`` block other than this one.
_FILENAME_RE = re.compile(
    r"^(?P<camera>.+)_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$"
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def parse_camera_and_timestamp(path: str | Path) -> tuple[str, str, datetime]:
    """Parse the camera name and acquisition datetime from a pose filename.

    Parameters
    ----------
    path:
        Path to (or bare name of) a pose file such as
        ``TopCamera_2026-06-11T03-00-00.h5``.

    Returns
    -------
    camera : str
        The camera name prefix (e.g. ``"TopCamera"``).
    timestamp : str
        The raw timestamp token from the filename (e.g.
        ``"2026-06-11T03-00-00"``). This is reused verbatim to locate the
        matching CSV on S3.
    dt : datetime.datetime
        The parsed (timezone-naive) acquisition datetime.

    Raises
    ------
    ValueError
        If *path* does not match the expected ``<Camera>_<timestamp>`` pattern.
    """
    stem = Path(path).stem
    match = _FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(
            f"Filename '{stem}' does not match expected pattern "
            "'<Camera>_<YYYY-MM-DDTHH-MM-SS>'."
        )
    camera = match.group("camera")
    timestamp = match.group("timestamp")
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%S")
    return camera, timestamp, dt


# ---------------------------------------------------------------------------
# Pose (.h5) loading
# ---------------------------------------------------------------------------
def load_pose_h5(path: str | Path, flatten: bool = True) -> pd.DataFrame:
    """Load a single DeepLabCut pose ``.h5`` file.

    Parameters
    ----------
    path:
        Path to a DLC ``.h5`` file (pandas/PyTables format).
    flatten:
        When ``True`` (default), drop the DLC ``scorer`` column level and
        flatten the remaining ``(bodypart, coord)`` MultiIndex into single-level
        ``"<bodypart>_<coord>"`` columns (e.g. ``"nose_x"``). When ``False``,
        the original ``scorer/bodyparts/coords`` column MultiIndex is preserved.

    Returns
    -------
    pandas.DataFrame
        Pose data with one row per video frame. The index is named ``"frame"``.
    """
    df = pd.read_hdf(path)
    df.index.name = "frame"
    if flatten:
        df = flatten_pose_columns(df)
    return df


def flatten_pose_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a DLC pose column MultiIndex to ``"<bodypart>_<coord>"`` names.

    Parameters
    ----------
    df:
        Pose DataFrame with a ``scorer/bodyparts/coords`` column MultiIndex, as
        returned by :func:`load_pose_h5` with ``flatten=False``.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with single-level columns ``"<bodypart>_<coord>"`` (the
        ``scorer`` level is dropped).
    """
    flat = df.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        if "scorer" in (flat.columns.names or []):
            flat = flat.droplevel("scorer", axis=1)
        flat.columns = [f"{bodypart}_{coord}" for bodypart, coord in flat.columns]
    return flat


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/prefix`` URI into ``(bucket, prefix)``.

    Parameters
    ----------
    uri:
        An ``s3://`` URI.

    Returns
    -------
    bucket : str
        The bucket name.
    prefix : str
        The key prefix, with any leading/trailing slashes stripped.

    Raises
    ------
    ValueError
        If *uri* does not start with ``s3://``.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI (must start with 's3://'): {uri!r}")
    without_scheme = uri[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.strip("/")


def get_s3_client(anonymous: bool = True) -> "S3Client":
    """Create a boto3 S3 client.

    Parameters
    ----------
    anonymous:
        When ``True`` (default), the client issues unsigned requests, which is
        required for anonymous access to public open-data buckets such as
        ``aind-open-data``. When ``False``, the default credential chain is used.

    Returns
    -------
    botocore.client.S3
        A configured S3 client.
    """
    import boto3

    if anonymous:
        from botocore import UNSIGNED
        from botocore.config import Config

        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def _camera_prefix(s3_video_uri: str, camera: str) -> str:
    """Return the S3 key prefix for a camera's frame-index directory."""
    _, prefix = _parse_s3_uri(s3_video_uri)
    return f"{prefix.rstrip('/')}/{camera}/"


def list_camera_csv_keys(
    s3_video_uri: str,
    camera: str,
    s3_client: "S3Client | None" = None,
    anonymous: bool = True,
) -> list[str]:
    """List all frame-index CSV keys for one camera, sorted by timestamp.

    Parameters
    ----------
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory that contains per-camera
        sub-directories (e.g.
        ``"s3://aind-open-data/854393_2026-06-09_19-34-26/behavior-videos"``).
    camera:
        Camera name whose sub-directory is searched (e.g. ``"TopCamera"``).
    s3_client:
        Optional pre-built S3 client. When ``None``, one is created via
        :func:`get_s3_client`.
    anonymous:
        Passed to :func:`get_s3_client` when *s3_client* is ``None``.

    Returns
    -------
    list[str]
        Fully-qualified S3 keys of the camera's ``.csv`` files, sorted
        chronologically by the timestamp embedded in each filename.
    """
    if s3_client is None:
        s3_client = get_s3_client(anonymous=anonymous)
    bucket, _ = _parse_s3_uri(s3_video_uri)
    prefix = _camera_prefix(s3_video_uri, camera)

    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"])

    def _sort_key(key: str) -> str:
        try:
            return parse_camera_and_timestamp(Path(key).name)[1]
        except ValueError:
            return Path(key).name

    return sorted(keys, key=_sort_key)


def load_harp_seconds(
    s3_video_uri: str,
    camera: str,
    timestamp: str,
    s3_client: "S3Client | None" = None,
    anonymous: bool = True,
    nrows: int | None = None,
) -> np.ndarray:
    """Load the ``Seconds`` (Harp time) column from a camera CSV on S3.

    Parameters
    ----------
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory (see
        :func:`list_camera_csv_keys`).
    camera:
        Camera name (e.g. ``"TopCamera"``).
    timestamp:
        Timestamp token identifying the file (e.g. ``"2026-06-11T03-00-00"``).
    s3_client:
        Optional pre-built S3 client. When ``None``, one is created via
        :func:`get_s3_client`.
    anonymous:
        Passed to :func:`get_s3_client` when *s3_client* is ``None``.
    nrows:
        Optional cap on the number of rows to read (useful for peeking at the
        first timestamp only).

    Returns
    -------
    numpy.ndarray
        1-D array of Harp timestamps in fractional seconds, one per frame.
    """
    if s3_client is None:
        s3_client = get_s3_client(anonymous=anonymous)
    bucket, _ = _parse_s3_uri(s3_video_uri)
    key = f"{_camera_prefix(s3_video_uri, camera)}{camera}_{timestamp}.csv"

    obj = s3_client.get_object(Bucket=bucket, Key=key)
    frame = pd.read_csv(
        io.BytesIO(obj["Body"].read()), usecols=["Seconds"], nrows=nrows
    )
    return frame["Seconds"].to_numpy()


def get_experiment_start_harp(
    s3_video_uri: str,
    camera: str,
    s3_client: "S3Client | None" = None,
    anonymous: bool = True,
) -> float:
    """Return the Harp time (s) of the first frame of the experiment.

    The experiment start is defined as the first ``Seconds`` value in the
    earliest CSV (by filename timestamp) in the camera's S3 directory.

    Parameters
    ----------
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory.
    camera:
        Camera name (e.g. ``"TopCamera"``).
    s3_client:
        Optional pre-built S3 client. When ``None``, one is created via
        :func:`get_s3_client`.
    anonymous:
        Passed to :func:`get_s3_client` when *s3_client* is ``None``.

    Returns
    -------
    float
        The first Harp timestamp of the earliest CSV, in fractional seconds.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found for the camera.
    """
    if s3_client is None:
        s3_client = get_s3_client(anonymous=anonymous)
    keys = list_camera_csv_keys(
        s3_video_uri, camera, s3_client=s3_client, anonymous=anonymous
    )
    if not keys:
        raise FileNotFoundError(
            f"No CSV files found for camera '{camera}' under {s3_video_uri!r}."
        )
    _, earliest_timestamp, _ = parse_camera_and_timestamp(Path(keys[0]).name)
    first_seconds = load_harp_seconds(
        s3_video_uri,
        camera,
        earliest_timestamp,
        s3_client=s3_client,
        anonymous=anonymous,
        nrows=1,
    )
    return float(first_seconds[0])


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------
def harp_to_datetime(
    harp_seconds: np.ndarray | pd.Series, tz: str = DEFAULT_TIMEZONE
) -> pd.Series:
    """Convert Harp seconds to timezone-aware wall-clock datetimes.

    The Aeon API decodes Harp seconds to a UTC datetime (1904-01-01 reference
    epoch); the result is then converted to the requested timezone.

    Parameters
    ----------
    harp_seconds:
        Harp timestamps in fractional seconds.
    tz:
        IANA timezone name for the output (default ``"America/Los_Angeles"``).

    Returns
    -------
    pandas.Series
        Timezone-aware ``datetime64[ns, tz]`` Series aligned to the input.
    """
    index = harp_seconds.index if isinstance(harp_seconds, pd.Series) else None
    seconds = pd.Series(np.asarray(harp_seconds, dtype="float64"), index=index)
    utc = aeon_api.to_datetime(seconds)
    # ``to_datetime`` returns UTC-aware values; convert to the target timezone.
    return utc.dt.tz_convert(tz)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_dataset(
    pose_dir: str | Path,
    s3_video_uri: str,
    tz: str = DEFAULT_TIMEZONE,
    anonymous: bool = True,
    order_by: str = "harp",
    experiment_start_harp: float | None = None,
    flatten: bool = True,
) -> pd.DataFrame:
    """Build a single time-ordered pose DataFrame with Harp timing columns.

    For every ``.h5`` file in *pose_dir* this:

    1. loads the DLC pose data,
    2. reads the matching camera CSV from S3 and aligns its ``Seconds`` (Harp
       time) column frame-for-frame,
    3. concatenates all files into one DataFrame ordered in time, and
    4. appends ``harp_time``, ``time_since_start`` and ``datetime_pacific``.

    The camera name used to locate the CSVs on S3 is derived from each pose
    filename (the ``<Camera>_<timestamp>`` prefix).

    Parameters
    ----------
    pose_dir:
        Local directory containing DLC pose ``.h5`` files.
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory holding per-camera CSV
        sub-directories, e.g.
        ``"s3://aind-open-data/854393_2026-06-09_19-34-26/behavior-videos"``.
    tz:
        IANA timezone for the ``datetime_pacific`` column (default
        ``"America/Los_Angeles"``).
    anonymous:
        When ``True`` (default), S3 is accessed with unsigned requests (required
        for public open-data buckets).
    order_by:
        ``"harp"`` (default) orders the concatenated frames by their Harp
        timestamp; ``"filename"`` preserves the chronological filename order
        without a global sort.
    experiment_start_harp:
        Optional override for the experiment start time (Harp seconds) used to
        compute ``time_since_start``. When ``None``, it is read from the earliest
        CSV in the camera's S3 directory via :func:`get_experiment_start_harp`.
    flatten:
        Passed to :func:`load_pose_h5`; flattens the DLC column MultiIndex when
        ``True`` (default).

    Returns
    -------
    pandas.DataFrame
        Concatenated pose data with these appended columns:

        * ``source_file`` — stem of the originating ``.h5`` file.
        * ``frame`` — original per-file, zero-based frame index.
        * ``harp_time`` — raw Harp timestamp in seconds.
        * ``time_since_start`` — ``harp_time`` minus the experiment start.
        * ``datetime_pacific`` — timezone-aware Pacific datetime.

        The index is reset to a clean ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If *pose_dir* contains no ``.h5`` files.
    ValueError
        If *order_by* is not ``"harp"`` or ``"filename"``.
    """
    if order_by not in ("harp", "filename"):
        raise ValueError("order_by must be 'harp' or 'filename'.")

    pose_dir = Path(pose_dir)
    pose_files = sorted(pose_dir.glob("*.h5"))
    if not pose_files:
        raise FileNotFoundError(f"No .h5 files found in {pose_dir}.")

    s3_client = get_s3_client(anonymous=anonymous)

    # Resolve camera(s) from the pose filenames and the experiment start time.
    cameras = {parse_camera_and_timestamp(p.name)[0] for p in pose_files}
    if experiment_start_harp is None:
        # Use the earliest camera (alphabetical) as the experiment reference when
        # more than one is present; typically there is exactly one.
        ref_camera = sorted(cameras)[0]
        experiment_start_harp = get_experiment_start_harp(
            s3_video_uri, ref_camera, s3_client=s3_client, anonymous=anonymous
        )

    per_file: list[pd.DataFrame] = []
    for path in pose_files:
        camera, timestamp, _ = parse_camera_and_timestamp(path.name)
        pose = load_pose_h5(path, flatten=flatten)

        harp = load_harp_seconds(
            s3_video_uri, camera, timestamp, s3_client=s3_client, anonymous=anonymous
        )

        n = len(pose)
        if len(harp) != n:
            warnings.warn(
                f"{path.name}: {n} pose frames but {len(harp)} CSV timestamps; "
                "aligning to the shorter length.",
                RuntimeWarning,
            )
            m = min(n, len(harp))
            pose = pose.iloc[:m]
            harp = harp[:m]

        pose = pose.reset_index()  # 'frame' becomes a column
        pose.insert(0, "source_file", path.stem)
        pose["harp_time"] = harp
        per_file.append(pose)

    combined = pd.concat(per_file, ignore_index=True)

    if order_by == "harp":
        combined = combined.sort_values("harp_time", kind="stable").reset_index(
            drop=True
        )

    combined["time_since_start"] = combined["harp_time"] - experiment_start_harp
    combined["datetime_pacific"] = harp_to_datetime(combined["harp_time"], tz=tz)

    return combined
