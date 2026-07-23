"""Unit tests for :mod:`pirouette_data.ingestion` (pure, no network)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pirouette_data import ingestion


# ---------------------------------------------------------------------------
# parse_camera_and_timestamp
# ---------------------------------------------------------------------------
def test_parse_camera_and_timestamp_basic():
    camera, timestamp, dt = ingestion.parse_camera_and_timestamp(
        "TopCamera_2026-06-11T03-00-00.h5"
    )
    assert camera == "TopCamera"
    assert timestamp == "2026-06-11T03-00-00"
    assert dt == datetime(2026, 6, 11, 3, 0, 0)


def test_parse_camera_and_timestamp_multiword_camera():
    # Register-style names (underscores in the "camera" part) still parse: only
    # the trailing <timestamp> block is split off.
    camera, timestamp, _ = ingestion.parse_camera_and_timestamp(
        "Commutator_AccumulatedCommutatorTurns_2026-06-09T19-00-00.csv"
    )
    assert camera == "Commutator_AccumulatedCommutatorTurns"
    assert timestamp == "2026-06-09T19-00-00"


def test_parse_camera_and_timestamp_accepts_path():
    camera, _, _ = ingestion.parse_camera_and_timestamp(
        "/some/dir/TopCamera_2026-06-11T03-00-00.h5"
    )
    assert camera == "TopCamera"


def test_parse_camera_and_timestamp_invalid():
    with pytest.raises(ValueError):
        ingestion.parse_camera_and_timestamp("not_a_valid_name.h5")


# ---------------------------------------------------------------------------
# _parse_s3_uri
# ---------------------------------------------------------------------------
def test_parse_s3_uri():
    bucket, prefix = ingestion._parse_s3_uri(
        "s3://aind-open-data/854393_2026-06-09_19-34-26/behavior-videos"
    )
    assert bucket == "aind-open-data"
    assert prefix == "854393_2026-06-09_19-34-26/behavior-videos"


def test_parse_s3_uri_strips_slashes():
    bucket, prefix = ingestion._parse_s3_uri("s3://bucket/a/b/")
    assert bucket == "bucket"
    assert prefix == "a/b"


def test_parse_s3_uri_invalid():
    with pytest.raises(ValueError):
        ingestion._parse_s3_uri("https://example.com/foo")


# ---------------------------------------------------------------------------
# flatten_pose_columns
# ---------------------------------------------------------------------------
def _make_dlc_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            ("scorerX", "left_ear", "x"),
            ("scorerX", "left_ear", "y"),
            ("scorerX", "left_ear", "likelihood"),
            ("scorerX", "right_ear", "x"),
            ("scorerX", "right_ear", "y"),
            ("scorerX", "right_ear", "likelihood"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.arange(12, dtype="float64").reshape(2, 6)
    return pd.DataFrame(data, columns=columns)


def test_flatten_pose_columns():
    flat = ingestion.flatten_pose_columns(_make_dlc_frame())
    assert list(flat.columns) == [
        "left_ear_x",
        "left_ear_y",
        "left_ear_likelihood",
        "right_ear_x",
        "right_ear_y",
        "right_ear_likelihood",
    ]
    # values unchanged
    assert flat["right_ear_x"].tolist() == [3.0, 9.0]


def test_flatten_pose_columns_idempotent_on_flat():
    flat = ingestion.flatten_pose_columns(_make_dlc_frame())
    again = ingestion.flatten_pose_columns(flat)
    assert list(again.columns) == list(flat.columns)


# ---------------------------------------------------------------------------
# harp_to_datetime
# ---------------------------------------------------------------------------
def test_harp_to_datetime_is_tz_aware_pacific():
    # 1904-01-01 UTC epoch + 0 s -> that instant in Pacific time.
    out = ingestion.harp_to_datetime([0.0], tz="America/Los_Angeles")
    assert str(out.dt.tz) == "America/Los_Angeles"
    # 1904-01-01 00:00 UTC == 1903-12-31 16:00 LMT/PST-ish; just assert year.
    assert out.iloc[0].year in (1903, 1904)


def test_harp_to_datetime_roundtrip_utc():
    # A known Harp second should map to a specific UTC wall-clock.
    seconds = 3863878502.681792
    out = ingestion.harp_to_datetime(pd.Series([seconds]), tz="UTC")
    # 3863878502.68 s after 1904-01-01 UTC -> 2026-06-09 19:35:02 UTC
    ts = out.iloc[0]
    assert ts.year == 2026 and ts.month == 6 and ts.day == 9
    assert ts.hour == 19


def test_harp_to_datetime_preserves_index():
    s = pd.Series([0.0, 1.0], index=[10, 20])
    out = ingestion.harp_to_datetime(s)
    assert list(out.index) == [10, 20]
