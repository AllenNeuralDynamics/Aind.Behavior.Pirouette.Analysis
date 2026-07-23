"""Unit tests for :mod:`pirouette_data.processing` (pure, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirouette_data import processing


def _chamber_df(
    length_px, width_px, n=5, like=1.0, jitter=0.0, rng=None, ox=0.0, oy=0.0
):
    """Axis-aligned chamber with upper-left at (ox, oy), plus a centre keypoint.

    Image coordinates (y down): upper edge at y=oy, lower edge at y=oy+width_px.
    """
    df = pd.DataFrame(
        {
            "ul_champber_x": np.full(n, ox),
            "ul_champber_y": np.full(n, oy),
            "ul_champber_likelihood": np.full(n, like),
            "ur_champber_x": np.full(n, ox + length_px),
            "ur_champber_y": np.full(n, oy),
            "ur_champber_likelihood": np.full(n, like),
            "lr_chamber_x": np.full(n, ox + length_px),
            "lr_chamber_y": np.full(n, oy + width_px),
            "lr_chamber_likelihood": np.full(n, like),
            "ll_chamber_x": np.full(n, ox),
            "ll_chamber_y": np.full(n, oy + width_px),
            "ll_chamber_likelihood": np.full(n, like),
            # a tracked keypoint at the chamber centre
            "nose_x": np.full(n, ox + length_px / 2),
            "nose_y": np.full(n, oy + width_px / 2),
            "nose_likelihood": np.full(n, like),
        }
    )
    if jitter and rng is not None:
        for c in df.columns:
            if c.endswith(("_x", "_y")):
                df[c] = df[c] + rng.normal(scale=jitter, size=n)
    return df


# ---------------------------------------------------------------------------
# estimate_chamber_scale
# ---------------------------------------------------------------------------
def test_scale_basic():
    df = _chamber_df(length_px=746.0, width_px=388.0)  # exactly 2 px per mm
    scale = processing.estimate_chamber_scale(df)
    assert scale.length_px == pytest.approx(746.0)
    assert scale.width_px == pytest.approx(388.0)
    assert scale.mm_per_px_x == pytest.approx(processing.CHAMBER_LENGTH_MM / 746.0)
    assert scale.mm_per_px_y == pytest.approx(processing.CHAMBER_WIDTH_MM / 388.0)
    # 2 px/mm -> 0.5 mm/px
    assert scale.mm_per_px_x == pytest.approx(0.5)
    assert scale.mm_per_px_y == pytest.approx(0.5)


def test_scale_median_robust_to_outliers():
    rng = np.random.default_rng(0)
    df = _chamber_df(length_px=746.0, width_px=388.0, n=999, jitter=1.0, rng=rng)
    # inject a few gross outliers
    df.loc[0, "ur_champber_x"] = 5000.0
    df.loc[1, "ll_chamber_y"] = -4000.0
    scale = processing.estimate_chamber_scale(df)
    assert scale.length_px == pytest.approx(746.0, abs=1.0)
    assert scale.width_px == pytest.approx(388.0, abs=1.0)


def test_scale_likelihood_filtering():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=4)
    # Corrupt one frame but mark it low-likelihood so it is excluded.
    df.loc[0, ["ur_champber_x"]] = 10000.0
    df.loc[0, ["ur_champber_likelihood"]] = 0.1
    scale = processing.estimate_chamber_scale(df, likelihood_threshold=0.5)
    assert scale.length_px == pytest.approx(746.0)


def test_scale_missing_columns():
    df = pd.DataFrame({"ul_champber_x": [0.0]})
    with pytest.raises(KeyError):
        processing.estimate_chamber_scale(df)


def test_scale_all_nan_raises():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=3)
    for c in df.columns:
        if c.endswith(("_x", "_y")):
            df[c] = np.nan
    with pytest.raises(ValueError):
        processing.estimate_chamber_scale(df)


# ---------------------------------------------------------------------------
# keypoint_columns
# ---------------------------------------------------------------------------
def test_keypoint_columns():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=2)
    bases = processing.keypoint_columns(df)
    assert "nose" in bases
    assert "ul_champber" in bases
    # likelihood-only / non-coordinate columns are excluded
    assert all(not b.endswith("_likelihood") for b in bases)


# ---------------------------------------------------------------------------
# append_mm_columns
# ---------------------------------------------------------------------------
def test_append_mm_columns_values():
    # Chamber offset from image origin to exercise origin subtraction.
    df = _chamber_df(length_px=746.0, width_px=388.0, n=3, ox=100.0, oy=50.0)
    out = processing.append_mm_columns(df)
    # Upper-left corner is the origin -> (0, 0) mm.
    assert out["ul_champber_x_mm"].iloc[0] == pytest.approx(0.0)
    assert out["ul_champber_y_mm"].iloc[0] == pytest.approx(0.0)
    # ur -> (length_mm, 0); ll -> (0, width_mm).
    assert out["ur_champber_x_mm"].iloc[0] == pytest.approx(processing.CHAMBER_LENGTH_MM)
    assert out["ur_champber_y_mm"].iloc[0] == pytest.approx(0.0)
    assert out["ll_chamber_y_mm"].iloc[0] == pytest.approx(processing.CHAMBER_WIDTH_MM)
    # nose at chamber centre -> (length_mm/2, width_mm/2).
    assert out["nose_x_mm"].iloc[0] == pytest.approx(processing.CHAMBER_LENGTH_MM / 2)
    assert out["nose_y_mm"].iloc[0] == pytest.approx(processing.CHAMBER_WIDTH_MM / 2)


def test_scale_reports_origin():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=4, ox=100.0, oy=50.0)
    scale = processing.estimate_chamber_scale(df)
    assert scale.origin_px_x == pytest.approx(100.0)
    assert scale.origin_px_y == pytest.approx(50.0)


def test_append_mm_columns_does_not_mutate_input():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=2)
    processing.append_mm_columns(df)
    assert "nose_x_mm" not in df.columns


def test_append_mm_columns_accepts_precomputed_scale():
    df = _chamber_df(length_px=746.0, width_px=388.0, n=2)
    scale = processing.ChamberScale(
        length_px=746.0,
        width_px=388.0,
        mm_per_px_x=2.0,
        mm_per_px_y=3.0,
        origin_px_x=0.0,
        origin_px_y=0.0,
    )
    out = processing.append_mm_columns(df, scale=scale)
    assert out["nose_x_mm"].iloc[0] == pytest.approx((746.0 / 2) * 2.0)
    assert out["nose_y_mm"].iloc[0] == pytest.approx((388.0 / 2) * 3.0)
