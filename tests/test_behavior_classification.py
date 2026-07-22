"""Unit tests for :mod:`pirouette_data.behavior_classification` (pure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirouette_data import behavior_classification as bc


# ---------------------------------------------------------------------------
# threshold estimation
# ---------------------------------------------------------------------------
def test_otsu_separates_bimodal():
    rng = np.random.default_rng(0)
    low = rng.normal(1.0, 0.2, 5000).clip(0)
    high = rng.normal(100.0, 5.0, 5000)
    thr = bc.otsu_threshold(np.concatenate([low, high]))
    assert 1.0 < thr < 100.0


def test_otsu_ignores_nan():
    v = np.array([1.0, 1.0, np.nan, 100.0, 100.0])
    thr = bc.otsu_threshold(v)
    assert np.isfinite(thr)


def test_otsu_all_nan_raises():
    with pytest.raises(ValueError):
        bc.otsu_threshold(np.array([np.nan, np.nan]))


def test_estimate_velocity_threshold_bad_method():
    with pytest.raises(ValueError):
        bc.estimate_velocity_threshold(np.array([1.0, 2.0]), method="kmeans")


# ---------------------------------------------------------------------------
# classification with a fixed threshold
# ---------------------------------------------------------------------------
def test_fixed_threshold_labels():
    speed = np.array([0.0, 10.0, 60.0, 5.0, 200.0])
    labels = bc.classify_rest_movement(
        speed, threshold=50.0, median_filter_size=1, use_abs=False
    )
    assert list(labels) == ["rest", "rest", "movement", "rest", "movement"]


def test_use_abs_treats_backward_as_movement():
    speed = np.array([-100.0, 0.0, -100.0])
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=1)
    assert list(labels) == ["movement", "rest", "movement"]


def test_use_abs_false_keeps_sign():
    speed = np.array([-100.0, 0.0, 100.0])
    labels = bc.classify_rest_movement(
        speed, threshold=50.0, median_filter_size=1, use_abs=False
    )
    assert list(labels) == ["rest", "rest", "movement"]


# ---------------------------------------------------------------------------
# median filtering of short bouts
# ---------------------------------------------------------------------------
def test_median_filter_removes_short_movement_bout():
    # A single movement frame amid rest -> becomes rest.
    speed = np.zeros(41)
    speed[20] = 100.0
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=5)
    assert set(labels) == {"rest"}


def test_median_filter_removes_short_rest_bout():
    # A single rest frame amid movement -> becomes movement.
    speed = np.full(41, 100.0)
    speed[20] = 0.0
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=5)
    assert set(labels) == {"movement"}


def test_median_filter_preserves_long_bout():
    speed = np.concatenate([np.zeros(20), np.full(50, 100.0), np.zeros(20)])
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=5)
    n_move = int((labels == "movement").sum())
    assert n_move == pytest.approx(50, abs=2)  # bout preserved (edges within filter)


def test_median_filter_size_coerced_to_odd():
    speed = np.zeros(21)
    speed[10] = 100.0
    # even size should still work (coerced to odd internally)
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=4)
    assert set(labels) == {"rest"}


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------
def test_nonfinite_labelled_rest():
    speed = np.array([100.0, np.nan, 100.0, 100.0, 100.0])
    labels = bc.classify_rest_movement(speed, threshold=50.0, median_filter_size=1)
    assert labels[1] == "rest"


# ---------------------------------------------------------------------------
# append_behavior_labels
# ---------------------------------------------------------------------------
def test_append_behavior_labels_adds_column_and_threshold():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(1, 0.2, 500).clip(0), rng.normal(100, 5, 500)])
    df = pd.DataFrame({"ear_velocity_smooth_mm_s": v})
    out = bc.append_behavior_labels(df, median_filter_size=1)
    assert "behavior" in out.columns
    assert "behavior" not in df.columns
    assert set(out["behavior"].unique()) <= {"rest", "movement"}
    assert "behavior_velocity_threshold" in out.attrs
    assert 1.0 < out.attrs["behavior_velocity_threshold"] < 100.0


def test_append_behavior_labels_fixed_threshold_counts():
    df = pd.DataFrame({"ear_velocity_smooth_mm_s": [0.0, 0.0, 100.0, 100.0, 100.0]})
    out = bc.append_behavior_labels(df, threshold=50.0, median_filter_size=1)
    assert list(out["behavior"]) == ["rest", "rest", "movement", "movement", "movement"]


def test_append_behavior_labels_missing_column():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(KeyError):
        bc.append_behavior_labels(df)
