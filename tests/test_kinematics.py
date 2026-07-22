"""Unit tests for :mod:`pirouette_data.kinematics` (pure, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirouette_data import kinematics


# ---------------------------------------------------------------------------
# commutator_heading_estimate
# ---------------------------------------------------------------------------
def test_heading_quarter_turns():
    turns = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = kinematics.commutator_heading_estimate(turns, reference_value=0.0)
    np.testing.assert_allclose(out, [0.0, 90.0, 180.0, 270.0, 0.0])


def test_heading_reference_defaults_to_first():
    turns = np.array([10.0, 10.25, 10.5])
    out = kinematics.commutator_heading_estimate(turns)  # reference = 10.0
    np.testing.assert_allclose(out, [0.0, 90.0, 180.0])


def test_heading_reference_subtraction():
    turns = np.array([183.5, 183.75, 184.0, 183.25])
    out = kinematics.commutator_heading_estimate(turns, reference_value=183.5)
    np.testing.assert_allclose(out, [0.0, 90.0, 180.0, 270.0])


def test_heading_offset_applied_and_wrapped():
    turns = np.array([0.0, 0.25, 0.75])
    out = kinematics.commutator_heading_estimate(
        turns, reference_value=0.0, offset_deg=90.0
    )
    np.testing.assert_allclose(out, [90.0, 180.0, 0.0])


def test_heading_negative_net_wraps_positive():
    turns = np.array([0.0, -0.25, -0.5])
    out = kinematics.commutator_heading_estimate(turns, reference_value=0.0)
    np.testing.assert_allclose(out, [0.0, 270.0, 180.0])


def test_heading_direction_flip():
    turns = np.array([0.0, 0.25, 0.5])
    out = kinematics.commutator_heading_estimate(
        turns, reference_value=0.0, direction=-1
    )
    np.testing.assert_allclose(out, [0.0, 270.0, 180.0])


def test_heading_multiple_full_turns_wrap():
    turns = np.array([0.0, 2.25, 5.5])  # many revolutions
    out = kinematics.commutator_heading_estimate(turns, reference_value=0.0)
    np.testing.assert_allclose(out, [0.0, 90.0, 180.0])


def test_heading_output_range():
    rng = np.linspace(-5, 5, 101)
    out = kinematics.commutator_heading_estimate(rng, reference_value=0.0)
    assert np.all(out >= 0.0) and np.all(out < 360.0)


# ---------------------------------------------------------------------------
# ear-based offset calibration
# ---------------------------------------------------------------------------
def _ear_frame(lx, ly, rx, ry, like=1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left_ear_x": [lx],
            "left_ear_y": [ly],
            "left_ear_likelihood": [like],
            "right_ear_x": [rx],
            "right_ear_y": [ry],
            "right_ear_likelihood": [like],
        }
    )


def test_offset_facing_up():
    # Ears horizontal (left at x=0, right at x=10, same y). Nose (forward) is the
    # +90 deg CCW rotation of left->right -> points up -> 90 deg (standard quadrant).
    df = _ear_frame(0.0, 0.0, 10.0, 0.0)
    assert kinematics.heading_offset_from_ears(df) == pytest.approx(90.0)


def test_offset_facing_right():
    # In image coords (y down): right ear below left ear -> forward points +x (right).
    df = _ear_frame(0.0, 0.0, 0.0, 10.0)
    assert kinematics.heading_offset_from_ears(df) == pytest.approx(0.0)


def test_offset_forward_sign_flips_180():
    df = _ear_frame(0.0, 0.0, 10.0, 0.0)
    a = kinematics.heading_offset_from_ears(df, forward_sign=1)
    b = kinematics.heading_offset_from_ears(df, forward_sign=-1)
    assert abs(((a - b) % 360.0) - 180.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# first_valid_ear_frame
# ---------------------------------------------------------------------------
def test_first_valid_ear_frame_skips_low_likelihood():
    df = pd.DataFrame(
        {
            "left_ear_x": [1.0, 2.0, 3.0],
            "left_ear_y": [1.0, 2.0, 3.0],
            "left_ear_likelihood": [0.1, 0.9, 0.9],
            "right_ear_x": [1.0, 2.0, 3.0],
            "right_ear_y": [1.0, 2.0, 3.0],
            "right_ear_likelihood": [0.9, 0.2, 0.9],
        }
    )
    # frame 0: left below threshold; frame 1: right below threshold; frame 2 ok.
    assert kinematics.first_valid_ear_frame(df, likelihood_threshold=0.5) == 2


def test_first_valid_ear_frame_skips_nan():
    df = _ear_frame(np.nan, 0.0, 10.0, 0.0)
    df = pd.concat([df, _ear_frame(0.0, 0.0, 10.0, 0.0)], ignore_index=True)
    assert kinematics.first_valid_ear_frame(df) == 1


def test_first_valid_ear_frame_raises_when_none():
    df = _ear_frame(0.0, 0.0, 10.0, 0.0, like=0.0)
    with pytest.raises(ValueError):
        kinematics.first_valid_ear_frame(df, likelihood_threshold=0.5)


def test_first_valid_ear_frame_missing_columns():
    df = pd.DataFrame({"left_ear_x": [0.0]})
    with pytest.raises(KeyError):
        kinematics.first_valid_ear_frame(df)
