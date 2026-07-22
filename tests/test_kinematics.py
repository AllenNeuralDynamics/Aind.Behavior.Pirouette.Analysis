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


# ---------------------------------------------------------------------------
# ear_heading_estimate (per-frame)
# ---------------------------------------------------------------------------
def _ear_series(lx, ly, rx, ry, like):
    return pd.DataFrame(
        {
            "left_ear_x": lx,
            "left_ear_y": ly,
            "left_ear_likelihood": like,
            "right_ear_x": rx,
            "right_ear_y": ry,
            "right_ear_likelihood": like,
        }
    )


def test_ear_heading_per_frame_values():
    # Frame 0: ears horizontal -> facing up (90). Frame 1: right ear below left
    # (image y down) -> facing right (0).
    df = _ear_series(
        lx=[0.0, 0.0], ly=[0.0, 0.0], rx=[10.0, 0.0], ry=[0.0, 10.0], like=[1.0, 1.0]
    )
    out = kinematics.ear_heading_estimate(df)
    np.testing.assert_allclose(out, [90.0, 0.0])


def test_ear_heading_output_in_range():
    rng = np.random.default_rng(0)
    n = 200
    df = _ear_series(
        lx=rng.normal(size=n),
        ly=rng.normal(size=n),
        rx=rng.normal(size=n) + 5,
        ry=rng.normal(size=n),
        like=np.ones(n),
    )
    out = kinematics.ear_heading_estimate(df)
    assert np.all(out >= 0.0) and np.all(out < 360.0)


def test_ear_heading_interpolates_missing():
    # Middle frame has a low-likelihood ear -> should be filled by interpolation
    # between the two valid neighbours (both at 90 deg -> stays 90).
    df = _ear_series(
        lx=[0.0, 0.0, 0.0],
        ly=[0.0, 0.0, 0.0],
        rx=[10.0, 10.0, 10.0],
        ry=[0.0, 0.0, 0.0],
        like=[1.0, 0.0, 1.0],
    )
    out = kinematics.ear_heading_estimate(df, likelihood_threshold=0.5)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, [90.0, 90.0, 90.0])


def test_ear_heading_circular_interpolation_shortest_path():
    # Neighbours at 350 and 10 deg -> shortest-path midpoint is 0, not 180.
    heading = np.array([350.0, np.nan, 10.0])
    filled = kinematics._interpolate_circular(heading)
    # 0 deg and 360 deg are the same angle; compare circularly.
    d = filled[1] % 360.0
    assert min(d, 360.0 - d) == pytest.approx(0.0, abs=1e-6)


def test_ear_heading_no_interpolation_keeps_nan():
    df = _ear_series(
        lx=[0.0, 0.0], ly=[0.0, 0.0], rx=[10.0, 10.0], ry=[0.0, 0.0], like=[1.0, 0.0]
    )
    out = kinematics.ear_heading_estimate(
        df, likelihood_threshold=0.5, interpolate=False
    )
    assert not np.isnan(out[0]) and np.isnan(out[1])


def test_append_ear_heading_adds_column():
    df = _ear_series(
        lx=[0.0, 0.0], ly=[0.0, 0.0], rx=[10.0, 10.0], ry=[0.0, 0.0], like=[1.0, 1.0]
    )
    out = kinematics.append_ear_heading(df)
    assert "ear_heading_deg" in out.columns
    assert "ear_heading_deg" not in df.columns  # original untouched
    np.testing.assert_allclose(out["ear_heading_deg"], [90.0, 90.0])


def test_interpolate_circular_edge_fill():
    # Leading/trailing NaNs -> filled with nearest valid value.
    heading = np.array([np.nan, 45.0, np.nan])
    filled = kinematics._interpolate_circular(heading)
    np.testing.assert_allclose(filled, [45.0, 45.0, 45.0])


# ---------------------------------------------------------------------------
# ear_midpoint
# ---------------------------------------------------------------------------
def _moving_ears_df(n=11, dt=0.1, step_x=1.0, like=1.0):
    """Rigid ears translating in +x. Facing = +x (right): left above right in y.

    Ears at left=(x_t, 0), right=(x_t, 10) (image coords, mm) -> orthogonal points
    +x, so the animal faces right and moves forward when x increases.
    """
    x = np.arange(n, dtype="float64") * step_x
    return pd.DataFrame(
        {
            "harp_time": np.arange(n, dtype="float64") * dt,
            "left_ear_x_mm": x.copy(),
            "left_ear_y_mm": np.zeros(n),
            "left_ear_likelihood": np.full(n, like),
            "right_ear_x_mm": x.copy(),
            "right_ear_y_mm": np.full(n, 10.0),
            "right_ear_likelihood": np.full(n, like),
        }
    )


def test_ear_midpoint_both_present():
    df = _moving_ears_df(n=3, step_x=2.0)
    mx, my = kinematics.ear_midpoint(df)
    np.testing.assert_allclose(mx, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(my, [5.0, 5.0, 5.0])  # midpoint of y=0 and y=10


def test_ear_midpoint_uses_present_ear():
    df = _moving_ears_df(n=3, step_x=2.0)
    df.loc[1, "right_ear_likelihood"] = 0.0  # right missing at frame 1
    mx, my = kinematics.ear_midpoint(df, likelihood_threshold=0.5)
    # frame 1 falls back to the left ear position (y = 0), not the midpoint (y=5)
    np.testing.assert_allclose(my, [5.0, 0.0, 5.0])
    np.testing.assert_allclose(mx, [0.0, 2.0, 4.0])


def test_ear_midpoint_interpolates_when_both_missing():
    df = _moving_ears_df(n=3, step_x=2.0)
    df.loc[1, ["left_ear_likelihood", "right_ear_likelihood"]] = 0.0
    mx, my = kinematics.ear_midpoint(df, likelihood_threshold=0.5)
    assert not np.isnan(mx).any() and not np.isnan(my).any()
    assert mx[1] == pytest.approx(2.0)  # linear between 0 and 4
    assert my[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ear_velocity_estimate / append_ear_velocity
# ---------------------------------------------------------------------------
def test_velocity_forward_positive():
    # Facing +x and moving +x at 1 mm / 0.1 s = 10 mm/s -> +10.
    df = _moving_ears_df(n=11, dt=0.1, step_x=1.0)
    inst, smooth = kinematics.ear_velocity_estimate(df, smoothing_sigma_s=0.0)
    np.testing.assert_allclose(inst, 10.0)
    np.testing.assert_allclose(smooth, 10.0)


def test_velocity_backward_negative():
    # Facing +x but moving -x -> negative (backward).
    df = _moving_ears_df(n=11, dt=0.1, step_x=-1.0)
    inst, _ = kinematics.ear_velocity_estimate(df, smoothing_sigma_s=0.0)
    np.testing.assert_allclose(inst, -10.0)


def test_velocity_projection_ignores_lateral():
    # Facing +x, moving purely in +y (lateral). projection -> ~0; signed_speed -> full.
    df = _moving_ears_df(n=11, dt=0.1, step_x=0.0)
    # move both ears in +y over time
    df["left_ear_y_mm"] = df["left_ear_y_mm"] + np.arange(11) * 1.0
    df["right_ear_y_mm"] = df["right_ear_y_mm"] + np.arange(11) * 1.0
    proj, _ = kinematics.ear_velocity_estimate(
        df, method="projection", smoothing_sigma_s=0.0
    )
    np.testing.assert_allclose(proj, 0.0, atol=1e-9)
    speed, _ = kinematics.ear_velocity_estimate(
        df, method="signed_speed", smoothing_sigma_s=0.0
    )
    np.testing.assert_allclose(np.abs(speed), 10.0)


def test_velocity_units_scale_with_dt():
    df = _moving_ears_df(n=11, dt=0.05, step_x=1.0)  # 1 mm / 0.05 s = 20 mm/s
    inst, _ = kinematics.ear_velocity_estimate(df, smoothing_sigma_s=0.0)
    np.testing.assert_allclose(inst, 20.0)


def test_velocity_smoothing_preserves_constant():
    df = _moving_ears_df(n=51, dt=0.1, step_x=1.0)
    inst, smooth = kinematics.ear_velocity_estimate(df, smoothing_sigma_s=0.1)
    # A constant velocity is unchanged by Gaussian smoothing (interior).
    np.testing.assert_allclose(smooth[10:-10], 10.0, atol=1e-6)


def test_append_ear_velocity_columns():
    df = _moving_ears_df(n=11, dt=0.1, step_x=1.0)
    out = kinematics.append_ear_velocity(df, smoothing_sigma_s=0.05)
    assert "ear_velocity_mm_s" in out.columns
    assert "ear_velocity_smooth_mm_s" in out.columns
    assert "ear_velocity_mm_s" not in df.columns  # input untouched
    np.testing.assert_allclose(out["ear_velocity_mm_s"], 10.0)


def test_velocity_missing_time_column_raises():
    df = _moving_ears_df(n=5)
    df = df.drop(columns=["harp_time"])
    with pytest.raises(KeyError):
        kinematics.ear_velocity_estimate(df)


def test_velocity_bad_method_raises():
    df = _moving_ears_df(n=5)
    with pytest.raises(ValueError):
        kinematics.ear_velocity_estimate(df, method="nope")
