"""Unit tests for :mod:`pirouette_data.visualization_gui` data layer + figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirouette_data import visualization_gui as viz


def _dataset(n=600, fps=60.0):
    """Small synthetic dataset resembling the build_dataset output."""
    t = np.arange(n) / fps
    start = pd.Timestamp("2026-06-10 20:00:00", tz="America/Los_Angeles")
    # experiment started 100 s before the first frame
    time_since_start = t + 100.0
    dt = start + pd.to_timedelta(t, unit="s")
    behavior = np.where((np.arange(n) // 60) % 2 == 0, "rest", "movement")
    return pd.DataFrame(
        {
            "source_file": ["TopCamera_2026-06-10T20-00-00"] * n,
            "frame": np.arange(n),
            "harp_time": 3.86e9 + time_since_start,
            "time_since_start": time_since_start,
            "datetime_pacific": dt,
            "behavior": behavior,
            "ear_velocity_smooth_mm_s": np.sin(t),
            "ear_heading_deg": (t * 10) % 360,
            "commutator_heading_deg": (t * 10 + 5) % 360,
            "left_ear_x_mm": np.cos(t) * 10 + 50,
            "left_ear_y_mm": np.sin(t) * 10 + 50,
            "right_ear_x_mm": np.cos(t) * 10 + 60,
            "right_ear_y_mm": np.sin(t) * 10 + 50,
        }
    )


def _units():
    return {
        0: {"spike_times": np.array([100.5, 101.0, 105.0, 200.0]), "amp": 1.0},
        7: {"spike_times": np.array([100.1, 100.2, 100.3]), "amp": 2.0},
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ext", [".parquet", ".pkl", ".csv"])
def test_load_dataset_roundtrip(tmp_path, ext):
    df = _dataset(50)
    p = tmp_path / f"d{ext}"
    if ext == ".parquet":
        df.to_parquet(p)
    elif ext == ".pkl":
        df.to_pickle(p)
    else:
        df.to_csv(p, index=False)
    out = viz.load_dataset(p)
    assert len(out) == 50
    assert pd.api.types.is_datetime64_any_dtype(out["datetime_pacific"])


def test_load_dataset_bad_ext(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        viz.load_dataset(p)


def test_load_units_and_ids(tmp_path):
    import pickle
    p = tmp_path / "u.pkl"
    with open(p, "wb") as f:
        pickle.dump(_units(), f)
    u = viz.load_units(p)
    assert viz.unit_ids(u) == [0, 7]


# ---------------------------------------------------------------------------
# time / spike alignment
# ---------------------------------------------------------------------------
def test_experiment_start_datetime():
    df = _dataset(10)
    start = viz.experiment_start_datetime(df)
    # first frame is 100 s after experiment start
    assert (df["datetime_pacific"].iloc[0] - start).total_seconds() == pytest.approx(100.0)


def test_spike_offset_applied():
    u = _units()
    raw = u[0]["spike_times"]
    shifted = viz.unit_spike_times_experiment(u, 0, spike_offset_s=1000.0)
    np.testing.assert_allclose(shifted, raw + 1000.0)


def test_spikes_to_datetime():
    df = _dataset(10)
    start = viz.experiment_start_datetime(df)
    dts = viz.spikes_to_datetime(np.array([0.0, 100.0]), start)
    assert dts[0] == start
    assert (dts[1] - start).total_seconds() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# firing rate / bouts / head position
# ---------------------------------------------------------------------------
def test_firing_rate_basic():
    spikes = np.arange(0, 10, 0.1)  # 10 Hz for 10 s
    centers, rate = viz.instantaneous_firing_rate(spikes, 0.0, 10.0, bin_s=0.1)
    assert centers.shape == rate.shape
    assert np.median(rate) == pytest.approx(10.0, rel=0.3)


def test_firing_rate_empty_window():
    c, r = viz.instantaneous_firing_rate(np.array([1.0]), 5.0, 5.0)
    assert c.size == 0 and r.size == 0


def test_behavior_bouts_alternate():
    df = _dataset(180)  # 3 bouts of 60 frames: rest, movement, rest
    bouts = viz.behavior_bouts(df)
    assert [b[2] for b in bouts] == ["rest", "movement", "rest"]


def test_head_position_midpoint():
    df = _dataset(5)
    hx, hy = viz.head_position_mm(df)
    # midpoint x = average of left/right ear x (which differ by 10)
    np.testing.assert_allclose(hx, (df["left_ear_x_mm"] + df["right_ear_x_mm"]) / 2)
    np.testing.assert_allclose(hy, df["left_ear_y_mm"])  # ear y equal here


def test_frame_mapping():
    df = _dataset(10)
    assert viz.frame_index_for_row(df, 5) == 5
    assert viz.video_path_for_row(df, 5, "/vids").name == "TopCamera_2026-06-10T20-00-00.mp4"


def test_stride():
    assert viz._stride(100, 50) == 2
    assert viz._stride(10, 50) == 1


def test_segments_and_segment_info():
    n = 120
    src = np.where(np.arange(n) < 60, "TopCamera_A", "TopCamera_B")
    df = pd.DataFrame({
        "source_file": src,
        "frame": np.concatenate([np.arange(60), np.arange(60)]),
        "time_since_start": np.arange(n) / 60.0,
    })
    assert viz.segments(df) == ["TopCamera_A", "TopCamera_B"]
    base, count, fps = viz.segment_info(df, "TopCamera_B")
    assert base == 60 and count == 60
    assert fps == pytest.approx(60.0, rel=0.05)


# ---------------------------------------------------------------------------
# figure builders (need plotly) + frame encoding
# ---------------------------------------------------------------------------
def test_build_timeseries_top_has_cursor():
    fig = viz.build_timeseries_top(_dataset(300))
    assert len(fig.layout.shapes) == 1  # the red cursor
    assert fig.layout.shapes[0].line.color == viz.CURSOR_COLOR


def test_build_head_position_current_marker():
    df = _dataset(300)
    hx, hy = viz.head_position_mm(df)
    ht = df["time_since_start"].to_numpy()
    fig = viz.build_head_position(hx, hy, ht, current_row=150, window_s=1.0)
    assert len(fig.data) == 2  # trail + current marker (no chamber)
    # markers only, no connecting line
    assert fig.data[0].mode == "markers"


def test_build_head_position_with_chamber_box():
    df = _dataset(300)
    hx, hy = viz.head_position_mm(df)
    ht = df["time_since_start"].to_numpy()
    chamber = {
        "ul_champber": (0.0, 0.0), "ur_champber": (373.0, 0.0),
        "lr_chamber": (373.0, 194.0), "ll_chamber": (0.0, 194.0),
    }
    fig = viz.build_head_position(hx, hy, ht, 150, 1.0, chamber=chamber)
    assert len(fig.data) == 3  # trail + current + chamber box
    assert fig.data[2].line.color == "black"


def test_chamber_corners_mm():
    df = pd.DataFrame({
        "ul_champber_x_mm": [0.0, 0.2], "ul_champber_y_mm": [0.0, 0.0],
        "ur_champber_x_mm": [373.0, 373.0], "ur_champber_y_mm": [0.0, 0.0],
        "lr_chamber_x_mm": [373.0, 373.0], "lr_chamber_y_mm": [194.0, 194.0],
        "ll_chamber_x_mm": [0.0, 0.0], "ll_chamber_y_mm": [194.0, 194.0],
    })
    corners = viz.chamber_corners_mm(df)
    assert set(corners) == {"ul_champber", "ur_champber", "lr_chamber", "ll_chamber"}
    assert corners["ur_champber"] == (373.0, 0.0)


def test_frame_to_data_uri_placeholder():
    uri = viz.frame_to_data_uri(None, "no video")
    assert uri.startswith("data:image/jpeg;base64,")
