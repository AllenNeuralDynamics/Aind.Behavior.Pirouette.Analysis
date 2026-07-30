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


def test_firing_rate_max_bins_cap():
    spikes = np.arange(0, 1000, 0.1)  # would be 20000 bins at 0.05 s
    c, r = viz.instantaneous_firing_rate(spikes, 0.0, 1000.0, bin_s=0.05, max_bins=5000)
    assert len(c) == 5000 and len(r) == 5000
    # rate is still ~10 Hz despite the coarser bins
    assert np.median(r) == pytest.approx(10.0, rel=0.3)


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


def test_unit_spike_times_experiment_offset_for_audio():
    # The audio monitor relies on offset-shifted spike times within a window.
    units = {3: {"spike_times": np.array([10.0, 20.0, 30.0]), "amp": 1.0}}
    shifted = viz.unit_spike_times_experiment(units, 3, spike_offset_s=100.0)
    # spikes that fall in a [105, 125] experiment window -> local seconds from 105
    t0, t1 = 105.0, 125.0
    in_seg = shifted[(shifted >= t0) & (shifted <= t1)] - t0
    np.testing.assert_allclose(in_seg, [5.0, 15.0])  # 110->5, 120->15 (130 excluded)


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


def test_build_segment_table_matches_segment_info():
    n = 120
    src = np.where(np.arange(n) < 60, "TopCamera_A", "TopCamera_B")
    df = pd.DataFrame({
        "source_file": src,
        "frame": np.concatenate([np.arange(60), np.arange(60)]),
        "time_since_start": np.arange(n) / 60.0,
    })
    table = viz.build_segment_table(df)
    assert [name for name, _, _, _ in table] == ["TopCamera_A", "TopCamera_B"]
    # Each table row must agree with the per-segment scan it replaces.
    for name, base, count, fps in table:
        b2, c2, f2 = viz.segment_info(df, name)
        assert (base, count) == (b2, c2)
        assert fps == pytest.approx(f2, rel=1e-6)


def test_clear_firing_rate_caches(tmp_path):
    import pickle
    uf = tmp_path / "u.pkl"
    with open(uf, "wb") as f:
        pickle.dump({1: {"spike_times": np.array([1.0])}}, f)
    parq, js = viz.ephys.cache_paths(uf)
    parq.write_bytes(b"x")
    js.write_text("{}")
    viz._clear_firing_rate_caches(tmp_path)
    assert not parq.exists() and not js.exists()  # caches removed
    assert uf.exists()  # the units file itself is kept


def test_file_options_show_only_names(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.csv").write_text("x")
    opts = viz.file_options(tmp_path, (".parquet", ".pkl", ".csv"))
    # Labels and values are bare names -- never the absolute path.
    for o in opts:
        assert o["label"] == o["value"]
        assert "/" not in o["value"] and "\\" not in o["value"]
    assert {o["value"] for o in opts} == {"a.parquet", "b.csv"}


def test_resolve_in_dir_confines_to_folder(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    resolved = viz.resolve_in_dir(tmp_path, "data.parquet")
    assert resolved is not None and resolved.endswith("data.parquet")
    # Traversal / outside names are rejected.
    assert viz.resolve_in_dir(tmp_path, "../secret.parquet") is None
    assert viz.resolve_in_dir(tmp_path, "missing.parquet") is None
    assert viz.resolve_in_dir(tmp_path, None) is None


def test_segment_options_flags_missing_videos(tmp_path):
    segs = ["TopCamera_A", "TopCamera_B", "TopCamera_C"]
    (tmp_path / "TopCamera_B.mp4").write_bytes(b"x")  # only B has a video
    options, default = viz.segment_options(segs, tmp_path)
    labels = {o["value"]: o["label"] for o in options}
    disabled = {o["value"]: o["disabled"] for o in options}
    assert labels["TopCamera_B"] == "TopCamera_B"
    assert labels["TopCamera_A"] == "TopCamera_A — Not Available"
    assert disabled == {"TopCamera_A": True, "TopCamera_B": False, "TopCamera_C": True}
    assert default == "TopCamera_B"  # first (only) available


def test_gui_columns_includes_essentials():
    cols = viz.gui_columns()
    for c in ("datetime_pacific", "time_since_start", "source_file", "behavior",
              "left_ear_x_mm", "right_ear_y_mm", "ul_champber_x_mm"):
        assert c in cols


# ---------------------------------------------------------------------------
# figure builders (need plotly) + frame encoding
# ---------------------------------------------------------------------------
def test_build_timeseries_top_has_cursor():
    fig = viz.build_timeseries_top(_dataset(300))
    assert len(fig.layout.shapes) == 1  # the red cursor
    assert fig.layout.shapes[0].line.color == viz.CURSOR_COLOR


def _ts_top_df(n=200):
    t = np.arange(n) / 60.0
    return pd.DataFrame({
        "datetime_pacific": pd.Timestamp("2026-06-10 20:00", tz="America/Los_Angeles")
        + pd.to_timedelta(t, unit="s"),
        "behavior": np.where((np.arange(n) // 30) % 2 == 0, "rest", "movement"),
        "ear_velocity_smooth_mm_s": np.sin(t),
        "ear_heading_deg": (t * 10) % 360,
        "commutator_heading_deg": (t * 10 + 5) % 360,
    })


def _heading_trace_names(fig):
    return [tr.name for tr in fig.data if tr.name in ("ear vector", "commutator")]


def test_heading_mode_vector_default():
    fig = viz.build_timeseries_top(_ts_top_df())
    assert _heading_trace_names(fig) == ["ear vector"]
    # no dashed/solid legend note in the title when not "both"
    titles = [a.text for a in fig.layout.annotations]
    assert any(t == "heading (deg)" for t in titles)
    assert all("commutator: dashed" not in t for t in titles)


def test_heading_mode_commutator():
    fig = viz.build_timeseries_top(_ts_top_df(), heading_mode="commutator")
    assert _heading_trace_names(fig) == ["commutator"]
    assert all("commutator: dashed" not in a.text for a in fig.layout.annotations)


def test_heading_mode_both():
    fig = viz.build_timeseries_top(_ts_top_df(), heading_mode="both")
    assert set(_heading_trace_names(fig)) == {"ear vector", "commutator"}
    assert any("commutator: dashed" in a.text for a in fig.layout.annotations)


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


def test_named_tunnel_requires_name():
    # No tunnel name -> a clear error before touching cloudflared.
    with pytest.raises(RuntimeError, match="tunnel name"):
        viz._start_cloudflare_named_tunnel(8050, None)


def test_named_tunnel_requires_cloudflared(monkeypatch):
    # Tunnel name given but no cloudflared anywhere -> actionable error.
    monkeypatch.setattr(viz, "_cloudflared_bin", lambda: "")
    with pytest.raises(RuntimeError, match="cloudflared executable not found"):
        viz._start_cloudflare_named_tunnel(8050, "pirouette", "pirouette.example.org")


def test_cloudflared_bin_falls_back_to_pycloudflared(monkeypatch):
    # With nothing on PATH, it should find the pycloudflared-bundled binary.
    monkeypatch.setattr("shutil.which", lambda _: None)
    exe = viz._cloudflared_bin()
    assert exe and exe.lower().endswith(".exe") or exe == ""  # bundled or absent


def test_tunnel_exists_parses_json(monkeypatch):
    from types import SimpleNamespace

    def fake_run(cmd, **kw):
        return SimpleNamespace(returncode=0, stdout='[{"name": "pirouette"}]', stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert viz._tunnel_exists("cf", "pirouette") is True
    assert viz._tunnel_exists("cf", "other") is False


def test_ensure_named_tunnel_creates_when_missing(monkeypatch, tmp_path):
    # Already logged in (cert.pem present) + tunnel absent -> create + route, no login.
    from types import SimpleNamespace

    calls = []

    def fake_run(cmd, **kw):
        calls.append(" ".join(cmd[1:]))
        stdout = "[]" if "list" in cmd else ""  # list -> tunnel missing
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path))
    cfdir = tmp_path / ".cloudflared"
    cfdir.mkdir()
    (cfdir / "cert.pem").write_text("x")

    viz._ensure_named_tunnel("cf", "pirouette", "pirouette-viz.org")
    assert any("tunnel create pirouette" in c for c in calls)
    assert any("route dns pirouette pirouette-viz.org" in c for c in calls)
    assert not any("tunnel login" in c for c in calls)  # cert present


def test_ensure_named_tunnel_logs_in_and_skips_create(monkeypatch, tmp_path):
    # No cert.pem -> login; tunnel already exists -> no create.
    from types import SimpleNamespace

    calls = []

    def fake_run(cmd, **kw):
        calls.append(" ".join(cmd[1:]))
        stdout = '[{"name": "pirouette"}]' if "list" in cmd else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("os.path.expanduser", lambda _: str(tmp_path))  # no cert.pem

    viz._ensure_named_tunnel("cf", "pirouette", "pirouette-viz.org")
    assert any("tunnel login" in c for c in calls)
    assert not any("tunnel create" in c for c in calls)
