"""Tests for :mod:`pirouette_data.animations` (spike-raster zoom animation)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pirouette_data import animations as anim


def _units(with_depth=True):
    u = {
        0: {"spike_times": np.array([0.0, 0.5, 1.0, 10.0, 100.0]), "depth": 800.0},
        1: {"spike_times": np.array([0.2, 0.9, 5.0, 50.0]), "depth": 200.0},
        2: {"spike_times": np.array([0.1, 2.0, 20.0]), "depth": 500.0},
    }
    if not with_depth:
        del u[1]["depth"]
    return u


def _write(tmp_path, units):
    p = tmp_path / "good_units.pkl"
    with open(p, "wb") as f:
        pickle.dump(units, f)
    return p


def test_load_requires_depth(tmp_path):
    p = _write(tmp_path, _units(with_depth=False))
    with pytest.raises(ValueError, match="depth"):
        anim.load_units_with_depth(p)


def test_load_requires_spike_times(tmp_path):
    u = _units()
    del u[2]["spike_times"]
    p = _write(tmp_path, u)
    with pytest.raises(ValueError, match="spike_times"):
        anim.load_units_with_depth(p)


def test_load_ok(tmp_path):
    p = _write(tmp_path, _units())
    units = anim.load_units_with_depth(p)
    assert set(units) == {0, 1, 2}


def test_prepare_units_orders_by_depth():
    per = anim.prepare_units(_units())
    # rank 0 = shallowest (depth 200), deepest (800) is the top rank
    assert [pu["depth"] for pu in per] == [200.0, 500.0, 800.0]
    assert [pu["rank"] for pu in per] == [0, 1, 2]
    # times are sorted
    for pu in per:
        assert np.all(np.diff(pu["times"]) >= 0)


def test_prepare_units_invert_depth():
    per = anim.prepare_units(_units(), invert_depth=True)
    assert [pu["depth"] for pu in per] == [800.0, 500.0, 200.0]


def test_pick_scale_bar():
    assert anim.pick_scale_bar(0.01) == (0.001, "1 ms")   # 10 ms view -> 1 ms bar
    assert anim.pick_scale_bar(0.5) == (0.001, "1 ms")
    assert anim.pick_scale_bar(5.0) == (1.0, "1 s")
    assert anim.pick_scale_bar(120.0) == (60.0, "1 min")
    assert anim.pick_scale_bar(7200.0) == (3600.0, "1 hour")
    assert anim.pick_scale_bar(129600.0) == (43200.0, "12 hours")


def test_orders_of_magnitude():
    assert anim.orders_of_magnitude(0.01, 129600.0) == pytest.approx(7.11, abs=0.01)
    assert anim.orders_of_magnitude(1.0, 1000.0) == pytest.approx(3.0)


def test_frame_points_narrow_keeps_all():
    per = anim.prepare_units(_units())
    x, y, r = anim.frame_points(per, 0.0, 1.05, cap=100)
    # spikes in [0, 1.05]: unit0 {0,0.5,1.0}, unit1 {0.2,0.9}, unit2 {0.1} = 6
    assert len(x) == 6
    assert set(r.tolist()) <= {0, 1, 2}
    assert np.all((x >= 0.0) & (x <= 1.05))


def test_frame_points_wide_subsamples_to_cap():
    # many spikes, small cap -> subsampled near cap, colours preserved
    units = {i: {"spike_times": np.arange(0, 1000, 0.1), "depth": float(i)}
             for i in range(5)}
    per = anim.prepare_units(units)
    x, y, r = anim.frame_points(per, 0.0, 1000.0, cap=500)
    assert 0 < len(x) <= 500 + 5  # ~cap (per-unit rounding slack)
    assert set(r.tolist()) == {0, 1, 2, 3, 4}  # every unit still represented


def test_frame_points_empty_window():
    per = anim.prepare_units(_units())
    x, y, r = anim.frame_points(per, 1000.0, 2000.0, cap=100)
    assert len(x) == 0 and len(r) == 0


def test_window_bounds_clamps():
    lo, hi = anim._window_bounds(5.0, 4.0, 0.0, 100.0)
    assert (lo, hi) == (3.0, 7.0)
    lo, hi = anim._window_bounds(1.0, 10.0, 0.0, 100.0)  # clamp to left edge
    assert lo == 0.0 and hi == 10.0
