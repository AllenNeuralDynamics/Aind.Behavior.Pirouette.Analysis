"""Tests for :mod:`pirouette_data.ephys` (firing rate + precomputed cache)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pirouette_data import ephys


def _units():
    # unit 1: steady 10 Hz for 100 s; unit 2: sparse.
    return {
        1: {"spike_times": np.arange(0.0, 100.0, 0.1)},
        2: {"spike_times": np.array([1.0, 2.0, 50.0, 99.0])},
    }


def _write_units(tmp_path):
    p = tmp_path / "good_units.pkl"
    with open(p, "wb") as f:
        pickle.dump(_units(), f)
    return p


def test_instantaneous_firing_rate_basic():
    spikes = np.arange(0, 10, 0.1)  # 10 Hz
    centers, rate = ephys.instantaneous_firing_rate(spikes, 0.0, 10.0, bin_s=0.1)
    assert centers.shape == rate.shape
    assert np.median(rate) == pytest.approx(10.0, rel=0.3)


def test_instantaneous_firing_rate_empty_window():
    c, r = ephys.instantaneous_firing_rate(np.array([1.0]), 5.0, 5.0)
    assert c.size == 0 and r.size == 0


def test_compute_firing_rates_downsamples():
    units = _units()
    uids = sorted(units)
    centers, rates = ephys.compute_firing_rates(
        units, uids, bin_s=0.05, smooth_s=0.2, max_points=200
    )
    assert len(centers) <= 200
    assert set(rates) == {"1", "2"}
    # shared grid: every unit's rate matches the centres length
    for r in rates.values():
        assert len(r) == len(centers)
        assert r.dtype == np.float32
    # unit 1 (~10 Hz) has a higher median rate than the sparse unit 2
    assert np.median(rates["1"]) > np.median(rates["2"])


def test_save_load_roundtrip(tmp_path):
    p = _write_units(tmp_path)
    units = _units()
    centers, rates = ephys.compute_firing_rates(
        units, sorted(units), bin_s=0.05, smooth_s=0.2, max_points=200
    )
    meta = ephys._meta(p, 0.05, 0.2, 200, sorted(units))
    ephys.save_firing_rates(p, centers, rates, meta)
    loaded = ephys.load_firing_rates(p)
    assert loaded is not None
    np.testing.assert_allclose(loaded["centers_s"], centers)
    np.testing.assert_allclose(loaded["rates"]["1"], rates["1"], rtol=1e-5)
    assert loaded["meta"] == meta


def test_ensure_reuses_then_recomputes(tmp_path, monkeypatch):
    p = _write_units(tmp_path)
    units = _units()
    calls = {"n": 0}
    real = ephys.compute_firing_rates

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ephys, "compute_firing_rates", counted)

    ephys.ensure_firing_rates(units, p, 0.05, 0.2, 200)
    assert calls["n"] == 1
    # Same params -> served from disk, no recompute.
    ephys.ensure_firing_rates(units, p, 0.05, 0.2, 200)
    assert calls["n"] == 1
    # Changed bin width -> recompute.
    ephys.ensure_firing_rates(units, p, 0.02, 0.2, 200)
    assert calls["n"] == 2


def test_cache_is_offset_independent(tmp_path):
    # Rates are stored over RAW spike seconds; the GUI applies the offset at view
    # time, so the cache must not encode any offset.
    p = _write_units(tmp_path)
    units = _units()
    cache = ephys.ensure_firing_rates(units, p, 0.05, 0.2, 200)
    assert "offset" not in cache["meta"]
    # centres start near the first raw spike (0 s), not shifted by any offset.
    assert cache["centers_s"][0] < 1.0
