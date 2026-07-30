"""Ephys utilities: instantaneous firing rate + a precomputed rate cache.

The instantaneous firing rate for a busy unit (millions of spikes) takes ~1 s to
compute, which made switching units in the GUI slow. This module computes it once
per units file and caches the (display-resolution) result to a parquet next to the
units file, so the GUI just reads it.

The cache is deliberately **offset- and dataset-independent**: rates are computed
over each unit's *raw* spike-time range (no experiment offset) and downsampled to
the plot resolution. At view time the GUI shifts the shared bin centres by the
current ``spike_offset_s`` and slices to the visible window -- so changing the
offset (or pairing the same units with a different pose dataset) needs no
recompute. Only ``bin_s`` / ``smooth_sigma_s`` / the units file itself invalidate
it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_VERSION = 2


def instantaneous_firing_rate(
    spike_exp_s: np.ndarray,
    t0_s: float,
    t1_s: float,
    bin_s: float = 0.05,
    smooth_sigma_s: float = 0.2,
    max_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smoothed instantaneous firing rate over ``[t0_s, t1_s]``.

    Parameters
    ----------
    spike_exp_s:
        Spike times (seconds) on whatever timeline ``t0_s``/``t1_s`` use.
    t0_s, t1_s:
        Time window (seconds) over which to compute the rate.
    bin_s:
        Requested histogram bin width (seconds).
    smooth_sigma_s:
        Gaussian smoothing sigma (seconds).
    max_bins:
        Optional cap on the number of bins (widens the effective bin width).

    Returns
    -------
    centers_s : numpy.ndarray
        Bin-centre times (seconds).
    rate : numpy.ndarray
        Firing rate in Hz at each bin centre.
    """
    from scipy.ndimage import gaussian_filter1d

    if t1_s <= t0_s:
        return np.array([]), np.array([])
    n_bins = max(1, int(round((t1_s - t0_s) / bin_s)))
    if max_bins is not None and n_bins > max_bins:
        n_bins = max_bins
    edges = np.linspace(t0_s, t1_s, n_bins + 1)
    width = (t1_s - t0_s) / n_bins
    spikes = np.asarray(spike_exp_s, dtype="float64")
    spikes = spikes[(spikes >= t0_s) & (spikes <= t1_s)]
    counts, _ = np.histogram(spikes, bins=edges)
    rate = counts / width
    sigma_bins = max(1e-6, smooth_sigma_s / width)
    rate = gaussian_filter1d(rate, sigma_bins, mode="nearest")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, rate


def _stride(n: int, max_points: int) -> int:
    """Stride to downsample *n* points to about *max_points*."""
    return max(1, int(np.ceil(n / max_points)))


# ---------------------------------------------------------------------------
# Precomputed firing-rate cache (per units file)
# ---------------------------------------------------------------------------
_CENTERS_COL = "__centers_s__"


def cache_paths(units_path: str | Path) -> tuple[Path, Path]:
    """(parquet, json) cache paths for a units file."""
    p = Path(units_path)
    return (p.with_suffix(p.suffix + ".firing_rate.parquet"),
            p.with_suffix(p.suffix + ".firing_rate.json"))


def _meta(units_path, bin_s, smooth_s, max_points, uids) -> dict:
    up = Path(units_path)
    return {
        "version": CACHE_VERSION,
        "units_file": up.name,
        "units_mtime": int(up.stat().st_mtime) if up.exists() else 0,
        "bin_s": float(bin_s),
        "smooth_sigma_s": float(smooth_s),
        "max_points": int(max_points),
        "n_units": len(uids),
    }


def _raw_spike_range(units: dict, uids: list) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for uid in uids:
        sp = np.asarray(units[uid]["spike_times"], dtype="float64")
        if sp.size:
            lo = min(lo, float(sp[0] if sp[0] <= sp[-1] else sp.min()))
            hi = max(hi, float(sp.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def compute_firing_rates(
    units: dict, uids: list, bin_s: float, smooth_s: float, max_points: int,
    progress=None,
) -> tuple[np.ndarray, dict]:
    """Downsampled firing rate for every unit over the raw spike range.

    Returns ``(centers_s, {str(unit_id): rate_float32})`` where ``centers_s`` is
    the shared (downsampled) bin-centre grid in *raw* spike seconds.
    """
    raw_t0, raw_t1 = _raw_spike_range(units, uids)
    centers_ds: np.ndarray | None = None
    stride = None
    rates: dict[str, np.ndarray] = {}
    for i, uid in enumerate(uids):
        spikes = np.asarray(units[uid]["spike_times"], dtype="float64")
        centers, rate = instantaneous_firing_rate(spikes, raw_t0, raw_t1, bin_s, smooth_s)
        if stride is None:
            stride = _stride(len(centers), max_points)
            centers_ds = centers[::stride].astype("float64")
        rates[str(uid)] = rate[::stride].astype("float32")
        if progress is not None:
            progress(i + 1, len(uids))
    if centers_ds is None:
        centers_ds = np.array([], dtype="float64")
    return centers_ds, rates


def save_firing_rates(units_path, centers_s, rates: dict, meta: dict) -> None:
    import pandas as pd

    parq, js = cache_paths(units_path)
    data = {_CENTERS_COL: np.asarray(centers_s, dtype="float64")}
    for uid, r in rates.items():
        data[str(uid)] = np.asarray(r, dtype="float32")
    pd.DataFrame(data).to_parquet(parq, index=False)
    js.write_text(json.dumps(meta))


def load_firing_rates(units_path) -> dict | None:
    """Load a cached firing-rate file, or ``None`` if absent/unreadable."""
    import pandas as pd

    parq, js = cache_paths(units_path)
    if not parq.exists() or not js.exists():
        return None
    try:
        meta = json.loads(js.read_text())
        df = pd.read_parquet(parq)
    except Exception:  # noqa: BLE001 - a corrupt cache -> recompute
        return None
    centers = df[_CENTERS_COL].to_numpy(dtype="float64")
    rates = {c: df[c].to_numpy(dtype="float32")
             for c in df.columns if c != _CENTERS_COL}
    return {"centers_s": centers, "rates": rates, "meta": meta}


def ensure_firing_rates(
    units: dict, units_path, bin_s: float, smooth_s: float, max_points: int,
    progress=None,
) -> dict:
    """Return the firing-rate cache for a units file, computing it if needed.

    Reuses an on-disk cache when its parameters (units-file mtime, ``bin_s``,
    ``smooth_sigma_s``, ``max_points``, version) match; otherwise recomputes and
    saves. The returned dict has ``centers_s`` (raw seconds), ``rates`` (by
    ``str(unit_id)``) and ``meta``.
    """
    uids = sorted(units.keys())
    want = _meta(units_path, bin_s, smooth_s, max_points, uids)
    cached = load_firing_rates(units_path)
    if cached is not None and cached.get("meta") == want:
        return cached
    centers_s, rates = compute_firing_rates(
        units, uids, bin_s, smooth_s, max_points, progress=progress
    )
    try:
        save_firing_rates(units_path, centers_s, rates, want)
    except Exception:  # noqa: BLE001 - still usable in-memory if the write fails
        pass
    return {"centers_s": centers_s, "rates": rates, "meta": want}
