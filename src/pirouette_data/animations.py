"""Spike-raster "Powers of Ten" zoom animation.

Renders a manually-curated spike raster (X = time, Y = each unit's true depth,
each unit its own colour) and smoothly zooms OUT from a 10 ms window to the full
~36 h recording -- like the *Powers of Ten* film. The X-axis units switch
ms -> s -> min -> hours as it widens, a dynamic scale bar switches through
1 ms / 1 s / 1 min / 1 hour / 10 hours, and a closing card reports how many orders
of magnitude in time were spanned.

The units file (a ``good_units.pkl``: ``{unit_id: {"spike_times", "depth", ...}}``)
must provide a ``depth`` per unit; the deepest units are drawn at the bottom (flip
with ``invert_depth``). Build an animation with :func:`make_animation`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# Glasbey 1024-colour categorical palette bundled in pirouette_data/assets/.
# This is the same npz that dartsort ships; bundling it here means the
# animation code has no runtime dependency on the dartsort package.
# Falls back to the dartsort package if the bundled copy is somehow absent,
# and ultimately to colorcet via unit_colors() as a last resort.
_dartsort_glasbey1024: "np.ndarray | None" = None
try:
    from importlib.resources import files as _importlib_files
    with np.load(
        str(_importlib_files("pirouette_data.assets").joinpath("glasbey1024.npz"))
    ) as _gb_npz:
        _dartsort_glasbey1024 = np.asarray(_gb_npz["glasbey1024"], dtype="float32")
except Exception:
    # Fallback: load from dartsort if installed (e.g. developer environment).
    try:
        with np.load(
            str(_importlib_files("dartsort.pretrained").joinpath("glasbey1024.npz"))
        ) as _gb_npz:
            _dartsort_glasbey1024 = np.asarray(_gb_npz["glasbey1024"], dtype="float32")
    except Exception:
        try:
            from dartsort.vis.colors import glasbey1024 as _gb  # type: ignore
            _dartsort_glasbey1024 = np.asarray(_gb, dtype="float32")
        except Exception:
            pass

# Pre-compute JCh (CIECAM02: J = lightness, C = chroma, h = hue) for every
# colour in the palette so the muted-range filter mirrors the glasbey library's
# lightness_bounds / chroma_bounds API — perceptually principled rather than
# an ad-hoc HSV threshold.  Done once at import time; if colorspacious is
# absent the draw block falls back to the HSV filter automatically.
_dartsort_glasbey1024_jch: "np.ndarray | None" = None
try:
    from colorspacious import cspace_convert as _cspace_convert  # type: ignore
    if _dartsort_glasbey1024 is not None:
        _rgb_01 = np.clip(_dartsort_glasbey1024[:, :3], 0.0, 1.0).astype("float64")
        _dartsort_glasbey1024_jch = _cspace_convert(
            _rgb_01, "sRGB1", "JCh"
        ).astype("float32")
    del _cspace_convert
except Exception:
    pass

# Scale-bar steps (seconds, label): the bar shows the largest of these that fits
# in the current view, so it "switches" as the zoom widens.
SCALE_STEPS: list[tuple[float, str]] = [
    (0.001,    "1 ms"),
    (0.1,      "100 ms"),
    (1.0,      "1 s"),
    (60.0,     "1 min"),
    (3600.0,   "1 hr"),
    (36000.0,  "10 hrs"),
    (360000.0, "100 hrs"),
]


def load_units_with_depth(path: str | Path) -> dict:
    """Load a units pickle, requiring ``spike_times`` and ``depth`` on every unit.

    Raises
    ------
    ValueError
        If any unit is missing ``depth`` (needed to order rows by probe depth) or
        ``spike_times``. The message lists how many units are affected.
    """
    with open(path, "rb") as f:
        units = pickle.load(f)
    if not isinstance(units, dict) or not units:
        raise ValueError(f"units file is empty or not a dict: {path}")
    no_depth = [u for u, v in units.items() if not isinstance(v, dict) or "depth" not in v]
    if no_depth:
        raise ValueError(
            f"{len(no_depth)} of {len(units)} units lack a 'depth' field. Depth is "
            "required to order units by their position on the probe -- add a "
            "'depth' entry to each unit in the units file."
        )
    no_spikes = [u for u, v in units.items() if "spike_times" not in v]
    if no_spikes:
        raise ValueError(f"{len(no_spikes)} units lack 'spike_times'.")
    return units


def prepare_units(units: dict, invert_depth: bool = False) -> list[dict]:
    """Per-unit records ordered by depth, with sorted spike times.

    Returns a list of ``{"id", "depth", "cidx", "times", "spike_depths"}`` ordered
    by depth.  ``cidx`` is the colour index in depth order.

    ``spike_depths`` is a per-spike float64 array aligned to ``times``.  If the
    unit dict contains a ``"spike_depths"`` key whose length matches
    ``spike_times``, those values are used (giving each spike its own y position).
    Otherwise every spike is plotted at the unit's scalar ``depth``.
    """
    ids = sorted(units.keys(), key=lambda u: float(units[u]["depth"]), reverse=invert_depth)
    out = []
    for cidx, u in enumerate(ids):
        raw_t = np.asarray(units[u]["spike_times"], dtype="float64")
        sort_idx = np.argsort(raw_t)
        t_sorted = raw_t[sort_idx]

        raw_sd = units[u].get("spike_depths")
        if raw_sd is not None and len(raw_sd) == len(raw_t):
            sd = np.asarray(raw_sd, dtype="float64")[sort_idx]
        else:
            sd = np.full(len(t_sorted), float(units[u]["depth"]), dtype="float64")

        out.append({"id": u, "depth": float(units[u]["depth"]), "cidx": cidx,
                    "times": t_sorted, "spike_depths": sd})
    return out


def unit_colors(n: int, palette: str = "glasbey") -> np.ndarray:
    """``n`` RGBA colours (one per unit, in depth order) for a named palette.

    ``"glasbey"`` / ``"dartsort"`` (default) uses the Glasbey optimal categorical
    palette from colorcet (256 maximally-distinct colours), the same algorithm
    dartsort uses for per-unit label colouring.  Falls back to ``"vivid"`` if
    colorcet is not installed.

    ``"muted"`` is a desaturated rainbow; ``"vivid"`` / ``"rainbow"`` is the full
    gist_rainbow.  Any other value is treated as a Matplotlib colormap name.
    """
    import matplotlib
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    p = (palette or "glasbey").lower()
    if p in ("glasbey", "dartsort"):
        # Glasbey optimal categorical palette — maximally distinct colours, the
        # same algorithm dartsort uses for per-unit label colouring (glasbey1024).
        # colorcet.b_glasbey_bw gives 256 RGBA-safe hex strings; we cycle if n > 256.
        try:
            import colorcet as cc
            from matplotlib.colors import to_rgba_array
            src = cc.b_glasbey_bw            # list of 256 hex strings
            palette_hex = [src[i % len(src)] for i in range(n)]
            return to_rgba_array(palette_hex).astype("float32")
        except ImportError:
            pass  # fall through to vivid
        return matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 0.90, n))
    if p == "muted":
        base = matplotlib.colormaps["gist_rainbow"].resampled(n)(np.arange(n))[:, :3]
        hsv = rgb_to_hsv(base)
        hsv[:, 1] *= 0.5
        hsv[:, 2] = 0.55 + 0.30 * hsv[:, 2]
        return np.column_stack([hsv_to_rgb(hsv), np.ones(n)])
    # "vivid": gist_rainbow sampled 0→0.90 — shallow units red-pink, deep purple.
    if p == "vivid":
        return matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 0.90, n))
    if p == "rainbow":
        p = "gist_rainbow"
    try:
        cmap = matplotlib.colormaps[p]
    except KeyError:
        cmap = matplotlib.colormaps["gist_rainbow"]
    return cmap.resampled(n)(np.arange(n))


def window_label(width_s: float) -> str:
    """Human-readable label for a window width (e.g. ``"70 s"``, ``"36 hours"``)."""
    if width_s < 1:
        return f"{width_s * 1000:.0f} ms"
    if width_s < 100:
        return f"{width_s:.0f} s"
    if width_s < 6000:
        return f"{width_s / 60:.0f} min"
    return f"{width_s / 3600:.0f} hours"


def x_axis_unit(
    width_s: float,
    thresholds: tuple[float, float, float] = (1.0, 100.0, 6000.0),
) -> tuple[float, str]:
    """(seconds-per-unit, unit name) for the x-axis at a given window width.

    The axis label switches ms → s → min → hours at the three threshold window
    widths (in seconds) given by *thresholds*.  Defaults: switch to s at 1 s,
    to min at 100 s, to hours at 6000 s (100 min).
    """
    t_ms_s, t_s_min, t_min_hr = thresholds
    if width_s < t_ms_s:
        return 0.001, "ms"
    if width_s < t_s_min:
        return 1.0, "s"
    if width_s < t_min_hr:
        return 60.0, "min"
    return 3600.0, "hours"


def _nice_step(raw: float) -> float:
    """Round *raw* up to the next 1 / 2 / 5 x 10**k (for tidy tick spacing)."""
    if raw <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if m * mag >= raw:
            return m * mag
    return 10.0 * mag


def pick_scale_bar(window_s: float) -> tuple[float, str]:
    """Largest ``SCALE_STEPS`` entry that fits within *window_s* (>= smallest)."""
    chosen = SCALE_STEPS[0]
    for dur, label in SCALE_STEPS:
        if dur <= window_s:
            chosen = (dur, label)
    return chosen


def orders_of_magnitude(window_start_s: float, window_end_s: float) -> float:
    """Number of base-10 orders of magnitude between two window widths."""
    if window_start_s <= 0 or window_end_s <= 0:
        return 0.0
    return float(np.log10(window_end_s / window_start_s))


def _window_bounds(center_s: float, width_s: float, lo_limit: float,
                   hi_limit: float) -> tuple[float, float]:
    """[center-w/2, center+w/2], clamped to [lo_limit, hi_limit] keeping the width."""
    half = width_s / 2.0
    lo, hi = center_s - half, center_s + half
    if lo < lo_limit:
        lo, hi = lo_limit, min(hi_limit, lo_limit + width_s)
    if hi > hi_limit:
        hi, lo = hi_limit, max(lo_limit, hi_limit - width_s)
    return lo, hi


def _window_bounds_left(anchor_s: float, width_s: float, lo_limit: float,
                        hi_limit: float) -> tuple[float, float]:
    """[anchor_s, anchor_s+width_s], clamped to recording bounds.

    The left edge is pinned at *anchor_s*; the window grows rightward as
    *width_s* increases.
    """
    lo = max(anchor_s, lo_limit)
    hi = min(lo + width_s, hi_limit)
    return lo, hi


def _window_bounds_center_right(center_s: float, width_s: float,
                                lo_limit: float, max_hi: float,
                                ) -> tuple[float, float]:
    """Centered zoom until the left edge reaches *lo_limit*, then expands right.

    For small windows the view is symmetrically centred on *center_s*.  Once
    the window is wide enough that the left edge would go below *lo_limit* the
    left edge is pinned there and the window continues growing rightward only —
    creating a seamless, non-jarring transition from centred to anchor-left.
    The right edge can grow up to *max_hi* (the full animation end, which may
    extend beyond the actual recording data).
    """
    lo = max(lo_limit, center_s - width_s / 2.0)
    hi = min(lo + width_s, max_hi)
    return lo, hi


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve *x* with a Gaussian kernel of std-dev *sigma* (in samples)."""
    r = int(np.ceil(3.5 * sigma))
    k = np.arange(-r, r + 1, dtype="float64")
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def frame_points(per_unit: list[dict], lo: float, hi: float, cap: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spikes within ``[lo, hi]`` across units, subsampled to ~*cap* total.

    Returns ``(times, spike_depths, colour_index)``.  Each spike's y position
    comes from ``pu["spike_depths"]`` — a per-spike array that equals the unit's
    mean ``depth`` when no per-spike estimates are available, or the individual
    depth estimate when they are.

    Mirrors the dartsort ``scatter_time_vs_depth`` approach: high-firing periods
    contribute proportionally more spikes to the drawn set, so temporal density
    variations appear as vertical striations at wide zoom.

    **Performance note**: subsampling is done *per unit* (proportional to each
    unit's count in the window) rather than via a single global concatenation.
    A global shuffle of 240 units × ~1.8 M spikes/unit = 432 M elements would
    allocate ~3.5 GB and take ~0.8 s per frame; per-unit proportional sampling
    allocates only the drawn subset (~500 k) and is ~15× faster.
    """
    # Pass 1: count per unit and compute total (searchsorted only — no alloc).
    slices: list[tuple[int, int]] = []
    total = 0
    for pu in per_unit:
        a = int(np.searchsorted(pu["times"], lo, "left"))
        b = int(np.searchsorted(pu["times"], hi, "right"))
        slices.append((a, b))
        total += b - a

    if total == 0:
        empty = np.array([], dtype="float64")
        return empty, empty, np.array([], dtype="int64")

    # Pass 2: collect (no subsampling) or per-unit proportional random sample.
    # Using a single seeded RNG shared across units keeps the global density
    # ratio correct and is reproducible frame-to-frame.
    xs, ys, cs = [], [], []
    if total <= cap:
        for pu, (a, b) in zip(per_unit, slices):
            if b > a:
                xs.append(pu["times"][a:b])
                ys.append(pu["spike_depths"][a:b])
                cs.append(np.full(b - a, pu["cidx"], dtype="int64"))
    else:
        frac = cap / total
        rng  = np.random.default_rng(seed=0)
        for pu, (a, b) in zip(per_unit, slices):
            m = b - a
            if m == 0:
                continue
            keep = max(1, int(round(m * frac)))
            # rng.choice with replace=False and keep << m uses an O(keep) hash
            # algorithm — fast even when m is in the millions.
            idx = rng.choice(m, size=keep, replace=False)
            idx.sort()   # restore temporal order
            xs.append(pu["times"][a:b][idx])
            ys.append(pu["spike_depths"][a:b][idx])
            cs.append(np.full(keep, pu["cidx"], dtype="int64"))

    if not xs:
        empty = np.array([], dtype="float64")
        return empty, empty, np.array([], dtype="int64")
    # Final concat is at most cap elements (~500 k × 24 B = ~12 MB) — fast.
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_c = np.concatenate(cs)
    # No global shuffle: the pixel-image path (wide zoom) handles draw order by
    # writing units depth-first into the buffer (last write wins at duplicate
    # pixels — a negligible bias).  The scatter path (narrow zoom) has so few
    # spikes in the window that order is irrelevant.  The previous permutation
    # added ~15 ms/frame for no visible benefit; it is removed here.
    return all_x, all_y, all_c


def frame_points_binned(
    per_unit: list[dict],
    lo: float,
    hi: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At most one tick per (unit × time bin) — density-invariant raster.

    Divides ``[lo, hi]`` into *n_bins* equal-width buckets.  For each unit,
    only the *first* spike found in each occupied bucket is emitted.  The
    output size is bounded by ``n_bins × n_units`` regardless of firing rate
    or window width, so the apparent raster density stays visually consistent
    as the zoom changes — analogous to dartsort's ``scatter_time_vs_depth``
    decimation strategy.

    Unlike :func:`frame_points`, no random cap is applied: presence is
    determined from the raw spike train, so quiet periods are never
    accidentally shown as active due to unlucky subsampling.
    """
    if n_bins < 1:
        n_bins = 1
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    for pu in per_unit:
        a = int(np.searchsorted(pu["times"], lo, "left"))
        b = int(np.searchsorted(pu["times"], hi, "right"))
        if b <= a:
            continue
        t = pu["times"][a:b]
        d = pu["spike_depths"][a:b]
        # Bin index for each spike: 0 … n_bins-1.
        # searchsorted on the *interior* edges (bin_edges[1:-1]) puts spikes
        # in [edges[k], edges[k+1]) into bucket k.
        bidx = np.searchsorted(bin_edges[1:-1], t)
        # One representative spike per occupied bucket (the first one found,
        # which is also the earliest because times are sorted).
        unique_b, first_idx = np.unique(bidx, return_index=True)
        bin_cx = 0.5 * (bin_edges[unique_b] + bin_edges[unique_b + 1])
        xs.append(bin_cx)
        ys.append(d[first_idx])
        cs.append(np.full(len(unique_b), pu["cidx"], dtype="int64"))
    if not xs:
        empty = np.array([], dtype="float64")
        return empty, empty, np.array([], dtype="int64")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def frame_points_fixed_k(
    per_unit: list[dict],
    lo: float,
    hi: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly sub-sample each unit to at most *k* ticks in [lo, hi].

    For a unit with *m* spikes in the window, retains the spike at index
    ``round(i * (m-1) / (k-1))`` for *i* in ``0 … k-1`` (i.e. evenly-spaced
    indices covering the full window, first and last spike always included).
    When ``m <= k`` all spikes are kept.

    This gives a **density-invariant** display: regardless of firing rate, each
    unit contributes at most *k* ticks.  With ``k=60`` across a ~1240-px pixel
    buffer, fill per unit ≈ 5 %.  Even if 10 units share the same y-pixel row,
    the combined fill stays ≈ 40 % — clearly dashed rather than solid.

    Unlike :func:`frame_points_binned`, the sub-sampled spikes retain their
    *exact* spike times (not bin-center positions), so temporal clustering
    (bursts, silences) is faithfully preserved.
    """
    if k < 1:
        k = 1
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    for pu in per_unit:
        a = int(np.searchsorted(pu["times"], lo, "left"))
        b = int(np.searchsorted(pu["times"], hi, "right"))
        m = b - a
        if m == 0:
            continue
        if m <= k:
            t = pu["times"][a:b]
            d = pu["spike_depths"][a:b]
        else:
            idx = np.round(np.linspace(0, m - 1, k)).astype(np.int64)
            t = pu["times"][a:b][idx]
            d = pu["spike_depths"][a:b][idx]
        xs.append(t)
        ys.append(d)
        cs.append(np.full(len(t), pu["cidx"], dtype="int64"))
    if not xs:
        empty = np.array([], dtype="float64")
        return empty, empty, np.array([], dtype="int64")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def _build_widths(
    window_start_s: float,
    window_end_s: float,
    n_base: int,
    easing: str = "ease-in-out",
    milestone_dwell_n: int = 0,
    slow_zones: list | None = None,
) -> np.ndarray:
    """Per-frame window-width sequence, with optional easing, milestone dwells, and slow zones.

    Parameters
    ----------
    easing:
        ``"ease-in-out"`` (default) — smoothstep; slows the start and end of the
        zoom so it doesn't feel mechanical. ``"linear"`` keeps constant log-speed.
    milestone_dwell_n:
        Extra frames to hold at each scale-bar milestone (where the scale-bar
        label switches). Gives viewers a moment to orient at each decade before
        the next zoom begins. Set to ``0`` to disable.
    slow_zones:
        List of zone tuples controlling per-region animation speed.  Two forms:

        * ``(lo_s, hi_s, factor)`` — constant weight: the zone plays at
          ``1/factor`` of the baseline log-speed (factor > 1 → slower).
        * ``(lo_s, hi_s, factor_start, factor_end)`` — linear ramp: the weight
          interpolates linearly (in log-width space) from ``factor_start`` at
          ``lo_s`` to ``factor_end`` at ``hi_s``, giving a smooth acceleration
          or deceleration across the zone.
    """
    log_start = np.log10(max(window_start_s, 1e-9))
    log_end   = np.log10(max(window_end_s,   1e-9))

    if slow_zones:
        # Build a piecewise weight function in log-width space via cumulative
        # integration, then invert to map uniform animation-time → log-width.
        # High-weight zones get proportionally more animation frames (slower).
        _N = 8192
        lw_fine = np.linspace(log_start, log_end, _N)
        w_fine  = np.ones(_N, dtype="float64")
        for zone in slow_zones:
            lo_s, hi_s = zone[0], zone[1]
            factor_start = float(zone[2])
            factor_end   = float(zone[3]) if len(zone) == 4 else factor_start
            lo_l = np.log10(max(lo_s, window_start_s))
            hi_l = np.log10(min(hi_s, window_end_s))
            if lo_l < hi_l:
                mask = (lw_fine >= lo_l) & (lw_fine <= hi_l)
                if factor_start == factor_end:
                    w_fine[mask] = factor_start
                else:
                    # Linear ramp in log-width space → smooth speed change.
                    t_zone = (lw_fine[mask] - lo_l) / (hi_l - lo_l)
                    w_fine[mask] = factor_start + (factor_end - factor_start) * t_zone
        cum = np.cumsum(w_fine)
        cum -= cum[0]
        cum /= cum[-1]          # CDF over log-width → maps anim-t to log-width

        # Apply easing to animation time, then look up log-widths via the CDF.
        t = np.linspace(0.0, 1.0, max(2, n_base))
        if easing == "ease-in-out":
            t = t * t * (3.0 - 2.0 * t)
        elif easing == "ease-in":
            t = t * t
        elif easing == "ease-out":
            t = 1.0 - (1.0 - t) ** 2
        lw = np.interp(t, cum, lw_fine)
        widths = 10.0 ** lw
    else:
        t = np.linspace(0.0, 1.0, max(2, n_base))
        if easing == "ease-in-out":
            t = t * t * (3.0 - 2.0 * t)          # smoothstep
        elif easing == "ease-in":
            t = t * t
        elif easing == "ease-out":
            t = 1.0 - (1.0 - t) ** 2
        # else "linear": keep t as-is
        widths = window_start_s * (window_end_s / window_start_s) ** t

    if milestone_dwell_n <= 0:
        return widths

    # Insert dwell frames at each scale-bar milestone that falls inside the zoom range.
    # IMPORTANT: dwell must be inserted BEFORE the frame that first crosses the
    # milestone, not after it.  If we append w first, the sequence becomes
    #   …, 99 ms, 101 ms, [100 ms × dwell], 102 ms, …
    # which causes the visible "extends then snaps back" blip.  Inserting the
    # dwell before w gives the correct monotone sequence:
    #   …, 99 ms, [100 ms × dwell], 101 ms, …
    milestones = sorted(s for s, _ in SCALE_STEPS if window_start_s < s < window_end_s)
    result: list[float] = []
    prev = float(widths[0])
    for w in widths:
        for ms in milestones:
            if prev < ms <= w:
                result.extend([ms] * milestone_dwell_n)
        result.append(float(w))
        prev = float(w)
    return np.array(result)


def make_animation(
    units_path: str | Path,
    out_path: str | Path,
    center_s: float | None = None,
    window_start_s: float = 0.001,
    window_end_s: float | None = None,
    duration_s: float = 20.0,
    hold_s: float = 3.0,
    fps: int = 30,
    cap: int = 500_000,
    dpi: int = 120,
    invert_depth: bool = False,
    palette: str = "glasbey",
    easing: str = "ease-in-out",
    milestone_dwell_s: float = 0.0,
    axis_unit_ms_to_s: float = 1.0,
    axis_unit_s_to_min: float = 100.0,
    axis_unit_min_to_hr: float = 6000.0,
    zoom_mode: str = "anchor-left",
    anchor_s: float | None = None,
    rate_bin_s: float = 1800.0,
    t0_pst_s: float = 0.0,
    seed: int = 0,
    title: str = "Manually curated spike output",
    raster_mode: str = "hybrid",
    charlie_max_spikes: int = 500_000,
    depth_lo_um: float | None = None,
    depth_hi_um: float | None = None,
    show_rate: bool = False,
    slow_zones: list[tuple[float, float, float]] | None = None,
    charlie_jitter_density: str = "none",
    show_probe_map: bool = False,
) -> Path:
    """Render + save the Powers-of-Ten spike-raster zoom animation.

    Parameters
    ----------
    units_path, out_path:
        Units pickle (needs per-unit ``depth``) and the output ``.mp4`` / ``.gif``.
    center_s:
        Time the zoom stays centred on (default: middle of the recording).
    window_start_s, window_end_s:
        Initial (1 ms) and final window widths (default final: full recording span).
    duration_s, hold_s, fps:
        Zoom-out length, closing-card hold, and frame rate.
    cap:
        Max spikes drawn per frame (subsampled; a wide view is sub-pixel dense).
    invert_depth:
        By default the deepest units are at the BOTTOM; set this to put them at
        the top.
    palette:
        Colour palette (``"muted"`` default; ``"rainbow"``, or any Matplotlib
        colormap name). See :func:`unit_colors`.
    easing:
        ``"ease-in-out"`` (default) slows the zoom at the start and end so it
        feels intentional rather than mechanical. ``"linear"`` gives constant
        log-speed (original behaviour). Also accepts ``"ease-in"`` / ``"ease-out"``.
    raster_mode:
        ``"hybrid"`` (default) — per-unit colour ticks blending into a coloured
        density heatmap at wide zoom.  ``"charlie"`` — dartsort-style: all spikes
        globally subsampled, coloured by depth via glasbey, drawn amplitude-order
        (high-amp on top), vertical ticks at narrow zoom that smear into a
        gaussian depth-scattered scatter plot as the window grows.
    charlie_max_spikes:
        Global spike count cap for ``raster_mode="charlie"`` (default 500 000).
        Spikes are chosen with a fixed random seed so renders are reproducible.
    charlie_jitter_density:
        How to compute the local density that scales each spike's Gaussian
        y-jitter in ``"charlie"`` mode.  ``"per-unit"`` (default) — each unit
        independently bins its own spikes across the visible window; a quiet unit
        keeps near-zero spread even when other units are bursting.  ``"global"``
        — all spikes from all units are pooled into one density estimate; spread
        tracks the population firing rate at each moment.  ``"none"`` — jitter
        is applied uniformly at full sigma regardless of local density, identical
        to the behaviour before adaptive jitter was introduced.
    show_rate:
        Show the population-rate panel below the raster (default ``False``).
        When ``False`` the raster is taller and the layout has 3 rows.
    slow_zones:
        List of ``(lo_s, hi_s, factor)`` triples that slow down specific
        timescale regions.  Defaults to ``[(3600, 36_000, 2.0),
        (36_000, 360_000, 4.0)]`` — half-speed from 1 hr → 10 hr and
        quarter-speed from 10 hr → 100 hr.  Pass ``[]`` for uniform speed.
    milestone_dwell_s:
        Seconds to hold at each scale-bar milestone (where the bar label
        switches: 1 ms → 100 ms → 1 s → … → 100 hrs). Default 0.5 s. Set to
        ``0`` to disable.
    axis_unit_ms_to_s, axis_unit_s_to_min, axis_unit_min_to_hr:
        Window widths (in seconds) at which the x-axis tick labels switch
        units.  Defaults: 1 s (ms→s), 100 s (s→min), 6000 s (min→hr).
        Tune these if labels feel too crowded or too sparse at a given zoom
        level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.transforms import blended_transform_factory

    # Colours follow ascending depth; top/bottom orientation is set on the axis.
    _raw_units = load_units_with_depth(units_path)
    per_unit = prepare_units(_raw_units)
    n_units = len(per_unit)
    _ = seed  # jitter is deterministic in x (see draw); seed kept for API stability

    t_lo = min(float(pu["times"][0]) for pu in per_unit if len(pu["times"]))
    t_hi = max(float(pu["times"][-1]) for pu in per_unit if len(pu["times"]))
    if center_s is None:
        center_s = 0.5 * (t_lo + t_hi)
    if window_end_s is None:
        window_end_s = max(window_start_s * 10, t_hi - t_lo)

    colors = unit_colors(n_units, palette)
    # Use the full spike-depth range for the y-axis so per-spike estimates that
    # stray slightly outside the unit-mean range are not clipped.
    d_lo = float(min(pu["spike_depths"].min() for pu in per_unit))
    d_hi = float(max(pu["spike_depths"].max() for pu in per_unit))
    if depth_lo_um is not None:
        d_lo = float(depth_lo_um)
    if depth_hi_um is not None:
        d_hi = float(depth_hi_um)
    d_pad = 0.0   # use exact depth range with no margin

    # Match dartsort's max_spikes_plot=500k: no subsampling for windows < ~350 s
    # (500k / (240 units × 6 Hz avg)).  Global random sampling (frame_points) means
    # dense periods contribute proportionally more points, revealing striations.
    _scatter_cap = max(cap, 500_000)

    # ── Per-unit colored density (replaces grayscale shadow heatmap) ─────────
    # For each unit, precompute a 1-D firing-rate density across _UNIT_HMAP_T
    # equal-width time bins (≈18 s/bin for a 100-hr recording).  Power-law
    # normalisation (exponent 3) makes quiet periods near-white and bursts
    # vivid.  draw() uses this to paint a coloured density background in each
    # unit's pixel row before stamping the sparse fixed-K ticks on top — so
    # colour identity is maintained at every zoom level and the tick→density
    # transition is seamless (same hue, only density vs. presence changes).
    _UNIT_HMAP_T = 20_000          # ≈18 s/bin @ 100 hr; 17 bins at 5-min zoom
    _unit_dens = np.zeros((n_units, _UNIT_HMAP_T), dtype="float32")
    for _ui, _hpu in enumerate(per_unit):
        _uht = np.clip(
            ((_hpu["times"] - t_lo) / (t_hi - t_lo) * _UNIT_HMAP_T).astype(np.int32),
            0, _UNIT_HMAP_T - 1,
        )
        _row = np.bincount(_uht, minlength=_UNIT_HMAP_T).astype("float32")
        _mx  = max(_row.max(), 1.0)
        _unit_dens[_ui] = (_row / _mx) ** 3.0   # power-law contrast
    del _row
    # Mean depth per unit — used every frame to map units to y-pixels.
    _unit_depths = np.array(
        [float(np.median(pu["spike_depths"])) for pu in per_unit], dtype="float32"
    )

    # ── "charlie" mode precompute ─────────────────────────────────────────────
    # Mirrors the dartsort scatter_time_vs_depth() aesthetic:
    #   • Random priority subsampling (not K-even) — bursts have more spikes so
    #     they are over-represented → temporal structure (vertical striations)
    #     is preserved at every zoom level.
    #   • Each spike has a precomputed random priority in [0, 1).  At render time
    #     we take the max_spikes highest-priority spikes in the visible window via
    #     np.argpartition (O(N), stable across frames → no flicker).
    #   • Spikes sorted by amplitude before drawing: low-amp first, high-amp last
    #     (on top) → well-isolated single units pop off the background.
    #   • Colour: dartsort's glasbey1024, palette indices scrambled so that
    #     depth-adjacent units get perceptually distinct hues (no gradient).
    if raster_mode == "charlie":
        _ch_a_list = []
        for pu in per_unit:
            _n_pu = len(pu["times"])
            # Try several amplitude key names; fall back to uniform 1.0
            _raw_u = _raw_units[pu["id"]]
            _raw_amp = (
                _raw_u.get("amplitudes")
                or _raw_u.get("denoised_ptp_amplitudes")
                or _raw_u.get("ptp_amplitudes")
            )
            if _raw_amp is not None and len(_raw_amp) == len(_raw_u["spike_times"]):
                # Re-sort to match the time-sorted order used in prepare_units()
                _raw_sort = np.argsort(
                    np.asarray(_raw_u["spike_times"], dtype="float64"))
                _ch_a_list.append(
                    np.asarray(_raw_amp, dtype="float32")[_raw_sort])
            else:
                _ch_a_list.append(np.ones(_n_pu, dtype="float32"))

        # Per-unit amplitude arrays (time-sorted, matching per_unit[i]["times"]).
        _ch_pu_amps = list(_ch_a_list)

        # Deterministic per-unit random priorities and y-jitter (same seed → stable).
        _rng_pu = np.random.default_rng(seed + 2)
        _ch_pu_priority = [
            _rng_pu.random(len(pu["times"])).astype("float32")
            for pu in per_unit
        ]
        _ch_pu_jitter = [
            _rng_pu.standard_normal(len(pu["times"])).astype("float32")
            for pu in per_unit
        ]

        # Colour palette: muted entries from glasbey1024, selected in the
        # perceptual JCh colorspace (CIECAM02 J = lightness, C = chroma).
        # This mirrors the glasbey library's lightness_bounds / chroma_bounds
        # API — dark (J < 55) and chromatically present (C > 25) keeps
        # dark magenta, dark blue, dark red, dark teal while definitively
        # excluding lime green (J~75), light cyan (J~85), pale yellow (J~90).
        # Muted upper chroma cap (C < 75) softens neon/vivid outliers.
        # Random shuffle gives categorical variety across depth-adjacent units.
        _rng_color = np.random.default_rng(seed + 5)
        if _dartsort_glasbey1024 is not None:
            _rgb3 = _dartsort_glasbey1024[:, :3].astype("float32")
            if _dartsort_glasbey1024_jch is not None:
                # Perceptual muted filter: dark & moderately saturated.
                _J = _dartsort_glasbey1024_jch[:, 0]
                _C = _dartsort_glasbey1024_jch[:, 1]
                _dark_idx = np.where((_J < 55.0) & (_C > 25.0) & (_C < 75.0))[0]
                if len(_dark_idx) < n_units:          # relax chroma cap
                    _dark_idx = np.where((_J < 55.0) & (_C > 20.0))[0]
                if len(_dark_idx) < n_units:          # relax lightness
                    _dark_idx = np.where((_J < 65.0) & (_C > 15.0))[0]
                if len(_dark_idx) < n_units:          # last resort: darkest J
                    _dark_idx = np.argsort(_J)[:max(n_units, 50)]
            else:
                # Fallback: HSV brightness/saturation filter.
                _V  = _rgb3.max(axis=1)
                _mn = _rgb3.min(axis=1)
                _S  = np.where(_V > 1e-6, (_V - _mn) / _V, 0.0)
                _dark_idx = np.where((_V < 0.65) & (_S > 0.4))[0]
                if len(_dark_idx) < n_units:
                    _dark_idx = np.where(_V < 0.65)[0]
                if len(_dark_idx) < n_units:
                    _dark_idx = np.argsort(_V)[:max(n_units, 50)]
            _perm      = _rng_color.permutation(len(_dark_idx))
            _color_idx = _dark_idx[_perm[np.arange(n_units, dtype=np.int64) % len(_dark_idx)]]
            _ch_colors = _rgb3[_color_idx]            # (n_units, 3) float32
        else:
            _base = unit_colors(n_units, "glasbey")[:, :3].astype("float32")
            _V_b  = _base.max(axis=1)
            _mn_b = _base.min(axis=1)
            _S_b  = np.where(_V_b > 1e-6, (_V_b - _mn_b) / _V_b, 0.0)
            _dark_idx_b = np.where((_V_b < 0.65) & (_S_b > 0.4))[0]
            if len(_dark_idx_b) < n_units:
                _dark_idx_b = np.argsort(_V_b)[:max(n_units, 50)]
            _perm2     = _rng_color.permutation(len(_dark_idx_b))
            _ch_colors = _base[_dark_idx_b[_perm2[np.arange(n_units) % len(_dark_idx_b)]]]

        # Visual transition schedule (shared with draw()).
        # Two independent log-width schedules:
        #   _CH_LOG_TICK_HI    — tick height: 5 px at ≤ 1 s → 1 px at exactly 1 min
        #   _CH_LOG_JITTER_LO/HI — Gaussian y-spread: onset at 500ms window, max at 10 hr
        # Alpha is constant 0.9 at all zoom levels.
        _CH_JITTER_MAX    = 12.0             # max y-jitter sigma (px) at widest zoom
        _CH_LOG_TICK_HI   = np.log10(60.0)  # tick → 1 px at 1-min window (single-pixel at 1 min)
        _CH_LOG_JITTER_LO = np.log10(0.5)   # jitter onset at 500ms window
        _CH_LOG_JITTER_HI = np.log10(36000.0)  # jitter max at 10-hr window
    else:
        # Ensure these names are defined so the draw() closure always compiles.
        (_ch_pu_amps, _ch_pu_priority, _ch_pu_jitter, _ch_colors,
         _CH_JITTER_MAX, _CH_LOG_TICK_HI,
         _CH_LOG_JITTER_LO, _CH_LOG_JITTER_HI) = (
            None, None, None, None, 0.0, 1.0, 0.0, 1.0)

    # Zoom-mode bookkeeping.
    if anchor_s is None:
        anchor_s = t_lo + 70.0          # default: 70 s into the recording
    # Tick labels are relative to: the anchor (anchor-left) or zoom centre (center).
    _tick_ref = anchor_s if zoom_mode != "center" else center_s

    # Population firing rate — two resolutions computed in one pass:
    #   Fine (10 s bins, unsmoothed): used in draw() for windows >= _RATE_SWITCH_S.
    #     Adaptive Gaussian smoothing is applied per-frame so wide zoom shows the
    #     circadian envelope and narrow zoom shows hourly fluctuations.
    #   On-the-fly (< _RATE_SWITCH_S): raw spike data histogrammed per frame so
    #     individual spikes are visible down to the 1 ms window.
    _FINE_BIN_S    = 10.0         # pre-computed bin width (seconds)
    _RATE_SWITCH_S = 100.0        # window widths below this use on-the-fly histograms
    _rate_bins_fine = np.arange(0.0, t_hi + _FINE_BIN_S, _FINE_BIN_S)
    _rate_cnt_fine  = np.zeros(max(1, len(_rate_bins_fine) - 1), dtype="float64")
    for _pu in per_unit:
        _c, _ = np.histogram(_pu["times"], bins=_rate_bins_fine)
        _rate_cnt_fine += _c.astype("float64")
    _rate_t_fine = 0.5 * (_rate_bins_fine[:-1] + _rate_bins_fine[1:])
    _rate_peak   = max(float(_rate_cnt_fine.max()), 1.0)   # counts/10 s at peak
    _rate_n_fine = _rate_cnt_fine / _rate_peak              # normalised to [0, 1]

    n_base = max(2, int(round(duration_s * fps)))
    n_dwell = max(0, int(round(milestone_dwell_s * fps)))
    _slow = slow_zones if slow_zones is not None else [
        # 0 → 100 ms window: base speed (good as-is).
        (0.0,            0.1,   7.5,   7.5),
        # 100 ms → 1 s: 2× slower than base.
        (0.1,            1.0,  15.0,  15.0),
        # 1 s → 1 hr: ramp 30→60, same profile as before the 1hr mark.
        (1.0,        3_600.0,  30.0,  60.0),
        # 1 hr → 10 hr: smooth ramp 60→120 — continuously doubles the frame density
        # entering the 1–100 hr slow window; no speed discontinuity at either edge.
        (3_600.0,   36_000.0,  60.0, 120.0),
        # 10 → 100 hr: hold at 120 (2× the previous 60) so the 1–100 hr window as
        # a whole lasts ~2× longer than before (was ≈5.5 s, now ≈11 s).
        (36_000.0,  360_000.0, 120.0, 120.0),
        # 100 → 1332 hr: ramp smoothly from 120 back down to near-zero so the exit
        # from the slow window is as gradual as the entry.
        (360_000.0, 4_795_200.0, 120.0,  0.5),
    ]
    widths = _build_widths(window_start_s, window_end_s, n_base, easing, n_dwell,
                           slow_zones=_slow)
    n_zoom = len(widths)
    n_hold = max(1, int(round(hold_s * fps)))

    # ------------------------------------------------------------------ figure --
    # Layout: [probe_map |] raster | scale-bar strip | [optional rate] | time-of-day
    # When show_probe_map=True a narrow probe column is added to the left; the
    # probe and raster share a y-axis (depth) and the depth tick labels live on
    # the probe panel so the raster shows none.
    fig = plt.figure(figsize=(16, 9))
    _probe_wratios = [1.0, 8.0] if show_probe_map else [1]
    _n_cols = 2 if show_probe_map else 1
    _raster_col = 1 if show_probe_map else 0
    if show_rate:
        # 4 rows: raster taller; tod halved
        gs = fig.add_gridspec(
            4, _n_cols,
            height_ratios=[7.0, 0.28, 1.8, 0.325],
            width_ratios=_probe_wratios,
            hspace=0.07, wspace=0.0,
            left=0.08, right=0.97, top=0.97, bottom=0.10,
        )
        if show_probe_map:
            ax_probe  = fig.add_subplot(gs[0, 0])
            ax_raster = fig.add_subplot(gs[0, _raster_col], sharey=ax_probe)
        else:
            ax_probe  = None
            ax_raster = fig.add_subplot(gs[0, _raster_col])
        ax_scale  = fig.add_subplot(gs[1, _raster_col], sharex=ax_raster)
        ax_rate   = fig.add_subplot(gs[2, _raster_col], sharex=ax_raster)
        ax_tod    = fig.add_subplot(gs[3, _raster_col], sharex=ax_raster)
    else:
        # 3 rows: raster | thin scale-bar strip | tod.
        # Dedicated scale-bar strip (0.20 ratio, thinner than original 0.28) so the
        # bar never overlaps raster spike data.
        _hspace  = 0.0  if show_probe_map else 0.07
        _top_gs  = 0.99 if show_probe_map else 0.97
        _bot_gs  = 0.05 if show_probe_map else 0.10   # 5% gives room for "56 days" tick label
        gs = fig.add_gridspec(
            3, _n_cols,
            height_ratios=[8.5, 0.20, 0.325],
            width_ratios=_probe_wratios,
            hspace=_hspace, wspace=0.0,
            left=0.08, right=0.97, top=_top_gs, bottom=_bot_gs,
        )
        if show_probe_map:
            # Probe spans ALL 3 rows — physical bottom = tod bottom.
            ax_probe  = fig.add_subplot(gs[0:3, 0])
            ax_raster = fig.add_subplot(gs[0, _raster_col])
        else:
            ax_probe  = None
            ax_raster = fig.add_subplot(gs[0, _raster_col])
        ax_scale  = fig.add_subplot(gs[1, _raster_col], sharex=ax_raster)
        ax_rate   = None
        ax_tod    = fig.add_subplot(gs[2, _raster_col], sharex=ax_raster)
    fig.patch.set_facecolor("white")

    # --- Raster panel ---
    ax_raster.set_facecolor("white")
    # No separate scatter artist or grayscale heatmap — the pixel buffer is used
    # at every zoom level.  At narrow zoom it shows sparse fixed-K tick marks;
    # at wide zoom it paints a per-unit coloured density background and stamps
    # ticks on top.  Both layers share the same Glasbey unit colours so the
    # transition is visually seamless.
    if invert_depth:
        ax_raster.set_ylim(d_lo - d_pad, d_hi + d_pad)
    else:
        ax_raster.set_ylim(d_hi + d_pad, d_lo - d_pad)
    if show_probe_map and ax_probe is not None:
        # ax_probe is physically taller than ax_raster (it spans extra rows so its
        # bottom edge reaches the tod strip bottom).  Without compensation µm/px
        # would differ and the same unit depth would appear at different screen heights.
        # Fix: read actual figure-fraction heights via get_position() and extend
        # ax_probe's lower ylim by the extra fraction — independent of hardcoded ratios.
        _raster_h   = ax_raster.get_position().height
        _probe_h    = ax_probe.get_position().height
        _data_range = (d_hi + d_pad) - (d_lo - d_pad)
        _probe_d_lo = (d_lo - d_pad) - (_probe_h / _raster_h - 1.0) * _data_range
        if invert_depth:
            ax_probe.set_ylim(_probe_d_lo, d_hi + d_pad)
        else:
            ax_probe.set_ylim(d_hi + d_pad, _probe_d_lo)
    if show_probe_map:
        # Depth axis lives on the probe panel; raster has no y-axis at all.
        ax_raster.set_ylabel("")
        ax_raster.tick_params(left=False, labelleft=False)
        for _sp in ("top", "right", "bottom", "left"):
            ax_raster.spines[_sp].set_visible(False)
    else:
        ax_raster.set_ylabel("depth (µm)")
        # Keep only the left spine (y-axis); remove top, right, and bottom.
        for _sp in ("top", "right", "bottom"):
            ax_raster.spines[_sp].set_visible(False)
    ax_raster.tick_params(bottom=False, labelbottom=False)

    # --- Scale-bar (dedicated strip when show_rate; embedded on ax_raster otherwise) ---
    if ax_scale is not None:
        ax_scale.set_facecolor("white")
        ax_scale.set_yticks([])
        for _sp in ax_scale.spines.values():
            _sp.set_visible(False)
        ax_scale.tick_params(bottom=False, labelbottom=False)
        _bar_ax = ax_scale
        _bar_y  = 0.35    # y in ax_scale axes fraction
        _bar_ty = 0.70
    else:
        # Draw scale bar near the bottom of the raster to avoid a white-space gap.
        _bar_ax = ax_raster
        _bar_y  = 0.04    # y in ax_raster axes fraction (4 % up from raster bottom)
        _bar_ty = 0.08

    blend = blended_transform_factory(_bar_ax.transData, _bar_ax.transAxes)
    (bar_line,) = _bar_ax.plot([], [], color="black", lw=3, transform=blend,
                                solid_capstyle="butt", zorder=10)
    bar_text = _bar_ax.text(0, _bar_ty, "", transform=blend,
                             ha="center", va="bottom", fontsize=10, color="black", zorder=10)

    # --- Population-rate panel (optional; zooms with raster) ---
    if show_rate and ax_rate is not None:
        ax_rate.set_facecolor("white")
        # Updatable filled polygon: draw() replaces vertices each frame via set_xy().
        rate_fill = ax_rate.fill([], [], color="#5599ee", alpha=0.80, lw=0)[0]
        ax_rate.set_ylim(0, 1.3)
        ax_rate.set_yticks([0.0, 0.5, 1.0])
        ax_rate.set_yticklabels(["0", "½", "1"], fontsize=7)
        ax_rate.set_ylabel("pop. rate\n(norm.)", fontsize=8, labelpad=2)
        ax_rate.tick_params(labelbottom=False)
        for _sp in ("top", "right"):
            ax_rate.spines[_sp].set_visible(False)
    else:
        rate_fill = None

    # --- Day / night panel (zooms with raster) ---
    from matplotlib.patches import Patch
    _DAWN_PST_S = 6 * 3600      # 6 AM PST in seconds from midnight
    _DUSK_PST_S = 20 * 3600     # 8 PM PST
    _TRANS_S    = 1800           # 30-min smooth dawn/dusk transitions
    _DAY_RGB    = np.array([1.00, 0.88, 0.25])   # sunny yellow
    _NIGHT_RGB  = np.array([0.25, 0.25, 0.25])   # dark grey

    _pst_s = (_rate_t_fine + t0_pst_s) % 86400.0
    _day_f = np.zeros_like(_pst_s)
    _m = (_pst_s >= _DAWN_PST_S) & (_pst_s < _DAWN_PST_S + _TRANS_S)
    _day_f[_m] = (_pst_s[_m] - _DAWN_PST_S) / _TRANS_S
    _day_f[(_pst_s >= _DAWN_PST_S + _TRANS_S) & (_pst_s < _DUSK_PST_S)] = 1.0
    _m = (_pst_s >= _DUSK_PST_S) & (_pst_s < _DUSK_PST_S + _TRANS_S)
    _day_f[_m] = 1.0 - (_pst_s[_m] - _DUSK_PST_S) / _TRANS_S
    _tod_rgb  = _day_f[:, None] * _DAY_RGB + (1 - _day_f[:, None]) * _NIGHT_RGB
    _tod_img  = _tod_rgb[np.newaxis, :, :]

    ax_tod.set_facecolor("white")   # beyond data extent shows white (missing data)
    ax_tod.imshow(_tod_img, aspect="auto", extent=[t_lo, t_hi, 0, 1],
                  origin="lower", interpolation="bilinear", zorder=0)
    ax_tod.set_ylim(0, 1)
    ax_tod.set_yticks([])
    # No axis labels, tick marks, tick values, or legend on the ToD strip.
    ax_tod.tick_params(bottom=False, labelbottom=False)
    ax_tod.set_xlabel("")
    ax_tod.set_ylabel("")
    for _sp in ax_tod.spines.values():
        _sp.set_visible(False)

    # No crossing markers or labels on the ToD strip — decoration only.

    # --- Probe-map panel (static; drawn once before the animation loop) -------
    if show_probe_map and ax_probe is not None:
        import os as _os
        from pirouette_data.vis import (
            ProbeGeometry as _ProbeGeometry,
            draw_probe_schematic as _draw_probe,
            overlay_waveforms as _overlay_wf,
        )
        # Read probe geometry from environment (matches .env PROBE_* variables).
        def _ef(name: str, default: float) -> float:
            v = _os.getenv(name)
            return default if not v or not v.strip() else float(v)
        def _ei(name: str, default: int) -> int:
            v = _os.getenv(name)
            return default if not v or not v.strip() else int(v)
        def _eb(name: str, default: bool) -> bool:
            v = _os.getenv(name)
            return default if not v or not v.strip() else v.strip().lower() in ("1","true","yes","on")
        _geo = _ProbeGeometry(
            shank_width       = _ef("PROBE_SHANK_WIDTH_UM",      70.0),
            tip_length        = _ef("PROBE_TIP_LENGTH_UM",       175.0),
            site_width        = _ef("PROBE_SITE_WIDTH_UM",        12.0),
            site_height       = _ef("PROBE_SITE_HEIGHT_UM",       12.0),
            vertical_pitch    = _ef("PROBE_VERTICAL_PITCH_UM",    15.0),
            horizontal_pitch  = _ef("PROBE_HORIZONTAL_PITCH_UM",  32.0),
            n_sites           = _ei("PROBE_N_SITES",               96),
            first_site_offset = _ef("PROBE_FIRST_SITE_OFFSET_UM", 20.0),
            stagger           = _eb("PROBE_STAGGER",              False),
        )

        # Draw shank outline + recording-site rectangles.
        ax_probe.set_facecolor("white")
        _draw_probe(ax_probe, _geo)

        # Build RGBA color map that exactly matches the raster animation colors.
        try:
            # charlie mode: _ch_colors is (n_units, 3) float32; per_unit order = cidx
            _probe_cm = {
                per_unit[_pi]["id"]: np.append(_ch_colors[_pi], 1.0).astype("float32")
                for _pi in range(n_units)
            }
        except NameError:
            # hybrid / non-charlie: colors is (n_units, 4) RGBA
            _probe_cm = {
                per_unit[_pi]["id"]: colors[_pi]
                for _pi in range(n_units)
            }

        # Overlay mean waveforms using the same unit colors as the raster.
        # All waveform parameters come from SPIKING_MAP_* env vars (matching .env).
        try:
            _overlay_wf(
                ax_probe, _raw_units, _geo,
                palette=_os.getenv("SPIKING_MAP_PALETTE", palette),
                invert_depth=_eb("SPIKING_MAP_INVERT_DEPTH", invert_depth),
                unit_color_map=_probe_cm,
                wf_x_span_um   = _ef("SPIKING_MAP_WF_X_SPAN_UM",    24.0),
                wf_amp_um      = _ef("SPIKING_MAP_WF_AMP_UM",        10.0),
                relative_amp   = _eb("SPIKING_MAP_WF_RELATIVE_AMP", False),
                jitter_step_um = _ef("SPIKING_MAP_JITTER_STEP_UM",    4.0),
                n_jitter       = _ei("SPIKING_MAP_N_JITTER",             5),
                jitter_y_um    = _ef("SPIKING_MAP_JITTER_Y_UM",       2.5),
                n_jitter_y     = _ei("SPIKING_MAP_N_JITTER_Y",           3),
                upsample       = _ei("SPIKING_MAP_WF_UPSAMPLE",          4),
                linewidth      = _ef("SPIKING_MAP_WF_LW",             0.5),
                alpha          = _ef("SPIKING_MAP_WF_ALPHA",          0.85),
                seed=seed,
            )
        except Exception:
            pass   # no waveform data in this pickle — shank outline still shown

        # Set probe x-limits so that 1 µm = 1 µm visually (sites appear square).
        # Compute axes dimensions from the GridSpec position (available immediately
        # when explicit left/right/top/bottom are used — no canvas.draw() needed).
        _pos_p   = ax_probe.get_position()          # Bbox in figure fraction
        _ax_w_in = _pos_p.width  * fig.get_figwidth()
        _ax_h_in = _pos_p.height * fig.get_figheight()
        _y_range = abs(ax_probe.get_ylim()[1] - ax_probe.get_ylim()[0])  # full probe ylim span
        _x_range = (_ax_w_in / _ax_h_in) * _y_range # µm needed for equal aspect
        _x_ctr   = _geo.shank_width / 2.0           # centre of probe shank
        ax_probe.set_xlim(_x_ctr - _x_range / 2.0, _x_ctr + _x_range / 2.0)

        # Style: left spine + depth y-ticks; no x-axis.
        ax_probe.xaxis.set_visible(False)
        for _sp in ("top", "right", "bottom"):
            ax_probe.spines[_sp].set_visible(False)
        ax_probe.spines["left"].set_visible(False)
        ax_probe.tick_params(axis="y", left=False, labelleft=False)
        ax_probe.set_ylabel("")
        # Unit-count label sits at the very top of the probe axes (99 % up),
        # just above the topmost recording sites. va='top' anchors the text top
        # at that fraction so the label hangs down into the visible area.
        ax_probe.text(0.5, 0.99, f"{n_units} units",
                      transform=ax_probe.transAxes,
                      ha="center", va="top", fontsize=8)

    # -------------------------------------------------------- draw / update --
    _unit_thresholds = (axis_unit_ms_to_s, axis_unit_s_to_min, axis_unit_min_to_hr)
    _last_uname = [None]

    # ── Pixel-image fast path for wide zoom ──────────────────────────────────
    # At windows >= _PX_S (where the _s formula hits 1.0 and a 1.7 px circle
    # is indistinguishable from a single pixel at video resolution), bypass
    # Agg's circle renderer entirely.  Instead, write spike colours directly
    # into a numpy RGBA buffer and display it with imshow.
    #
    # Benchmark at 36-hr zoom (500 k points):
    #   scatter "o"  → ~1.9 s/frame   (Agg rasterises 500 k circle paths)
    #   numpy pixels → ~5 ms/frame    (array indexing + imshow blit)
    #   Speedup: ~400×, ~90% of animation frames.
    #
    # The pixel mapping uses the same formula for both invert_depth modes:
    #   row 0 = d_lo - d_pad,  row img_h-1 = d_hi + d_pad (data coords).
    # matplotlib's y-axis inversion (set_ylim(hi, lo)) flips the display
    # automatically — no conditional code needed in draw().
    _PX_S    = 30.0              # window width (s) where _s formula = 1.0
    _d_lo_px = d_lo - d_pad     # data y at the bottom of the imshow extent
    _d_full  = (d_hi + d_pad) - (d_lo - d_pad)   # full depth span of the image

    # Density-blend schedule: 0 at ≤1 min (pure ticks), 1 at ≥1 hr (full density).
    _HMAP_LOG_LO   = np.log10(60.0)      # log10(1 min) — blend start
    _HMAP_LOG_HI   = np.log10(3600.0)    # log10(1 hr)  — blend end
    _HMAP_LOG_SPAN = _HMAP_LOG_HI - _HMAP_LOG_LO

    # Lazy state: built on the first wide-zoom frame so axes are fully laid out.
    _pxim_state: list = [None]   # None → not yet created; else (im, buf, iw, ih)

    def _ensure_pxim():
        """Return cached (im, buf, iw, ih) — create on first call."""
        if _pxim_state[0] is not None:
            return _pxim_state[0]
        # By the time we reach wide zoom, FuncAnimation has already rendered
        # many frames, so the Agg renderer and axes layout are fully initialised.
        try:
            renderer = fig.canvas.get_renderer()
            bbox = ax_raster.get_window_extent(renderer)
            iw = max(100, int(np.ceil(bbox.width)))
            ih = max(50,  int(np.ceil(bbox.height)))
        except Exception:
            # Fallback: estimate from figure size and GridSpec params.
            fw_px = int(round(fig.get_size_inches()[0] * dpi))
            fh_px = int(round(fig.get_size_inches()[1] * dpi))
            iw = max(100, int(fw_px * (0.97 - 0.08)))
            ih = max(50,  int(fh_px * (5 / 7.73) * (0.97 - 0.10)))
        buf = np.ones((ih, iw, 4), dtype="float32")   # RGBA white
        im  = ax_raster.imshow(
            buf, aspect="auto", origin="lower", interpolation="nearest",
            extent=[t_lo, t_hi, _d_lo_px, _d_lo_px + _d_full],
            zorder=1, visible=False,   # starts hidden; draw() calls im.set_visible(True)
        )
        _pxim_state[0] = (im, buf, iw, ih)
        return _pxim_state[0]

    def draw(lo, hi):
        width = hi - lo

        # ── Density-blend factor (_db): 0 at ≤30 s, 1 at ≥5 min ────────────
        # Smoothstep in log-width space.  At _db=0 only fixed-K ticks show;
        # at _db=1 the per-unit coloured density background dominates and ticks
        # accent burst columns.  Same colour for both layers → seamless blend.
        if width <= 10 ** _HMAP_LOG_LO:
            _db = 0.0
        elif width >= 10 ** _HMAP_LOG_HI:
            _db = 1.0
        else:
            _ht = np.clip(
                (np.log10(width) - _HMAP_LOG_LO) / _HMAP_LOG_SPAN, 0.0, 1.0
            )
            _db = _ht * _ht * (3.0 - 2.0 * _ht)

        # ── Single pixel-buffer path (all zoom levels) ────────────────────────
        im, buf, iw, ih = _ensure_pxim()
        im.set_visible(True)
        buf[:] = 1.0   # reset to RGBA white

        if raster_mode == "charlie":
            # ── charlie raster (dartsort scatter_time_vs_depth aesthetic) ────
            # Three independent log-width visual schedules:
            #   tick half-height — 5 px at ≤ 1 ms → 1 px by 1 min (striations at short zoom)
            #   Gaussian y-jitter — 0 until 30 s, grows to _CH_JITTER_MAX px at 10 hr
            #   alpha             — constant 0.9
            #
            # Random priority subsampling (dartsort style): each spike has a
            # precomputed random priority.  We keep the max_spikes highest-priority
            # spikes in the visible window via argpartition (O(N), no sort needed).
            # Bursts → more spikes → more high-priority candidates → burst columns
            # remain visible as vertical striations at all zoom levels.
            # After subsampling, spikes are sorted by amplitude so that high-amp
            # units are drawn last (on top), matching the dartsort look.
            _ch_lw = np.log10(max(width, 1e-3))
            _ch_wn_tick   = float(np.clip(
                _ch_lw / _CH_LOG_TICK_HI, 0.0, 1.0))
            _ch_wn_jitter = float(np.clip(
                (_ch_lw - _CH_LOG_JITTER_LO)
                / (_CH_LOG_JITTER_HI - _CH_LOG_JITTER_LO), 0.0, 1.0))
            _ch_half   = max(0, round(5.0 * (1.0 - _ch_wn_tick)))
            _ch_half_x = 1 if width < 3.0 else 0
            _ch_sigma  = _CH_JITTER_MAX * _ch_wn_jitter

            # Collect spikes for all units in the visible window.
            # Per-unit density mode also computes each unit's own local density
            # scale here, before concatenation, so quiet units are unaffected by
            # other units' bursts.
            _n_db = 200   # x-bins for density estimation
            _tvp, _dvp, _avp, _cvp, _pvp, _jvp, _dsp = [], [], [], [], [], [], []
            for _ipu, _pu in enumerate(per_unit):
                _ai = int(np.searchsorted(_pu["times"], lo, "left"))
                _bi = int(np.searchsorted(_pu["times"], hi, "right"))
                _m  = _bi - _ai
                if _m == 0:
                    continue
                _sl = slice(_ai, _bi)
                _t_pu = _pu["times"][_sl]
                _tvp.append(_t_pu)
                _dvp.append(_pu["spike_depths"][_sl])
                _avp.append(_ch_pu_amps[_ipu][_sl])
                _cvp.append(np.full(_m, _pu["cidx"], dtype=np.int32))
                _pvp.append(_ch_pu_priority[_ipu][_sl])
                _jvp.append(_ch_pu_jitter[_ipu][_sl])
                if charlie_jitter_density == "per-unit" and _ch_sigma > 0.0:
                    # Density scale from this unit's own spikes only.
                    _xb_pu   = np.clip(
                        ((_t_pu - lo) / (hi - lo) * _n_db).astype(np.int32),
                        0, _n_db - 1)
                    _bc_pu   = np.bincount(_xb_pu, minlength=_n_db).astype("float32")
                    _mbc_pu  = max(_bc_pu.max(), 1.0)
                    _dsp.append(_bc_pu[_xb_pu] / _mbc_pu)
                else:
                    _dsp.append(None)   # filled after global density computed

            if _tvp:
                _tv = np.concatenate(_tvp)
                _dv = np.concatenate(_dvp).astype("float32")
                _av = np.concatenate(_avp)
                _cv = np.concatenate(_cvp)
                _pv = np.concatenate(_pvp)
                _jv = np.concatenate(_jvp)
                # Build density-scale array (_ds) aligned with the full spike list.
                # "none" mode skips density entirely — _ds stays None and jitter
                # is applied as plain _jv * _ch_sigma (uniform, not adaptive).
                if _ch_sigma > 0.0 and charlie_jitter_density != "none":
                    if charlie_jitter_density == "per-unit":
                        _ds = np.concatenate(_dsp)
                    else:
                        # Global: bin all spikes together.
                        _xb_gl  = np.clip(
                            ((_tv - lo) / (hi - lo) * _n_db).astype(np.int32),
                            0, _n_db - 1)
                        _bc_gl  = np.bincount(_xb_gl, minlength=_n_db).astype("float32")
                        _ds     = _bc_gl[_xb_gl] / max(_bc_gl.max(), 1.0)
                else:
                    _ds = None
                # Priority subsample if actual count exceeds cap.
                if len(_tv) > charlie_max_spikes:
                    _keep = np.argpartition(_pv, -charlie_max_spikes)[
                        -charlie_max_spikes:
                    ]
                    _tv = _tv[_keep]; _dv = _dv[_keep]; _av = _av[_keep]
                    _cv = _cv[_keep]; _jv = _jv[_keep]
                    if _ds is not None:
                        _ds = _ds[_keep]
                # Amplitude sort: high → low so high-amp units claim pixels first
                # under paint-on-white-only rendering (first writer wins).
                _ao = np.argsort(_av, kind="stable")[::-1]
                _tv, _dv, _cv, _jv = _tv[_ao], _dv[_ao], _cv[_ao], _jv[_ao]
                if _ds is not None:
                    _ds = _ds[_ao]
                _px_x = np.clip(
                    ((_tv - lo) / (hi - lo) * iw).astype(np.int32), 0, iw - 1)
                _px_y = ((_dv - _d_lo_px) / _d_full * ih).astype(np.int32)
                if _ch_sigma > 0.0:
                    if charlie_jitter_density == "none":
                        # Uniform jitter: every spike gets the full sigma scale.
                        _px_y = _px_y + (_jv * _ch_sigma).astype(np.int32)
                    elif _ds is not None:
                        # Adaptive jitter: scale by per-unit or global density.
                        _px_y = _px_y + (_jv * _ds * _ch_sigma).astype(np.int32)
                _px_y = np.clip(_px_y, 0, ih - 1)
                for _dy in range(-_ch_half, _ch_half + 1):
                    _py = np.clip(_px_y + _dy, 0, ih - 1)
                    for _dx in range(-_ch_half_x, _ch_half_x + 1):
                        _ppx = np.clip(_px_x + _dx, 0, iw - 1)
                        # Paint-on-white-only: skip pixels already claimed by
                        # another spike so each pixel shows exactly one unit's
                        # colour — no overwrite blending, sharper striation edges.
                        _wh = buf[_py, _ppx, 3] == 1.0
                        if _wh.any():
                            buf[_py[_wh], _ppx[_wh], :3] = _ch_colors[_cv[_wh]]
                            buf[_py[_wh], _ppx[_wh],  3] = 0.9

        else:
            # ── hybrid raster ─────────────────────────────────────────────────
            # Tick height: fill each unit's full allocated pixel band, with a
            # generous minimum so ticks look like vertical dashes, not squares.
            # _ppu is the per-unit pixel budget; floor at 5 → 11 px minimum height.
            # K-subsampling (3 % fill) keeps adjacent-unit bleed invisible because
            # 97 % of x-columns are empty even when ticks span multiple unit rows.
            _ppu  = max(1, ih // max(1, n_units))
            _half = max(5, _ppu // 2)

            # Tick width (x-spread): 3 px at narrow zoom for visibility, 1 px at
            # ≥ 3 s so the ticks stay thin and don't smear into bands.
            if width < 3.0:
                _half_x = 1      # 3 px wide  (< 3 s)
            else:
                _half_x = 0      # 1 px wide  (≥ 3 s)

            # K ticks per unit: smoothly interpolated from 3 % at ≤ 30 s to 1 % at
            # ≥ 60 s (1 min) in log-width space so the density of ticks tapers
            # gradually rather than jumping at any single frame.
            _k_norm = np.clip(
                (np.log10(max(width, 1e-3)) - np.log10(30.0))
                / (np.log10(60.0) - np.log10(30.0)),
                0.0, 1.0,
            )
            _k_frac = 0.03 * (1.0 - _k_norm) + 0.01 * _k_norm   # 3 % → 1 %
            _k_px = max(5, int(iw * _k_frac))

            # Layer 1 — per-unit coloured density background ──────────────────
            # Each unit's pixel row is tinted white → unit colour proportional
            # to its local firing-rate density × _db.  Invisible at narrow zoom;
            # fully coloured at wide zoom.  Uses the precomputed _unit_dens
            # (n_units × _UNIT_HMAP_T) mapped to display x-columns.
            if _db > 0.0:
                _unit_py = np.clip(
                    ((_unit_depths - _d_lo_px) / _d_full * ih).astype(np.int32),
                    0, ih - 1,
                )
                _t_cx = lo + (np.arange(iw, dtype="float32") + 0.5) / iw * (hi - lo)
                _t_bi = np.clip(
                    ((_t_cx - t_lo) / (t_hi - t_lo) * _UNIT_HMAP_T).astype(np.int32),
                    0, _UNIT_HMAP_T - 1,
                )
                # _dm: (n_units, iw) density × blend per column
                _dm = _unit_dens[:, _t_bi] * _db
                # _bg: (n_units, iw, 3) — lerp white→unit colour
                _bg = 1.0 - _dm[:, :, np.newaxis] * (
                    1.0 - colors[:n_units, :3][:, np.newaxis, :])
                for _dy in range(-_half, _half + 1):
                    _pys = np.clip(_unit_py + _dy, 0, ih - 1)
                    buf[_pys, :, :3] = _bg
                    buf[_pys, :, 3]  = 1.0

            # Layer 2 — fixed-K tick marks, cross-fading with density layer ───
            # _tick_vis is the square of the complement of _db so ticks fade out
            # quickly once the density layer begins to show (rather than lingering
            # at half-strength through the whole 1-min → 1-hr transition).
            #   _db=0   (≤1 min)  → _tick_vis=1.0  full ticks, no density
            #   _db=0.5 (~6 min)  → _tick_vis=0.25  mostly gone
            #   _db=1.0 (≥1 hr)   → _tick_vis=0.0  invisible
            _tick_vis = (1.0 - _db) ** 2
            if _tick_vis > 0.0:
                x, y, c = frame_points_fixed_k(per_unit, lo, hi, _k_px)
                if len(x):
                    px_x = np.clip(
                        ((x - lo) / (hi - lo) * iw).astype(np.int32), 0, iw - 1)
                    px_y = np.clip(
                        ((y - _d_lo_px) / _d_full * ih).astype(np.int32), 0, ih - 1)
                    _combo = px_x.astype(np.int64) * n_units + c
                    _, _first = np.unique(_combo, return_index=True)
                    _ux, _uy, _uc = px_x[_first], px_y[_first], c[_first]
                    _tc = colors[_uc, :3]          # (n_ticks, 3) full unit colour
                    for _dy in range(-_half, _half + 1):
                        _py = np.clip(_uy + _dy, 0, ih - 1)
                        for _dx in range(-_half_x, _half_x + 1):
                            _px = np.clip(_ux + _dx, 0, iw - 1)
                            # Blend: full tick colour at narrow zoom; fades to
                            # density background colour as wide zoom approaches.
                            buf[_py, _px, :3] = (
                                _tick_vis * _tc
                                + (1.0 - _tick_vis) * buf[_py, _px, :3]
                            )
                            buf[_py, _px, 3] = 1.0

        im.set_data(buf)
        im.set_extent([lo, hi, _d_lo_px, _d_lo_px + _d_full])

        ax_raster.set_xlim(lo, hi)   # propagates to ax_scale, ax_rate, ax_tod via sharex
        # Scale bar on the dedicated strip — centred in the window so the text
        # stays stationary (in center-zoom mode x_mid = center_s = constant).
        dur, label = pick_scale_bar(width / 2.0)   # cap bar at ½ x-axis length
        x_mid = (lo + hi) / 2.0
        bar_line.set_data([x_mid - dur / 2.0, x_mid + dur / 2.0], [_bar_y, _bar_y])
        bar_text.set_position((x_mid, _bar_ty))
        bar_text.set_text(label)
        # Dynamic x-axis tick labels on the bottom (ax_tod) panel.
        scale, uname = x_axis_unit(width, _unit_thresholds)
        step = _nice_step(width / scale / 6.0) * scale
        first = np.ceil(lo / step) * step
        n_t = max(0, int(np.floor((hi - first) / step)) + 1)
        ticks = first + np.arange(n_t) * step
        ticks = ticks[(ticks >= lo - step * 1e-6) & (ticks <= hi + step * 1e-6)]
        # Always include the reference point (center_s in center mode, anchor_s
        # otherwise) so that "0" is permanently visible in the tick labels.
        # Remove any regular tick within 60 % of a step of _tick_ref first so
        # the "0" label never crowds an adjacent label.
        if lo <= _tick_ref <= hi:
            mask = np.abs(ticks - _tick_ref) >= step * 0.6
            ticks = np.sort(np.concatenate([ticks[mask], [_tick_ref]]))
        # ToD panel: at full span show "0 hr" / "1332 hr" endpoints; otherwise blank.
        _at_full_span = (hi - lo) >= window_end_s * 0.99
        if _at_full_span:
            ax_tod.set_xticks([0.0, window_end_s])
            ax_tod.set_xticklabels(["0 hr", "1332 hours\n56 days"], fontsize=7)
            ax_tod.tick_params(bottom=True, labelbottom=True, length=3, pad=2)
            for _lbl in ax_tod.get_xticklabels():
                _lbl.set_clip_on(False)   # let "56 days" extend below the axes edge
        else:
            ax_tod.set_xticks([])
            ax_tod.tick_params(bottom=False, labelbottom=False)
        # Adaptive-resolution population rate (only when show_rate is True).
        if show_rate and rate_fill is not None:
            # Wide zoom (>= _RATE_SWITCH_S): pre-computed 10 s bins with adaptive
            # Gaussian smoothing.  Narrow zoom: histogram from raw per-unit data.
            if width >= _RATE_SWITCH_S:
                _iv0 = max(0, int(np.searchsorted(_rate_t_fine, lo)) - 1)
                _iv1 = min(len(_rate_t_fine), int(np.searchsorted(_rate_t_fine, hi)) + 1)
                if _iv1 - _iv0 >= 2:
                    _rt = _rate_t_fine[_iv0:_iv1]
                    _rv = _rate_n_fine[_iv0:_iv1]
                    _sigma_b = max(0.5, (_iv1 - _iv0) / 25.0)
                    _rn = _gaussian_smooth(_rv, _sigma_b)
                else:
                    _rt, _rn = np.array([lo, hi]), np.zeros(2)
            else:
                _bw = max(width / 200.0, 1e-4)
                _n_bins = max(1, int(np.ceil(width / _bw)))
                _bins_r = np.linspace(lo, hi, _n_bins + 1)
                _cnt = np.zeros(_n_bins, dtype="float64")
                for _pu in per_unit:
                    _i0 = int(np.searchsorted(_pu["times"], lo))
                    _i1 = int(np.searchsorted(_pu["times"], hi))
                    if _i1 > _i0:
                        _c, _ = np.histogram(_pu["times"][_i0:_i1], bins=_bins_r)
                        _cnt += _c.astype("float64")
                _sigma_b = max(0.5, _n_bins / 20.0)
                _cnt_sm  = _gaussian_smooth(_cnt, _sigma_b)
                _rt = 0.5 * (_bins_r[:-1] + _bins_r[1:])
                _rn = np.clip(_cnt_sm * _FINE_BIN_S / (_bw * _rate_peak), 0.0, 1.3)
            if len(_rt) >= 2:
                _fx = np.concatenate([[_rt[0]], _rt, [_rt[-1]]])
                _fy = np.concatenate([[0.0], _rn, [0.0]])
                rate_fill.set_xy(np.column_stack([_fx, _fy]))

    def update(frame):
        width = widths[frame] if frame < n_zoom else window_end_s
        if zoom_mode == "center":
            lo, hi = _window_bounds(center_s, width, t_lo, t_hi)
        elif zoom_mode == "center-right":
            # Centred zoom until the left edge reaches t_lo, then expands right
            # up to window_end_s (which may extend well beyond the data end).
            lo, hi = _window_bounds_center_right(
                center_s, width, t_lo, window_end_s)
        else:
            lo, hi = _window_bounds_left(anchor_s, width, t_lo, t_hi)
        draw(lo, hi)
        im, *_ = _pxim_state[0] or (None,)
        _artists = [bar_line, bar_text]
        if rate_fill is not None:
            _artists.append(rate_fill)
        if im is not None:
            _artists.insert(0, im)
        return tuple(_artists)

    total = n_zoom + n_hold
    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=1000 / fps, blit=False
    )

    def _progress(i, n):
        pct = (i + 1) / n
        bar = "#" * int(pct * 40)
        print(f"\r  rendering [{bar:<40}] {pct * 100:3.0f}%  ({i + 1}/{n})",
              end="", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        # -preset veryfast keeps x264 from being the bottleneck (the default preset
        # falls behind at 1080p and stalls the render); crf gives good quality.
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"],
        )
    anim.save(str(out_path), writer=writer, dpi=dpi, progress_callback=_progress)
    print()  # newline after the progress bar
    plt.close(fig)
    return out_path
