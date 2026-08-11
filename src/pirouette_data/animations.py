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

# Scale-bar steps (seconds, label): the bar shows the largest of these that fits
# in the current view, so it "switches" as the zoom widens.
SCALE_STEPS: list[tuple[float, str]] = [
    (0.001, "1 ms"),
    (0.1,   "100 ms"),
    (1.0,   "1 s"),
    (60.0,  "1 min"),
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


def _build_widths(
    window_start_s: float,
    window_end_s: float,
    n_base: int,
    easing: str = "ease-in-out",
    milestone_dwell_n: int = 0,
) -> np.ndarray:
    """Per-frame window-width sequence, with optional easing and milestone dwells.

    Parameters
    ----------
    easing:
        ``"ease-in-out"`` (default) — smoothstep; slows the start and end of the
        zoom so it doesn't feel mechanical. ``"linear"`` keeps constant log-speed.
    milestone_dwell_n:
        Extra frames to hold at each scale-bar milestone (where the scale-bar
        label switches). Gives viewers a moment to orient at each decade before
        the next zoom begins. Set to ``0`` to disable.
    """
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
    milestones = sorted(s for s, _ in SCALE_STEPS if window_start_s < s < window_end_s)
    result: list[float] = []
    prev = float(widths[0])
    for w in widths:
        result.append(float(w))
        for ms in milestones:
            if prev < ms <= w:
                result.extend([ms] * milestone_dwell_n)
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
    milestone_dwell_s: float = 0.5,
    axis_unit_ms_to_s: float = 1.0,
    axis_unit_s_to_min: float = 100.0,
    axis_unit_min_to_hr: float = 6000.0,
    zoom_mode: str = "anchor-left",
    anchor_s: float | None = None,
    rate_bin_s: float = 1800.0,
    t0_pst_s: float = 0.0,
    seed: int = 0,
    title: str = "Manually curated spike output",
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
    per_unit = prepare_units(load_units_with_depth(units_path))
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
    d_pad = 0.04 * (d_hi - d_lo) + 5.0

    # Match dartsort's max_spikes_plot=500k: no subsampling for windows < ~350 s
    # (500k / (240 units × 6 Hz avg)).  Global random sampling (frame_points) means
    # dense periods contribute proportionally more points, revealing striations.
    _scatter_cap = max(cap, 500_000)

    # ── Precomputed 2D density heatmap ───────────────────────────────────────
    # 10 k time bins × 400 depth bins.  Per-row normalisation means each depth
    # position (unit) shows its own temporal firing rate normalised to [0, 1],
    # so units with wildly different mean rates are equally visible.
    # The image is static — matplotlib pans/zooms it as set_xlim() changes —
    # so there is no per-frame computation.  draw() fades it in smoothly by
    # calling hmap_im.set_alpha() as the window grows past ~30 s.
    _HMAP_T = 10_000
    _HMAP_D = 400
    _hmap_bins_t = np.linspace(t_lo, t_hi, _HMAP_T + 1)
    _hmap_bins_d = np.linspace(d_lo - d_pad, d_hi + d_pad, _HMAP_D + 1)
    _hmap = np.zeros((_HMAP_D, _HMAP_T), dtype="float64")
    for _hpu in per_unit:
        # Vectorised 2-D binning via searchsorted on the pre-sorted arrays.
        _ht = np.clip(
            np.searchsorted(_hmap_bins_t[1:], _hpu["times"], side="right"),
            0, _HMAP_T - 1,
        )
        _hd = np.clip(
            np.searchsorted(_hmap_bins_d[1:], _hpu["spike_depths"], side="right"),
            0, _HMAP_D - 1,
        )
        _hmap += np.bincount(_hd * _HMAP_T + _ht,
                             minlength=_HMAP_D * _HMAP_T).reshape(_HMAP_D, _HMAP_T)
    _hmap_row_max = np.maximum(_hmap.max(axis=1, keepdims=True), 1.0)
    _hmap_norm = (_hmap / _hmap_row_max).astype("float32")
    del _hmap, _hmap_row_max   # free ~30 MB

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
    widths = _build_widths(window_start_s, window_end_s, n_base, easing, n_dwell)
    n_zoom = len(widths)
    n_hold = max(1, int(round(hold_s * fps)))

    # ------------------------------------------------------------------ figure --
    # 4 rows: raster | scale-bar strip | full-recording rate | full-recording tod
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[5, 0.28, 1.8, 0.65],
        hspace=0.07,
        left=0.08, right=0.97, top=0.97, bottom=0.10,
    )
    ax_raster = fig.add_subplot(gs[0])
    ax_scale  = fig.add_subplot(gs[1], sharex=ax_raster)   # scale bar; follows zoom
    ax_rate   = fig.add_subplot(gs[2], sharex=ax_raster)   # zooms with raster
    ax_tod    = fig.add_subplot(gs[3], sharex=ax_raster)   # zooms with raster
    fig.patch.set_facecolor("white")

    # --- Raster panel ---
    ax_raster.set_facecolor("white")
    # rasterized=True composites points during save — same as dartsort.
    # marker="o" (filled circle) and s=1 are dartsort's defaults; edgecolors/lw=0
    # avoids marker outlines that would dominate at small sizes.
    scat = ax_raster.scatter([], [], s=1, marker="o", edgecolors="none",
                             linewidths=0, rasterized=True, zorder=1)

    # Precomputed density heatmap — zorder=2 places it on top of the scatter /
    # pixel buffer (both at zorder=1).  draw() sets alpha to 0 at narrow zoom
    # and ramps to _HMAP_MAX_ALPHA as the window crosses 30 s → 1 hr.
    # "Blues" maps sparse (0) → near-white and dense (1) → dark blue, so
    # high-firing time periods appear as dark vertical stripes (striations).
    hmap_im = ax_raster.imshow(
        _hmap_norm,
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        aspect="auto",
        origin="lower",
        extent=[t_lo, t_hi, d_lo - d_pad, d_hi + d_pad],
        alpha=0.0,          # starts invisible; set per-frame in draw()
        zorder=2,
        interpolation="bilinear",
    )
    if invert_depth:
        ax_raster.set_ylim(d_lo - d_pad, d_hi + d_pad)
    else:
        ax_raster.set_ylim(d_hi + d_pad, d_lo - d_pad)
    ax_raster.set_ylabel("unit depth (µm)")
    ax_raster.tick_params(labelbottom=False)

    # --- Scale-bar strip (shares X with raster; no content except the bar) ---
    ax_scale.set_facecolor("white")
    ax_scale.set_yticks([])
    for _sp in ax_scale.spines.values():
        _sp.set_visible(False)
    ax_scale.tick_params(bottom=False, labelbottom=False)

    blend = blended_transform_factory(ax_scale.transData, ax_scale.transAxes)
    (bar_line,) = ax_scale.plot([], [], color="black", lw=3, transform=blend,
                                 solid_capstyle="butt")
    bar_text = ax_scale.text(0, 0.78, "", transform=blend,
                              ha="center", va="bottom", fontsize=10, color="black")

    # --- Population-rate panel (zooms with raster; adaptive-resolution per frame) ---
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

    ax_tod.imshow(_tod_img, aspect="auto", extent=[t_lo, t_hi, 0, 1],
                  origin="lower", interpolation="bilinear", zorder=0)
    ax_tod.set_ylim(0, 1)
    ax_tod.set_yticks([])
    ax_tod.set_ylabel("time\nof day\n(PST)", fontsize=7, labelpad=2)
    ax_tod.tick_params(bottom=True, labelbottom=True, labelsize=8)
    for _sp in ("top", "right", "left"):
        ax_tod.spines[_sp].set_visible(False)

    # Dawn / dusk crossing markers — start invisible; draw() toggles them as the
    # zoom window reaches each crossing, avoiding font-renderer overflow at narrow zoom.
    _blend_tod = blended_transform_factory(ax_tod.transData, ax_tod.transAxes)
    _crossings: list[tuple[float, object, object]] = []
    for _d in range(-1, 8):
        for _t_pst, _lbl in [(_DAWN_PST_S, "6 AM"), (_DUSK_PST_S, "8 PM")]:
            _t_cross = _d * 86400.0 + _t_pst - t0_pst_s
            if t_lo <= _t_cross <= t_hi:
                _vl = ax_tod.axvline(_t_cross, color="white", lw=0.6, alpha=0.55,
                                     zorder=2, visible=False)
                _tx = ax_tod.text(_t_cross, 0.88, _lbl, ha="center", va="top",
                                  fontsize=6, color="white", zorder=3,
                                  transform=_blend_tod, visible=False)
                _crossings.append((_t_cross, _tx, _vl))

    # Legend: day vs night, placed just above the tod strip (does not share x-space).
    ax_tod.legend(
        handles=[
            Patch(facecolor=tuple(_DAY_RGB),                    label="Day  (6 AM – 8 PM PST)"),
            Patch(facecolor=tuple(_NIGHT_RGB), edgecolor="0.5", label="Night  (8 PM – 6 AM PST)"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.06),
        bbox_transform=ax_tod.transAxes,
        fontsize=7, ncol=2,
        framealpha=0.92, facecolor="white", edgecolor="0.8",
        handlelength=1.2, handleheight=0.9,
        borderpad=0.4, columnspacing=0.8,
    )

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

    # Heatmap blend: ramp alpha 0 → _HMAP_MAX_ALPHA over one log-decade in
    # window width, starting where the pixel path begins (_PX_S = 30 s) and
    # reaching full opacity at 1 hour.  Smoothstep gives an imperceptible
    # transition — no jarring switch, just gradual appearance of the blue
    # striation overlay as the zoom widens past 30 s → 3600 s.
    _HMAP_MAX_ALPHA = 0.72        # at 1 hr+: heatmap at 72 % opacity
    _HMAP_LOG_LO    = np.log10(_PX_S)        # log10(30)
    _HMAP_LOG_HI    = np.log10(3600.0)       # log10(1 hr)
    _HMAP_LOG_SPAN  = _HMAP_LOG_HI - _HMAP_LOG_LO

    # Lazy state: built on the first wide-zoom frame so axes are fully laid out.
    _pxim_state: list = [None]   # None → not yet created; else (im, buf, iw, ih)
    _use_pxim = [False]          # True → pixel imshow is the visible artist

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
            zorder=1, visible=False,   # below hmap_im (zorder=2)
        )
        _pxim_state[0] = (im, buf, iw, ih)
        return _pxim_state[0]

    def draw(lo, hi):
        width = hi - lo

        # ── Heatmap blend ────────────────────────────────────────────────────
        # Fade the precomputed density overlay in as the window passes ~30 s.
        # Below _PX_S: alpha=0 (scatter-only, no heatmap visible).
        # Above 1 hr: alpha=_HMAP_MAX_ALPHA (striation pattern fully visible).
        # Smoothstep in log-window-width space gives a gentle ramp.
        if width <= _PX_S:
            _hmap_a = 0.0
        elif width >= 3600.0:
            _hmap_a = _HMAP_MAX_ALPHA
        else:
            _ht = (np.log10(width) - _HMAP_LOG_LO) / _HMAP_LOG_SPAN
            _hmap_a = _HMAP_MAX_ALPHA * _ht * _ht * (3.0 - 2.0 * _ht)
        hmap_im.set_alpha(_hmap_a)

        # Adaptive marker size matching dartsort's s=1 at wide zoom, thicker at
        # narrow zoom so individual spikes are legible when zoomed in.
        # dartsort uses s=1 fixed; we floor at 1 and scale up at narrow zoom.
        #   1 ms → 10 pt²,  1 s → 4 pt²,  100 s → 1 pt²,  100 hr → 1 pt²
        # At 120 DPI s=1 ≈ 1.7 px diam — dense periods visibly fuller than sparse.
        _s = float(max(1.0, 4.0 - 2.0 * np.log10(max(width, 1e-2))))

        x, y, c = frame_points(per_unit, lo, hi, _scatter_cap)

        if width >= _PX_S:
            # ── Fast path: numpy pixel writes (bypasses Agg circle renderer) ──
            # Cuts per-frame render from ~1.9 s (scatter "o") to ~5 ms.
            im, buf, iw, ih = _ensure_pxim()
            if not _use_pxim[0]:
                # Transition scatter → imshow: hide scatter, show pixel image.
                scat.set_offsets(np.empty((0, 2)))   # free memory
                scat.set_visible(False)
                im.set_visible(True)
                _use_pxim[0] = True
            buf[:] = 1.0   # reset to RGBA white (fast memset)
            if len(x):
                px_x = np.clip(
                    ((x - lo) / (hi - lo) * iw).astype(np.int32), 0, iw - 1)
                px_y = np.clip(
                    ((y - _d_lo_px) / _d_full * ih).astype(np.int32), 0, ih - 1)
                buf[px_y, px_x, :3] = colors[c, :3]
                buf[px_y, px_x,  3] = 1.0
            im.set_data(buf)
            im.set_extent([lo, hi, _d_lo_px, _d_lo_px + _d_full])
        else:
            # ── Narrow zoom: scatter with adaptive circle markers ─────────────
            # Thick circles (s > 1) keep individual spikes legible at ms zoom.
            if _use_pxim[0]:
                _pxim_state[0][0].set_visible(False)
                scat.set_visible(True)
                _use_pxim[0] = False
            if len(x):
                scat.set_offsets(np.column_stack([x, y]))
                scat.set_facecolor(colors[c])
                scat.set_sizes(np.full(len(x), _s))
            else:
                scat.set_offsets(np.empty((0, 2)))

        ax_raster.set_xlim(lo, hi)   # propagates to ax_scale, ax_rate, ax_tod via sharex
        # Scale bar on the dedicated strip.
        dur, label = pick_scale_bar(width)
        x0 = lo + 0.05 * width
        bar_line.set_data([x0, x0 + dur], [0.35, 0.35])
        bar_text.set_position((x0 + dur / 2.0, 0.70))
        bar_text.set_text(label)
        # Dynamic x-axis tick labels on the bottom (ax_tod) panel.
        scale, uname = x_axis_unit(width, _unit_thresholds)
        step = _nice_step(width / scale / 6.0) * scale
        first = np.ceil(lo / step) * step
        n_t = max(0, int(np.floor((hi - first) / step)) + 1)
        ticks = first + np.arange(n_t) * step
        ticks = ticks[(ticks >= lo - step * 1e-6) & (ticks <= hi + step * 1e-6)]
        ax_tod.set_xticks(ticks)
        ax_tod.set_xticklabels([f"{(tk - _tick_ref) / scale:g}" for tk in ticks])
        if uname != _last_uname[0]:
            ax_tod.set_xlabel(f"time ({uname})", fontsize=9)
            _last_uname[0] = uname
        # Adaptive-resolution population rate.
        # Wide zoom (>= _RATE_SWITCH_S): slice the pre-computed 10 s bins and apply
        # Gaussian smoothing scaled to the number of visible bins (so circadian patterns
        # emerge naturally as the window grows to 100 hrs).
        # Narrow zoom (< _RATE_SWITCH_S): histogram spikes from raw per-unit data so
        # individual firing events are visible down to the 1 ms window.
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
            _bw = max(width / 200.0, 1e-4)   # aim for ~200 bins; floor at 0.1 ms
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
            # Normalise to the same [0, 1] scale as the pre-computed fine rate:
            # peak rate = _rate_peak counts / _FINE_BIN_S seconds;
            # on-the-fly rate density = cnt_sm / _bw counts/s.
            _rn = np.clip(_cnt_sm * _FINE_BIN_S / (_bw * _rate_peak), 0.0, 1.3)
        if len(_rt) >= 2:
            _fx = np.concatenate([[_rt[0]], _rt, [_rt[-1]]])
            _fy = np.concatenate([[0.0], _rn, [0.0]])
            rate_fill.set_xy(np.column_stack([_fx, _fy]))
        # PST crossing labels: only show when the crossing is within the current window.
        for _t_cross, _tx, _vl in _crossings:
            _vis = lo <= _t_cross <= hi
            _tx.set_visible(_vis)
            _vl.set_visible(_vis)

    def update(frame):
        width = widths[frame] if frame < n_zoom else window_end_s
        if zoom_mode == "center":
            lo, hi = _window_bounds(center_s, width, t_lo, t_hi)
        else:
            lo, hi = _window_bounds_left(anchor_s, width, t_lo, t_hi)
        draw(lo, hi)
        return scat, bar_line, bar_text, rate_fill

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
