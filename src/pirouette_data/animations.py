"""Spike-raster "Powers of Ten" zoom animation.

Renders a manually-curated spike raster (X = time, Y = unit ordered by depth, each
unit its own colour) and smoothly zooms OUT from a 10 ms window to the full ~36 h
recording -- like the *Powers of Ten* film. A dynamic scale bar at the bottom
switches through 1 ms / 1 s / 1 min / 1 hour / 12 hours as the view widens, and a
closing card reports how many orders of magnitude in time were spanned.

The units file (a ``good_units.pkl``: ``{unit_id: {"spike_times", "depth", ...}}``)
must provide a ``depth`` per unit so rows can be ordered "highest relative to the
probe at the top". Build an animation with :func:`make_animation`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# Scale-bar steps (seconds, label): the bar shows the largest of these that fits
# in the current view, so it "switches" as the zoom widens.
SCALE_STEPS: list[tuple[float, str]] = [
    (0.001, "1 ms"),
    (1.0, "1 s"),
    (60.0, "1 min"),
    (3600.0, "1 hour"),
    (43200.0, "12 hours"),
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

    Returns a list of ``{"id", "depth", "rank", "times"}`` sorted so ``rank`` 0 is
    at the bottom and the deepest unit is at the top (flip with *invert_depth*).
    """
    ids = sorted(units.keys(), key=lambda u: float(units[u]["depth"]), reverse=invert_depth)
    out = []
    for rank, u in enumerate(ids):
        t = np.asarray(units[u]["spike_times"], dtype="float64")
        out.append({"id": u, "depth": float(units[u]["depth"]), "rank": rank,
                    "times": np.sort(t)})
    return out


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


def frame_points(per_unit: list[dict], lo: float, hi: float, cap: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spikes within ``[lo, hi]`` across units, subsampled to ~*cap* total.

    Returns ``(times, ranks, rank_ints)`` -- x, jittered-y base (the unit rank),
    and the integer rank for colour lookup. Subsampling is proportional per unit,
    so relative firing density (and each unit's colour) is preserved; a narrow
    window keeps every spike (full resolution where you can see them).
    """
    slices = []
    total = 0
    for pu in per_unit:
        a = int(np.searchsorted(pu["times"], lo, "left"))
        b = int(np.searchsorted(pu["times"], hi, "right"))
        slices.append((a, b))
        total += b - a
    frac = 1.0 if total <= cap or total == 0 else cap / total
    xs, ranks = [], []
    for pu, (a, b) in zip(per_unit, slices):
        m = b - a
        if m == 0:
            continue
        if frac < 1.0:
            keep = max(1, int(m * frac))
            idx = np.linspace(a, b - 1, keep).astype("int64")
            tt = pu["times"][idx]
        else:
            tt = pu["times"][a:b]
        xs.append(tt)
        ranks.append(np.full(len(tt), pu["rank"], dtype="int64"))
    if not xs:
        empty = np.array([], dtype="float64")
        return empty, empty, np.array([], dtype="int64")
    x = np.concatenate(xs)
    r = np.concatenate(ranks)
    return x, r.astype("float64"), r


def make_animation(
    units_path: str | Path,
    out_path: str | Path,
    center_s: float | None = None,
    window_start_s: float = 0.01,
    window_end_s: float | None = None,
    duration_s: float = 20.0,
    hold_s: float = 3.0,
    fps: int = 30,
    cap: int = 60000,
    dpi: int = 120,
    invert_depth: bool = False,
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
        Initial (10 ms) and final window widths (default final: full recording).
    duration_s, hold_s, fps:
        Zoom-out length, closing-card hold, and frame rate.
    cap:
        Max spikes drawn per frame (subsampled; a wide view is sub-pixel dense).
    invert_depth:
        Flip the depth ordering if "highest at top" comes out upside-down.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.transforms import blended_transform_factory

    per_unit = prepare_units(load_units_with_depth(units_path), invert_depth=invert_depth)
    n_units = len(per_unit)
    _ = seed  # jitter is deterministic in x (see draw); seed kept for API stability

    t_lo = min(float(pu["times"][0]) for pu in per_unit if len(pu["times"]))
    t_hi = max(float(pu["times"][-1]) for pu in per_unit if len(pu["times"]))
    if center_s is None:
        center_s = 0.5 * (t_lo + t_hi)
    if window_end_s is None:
        window_end_s = max(window_start_s * 10, t_hi - t_lo)

    colors = matplotlib.colormaps["gist_rainbow"].resampled(n_units)(np.arange(n_units))

    n_zoom = max(2, int(round(duration_s * fps)))
    n_hold = max(1, int(round(hold_s * fps)))
    ratio = window_end_s / window_start_s
    widths = window_start_s * ratio ** (np.arange(n_zoom) / (n_zoom - 1))
    n_oom = orders_of_magnitude(window_start_s, window_end_s)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scat = ax.scatter([], [], s=3, c=[], edgecolors="none", rasterized=True)
    ax.set_ylim(-1, n_units)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit #  (deep → top)")
    ax.set_title(title)

    blend = blended_transform_factory(ax.transData, ax.transAxes)
    (bar_line,) = ax.plot([], [], color="black", lw=3, transform=blend,
                          solid_capstyle="butt")
    bar_text = ax.text(0, 0.075, "", transform=blend, ha="center", va="bottom",
                       fontsize=11, color="black")
    end_text = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center",
                       va="center", fontsize=18, color="black", alpha=0.0,
                       bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    def draw(lo, hi):
        x, y, r = frame_points(per_unit, lo, hi, cap)
        if len(x):
            # Small band spread, deterministic in x so it's stable across frames.
            yj = y + 0.42 * np.sin(x * 997.0)
            scat.set_offsets(np.column_stack([x, yj]))
            scat.set_color(colors[r])
        else:
            scat.set_offsets(np.empty((0, 2)))
        ax.set_xlim(lo, hi)
        width = hi - lo
        dur, label = pick_scale_bar(width)
        x0 = lo + 0.05 * width
        bar_line.set_data([x0, x0 + dur], [0.05, 0.05])
        bar_text.set_position((x0 + dur / 2.0, 0.075))
        bar_text.set_text(label)

    def update(frame):
        if frame < n_zoom:
            width = widths[frame]
            lo, hi = _window_bounds(center_s, width, t_lo, t_hi)
            draw(lo, hi)
            end_text.set_alpha(0.0)
        else:
            lo, hi = _window_bounds(center_s, window_end_s, t_lo, t_hi)
            draw(lo, hi)
            a = min(1.0, (frame - n_zoom + 1) / max(1, n_hold * 0.5))
            end_text.set_alpha(a)
            end_text.set_text(
                f"Spanned {n_oom:.1f} orders of magnitude in time\n"
                f"(10 ms → {window_end_s / 3600:.0f} hours)"
            )
        return scat, bar_line, bar_text, end_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_zoom + n_hold, interval=1000 / fps, blit=False
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=4000),
                  dpi=dpi)
    plt.close(fig)
    return out_path
