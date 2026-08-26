"""Probe schematic visualization helpers (Neuropixels 2.0).

Provides :class:`ProbeGeometry` (one-shank geometry) and
:func:`draw_probe_schematic` / :func:`make_probe_figure` for rendering
publication-quality probe schematics with optional per-site colour coding.

Coordinate convention
---------------------
All lengths are in **micrometres (µm)**.  The origin (0, 0) sits at the probe
tip (the bottom-most point of the shank).  *y* increases toward the brain
surface, i.e. upward in the default plot orientation.

Site ordering
-------------
Sites are stored column-by-column (left → right), bottom → top within each
column.  Indices 0 … (n_per_col − 1) belong to column 0 (left shank side);
indices n_per_col … n_sites − 1 belong to column 1 (right shank side).

NP 2.0 default parameters (from spec sheet)
--------------------------------------------
    shank_width       = 70  µm
    tip_length        = 175 µm  (tapered region from tip point to rectangular body)
    site_width/height = 12 × 12 µm
    vertical_pitch    = 15  µm  (front-edge to front-edge = center-to-center)
    horizontal_pitch  = 32  µm  (right-edge to right-edge = center-to-center)
    n_sites           = 96  (48 per column, y-aligned)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_SHANK_FC = "#FFFFFF"   # white shank fill
_DEFAULT_SHANK_EC = "#000000"   # black shank outline
_DEFAULT_SITE_FC  = "#FFFFFF"   # white site fill
_DEFAULT_SITE_EC  = "#000000"   # black site edge


# ---------------------------------------------------------------------------
# Probe geometry
# ---------------------------------------------------------------------------

@dataclass
class ProbeGeometry:
    """Physical dimensions (µm) of one shank of a Neuropixels 2.0 probe.

    All linear quantities are in micrometres.  The defaults match the NP 2.0
    spec sheet.
    """

    shank_width: float = 70.0
    """Full width of the rectangular shank body."""

    tip_length: float = 175.0
    """Length of the tapered tip region (tip point → rectangular body)."""

    site_width: float = 12.0
    """Width of one recording site."""

    site_height: float = 12.0
    """Height of one recording site."""

    vertical_pitch: float = 15.0
    """Front-edge-to-front-edge (≡ center-to-center for equal-height sites)
    vertical spacing between adjacent sites within a column."""

    horizontal_pitch: float = 32.0
    """Right-edge-to-right-edge (≡ center-to-center for equal-width sites)
    horizontal spacing between the two site columns."""

    n_sites: int = 96
    """Total number of recording sites to display."""

    n_columns: int = 2
    """Number of site columns on the shank."""

    first_site_offset: float = 20.0
    """Distance (µm) from the tip/body junction to the first site *centre*."""

    stagger: bool = False
    """If *True*, column 1 is shifted upward by ``vertical_pitch / 2`` relative
    to column 0 (checkerboard layout).  Default *False* keeps both columns
    aligned along y."""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_sites_per_column(self) -> int:
        """Number of sites in each column."""
        return self.n_sites // self.n_columns

    @property
    def column_x_centers(self) -> list[float]:
        """X-coordinate of each column centre, centred symmetrically on the shank."""
        total_span = (self.n_columns - 1) * self.horizontal_pitch
        left_x = (self.shank_width - total_span) / 2.0
        return [left_x + i * self.horizontal_pitch for i in range(self.n_columns)]

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def site_centers(self) -> np.ndarray:
        """Return ``(n_sites, 2)`` array of ``[x, y]`` site centres in µm.

        Sites are ordered column-by-column (left → right), bottom → top
        within each column.
        """
        n = self.n_sites_per_column
        rows: list[list[float]] = []
        for col_i, cx in enumerate(self.column_x_centers):
            stagger_off = (col_i * self.vertical_pitch / 2.0) if self.stagger else 0.0
            y0 = self.tip_length + self.first_site_offset + stagger_off
            for row_i in range(n):
                rows.append([cx, y0 + row_i * self.vertical_pitch])
        return np.asarray(rows, dtype=float)

    def shank_outline_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Polygon vertices ``(xs, ys)`` for the shank silhouette (µm).

        Traces counterclockwise:
        tip point → lower-left body corner → upper-left → upper-right →
        lower-right body corner → tip point.
        """
        w = self.shank_width
        t = self.tip_length
        all_y = self.site_centers()[:, 1]
        top_y = all_y.max() + self.site_height / 2.0 + self.first_site_offset
        xs = np.array([w / 2,  0.0,   0.0,    w,      w,      w / 2])
        ys = np.array([0.0,    t,     top_y,  top_y,  t,      0.0])
        return xs, ys


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _add_scale_bar(
    ax: plt.Axes,
    x0: float,
    y: float,
    bar_um: float,
    *,
    color: str = "#303030",
    lw: float = 1.0,
    tick_half: float = 3.0,
    fontsize: float = 5.0,
) -> None:
    """Draw a horizontal scale bar at *(x0, y)* of length *bar_um* µm."""
    x1 = x0 + bar_um
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw,
            solid_capstyle="butt", zorder=4, clip_on=False)
    for xv in (x0, x1):
        ax.plot([xv, xv], [y - tick_half, y + tick_half],
                color=color, linewidth=lw * 0.8, zorder=4, clip_on=False)
    ax.text(
        (x0 + x1) / 2, y - tick_half - 2,
        f"{bar_um:.0f} µm",
        ha="center", va="top", fontsize=fontsize, color=color,
        zorder=4, clip_on=False,
    )


# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------

def draw_probe_schematic(
    ax: plt.Axes,
    geometry: ProbeGeometry,
    *,
    site_values: np.ndarray | None = None,
    cmap: str | mcolors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    site_color: str = _DEFAULT_SITE_FC,
    site_edgecolor: str = _DEFAULT_SITE_EC,
    shank_facecolor: str = _DEFAULT_SHANK_FC,
    shank_edgecolor: str = _DEFAULT_SHANK_EC,
    shank_linewidth: float = 0.7,
    site_linewidth: float = 0.4,
    label_sites: Sequence[int] | None = None,
    label_fontsize: float = 4.5,
    annotate_columns: bool = False,
) -> mcolors.Normalize | None:
    """Draw the probe schematic onto *ax*.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw into.  The caller controls titles, axis
        labels, and figure layout.
    geometry:
        Probe geometry configuration.
    site_values:
        Per-site scalar array of shape ``(n_sites,)``.  When supplied, sites
        are colour-mapped; otherwise every site is drawn in *site_color*.
    cmap:
        Matplotlib colormap name or object (used only when *site_values* is
        given).
    vmin / vmax:
        Colour limits.  Default: data min / max.
    site_color:
        Uniform fill colour applied when *site_values* is not given.
    site_edgecolor:
        Edge colour for the site rectangles.
    shank_facecolor / shank_edgecolor:
        Fill and outline colours for the shank polygon.
    shank_linewidth / site_linewidth:
        Line widths for the shank outline and individual site edges.
    label_sites:
        Site indices whose channel number is printed inside the site square.
    label_fontsize:
        Font size for the site channel labels.
    annotate_columns:
        If *True*, draw "col 0" / "col 1" text below the lowest site of each
        column.

    Returns
    -------
    norm : Normalize | None
        The normalizer used for colour-mapping (useful for creating a
        colourbar), or *None* if *site_values* was not provided.
    """
    # --- Shank silhouette ---------------------------------------------------
    xs, ys = geometry.shank_outline_xy()
    ax.add_patch(mpatches.Polygon(
        np.column_stack([xs, ys]),
        closed=True,
        facecolor=shank_facecolor,
        edgecolor=shank_edgecolor,
        linewidth=shank_linewidth,
        zorder=1,
    ))

    # --- Colour mapping for sites ------------------------------------------
    norm: mcolors.Normalize | None = None
    mapper: mcm.ScalarMappable | None = None
    if site_values is not None:
        sv = np.asarray(site_values, dtype=float)
        lo = float(np.nanmin(sv)) if vmin is None else vmin
        hi = float(np.nanmax(sv)) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        mapper = mcm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mapper.set_array(sv)

    # --- Site rectangles ----------------------------------------------------
    centers = geometry.site_centers()
    hw = geometry.site_width / 2.0
    hh = geometry.site_height / 2.0
    label_set = set(label_sites) if label_sites is not None else set()

    for i, (cx, cy) in enumerate(centers):
        fc = mapper.to_rgba(site_values[i]) if mapper is not None else site_color
        ax.add_patch(mpatches.Rectangle(
            (cx - hw, cy - hh),
            geometry.site_width,
            geometry.site_height,
            facecolor=fc,
            edgecolor=site_edgecolor,
            linewidth=site_linewidth,
            zorder=2,
        ))
        if i in label_set:
            ax.text(
                cx, cy, str(i),
                ha="center", va="center",
                fontsize=label_fontsize,
                color="white",
                fontweight="bold",
                zorder=3,
            )

    # --- Column labels ------------------------------------------------------
    if annotate_columns and geometry.n_columns > 1:
        for col_i in range(geometry.n_columns):
            start_idx = col_i * geometry.n_sites_per_column
            cx_bot = centers[start_idx, 0]
            cy_bot = centers[start_idx, 1]
            ax.text(
                cx_bot, cy_bot - hh - 3.5,
                f"col {col_i}",
                ha="center", va="top",
                fontsize=max(label_fontsize - 0.5, 3.0),
                color="#505050",
                zorder=3,
            )

    return norm


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def make_probe_figure(
    geometry: ProbeGeometry | None = None,
    *,
    site_values: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str = "Value",
    title: str = "Neuropixels 2.0 — 1 shank, bottom 96 sites",
    fig_width: float = 2.5,
    dpi: int = 150,
    probe_label: str = "NP 2.0",
    scale_bar: bool = True,
    scale_bar_um: float = 100.0,
    x_margin: float = 35.0,
    y_margin: float = 30.0,
    **draw_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a new figure with the probe schematic and return *(fig, ax)*.

    The figure height is computed from the probe's physical aspect ratio so
    that ``ax.set_aspect("equal")`` fills the plot area without large margins.

    Parameters
    ----------
    geometry:
        Probe geometry; defaults to :class:`ProbeGeometry` (NP 2.0 spec).
    site_values:
        Optional per-site scalar array ``(n_sites,)`` for colour coding.
    cmap / vmin / vmax:
        Colour-map arguments forwarded to :func:`draw_probe_schematic`.
    colorbar:
        Add a colourbar when *site_values* is provided.
    colorbar_label:
        Colourbar axis label.
    title:
        Figure title.
    fig_width:
        Figure width in inches.
    dpi:
        Figure DPI (for raster formats).
    probe_label:
        Short italic label printed above the shank.
    scale_bar:
        Draw a horizontal scale bar to the right of the shank.
    scale_bar_um:
        Scale bar length in µm.
    x_margin / y_margin:
        Extra white space (µm) added around the shank outline.
    **draw_kwargs:
        Additional keyword arguments forwarded to :func:`draw_probe_schematic`.

    Returns
    -------
    (fig, ax)
    """
    if geometry is None:
        geometry = ProbeGeometry()

    xs_out, ys_out = geometry.shank_outline_xy()

    # --- Data-coordinate bounding box (µm) ---------------------------------
    # Include room for the scale bar on the right.
    x_lo = xs_out.min() - x_margin
    x_hi = xs_out.max() + (scale_bar_um + 20.0 if scale_bar else x_margin)
    y_lo = ys_out.min() - y_margin
    y_hi = ys_out.max() + y_margin

    data_w = x_hi - x_lo
    data_h = y_hi - y_lo

    # Figure height preserves the data aspect ratio.
    fig_height = min(fig_width * (data_h / data_w), 24.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")

    # --- Draw probe --------------------------------------------------------
    norm = draw_probe_schematic(
        ax, geometry,
        site_values=site_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **draw_kwargs,
    )

    # --- Scale bar ---------------------------------------------------------
    if scale_bar and scale_bar_um > 0:
        sb_x0 = xs_out.max() + 8.0
        sb_y  = ys_out.min() + geometry.tip_length * 0.6
        _add_scale_bar(ax, sb_x0, sb_y, scale_bar_um)

    # --- Probe-type label above shank --------------------------------------
    if probe_label:
        ax.text(
            geometry.shank_width / 2, ys_out.max() + 8.0,
            probe_label,
            ha="center", va="bottom",
            fontsize=6, fontstyle="italic", color="#404040",
        )

    # --- Axis limits and appearance ----------------------------------------
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("")
    ax.set_ylabel("Distance from tip (µm)", fontsize=7, labelpad=4)
    ax.tick_params(axis="y", labelsize=6)
    ax.xaxis.set_visible(False)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)

    if title:
        ax.set_title(title, fontsize=7.5, pad=5)

    # --- Colourbar ---------------------------------------------------------
    if colorbar and norm is not None and site_values is not None:
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.03, aspect=28)
        cbar.set_label(colorbar_label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_linewidth(0.5)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Waveform overlay
# ---------------------------------------------------------------------------

def _palette_rgba(n: int, palette: str = "glasbey") -> np.ndarray:
    """Return ``(n, 4)`` float32 RGBA array for *palette*.

    Mirrors :func:`pirouette_data.animations.unit_colors` exactly so colours
    match the raster animation when the same *palette* is used.

    Priority
    --------
    1. ``colorcet.b_glasbey_bw`` (if installed),
    2. Bundled ``pirouette_data/assets/glasbey1024.npz``,
    3. ``gist_rainbow`` fallback.
    """
    p = (palette or "glasbey").lower()
    if p in ("glasbey", "dartsort"):
        try:
            import colorcet as cc
            from matplotlib.colors import to_rgba_array
            src = cc.b_glasbey_bw
            return to_rgba_array([src[i % len(src)] for i in range(n)]).astype("float32")
        except ImportError:
            pass
        try:
            from importlib.resources import files as _files
            with np.load(
                str(_files("pirouette_data.assets").joinpath("glasbey1024.npz"))
            ) as npz:
                rgb = np.clip(
                    np.asarray(npz["glasbey1024"], dtype="float32")[:, :3], 0.0, 1.0
                )
                alpha = np.ones((len(rgb), 1), dtype="float32")
                arr = np.concatenate([rgb, alpha], axis=1)
                return arr[[i % len(arr) for i in range(n)]]
        except Exception:
            pass
        return plt.get_cmap("gist_rainbow")(np.linspace(0.0, 0.9, n)).astype("float32")

    try:
        return plt.get_cmap(p)(np.linspace(0.0, 0.9, n)).astype("float32")
    except Exception:
        return plt.get_cmap("gist_rainbow")(np.linspace(0.0, 0.9, n)).astype("float32")


def unit_rgba_map(
    units: dict,
    palette: str = "glasbey",
    invert_depth: bool = True,
) -> dict:
    """Return ``{unit_id: rgba_array(4,)}`` matching the raster-animation colour assignment.

    Units are sorted by depth (descending when *invert_depth* is ``True``,
    matching ``RASTER_INVERT_DEPTH=true``), and colour indices are assigned in
    that order — identical to :func:`pirouette_data.animations.prepare_units`.

    Parameters
    ----------
    units:
        Units dict loaded from the pickle file.
    palette:
        Colour palette name (default ``"glasbey"``).
    invert_depth:
        Sort deepest unit first (``cidx`` 0 = deepest).  Set to ``True`` to
        match the raster animation default.
    """
    ids = sorted(units, key=lambda u: float(units[u]["depth"]), reverse=invert_depth)
    rgba = _palette_rgba(len(ids), palette)
    return {uid: rgba[cidx] for cidx, uid in enumerate(ids)}


def pick_representative_waveform(unit: dict) -> tuple[np.ndarray, int]:
    """Select the channel at unit depth with the largest peak-to-trough amplitude.

    Algorithm
    ---------
    1. Filter ``waveform_channels`` to those whose y-coordinate in
       ``waveform_positions`` (column 1, depth in µm) matches ``unit["depth"]``.
    2. Among those channels, pick the one with the largest ``max − min`` in
       ``mean_waveform`` (peak-to-trough).
    3. Fall back to all channels if none match the depth exactly.

    Parameters
    ----------
    unit:
        Dict with keys ``mean_waveform`` ``(n_samples, n_ch)``,
        ``waveform_channels`` ``(n_ch,)``, ``waveform_positions`` ``(n_ch, 2)``,
        and ``depth`` (float).

    Returns
    -------
    waveform : ndarray, shape (n_samples,)
    channel_id : int
    """
    depth = float(unit["depth"])
    mw = np.asarray(unit["mean_waveform"], dtype=float)    # (n_samples, n_ch)
    wc = np.asarray(unit["waveform_channels"], dtype=int)   # (n_ch,)
    wp = np.asarray(unit["waveform_positions"], dtype=float)  # (n_ch, 2)

    at_depth = np.where(np.abs(wp[:, 1] - depth) < 1.0)[0]
    if len(at_depth) == 0:
        at_depth = np.arange(len(wc))

    best = int(at_depth[np.argmax(np.ptp(mw[:, at_depth], axis=0))])
    return mw[:, best].copy(), int(wc[best])


def _random_jitter_2d(
    n: int,
    n_x: int, step_x: float,
    n_y: int, step_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_offsets, y_offsets)`` arrays of length *n*.

    Offsets are drawn independently and uniformly at random within
    ``[−x_max, +x_max]`` and ``[−y_max, +y_max]``, where the half-extents
    are derived from the grid parameters:

    * ``x_max = (n_x − 1) / 2 × step_x``  (e.g. n_x=5, step_x=4 → ±8 µm)
    * ``y_max = (n_y − 1) / 2 × step_y``  (e.g. n_y=5, step_y=1.25 → ±2.5 µm)

    Pass a seeded :class:`numpy.random.Generator` for reproducible output.
    """
    x_max = (n_x - 1) / 2.0 * step_x
    y_max = (n_y - 1) / 2.0 * step_y
    x_off = rng.uniform(-x_max, x_max, n)
    y_off = rng.uniform(-y_max, y_max, n)
    return x_off, y_off


def overlay_waveforms(
    ax: plt.Axes,
    units: dict,
    geometry: ProbeGeometry,
    *,
    palette: str = "glasbey",
    invert_depth: bool = True,
    unit_color_map: dict | None = None,
    wf_x_span_um: float = 14.0,
    wf_amp_um: float = 10.0,
    relative_amp: bool = False,
    jitter_step_um: float = 4.0,
    n_jitter: int = 5,
    jitter_y_um: float = 4.0,
    n_jitter_y: int = 3,
    upsample: int = 1,
    linewidth: float = 0.5,
    alpha: float = 0.85,
    seed: int = 0,
) -> dict:
    """Overlay mean waveforms for every unit on *ax*.

    Each unit is represented by its best-amplitude waveform at the probe depth
    ``unit["depth"]``.  The recording time axis maps to the horizontal axis of
    the schematic, centred on the left or right column depending on whether the
    selected channel ID is even (left) or odd (right).  When *relative_amp* is
    ``True`` waveforms are scaled relative to the largest-amplitude unit in the
    dataset (that unit reaches *wf_amp_um*; all others are proportionally
    smaller).  When *relative_amp* is ``False`` (default) every waveform is
    drawn at the same visual height (*wf_amp_um*).  Random uniform x and y
    jitter (seeded for reproducibility) is applied within each column to
    minimise overlap between nearby units.

    Parameters
    ----------
    ax:
        Axes containing the probe schematic (from :func:`make_probe_figure`).
    units:
        Units dict loaded from the pickle file.
    geometry:
        Probe geometry (determines left/right column x centres).
    palette:
        Colour palette — must match the raster-animation palette for identical
        colours (default ``"glasbey"``).
    invert_depth:
        Sort deepest unit first when assigning colour indices (default ``True``,
        matching ``RASTER_INVERT_DEPTH=true``).
    wf_x_span_um:
        Total horizontal extent of each waveform in µm.
    wf_amp_um:
        Peak-to-trough amplitude in µm.  When *relative_amp* is ``False`` (default)
        every waveform is drawn at this height.  When *relative_amp* is ``True``
        this is the height of the largest-amplitude unit; others scale down
        proportionally.
    relative_amp:
        If ``True``, scale waveform heights relative to the dataset-wide maximum
        peak-to-trough amplitude.  If ``False`` (default), all waveforms are drawn
        at the same height (*wf_amp_um*).
    seed:
        Integer seed for the random-jitter RNG (default ``0``).  Change to get a
        different but still reproducible layout.
    jitter_step_um:
        Horizontal step between successive x jitter positions in µm.
    n_jitter:
        Number of distinct x jitter positions to cycle through.
    jitter_y_um:
        Step between y jitter positions in µm — shifts baselines apart so
        co-located units do not stack exactly on top of each other.
    n_jitter_y:
        Number of distinct y jitter positions (default 3: −jitter_y, 0, +jitter_y).
    upsample:
        Upsample factor applied to each waveform before rendering (default 4 ×
        the 150-sample recording → 600 display points) for smoother curves.
    linewidth:
        Waveform line width (pts).
    alpha:
        Waveform line opacity.

    Returns
    -------
    color_map : dict
        ``{unit_id: rgba_array}`` — useful for building a legend.
    """
    color_map = (
        unit_color_map
        if unit_color_map is not None
        else unit_rgba_map(units, palette=palette, invert_depth=invert_depth)
    )
    rng = np.random.default_rng(seed)

    # --- Group units by column -----------------------------------------------
    col_groups: dict[int, list] = {c: [] for c in range(geometry.n_columns)}
    for uid, udata in units.items():
        try:
            wf, ch_id = pick_representative_waveform(udata)
        except Exception:
            continue
        col_groups[ch_id % geometry.n_columns].append((uid, udata, wf))

    # --- Amplitude normalisation denominator --------------------------------
    # relative_amp=True  → divide by dataset-wide max ptp so amplitudes are
    #                       proportional (largest unit fills wf_amp_um).
    # relative_amp=False → divide per-unit by its own ptp (uniform height).
    if relative_amp:
        all_ptps = [np.ptp(wf) for group in col_groups.values() for _, _, wf in group]
        global_max_ptp = max(all_ptps) if all_ptps else 1.0
        if global_max_ptp < 1e-9:
            global_max_ptp = 1.0
    else:
        global_max_ptp = None  # use per-unit ptp below

    # --- Draw waveforms per column ------------------------------------------
    # Pre-build upsampled time grid (unit-normalised 0→1)
    t_orig = np.linspace(0.0, 1.0, 150)           # assume 150 recording samples
    n_up   = max(150, 150 * upsample)
    t_up   = np.linspace(0.0, 1.0, n_up)

    for col, group in col_groups.items():
        x_off, y_off = _random_jitter_2d(
            len(group),
            n_jitter, jitter_step_um,
            n_jitter_y, jitter_y_um,
            rng,
        )
        cx = geometry.column_x_centers[col]
        half_span = wf_x_span_um / 2.0

        for (uid, udata, wf), jx, jy in zip(group, x_off, y_off):
            depth = float(udata["depth"])
            color = color_map[uid]

            ptp = np.ptp(wf)
            if ptp < 1e-9:
                continue
            denom = global_max_ptp if global_max_ptp is not None else ptp
            wf_norm = (wf - wf.mean()) / denom * wf_amp_um

            # Upsample for smoother display curves
            t_src = np.linspace(0.0, 1.0, len(wf_norm))
            wf_up = np.interp(t_up, t_src, wf_norm)

            # x: time axis → µm, centred at column centre + x jitter
            x_c = cx + jx
            xs = np.linspace(x_c - half_span, x_c + half_span, n_up)
            # y: amplitude around unit depth + y jitter (separates baselines)
            ys = wf_up + depth + jy

            ax.plot(
                xs, ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=5,
            )

    return color_map
