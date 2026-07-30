"""Interactive Dash GUI for exploring a Pirouette dataset with ephys spikes.

Loads a per-frame dataset (``.parquet`` / ``.pkl`` / ``.csv`` from
:mod:`pirouette_data`) together with a ``good_units.pkl`` spike-times file, and
serves a browser app that shows, side by side:

* the tracked video frame (scrubbable with a slider), with the frame number,
  time since experiment start, and Pacific wall-clock time, and
* a stack of time-aligned plots with a red cursor at the current frame:
  behaviour (rest/movement), smoothed velocity, heading (commutator + ear
  vector), head position (inferno-coloured by time, windowed), the selected
  unit's spike raster, and its instantaneous firing rate.

Spike times are shifted by ``spike_offset_s`` so they are referenced to the
**start of the experiment** (the dataset's ``time_since_start`` origin) rather
than the ephys/processing-chunk start.

The app is hosted on the machine that holds the data; viewers explore it in a
browser (LAN, or via a tunnel such as ngrok) without needing the files locally.

Run with :func:`run` or ``scripts/run_gui.py``.
"""

from __future__ import annotations

import base64
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ephys

# Firing rate lives in the ephys module now; re-exported for backwards-compat.
from .ephys import instantaneous_firing_rate

# --- Column names produced by the build pipeline ---
COL_TIME = "time_since_start"
COL_DATETIME = "datetime_pacific"
COL_SOURCE = "source_file"
COL_FRAME = "frame"
COL_BEHAVIOR = "behavior"
COL_VELOCITY = "ear_velocity_smooth_mm_s"
COL_EAR_HEADING = "ear_heading_deg"
COL_COMM_HEADING = "commutator_heading_deg"
LEFT_EAR = "left_ear"
RIGHT_EAR = "right_ear"

REST_COLOR = "#9e9e9e"  # gray
MOVE_COLOR = "#fa8072"  # salmon
CURSOR_COLOR = "#e53935"  # red

MAX_PLOT_POINTS = 12000  # timeseries downsample target
MAX_RASTER_SPIKES = 40000  # cap markers in the spike raster (subsample if more)
# Precomputed firing-rate cache resolution over each unit's RAW spike range. Kept
# well above MAX_PLOT_POINTS so that a pose-window slice (often a fraction of the
# raw range) still has >= MAX_PLOT_POINTS points to downsample from for display.
FR_CACHE_POINTS = 60000


# ---------------------------------------------------------------------------
# Data layer (no Dash dependency — unit tested)
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a per-frame dataset from ``.parquet``, ``.pkl``, or ``.csv``.

    Parameters
    ----------
    path:
        Dataset file path.
    columns:
        If given, load only these columns (intersected with those present) for
        ``.parquet``/``.csv``. ``.pkl`` always loads in full.

    Returns
    -------
    pandas.DataFrame
        The dataset with ``datetime_pacific`` parsed as a datetime.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        # Read only the columns the GUI needs -- these datasets are wide (45 cols,
        # GBs); selecting ~20 columns roughly halves read time and memory.
        if columns is not None:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema.names)
            use = [c for c in columns if c in available]
            df = pd.read_parquet(path, columns=use or None)
        else:
            df = pd.read_parquet(path)
    elif suffix == ".pkl":
        df = pd.read_pickle(path)  # pickled frames can't be column-filtered cheaply
    elif suffix == ".csv":
        if columns is not None:
            import csv as _csv

            with open(path, newline="") as fh:
                header = next(_csv.reader(fh))
            use = [c for c in columns if c in header]
            df = pd.read_csv(path, usecols=use or None)
        else:
            df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix} ({path})")

    if COL_DATETIME in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df[COL_DATETIME]
    ):
        # ISO8601 handles the tz offset + variable fractional-second precision
        # written to CSV.
        df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME], format="ISO8601")
    return df.reset_index(drop=True)


def load_units(path: str | Path) -> dict:
    """Load a spike-times dict (``{unit_id: {'spike_times', 'amp', ...}}``).

    Parameters
    ----------
    path:
        Path to a pickle file (e.g. ``good_units.pkl``).

    Returns
    -------
    dict
        The unpickled units mapping.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def unit_ids(units: dict) -> list:
    """Return the sorted unit identifiers in *units*."""
    return sorted(units.keys())


def experiment_start_datetime(df: pd.DataFrame) -> pd.Timestamp:
    """Pacific datetime of the experiment start (``time_since_start == 0``).

    Derived from the first row: ``datetime_pacific - time_since_start`` so it is
    correct even when the dataset does not begin at the experiment origin.
    """
    return df[COL_DATETIME].iloc[0] - pd.Timedelta(seconds=float(df[COL_TIME].iloc[0]))


def unit_spike_times_experiment(
    units: dict, unit_id, spike_offset_s: float = 0.0
) -> np.ndarray:
    """Spike times of *unit_id* referenced to the experiment start (seconds).

    Parameters
    ----------
    units:
        Units mapping from :func:`load_units`.
    unit_id:
        Key into *units*.
    spike_offset_s:
        Constant offset (seconds) added to the raw spike times to move them from
        the ephys/processing-chunk reference onto the experiment timeline
        (``= ephys_start - experiment_start``).

    Returns
    -------
    numpy.ndarray
        Spike times in seconds since the experiment start.
    """
    spikes = np.asarray(units[unit_id]["spike_times"], dtype="float64")
    return spikes + float(spike_offset_s)


def spikes_to_datetime(
    spike_exp_s: np.ndarray, exp_start_dt: pd.Timestamp
) -> pd.DatetimeIndex:
    """Convert experiment-referenced spike seconds to Pacific datetimes."""
    return exp_start_dt + pd.to_timedelta(np.asarray(spike_exp_s), unit="s")


def behavior_bouts(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Contiguous behaviour bouts as ``(start_dt, end_dt, label)`` tuples."""
    if COL_BEHAVIOR not in df.columns:
        return []
    labels = df[COL_BEHAVIOR].to_numpy()
    times = df[COL_DATETIME].to_numpy()
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    bouts = []
    for s, e in zip(starts, ends):
        bouts.append((times[s], times[e - 1], str(labels[s])))
    return bouts


CHAMBER_CORNERS = ["ul_champber", "ur_champber", "lr_chamber", "ll_chamber"]


def gui_columns() -> list[str]:
    """Columns the GUI actually reads -- used to load wide datasets faster."""
    cols = [COL_TIME, COL_DATETIME, COL_SOURCE, COL_FRAME, COL_BEHAVIOR,
            COL_VELOCITY, COL_EAR_HEADING, COL_COMM_HEADING]
    for ear in (LEFT_EAR, RIGHT_EAR):
        cols += [f"{ear}_x_mm", f"{ear}_y_mm"]
    for corner in CHAMBER_CORNERS:
        cols += [f"{corner}_x_mm", f"{corner}_y_mm"]
    return cols


def chamber_corners_mm(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Median (x, y) mm position of each chamber corner (empty if columns absent)."""
    corners: dict[str, tuple[float, float]] = {}
    for corner in CHAMBER_CORNERS:
        xc, yc = f"{corner}_x_mm", f"{corner}_y_mm"
        if xc in df.columns and yc in df.columns:
            corners[corner] = (
                float(np.nanmedian(df[xc])),
                float(np.nanmedian(df[yc])),
            )
    return corners


def head_position_mm(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ear-midpoint head position (mm): mean of the left/right ear mm columns."""
    lx = df[f"{LEFT_EAR}_x_mm"].to_numpy(dtype="float64")
    ly = df[f"{LEFT_EAR}_y_mm"].to_numpy(dtype="float64")
    rx = df[f"{RIGHT_EAR}_x_mm"].to_numpy(dtype="float64")
    ry = df[f"{RIGHT_EAR}_y_mm"].to_numpy(dtype="float64")
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def _stride(n: int, max_points: int = MAX_PLOT_POINTS) -> int:
    """Stride to downsample *n* points to about *max_points*."""
    return max(1, int(np.ceil(n / max_points)))


def video_path_for_row(df: pd.DataFrame, row: int, video_dir: str | Path) -> Path:
    """Path to the mp4 for a dataset row (``<source_file>.mp4``)."""
    return Path(video_dir) / f"{df[COL_SOURCE].iloc[row]}.mp4"


def frame_index_for_row(df: pd.DataFrame, row: int) -> int:
    """Per-file (video) frame index for a dataset row."""
    return int(df[COL_FRAME].iloc[row])


def segments(df: pd.DataFrame) -> list[str]:
    """Ordered unique ``source_file`` values (one per video segment)."""
    return list(dict.fromkeys(df[COL_SOURCE].tolist()))


def segment_info(df: pd.DataFrame, source_file: str) -> tuple[int, int, float]:
    """Return ``(base_row, n_frames, fps)`` for a video segment.

    ``base_row`` is the dataset row index of the segment's first frame, so a
    within-video time ``t`` maps to global row ``base_row + round(t * fps)``.
    """
    idx = np.flatnonzero(df[COL_SOURCE].to_numpy() == source_file)
    base = int(idx[0])
    n = int(idx.size)
    if n > 1:
        dt = float(np.median(np.diff(df[COL_TIME].to_numpy()[idx])))
        fps = 1.0 / dt if dt > 0 else 60.0
    else:
        fps = 60.0
    return base, n, fps


def segment_options(
    segs: list[str], video_dir: str | Path
) -> tuple[list[dict], str | None]:
    """Dropdown options for the hour segments + the default selection.

    Hours whose ``<name>.mp4`` is missing from ``video_dir`` are labelled
    ``"<name> — Not Available"`` and disabled; the default is the first hour that
    does have a video (so playback works out of the box).
    """
    video_dir = Path(video_dir)
    have = {s: (video_dir / f"{s}.mp4").exists() for s in segs}
    options = [
        {"label": s if have[s] else f"{s} — Not Available",
         "value": s, "disabled": not have[s]}
        for s in segs
    ]
    default = next((s for s in segs if have[s]), segs[0] if segs else None)
    return options, default


def build_segment_table(df: pd.DataFrame) -> list[tuple[str, int, int, float]]:
    """Ordered ``(name, base_row, n_frames, fps)`` per segment, in ONE pass.

    Segments are contiguous blocks of ``source_file``; computing them all at once
    avoids rescanning the (multi-million-row) source column once per segment,
    which dominated load time on large datasets.
    """
    src = df[COL_SOURCE].to_numpy()
    t = df[COL_TIME].to_numpy(dtype="float64")
    codes = pd.factorize(df[COL_SOURCE], sort=False)[0]  # int codes: fast compare
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(df)]))
    table: list[tuple[str, int, int, float]] = []
    for a, b in zip(starts, ends):
        n = int(b - a)
        if n > 1:
            dt = float(np.median(np.diff(t[a:b])))
            fps = 1.0 / dt if dt > 0 else 60.0
        else:
            fps = 60.0
        table.append((str(src[a]), int(a), n, fps))
    return table


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
class FrameReader:
    """Caches ``cv2.VideoCapture`` objects and reads frames by index.

    ``cv2.VideoCapture`` is not thread-safe: concurrent ``set``/``read`` from the
    threaded web server (which happens during playback) crashes FFmpeg. All
    access is therefore serialised with a lock.
    """

    def __init__(self, video_dir: str | Path):
        self.video_dir = Path(video_dir)
        self._caps: dict[str, object] = {}
        self._lock = threading.Lock()

    def _capture(self, source_file: str):
        import cv2

        if source_file not in self._caps:
            path = self.video_dir / f"{source_file}.mp4"
            self._caps[source_file] = cv2.VideoCapture(str(path)) if path.exists() else None
        return self._caps[source_file]

    def frame(self, source_file: str, frame_index: int) -> "np.ndarray | None":
        """Return an RGB frame array, or ``None`` if unavailable (thread-safe)."""
        import cv2

        with self._lock:
            cap = self._capture(source_file)
            if cap is None or not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        with self._lock:
            for cap in self._caps.values():
                if cap is not None:
                    cap.release()
            self._caps.clear()


def frame_to_data_uri(rgb: "np.ndarray | None", placeholder_text: str = "") -> str:
    """Encode an RGB array as a base64 JPEG data URI (or a placeholder)."""
    import cv2

    if rgb is None:
        rgb = np.full((360, 480, 3), 40, dtype="uint8")
        cv2.putText(
            rgb,
            placeholder_text or "no video",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode("ascii") if ok else ""
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    """Server-side state shared across callbacks for the hosted session."""

    dataset_dir: Path
    units_dir: Path
    video_dir: Path
    spike_offset_s: float = 0.0
    show_all_spikes: bool = False
    firing_rate_bin_s: float = 0.05
    firing_rate_smooth_s: float = 0.2
    heading_mode: str = "vector"

    df: pd.DataFrame | None = None
    units: dict | None = None
    exp_start_dt: pd.Timestamp | None = None
    reader: FrameReader | None = None
    head_x: np.ndarray | None = None
    head_y: np.ndarray | None = None
    head_t: np.ndarray | None = None  # experiment seconds
    head_ms: np.ndarray | None = None  # wall-clock epoch ms (for datetime colour)
    chamber: dict | None = None
    unit_id: object = None
    bottom_cache: dict = field(default_factory=dict)
    segtable: list | None = None  # (name, base, n, fps) per segment, computed once
    fr_cache: dict | None = None  # precomputed firing rates for the loaded units
    autoselect_unit: object = None  # unit _load just auto-picked (consumed once)
    load_status_msg: str = ""  # last dataset "Loaded …" summary, to re-show
    load_counter: int = 0  # nonce so the load-info store always changes

    def load(self, dataset_path: str | Path, units_path: str | Path, offset_s: float):
        """Load a dataset + units file into the state."""
        self.df = load_dataset(dataset_path, columns=gui_columns())
        self.units = load_units(units_path)
        self.spike_offset_s = float(offset_s)
        self.exp_start_dt = experiment_start_datetime(self.df)
        self.reader = FrameReader(self.video_dir)
        self.head_x, self.head_y = head_position_mm(self.df)
        self.head_t = self.df[COL_TIME].to_numpy(dtype="float64")
        # Wall-clock epoch ms (tz-naive) for datetime colouring of the head plot.
        naive = self.df[COL_DATETIME].dt.tz_localize(None).to_numpy()
        self.head_ms = naive.astype("datetime64[ms]").astype("int64")
        self.chamber = chamber_corners_mm(self.df)
        self.unit_id = unit_ids(self.units)[0]
        self.bottom_cache = {}
        self.segtable = build_segment_table(self.df)  # one pass, reused everywhere
        # Precomputed firing rates (built before launch; recomputed here only if the
        # cache is missing/stale). Makes switching units near-instant.
        self.fr_cache = ephys.ensure_firing_rates(
            self.units, units_path, self.firing_rate_bin_s,
            self.firing_rate_smooth_s, FR_CACHE_POINTS,
        )
        return self


def _list_files(directory: Path, suffixes: tuple[str, ...]) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        str(p) for p in directory.iterdir() if p.suffix.lower() in suffixes
    )


def file_options(directory: Path, suffixes: tuple[str, ...]) -> list[dict]:
    """Dropdown options for files in *directory*, showing/sending only the file
    NAME (never the full path, which would leak the server's directory layout to
    viewers)."""
    return [{"label": Path(p).name, "value": Path(p).name}
            for p in _list_files(directory, suffixes)]


def resolve_in_dir(directory: Path, name: str | None) -> str | None:
    """Resolve a bare file *name* to a full path inside *directory*.

    Returns ``None`` unless the name is a real file directly in *directory* (so a
    dropdown value can only ever load a file from the configured folder -- no path
    traversal, and the client never sees or sends an absolute path).
    """
    if not name:
        return None
    base = Path(directory).resolve()
    path = (base / Path(name).name).resolve()
    if path.parent != base or not path.is_file():
        return None
    return str(path)


# ---------------------------------------------------------------------------
# Figure builders (Plotly)
# ---------------------------------------------------------------------------
def build_timeseries_top(df: pd.DataFrame, heading_mode: str = "vector"):
    """Behaviour + velocity + heading, shared time x-axis, with a red cursor.

    Parameters
    ----------
    heading_mode:
        Which heading trace(s) to plot: ``"vector"`` (ear-vector only, default),
        ``"commutator"`` (commutator only), or ``"both"``. The dashed/solid
        legend note is only added to the heading title when ``"both"`` is used.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    heading_title = "heading (deg)"
    if heading_mode == "both":
        heading_title += "  -  commutator: dashed, ear-vector: solid"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[0.08, 0.46, 0.46],
        subplot_titles=(
            "behaviour  (rest: gray, movement: salmon)",
            "smoothed velocity (mm/s)",
            heading_title,
        ),
    )

    # Behaviour as a thin 2-colour heatmap row. Plot as tz-naive wall-clock so
    # the client-side cursor (also naive) lines up exactly.
    s = _stride(len(df))
    x = df[COL_DATETIME].iloc[::s].dt.tz_localize(None)
    beh = (df[COL_BEHAVIOR].iloc[::s] == "movement").astype(int).to_numpy()
    fig.add_trace(
        go.Heatmap(
            x=x, z=[beh], colorscale=[[0, REST_COLOR], [1, MOVE_COLOR]],
            showscale=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=x, y=df[COL_VELOCITY].iloc[::s], line=dict(color="black", width=1),
            name="velocity",
        ),
        row=2, col=1,
    )
    if heading_mode in ("commutator", "both"):
        # Dashed only when both are shown (to distinguish); solid if alone.
        dash = "dash" if heading_mode == "both" else "solid"
        fig.add_trace(
            go.Scattergl(
                x=x, y=df[COL_COMM_HEADING].iloc[::s],
                line=dict(color="black", width=1, dash=dash), name="commutator",
            ),
            row=3, col=1,
        )
    if heading_mode in ("vector", "both"):
        fig.add_trace(
            go.Scattergl(
                x=x, y=df[COL_EAR_HEADING].iloc[::s],
                line=dict(color="black", width=1), name="ear vector",
            ),
            row=3, col=1,
        )

    x0 = x.iloc[0]
    # Hidden Plotly cursor (kept so the figure/tests still carry it); the VISIBLE
    # red cursor is a DOM overlay moved by CSS transform -- Plotly.relayout is far
    # too slow (100-200 ms) on these heavy figures to move a shape per frame.
    fig.add_shape(
        type="line", x0=x0, x1=x0, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=CURSOR_COLOR, width=2.5), opacity=0,
    )
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="mm/s", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=3, col=1)
    # Zoom only affects time (x); keep the y-scale constant.
    fig.update_yaxes(fixedrange=True)
    fig.update_annotations(font_size=12)  # subplot titles hold the labels
    fig.update_layout(
        autosize=True, margin=dict(l=55, r=25, t=30, b=20),
        showlegend=False, template="plotly_white", uirevision="ts-top",
    )
    return fig


def build_timeseries_bottom(
    spike_dt: pd.DatetimeIndex,
    rate_dt: pd.DatetimeIndex,
    rate: np.ndarray,
    x0,
    x_range,
    unit_label: str,
):
    """Spike raster + instantaneous firing rate, shared x, with a red cursor."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.4, 0.6],
        subplot_titles=(f"spikes (unit {unit_label})", "firing rate (Hz)"),
    )
    fig.add_trace(
        go.Scattergl(
            x=spike_dt, y=np.zeros(len(spike_dt)), mode="markers",
            marker=dict(symbol="line-ns-open", color="black", size=14, line=dict(width=1)),
            name="spikes", hoverinfo="x",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=rate_dt, y=rate, line=dict(color="#1565c0", width=1), name="rate"),
        row=2, col=1,
    )
    fig.add_shape(  # hidden; visible cursor is the DOM overlay (see build_layout)
        type="line", x0=x0, x1=x0, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=CURSOR_COLOR, width=2.5), opacity=0,
    )
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    # Zoom only affects time (x); keep the y-scale constant.
    fig.update_yaxes(fixedrange=True)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    fig.update_layout(
        autosize=True, margin=dict(l=55, r=25, t=30, b=20), showlegend=False,
        template="plotly_white", uirevision=f"ts-bottom-{unit_label}",
    )
    return fig


def _ms_colorbar_ticks(color_ms: np.ndarray, n: int = 4) -> dict:
    """Colorbar tick config that labels epoch-ms values as ``HH:MM:SS``."""
    cs = np.asarray(color_ms, dtype="float64")
    cs = cs[np.isfinite(cs)]
    if cs.size == 0:
        return {}
    lo, hi = float(cs.min()), float(cs.max())
    if hi <= lo:
        hi = lo + 1.0
    vals = np.linspace(lo, hi, n)
    txt = [pd.Timestamp(v, unit="ms").strftime("%H:%M:%S") for v in vals]
    return dict(tickmode="array", tickvals=vals.tolist(), ticktext=txt)


def build_head_position(
    head_x: np.ndarray,
    head_y: np.ndarray,
    color_ms: np.ndarray,
    current_row: int,
    window_s: float,
    fps: float = 60.0,
    chamber: dict | None = None,
):
    """Spatial head-position trail over a time window, inferno-coloured by time.

    Points are drawn as markers (no connecting line) coloured by wall-clock time
    (*color_ms* = epoch milliseconds), with a ``HH:MM:SS`` colorbar to match the
    timeseries plots. If *chamber* corner positions are given, a black box marks
    the chamber walls and the axes are bounded to it.
    """
    import plotly.graph_objects as go

    half = int(round(window_s * fps))
    lo = max(0, current_row - half)
    hi = min(len(head_x), current_row + 1)
    xs, ys, cs = head_x[lo:hi], head_y[lo:hi], color_ms[lo:hi]

    marker = dict(
        size=5, color=cs, colorscale="Inferno", showscale=True,
        colorbar=dict(title="time", thickness=12, **_ms_colorbar_ticks(cs)),
    )
    if len(cs):
        marker["cmin"], marker["cmax"] = float(np.min(cs)), float(np.max(cs))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=xs, y=ys, mode="markers", marker=marker,
                     name="trail", hoverinfo="skip")
    )
    if hi > lo:
        fig.add_trace(
            go.Scattergl(
                x=[head_x[current_row]], y=[head_y[current_row]], mode="markers",
                marker=dict(size=12, color=CURSOR_COLOR, line=dict(color="white", width=1)),
                name="current", hoverinfo="skip",
                # Hidden: the live current-position dot is a DOM overlay (#head-dot)
                # moved by CSS transform each frame, which the video can't outrun.
                opacity=0,
            )
        )

    order = ["ul_champber", "ur_champber", "lr_chamber", "ll_chamber"]
    if chamber and all(c in chamber for c in order):
        bx = [chamber[c][0] for c in order] + [chamber[order[0]][0]]
        by = [chamber[c][1] for c in order] + [chamber[order[0]][1]]
        fig.add_trace(
            go.Scatter(
                x=bx, y=by, mode="lines", line=dict(color="black", width=2),
                name="chamber", hoverinfo="skip",
            )
        )
        cxs = [chamber[c][0] for c in order]
        cys = [chamber[c][1] for c in order]
        mx = (max(cxs) - min(cxs)) * 0.01 + 1  # just a little larger than the chamber
        my = (max(cys) - min(cys)) * 0.01 + 1
        # Honour the tight ranges exactly (no equal-aspect padding); the axes hug
        # the chamber even if that means slightly non-proportional scaling.
        fig.update_xaxes(range=[min(cxs) - mx, max(cxs) + mx], constrain="domain")
        fig.update_yaxes(range=[max(cys) + my, min(cys) - my])  # image convention (y down)
    else:
        fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        autosize=True, margin=dict(l=55, r=20, t=30, b=20), showlegend=False,
        template="plotly_white", title="head position (mm), time-coloured",
        uirevision="head",
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
def create_app(
    dataset_dir: str | Path,
    units_dir: str | Path,
    video_dir: str | Path,
    spike_offset_s: float = 0.0,
    head_window_s: float = 10.0,
    show_all_spikes: bool = False,
    firing_rate_bin_s: float = 0.05,
    firing_rate_smooth_s: float = 0.2,
    heading_mode: str = "vector",
):
    """Build the Dash application.

    Parameters
    ----------
    dataset_dir:
        Directory scanned for dataset files (``.parquet`` / ``.pkl`` / ``.csv``).
    units_dir:
        Directory scanned for spike-times ``.pkl`` files.
    video_dir:
        Directory of ``<source_file>.mp4`` tracked videos.
    spike_offset_s:
        Default spike offset (seconds) to the experiment reference.
    head_window_s:
        Default head-position trail window (seconds).

    Returns
    -------
    dash.Dash
        The configured app. Call ``app.run(...)`` (or :func:`run`) to serve it.
    """
    from dash import Dash, Input, Output, State, dcc, html, no_update
    from flask import abort, send_file

    state = AppState(
        dataset_dir=Path(dataset_dir),
        units_dir=Path(units_dir),
        video_dir=Path(video_dir),
        spike_offset_s=spike_offset_s,
        show_all_spikes=show_all_spikes,
        firing_rate_bin_s=firing_rate_bin_s,
        firing_rate_smooth_s=firing_rate_smooth_s,
        heading_mode=heading_mode,
    )

    # Show only file NAMES in the dropdowns (values are names too) so the server's
    # absolute paths are never exposed to viewers.
    dataset_opts = file_options(state.dataset_dir, (".parquet", ".pkl", ".csv"))
    unit_opts = file_options(state.units_dir, (".pkl",))

    app = Dash(__name__, title="Pirouette explorer")

    # Crosshair cursor over the plots (override Plotly's ew/ns-resize cursor that
    # appears when the y-axis is fixed).
    app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      .js-plotly-plot .nsewdrag,
      .js-plotly-plot .nsewdrag.cursor-ew-resize,
      .js-plotly-plot .nsewdrag.cursor-ns-resize { cursor: crosshair !important; }
      /* Indeterminate loading bar (we can't get true % from the sync read). */
      .load-bar-track { position: relative; height: 8px; width: 100%;
        background: #e0e0e0; border-radius: 4px; overflow: hidden; }
      .load-bar-track .fill { position: absolute; top: 0; height: 100%;
        width: 40%; background: #1565c0; border-radius: 4px;
        animation: loadslide 1.1s ease-in-out infinite; }
      @keyframes loadslide { 0% { left: -40%; } 100% { left: 100%; } }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""

    # Serve the mp4 files to the browser's native <video> player (range requests
    # supported), so playback is decoded/buffered client-side and stays smooth
    # even over a tunnel.
    @app.server.route("/pirouette-video/<path:name>")
    def _serve_video(name):
        base = state.video_dir.resolve()
        path = (base / name).resolve()
        if path.parent != base or path.suffix.lower() != ".mp4" or not path.exists():
            abort(404)
        return send_file(str(path), mimetype="video/mp4", conditional=True)

    controls = html.Div(
        [
            html.Div([
                html.Label("Dataset"),
                dcc.Dropdown(id="dataset", options=dataset_opts,
                             value=dataset_opts[0]["value"] if dataset_opts else None,
                             clearable=False),
            ], style={"flex": "3"}),
            html.Div([
                html.Label("Spike units"),
                dcc.Dropdown(id="unitsfile", options=unit_opts,
                             value=unit_opts[0]["value"] if unit_opts else None,
                             clearable=False),
            ], style={"flex": "3"}),
            html.Div([
                html.Label("Spike offset (s)"),
                dcc.Input(id="offset", type="number", value=spike_offset_s, step=0.001,
                          debounce=True, style={"width": "100%"}),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Unit"),
                dcc.Dropdown(id="unit", options=[], clearable=False),
            ], style={"flex": "1"}),
            html.Div([
                # Status + progress bar sit directly above the Load button.
                html.Div(id="load-status", children="⏳ Loading…",
                         style={"fontSize": "11px", "color": "#555",
                                "minHeight": "14px", "marginBottom": "3px",
                                "lineHeight": "1.2"}),
                html.Div(html.Div(className="fill"), id="load-bar",
                         className="load-bar-track", style={"marginBottom": "5px"}),
                html.Button("Load", id="load", n_clicks=0, style={"width": "100%"}),
            ], style={"flex": "2"}),
        ],
        style={"display": "flex", "gap": "10px", "alignItems": "flex-end",
               "padding": "8px"},
    )

    # responsive: figures fill their flex cells and refit on window resize, so the
    # GUI auto-sizes to whatever screen it's shown on. The right column is a flex
    # column that fills the viewport height; the three plots share it by flex-grow
    # (so none gets clipped), each graph filling its cell at height 100%.
    graph_config = {"scrollZoom": True, "displaylogo": False, "responsive": True}
    graph_fill = {"height": "100%", "width": "100%"}
    wrap_top = {"position": "relative", "flex": "1.15 1 0", "minHeight": "0"}
    wrap_head = {"position": "relative", "flex": "1 1 0", "minHeight": "0"}
    wrap_bot = {"position": "relative", "flex": "1.1 1 0", "minHeight": "0"}

    # Equal-width columns: the video and the plots share the same horizontal
    # extent. The video sizes naturally to that width (no letterbox); the plots
    # are stacked taller for a bigger view.
    left = html.Div(
        [
            # Native HTML5 player: smooth, browser-buffered playback + scrubbing.
            # preload="auto" tells the browser to buffer ahead aggressively so
            # playback doesn't stall waiting on the next chunk.
            html.Video(
                id="video", controls=True, autoPlay=False, preload="auto",
                style={"width": "100%", "maxHeight": "55vh", "objectFit": "contain",
                       "border": "1px solid #ccc", "background": "#000"},
            ),
            html.Div(id="frame-info",
                     style={"fontFamily": "monospace", "padding": "6px 0",
                            "whiteSpace": "pre-line"}),
            # Global scrubber across the whole session; jumps to the right video.
            # updatemode="drag" so the red cursor follows the handle live.
            dcc.Slider(id="frame", min=0, max=1, step=1, value=0, marks=None,
                       tooltip={"placement": "bottom"}, updatemode="drag"),
            html.Div([
                html.Label("Segment (hour)"),
                dcc.Dropdown(id="segment", options=[], clearable=False,
                             style={"flex": "1"}),
                html.Label("Speed", style={"marginLeft": "10px"}),
                dcc.Dropdown(
                    id="speed",
                    options=[{"label": f"{s}x", "value": s}
                             for s in (0.25, 0.5, 1, 2, 4, 10)],
                    value=1, clearable=False, style={"width": "90px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                      "paddingTop": "8px"}),
            html.Div([
                html.Label("Head-trail window (s)"),
                dcc.Slider(id="window", min=1, max=120, step=1, value=head_window_s,
                           marks={1: "1", 30: "30", 60: "60", 120: "120"}),
            ], style={"paddingTop": "10px"}),
            html.Div([
                dcc.Checklist(
                    id="listen",
                    options=[{"label": " 🔊 listen to spikes", "value": "on"}],
                    value=[],
                    style={"fontSize": "16px", "fontWeight": "bold"},
                ),
                html.Label("Audio gain", style={"fontSize": "15px"}),
                dcc.Slider(id="gain", min=0, max=1, step=0.05, value=0.35,
                           marks={0: "0", 0.5: "0.5", 1: "1"}),
                html.Button("Plot Reset", id="plot-reset", n_clicks=0,
                            style={"marginTop": "12px", "width": "100%"}),
            ], style={"paddingTop": "14px"}),
            # Poll the video's playback time to drive the plot cursor + head plot
            # (fast, so they track the video closely).
            dcc.Interval(id="sync", interval=40, n_intervals=0),
            dcc.Store(id="seg"),
            dcc.Store(id="segmap"),
            dcc.Store(id="seek"),
            dcc.Store(id="segspikes"),
            # Server -> client "load finished" signal ({msg, n}); the visible
            # status/bar are driven ONLY by client-side JS (Dash can't reliably
            # share one output between a clientside and a server callback).
            dcc.Store(id="load-info"),
            html.Div(id="_dummy", style={"display": "none"}),
            html.Div(id="_dummy2", style={"display": "none"}),
            html.Div(id="_dummy3", style={"display": "none"}),
            html.Div(id="_dummy4", style={"display": "none"}),
            html.Div(id="_dummy5", style={"display": "none"}),
            html.Div(id="_dummy6", style={"display": "none"}),
            html.Div(id="_dummy7", style={"display": "none"}),
            html.Div(id="_dummy8", style={"display": "none"}),
            html.Div(id="_dummy9", style={"display": "none"}),
        ],
        style={"flex": "1", "minWidth": "560px", "padding": "8px",
               "overflowY": "auto", "minHeight": "0"},
    )

    right = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="ts-top", config=graph_config, style=graph_fill),
                    html.Div(id="cursor-top", style={
                        "position": "absolute", "left": "0", "top": "0",
                        "width": "2px", "height": "100%", "background": CURSOR_COLOR,
                        "transform": "translateX(-100px)", "pointerEvents": "none",
                        "display": "none", "zIndex": "10",
                    }),
                ],
                style=wrap_top,
            ),
            # The head graph is wrapped so a lightweight DOM dot can be overlaid on
            # it: the current-position marker is moved by a CSS transform (GPU
            # compositor) every animation frame, which tracks the video with no lag
            # -- Plotly.restyle can't keep up at high playback speed.
            html.Div(
                [
                    dcc.Graph(id="head", config=graph_config, style=graph_fill),
                    html.Div(id="head-dot", style={
                        "position": "absolute", "left": "0", "top": "0",
                        "boxSizing": "border-box",
                        "width": "13px", "height": "13px", "borderRadius": "50%",
                        "background": CURSOR_COLOR, "border": "1.5px solid white",
                        "boxShadow": "0 0 3px rgba(0,0,0,0.6)",
                        "transform": "translate(-100px,-100px)",
                        "pointerEvents": "none", "display": "none", "zIndex": "10",
                    }),
                ],
                style=wrap_head,
            ),
            html.Div(
                [
                    dcc.Graph(id="ts-bottom", config=graph_config, style=graph_fill),
                    html.Div(id="cursor-bot", style={
                        "position": "absolute", "left": "0", "top": "0",
                        "width": "2px", "height": "100%", "background": CURSOR_COLOR,
                        "transform": "translateX(-100px)", "pointerEvents": "none",
                        "display": "none", "zIndex": "10",
                    }),
                ],
                style=wrap_bot,
            ),
        ],
        style={"flex": "1", "padding": "8px", "display": "flex",
               "flexDirection": "column", "minWidth": "0", "minHeight": "0"},
    )

    app.layout = html.Div([
        html.H3("Pirouette dataset explorer", style={"padding": "0 8px", "margin": "6px 0"}),
        controls,
        # Fill the viewport below the title + controls; the right column's plots
        # flex to share this height (so the firing-rate plot is never clipped), and
        # the left column scrolls internally on short screens.
        html.Div(
            [left, right],
            style={"display": "flex", "gap": "8px", "alignItems": "stretch",
                   "height": "calc(100vh - 96px)", "minHeight": "420px"},
        ),
    ])

    # ---- helpers bound to state ----
    def _rate_window(t0, t1, offset):
        """Precomputed firing-rate points within ``[t0, t1]`` (experiment secs).

        The cache holds the rate over raw spike seconds; shift the shared bin
        centres by ``offset`` and slice to the window. Falls back to an on-the-fly
        compute if the cache is somehow missing.
        """
        fr = state.fr_cache
        rate = fr["rates"].get(str(state.unit_id)) if fr else None
        if fr is None or rate is None:
            spikes = unit_spike_times_experiment(state.units, state.unit_id, offset)
            centers, r = instantaneous_firing_rate(
                spikes, t0, t1, bin_s=state.firing_rate_bin_s,
                smooth_sigma_s=state.firing_rate_smooth_s,
            )
            rs = _stride(len(centers))
            return (spikes_to_datetime(centers[::rs], state.exp_start_dt)
                    .tz_localize(None), r[::rs])
        centers_exp = fr["centers_s"] + float(offset)
        mask = (centers_exp >= t0) & (centers_exp <= t1)
        cw, rw = centers_exp[mask], rate[mask]
        # The cache is finer than the display; downsample the window to the plot
        # target (matches the old shipped resolution).
        rs = _stride(len(cw))
        rate_dt = spikes_to_datetime(cw[::rs], state.exp_start_dt).tz_localize(None)
        return rate_dt, rw[::rs]

    def _bottom_figure():
        df = state.df
        key = (state.unit_id, round(float(state.spike_offset_s), 3))
        cached = state.bottom_cache.get(key)
        if cached is None:
            spikes = unit_spike_times_experiment(
                state.units, state.unit_id, state.spike_offset_s
            )
            t0, t1 = float(df[COL_TIME].iloc[0]), float(df[COL_TIME].iloc[-1])
            in_range = spikes[(spikes >= t0) & (spikes <= t1)]
            # Cap the raster: converting/rendering hundreds of thousands of ticks
            # is slow and unreadable; a uniform subsample looks the same. Set
            # show_all_spikes to render every tick (slower for busy units).
            if not state.show_all_spikes and in_range.size > MAX_RASTER_SPIKES:
                idx = np.linspace(0, in_range.size - 1, MAX_RASTER_SPIKES).astype("int64")
                in_range = in_range[idx]
            # tz-naive wall-clock so the client-side cursor lines up exactly.
            spike_dt = spikes_to_datetime(in_range, state.exp_start_dt).tz_localize(None)
            # Firing rate from the PRECOMPUTED cache: shift the shared raw-spike bin
            # centres by the current offset and slice to the visible window -- no
            # per-unit recompute (that ~1 s histogram+smooth was the bottleneck).
            rate_dt, rate_ds = _rate_window(t0, t1, state.spike_offset_s)
            cached = (spike_dt, rate_dt, rate_ds)
            state.bottom_cache[key] = cached
        spike_dt, rate_dt, rate_ds = cached
        x0 = df[COL_DATETIME].iloc[0].tz_localize(None)
        x_range = (
            df[COL_DATETIME].iloc[0].tz_localize(None),
            df[COL_DATETIME].iloc[-1].tz_localize(None),
        )
        return build_timeseries_bottom(
            spike_dt, rate_dt, rate_ds, x0, x_range, str(state.unit_id)
        )

    # ---- callbacks ----
    @app.callback(
        Output("ts-top", "figure"),
        Output("ts-bottom", "figure"),
        Output("unit", "options"),
        Output("unit", "value"),
        Output("segment", "options"),
        Output("segment", "value"),
        Output("frame", "max"),
        Output("frame", "value"),
        Output("frame", "marks"),
        Output("segmap", "data"),
        Output("load-info", "data"),
        Input("load", "n_clicks"),
        # dataset/units are State -> they load only when the Load button is pressed.
        State("dataset", "value"),
        State("unitsfile", "value"),
        State("offset", "value"),
        prevent_initial_call=False,
    )
    def _load(_clicks, dataset_name, units_name, offset):
        # Dropdown values are bare file names; resolve them to paths inside the
        # configured folders (guards against anything outside those folders).
        dataset_path = resolve_in_dir(state.dataset_dir, dataset_name)
        units_path = resolve_in_dir(state.units_dir, units_name)
        state.load_counter += 1
        if not dataset_path or not units_path:
            msg = "⚠ Select a dataset and a units file, then Load."
            return (no_update,) * 10 + ({"msg": msg, "n": state.load_counter},)
        state.load(dataset_path, units_path, offset or 0.0)
        options = [{"label": f"unit {u}", "value": u} for u in unit_ids(state.units)]
        top_fig = build_timeseries_top(state.df, heading_mode=state.heading_mode)
        # Use the precomputed segment table (one pass) instead of rescanning the
        # source column per segment. Global map: row -> (segment, fps, start time)
        # so the slider can move the cursor + seek the right video client-side.
        table = state.segtable or build_segment_table(state.df)
        dt_col = state.df[COL_DATETIME]
        segmap = []
        for name, base, n, fps in table:
            start_ms = int(dt_col.iloc[base].tz_localize(None).value // 1_000_000)
            segmap.append({"name": name, "base": base, "n": n, "fps": fps,
                           "startMs": start_ms})
        # Thin the tick labels so they never overlap: show at most ~MAX marks,
        # spaced evenly across the segments (scales with dataset size).
        MAX_MARKS = 8
        stepm = max(1, -(-len(table) // MAX_MARKS))  # ceil(len/MAX)
        marks = {
            int(base): pd.Timestamp(dt_col.iloc[base]).strftime("%H:%M")
            for i, (name, base, n, fps) in enumerate(table) if i % stepm == 0
        }
        segs = [name for name, _, _, _ in table]
        # Flag hours whose video file is missing (disabled + "Not Available") and
        # default to the first hour that has a video.
        seg_options, default_seg = segment_options(segs, state.video_dir)
        n_frames = len(state.df)
        frames_txt = (f"{n_frames / 1e6:.1f}M" if n_frames >= 1e6
                      else f"{n_frames:,}")
        n_avail = sum(1 for o in seg_options if not o["disabled"])
        status = (f"✓ Loaded · {len(segs)} hr ({n_avail} with video) · "
                  f"{frames_txt} frames")
        # The unit dropdown value changes here too, which fires _select_unit; mark
        # that auto-selection so it keeps this dataset summary rather than replacing
        # it with a "Loaded unit X" message.
        state.autoselect_unit = state.unit_id
        state.load_status_msg = status
        return (
            top_fig,
            _bottom_figure(),
            options,
            state.unit_id,
            seg_options,
            default_seg,
            len(state.df) - 1,
            0,
            marks,
            segmap,
            {"msg": status, "n": state.load_counter},
        )

    # The instant Load is clicked OR the dataset/units/unit is changed, flip to
    # "Loading…" + show the bar (client-side, so it appears before the server
    # callback runs; the server callback then writes the "Loaded" confirmation and
    # hides the bar). This is what makes changing a file/unit auto-load with visible
    # progress.
    app.clientside_callback(
        """
        function(n, unit) {
            var ctx = window.dash_clientside.callback_context;
            var trig = ctx && ctx.triggered && ctx.triggered[0];
            var id = (trig && trig.prop_id) ? trig.prop_id.split('.')[0] : '';
            var msg = (id === 'unit') ? '⏳ Loading unit…' : '⏳ Loading dataset…';
            // Write the DOM directly (the status/bar are client-owned; the server
            // signals completion via the load-info store -> the confirm callback).
            // Triggers: the Load button (dataset) and the unit dropdown -- NOT the
            // dataset/units file dropdowns (those load only on Load press).
            var s = document.getElementById('load-status');
            if (s) { s.textContent = msg; }
            var b = document.getElementById('load-bar');
            if (b) { b.style.display = 'block'; }
            return '';
        }
        """,
        Output("_dummy8", "children"),
        Input("load", "n_clicks"),
        Input("unit", "value"),
        prevent_initial_call=True,
    )

    # When a load finishes (server bumps load-info), write the confirmation text and
    # hide the bar -- client-side, so it never conflicts with the "Loading" writer.
    app.clientside_callback(
        """
        function(info) {
            if (info && info.msg) {
                var s = document.getElementById('load-status');
                if (s) { s.textContent = info.msg; }
                var b = document.getElementById('load-bar');
                if (b) { b.style.display = 'none'; }
            }
            return '';
        }
        """,
        Output("_dummy9", "children"),
        Input("load-info", "data"),
        prevent_initial_call=False,
    )

    def _segment_row(seg):
        """(base, n, fps) for a segment from the cached table (O(1)-ish)."""
        for name, base, n, fps in (state.segtable or []):
            if name == seg:
                return base, n, fps
        return segment_info(state.df, seg)  # fallback

    @app.callback(
        Output("video", "src"),
        Input("segment", "value"),
        prevent_initial_call=True,
    )
    def _load_video(seg):
        # Tiny, instant response: point the <video> at the chosen hour so it starts
        # loading immediately, WITHOUT waiting on the multi-MB head-data payload
        # (which ships from the separate callback below).
        if not seg:
            return no_update
        return f"/pirouette-video/{seg}.mp4"

    @app.callback(
        Output("seg", "data"),
        Output("head", "figure"),
        Input("segment", "value"),
        State("window", "value"),
        prevent_initial_call=True,
    )
    def _load_segment(seg, window_s):
        # Ship this segment's head-position data to the browser (so the head plot +
        # dot update client-side during playback) and build the initial head figure.
        if state.df is None or not seg:
            return no_update, no_update
        base, n, fps = _segment_row(seg)
        # tz-naive wall-clock ms so the client-side cursor aligns with the axis.
        start_ms = int(state.df[COL_DATETIME].iloc[base].tz_localize(None).value // 1_000_000)
        sl = slice(base, base + n)
        seg_store = {
            "base": base, "n": n, "fps": fps, "name": seg, "startMs": start_ms,
            # 0.1 mm precision is plenty for the trail and roughly halves the
            # payload vs 2 decimals (this ships on every segment switch).
            "hx": np.round(state.head_x[sl], 1).tolist(),
            "hy": np.round(state.head_y[sl], 1).tolist(),
            "ht": np.round(state.head_t[sl], 3).tolist(),
        }
        head_fig = build_head_position(
            state.head_x, state.head_y, state.head_ms, base,
            float(window_s or 10.0), chamber=state.chamber,
        )
        # Ship the head plot's home axis ranges so Plot Reset can restore the exact
        # tight view (its axes aren't auto-ranged).
        hx_rng = head_fig.layout.xaxis.range
        hy_rng = head_fig.layout.yaxis.range
        seg_store["head_xrange"] = list(hx_rng) if hx_rng else None
        seg_store["head_yrange"] = list(hy_rng) if hy_rng else None
        return seg_store, head_fig

    # Slider drag -> move the red cursor live (client-side, in sync with the
    # handle) and seek the video. Within the current hour the seek is client-side
    # too; crossing into another hour switches the video via the server.
    app.clientside_callback(
        """
        function(row, segmap, seg) {
            var nou = window.dash_clientside.no_update;
            if (row == null || !segmap || !segmap.length) { return [nou, nou]; }
            var s = segmap[segmap.length - 1];
            for (var k = 0; k < segmap.length; k++) {
                var m = segmap[k];
                if (row >= m.base && row < m.base + m.n) { s = m; break; }
            }
            var localT = (row - s.base) / s.fps;
            if (localT < 0) { localT = 0; }
            // Move the cursor immediately (DOM overlay) so it tracks the handle.
            if (window.__placeCursor) {
                window.__placeCursor(s.startMs + localT * 1000);
            }
            var curName = seg && seg.name;
            var v = document.getElementById('video');
            if (s.name === curName) {
                if (v) { try { v.currentTime = localT; } catch (e) {} }
                return [nou, nou];
            }
            // Different hour: load that video (server) + pending seek.
            return [s.name, {seg: s.name, t: localT}];
        }
        """,
        Output("segment", "value", allow_duplicate=True),
        Output("seek", "data"),
        Input("frame", "value"),
        State("segmap", "data"),
        State("seg", "data"),
        prevent_initial_call=True,
    )

    # Each tick: apply any pending seek (once the right video is ready) and read
    # the video's currentTime -> global row (client-side, cheap).
    # Everything that must track the video during playback is updated CLIENT-SIDE
    # (no server round-trips): the red cursor, the head-position trail, and the
    # frame-info text. Data for the current segment lives in the `seg` store.
    app.clientside_callback(
        """
        function(_n, seek, windowS, listen, segspikes) {
            var nou = window.dash_clientside.no_update;
            var v = document.getElementById('video');
            var seg = window.__seg;  // mirrored once per segment (not marshalled/tick)
            if (!v || !seg || !seg.hx) { return [nou, nou]; }
            var ct = v.currentTime || 0;
            var clearSeek = nou;
            // Apply a pending seek once the right video is loaded, then consume
            // it. Only jump if we're not already near the target, so we never
            // re-pin currentTime every tick (which would stall playback).
            if (seek && seek.seg && v.readyState >= 1 &&
                v.currentSrc && v.currentSrc.indexOf(seek.seg) >= 0) {
                if (Math.abs(ct - seek.t) > 0.25) { v.currentTime = seek.t; ct = seek.t; }
                clearSeek = null;
            }
            var moved = ct !== window.__lastCt;
            // Skip the heavy cursor/head/info work only when paused AND nothing
            // moved (keeps mouse-wheel zoom smooth). During playback the video is
            // not paused, so the cursor always advances.
            if (!moved && v.paused) { return [clearSeek, nou]; }
            window.__lastCt = ct;
            var nowT = (window.performance && performance.now)
                ? performance.now() : Date.now();
            var Plotly = window.Plotly;
            function plotDiv(id) {
                var el = document.getElementById(id);
                return el && (el.classList.contains('js-plotly-plot')
                    ? el : el.querySelector('.js-plotly-plot'));
            }

            // Current frame index off the video's own frame rate (as in place()).
            // Reference this frame's ACTUAL timestamp for the cursor + info, not a
            // linear startMs+ct*1000 -- the video duration is slightly longer than
            // the data span, so linear time drifts ahead of the data samples (and
            // the head dot) over the hour.
            var nFrames = seg.hx.length;
            var fpsEff = (v.duration && isFinite(v.duration) && v.duration > 0)
                ? (nFrames / v.duration) : seg.fps;
            var curIdx = Math.round(ct * fpsEff);
            if (curIdx < 0) { curIdx = 0; }
            if (curIdx > nFrames - 1) { curIdx = nFrames - 1; }
            // frameT = current frame's time from segment start in the DATA timeline
            // (what segspikes are referenced to); frameMs = its wall-clock ms.
            var frameT = seg.ht[curIdx] - seg.ht[0];
            var frameMs = seg.startMs + frameT * 1000;

            // Red time cursor: move the DOM overlay (cheap CSS transform, not a
            // 100-200 ms Plotly.relayout) to this frame's wall-clock time. rVFC
            // moves it per frame during playback; this 40 ms loop covers paused
            // seeks and is the fallback where rVFC is unavailable.
            //
            // Seek hold: right after a click the video's frame lags (it's still
            // seeking; a cross-segment click is also loading a new hour + buffering),
            // so tracking frameMs would snap the cursor back to the old/stale spot.
            // Pin the cursor to the clicked wall-clock time until the REAL frame time
            // (segment-aware -- frameMs uses the loaded segment's startMs) reaches it,
            // then resume tracking. A safety timeout clears a hold that never lands.
            if (window.__cursorHoldMs != null) {
                var reached = Math.abs(frameMs - window.__cursorHoldMs) <= 500;
                var tooLong = window.__cursorHoldT != null
                    && (nowT - window.__cursorHoldT) > 6000;
                if (reached || tooLong) {
                    window.__cursorHoldMs = null;
                    window.__cursorHoldT = null;
                }
            }
            var cursorAt = (window.__cursorHoldMs != null)
                ? window.__cursorHoldMs : frameMs;
            if (window.__placeCursor && typeof seg.startMs === 'number') {
                window.__placeCursor(cursorAt);
            }
            window.__cursorMs = cursorAt;

            // Auto-follow: when zoomed in and playing, PAGE the window forward only
            // when the cursor reaches the right edge (or lands outside after a
            // seek). This replaces the old centre-every-tick behaviour, which ran a
            // slow relayout ~16x/s and fought the user's own zoom/click. It's also
            // suppressed briefly after a manual zoom/pan/click so it doesn't yank
            // the view the user just set.
            var suppressed = window.__suppressFollowUntil
                && nowT < window.__suppressFollowUntil;
            if (Plotly && typeof seg.startMs === 'number' && !v.paused
                && !suppressed) {
                var sm = window.__segmap;
                var fullSpan = Infinity;
                if (sm && sm.length) {
                    var last = sm[sm.length - 1];
                    fullSpan = last.startMs + (last.n / last.fps) * 1000
                               - sm[0].startMs;
                }
                var tg = plotDiv('ts-top');
                if (tg && tg._fullLayout && tg._fullLayout.xaxis) {
                    var _toMs = function (x) {
                        if (typeof x === 'number') return x;
                        var s = String(x);
                        if (s.indexOf('Z') < 0 && s.indexOf('+') < 0) {
                            s = s.replace(' ', 'T') + 'Z';
                        }
                        return new Date(s).getTime();
                    };
                    var r0 = _toMs(tg._fullLayout.xaxis.range[0]);
                    var r1 = _toMs(tg._fullLayout.xaxis.range[1]);
                    var W = r1 - r0;
                    var curMs = frameMs;  // same frame-accurate time as the cursor
                    // Page only when the cursor nears/passes the right edge or is
                    // behind the window (after a seek) -- not while it's comfortably
                    // in view. New window puts the cursor ~10% from the left so most
                    // of the window shows what's coming.
                    if (W > 0 && W < 0.9 * fullSpan
                        && (curMs > r0 + 0.9 * W || curMs < r0)) {
                        var _fmt = function (ms) {
                            return new Date(ms).toISOString()
                                .replace('T', ' ').replace('Z', '');
                        };
                        var nr0 = curMs - 0.1 * W;
                        var rng = [_fmt(nr0), _fmt(nr0 + W)];
                        window.__xsyncKey = JSON.stringify(rng);
                        ['ts-top', 'ts-bottom'].forEach(function (gid) {
                            var gd = plotDiv(gid);
                            if (!gd || !gd.layout) return;
                            var upd = {};
                            ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'].forEach(
                                function (ax) {
                                    if (gd.layout[ax]) upd[ax + '.range'] = rng;
                                });
                            try { Plotly.relayout(gd, upd); } catch (e) {}
                        });
                    }
                }
            }

            // Head-position trail + current marker (windowed), using the shared
            // frame index (video frame rate) computed above.
            var n = nFrames;
            var i = curIdx;
            var win = Math.round((windowS || 10) * fpsEff);
            var lo = Math.max(0, i - win);
            var hgd = plotDiv('head');
            if (hgd && Plotly) {
                try {
                    // The current-position marker is updated on rAF (below). Here we
                    // only rebuild the trail + colorbar, which is much heavier -- run
                    // it ~10x/s while playing (every tick when paused, so scrubbing
                    // stays fresh) rather than every 40 ms tick.
                    var freshTrail = v.paused
                        || window.__lastTrailT === undefined
                        || (nowT - window.__lastTrailT) >= 100;
                    if (freshTrail) {
                        window.__lastTrailT = nowT;
                        var CAP = 200;
                        var step = Math.max(1, Math.ceil((i - lo + 1) / CAP));
                        var xs = [], ys = [], color = [];
                        for (var j = lo; j <= i; j += step) {
                            xs.push(seg.hx[j]); ys.push(seg.hy[j]);
                            color.push(seg.startMs + (seg.ht[j] - seg.ht[0]) * 1000);
                        }
                        if ((i - lo) % step !== 0) {  // always include current point
                            xs.push(seg.hx[i]); ys.push(seg.hy[i]);
                            color.push(seg.startMs + (seg.ht[i] - seg.ht[0]) * 1000);
                        }
                        var cmin = color[0], cmax = color[color.length - 1];
                        if (cmax <= cmin) { cmax = cmin + 1; }
                        var tv = [], tt = [], NT = 4;
                        for (var t = 0; t < NT; t++) {
                            var val = cmin + (cmax - cmin) * t / (NT - 1);
                            tv.push(val);
                            tt.push(new Date(val).toISOString().slice(11, 19));
                        }
                        Plotly.restyle(hgd, {
                            x: [xs], y: [ys], 'marker.color': [color],
                            'marker.cmin': [cmin], 'marker.cmax': [cmax],
                            'marker.cauto': [false],
                            'marker.colorbar.tickmode': ['array'],
                            'marker.colorbar.tickvals': [tv],
                            'marker.colorbar.ticktext': [tt],
                        }, [0]);
                    }
                } catch (e) { /* not ready */ }
            }

            // Frame info (client-side; frameMs is the current frame's wall-clock).
            var wall = new Date(frameMs).toISOString()
                .replace('T', ' ').replace('Z', '');
            var row = seg.base + i;
            var tss = seg.ht[i];
            var info = 'frame ' + row.toLocaleString() +
                '  (' + seg.name + ' #' + i + ')\\n' +
                't since start: ' + tss.toFixed(3) + ' s  (' +
                (tss / 3600).toFixed(3) + ' h)\\n' +
                'PST: ' + wall;

            // Audible spike monitor: schedule a click for each spike the cursor
            // crossed since the last tick, spread across the tick's real duration
            // so it crackles and stays in sync at any playback speed.
            var listenOn = listen && listen.indexOf && listen.indexOf('on') >= 0;
            if (listenOn && window.__audioCtx && window.__pop &&
                segspikes && segspikes.length && !v.paused) {
                var actx = window.__audioCtx;
                var nowP = (window.performance && performance.now)
                    ? performance.now() : Date.now();
                var realDt = (window.__lastTickPerf !== undefined)
                    ? (nowP - window.__lastTickPerf) : 40;
                window.__lastTickPerf = nowP;
                realDt = Math.min(Math.max(realDt, 5), 500) / 1000;  // seconds
                var prev = window.__lastSpikeCt;
                // Compare against the frame's DATA-timeline time (same reference as
                // segspikes), not the linear video time, so pops fire exactly when
                // the cursor crosses each spike. Forward-advance only (not a seek);
                // 8 s covers even 10x with laggy ticks, seeks are larger.
                if (prev !== undefined && frameT > prev && (frameT - prev) <= 8.0) {
                    var a = 0, b = segspikes.length;
                    while (a < b) {  // first index with segspikes[idx] > prev
                        var mm = (a + b) >> 1;
                        if (segspikes[mm] <= prev) a = mm + 1; else b = mm;
                    }
                    var span = frameT - prev, base = actx.currentTime, cnt = 0;
                    for (var si = a; si < segspikes.length && segspikes[si] <= frameT;
                         si++) {
                        if (cnt >= 60) break;  // avoid extreme bursts
                        var frac = span > 0 ? (segspikes[si] - prev) / span : 0;
                        window.__pop(actx, base + frac * realDt);
                        cnt++;
                    }
                }
                window.__lastSpikeCt = frameT;
            }
            return [clearSeek, info];
        }
        """,
        Output("seek", "data", allow_duplicate=True),
        Output("frame-info", "children"),
        Input("sync", "n_intervals"),
        State("seek", "data"),
        State("window", "value"),
        State("listen", "value"),
        State("segspikes", "data"),
        prevent_initial_call=True,
    )

    # Mirror the per-segment head data to a window global (once per segment, not
    # marshalled every tick) and drive the current head dot -- the one thing that
    # must update every video frame -- from a per-frame callback. Moving the dot is
    # a single CSS transform, so it keeps full frame rate without falling behind
    # (the heavier Plotly cursor/trail live on the 40 ms loop). rVFC fires once per
    # *presented* frame (fresh mediaTime); rAF+currentTime is the fallback.
    app.clientside_callback(
        """
        function(seg) {
            window.__seg = seg;
            // Head plot's home ranges for Plot Reset (updated once per segment).
            if (seg && seg.head_xrange && seg.head_yrange) {
                window.__headHome = {x: seg.head_xrange, y: seg.head_yrange};
            }
            if (window.__headSync) { return ''; }
            window.__headSync = true;
            function pdiv(id) {
                var el = document.getElementById(id);
                return el && (el.classList.contains('js-plotly-plot')
                    ? el : el.querySelector('.js-plotly-plot'));
            }
            // Move the red time cursor on both timeseries figures to wall-clock
            // `ms` by translating a DOM overlay line (compositor-only). This
            // replaces Plotly.relayout of a shape, which costs 100-200 ms on these
            // heavy figures and was the bulk of the click/playback latency. Hides
            // the line when the time is outside the current (zoomed) x-view.
            window.__placeCursor = function (ms) {
                try {
                    window.__cursorMs = ms;
                    var pairs = [['ts-top', 'cursor-top'], ['ts-bottom', 'cursor-bot']];
                    for (var pi = 0; pi < pairs.length; pi++) {
                        var g = pdiv(pairs[pi][0]);
                        var line = document.getElementById(pairs[pi][1]);
                        if (!g || !g._fullLayout || !g._fullLayout.xaxis || !line) {
                            continue;
                        }
                        var xa = g._fullLayout.xaxis, lay = g._fullLayout;
                        var toMs = function (x) {
                            if (typeof x === 'number') { return x; }
                            var t = String(x);
                            if (t.indexOf('Z') < 0 && t.indexOf('+') < 0) {
                                t = t.replace(' ', 'T') + 'Z';  // naive == UTC here
                            }
                            return new Date(t).getTime();
                        };
                        var r0 = toMs(xa.range[0]), r1 = toMs(xa.range[1]);
                        if (!(r1 > r0)) { continue; }
                        var frac = (ms - r0) / (r1 - r0);
                        if (frac < -0.002 || frac > 1.002) {
                            line.style.display = 'none';
                            continue;
                        }
                        var top = lay.margin ? lay.margin.t : 0;
                        var h = lay.height - (lay.margin
                            ? (lay.margin.t + lay.margin.b) : 0);
                        line.style.top = top + 'px';
                        line.style.height = h + 'px';
                        line.style.transform = 'translateX('
                            + (xa._offset + frac * xa._length) + 'px)';
                        line.style.display = 'block';
                    }
                } catch (e) {}
            };
            // Move ONLY the head dot for a media time `mt` (seconds). This is a
            // single CSS transform (compositor-only, no Plotly redraw), so it can
            // run for every presented video frame without falling behind. The
            // timeseries cursor is a Plotly shape moved on the slower 40 ms loop --
            // relayout is too heavy to call per frame and would starve this update.
            function place(mt) {
                try {
                    var s = window.__seg;
                    if (!s || !s.hx) { return; }
                    // Index off the VIDEO's own frame rate (frames / duration), not
                    // the data-derived s.fps -- the latter is slightly low here and
                    // makes the frame index drift behind the video (tens of frames
                    // deep into an hour). n/duration ties the dot to the frame on
                    // screen.
                    var vid = document.getElementById('video');
                    var n = s.hx.length;
                    var fpsEff = (vid && vid.duration && isFinite(vid.duration)
                        && vid.duration > 0) ? (n / vid.duration) : s.fps;
                    var i = Math.round(mt * fpsEff);
                    if (i < 0) { i = 0; }
                    if (i > n - 1) { i = n - 1; }
                    var hgd = pdiv('head');
                    var dot = document.getElementById('head-dot');
                    if (hgd && hgd._fullLayout && dot) {
                        var xa = hgd._fullLayout.xaxis;
                        var ya = hgd._fullLayout.yaxis;
                        if (xa && ya && xa._length && ya._length) {
                            var px = xa._offset + (s.hx[i] - xa.range[0])
                                / (xa.range[1] - xa.range[0]) * xa._length;
                            var py = ya._offset + (ya.range[1] - s.hy[i])
                                / (ya.range[1] - ya.range[0]) * ya._length;
                            dot.style.transform = 'translate('
                                + (px - 6.5) + 'px,' + (py - 6.5) + 'px)';
                            dot.style.display = 'block';
                        }
                    }
                    // Cursor at this frame's wall-clock time (aligns with the data
                    // and spike ticks, same as the head dot) -- but NOT while a click
                    // hold is active (the 40 ms sync loop pins the cursor to the click
                    // and clears the hold once the frame reaches it).
                    if (window.__placeCursor && window.__cursorHoldMs == null) {
                        window.__placeCursor(s.startMs + (s.ht[i] - s.ht[0]) * 1000);
                    }
                } catch (e) {}
            }
            var v = document.getElementById('video');
            if (v && typeof v.requestVideoFrameCallback === 'function') {
                // Frame-accurate: metadata.mediaTime is the presentation time of
                // the frame now on screen. Fires per presented frame (and once
                // after any seek/pause), so the dot tracks the visible frame.
                var vcb = function (now, metadata) {
                    place(metadata.mediaTime);
                    var vid = document.getElementById('video');
                    if (vid) { vid.requestVideoFrameCallback(vcb); }
                };
                v.requestVideoFrameCallback(vcb);
            } else {
                // Fallback: rAF reading currentTime (a few browsers lack rVFC).
                var raf = function () {
                    var vid = document.getElementById('video');
                    if (vid) {
                        var t = vid.currentTime || 0;
                        if (t !== window.__rafCt) { window.__rafCt = t; place(t); }
                    }
                    window.requestAnimationFrame(raf);
                };
                window.requestAnimationFrame(raf);
            }
            return '';
        }
        """,
        Output("_dummy7", "children"),
        Input("seg", "data"),
        prevent_initial_call=True,
    )

    # Apply the playback speed to the native player (client-side).
    app.clientside_callback(
        """
        function(s) {
            var v = document.getElementById('video');
            if (v && s) { v.playbackRate = s; }
            return '';
        }
        """,
        Output("_dummy", "children"),
        Input("speed", "value"),
    )

    # Arm the Web Audio spike monitor when "listen to spikes" is toggled on
    # (toggling is the user gesture that lets the browser start audio).
    app.clientside_callback(
        """
        function(val) {
            var on = val && val.indexOf && val.indexOf('on') >= 0;
            if (on) {
                if (!window.__audioCtx) {
                    var AC = window.AudioContext || window.webkitAudioContext;
                    if (AC) window.__audioCtx = new AC();
                }
                if (window.__audioCtx && window.__audioCtx.state === 'suspended') {
                    window.__audioCtx.resume();
                }
                window.__pop = function (ctx, when) {
                    try {
                        var dur = 0.006, sr = ctx.sampleRate, n = Math.floor(sr * dur);
                        var buf = ctx.createBuffer(1, n, sr);
                        var d = buf.getChannelData(0);
                        for (var k = 0; k < n; k++) {
                            d[k] = (Math.random() * 2 - 1) * Math.pow(1 - k / n, 2);
                        }
                        var s = ctx.createBufferSource(); s.buffer = buf;
                        var g = ctx.createGain();
                        g.gain.value = (window.__audioGain != null) ? window.__audioGain : 0.35;
                        s.connect(g); g.connect(ctx.destination);
                        s.start(when || ctx.currentTime);
                    } catch (e) { /* ignore */ }
                };
                window.__lastSpikeCt = undefined;  // resync on enable
            }
            return '';
        }
        """,
        Output("_dummy4", "children"),
        Input("listen", "value"),
        prevent_initial_call=True,
    )

    # Audio gain slider -> click volume.
    app.clientside_callback(
        "function(g){ window.__audioGain = (g == null) ? 0.35 : g; return ''; }",
        Output("_dummy5", "children"),
        Input("gain", "value"),
        prevent_initial_call=False,
    )

    # "Plot Reset": restore the timeseries figures to their full span (auto-range
    # x; y stays fixed) AND the head plot to its home view. Clearing __xsyncKey
    # lets the next real zoom re-sync the two timeseries figures.
    app.clientside_callback(
        """
        function(n) {
            if (!n) return '';
            function gdOf(id) {
                var el = document.getElementById(id);
                return el && (el.classList.contains('js-plotly-plot')
                    ? el : el.querySelector('.js-plotly-plot'));
            }
            ['ts-top', 'ts-bottom'].forEach(function (id) {
                var gd = gdOf(id);
                if (gd && window.Plotly && gd.layout) {
                    var upd = {};
                    ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'].forEach(function (ax) {
                        if (gd.layout[ax]) upd[ax + '.autorange'] = true;
                    });
                    try { window.Plotly.relayout(gd, upd); } catch (e) {}
                }
            });
            // Head plot: restore the tight home ranges captured on load (its axes
            // aren't auto-ranged, so re-apply the exact original range).
            var hgd = gdOf('head');
            if (hgd && window.Plotly) {
                var home = window.__headHome;
                try {
                    if (home && home.x && home.y) {
                        window.Plotly.relayout(hgd, {
                            'xaxis.range': home.x, 'yaxis.range': home.y,
                        });
                    } else {
                        window.Plotly.relayout(hgd,
                            {'xaxis.autorange': true, 'yaxis.autorange': true});
                    }
                } catch (e) {}
            }
            window.__xsyncKey = null;
            // Re-place the cursor overlay for the restored x-ranges.
            if (window.__placeCursor && window.__cursorMs != null) {
                setTimeout(function () { window.__placeCursor(window.__cursorMs); }, 0);
            }
            return '';
        }
        """,
        Output("_dummy6", "children"),
        Input("plot-reset", "n_clicks"),
        prevent_initial_call=True,
    )


    # Link the x-axis (time) range of the two timeseries figures: zoom/pan in one
    # applies the same range to the other. Runs client-side so it's instant and
    # the per-tick cursor relayout (which also fires relayoutData) is a cheap
    # no-op here.
    app.clientside_callback(
        """
        function(rdTop, rdBot) {
            function extractX(rd) {
                if (!rd) return null;
                for (var k in rd) {
                    if (k.indexOf('xaxis') === 0 &&
                        k.indexOf('autorange') >= 0 && rd[k] === true) return 'auto';
                }
                var lo, hi;
                for (var k in rd) {
                    if (/^xaxis\\d*\\.range\\[0\\]$/.test(k)) lo = rd[k];
                    else if (/^xaxis\\d*\\.range\\[1\\]$/.test(k)) hi = rd[k];
                    else if (/^xaxis\\d*\\.range$/.test(k)) { lo = rd[k][0]; hi = rd[k][1]; }
                }
                if (lo !== undefined && hi !== undefined) return [lo, hi];
                return null;
            }
            function apply(id, rng) {
                var el = document.getElementById(id);
                var gd = el && (el.classList.contains('js-plotly-plot')
                    ? el : el.querySelector('.js-plotly-plot'));
                if (!gd || !window.Plotly || !gd.layout) return;
                var upd = {};
                ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'].forEach(function (ax) {
                    if (gd.layout[ax]) {
                        if (rng === 'auto') { upd[ax + '.autorange'] = true; }
                        else { upd[ax + '.range'] = rng; }
                    }
                });
                try { window.Plotly.relayout(gd, upd); } catch (e) {}
            }
            var ctx = window.dash_clientside.callback_context;
            var trig = ctx && ctx.triggered && ctx.triggered[0];
            if (!trig || !trig.prop_id) return '';
            var src = trig.prop_id.split('.')[0];
            var rng = extractX(src === 'ts-top' ? rdTop : rdBot);
            if (rng === null) return '';
            var key = JSON.stringify(rng);
            if (window.__xsyncKey === key) return '';
            window.__xsyncKey = key;
            apply(src === 'ts-top' ? 'ts-bottom' : 'ts-top', rng);
            // This is a user zoom/pan (follow's own relayouts are caught by the
            // __xsyncKey guard above): pause auto-follow briefly so it doesn't yank
            // the view the user just set.
            var nowP = (window.performance && performance.now)
                ? performance.now() : Date.now();
            window.__suppressFollowUntil = nowP + 2500;
            // Keep the cursor overlay aligned with the new (zoomed/panned) x-range.
            if (window.__placeCursor && window.__cursorMs != null) {
                window.__placeCursor(window.__cursorMs);
            }
            return '';
        }
        """,
        Output("_dummy2", "children"),
        Input("ts-top", "relayoutData"),
        Input("ts-bottom", "relayoutData"),
        prevent_initial_call=True,
    )

    # Click ANYWHERE on either timeseries figure to set the time (not just on a
    # data point): a native listener converts the click's pixel x to a time,
    # maps it to the nearest frame, and sets the slider (which seeks the video
    # and moves the cursor). segmap is mirrored to a window global for the
    # listener; the listener is attached once.
    app.clientside_callback(
        """
        function(segmap) {
            window.__segmap = segmap;
            if (window.__clickAttached) return '';
            window.__clickAttached = true;
            function toMs(v) {
                if (typeof v === 'number') return v;
                var s = String(v);
                if (s.indexOf('Z') < 0 && s.indexOf('+') < 0) s = s.replace(' ', 'T') + 'Z';
                return new Date(s).getTime();
            }
            // Click-to-seek. We can't rely on the native 'click' event: Plotly's
            // drag layer swaps in a full-screen "dragcover" as soon as the pointer
            // moves even slightly, so mouseup lands on a different element and no
            // 'click' fires (still clicks worked, moved ones didn't). Instead treat
            // mousedown+mouseup on a plot with < 6px movement as a click; a real
            // drag (> 6px) is left to Plotly for zoom/pan.
            function seekFromPoint(container, clientX) {
                try {
                    var gd = container.querySelector('.js-plotly-plot');
                    if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis) return;
                    var xa = gd._fullLayout.xaxis;
                    var rect = gd.getBoundingClientRect();
                    var px = clientX - rect.left - xa._offset;
                    if (px < 0 || px > xa._length) return;
                    var r0 = toMs(xa.range[0]), r1 = toMs(xa.range[1]);
                    var ms = r0 + (px / xa._length) * (r1 - r0);
                    var sm = window.__segmap;
                    if (!sm || !sm.length) return;
                    var s = null;
                    for (var k = 0; k < sm.length; k++) {
                        var m = sm[k];
                        var end = m.startMs + (m.n / m.fps) * 1000;
                        if (ms >= m.startMs && ms < end) { s = m; break; }
                    }
                    if (!s) { s = ms < sm[0].startMs ? sm[0] : sm[sm.length - 1]; }
                    var localT = (ms - s.startMs) / 1000;
                    if (localT < 0) localT = 0;
                    var maxT = (s.n - 1) / s.fps;
                    if (localT > maxT) localT = maxT;

                    var v = document.getElementById('video');
                    var wasPlaying = !!(v && !v.paused);
                    var curName = window.__seg && window.__seg.name;

                    var nowP = (window.performance && performance.now)
                        ? performance.now() : Date.now();
                    // Don't let auto-follow yank the view right after a click.
                    window.__suppressFollowUntil = nowP + 2500;
                    // Pin the cursor to the clicked wall-clock time immediately (it's
                    // valid on the full-session x-axis regardless of which hour is
                    // loaded) and HOLD it there until the video -- including a newly
                    // loaded segment -- actually reaches it. This makes the line land
                    // on the click across hours and through buffering, instead of
                    // snapping back to the old/stale position. The hold clears in the
                    // sync loop when the real frame time reaches it (segment-aware).
                    window.__cursorHoldMs = ms;
                    window.__cursorHoldT = nowP;
                    if (window.__placeCursor) { window.__placeCursor(ms); }
                    if (v && s.name === curName) {
                        // Same hour: seek natively RIGHT NOW (no Dash round-trip) and
                        // keep playing if it was. Coalesce rapid clicks: if a seek is
                        // already in flight, remember the latest target and apply it
                        // when the current seek finishes (setting currentTime on every
                        // click thrashes the decoder).
                        if (v.seeking) {
                            window.__pendingSeek = localT;
                        } else {
                            try { v.currentTime = localT; } catch (e) {}
                        }
                        if (wasPlaying) {
                            var pp = v.play();
                            if (pp && pp.catch) { pp.catch(function () {}); }
                        }
                    } else {
                        // Different hour: switch the video via the slider -> _drag
                        // -> server load, and resume playback once it's ready if it
                        // was playing (the 'canplay' handler honours __autoPlayNext).
                        window.__autoPlayNext = wasPlaying;
                        var row = s.base + Math.round(localT * s.fps);
                        var input = document.querySelector('#frame input');
                        if (input) {
                            var setter = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype, 'value').set;
                            setter.call(input, String(row));
                            input.dispatchEvent(new Event('input', {bubbles: true}));
                            input.dispatchEvent(new Event('change', {bubbles: true}));
                        }
                    }
                } catch (e) { /* ignore */ }
            }
            var __mdown = null;
            document.addEventListener('mousedown', function (evt) {
                if (evt.target.closest && evt.target.closest('.modebar')) {
                    __mdown = null; return;
                }
                var c = evt.target.closest &&
                    evt.target.closest('#ts-top, #ts-bottom');
                __mdown = c ? {x: evt.clientX, y: evt.clientY, c: c} : null;
            }, true);
            document.addEventListener('mouseup', function (evt) {
                var md = __mdown; __mdown = null;
                if (!md) { return; }
                if (Math.abs(evt.clientX - md.x) > 6 ||
                    Math.abs(evt.clientY - md.y) > 6) { return; }  // drag -> zoom
                seekFromPoint(md.c, evt.clientX);
            }, true);

            // Load button: show the progress bar on the REAL click, synchronously,
            // before Dash even dispatches -- a Dash clientside can't be relied on to
            // paint before the heavy server _load runs. The confirm callback (on the
            // load-info store) writes "Loaded" and hides the bar when done.
            var loadBtn = document.getElementById('load');
            if (loadBtn && !window.__loadBtnAttached) {
                window.__loadBtnAttached = true;
                loadBtn.addEventListener('click', function () {
                    var s = document.getElementById('load-status');
                    if (s) { s.textContent = '⏳ Loading dataset…'; }
                    var b = document.getElementById('load-bar');
                    if (b) { b.style.display = 'block'; }
                });
            }

            // Auto-advance: when an hour finishes, load the next segment and keep
            // playing. 'ended' jumps the slider to the next hour's first frame
            // (which seeks/loads the video); 'canplay' resumes playback.
            var vid = document.getElementById('video');
            if (vid && !window.__videoAutoAttached) {
                window.__videoAutoAttached = true;
                vid.addEventListener('ended', function () {
                    var sm = window.__segmap;
                    if (!sm || !sm.length) return;
                    var src = vid.currentSrc || '';
                    var idx = -1;
                    for (var k = 0; k < sm.length; k++) {
                        if (src.indexOf(sm[k].name) >= 0) { idx = k; break; }
                    }
                    if (idx < 0 || idx + 1 >= sm.length) return;  // last / unknown
                    window.__autoPlayNext = true;
                    var input = document.querySelector('#frame input');
                    if (input) {
                        var setter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value').set;
                        setter.call(input, String(sm[idx + 1].base));
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                });
                vid.addEventListener('canplay', function () {
                    if (window.__autoPlayNext) {
                        window.__autoPlayNext = false;
                        var p = vid.play();
                        if (p && p.catch) p.catch(function () {});
                    }
                });
                // Apply the latest coalesced click target once the in-flight seek
                // finishes, so a burst of rapid clicks resolves to one final seek.
                vid.addEventListener('seeked', function () {
                    if (window.__pendingSeek != null) {
                        var t = window.__pendingSeek;
                        window.__pendingSeek = null;
                        try { vid.currentTime = t; } catch (e) {}
                    }
                });
            }
            return '';
        }
        """,
        Output("_dummy3", "children"),
        Input("segmap", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("segspikes", "data"),
        Input("seg", "data"),
        Input("unit", "value"),
        Input("offset", "value"),
        prevent_initial_call=True,
    )
    def _segment_spikes(seg, unit, offset):
        # Spike times (seconds from the segment start) for the selected unit
        # within the current video hour — used by the client-side audio monitor.
        if state.df is None or not seg or unit is None:
            return no_update
        base, n = seg["base"], seg["n"]
        t0 = float(state.df[COL_TIME].iloc[base])
        t1 = float(state.df[COL_TIME].iloc[base + n - 1])
        spikes = unit_spike_times_experiment(state.units, unit, float(offset or 0.0))
        in_seg = spikes[(spikes >= t0) & (spikes <= t1)] - t0
        return in_seg.tolist()

    @app.callback(
        Output("ts-bottom", "figure", allow_duplicate=True),
        Output("load-info", "data", allow_duplicate=True),
        Input("unit", "value"),
        prevent_initial_call=True,
    )
    def _select_unit(unit_value):
        # Selecting a unit updates the spike + firing-rate plots immediately, with
        # the progress bar shown while it loads (the top/head plots stay as-is).
        if state.df is None or unit_value is None:
            return no_update, no_update
        state.load_counter += 1
        if unit_value == state.autoselect_unit:
            # Change came from loading a dataset (auto-picked first unit); _load
            # already built the bottom plot -- keep the dataset summary.
            state.autoselect_unit = None  # consume (real unit ids are never None)
            return no_update, {"msg": state.load_status_msg, "n": state.load_counter}
        state.unit_id = unit_value
        fig = _bottom_figure()
        return fig, {"msg": f"✓ Loaded unit {unit_value}", "n": state.load_counter}

    @app.callback(
        Output("ts-bottom", "figure", allow_duplicate=True),
        Input("offset", "value"),
        prevent_initial_call=True,
    )
    def _set_offset(offset):
        if state.df is None or offset is None:
            return no_update
        state.spike_offset_s = float(offset)
        return _bottom_figure()

    return app


def _lan_ip() -> str:
    """Best-effort primary LAN IP of this machine."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _reset_ngrok() -> None:
    """Terminate any ngrok agent left running from a previous session.

    A crashed or re-launched run can leave an ngrok agent holding the tunnel,
    causing ``ERR_NGROK_334`` ("endpoint is already online") on the next start.
    """
    import platform
    import subprocess
    import time

    try:
        from pyngrok import ngrok

        ngrok.kill()  # stop the agent this process may have started
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ngrok.exe"],
                           capture_output=True, check=False)
        else:
            subprocess.run(["pkill", "-x", "ngrok"], capture_output=True, check=False)
    except Exception:
        pass
    time.sleep(1)  # let the tunnel/ports free up


def _start_cloudflare_tunnel(port: int) -> str:
    """Open a Cloudflare *quick* tunnel and return the public URL.

    Needs no account and shows no interstitial, but the URL is random and
    Cloudflare tears it down after a while — use a *named* tunnel
    (:func:`_start_cloudflare_named_tunnel`) for a link that lasts weeks.
    Downloads the ``cloudflared`` helper on first use.
    """
    from pycloudflared import try_cloudflare

    return try_cloudflare(port=port).tunnel


# Keep tunnel subprocesses alive for the lifetime of the server process.
_TUNNEL_PROCS: list = []


def _cloudflared_bin() -> str:
    """Locate a ``cloudflared`` executable.

    Prefer one on ``PATH`` (a proper ``winget`` install), else fall back to the
    copy ``pycloudflared`` downloads for the quick tunnel — so the named tunnel
    needs no separate install. Returns ``""`` if none is found.
    """
    import os
    import shutil

    exe = shutil.which("cloudflared")
    if exe:
        return exe
    try:
        from pycloudflared.util import get_info

        cand = getattr(get_info(), "executable", None)
        if cand and os.path.exists(cand):
            return cand
    except Exception:  # noqa: BLE001 - fallback is best-effort
        pass
    return ""


def _tunnel_exists(exe: str, tunnel: str) -> bool:
    """Return True if a named tunnel already exists (requires prior login)."""
    import json
    import subprocess

    try:
        r = subprocess.run(
            [exe, "tunnel", "list", "--output", "json"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            return False
        return any(t.get("name") == tunnel for t in json.loads(r.stdout or "[]"))
    except Exception:  # noqa: BLE001 - treat any parse/exec failure as "missing"
        return False


def _ensure_named_tunnel(exe: str, tunnel: str, hostname: str | None) -> None:
    """Provision the named tunnel on first run so the GUI "just works".

    Idempotent: logs in (once, via browser), creates the tunnel, and routes the
    hostname only if those steps haven't already been done. The single browser
    authorization on first ``login`` is unavoidable (Cloudflare OAuth); every run
    afterwards is fully automatic.
    """
    import os
    import subprocess
    from pathlib import Path

    cert = Path(os.path.expanduser("~")) / ".cloudflared" / "cert.pem"
    if not cert.exists():
        print("  [share] first-time Cloudflare login — a browser window will open;"
              f" authorize the {hostname or 'chosen'} zone to continue...")
        subprocess.run([exe, "tunnel", "login"], check=True)

    if not _tunnel_exists(exe, tunnel):
        print(f"  [share] creating named tunnel '{tunnel}' (one-time)...")
        subprocess.run([exe, "tunnel", "create", tunnel], check=True)

    if hostname:
        r = subprocess.run(
            [exe, "tunnel", "route", "dns", tunnel, hostname],
            capture_output=True, text=True, check=False,
        )
        blob = (r.stdout + r.stderr).lower()
        if r.returncode != 0 and "already" not in blob and "exists" not in blob:
            print(f"  [share] DNS route warning: {(r.stderr or r.stdout).strip()}")


def _start_cloudflare_named_tunnel(
    port: int, tunnel: str | None, hostname: str | None = None
) -> str:
    """Run a *named* Cloudflare tunnel and return its stable public URL.

    A named tunnel keeps the **same** hostname across restarts and stays up as
    long as the process runs (or as a service — see the README), so the link is
    good for weeks rather than the minutes/hours of a quick tunnel. It requires a
    one-time setup with a domain you control on a (free) Cloudflare account::

        cloudflared tunnel login
        cloudflared tunnel create pirouette
        cloudflared tunnel route dns pirouette pirouette.<your-domain>

    after which this launches ``cloudflared tunnel run`` pointed at the local app.
    """
    import subprocess

    if not tunnel:
        raise RuntimeError(
            "cloudflare-named needs a tunnel name (set CLOUDFLARE_TUNNEL or "
            "--cloudflare-tunnel). Create one with `cloudflared tunnel create <name>`."
        )
    exe = _cloudflared_bin()
    if not exe:
        raise RuntimeError(
            "cloudflared executable not found (neither on PATH nor bundled with "
            "pycloudflared). Run `uv sync --extra gui`, or install it with "
            "`winget install --id Cloudflare.cloudflared`, then complete the "
            "one-time tunnel setup (login / create / route dns)."
        )
    _ensure_named_tunnel(exe, tunnel, hostname)  # provision on first run
    cmd = [exe, "tunnel", "--url", f"http://localhost:{port}", "run", tunnel]
    proc = subprocess.Popen(cmd)  # noqa: S603 - args are ours, not user input
    _TUNNEL_PROCS.append(proc)
    return f"https://{hostname}" if hostname else f"(named tunnel '{tunnel}' is running)"


def _start_ngrok_tunnel(port: int) -> str:
    """Open an ngrok tunnel and return the public URL (requires an auth token)."""
    import os

    from pyngrok import ngrok

    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
    _reset_ngrok()  # clear any tunnel left over from a previous run
    return ngrok.connect(port, "http").public_url


def _precompute_firing_rates(units_dir, bin_s: float, smooth_s: float) -> None:
    """Ensure every units file in *units_dir* has a firing-rate cache on disk.

    Runs once before serving; a matching cache is reused (fast), otherwise it is
    computed and saved. Progress is printed so the operator sees the one-time cost.
    """
    files = _list_files(units_dir, (".pkl",))
    if not files:
        return
    for uf in files:
        name = Path(uf).name
        parq, js = ephys.cache_paths(uf)
        if parq.exists() and js.exists():
            print(f"  [firing-rate] {name}: cached")
            continue
        try:
            units = load_units(uf)
        except Exception as exc:  # noqa: BLE001 - skip unreadable units files
            print(f"  [firing-rate] {name}: skipped ({exc})")
            continue
        n = len(units)
        print(f"  [firing-rate] {name}: computing for {n} units (one-time)...")

        def _progress(done, total, _name=name):
            if done == total or done % 25 == 0:
                print(f"      {_name}: {done}/{total}")

        ephys.ensure_firing_rates(units, uf, bin_s, smooth_s, FR_CACHE_POINTS,
                                  progress=_progress)
        print(f"  [firing-rate] {name}: done")


def run(
    dataset_dir: str | Path,
    units_dir: str | Path,
    video_dir: str | Path,
    spike_offset_s: float = 0.0,
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False,
    share: bool = False,
    share_method: str = "cloudflare",
    cloudflare_tunnel: str | None = None,
    cloudflare_hostname: str | None = None,
    show_all_spikes: bool = False,
    firing_rate_bin_s: float = 0.05,
    firing_rate_smooth_s: float = 0.2,
    heading_mode: str = "vector",
) -> None:
    """Create and serve the app, printing the URLs to share.

    Parameters
    ----------
    host:
        Bind address. ``"0.0.0.0"`` (default) exposes the app on the local
        network so viewers on the same Wi-Fi/LAN can open the printed LAN URL.
        ``"127.0.0.1"`` restricts it to this machine only.
    share:
        When ``True``, open a public tunnel and print an ``https`` URL that works
        for viewers **anywhere** (data stays on this machine).
    share_method:
        ``"cloudflare"`` (default) uses a Cloudflare quick tunnel — no account,
        no interstitial, but a random URL that lasts only minutes/hours.
        ``"cloudflare-named"`` runs a *named* tunnel (needs ``cloudflare_tunnel``
        + a one-time domain setup) for a **stable URL that lasts weeks**.
        ``"ngrok"`` uses ngrok (needs ``NGROK_AUTHTOKEN``; free tier shows a
        warning page).
    cloudflare_tunnel, cloudflare_hostname:
        Named-tunnel name and its routed hostname (e.g. ``pirouette`` and
        ``pirouette.example.org``). Used only when ``share_method`` is
        ``"cloudflare-named"``.
    """
    app = create_app(
        dataset_dir, units_dir, video_dir, spike_offset_s=spike_offset_s,
        show_all_spikes=show_all_spikes,
        firing_rate_bin_s=firing_rate_bin_s,
        firing_rate_smooth_s=firing_rate_smooth_s,
        heading_mode=heading_mode,
    )

    # Precompute the instantaneous firing rate for every spiking (units) file so
    # switching units in the GUI is instant. Cached to a parquet next to each units
    # file; only recomputed when the file or bin/smoothing params change.
    _precompute_firing_rates(units_dir, firing_rate_bin_s, firing_rate_smooth_s)
    if show_all_spikes:
        print("  [note] SHOW_ALL_SPIKES=true renders EVERY spike tick -- busy units "
              "ship tens of MB per unit switch and are slow (especially over the "
              "public link). Set SHOW_ALL_SPIKES=false for fast switching "
              f"({MAX_RASTER_SPIKES:,}-tick subsample, visually the same).")

    print("\nPirouette explorer — share one of these links:")
    print(f"  this machine : http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"  same network : http://{_lan_ip()}:{port}")
    if share:
        method = (share_method or "cloudflare").lower()
        try:
            if method == "ngrok":
                public = _start_ngrok_tunnel(port)
            elif method in ("cloudflare-named", "named"):
                public = _start_cloudflare_named_tunnel(
                    port, cloudflare_tunnel, cloudflare_hostname
                )
            else:
                public = _start_cloudflare_tunnel(port)
            print(f"  public       : {public}   <-- send this to anyone")
            if method in ("cloudflare-named", "named"):
                print("                 (stable URL — keep this process running; "
                      "see README to run it as a service for weeks)")
        except Exception as exc:  # noqa: BLE001 - report and continue serving
            print(f"  [share] {method} tunnel failed: {exc}")
            print("  [share] run `uv sync --extra gui` to install the tunnel helper.")
    print(
        "\nNote: for the LAN link, allow Python through the Windows Firewall when "
        "prompted.\n"
    )
    # threaded=True so video range requests are served concurrently with the app's
    # other requests/callbacks -- a single-threaded server blocks during a video
    # chunk, which shows up as playback buffering/stalls.
    app.run(host=host, port=port, debug=debug, threaded=True)
