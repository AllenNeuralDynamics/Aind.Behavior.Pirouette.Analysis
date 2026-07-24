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


# ---------------------------------------------------------------------------
# Data layer (no Dash dependency — unit tested)
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a per-frame dataset from ``.parquet``, ``.pkl``, or ``.csv``.

    Parameters
    ----------
    path:
        Dataset file path.

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
        df = pd.read_parquet(path)
    elif suffix == ".pkl":
        df = pd.read_pickle(path)
    elif suffix == ".csv":
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
        Experiment-referenced spike times (seconds).
    t0_s, t1_s:
        Time window (experiment seconds) over which to compute the rate.
    bin_s:
        Requested histogram bin width (seconds).
    smooth_sigma_s:
        Gaussian smoothing sigma (seconds).
    max_bins:
        Optional cap on the number of bins. Over long windows the ``bin_s``
        resolution can imply millions of bins that are far finer than the plot
        can show; capping keeps the computation fast (the effective bin width
        widens to ``(t1 - t0) / max_bins``).

    Returns
    -------
    centers_s : numpy.ndarray
        Bin-centre times (experiment seconds).
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

    df: pd.DataFrame | None = None
    units: dict | None = None
    exp_start_dt: pd.Timestamp | None = None
    reader: FrameReader | None = None
    head_x: np.ndarray | None = None
    head_y: np.ndarray | None = None
    head_t: np.ndarray | None = None  # experiment seconds
    chamber: dict | None = None
    unit_id: object = None
    bottom_cache: dict = field(default_factory=dict)

    def load(self, dataset_path: str | Path, units_path: str | Path, offset_s: float):
        """Load a dataset + units file into the state."""
        self.df = load_dataset(dataset_path)
        self.units = load_units(units_path)
        self.spike_offset_s = float(offset_s)
        self.exp_start_dt = experiment_start_datetime(self.df)
        self.reader = FrameReader(self.video_dir)
        self.head_x, self.head_y = head_position_mm(self.df)
        self.head_t = self.df[COL_TIME].to_numpy(dtype="float64")
        self.chamber = chamber_corners_mm(self.df)
        self.unit_id = unit_ids(self.units)[0]
        self.bottom_cache = {}
        return self


def _list_files(directory: Path, suffixes: tuple[str, ...]) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        str(p) for p in directory.iterdir() if p.suffix.lower() in suffixes
    )


# ---------------------------------------------------------------------------
# Figure builders (Plotly)
# ---------------------------------------------------------------------------
def build_timeseries_top(df: pd.DataFrame):
    """Behaviour + velocity + heading, shared time x-axis, with a red cursor."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[0.08, 0.46, 0.46],
        subplot_titles=(
            "behaviour  (rest: gray, movement: salmon)",
            "smoothed velocity (mm/s)",
            "heading (deg)  -  commutator: dashed, ear-vector: solid",
        ),
    )

    # Behaviour as a thin 2-colour heatmap row.
    s = _stride(len(df))
    x = df[COL_DATETIME].iloc[::s]
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
    fig.add_trace(
        go.Scattergl(
            x=x, y=df[COL_COMM_HEADING].iloc[::s],
            line=dict(color="black", width=1, dash="dash"), name="commutator",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=x, y=df[COL_EAR_HEADING].iloc[::s],
            line=dict(color="black", width=1), name="ear vector",
        ),
        row=3, col=1,
    )

    x0 = df[COL_DATETIME].iloc[0]
    fig.add_shape(
        type="line", x0=x0, x1=x0, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=CURSOR_COLOR, width=2.5),
    )
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="mm/s", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=3, col=1)
    fig.update_annotations(font_size=12)  # subplot titles hold the labels
    fig.update_layout(
        height=460, margin=dict(l=55, r=25, t=30, b=20),
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
    fig.add_shape(
        type="line", x0=x0, x1=x0, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=CURSOR_COLOR, width=2.5),
    )
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    fig.update_layout(
        height=320, margin=dict(l=55, r=25, t=30, b=20), showlegend=False,
        template="plotly_white", uirevision=f"ts-bottom-{unit_label}",
    )
    return fig


def build_head_position(
    head_x: np.ndarray,
    head_y: np.ndarray,
    head_t: np.ndarray,
    current_row: int,
    window_s: float,
    fps: float = 60.0,
    chamber: dict | None = None,
):
    """Spatial head-position trail over a time window, inferno-coloured by time.

    Points are drawn as time-coloured markers (no connecting line); if *chamber*
    corner positions are given, a black box marks the chamber walls and the axes
    are bounded to it.
    """
    import plotly.graph_objects as go

    half = int(round(window_s * fps))
    lo = max(0, current_row - half)
    hi = min(len(head_x), current_row + 1)
    xs, ys, ts = head_x[lo:hi], head_y[lo:hi], head_t[lo:hi]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=5, color=ts, colorscale="Inferno",
                colorbar=dict(title="t (s)", thickness=12), showscale=True,
            ),
            name="trail", hoverinfo="skip",
        )
    )
    if hi > lo:
        fig.add_trace(
            go.Scattergl(
                x=[head_x[current_row]], y=[head_y[current_row]], mode="markers",
                marker=dict(size=12, color=CURSOR_COLOR, line=dict(color="white", width=1)),
                name="current", hoverinfo="skip",
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
        height=480, margin=dict(l=55, r=20, t=30, b=20), showlegend=False,
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
    from dash import Dash, Input, Output, Patch, State, dcc, html, no_update
    from flask import abort, send_file

    state = AppState(
        dataset_dir=Path(dataset_dir),
        units_dir=Path(units_dir),
        video_dir=Path(video_dir),
        spike_offset_s=spike_offset_s,
        show_all_spikes=show_all_spikes,
        firing_rate_bin_s=firing_rate_bin_s,
        firing_rate_smooth_s=firing_rate_smooth_s,
    )

    datasets = _list_files(state.dataset_dir, (".parquet", ".pkl", ".csv"))
    unit_files = _list_files(state.units_dir, (".pkl",))

    app = Dash(__name__, title="Pirouette explorer")

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
                dcc.Dropdown(id="dataset", options=datasets,
                             value=datasets[0] if datasets else None, clearable=False),
            ], style={"flex": "3"}),
            html.Div([
                html.Label("Spike units"),
                dcc.Dropdown(id="unitsfile", options=unit_files,
                             value=unit_files[0] if unit_files else None, clearable=False),
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
                html.Label(" "),
                html.Button("Load", id="load", n_clicks=0, style={"width": "100%"}),
            ], style={"flex": "1"}),
        ],
        style={"display": "flex", "gap": "10px", "alignItems": "flex-end",
               "padding": "8px"},
    )

    graph_config = {"scrollZoom": True, "displaylogo": False}

    # Equal-width columns: the video and the plots share the same horizontal
    # extent. The video sizes naturally to that width (no letterbox); the plots
    # are stacked taller for a bigger view.
    left = html.Div(
        [
            # Native HTML5 player: smooth, browser-buffered playback + scrubbing.
            html.Video(
                id="video", controls=True, autoPlay=False,
                style={"width": "100%", "border": "1px solid #ccc", "background": "#000"},
            ),
            html.Div(id="frame-info", style={"fontFamily": "monospace", "padding": "6px 0"}),
            # Global scrubber across the whole session; jumps to the right video.
            dcc.Slider(id="frame", min=0, max=1, step=1, value=0, marks=None,
                       tooltip={"placement": "bottom"}, updatemode="mouseup"),
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
            # Poll the video's playback time to drive the plot cursor.
            dcc.Interval(id="sync", interval=150, n_intervals=0),
            dcc.Store(id="seg"),
            dcc.Store(id="vrow"),
            dcc.Store(id="seek"),
            html.Div(id="_dummy", style={"display": "none"}),
        ],
        style={"flex": "1", "minWidth": "560px", "padding": "8px"},
    )

    right = html.Div(
        [
            dcc.Graph(id="ts-top", config=graph_config),
            dcc.Graph(id="head", config=graph_config),
            dcc.Graph(id="ts-bottom", config=graph_config),
        ],
        style={"flex": "1", "padding": "8px"},
    )

    app.layout = html.Div([
        html.H3("Pirouette dataset explorer", style={"padding": "0 8px"}),
        controls,
        html.Div([left, right], style={"display": "flex"}),
    ])

    # ---- helpers bound to state ----
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
            spike_dt = spikes_to_datetime(in_range, state.exp_start_dt)
            # Firing rate at full (fine) resolution so the trace stays smooth;
            # downsample the plotted points BEFORE the datetime conversion (that
            # conversion of the full fine grid was the main cost, not the bins).
            centers, rate = instantaneous_firing_rate(
                spikes, t0, t1,
                bin_s=state.firing_rate_bin_s,
                smooth_sigma_s=state.firing_rate_smooth_s,
            )
            rs = _stride(len(centers))
            rate_dt = spikes_to_datetime(centers[::rs], state.exp_start_dt)
            rate_ds = rate[::rs]
            cached = (spike_dt, rate_dt, rate_ds)
            state.bottom_cache[key] = cached
        spike_dt, rate_dt, rate_ds = cached
        x0 = df[COL_DATETIME].iloc[0]
        x_range = (df[COL_DATETIME].iloc[0], df[COL_DATETIME].iloc[-1])
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
        Input("load", "n_clicks"),
        State("dataset", "value"),
        State("unitsfile", "value"),
        State("offset", "value"),
        prevent_initial_call=False,
    )
    def _load(_clicks, dataset_path, units_path, offset):
        if not dataset_path or not units_path:
            return (no_update,) * 9
        state.load(dataset_path, units_path, offset or 0.0)
        options = [{"label": f"unit {u}", "value": u} for u in unit_ids(state.units)]
        segs = segments(state.df)
        # Slider marks at each segment start, labelled with the Pacific hour.
        marks = {}
        for s in segs:
            base, _, _ = segment_info(state.df, s)
            marks[int(base)] = pd.Timestamp(
                state.df[COL_DATETIME].iloc[base]
            ).strftime("%H:%M")
        return (
            build_timeseries_top(state.df),
            _bottom_figure(),
            options,
            state.unit_id,
            [{"label": s, "value": s} for s in segs],
            segs[0],
            len(state.df) - 1,
            0,
            marks,
        )

    @app.callback(
        Output("video", "src"),
        Output("seg", "data"),
        Input("segment", "value"),
        prevent_initial_call=True,
    )
    def _load_segment(seg):
        # Point the <video> at the chosen hour and store its row/fps mapping.
        if state.df is None or not seg:
            return no_update, no_update
        base, n, fps = segment_info(state.df, seg)
        return f"/pirouette-video/{seg}.mp4", {"base": base, "n": n, "fps": fps, "name": seg}

    @app.callback(
        Output("segment", "value", allow_duplicate=True),
        Output("seek", "data"),
        Input("frame", "value"),
        State("seg", "data"),
        prevent_initial_call=True,
    )
    def _seek(row, seg):
        # Slider -> jump to the video/hour containing this row and seek to it.
        if state.df is None or row is None:
            return no_update, no_update
        row = int(min(max(0, row), len(state.df) - 1))
        name = state.df[COL_SOURCE].iloc[row]
        base, _, fps = segment_info(state.df, name)
        seek = {"seg": name, "t": (row - base) / fps}
        current = seg.get("name") if seg else None
        seg_out = no_update if current == name else name
        return seg_out, seek

    # Each tick: apply any pending seek (once the right video is ready) and read
    # the video's currentTime -> global row (client-side, cheap).
    app.clientside_callback(
        """
        function(_n, seg, seek) {
            var nou = window.dash_clientside.no_update;
            var v = document.getElementById('video');
            if (!v || !seg) { return [nou, nou]; }
            var clearSeek = nou;
            if (seek && seek.seg && v.readyState >= 1 &&
                v.currentSrc && v.currentSrc.indexOf(seek.seg) >= 0) {
                v.currentTime = seek.t;
                clearSeek = null;
            }
            var row = seg.base + Math.round((v.currentTime || 0) * seg.fps);
            var maxr = seg.base + seg.n - 1;
            if (row > maxr) { row = maxr; }
            var out = row;
            if (window.__pirRow === row) { out = nou; }
            else { window.__pirRow = row; }
            return [out, clearSeek];
        }
        """,
        Output("vrow", "data"),
        Output("seek", "data", allow_duplicate=True),
        Input("sync", "n_intervals"),
        State("seg", "data"),
        State("seek", "data"),
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

    @app.callback(
        Output("ts-bottom", "figure", allow_duplicate=True),
        Input("unit", "value"),
        prevent_initial_call=True,
    )
    def _select_unit(unit_value):
        # Selecting a unit updates the spike + firing-rate plots immediately.
        if state.df is None or unit_value is None:
            return no_update
        state.unit_id = unit_value
        return _bottom_figure()

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

    @app.callback(
        Output("frame-info", "children"),
        Output("head", "figure"),
        Output("ts-top", "figure", allow_duplicate=True),
        Output("ts-bottom", "figure", allow_duplicate=True),
        Input("vrow", "data"),
        Input("window", "value"),
        prevent_initial_call=True,
    )
    def _sync(row, window_s):
        # Driven by the video playhead (vrow): move the cursor + update info/head.
        if state.df is None or row is None:
            return no_update, no_update, no_update, no_update
        row = int(min(max(0, row), len(state.df) - 1))
        src = state.df[COL_SOURCE].iloc[row]
        fidx = frame_index_for_row(state.df, row)
        t_s = float(state.df[COL_TIME].iloc[row])
        dt = state.df[COL_DATETIME].iloc[row]
        info = html.Div([
            html.Span(f"frame {row:,} / {len(state.df) - 1:,}  ({src} #{fidx})"),
            html.Br(),
            html.Span(f"t since start: {t_s:,.3f} s  ({t_s / 3600:.3f} h)"),
            html.Br(),
            html.Span(f"PST: {pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"),
        ])
        head_fig = build_head_position(
            state.head_x, state.head_y, state.head_t, row, float(window_s),
            chamber=state.chamber,
        )
        cursor = pd.Timestamp(dt).isoformat()
        patch_top, patch_bottom = Patch(), Patch()
        patch_top["layout"]["shapes"][0]["x0"] = cursor
        patch_top["layout"]["shapes"][0]["x1"] = cursor
        patch_bottom["layout"]["shapes"][0]["x0"] = cursor
        patch_bottom["layout"]["shapes"][0]["x1"] = cursor
        return info, head_fig, patch_top, patch_bottom

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
    """Open a Cloudflare quick tunnel and return the public URL.

    Needs no account and shows no interstitial. Downloads the ``cloudflared``
    helper on first use.
    """
    from pycloudflared import try_cloudflare

    return try_cloudflare(port=port).tunnel


def _start_ngrok_tunnel(port: int) -> str:
    """Open an ngrok tunnel and return the public URL (requires an auth token)."""
    import os

    from pyngrok import ngrok

    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
    _reset_ngrok()  # clear any tunnel left over from a previous run
    return ngrok.connect(port, "http").public_url


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
    show_all_spikes: bool = False,
    firing_rate_bin_s: float = 0.05,
    firing_rate_smooth_s: float = 0.2,
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
        no browser interstitial. ``"ngrok"`` uses ngrok (needs ``NGROK_AUTHTOKEN``
        and shows a warning page on the free tier).
    """
    app = create_app(
        dataset_dir, units_dir, video_dir, spike_offset_s=spike_offset_s,
        show_all_spikes=show_all_spikes,
        firing_rate_bin_s=firing_rate_bin_s,
        firing_rate_smooth_s=firing_rate_smooth_s,
    )

    print("\nPirouette explorer — share one of these links:")
    print(f"  this machine : http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"  same network : http://{_lan_ip()}:{port}")
    if share:
        method = (share_method or "cloudflare").lower()
        try:
            if method == "ngrok":
                public = _start_ngrok_tunnel(port)
            else:
                public = _start_cloudflare_tunnel(port)
            print(f"  public       : {public}   <-- send this to anyone")
        except Exception as exc:  # noqa: BLE001 - report and continue serving
            print(f"  [share] {method} tunnel failed: {exc}")
            print("  [share] run `uv sync --extra gui` to install the tunnel helper.")
    print(
        "\nNote: for the LAN link, allow Python through the Windows Firewall when "
        "prompted.\n"
    )
    app.run(host=host, port=port, debug=debug)
