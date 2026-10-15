"""Plot accumulated commutator turns for every Pirouette experiment.

One line per experiment (session), coloured by subject ID. For each session the
hourly ``Commutator_AccumulatedCommutatorTurns_*.csv`` files in
``s3://aind-open-data/<session>/behavior/Commutator/`` are downloaded
(anonymously), concatenated in time order and plotted against hours since the
start of the recording.

Crash handling
--------------
On some early sessions the commutator state was not recovered after a crash, so
the accumulated value either reset to ~0 or restarted at a wrong value. Any
sample-to-sample change larger than ``--jump-threshold`` turns (also checked
across file boundaries) is treated as such an artifact: the value is held at
the previous one and accumulation continues from there.

Downloaded + processed data are cached as small CSVs in ``--cache-dir`` so
re-plotting is fast.

Usage
-----
    python scripts/population_commutator_turns.py
    python scripts/population_commutator_turns.py --save-dir figures --no-show
"""

from __future__ import annotations

import argparse
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pirouette_data.ingestion import get_s3_client

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKET = "aind-open-data"
FILE_TAG = "Commutator_AccumulatedCommutatorTurns"

SESSIONS = [
    "854393_2026-08-28_20-34-08",
    "872757_2026-08-24_20-30-12",
    "842456_2026-08-11_22-26-31",
    "841322_2026-08-05_21-39-18",
    "854393_2026-06-09_19-34-26",
    "801055_2026-05-18_23-41-18",
    "842456_2026-03-20_20-49-05",
    "841322_2026-03-19_18-56-42",
    "835602_2026-01-06_22-17-15",
    "801055_2026-01-06_22-49-36",
]


def list_turn_keys(client, session: str) -> list[str]:
    """Return the accumulated-turns CSV keys of a session, in time order."""
    prefix = f"{session}/behavior/Commutator/{FILE_TAG}"
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".csv")]
    # Filenames embed an ISO timestamp, so lexicographic order is chronological.
    return sorted(keys)


def read_key(client, key: str) -> pd.DataFrame:
    body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), usecols=["Seconds", "Value"])


def correct_jumps(values: np.ndarray, jump_threshold: float) -> np.ndarray:
    """Accumulate *values*, ignoring changes larger than *jump_threshold*.

    Jumps (resets / wrong restored state after a crash) are replaced by a zero
    step, i.e. the jumped value is set to the previous one and accumulation
    continues from there. The result starts at 0.
    """
    steps = np.diff(values, prepend=values[0])
    steps[np.abs(steps) > jump_threshold] = 0.0
    return np.cumsum(steps)


def load_session(
    session: str, cache_dir: Path, jump_threshold: float, bin_s: float, workers: int
) -> pd.DataFrame:
    """Return a DataFrame (``hours``, ``turns``) for one session, using the cache."""
    cache = cache_dir / f"{session}_commutator_turns.csv"
    if cache.exists():
        return pd.read_csv(cache)

    client = get_s3_client(anonymous=True)
    keys = list_turn_keys(client, session)
    if not keys:
        raise FileNotFoundError(f"No commutator files found for {session}")

    # Download in parallel; ex.map preserves chronological order.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = list(
            tqdm(ex.map(lambda k: read_key(client, k), keys), total=len(keys),
                 desc=session)
        )
    df = pd.concat(frames, ignore_index=True)
    # Some files contain stray non-numeric rows; coerce and drop them.
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    turns = correct_jumps(df["Value"].to_numpy(), jump_threshold)
    hours = (df["Seconds"].to_numpy() - df["Seconds"].iloc[0]) / 3600.0

    # Downsample (keep the first sample of each time bin) to keep plots light.
    bins = np.floor(hours * 3600.0 / bin_s).astype(int)
    keep = np.r_[True, np.diff(bins) != 0]
    out = pd.DataFrame({"hours": hours[keep], "turns": turns[keep]})
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


# Figure style (matches twists_polished.pdf): Arial, thick offset spines.
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 18
SPINE_LW = 2.5
SPINE_OFFSET = 10  # points


def style_axes(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Offset, thick, minimal axes with ticks only at the ends and middle."""
    from matplotlib.ticker import MaxNLocator

    loc = MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
    xt = loc.tick_values(*xlim)
    yt = [np.floor(ylim[0] / 1000) * 1000, np.ceil(ylim[1] / 1000) * 1000]
    ax.set_xlim(xt[0], xt[-1])
    ax.set_ylim(yt[0], yt[-1])
    ax.set_xticks([xt[0], (xt[0] + xt[-1]) / 2, xt[-1]])
    ax.set_yticks([yt[0], 0, yt[-1]] if yt[0] < 0 < yt[-1] else [yt[0], yt[-1]])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_LW)
        ax.spines[side].set_position(("outward", SPINE_OFFSET))
    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.spines["bottom"].set_bounds(xt[0], xt[-1])
    ax.tick_params(width=SPINE_LW, length=7, labelsize=TICK_FS)


def plot_population(data: dict[str, pd.DataFrame], save_dir: Path | None, show: bool):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,  # editable text in the PDF
        "svg.fonttype": "none",
    })
    subjects = sorted({s.split("_")[0] for s in data})
    cmap = plt.get_cmap("tab10")
    colors = {subj: cmap(i % 10) for i, subj in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(9, 7))
    labelled: set[str] = set()
    for session, df in data.items():
        subj = session.split("_")[0]
        ax.plot(
            df["hours"], df["turns"], color=colors[subj], linewidth=2,
            label=subj if subj not in labelled else None,
        )
        labelled.add(subj)

    ax.set_xlabel("Time (hours)", fontsize=LABEL_FS)
    ax.set_ylabel("Accumulated turns (#)", fontsize=LABEL_FS)
    all_turns = np.concatenate([d["turns"].to_numpy() for d in data.values()])
    xmax = max(d["hours"].iloc[-1] for d in data.values())
    style_axes(ax, (0, xmax), (min(all_turns.min(), 0), max(all_turns.max(), 0)))
    ax.legend(title="Subject ID", loc="lower right", frameon=False,
              fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, handlelength=1.2)
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            out = (save_dir / f"population_commutator_turns.{ext}").resolve()
            fig.savefig(out, dpi=200)
            print(f"Saved {out}")
    if show:
        plt.show()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sessions", nargs="+", default=SESSIONS,
                   help="Session folder names in s3://aind-open-data.")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "cache" / "commutator")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--jump-threshold", type=float, default=1.0,
                   help="Max plausible change in turns between consecutive samples.")
    p.add_argument("--bin-seconds", type=float, default=10.0,
                   help="Downsampling bin width for plotting.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    for s in args.sessions:
        if not re.fullmatch(r"\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", s):
            raise ValueError(f"Unexpected session name: {s!r}")

    data = {
        s: load_session(s, args.cache_dir, args.jump_threshold,
                        args.bin_seconds, args.workers)
        for s in args.sessions
    }
    for s, df in data.items():
        print(f"{s}: {df['hours'].iloc[-1]:.1f} h, {df['turns'].iloc[-1]:.0f} turns")
    plot_population(data, args.save_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
