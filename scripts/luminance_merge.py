"""Merge step: combine per-chunk results into the final normalized luminance trace.

Reads every ``*_luminance_chunk.npz`` found (recursively) under ``--chunks-dir``
(as written by ``scripts/luminance_chunks.py``), orders them by Harp time, and
applies the whole-session steps that cannot be done per chunk: the two-sample
average (so chunk boundaries are seamless) and the min-max scaling (see --scaling).

Outputs, written to ``--save-dir`` as ``<session>_<camera>_luminance.*``:

* ``.npy`` — float64 ``(N, 2)``: column 0 = Harp timestamp (s), column 1 =
  normalized luminance in [0, 1].
* ``.png`` — normalized luminance (black trace) vs. Pacific wall-clock time.
* ``.parquet`` (or ``.csv``) — full table with all columns.

If no chunk files are found, a ``NO_CHUNKS_SKIPPED.txt`` marker is written (so a
pipeline step always produces a file) and the script exits.

Parameters come from CLI flags and/or ``.env`` (CLI overrides ``.env``):
``CHUNKS_DIR``, ``SAVE_DIR``, ``OUTPUT_FORMAT``.

Usage
-----
    python scripts/luminance_merge.py --chunks-dir C:/results/chunks \\
        --save-dir C:/results/luminance
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pirouette_data import luminance


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--chunks-dir", default=os.getenv("CHUNKS_DIR"),
                   help="Directory (searched recursively) holding the "
                        "*_luminance_chunk.npz files.")
    p.add_argument("--save-dir", default=os.getenv("SAVE_DIR"),
                   help="Directory for the .npy / .png / table outputs "
                        "(created if missing).")
    p.add_argument("--scaling", choices=luminance.SCALINGS,
                   default=os.getenv("SCALING", luminance.DEFAULT_SCALING),
                   help="minmax: min->0, max->1, no clipping (default); absolute: "
                        "mean pixel/255; percentile: clipped to the percentiles.")
    p.add_argument("--low-percentile", type=float,
                   default=luminance.DEFAULT_LOW_PERCENTILE,
                   help="Percentile mapped to 0 (relative scaling).")
    p.add_argument("--high-percentile", type=float,
                   default=luminance.DEFAULT_HIGH_PERCENTILE,
                   help="Percentile mapped to 1 (relative scaling).")
    p.add_argument("--format", dest="output_format", choices=("parquet", "csv"),
                   default=os.getenv("OUTPUT_FORMAT", "parquet"))
    p.add_argument("--shade-gaps", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Shade recording gaps salmon in the plot and break the line "
                        "there (default on).")
    p.add_argument("--gap-seconds", type=float, default=None,
                   help="Spacing (s) above which samples count as a gap "
                        "(default: 10x the median spacing).")
    p.add_argument("--show", action="store_true",
                   help="Also display the plot in a window (it is always saved).")
    args = p.parse_args()

    missing = [n for n in ("chunks_dir", "save_dir") if not getattr(args, n)]
    if missing:
        raise SystemExit(
            "Missing required parameters (set via CLI flags or .env): "
            + ", ".join(missing)
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.chunks_dir).rglob(f"*{luminance.CHUNK_RESULT_SUFFIX}"))
    if not paths:
        marker = save_dir / "NO_CHUNKS_SKIPPED.txt"
        marker.write_text(
            f"No *{luminance.CHUNK_RESULT_SUFFIX} files found under "
            f"{args.chunks_dir}.\n"
        )
        print(f"No chunk files found; wrote {marker}")
        return

    tables, meta = luminance.load_chunk_results(paths)
    df = luminance.combine_chunk_tables(
        tables, low=args.low_percentile, high=args.high_percentile,
        scaling=args.scaling,
    )
    stem = f"{meta['session']}_{meta['camera']}_luminance"
    out = luminance.save_outputs(df, save_dir, stem, args.output_format, args.show,
                                args.shade_gaps, args.gap_seconds)
    print(f"Merged {len(paths)} chunk files, {len(df):,} samples -> "
          + ", ".join(str(v) for v in out.values()))


if __name__ == "__main__":
    main()
