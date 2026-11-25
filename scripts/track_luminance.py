"""Track normalized luminance (external lighting proxy) from behavior video.

For each video chunk of one camera on S3, streams only the sampled frames (by
HTTP range request), computes their mean luminance, averages each sample with the
previous one, then rescales to the session min-max (0-1; see --scaling).
See :mod:`pirouette_data.luminance` for the sampling modes and their speed.

Outputs (also see ``luminance_chunks.py`` / ``luminance_merge.py`` for a
distributed version of this same pipeline), all written to ``--save-dir`` (``<session>_<camera>_luminance.*``):

* ``.npy`` — float64 array of shape ``(N, 2)``: column 0 = Harp timestamp (s, from
  the camera CSV), column 1 = normalized luminance in [0, 1].
* ``.png`` — normalized luminance (black trace) vs. Pacific wall-clock time.
* ``.parquet`` (or ``.csv``) — full table with all columns.

A per-chunk cache is kept in ``<save_dir>/luminance_cache/<session>_<camera>/`` so
an interrupted run resumes where it stopped.

Parameters come from CLI flags and/or ``.env`` (CLI overrides ``.env``):
``DATA_DIR``, ``SAVE_DIR``, ``ANONYMOUS_S3``, ``MAX_FILES``, ``OUTPUT_FORMAT``,
``SAMPLE_HZ``, ``SAMPLE_MODE``, ``WORKERS``, ``MAX_DECODES``, ``IO_THREADS``, ``DECODE_THREADS``.

Usage
-----
    python scripts/track_luminance.py --limit-files 1     # quick test
    python scripts/track_luminance.py --sample-hz 0.1     # one sample per ~10 s
    python scripts/track_luminance.py --data-dir s3://aind-open-data/854393_2026-08-28_20-34-08
    python scripts/track_luminance.py --save-dir D:/results --show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pirouette_data import luminance


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int_or_none(name: str) -> int | None:
    val = os.getenv(name)
    return None if val is None or val.strip() == "" else int(val)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR"),
                   help="S3 session URI (s3://aind-open-data/<session>).")
    p.add_argument("--save-dir", default=os.getenv("SAVE_DIR"),
                   help="Directory for the .npy / .png / table outputs and the cache "
                        "(created if missing).")
    p.add_argument("--camera", default="TopCamera",
                   help="Camera sub-directory under behavior-videos.")
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
    p.add_argument("--anonymous-s3", action=argparse.BooleanOptionalAction,
                   default=_env_bool("ANONYMOUS_S3", True),
                   help="Access S3 with unsigned requests.")
    p.add_argument("--limit-files", type=int, default=_env_int_or_none("MAX_FILES"),
                   help="Process only the first N video chunks.")
    p.add_argument("--sample-hz", type=float,
                   default=float(os.getenv("SAMPLE_HZ") or 0.25),
                   help="Target sampling rate in Hz (default 0.25 = one keyframe "
                        "per GOP, the fastest path). Higher rates need --mode exact.")
    p.add_argument("--mode", choices=("auto", "keyframe", "exact"),
                   default=os.getenv("SAMPLE_MODE", "keyframe"),
                   help="keyframe: only keyframes (fast, <= 1 sample/GOP); exact: "
                        "decode whole GOPs for any rate (slow); auto: pick by rate.")
    p.add_argument("--workers", type=int, default=_env_int_or_none("WORKERS"),
                   help="Chunks in flight at once (default: max_decodes + 1). Each holds "
                        "~140 MB, so raising this raises memory use.")
    p.add_argument("--max-decodes", type=int, default=_env_int_or_none("MAX_DECODES"),
                   help="Max simultaneous ffmpeg decodes = CPU cores used "
                        "(default: half the cores).")
    p.add_argument("--io-threads", type=int, default=int(os.getenv("IO_THREADS") or 8),
                   help="Parallel S3 range requests per chunk.")
    p.add_argument("--decode-threads", type=int,
                   default=int(os.getenv("DECODE_THREADS") or 1),
                   help="ffmpeg decode threads per chunk.")
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

    missing = [n for n in ("data_dir", "save_dir") if not getattr(args, n)]
    if missing:
        raise SystemExit(
            "Missing required parameters (set via CLI flags or .env): "
            + ", ".join(missing)
        )

    data_dir = args.data_dir.rstrip("/")
    session = data_dir.rsplit("/", 1)[-1]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = save_dir / "luminance_cache" / f"{session}_{args.camera}"

    df = luminance.build_luminance_dataset(
        f"{data_dir}/behavior-videos",
        camera=args.camera,
        cache_dir=cache_dir,
        anonymous=args.anonymous_s3,
        max_files=args.limit_files,
        low=args.low_percentile,
        high=args.high_percentile,
        scaling=args.scaling,
        sample_hz=args.sample_hz,
        mode=args.mode,
        n_workers=args.workers,
        max_decodes=args.max_decodes,
        io_threads=args.io_threads,
        decode_threads=args.decode_threads,
    )

    stem = f"{session}_{args.camera}_luminance"
    out = luminance.save_outputs(df, save_dir, stem, args.output_format, args.show,
                                args.shade_gaps, args.gap_seconds)
    print(f"Saved {len(df):,} samples -> " + ", ".join(str(v) for v in out.values()))


if __name__ == "__main__":
    main()
