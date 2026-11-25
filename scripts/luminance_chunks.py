"""Per-chunk step: sampled luminance + Harp timestamps for one slice of chunks.

Designed to run many times in parallel (e.g. one Code Ocean capsule instance per
shard). The camera's video chunks (``<Camera>_<timestamp>.mp4`` / ``.csv``) are
sorted chronologically and split into ``--num-shards`` contiguous, near-equal
slices; this process handles slice ``--shard-index`` (zero-based).

For each chunk it writes ``<chunk>_luminance_chunk.npz`` to ``--save-dir``
(Harp time, frame index, per-sample luminance, plus session/camera/rate/mode). Feed
all of those files to ``scripts/luminance_merge.py`` to get the final trace.

* Chunks whose output file already exists in ``--save-dir`` are skipped, so a
  restarted shard resumes.
* A shard with no chunks (``num_shards`` > number of chunks) writes a
  ``*_SKIPPED.txt`` marker instead, so downstream pipeline connections always
  receive at least one file.

Parameters come from CLI flags and/or ``.env`` (CLI overrides ``.env``):
``DATA_DIR``, ``SAVE_DIR``, ``ANONYMOUS_S3``, ``MAX_FILES``, ``SAMPLE_HZ``,
``SAMPLE_MODE``, ``SHARD_INDEX``, ``NUM_SHARDS``, ``WORKERS``, ``MAX_DECODES``,
``IO_THREADS``, ``DECODE_THREADS``.

Usage
-----
    python scripts/luminance_chunks.py --data-dir s3://aind-open-data/<session> \\
        --save-dir C:/results/chunks --shard-index 0 --num-shards 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pirouette_data import ingestion, luminance


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
                   help="Directory for the per-chunk .npz files (created if missing).")
    p.add_argument("--camera", default="TopCamera",
                   help="Camera sub-directory under behavior-videos.")
    p.add_argument("--shard-index", type=int,
                   default=int(os.getenv("SHARD_INDEX") or 0),
                   help="Zero-based index of the slice of chunks to process.")
    p.add_argument("--num-shards", type=int,
                   default=int(os.getenv("NUM_SHARDS") or 1),
                   help="Total number of slices the chunks are split into.")
    p.add_argument("--limit-files", type=int, default=_env_int_or_none("MAX_FILES"),
                   help="Consider only the first N chunks of the session (testing); "
                        "applied before sharding.")
    p.add_argument("--sample-hz", type=float,
                   default=float(os.getenv("SAMPLE_HZ") or 0.25),
                   help="Target sampling rate in Hz (0.25 = one keyframe per GOP).")
    p.add_argument("--mode", choices=("auto", "keyframe", "exact"),
                   default=os.getenv("SAMPLE_MODE", "keyframe"))
    p.add_argument("--anonymous-s3", action=argparse.BooleanOptionalAction,
                   default=_env_bool("ANONYMOUS_S3", True),
                   help="Access S3 with unsigned requests.")
    p.add_argument("--workers", type=int, default=_env_int_or_none("WORKERS"),
                   help="Chunks in flight at once (default: max_decodes + 1; each "
                        "holds ~140 MB).")
    p.add_argument("--max-decodes", type=int, default=_env_int_or_none("MAX_DECODES"),
                   help="Max simultaneous ffmpeg decodes (default: half the cores).")
    p.add_argument("--io-threads", type=int, default=int(os.getenv("IO_THREADS") or 8),
                   help="Parallel S3 range requests per chunk.")
    p.add_argument("--decode-threads", type=int,
                   default=int(os.getenv("DECODE_THREADS") or 1))
    args = p.parse_args()

    missing = [n for n in ("data_dir", "save_dir") if not getattr(args, n)]
    if missing:
        raise SystemExit(
            "Missing required parameters (set via CLI flags or .env): "
            + ", ".join(missing)
        )

    data_dir = args.data_dir.rstrip("/")
    session = data_dir.rsplit("/", 1)[-1]
    video_uri = f"{data_dir}/behavior-videos"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    s3 = luminance.make_s3_client(args.anonymous_s3, max_connections=8)
    keys = ingestion.list_camera_csv_keys(video_uri, args.camera, s3_client=s3)
    if not keys:
        raise SystemExit(f"No CSV files found for {args.camera!r} under {video_uri!r}.")
    if args.limit_files is not None:
        keys = keys[: args.limit_files]
    mine = luminance.select_shard(keys, args.shard_index, args.num_shards)

    tag = f"shard {args.shard_index} of {args.num_shards}"
    if not mine:
        marker = (
            save_dir
            / f"{session}_{args.camera}_shard{args.shard_index:04d}_SKIPPED.txt"
        )
        marker.write_text(
            f"{tag}: no chunks assigned ({len(keys)} chunks in session, "
            f"{args.num_shards} shards).\n"
        )
        print(f"{tag}: nothing to do; wrote {marker}")
        return

    todo = [
        k for k in mine
        if not (save_dir / f"{Path(k).stem}{luminance.CHUNK_RESULT_SUFFIX}").exists()
    ]
    print(f"{tag}: {len(mine)} chunks assigned, {len(mine) - len(todo)} already done.")
    for key, table, used in luminance.process_chunks(
        video_uri, args.camera, todo, sample_hz=args.sample_hz, mode=args.mode,
        anonymous=args.anonymous_s3, n_workers=args.workers,
        max_decodes=args.max_decodes, io_threads=args.io_threads,
        decode_threads=args.decode_threads,
    ):
        luminance.save_chunk_result(
            save_dir, table, session, args.camera, args.sample_hz, used
        )
    print(f"{tag}: wrote {len(todo)} chunk files to {save_dir}")


if __name__ == "__main__":
    main()
