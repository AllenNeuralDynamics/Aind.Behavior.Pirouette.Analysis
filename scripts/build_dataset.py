"""Build and save the combined Pirouette dataset.

Runs the full pipeline for one session and writes a single CSV containing:

* pose keypoints in **pixels** and **millimetres**,
* **heading** estimates — per-frame ear-vector (``ear_heading_deg``) and
  commutator (``commutator_heading_deg``),
* **velocity** — instantaneous (``ear_velocity_mm_s``) and smoothed
  (``ear_velocity_smooth_mm_s``),
* **behaviour** classification (``behavior``: rest / movement),
* timing columns (``harp_time``, ``time_since_start``, ``datetime_pacific``).

Parameters come from CLI flags and/or a ``.env`` file (CLI overrides ``.env``);
see :mod:`pirouette_data.cli`. The output is written to
``<save_dir>/<session>_pirouette_dataset.csv`` where ``<session>`` is the root
datetime folder of the S3 data dir (e.g. ``854393_2026-06-09_19-34-26``).

Usage
-----
    python scripts/build_dataset.py                     # all params from .env
    python scripts/build_dataset.py --data-dir s3://... # override .env
    python scripts/build_dataset.py --limit-files 1     # quick partial build
"""

from __future__ import annotations

import time

import pandas as pd

from pirouette_data import behavior_classification as bc
from pirouette_data import ingestion, kinematics, processing
from pirouette_data.cli import BuildConfig, resolve_config


def build_dataframe(config: BuildConfig) -> pd.DataFrame:
    """Run the full pipeline and return the combined DataFrame.

    Parameters
    ----------
    config:
        Resolved :class:`pirouette_data.cli.BuildConfig`.

    Returns
    -------
    pandas.DataFrame
        The combined per-frame dataset.
    """
    # 1. Pose (pixels) + Harp time + datetime, concatenated & time-ordered.
    print(f"[1/6] Loading pose + Harp time from {config.s3_video_uri} ...")
    df = ingestion.build_dataset(
        config.pose_dir,
        config.s3_video_uri,
        anonymous=config.anonymous_s3,
        max_files=config.max_files,
    )
    print(f"      pose frames: {df.shape}")

    # 2. Pixel -> mm (chamber-calibrated, origin at upper-left corner).
    print("[2/6] Converting keypoints to mm ...")
    df = processing.append_mm_columns(
        df,
        length_mm=config.chamber_length_mm,
        width_mm=config.chamber_width_mm,
        likelihood_threshold=config.likelihood_threshold,
    )

    # 3. Per-frame ear-vector heading.
    print("[3/6] Ear-vector heading ...")
    df = kinematics.append_ear_heading(
        df,
        likelihood_threshold=config.likelihood_threshold,
        forward_sign=config.forward_sign,
    )

    # 4. Commutator heading (offset auto-calibrated from the ears).
    print(f"[4/6] Commutator heading from {config.s3_behavior_uri} ...")
    df = kinematics.append_commutator_heading(
        df,
        config.s3_behavior_uri,
        direction=config.commutator_direction,
        forward_sign=config.forward_sign,
        likelihood_threshold=config.likelihood_threshold,
        anonymous=config.anonymous_s3,
    )

    # 5. Velocity: instantaneous + smoothed (signed forward/backward).
    print("[5/6] Ear-midpoint velocity (instantaneous + smoothed) ...")
    df = kinematics.append_ear_velocity(
        df,
        likelihood_threshold=config.likelihood_threshold,
        forward_sign=config.forward_sign,
        smoothing_sigma=config.smoothing_sigma,
        method=config.velocity_method,
    )

    # 6. Rest / movement classification (on the smoothed velocity).
    print("[6/6] Behaviour classification ...")
    df = bc.append_behavior_labels(
        df,
        min_bout_s=config.min_bout_s,
        bridge_gap_s=config.bridge_gap_s,
        log=config.log_otsu,
    )

    return df


def main(argv: list[str] | None = None) -> None:
    """Resolve config, build the dataset, and save it to CSV."""
    config = resolve_config(argv)
    print("Pirouette dataset build")
    print(f"  session   : {config.session_name}")
    print(f"  pose_dir  : {config.pose_dir}")
    print(f"  data_dir  : {config.data_dir}")
    print(f"  save_dir  : {config.save_dir}")
    if config.max_files is not None:
        print(f"  limit     : first {config.max_files} pose file(s)")

    start = time.perf_counter()
    df = build_dataframe(config)

    config.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_path
    if config.output_format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - start
    print(
        f"\nSaved {df.shape[0]:,} rows x {df.shape[1]} cols to {out_path} "
        f"({elapsed:.1f}s)"
    )
    print(f"  behaviour threshold: {df.attrs.get('behavior_velocity_threshold', float('nan')):.1f} mm/s")


if __name__ == "__main__":
    main()
