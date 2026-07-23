"""Command-line configuration for the dataset build script.

Resolves build parameters with this precedence (highest first):

1. explicit command-line arguments,
2. a local ``.env`` file / environment variables,
3. built-in defaults.

``scripts/build_dataset.py`` calls :func:`resolve_config` to obtain a
:class:`BuildConfig`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Env parsing helpers
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return default if val is None or val.strip() == "" else float(val)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return default if val is None or val.strip() == "" else int(val)


def _env_int_or_none(name: str) -> int | None:
    val = os.getenv(name)
    return None if val is None or val.strip() == "" else int(val)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BuildConfig:
    """Resolved parameters for a dataset build."""

    pose_dir: Path
    data_dir: str  # AWS S3 session URI, e.g. s3://aind-open-data/854393_2026-06-09_19-34-26
    save_dir: Path
    likelihood_threshold: float
    smoothing_sigma: float
    chamber_length_mm: float
    chamber_width_mm: float
    forward_sign: int
    commutator_direction: int
    velocity_method: str
    min_bout_s: float
    bridge_gap_s: float
    log_otsu: bool
    anonymous_s3: bool
    max_files: int | None

    @property
    def session_name(self) -> str:
        """Root datetime folder of the S3 dir, e.g. ``854393_2026-06-09_19-34-26``."""
        return self.data_dir.rstrip("/").rsplit("/", 1)[-1]

    @property
    def s3_video_uri(self) -> str:
        """S3 URI of the per-camera frame-index CSV directory."""
        return f"{self.data_dir.rstrip('/')}/behavior-videos"

    @property
    def s3_behavior_uri(self) -> str:
        """S3 URI of the ``behavior`` directory (commutator, etc.)."""
        return f"{self.data_dir.rstrip('/')}/behavior"

    @property
    def output_path(self) -> Path:
        """Destination CSV: ``<save_dir>/<session>_pirouette_dataset.csv``."""
        return self.save_dir / f"{self.session_name}_pirouette_dataset.csv"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, using environment variables as defaults.

    Call :func:`load_dotenv` (done by :func:`resolve_config`) before this so
    ``.env`` values populate the defaults.
    """
    p = argparse.ArgumentParser(
        prog="build_dataset.py",
        description=(
            "Build the combined Pirouette dataset (pose px+mm, ear & commutator "
            "heading, velocity, behaviour) and save it as a CSV."
        ),
    )

    # Paths (required; may come from .env)
    p.add_argument("--pose-dir", default=os.getenv("POSE_DIR"),
                   help="Local directory of DeepLabCut pose .h5 files.")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR"),
                   help="AWS S3 session URI (e.g. s3://aind-open-data/854393_2026-06-09_19-34-26).")
    p.add_argument("--save-dir", default=os.getenv("SAVE_DIR"),
                   help="Directory to write <session>_pirouette_dataset.csv.")

    # Numeric parameters
    p.add_argument("--likelihood-threshold", type=float,
                   default=_env_float("LIKELIHOOD_THRESHOLD", 0.6),
                   help="Minimum DLC likelihood for keypoints.")
    p.add_argument("--smoothing-sigma", type=float,
                   default=_env_float("SMOOTHING_SIGMA", 1.5),
                   help="Gaussian sigma (frames) for velocity smoothing.")
    p.add_argument("--chamber-length-mm", type=float,
                   default=_env_float("CHAMBER_LENGTH_MM", 373.0),
                   help="Known chamber length in mm (x-axis scale).")
    p.add_argument("--chamber-width-mm", type=float,
                   default=_env_float("CHAMBER_WIDTH_MM", 194.0),
                   help="Known chamber width in mm (y-axis scale).")
    p.add_argument("--forward-sign", type=int, choices=(1, -1),
                   default=_env_int("FORWARD_SIGN", 1),
                   help="Nose-ward orthogonal sign for heading/velocity (flip if 180 out).")
    p.add_argument("--commutator-direction", type=int, choices=(1, -1),
                   default=_env_int("COMMUTATOR_DIRECTION", 1),
                   help="Commutator rotation sign.")
    p.add_argument("--velocity-method", choices=("signed_speed", "projection"),
                   default=os.getenv("VELOCITY_METHOD", "signed_speed"),
                   help="Signed velocity convention.")
    p.add_argument("--min-bout-s", type=float, default=_env_float("MIN_BOUT_S", 0.5),
                   help="Minimum movement-bout duration (s) for classification.")
    p.add_argument("--bridge-gap-s", type=float, default=_env_float("BRIDGE_GAP_S", 0.2),
                   help="Bridge movement dips shorter than this (s).")

    # Booleans (--flag / --no-flag)
    p.add_argument("--log-otsu", action=argparse.BooleanOptionalAction,
                   default=_env_bool("LOG_OTSU", True),
                   help="Use log-scale Otsu threshold (captures slow movement).")
    p.add_argument("--anonymous-s3", action=argparse.BooleanOptionalAction,
                   default=_env_bool("ANONYMOUS_S3", True),
                   help="Access S3 with unsigned (anonymous) requests.")

    # Testing / partial builds
    p.add_argument("--limit-files", type=int, default=_env_int_or_none("MAX_FILES"),
                   help="Process only the first N pose files (for quick runs).")
    return p


def resolve_config(argv: list[str] | None = None, use_dotenv: bool = True) -> BuildConfig:
    """Resolve the build configuration from CLI args, ``.env`` and defaults.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv``).
    use_dotenv:
        When ``True`` (default), load a local ``.env`` into the environment
        before reading defaults. ``.env`` never overrides variables already set
        in the real environment.

    Returns
    -------
    BuildConfig
        The fully resolved configuration.

    Raises
    ------
    SystemExit
        If any required path (pose_dir, data_dir, save_dir) is unset.
    """
    if use_dotenv:
        load_dotenv()
    args = build_parser().parse_args(argv)

    missing = [
        name
        for name in ("pose_dir", "data_dir", "save_dir")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(
            "Missing required parameters (set via CLI flags or .env): "
            + ", ".join(missing)
        )

    return BuildConfig(
        pose_dir=Path(args.pose_dir),
        data_dir=args.data_dir,
        save_dir=Path(args.save_dir),
        likelihood_threshold=args.likelihood_threshold,
        smoothing_sigma=args.smoothing_sigma,
        chamber_length_mm=args.chamber_length_mm,
        chamber_width_mm=args.chamber_width_mm,
        forward_sign=args.forward_sign,
        commutator_direction=args.commutator_direction,
        velocity_method=args.velocity_method,
        min_bout_s=args.min_bout_s,
        bridge_gap_s=args.bridge_gap_s,
        log_otsu=args.log_otsu,
        anonymous_s3=args.anonymous_s3,
        max_files=args.limit_files,
    )
