"""Unit tests for :mod:`pirouette_data.cli`."""

from __future__ import annotations

from pathlib import Path

import pytest

from pirouette_data import cli

REQUIRED_ENV = {
    "POSE_DIR": "/data/pose",
    "DATA_DIR": "s3://aind-open-data/854393_2026-06-09_19-34-26",
    "SAVE_DIR": "/data/out",
}


def _set_env(monkeypatch, **extra):
    for k, v in {**REQUIRED_ENV, **extra}.items():
        monkeypatch.setenv(k, str(v))


# ---------------------------------------------------------------------------
# defaults / env
# ---------------------------------------------------------------------------
def test_env_provides_defaults(monkeypatch):
    _set_env(monkeypatch)
    cfg = cli.resolve_config([], use_dotenv=False)
    assert cfg.pose_dir == Path("/data/pose")
    assert cfg.data_dir == "s3://aind-open-data/854393_2026-06-09_19-34-26"
    assert cfg.save_dir == Path("/data/out")
    # built-in defaults
    assert cfg.likelihood_threshold == 0.6
    assert cfg.smoothing_sigma == 1.5
    assert cfg.log_otsu is True
    assert cfg.anonymous_s3 is True
    assert cfg.max_files is None


def test_env_values_are_read(monkeypatch):
    _set_env(monkeypatch, SMOOTHING_SIGMA="3.0", MIN_BOUT_S="1.0", LOG_OTSU="false")
    cfg = cli.resolve_config([], use_dotenv=False)
    assert cfg.smoothing_sigma == 3.0
    assert cfg.min_bout_s == 1.0
    assert cfg.log_otsu is False


# ---------------------------------------------------------------------------
# CLI overrides env
# ---------------------------------------------------------------------------
def test_cli_overrides_env(monkeypatch):
    _set_env(monkeypatch, SMOOTHING_SIGMA="3.0")
    cfg = cli.resolve_config(
        ["--smoothing-sigma", "0.5", "--data-dir", "s3://bucket/999_2020-01-01_00-00-00"],
        use_dotenv=False,
    )
    assert cfg.smoothing_sigma == 0.5  # CLI wins over env's 3.0
    assert cfg.data_dir == "s3://bucket/999_2020-01-01_00-00-00"


def test_cli_boolean_override(monkeypatch):
    _set_env(monkeypatch, LOG_OTSU="true", ANONYMOUS_S3="true")
    cfg = cli.resolve_config(["--no-log-otsu", "--no-anonymous-s3"], use_dotenv=False)
    assert cfg.log_otsu is False
    assert cfg.anonymous_s3 is False


# ---------------------------------------------------------------------------
# derived properties
# ---------------------------------------------------------------------------
def test_derived_paths_and_session(monkeypatch):
    _set_env(monkeypatch)
    cfg = cli.resolve_config([], use_dotenv=False)
    assert cfg.session_name == "854393_2026-06-09_19-34-26"
    assert cfg.s3_video_uri.endswith("/behavior-videos")
    assert cfg.s3_behavior_uri.endswith("/behavior")
    # default format is parquet
    assert cfg.output_format == "parquet"
    assert cfg.output_path == Path(
        "/data/out/854393_2026-06-09_19-34-26_pirouette_dataset.parquet"
    )


def test_format_csv_override(monkeypatch):
    _set_env(monkeypatch)
    cfg = cli.resolve_config(["--format", "csv"], use_dotenv=False)
    assert cfg.output_format == "csv"
    assert cfg.output_path.suffix == ".csv"


# ---------------------------------------------------------------------------
# missing required
# ---------------------------------------------------------------------------
def test_missing_required_raises(monkeypatch):
    for k in REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        cli.resolve_config([], use_dotenv=False)
