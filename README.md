# Aind.Behavior.Pirouette.Analysis

Code for processing and analyzing chronic ephys and behavior (Pirouette) data.

The `pirouette_data` package turns local DeepLabCut pose‑tracking `.h5` files and
the matching behavior streams in the `aind-open-data` S3 bucket into a single,
analysis‑ready, per‑frame dataset: pose in pixels **and** millimetres, heading
estimates, signed velocity, and a rest/movement behaviour label — all aligned to
Harp time.

---

## Overview

The pipeline builds one time‑ordered DataFrame per session with these column
groups:

| Group | Columns | Source |
|---|---|---|
| Pose (pixels) | `<bodypart>_x/_y/_likelihood` | DeepLabCut `.h5` |
| Pose (mm) | `<bodypart>_x_mm/_y_mm` | chamber‑calibrated, origin at upper‑left corner |
| Timing | `harp_time`, `time_since_start`, `datetime_pacific`, `source_file`, `frame` | camera CSV (S3) + Aeon API |
| Heading | `ear_heading_deg`, `commutator_heading_deg` | ear keypoints / commutator turns (S3) |
| Velocity | `ear_velocity_mm_s`, `ear_velocity_smooth_mm_s` | ear‑midpoint, signed forward/backward |
| Behaviour | `behavior` (`rest` / `movement`) | velocity threshold + min‑bout filter |

`time_since_start` is measured from the **first timestamp of the whole
experiment** (the earliest camera CSV in S3), not the first pose frame.
`datetime_pacific` is timezone‑aware (`America/Los_Angeles`, DST‑aware).

---

## Installation

Uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv sync --all-extras          # create .venv and install everything
```

Alternatively, without uv:

```bash
pip install -e .
```

### Jupyter kernel

```bash
uv run python -m ipykernel install --user --name pirouette_data \
  --display-name "Python (pirouette_data)"
```

Then select **Python (pirouette_data)** in Jupyter / VS Code.

---

## Package layout

`src`‑layout, import name `pirouette_data`:

| Module | Purpose |
|---|---|
| `ingestion` | Load DLC pose `.h5`, join per‑frame Harp time from the S3 camera CSVs, concatenate into one time‑ordered DataFrame (`build_dataset`). |
| `processing` | Chamber‑based pixel→mm calibration; append `<bodypart>_x_mm/_y_mm` relative to the upper‑left corner (`estimate_chamber_scale`, `append_mm_columns`). |
| `kinematics` | Heading (ear‑vector & commutator) and signed ear‑midpoint velocity, instantaneous + Gaussian‑smoothed (`append_ear_heading`, `append_commutator_heading`, `append_ear_velocity`). |
| `behavior_classification` | Velocity‑threshold rest/movement labels with unsupervised (Otsu) threshold and a minimum‑bout‑duration filter (`append_behavior_labels`). |
| `cli` | Command‑line / `.env` configuration for the build script (`resolve_config`, `BuildConfig`). |

---

## Building a dataset

`scripts/build_dataset.py` runs the whole pipeline for one session and saves
`<save_dir>/<session>_pirouette_dataset.parquet`, where `<session>` is the S3
root datetime folder (e.g. `854393_2026-06-09_19-34-26`).

```bash
uv run python scripts/build_dataset.py                 # all params from .env
uv run python scripts/build_dataset.py --format csv     # CSV instead of parquet
uv run python scripts/build_dataset.py --limit-files 1  # quick partial build
uv run python scripts/build_dataset.py --data-dir s3://aind-open-data/<session>
```

### Configuration (`.env` + CLI)

Parameters are resolved with precedence **CLI flags > `.env` > built‑in
defaults**. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| `.env` key / CLI flag | Default | Meaning |
|---|---|---|
| `POSE_DIR` / `--pose-dir` | — | Local directory of DLC pose `.h5` files |
| `DATA_DIR` / `--data-dir` | — | AWS S3 session URI (`s3://aind-open-data/<session>`) |
| `SAVE_DIR` / `--save-dir` | — | Output directory |
| `OUTPUT_FORMAT` / `--format` | `parquet` | `parquet` or `csv` |
| `LIKELIHOOD_THRESHOLD` / `--likelihood-threshold` | `0.6` | Minimum DLC likelihood |
| `CHAMBER_LENGTH_MM` / `--chamber-length-mm` | `373` | Chamber length (x scale) |
| `CHAMBER_WIDTH_MM` / `--chamber-width-mm` | `194` | Chamber width (y scale) |
| `SMOOTHING_SIGMA` / `--smoothing-sigma` | `1.5` | Gaussian sigma (frames) for velocity |
| `VELOCITY_METHOD` / `--velocity-method` | `signed_speed` | `signed_speed` or `projection` |
| `FORWARD_SIGN` / `--forward-sign` | `1` | Nose‑ward orthogonal sign (flip if heading 180° out) |
| `COMMUTATOR_DIRECTION` / `--commutator-direction` | `1` | Commutator rotation sign |
| `MIN_BOUT_S` / `--min-bout-s` | `0.5` | Minimum movement‑bout duration (s) |
| `BRIDGE_GAP_S` / `--bridge-gap-s` | `0.2` | Bridge movement dips shorter than this (s) |
| `LOG_OTSU` / `--log-otsu` | `true` | Log‑scale Otsu threshold (captures slow movement) |
| `ANONYMOUS_S3` / `--anonymous-s3` | `true` | Unsigned access to the public bucket |
| `MAX_FILES` / `--limit-files` | all | Process only the first N pose files |

`.env` is git‑ignored; `.env.example` is the committed template.

### Library usage

The same steps are available programmatically:

```python
from pirouette_data import ingestion, processing, kinematics
from pirouette_data import behavior_classification as bc

s3 = "s3://aind-open-data/854393_2026-06-09_19-34-26"
df = ingestion.build_dataset("path/to/pose_data", f"{s3}/behavior-videos")
df = processing.append_mm_columns(df, likelihood_threshold=0.6)
df = kinematics.append_ear_heading(df, likelihood_threshold=0.6)
df = kinematics.append_commutator_heading(df, f"{s3}/behavior")
df = kinematics.append_ear_velocity(df, likelihood_threshold=0.6, smoothing_sigma=1.5)
df = bc.append_behavior_labels(df)          # smoothed velocity, log-Otsu, 0.5 s min bout
```

---

## Methods notes

- **Harp time → datetime** uses the [Aeon API](https://github.com/SainsburyWellcomeCentre/aeon_api)
  (`swc.aeon`); Harp seconds are relative to the 1904‑01‑01 UTC epoch, converted to Pacific.
- **Pixel → mm** uses the four chamber corners: length (373 mm) = median of the
  horizontal edges, width (194 mm) = median of the vertical edges; coordinates
  are expressed relative to the upper‑left corner.
- **Heading** (0° = facing right, standard quadrant) is the vector orthogonal to
  the inter‑aural axis; the commutator heading is anchored to this via an offset
  calibrated from the first frame with both ears tracked.
- **Velocity** is the ear‑midpoint displacement in mm/s, signed forward/backward
  by the heading; missing ears fall back to the present ear, then interpolation.
- **Behaviour**: log‑scale Otsu picks a threshold just above the rest floor so
  slow movement is detected, and a minimum‑bout filter keeps only movement that
  is continuous for ≥ `min_bout_s`.

---

## Notebooks

Runnable demos in `notebooks/` (kernel *Python (pirouette_data)*):

| Notebook | Shows |
|---|---|
| `load_pose_h5.ipynb` | Loading DLC pose `.h5` files |
| `ingest_pose_with_harp_time.ipynb` | Building the pose + Harp‑time dataset |
| `pixels_to_mm.ipynb` | Chamber calibration, pixel→mm conversion |
| `commutator_heading.ipynb` | Commutator‑derived heading |
| `ear_velocity.ipynb` | Velocity + smoothing‑sigma optimisation |
| `behavior_classification.ipynb` | Rest/movement classification |

---

## Development

- Tests: `uv run pytest`
- Lint / format: `uv run ruff check .` and `uv run black .`
- Code should be developed in the user's sandbox on a dev branch; submit a PR
  with **Prattbuw** as reviewer. AI‑generated lines that are ambiguous must be
  highlighted in the PR description.
- Every function needs a docstring (rendered via
  [mkdocstrings](https://mkdocstrings.github.io/python/)).

## Data structure

Recommended dataset structure: `subject_id/session/run`.

## Issue reporting

Report unresolved issues in the repository's **Issues** tab.
