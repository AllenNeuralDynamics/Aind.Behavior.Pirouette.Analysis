"""Tests for pirouette_data.qc (spike-sorting subsampling QC)."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from pirouette_data.qc import (
    find_run_dirs,
    label_from_folder,
    load_fit_summary,
    load_runs,
    parse_find_templates_durations,
    plot_find_templates_time,
    plot_num_clusters,
)

# Excerpt of a find_templates capsule log: one Start/Done pair per fit.
LOG = (
    "Starting DARTsort model fitting -> ../results/dartsort_rec_fit0000  RAM: 1 GB\n"
    "noise line\nDone. Total time: 4267.8s\n"
    "  Starting DARTsort model fitting -> ../results/dartsort_rec_fit0001  RAM: 1 GB\n"
    "Done. Total time: 5481.2s\n"
)


def _write(folder, rows):
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "fit_summary.csv", index=False)


@pytest.fixture
def root(tmp_path):
    _write(tmp_path / "10m_base_5_powers", [
        {"run_name": "rec_fit0001", "subsampling_minutes": 20, "num_units": 700,
         "fit_time_min": 90, "status": "ok"},
        {"run_name": "rec_fit0000", "subsampling_minutes": 10, "num_units": 600,
         "fit_time_min": 70, "status": "ok"},
    ])
    (tmp_path / "10m_base_5_powers" / "output").write_text(LOG)
    _write(tmp_path / "5m_base_6_powers", [
        {"subsampling_minutes": 5, "num_units": 400, "fit_time_min": 60, "status": "ok"},
        {"subsampling_minutes": 10, "num_units": 0, "fit_time_min": 0, "status": "failed"},
    ])
    (tmp_path / "not_a_result").mkdir()
    return tmp_path


def test_label_from_folder():
    assert label_from_folder("C:/x/10m_base_5_powers_motion") == "10m base 5 powers motion"


def test_find_run_dirs_ignores_folders_without_summary(root):
    assert [d.name for d in find_run_dirs(root)] == ["10m_base_5_powers", "5m_base_6_powers"]


def test_load_fit_summary_sorts_and_drops_failed(root):
    df = load_fit_summary(root / "10m_base_5_powers")
    assert list(df["subsampling_minutes"]) == [10, 20]
    with pytest.warns(UserWarning, match="status"):
        assert len(load_fit_summary(root / "5m_base_6_powers")) == 1


def test_missing_column_raises(tmp_path):
    _write(tmp_path / "bad", [{"subsampling_minutes": 1}])
    with pytest.raises(ValueError, match="missing columns"):
        load_fit_summary(tmp_path / "bad")


def test_load_runs_keys_and_empty(root, tmp_path):
    with pytest.warns(UserWarning):
        runs = load_runs(root)
    assert list(runs) == ["10m base 5 powers", "5m base 6 powers"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "empty")


def test_plots_have_legend_labels(root):
    with pytest.warns(UserWarning):
        runs = load_runs(root)
    for fn in (plot_num_clusters, plot_find_templates_time):
        fig = fn(runs)
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert texts == list(runs)
        fn(runs, log_x=False)


def test_parse_find_templates_durations(tmp_path):
    log = tmp_path / "output"
    # A trailing 'Done' with no preceding fit start is ignored.
    log.write_text(LOG + "Done. Total time: 1.0s\n")
    assert parse_find_templates_durations(log) == {"fit0000": 4267.8, "fit0001": 5481.2}


def test_find_templates_s_from_log(root):
    df = load_fit_summary(root / "10m_base_5_powers")
    assert list(df["find_templates_s"]) == [4267.8, 5481.2]  # sorted by subsampling time


def test_find_templates_s_falls_back_to_fit_time(root):
    with pytest.warns(UserWarning, match="find-templates duration"):
        df = load_fit_summary(root / "5m_base_6_powers", only_ok=False)
    assert list(df["find_templates_s"]) == [60 * 60.0, 0.0]


def test_find_templates_plot_ylabel(root):
    with pytest.warns(UserWarning):
        runs = load_runs(root)
    fig = plot_find_templates_time(runs)
    assert fig.axes[0].get_ylabel() == "find template run duration (h)"
