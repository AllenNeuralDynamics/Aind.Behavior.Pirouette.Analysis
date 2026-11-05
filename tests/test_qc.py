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
    plot_num_clusters,
    plot_run_time,
)


def _write(folder, rows):
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "fit_summary.csv", index=False)


@pytest.fixture
def root(tmp_path):
    _write(tmp_path / "10m_base_5_powers", [
        {"subsampling_minutes": 20, "num_units": 700, "fit_time_min": 90, "status": "ok"},
        {"subsampling_minutes": 10, "num_units": 600, "fit_time_min": 70, "status": "ok"},
    ])
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
    for fn in (plot_num_clusters, plot_run_time):
        fig = fn(runs)
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert texts == list(runs)
        fn(runs, log_x=False)
