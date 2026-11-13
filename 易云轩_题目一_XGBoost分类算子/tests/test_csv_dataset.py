"""CSV input validation and integration with the existing comparison workflows."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from compare import parse_configuration
from experiment_config import (dataset_metadata, generate_dataset, load_dataset,
                               load_model_configuration, normalize_dataset, normalize_experiment)


def csv_config(tmp_path, frame=None):
    if frame is None:
        frame = pd.DataFrame({"a": np.arange(80), "b": np.arange(80) % 7,
                              "label": ["no", "yes"] * 40})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return {"source": "csv", "path": str(path), "target_column": "label"}


def test_load_csv_and_selected_order(tmp_path):
    dataset = csv_config(tmp_path)
    experiment = normalize_experiment()
    X, y = load_dataset(experiment, dataset)
    assert list(X.columns) == ["a", "b"]
    assert X.dtypes.tolist() == [np.dtype("float32")] * 2
    assert y.tolist() == ["no", "yes"] * 40
    dataset["feature_columns"] = ["b", "a"]
    selected, labels = load_dataset(experiment, dataset, float32=True)
    np.testing.assert_array_equal(selected, X[["b", "a"]])
    np.testing.assert_array_equal(labels, y)
    assert selected.shape == (80, 2)  # append_constant applies only to synthetic data
    info = dataset_metadata(selected, labels, experiment, dataset)
    assert info["source"] == "csv" and info["feature_columns"] == ["b", "a"]
    assert info == dataset_metadata(*load_dataset(experiment, dataset), experiment, dataset)


@pytest.mark.parametrize("dataset", [
    [], {"source": "parquet"}, {"source": 1}, {"source": "csv"},
    {"source": "csv", "path": "data.csv"}, {"typo": 1},
    {"feature_columns": []}, {"feature_columns": "a"},
    {"feature_columns": [1]}, {"feature_columns": ["a", "a"]},
    {"target_column": "a", "feature_columns": ["a"]},
])
def test_invalid_dataset_settings(dataset):
    with pytest.raises(ValueError, match="dataset"):
        normalize_dataset(dataset)


@pytest.mark.parametrize("frame, update, message", [
    ({"a": [1, 2]}, {}, "target column"),
    ({"a": [1, 2], "label": [0, 1]}, {"feature_columns": ["missing"]}, "feature columns"),
    ({"a": ["red", "blue"] * 2, "label": [0, 1] * 2}, {}, "non-numeric"),
    ({"a": [np.inf] * 4, "label": [0, 1] * 2}, {}, "infinity"),
    ({"a": [1e40] * 4, "label": [0, 1] * 2}, {}, "float32"),
    ({"a": [1] * 4, "label": [0, 1, None, 0]}, {}, "nonmissing"),
    ({"a": [1] * 4, "label": [0, np.inf, 0, np.inf]}, {}, "finite"),
    ({"a": [1] * 4, "label": [0] * 4}, {}, "exactly two"),
    ({"a": [1] * 6, "label": [0, 1, 2] * 2}, {}, "exactly two"),
    ({"a": [1] * 4, "label": [0, 0, 0, 1]}, {}, "at least two"),
    ({"label": [0, 1] * 2}, {}, "at least one feature"),
    ({"a": [], "label": []}, {}, "contain rows"),
])
def test_invalid_csv_data(tmp_path, frame, update, message):
    dataset = {**csv_config(tmp_path, pd.DataFrame(frame)), **update}
    with pytest.raises(ValueError, match=message):
        load_dataset(normalize_experiment(), dataset)


def test_missing_empty_and_too_small_csv(tmp_path):
    dataset = {"source": "csv", "path": str(tmp_path / "missing.csv"), "target_column": "label"}
    with pytest.raises(ValueError, match="file not found"):
        load_dataset(normalize_experiment(), dataset)
    Path(dataset["path"]).touch()
    with pytest.raises(ValueError, match="Cannot read CSV"):
        load_dataset(normalize_experiment(), dataset)
    dataset = csv_config(tmp_path, pd.DataFrame({"a": [1] * 4, "label": [0, 1] * 2}))
    with pytest.raises(ValueError, match="splits each need"):
        load_dataset(normalize_experiment(), dataset)


def test_numeric_missing_values_preserved(tmp_path):
    dataset = csv_config(tmp_path, pd.DataFrame({"a": [1, np.nan] * 40, "label": [0, 1] * 40}))
    X, _ = load_dataset(normalize_experiment(), dataset)
    assert X["a"].isna().sum() == 40


@pytest.mark.parametrize("workflow", ["compare", "models"])
def test_omitted_dataset_keeps_legacy_generation(workflow):
    experiment = normalize_experiment({"samples": 80}, workflow=workflow)
    expected = generate_dataset(experiment, float32=workflow == "compare")
    actual = load_dataset(experiment, float32=workflow == "compare")
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert dataset_metadata(*actual, experiment)["source"] == "sklearn.datasets.make_classification"
    assert parse_configuration([])[1]["dataset"] == {"source": "synthetic"}
    legacy = [{"name": "legacy", "framework": "single"}]
    assert load_model_configuration(legacy) == (legacy, {}, None)
    assert load_model_configuration({"models": legacy}, include_dataset=True)[3] == {"source": "synthetic"}


def test_csv_cli_shared_data_splits_and_existing_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = csv_config(tmp_path)
    common = {"dataset": dataset, "experiment": {"model_seed": 23, "test_size": .25},
              "feature_reduction": {"enabled": True, "method": "pca", "n_components": 1}}
    model = {"n_estimators": 3, "max_depth": 2}
    configs = {
        "compare": {**common, "model": model, "distributed": {"n_workers": 2, "total_threads": 2}},
        "compare_models": {**common, "models": [
            {"name": "single", "framework": "single", "model": {**model, "n_jobs": 2}},
            {"name": "dask", "framework": "dask", "model": {**model, "n_jobs": 1},
             "distributed": {"n_workers": 2, "threads_per_worker": 1}}]},
    }
    hashes, splits = [], []
    for script, config in configs.items():
        path = tmp_path / f"{script}.json"
        path.write_text(json.dumps(config))
        output = tmp_path / script
        run = subprocess.run([sys.executable, str(root / f"{script}.py"), "--config", str(path),
                              "--output", str(output)], cwd=root, text=True, capture_output=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        result = json.loads((output / "comparison.json").read_text())
        hashes.append(result["dataset"]["sha256"])
        assert result["dataset"]["features"] == 2 and result["dataset"]["samples"] == 80
        assert "generation_params" not in result["dataset"]
        assert (output / "terminal_report.txt").exists()
        if script == "compare":
            splits.append(np.load(output / "validation_predictions.npz")["indices"])
            assert result["comparison"]["identical_split"]
            assert all(r["reload_verified"] and r["transformed_features"] == 1 for r in result["runs"].values())
            assert (output / "comparison.png").exists()
        else:
            splits.append(result["split"]["validation_indices"])
            assert all(r["split_verified"] and r["complexity"]["features"] == 1 for r in result["runs"])
    _, labels = load_dataset(normalize_experiment(), dataset)
    expected = train_test_split(np.arange(80), test_size=.25, random_state=23, stratify=labels)[1]
    for split in splits:
        np.testing.assert_array_equal(split, expected)
    assert hashes[0] == hashes[1]
