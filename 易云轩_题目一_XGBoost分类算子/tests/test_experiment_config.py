import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from compare import parse_configuration
from compare_models import compare_models
from distributed_xgb import DistributedXGBClassifier
from experiment_config import (generate_dataset, load_model_configuration, normalize_experiment,
                               resolve_reduction)


def experiment():
    return dict(samples=240, n_features=12, n_informative=6, n_redundant=2, n_repeated=1,
                n_classes=2, class_sep=1.6, flip_y=.02, weights=[.6, .4], seed=17,
                model_seed=23, test_size=.25, append_constant=False)


@pytest.mark.parametrize("workflow", ["compare", "models"])
def test_legacy_generation_is_identical(workflow):
    config = normalize_experiment({"samples": 120}, workflow=workflow)
    X, y = generate_dataset(config, float32=workflow == "compare")
    features, informative, redundant = (30, 20, 5) if workflow == "compare" else (20, 12, 4)
    expected, labels = make_classification(n_samples=120, n_features=features, n_informative=informative,
                                          n_redundant=redundant, random_state=42)
    if workflow == "compare":
        expected = np.column_stack([expected.astype(np.float32), np.ones(120, dtype=np.float32)])
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(y, labels)


def test_aliases_and_explicit_cli_overrides(tmp_path):
    data = {"experiment": {**experiment(), "n_samples": 240, "random_state": 17},
            "model": {"random_state": 23},
            "feature_reduction": {"enabled": True, "method": "pca", "n_components": .9}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    _, config = parse_configuration(["--config", str(path)])
    assert config["experiment"]["n_features"] == 12
    assert config["experiment"]["seed"] == 17
    assert config["model"]["random_state"] == 23
    assert config["feature_reduction"] == {"method": "pca", "n_components": .9}
    _, config = parse_configuration(["--config", str(path), "--samples", "260", "--seed", "19",
                                    "--model-seed", "29", "--feature-reduction", "none"])
    assert config["experiment"]["samples"] == 260
    assert config["experiment"]["seed"] == 19
    assert config["model"]["random_state"] == config["experiment"]["model_seed"] == 29
    assert config["feature_reduction"] is None


@pytest.mark.parametrize("invalid", [
    {"samples": 21, "n_samples": 22}, {"seed": 1, "random_state": 2},
    {"n_features": 3}, {"n_repeated": -1}, {"n_informative": 1}, {"n_classes": 3},
    {"class_sep": 0}, {"flip_y": 1.1}, {"weights": [.2, .2]}, {"weights": [-.1, 1.1]},
    {"weights": [True]}, {"seed": None}, {"seed": 2**32}, {"model_seed": -1},
    {"samples": 20, "test_size": 0.001}, {"append_constant": 1}, {"typo": 7},
])
def test_invalid_data_configuration(invalid):
    with pytest.raises(ValueError):
        normalize_experiment(invalid)


@pytest.mark.parametrize("invalid", [
    {"enabled": "false"}, {"enabled": True, "method": "typo"},
    {"enabled": True, "method": "pca", "n_components": 1.0},
    {"enabled": True, "method": "variance_threshold", "threshold": -1}, {"typo": 1},
])
def test_invalid_reduction_configuration(invalid):
    with pytest.raises(ValueError):
        resolve_reduction(invalid)


def test_reduction_switch_and_legacy_list():
    assert resolve_reduction({}) == {"method": "variance_threshold", "threshold": 0.0}
    assert resolve_reduction({"enabled": False, "method": "pca", "n_components": .8}) is None
    assert resolve_reduction({"enabled": True, "method": "pca", "n_components": 3}) == {"method": "pca", "n_components": 3}
    assert resolve_reduction({"method": "variance_threshold", "threshold": .1}) == {"method": "variance_threshold", "threshold": .1}
    old = [{"name": "old", "framework": "single"}]
    assert load_model_configuration(old) == (old, {}, None)


def test_nondefault_generator_matches_sklearn():
    config = normalize_experiment(experiment(), workflow="models")
    actual_X, actual_y = generate_dataset(config)
    expected_X, expected_y = make_classification(n_samples=240, n_features=12, n_informative=6,
                                                n_redundant=2, n_repeated=1, n_classes=2,
                                                class_sep=1.6, flip_y=.02, weights=[.6, .4], random_state=17)
    np.testing.assert_array_equal(actual_X, expected_X)
    np.testing.assert_array_equal(actual_y, expected_y)


@pytest.mark.parametrize("params", [{"colsample_bylevel": 0}, {"colsample_bynode": 0},
                                    {"colsample_bylevel": 1.1}, {"colsample_bynode": True}])
def test_invalid_feature_sampling(params):
    with pytest.raises(ValueError, match="must be in"):
        DistributedXGBClassifier(params=params)


def test_structured_config_with_provided_arrays(tmp_path):
    settings = experiment()
    X, y = generate_dataset(normalize_experiment(settings, workflow="models"))
    source = {"experiment": settings, "feature_reduction": {"enabled": True, "method": "pca", "n_components": 3},
              "models": [{"name": name, "framework": "single", "model": {"n_estimators": 3, "n_jobs": 1}}
                         for name in ["a", "b"]]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(source))
    results = compare_models(X, y, path, output_dir=tmp_path)
    assert results["split"]["random_state"] == 23
    assert len(results["split"]["validation_indices"]) == 60
    assert results["reduction"] == {"method": "pca", "n_components": 3}
    assert all(run["complexity"]["features"] == 3 for run in results["runs"])
    assert results["dataset"]["source"] == "provided_arrays"


def test_legacy_array_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps([{"name": "legacy", "framework": "single", "model": {"n_estimators": 2, "n_jobs": 1}}]))
    output = tmp_path / "output"
    run = subprocess.run([str(root / '.venv/bin/python'), 'compare_models.py', '--config', str(source),
                          '--samples', '120', '--seed', '31', '--test-size', '.25', '--output', str(output)],
                         cwd=root, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / 'comparison.json').read_text())
    assert result['dataset']['features'] == 20 and result['dataset']['samples'] == 120
    assert result['dataset']['seed'] == result['runs'][0]['effective_params']['random_state'] == 31
    assert len(result['split']['validation_indices']) == 30
    assert result['reduction'] is None


@pytest.mark.parametrize("script", ["compare.py", "compare_models.py"])
def test_end_to_end_configured_data_sampling_reduction(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = experiment()
    model = dict(n_estimators=4, max_depth=2, colsample_bytree=.8,
                 colsample_bylevel=.75, colsample_bynode=.6)
    if script == "compare.py":
        source = {"experiment": settings, "feature_reduction": {"enabled": True, "method": "pca", "n_components": 4},
                  "model": model, "distributed": {"n_workers": 2, "total_threads": 2}}
    else:
        settings["append_constant"] = True
        source = {"experiment": settings,
                  "feature_reduction": {"enabled": True, "method": "variance_threshold", "threshold": 0.0},
                  "models": [{"name": "single", "framework": "single", "model": {**model, "n_jobs": 2}},
                             {"name": "dask", "framework": "dask", "model": {**model, "n_jobs": 1},
                              "distributed": {"n_workers": 2, "threads_per_worker": 1}}]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(source))
    output = tmp_path / "output"
    run = subprocess.run([str(root / '.venv/bin/python'), script, "--config", str(path), "--output", str(output)],
                         cwd=root, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    results = json.loads((output / "comparison.json").read_text())
    assert results["dataset"]["samples"] == 240
    assert results["dataset"]["features"] == (12 if script == "compare.py" else 13)
    assert results["dataset"]["seed"] == 17
    assert results["dataset"]["generation_params"]["n_repeated"] == 1
    runs = list(results["runs"].values()) if script == "compare.py" else results["runs"]
    for record in runs:
        for key in ("colsample_bytree", "colsample_bylevel", "colsample_bynode"):
            assert record["effective_params"][key] == model[key]
            native = record["native_booster_config"]["learner"]["gradient_booster"]["tree_train_param"]
            assert float(native[key]) == pytest.approx(model[key])
        assert record["effective_params"]["random_state"] == 23
        reduced = record["transformed_features"] if script == "compare.py" else record["complexity"]["features"]
        assert reduced == (4 if script == "compare.py" else 12)
    _, y = generate_dataset(normalize_experiment(settings))
    expected = train_test_split(np.arange(240), test_size=.25, random_state=23, stratify=y)
    if script == "compare.py":
        saved = np.load(output / "validation_predictions.npz")
        np.testing.assert_array_equal(saved["indices"], expected[1])
        assert all(record["reload_verified"] for record in runs)
    else:
        np.testing.assert_array_equal(results["split"]["train_indices"], expected[0])
        np.testing.assert_array_equal(results["split"]["validation_indices"], expected[1])
