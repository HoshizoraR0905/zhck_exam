"""Configuration/reporting checks; the existing operator tests remain unchanged."""
import json
from pathlib import Path

import pytest

from compare import format_comparison, parse_configuration
from distributed_xgb import DistributedXGBClassifier


def test_defaults_and_worker_alias():
    _, config = parse_configuration(["--workers", "4"])
    assert config["distributed"]["threads_per_worker"] == 1
    assert config["model"] == {k: v for k, v in DistributedXGBClassifier.DEFAULTS.items() if k != "n_jobs"}


def test_json_and_cli_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"n_estimators": 5, "max_delta_step": 1,
                                         "scale_pos_weight": 2, "colsample_bynode": .8},
                                "distributed": {"n_workers": 2, "total_threads": 4},
                                "experiment": {"samples": 1000, "seed": 7}}))
    _, config = parse_configuration(["--config", str(path), "--n-workers", "4", "--total-threads", "8",
                                    "--estimators", "9", "--seed", "13"])
    assert config["model"]["n_estimators"] == 9
    assert config["model"]["max_delta_step"] == 1
    assert config["model"]["random_state"] == config["experiment"]["seed"] == 13
    assert config["experiment"]["samples"] == 1000
    assert config["distributed"]["threads_per_worker"] == 2


@pytest.mark.parametrize("configuration", [
    {"model": {"n_jobs": 2}}, {"model": {"unknown_parameter": 1}},
    {"model": {"n_workers": 4}}, {"distributed": {"total_threads": 3}},
    {"distributed": {"threads_per_worker": 3}}, {"distributed": {"n_workers": 0}},
    {"distributed": {"n_workers": True}}, {"distributed": {"typo": 1}},
    {"experiment": {"test_size": 1}}, {"model": []}, {"typo": {}},
])
def test_conflicts_fail_before_training(tmp_path, configuration, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    with pytest.raises(SystemExit) as error:
        parse_configuration(["--config", str(path)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("constraints", ["(1,0,-1)", (1, 0, -1), None])
def test_valid_multifeature_constraints(constraints):
    model = DistributedXGBClassifier(params={"monotone_constraints": constraints})
    assert model.params["monotone_constraints"] == constraints


def test_format_uses_saved_values():
    # Historical JSON remains unchanged. Add only the new reporting metadata.
    results = json.loads((Path(__file__).resolve().parents[1] / "results/comparison.json").read_text())
    results["dataset"].update(train_samples=160000, validation_samples=40000, test_size=.2)
    dask = results["runs"]["dask"]
    dask["distributed_config"] = {"threads_per_worker": dask["n_jobs"]}
    report = format_comparison(results)
    assert f"Workers verified: {results['workers']}/{results['workers']}" in report
    assert "160,000 train / 40,000 validation" in report
    for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("auc_roc", "ROC-AUC")]:
        row = next(line for line in report.splitlines() if line.startswith(label))
        assert row.split()[-2:] == [f"{results['runs'][mode]['metrics'][key]:.6f}" for mode in ("single", "dask")]
    for key, label in [("preprocessing_seconds", "Preprocessing"), ("cluster_startup_seconds", "Cluster startup"),
                       ("data_distribution_seconds", "Data distribution"), ("fit_seconds", "Fit")]:
        row = next(line for line in report.splitlines() if line.startswith(label))
        assert row.split()[-2:] == [f"{results['runs'][mode]['timings'][key]:.3f}" for mode in ("single", "dask")]
    row = next(line for line in report.splitlines() if line.startswith("Total train"))
    assert row.split()[-2:] == [f"{results['runs'][mode]['train_total_seconds']:.3f}" for mode in ("single", "dask")]
    comparison = results["comparison"]
    assert f"Prediction-label agreement: {comparison['label_agreement']:.4%}" in report
    for key in ("mean", "p95", "max"):
        assert f"{comparison[f'probability_abs_difference_{key}']:.8f}" in report
    for line in report.splitlines():
        assert "tcp://" not in line and '"pid"' not in line and '"rank"' not in line
    assert f"{results['comparison']['fit_speedup_single_over_dask']:.3f}x" in report
