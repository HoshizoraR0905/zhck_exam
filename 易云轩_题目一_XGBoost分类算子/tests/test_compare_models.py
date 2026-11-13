import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from compare_models import compare_models, format_model_comparison
from distributed_xgb import DistributedXGBClassifier


@pytest.fixture
def data():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=13)
    return pd.DataFrame(np.column_stack([X, np.ones(len(X))])), np.where(y, "yes", "no")


@pytest.mark.parametrize("reduction", [{"method": "variance_threshold", "threshold": 0.0},
                                       {"method": "pca", "n_components": 3}])
def test_shared_split_preprocessing_parameters_and_outputs(data, tmp_path, monkeypatch, capsys, reduction):
    X, y = data
    captured = []
    original = DistributedXGBClassifier.train

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(self)
        return result

    monkeypatch.setattr(DistributedXGBClassifier, "train", record)
    configurations = [
        {"name": "shallow", "framework": "single", "model": {"n_estimators": 4, "max_depth": 1, "n_jobs": 1}},
        {"name": "deeper", "framework": "single", "model": {"n_estimators": 6, "max_depth": 2,
                                                                  "n_jobs": 1, "scale_pos_weight": 1.2}},
    ]
    path = tmp_path / "models.json"
    path.write_text(json.dumps(configurations))
    results = compare_models(X, y, path, random_state=13, test_size=.25,
                             reduction=reduction, output_dir=tmp_path, plot=True)
    a, b = captured
    np.testing.assert_array_equal(a.train_indices_, b.train_indices_)
    np.testing.assert_array_equal(a.validation_indices_, b.validation_indices_)
    assert results["split"]["train_indices"] == a.train_indices_.tolist()
    assert results["split"]["validation_indices"] == a.validation_indices_.tolist()
    for operator in captured:
        training = X.iloc[operator.train_indices_].to_numpy(dtype=np.float32)
        if reduction["method"] == "pca":
            np.testing.assert_allclose(operator.reducer_.mean_, training.mean(axis=0), atol=1e-6)
        else:
            np.testing.assert_allclose(operator.reducer_.variances_, training.var(axis=0, dtype=np.float64), atol=1e-6)
    assert b.params["scale_pos_weight"] == 1.2
    assert b.params["max_depth"] == 2 and a.params["max_depth"] == 1
    for run, operator, rounds in zip(results["runs"], captured, [4, 6]):
        assert run["effective_params"] == operator.params
        assert run["validation_metrics"] == operator.evaluate()
        assert run["train_metrics"] == operator.evaluate(X.iloc[operator.train_indices_], y[operator.train_indices_])
        assert run["complexity"]["trees"] == run["complexity"]["rounds"] == rounds
        assert run["complexity"]["nodes"] >= run["complexity"]["leaves"] >= rounds
    assert json.loads((tmp_path / "comparison.json").read_text()) == results
    csv = pd.read_csv(tmp_path / "comparison.csv")
    assert csv["name"].tolist() == ["shallow", "deeper"]
    np.testing.assert_allclose(csv["validation_metrics.auc_roc"], [r["validation_metrics"]["auc_roc"] for r in results["runs"]])
    np.testing.assert_allclose(csv["timings.fit_seconds"], [r["timings"]["fit_seconds"] for r in results["runs"]])
    report = format_model_comparison(results)
    assert (tmp_path / "terminal_report.txt").read_text() == report + "\n"
    assert report in capsys.readouterr().out
    assert (tmp_path / "train_validation.png").stat().st_size > 0


def test_single_and_real_dask_with_derived_threads(data, tmp_path):
    X, y = data
    results = compare_models(X, y, [
        {"name": "single", "framework": "single", "model": {"n_estimators": 3, "n_jobs": 2}},
        {"name": "dask", "framework": "dask", "model": {"n_estimators": 3},
         "distributed": {"n_workers": 2, "threads_per_worker": 1}},
    ], output_dir=tmp_path)
    single, dask = results["runs"]
    assert single["total_threads"] == dask["total_threads"] == 2
    assert dask["effective_params"]["n_jobs"] == dask["distributed_config"]["threads_per_worker"] == 1
    assert {e["rank"] for e in dask["worker_evidence"]} == {0, 1}
    assert all(e["world_size"] == 2 for e in dask["worker_evidence"])
    assert all(run["split_verified"] for run in results["runs"])
    assert not (tmp_path / "train_validation.png").exists()


@pytest.mark.parametrize("invalid", [
    {"name": "ok", "framework": "single"},
    {"name": "", "framework": "single"},
    {"name": "bad", "framework": "unknown"},
    {"name": "bad", "framework": "single", "model": {"random_state": 7}},
    {"name": "bad", "framework": "single", "model": {"unknown_parameter": 1}},
    {"name": "bad", "framework": "single", "distributed": {"n_workers": 2}},
    {"name": "bad", "framework": "dask", "model": {"n_jobs": 2}, "distributed": {"threads_per_worker": 1}},
    {"name": "bad", "framework": "dask", "distributed": {"n_workers": 0}},
    {"name": "bad", "framework": "dask", "distributed": {"unknown_setting": 1}},
    {"name": "bad", "framework": "single", "model": []},
    {"name": "bad", "framework": "single", "typo": {}},
])
def test_invalid_configuration_fails_before_any_training(data, tmp_path, monkeypatch, invalid):
    def unexpected_training(*args, **kwargs):
        pytest.fail("No training should occur when any configuration is invalid")

    monkeypatch.setattr(DistributedXGBClassifier, "train", unexpected_training)
    with pytest.raises(ValueError):
        compare_models(*data, [{"name": "ok", "framework": "single"}, invalid], output_dir=tmp_path)


@pytest.mark.parametrize("configurations", [[], {}, "not_a_file.json"])
def test_invalid_configuration_list(data, tmp_path, configurations):
    with pytest.raises((ValueError, OSError)):
        compare_models(*data, configurations, output_dir=tmp_path)


def test_runtime_split_consistency_guard(data, tmp_path, monkeypatch):
    original = DistributedXGBClassifier.train
    calls = 0

    def altered_split(self, *args, **kwargs):
        nonlocal calls
        original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            self.validation_indices_ = self.validation_indices_[::-1]
        return self

    monkeypatch.setattr(DistributedXGBClassifier, "train", altered_split)
    with pytest.raises(RuntimeError, match="split differs"):
        compare_models(*data, [{"name": name, "framework": "single", "model": {"n_estimators": 2}}
                              for name in ["first", "second"]], output_dir=tmp_path)
