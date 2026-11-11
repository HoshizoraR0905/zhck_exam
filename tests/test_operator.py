import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from distributed_xgb import DistributedXGBClassifier


@pytest.fixture
def data():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=6, random_state=7)
    return np.column_stack([X, np.ones(len(X))]), y


@pytest.mark.parametrize("params", [dict(max_depth=0), dict(n_estimators=-1), dict(n_jobs=0),
    dict(max_bin=1), dict(learning_rate=2), dict(subsample=0), dict(gamma=-1),
    dict(random_state=None), dict(unknown_typo=3), dict(objective="multi:softprob"),
    dict(device="cuda"), dict(tree_method="exact"), dict(max_leaves=-1), dict(nthread=4),
    dict(early_stopping_rounds=0), dict(importance_type="typo")])
def test_invalid_params(params):
    with pytest.raises(ValueError):
        DistributedXGBClassifier(**params)


def test_extra_native_param():
    model = DistributedXGBClassifier(params={"max_leaves": 12, "grow_policy": "lossguide"})
    assert model.params["max_leaves"] == 12


def test_split_reduction_metrics_persistence(data, tmp_path):
    X, y = data
    frame = pd.DataFrame(X, columns=[f"column{i}" for i in range(X.shape[1])])
    model = DistributedXGBClassifier("single", n_estimators=8)
    assert model.feature_reduction(frame, y) is model
    assert not hasattr(model, "reducer_")
    model.train(frame, y)
    assert model.reducer_.n_features_in_ == 11
    assert len(model.output_feature_names_) == 10
    expected_variance = X[model.train_indices_].astype(np.float32).var(axis=0, dtype=np.float64)
    np.testing.assert_allclose(model.reducer_.variances_, expected_variance, rtol=1e-6, atol=1e-6)
    assert set(model.evaluate(plot_dir=tmp_path / "plots")) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
    assert len(list((tmp_path / "plots").glob("*.png"))) == 3
    loaded = DistributedXGBClassifier.load_model(model.save_model(tmp_path))
    np.testing.assert_array_equal(model.predict_proba(frame), loaded.predict_proba(frame))
    assert loaded.evaluate(frame, y)["auc_roc"] > .8
    with pytest.raises(ValueError, match="explicit"):
        loaded.evaluate()
    with pytest.raises(ValueError, match="columns"):
        loaded.predict(frame[frame.columns[::-1]])
    model.set_params(max_depth=3)
    with pytest.raises(RuntimeError):
        model.predict(frame)


@pytest.mark.parametrize("components", [3, .9])
def test_pca(data, components, tmp_path):
    X, y = data
    model = DistributedXGBClassifier("single", n_estimators=3,
                                     reduction={"method": "pca", "n_components": components}).train(X, y)
    np.testing.assert_allclose(model.reducer_.mean_, X[model.train_indices_].mean(axis=0), atol=1e-6)
    if isinstance(components, int):
        assert model.reducer_.n_components_ == components
    else:
        assert model.reducer_.explained_variance_ratio_.sum() >= components
    loaded = DistributedXGBClassifier.load_model(model.save_model(tmp_path))
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))


def test_input_and_reducer_errors(data):
    X, y = data
    model = DistributedXGBClassifier("single", n_estimators=3)
    with pytest.raises(RuntimeError):
        model.predict(X)
    for bad in [dict(method="pca", n_components=1.0), dict(method="variance_threshold", variance_ratio=.95),
                dict(method="variance_threshold", threshold=-1), dict(method="typo")]:
        with pytest.raises(ValueError):
            model.feature_reduction(**bad)
    for bad_x, bad_y in [(X[:0], y[:0]), (X, y[:-1]), (X, np.zeros(len(y))), (X.ravel(), y)]:
        with pytest.raises(ValueError):
            model.train(bad_x, bad_y)
    model.train(X, np.where(y, "yes", "no"))
    assert set(model.predict(X)) <= {"yes", "no"}
    with pytest.raises(ValueError):
        model.predict(X[:, :-1])
    with pytest.raises(ValueError):
        model.evaluate(X, None)


def test_real_distributed_training(data, tmp_path):
    from distributed import Client
    X, y = data
    single = DistributedXGBClassifier("single", n_estimators=5, n_jobs=2).train(X, y)
    distributed = DistributedXGBClassifier("dask", n_estimators=5, n_jobs=1).train(
        X, y, distributed_config={"n_workers": 2, "threads_per_worker": 1})
    np.testing.assert_array_equal(single.validation_indices_, distributed.validation_indices_)
    assert {e["rank"] for e in distributed.worker_evidence_} == {0, 1}
    assert len({e["pid"] for e in distributed.worker_evidence_}) == 2
    assert all(e["world_size"] == 2 for e in distributed.worker_evidence_)
    assert distributed.evaluate()["accuracy"] > .7
    with pytest.raises(ValueError, match="No.*client"):
        Client.current()
    loaded = DistributedXGBClassifier.load_model(distributed.save_model(tmp_path))
    np.testing.assert_array_equal(loaded.predict_proba(X), distributed.predict_proba(X))


def test_distributed_config_validation(data):
    X, y = data
    for config in [{"typo": 1}, {"n_workers": 0}, {"threads_per_worker": 4}]:
        with pytest.raises(ValueError):
            DistributedXGBClassifier(n_jobs=1).train(X, y, distributed_config=config)
