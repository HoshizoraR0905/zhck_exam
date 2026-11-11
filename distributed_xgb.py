"""Binary XGBoost operator with local inference and process-based Dask training."""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class _Participation(xgb.callback.TrainingCallback):
    """Evidence collected inside the actual collective training call, on each rank."""
    def after_training(self, model):
        from distributed import get_worker
        from xgboost import collective
        worker = get_worker()
        worker.xgb_participation = {
            "address": worker.address, "pid": os.getpid(),
            "rank": collective.get_rank(), "world_size": collective.get_world_size(),
        }
        return model


def _evidence(dask_worker=None):
    return getattr(dask_worker, "xgb_participation", None)


def _reset_evidence(dask_worker=None):
    dask_worker.xgb_participation = None


class DistributedXGBClassifier:
    """Use framework='single' or 'dask'. Inputs are in-memory numeric tables.

    feature_reduction configures a reducer; only train fits it, after splitting.
    All predictions accept ORIGINAL features. Binary labels may be strings.
    """
    DEFAULTS = dict(learning_rate=0.1, max_depth=6, n_estimators=100,
                    subsample=1.0, colsample_bytree=1.0, min_child_weight=1.0,
                    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                    objective="binary:logistic", tree_method="hist", max_bin=512,
                    random_state=42, n_jobs=2, device="cpu", eval_metric="logloss")

    def __init__(self, framework="dask", params=None, reduction=None, **kwargs):
        if framework not in {"single", "dask"}:
            raise ValueError("framework must be 'single' or 'dask'")
        self.framework = framework
        self.params = dict(self.DEFAULTS)
        self.model_ = None
        self.reduction = None
        self.set_params(**{**(params or {}), **kwargs})
        if reduction is not None:
            self.feature_reduction(**reduction)

    def set_params(self, **params):
        candidate = {**self.params, **params}
        for name in ("max_depth", "n_estimators", "max_bin", "n_jobs"):
            v = candidate[name]
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if candidate["max_bin"] < 2:
            raise ValueError("max_bin must be >= 2")
        seed = candidate["random_state"]
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        for name in ("learning_rate", "subsample", "colsample_bytree"):
            v = candidate[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("min_child_weight", "gamma", "reg_alpha", "reg_lambda"):
            v = candidate[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if candidate["objective"] != "binary:logistic":
            raise ValueError("Only objective='binary:logistic' is supported")
        if candidate["device"] != "cpu":
            raise ValueError("This operator supports device='cpu' only")
        if candidate["tree_method"] not in {"hist", "approx", "exact"}:
            raise ValueError("tree_method must be hist, approx, or exact")
        if self.framework == "dask" and candidate["tree_method"] == "exact":
            raise ValueError("Dask does not support tree_method='exact'")
        if candidate.get("booster", "gbtree") not in {"gbtree", "dart"}:
            raise ValueError("Only tree boosters gbtree and dart are supported")
        if candidate.get("enable_categorical", False):
            raise ValueError("Categorical features must be encoded before using this numeric operator")
        metric = candidate.get("eval_metric")
        if callable(metric):
            raise ValueError("Use named XGBoost evaluation metrics; callable metrics are unsupported")
        if candidate.get("importance_type") not in {None, "weight", "gain", "cover", "total_gain", "total_cover"}:
            raise ValueError("Unsupported importance_type")
        rounds = candidate.get("early_stopping_rounds")
        if rounds is not None and (isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1):
            raise ValueError("early_stopping_rounds must be a positive integer")
        # Aliases could silently override validated values and break CPU comparisons.
        if set(candidate) & {"eta", "nthread", "seed", "lambda", "alpha", "num_class"}:
            raise ValueError("Use canonical learning_rate/n_jobs/random_state/reg_lambda/reg_alpha; num_class is unsupported")
        # The installed native engine validates extra booster parameters and values.
        # sklearn-only controls are recognized via the installed wrapper, not a fixed allowlist.
        native = xgb.XGBClassifier(**candidate).get_xgb_params()
        native.update(n_jobs=1, validate_parameters=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                xgb.train(native, xgb.DMatrix(np.array([[0.], [1.], [2.], [3.]]),
                                            label=[0, 1, 0, 1]), num_boost_round=1)
            unused = [str(w.message) for w in caught if "not used" in str(w.message)]
            if unused:
                raise ValueError("Unknown or unused XGBoost parameter: " + " ".join(unused))
        except xgb.core.XGBoostError as exc:
            raise ValueError(f"Invalid XGBoost configuration: {exc}") from exc
        self.params = candidate
        self.model_ = None
        return self

    def feature_reduction(self, X=None, y=None, method="variance_threshold", **kwargs):
        """Configure only; X/y are accepted for interface compatibility, never fitted here.

        method=None disables. VarianceThreshold uses threshold (absolute variance,
        >=0); PCA uses n_components (integer count or explained variance in (0,1)).
        """
        if method is None:
            if kwargs:
                raise ValueError("Disabled reduction accepts no parameters")
            self.reduction = None
        elif method == "variance_threshold":
            if set(kwargs) - {"threshold"}:
                raise ValueError("VarianceThreshold accepts threshold only")
            threshold = kwargs.get("threshold", 0.0)
            if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
                raise ValueError("threshold must be finite and >= 0")
            self.reduction = dict(method=method, threshold=threshold)
        elif method == "pca":
            if set(kwargs) - {"n_components"}:
                raise ValueError("PCA accepts n_components only")
            n = kwargs.get("n_components", 0.95)
            if isinstance(n, bool) or not ((isinstance(n, int) and n > 0) or
                                         (isinstance(n, float) and 0 < n < 1)):
                raise ValueError("n_components must be a positive integer or a float in (0,1)")
            self.reduction = dict(method=method, n_components=n)
        else:
            raise ValueError("method must be None, variance_threshold, or pca")
        self.model_ = None
        return self

    @staticmethod
    def _array(X):
        try:
            a = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric features") from exc
        if a.ndim != 2 or min(a.shape) == 0 or np.isinf(a).any():
            raise ValueError("X must be a nonempty 2D table without infinities")
        return a

    def train(self, X, y, test_size=0.2, distributed_config=None):
        self.model_ = None
        a = self._array(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(a) or pd.isna(labels).any():
            raise ValueError("y must be a nonmissing 1D vector matching X")
        encoder = LabelEncoder().fit(labels)
        if len(encoder.classes_) != 2:
            raise ValueError("Exactly two label classes are required")
        if isinstance(test_size, bool) or not isinstance(test_size, float) or not 0 < test_size < 1:
            raise ValueError("test_size must be a float in (0,1)")
        config = dict(distributed_config or {})
        if self.framework == "single" and config:
            raise ValueError("distributed_config is only used by framework='dask'")
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        if self.feature_names_in_ is not None and len(set(self.feature_names_in_)) != a.shape[1]:
            raise ValueError("DataFrame feature names must be unique")
        self.n_features_in_ = a.shape[1]
        self.encoder_ = encoder
        self.classes_ = encoder.classes_
        yi = encoder.transform(labels)
        self.train_indices_, self.validation_indices_ = train_test_split(
            np.arange(len(a)), test_size=test_size, random_state=self.params["random_state"], stratify=yi)
        tr, va = self.train_indices_, self.validation_indices_
        self.X_validation_ = a[va].copy()  # original space, reused by evaluate
        self.y_validation_ = labels[va].copy()
        start = time.perf_counter()
        self.reducer_ = None
        if self.reduction:
            options = {k: v for k, v in self.reduction.items() if k != "method"}
            self.reducer_ = (VarianceThreshold(**options) if self.reduction["method"] == "variance_threshold"
                             else PCA(**options, svd_solver="full"))
        train_x = a[tr]
        valid_x = a[va]
        if self.reducer_ is not None:
            if not np.isfinite(a).all():
                raise ValueError("Feature reduction requires finite input; impute missing data first")
            train_x = self.reducer_.fit_transform(train_x)
            valid_x = self.reducer_.transform(valid_x)
        self.timings_ = {"preprocessing_seconds": time.perf_counter() - start,
                         "cluster_startup_seconds": 0.0, "data_distribution_seconds": 0.0}
        if self.reducer_ is None:
            self.output_feature_names_ = ([str(v) for v in self.feature_names_in_] if self.feature_names_in_ is not None
                                          else [f"f{i}" for i in range(a.shape[1])])
        elif isinstance(self.reducer_, PCA):
            self.output_feature_names_ = [f"PC{i+1}" for i in range(train_x.shape[1])]
        else:
            names = np.array(self.feature_names_in_ if self.feature_names_in_ is not None
                             else [f"f{i}" for i in range(a.shape[1])])
            self.output_feature_names_ = names[self.reducer_.get_support()].astype(str).tolist()
        self.worker_evidence_ = []
        if self.framework == "single":
            model = xgb.XGBClassifier(**self.params)
            start = time.perf_counter()
            model.fit(train_x, yi[tr], eval_set=[(valid_x, yi[va])], verbose=False)
            self.timings_["fit_seconds"] = time.perf_counter() - start
        else:
            model = self._train_dask(train_x, yi[tr], valid_x, yi[va], config)
        self.model_ = model
        return self

    def _train_dask(self, X, y, vx, vy, config):
        import dask.array as da
        from dask import delayed
        from distributed import Client, LocalCluster, wait
        from xgboost.dask import DaskXGBClassifier

        unknown = set(config) - {"n_workers", "threads_per_worker", "memory_limit", "scheduler_address"}
        if unknown:
            raise ValueError(f"Unknown distributed_config keys: {sorted(unknown)}")
        n_workers = config.get("n_workers", 2)
        threads = config.get("threads_per_worker", self.params["n_jobs"])
        for key, value in {"n_workers": n_workers, "threads_per_worker": threads}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if threads != self.params["n_jobs"]:
            raise ValueError("threads_per_worker must equal n_jobs (threads per Dask worker)")
        if min(len(X), len(vx)) < n_workers:
            raise ValueError("Both train and validation need at least one row per worker")
        cluster = client = None
        try:
            start = time.perf_counter()
            if config.get("scheduler_address"):
                client = Client(config["scheduler_address"], set_as_default=False)
            else:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads,
                                       processes=True, dashboard_address=None, host="127.0.0.1",
                                       memory_limit=config.get("memory_limit", "auto"))
                client = Client(cluster, set_as_default=False)
            client.wait_for_workers(n_workers, timeout=60)
            workers = sorted(client.scheduler_info()["workers"])
            if len(workers) != n_workers:
                raise ValueError("Connected cluster worker count must equal n_workers")
            info = client.scheduler_info()["workers"]
            if any(info[w]["nthreads"] != threads for w in workers):
                raise ValueError("Connected worker thread counts must equal threads_per_worker")
            client.run(_reset_evidence)
            self.timings_["cluster_startup_seconds"] = time.perf_counter() - start
            start = time.perf_counter()

            def distribute(a):
                chunks = []
                for w, block in zip(workers, np.array_split(a, n_workers)):
                    future = client.scatter(block, workers=[w], broadcast=False, hash=False)
                    chunks.append(da.from_delayed(delayed(future), shape=block.shape, dtype=block.dtype))
                return da.concatenate(chunks, axis=0)

            dx, dy, dvx, dvy = map(distribute, (X, y, vx, vy))
            dx, dy, dvx, dvy = client.persist([dx, dy, dvx, dvy])
            with client.as_current():
                wait([dx, dy, dvx, dvy])
            self.timings_["data_distribution_seconds"] = time.perf_counter() - start
            p = dict(self.params)
            p["callbacks"] = [*(p.get("callbacks") or []), _Participation()]
            trained = DaskXGBClassifier(**p)
            trained.client = client
            start = time.perf_counter()
            with client.as_current():
                trained.fit(dx, dy, eval_set=[(dvx, dvy)], verbose=False)
            self.timings_["fit_seconds"] = time.perf_counter() - start
            self.worker_evidence_ = [e for e in client.run(_evidence).values() if e is not None]
            ranks = {e["rank"] for e in self.worker_evidence_}
            pids = {e["pid"] for e in self.worker_evidence_}
            if (ranks != set(range(n_workers)) or len(pids) != n_workers or
                    any(e["world_size"] != n_workers for e in self.worker_evidence_)):
                raise RuntimeError(f"Worker participation verification failed: {self.worker_evidence_}")
            # Convert to a plain sklearn wrapper: no client, futures, or cluster in artifact.
            local = xgb.XGBClassifier(**{k: v for k, v in self.params.items() if k != "callbacks"})
            local.load_model(bytearray(trained.get_booster().save_raw(raw_format="ubj")))
            return local
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Call train before prediction, evaluation, or saving")

    def _transform(self, X):
        self._require_fitted()
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            if list(X.columns) != self.feature_names_in_:
                raise ValueError("DataFrame columns/order differ from training")
        a = self._array(X)
        if a.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} original features")
        return self.reducer_.transform(a) if self.reducer_ is not None else a

    def predict_proba(self, X):
        transformed = self._transform(X)
        return self.model_.predict_proba(transformed)

    def predict(self, X):
        self._require_fitted()
        return self.encoder_.inverse_transform((self.predict_proba(X)[:, 1] >= 0.5).astype(int))

    def evaluate(self, X_test=None, y_test=None, plot_dir=None):
        self._require_fitted()
        if (X_test is None) != (y_test is None):
            raise ValueError("Provide both X_test and y_test, or neither")
        if X_test is None and not hasattr(self, "X_validation_"):
            raise ValueError("Reloaded models require explicit evaluation data")
        X = self.X_validation_ if X_test is None else X_test
        y = self.y_validation_ if y_test is None else np.asarray(y_test)
        if np.asarray(y).ndim != 1 or len(y) != len(X):
            raise ValueError("y_test must be a 1D vector matching X_test")
        truth = self.encoder_.transform(y)
        if len(np.unique(truth)) != 2:
            raise ValueError("AUC evaluation requires both classes")
        prob = self.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = dict(accuracy=float(accuracy_score(truth, pred)),
                       precision=float(precision_score(truth, pred, zero_division=0)),
                       recall=float(recall_score(truth, pred, zero_division=0)),
                       f1=float(f1_score(truth, pred, zero_division=0)),
                       auc_roc=float(roc_auc_score(truth, prob)))
        if plot_dir is not None:
            self._plots(truth, pred, prob, metrics, Path(plot_dir))
        return metrics

    def _plots(self, truth, pred, prob, metrics, directory):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        directory.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(truth, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=self.classes_, yticklabels=self.classes_)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        fig.tight_layout(); fig.savefig(directory / "confusion_matrix.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(truth, prob)
        ax.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.5f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        ax.legend(); fig.tight_layout(); fig.savefig(directory / "roc.png", dpi=160); plt.close(fig)
        importance = self.model_.feature_importances_
        order = np.argsort(importance)[-20:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(self.output_feature_names_)[order], importance[order])
        ax.set(xlabel="Normalized gain", title="Feature importance (transformed space)")
        fig.tight_layout(); fig.savefig(directory / "feature_importance.png", dpi=160); plt.close(fig)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    def save_model(self, path):
        """Save a trusted joblib bundle, omitting cached validation rows/indices."""
        self._require_fitted()
        path = Path(path)
        if path.suffix != ".joblib":
            path = path / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k not in {
            "X_validation_", "y_validation_", "train_indices_", "validation_indices_"}}
        joblib.dump({"format_version": 1, "state": state}, path, compress=3)
        return path

    @classmethod
    def load_model(cls, path):
        """Load trusted artifacts only. Reloaded evaluate requires explicit data."""
        path = Path(path)
        if path.is_dir():
            path = path / "model.joblib"
        bundle = joblib.load(path)
        if bundle.get("format_version") != 1:
            raise ValueError("Unsupported model bundle version")
        instance = cls.__new__(cls)
        instance.__dict__.update(bundle["state"])
        return instance
