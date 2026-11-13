"""Shared dataset and reduction configuration for the two comparison scripts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def normalize_experiment(values=None, *, workflow="compare", overrides=None):
    if not isinstance(values if values is not None else {}, dict):
        raise ValueError("experiment must be a JSON object")
    values = dict(values or {})
    for alias, canonical in [("n_samples", "samples"), ("random_state", "seed")]:
        if alias in values:
            if canonical in values and values[canonical] != values[alias]:
                raise ValueError(f"experiment.{alias} conflicts with experiment.{canonical}")
            values[canonical] = values.pop(alias)
    defaults = dict(samples=200000, n_features=30, n_informative=20, n_redundant=5,
                    n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0,
                    flip_y=0.01, weights=None, seed=42, model_seed=None, test_size=0.2,
                    append_constant=True)
    if workflow == "models":
        defaults.update(samples=2000, n_features=20, n_informative=12,
                        n_redundant=4, append_constant=False)
    extra = set(values) - set(defaults)
    if extra:
        raise ValueError(f"Unknown experiment keys: {sorted(extra)}")
    result = {**defaults, **values, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    if result["model_seed"] is None:
        result["model_seed"] = result["seed"]
    for name, minimum in [("samples", 20), ("n_features", 1), ("n_informative", 1),
                          ("n_redundant", 0), ("n_repeated", 0), ("n_classes", 2),
                          ("n_clusters_per_class", 1), ("seed", 0), ("model_seed", 0)]:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if result["n_classes"] != 2:
        raise ValueError("n_classes must be 2 (binary classification only)")
    if any(result[key] >= 2**32 for key in ("seed", "model_seed")):
        raise ValueError("seed/model_seed must be in [0, 2**32)")
    if result["n_informative"] + result["n_redundant"] + result["n_repeated"] > result["n_features"]:
        raise ValueError("n_informative + n_redundant + n_repeated must be <= n_features")
    if result["n_informative"] < math.log2(2 * result["n_clusters_per_class"]):
        raise ValueError("2 * n_clusters_per_class must be <= 2**n_informative")
    for name, low, high, inclusive_low in [("class_sep", 0, math.inf, False),
                                          ("flip_y", 0, 1, True), ("test_size", 0, 1, False)]:
        value = result[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or not (value >= low if inclusive_low else value > low)
                or (value > high if name == "flip_y" else value >= high)):
            raise ValueError(f"{name} has an invalid range (class_sep >0, flip_y in [0,1], test_size in (0,1))")
    if not isinstance(result["test_size"], float):
        raise ValueError("test_size must be a float in (0,1)")
    if not isinstance(result["append_constant"], bool):
        raise ValueError("append_constant must be true or false")
    weights = result["weights"]
    if weights is not None:
        if (not isinstance(weights, (list, tuple)) or len(weights) not in (1, 2)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) or not 0 <= v <= 1 for v in weights)
                or (len(weights) == 2 and not np.isclose(sum(weights), 1.0))):
            raise ValueError("weights must be null, one probability, or two probabilities summing to 1")
    if min(math.ceil(result["samples"] * result["test_size"]),
           result["samples"] - math.ceil(result["samples"] * result["test_size"])) < 2:
        raise ValueError("Both train and validation splits need at least two samples")
    return result


def generate_dataset(experiment, *, float32=False):
    names = ("n_features", "n_informative", "n_redundant", "n_repeated", "n_classes",
             "n_clusters_per_class", "class_sep", "flip_y", "weights")
    X, y = make_classification(n_samples=experiment["samples"], random_state=experiment["seed"],
                               **{key: experiment[key] for key in names})
    if float32:
        X = X.astype(np.float32)
    if experiment["append_constant"]:
        X = np.column_stack([X, np.ones(len(X), dtype=X.dtype)])
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Generated labels need two classes with >=2 samples each; increase samples or adjust weights/flip_y")
    return X, y


def normalize_dataset(values=None):
    if values is None:
        return {"source": "synthetic"}
    if not isinstance(values, dict) or set(values) - {"source", "path", "target_column", "feature_columns"}:
        raise ValueError("dataset accepts source, path, target_column, feature_columns only")
    result = {"source": "synthetic", **values}
    if result["source"] not in ("synthetic", "csv"):
        raise ValueError('dataset.source must be "synthetic" or "csv"')
    if result["source"] == "csv":
        for key in ("path", "target_column"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                raise ValueError(f"dataset.{key} is required for CSV and must be a nonempty string")
    if "feature_columns" in result:
        columns = result["feature_columns"]
        if (not isinstance(columns, list) or not columns
                or any(not isinstance(c, str) or not c.strip() for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("dataset.feature_columns must be a nonempty array of unique column names")
        if result.get("target_column") in columns:
            raise ValueError("dataset.feature_columns must not include the target column")
    return result


def load_dataset(experiment, dataset=None, *, float32=False):
    """Load once per comparison. Relative CSV paths use the working directory."""
    dataset = normalize_dataset(dataset)
    if dataset["source"] == "synthetic":
        return generate_dataset(experiment, float32=float32)
    path = Path(dataset["path"])
    if not path.is_file():
        raise ValueError(f"CSV file not found or not a file: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read CSV file {path}: {exc}") from exc
    target = dataset["target_column"]
    if target not in frame.columns:
        raise ValueError(f"CSV target column not found: {target}")
    columns = dataset.get("feature_columns", [c for c in frame.columns if c != target])
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV feature columns not found: {missing}")
    if frame.empty or not columns:
        raise ValueError("CSV must contain rows and at least one feature column")
    nonnumeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if nonnumeric:
        raise ValueError(f"CSV features must be numeric; non-numeric columns: {nonnumeric}")
    # The operator consumes float32; use the same representation in both scripts.
    with np.errstate(over="ignore"):
        X = frame[columns].astype(np.float32)
    if np.isinf(X.to_numpy()).any():
        raise ValueError("CSV features contain infinity or values outside the float32 range")
    y = frame[target].to_numpy()
    if pd.isna(y).any() or (pd.api.types.is_numeric_dtype(y) and not np.isfinite(y).all()):
        raise ValueError("CSV labels must be nonmissing and finite")
    counts = pd.Series(y).value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("CSV labels require exactly two classes with at least two samples each")
    validation_size = math.ceil(len(y) * experiment["test_size"])
    if min(validation_size, len(y) - validation_size) < 2:
        raise ValueError("CSV train and validation splits each need at least two samples; adjust test_size")
    return X, y


def dataset_metadata(X, y, experiment, dataset=None):
    dataset = normalize_dataset(dataset)
    if dataset["source"] == "csv":
        return {**dataset, "feature_columns": list(X.columns), "samples": len(y),
                "features": X.shape[1], "test_size": experiment["test_size"],
                "sha256": hashlib.sha256(X.to_numpy().tobytes()
                                         + pd.util.hash_array(y).tobytes()).hexdigest()}
    return {"source": "sklearn.datasets.make_classification", "samples": len(y),
            "features": X.shape[1], "generated_features": experiment["n_features"],
            "seed": experiment["seed"], "test_size": experiment["test_size"],
            "generation_params": dict(experiment),
            "sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()}


def resolve_reduction(values=None, *, default=None, method=None, threshold=None, n_components=None):
    raw = dict(default or {}) if values is None else values
    if not isinstance(raw, dict) or set(raw) - {"enabled", "method", "threshold", "n_components"}:
        raise ValueError("feature_reduction accepts enabled, method, threshold, n_components only")
    raw = dict(raw)
    enabled = raw.get("enabled", (values is not None or bool(raw))
                      and raw.get("method", "variance_threshold") is not None)
    if not isinstance(enabled, bool):
        raise ValueError("feature_reduction.enabled must be true or false")
    if method is not None:
        enabled = method != "none"
        raw["method"] = method
    if not enabled:
        if threshold is not None or n_components is not None:
            raise ValueError("Enable --feature-reduction variance_threshold/pca before overriding its parameters")
        return None
    selected = raw.get("method", "variance_threshold")
    if selected == "variance_threshold":
        if n_components is not None:
            raise ValueError("--n-components requires PCA")
        value = threshold if threshold is not None else raw.get("threshold", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("feature_reduction.threshold must be finite and >=0")
        return {"method": selected, "threshold": value}
    if selected == "pca":
        if threshold is not None:
            raise ValueError("--threshold requires variance_threshold")
        value = n_components if n_components is not None else raw.get("n_components", 0.95)
        if isinstance(value, bool) or not ((isinstance(value, int) and value > 0)
                                         or (isinstance(value, float) and 0 < value < 1)):
            raise ValueError("feature_reduction.n_components must be a positive integer or a float in (0,1)")
        return {"method": selected, "n_components": value}
    raise ValueError("feature_reduction.method must be variance_threshold or pca")


def load_model_configuration(source, *, include_dataset=False):
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text())
    if isinstance(source, list):
        result = (source, {}, None)
        return (*result, normalize_dataset()) if include_dataset else result
    if not isinstance(source, dict) or set(source) - {"experiment", "dataset", "feature_reduction", "models"}:
        raise ValueError("Model configuration must be a list or an object with experiment, dataset, feature_reduction, models")
    if "models" not in source:
        raise ValueError("Model configuration requires a models array")
    if not isinstance(source["models"], list) or not source["models"]:
        raise ValueError("models must be a nonempty array")
    dataset = normalize_dataset(source.get("dataset"))
    result = (source["models"], source.get("experiment", {}), source.get("feature_reduction"))
    return (*result, dataset) if include_dataset else result


def add_experiment_arguments(parser):
    parser.add_argument("--samples", "--n-samples", dest="samples", type=int)
    parser.add_argument("--seed", "--random-state", dest="seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--test-size", type=float)
    for name in ("n_features", "n_informative", "n_redundant", "n_repeated", "n_classes", "n_clusters_per_class"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--class-sep", type=float)
    parser.add_argument("--flip-y", type=float)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--append-constant", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-reduction", choices=["none", "variance_threshold", "pca"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--n-components", type=lambda text: float(text) if any(c in text for c in ".eE") else int(text))


def experiment_overrides(args):
    keys = ("samples", "seed", "model_seed", "test_size", "n_features", "n_informative", "n_redundant",
            "n_repeated", "n_classes", "n_clusters_per_class", "class_sep", "flip_y", "weights", "append_constant")
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
