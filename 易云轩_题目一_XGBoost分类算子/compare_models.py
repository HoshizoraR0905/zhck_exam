"""Compare explicitly named XGBoost configurations; no search or tuning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare import training_diagnostics
from distributed_xgb import DistributedXGBClassifier
from experiment_config import (add_experiment_arguments, dataset_metadata, experiment_overrides,
                               load_dataset, load_model_configuration, normalize_experiment, resolve_reduction)

_UNSET = object()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Comparison configuration must be JSON-compatible, got {type(value).__name__}")


def format_model_comparison(results):
    """Render scores and timings directly from the complete result dictionary."""
    scores, complexity = [], []
    for run in results["runs"]:
        scores.append({"Name": run["name"], "Mode": run["framework"],
                       **{f"{prefix} {label}": run[source][key]
                          for prefix, source in [("Train", "train_metrics"), ("Val", "validation_metrics")]
                          for key, label in [("auc_roc", "AUC"), ("accuracy", "Acc"), ("f1", "F1")]},
                       "Fit (s)": run["timings"]["fit_seconds"]})
        complexity.append({"Name": run["name"], "Workers": run["workers"],
                           "Threads/worker": run["effective_params"]["n_jobs"],
                           "Total threads": run["total_threads"],
                           **run["complexity"]})
    split = results["split"]
    return (f"User-defined XGBoost comparison ({len(results['runs'])} configurations)\n"
            f"Samples: {results['dataset']['samples']:,} | Input features: {results['dataset']['features']} | "
            f"Identical split: {len(split['train_indices']):,} train / "
            f"{len(split['validation_indices']):,} validation | Model/split seed: {split['random_state']}\n"
            + (f"Data seed: {results['dataset']['seed']}\n" if "seed" in results['dataset'] else "") + "\n"
            + pd.DataFrame(scores).to_string(index=False, float_format=lambda v: f"{v:.6f}",
                                            formatters={"Fit (s)": lambda v: f"{v:.3f}"})
            + "\n\nModel complexity and execution resources\n"
            + pd.DataFrame(complexity).to_string(index=False)
            + "\n\nScores: 0–1. Fit time: seconds, excluding cluster startup and preprocessing."
              "\nComplexity counts describe all stored trees, including trees retained after early stopping."
              "\nRows follow input order; no configuration is selected automatically.")


def compare_models(X, y, configurations, *, test_size=None, random_state=None,
                   reduction=_UNSET, output_dir="results/model_comparison", plot=False,
                   dataset_info=None):
    """Train/evaluate a list, a shared configuration object, or its JSON file.

    Each entry contains name, framework, model, and optional distributed settings.
    The shared random_state is also the operator's split seed; conflicting per-model
    seeds are rejected. Reduction is shared and fitted by the operator after splitting.
    Returns complete results as a dictionary; CSV, JSON and a terminal table are saved.
    Supplied X/y remain authoritative; only the CLI loads the configured dataset.
    """
    configurations, raw_experiment, raw_reduction = load_model_configuration(configurations)
    experiment = normalize_experiment(raw_experiment, workflow="models",
                                      overrides={"test_size": test_size, "model_seed": random_state})
    test_size, random_state = experiment["test_size"], experiment["model_seed"]
    reduction = resolve_reduction(raw_reduction if reduction is _UNSET else reduction)
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("configurations must be a nonempty list, or a JSON file containing that list")
    prepared, names = [], set()
    for spec in configurations:
        if not isinstance(spec, dict) or set(spec) - {"name", "framework", "model", "distributed"}:
            raise ValueError("Each configuration accepts only name, framework, model, distributed")
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Configuration names must be nonempty, unique strings")
        names.add(name)
        params, execution = spec.get("model", {}), spec.get("distributed", {})
        if not isinstance(params, dict) or not isinstance(execution, dict):
            raise ValueError(f"{name}: model and distributed must be dictionaries")
        params, execution = dict(params), dict(execution)
        if params.get("random_state", random_state) != random_state:
            raise ValueError(f"{name}: model.random_state must equal shared random_state={random_state} for identical splits")
        params["random_state"] = random_state
        framework = spec.get("framework")
        if framework == "single" and execution:
            raise ValueError(f"{name}: distributed settings require framework='dask'")
        if framework == "dask":
            extra = set(execution) - {"n_workers", "threads_per_worker", "memory_limit", "scheduler_address"}
            if extra:
                raise ValueError(f"{name}: unknown distributed settings: {sorted(extra)}")
            execution.setdefault("n_workers", 2)
            threads = execution.setdefault("threads_per_worker", params.get("n_jobs", DistributedXGBClassifier.DEFAULTS["n_jobs"]))
            params.setdefault("n_jobs", threads)
            for key in ("n_workers", "threads_per_worker"):
                value = execution[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name}: {key} must be a positive integer")
            if params["n_jobs"] != threads:
                raise ValueError(f"{name}: model.n_jobs must equal distributed.threads_per_worker")
            execution.setdefault("memory_limit", "auto")
        # Validate every entry before training the first one. Python configurations
        # have the same serializable parameter contract as the JSON input format.
        json.dumps({"model": params, "distributed": execution}, default=_json_value, allow_nan=False)
        operator = DistributedXGBClassifier(framework, params=params, reduction=reduction)
        prepared.append((name, operator, execution))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {"dataset": {}, "experiment": experiment,
               "split": {"random_state": random_state, "test_size": test_size},
               "reduction": reduction, "runs": []}
    reference_train = reference_validation = None
    labels = np.asarray(y)
    for name, operator, execution in prepared:
        start = time.perf_counter()
        with training_diagnostics(output / "training.log"), threadpool_limits(limits=1, user_api="blas"):
            operator.train(X, y, test_size=test_size,
                           distributed_config=execution if operator.framework == "dask" else None)
        elapsed = time.perf_counter() - start
        train, validation = operator.train_indices_, operator.validation_indices_
        if reference_train is None:
            reference_train, reference_validation = train.copy(), validation.copy()
            results["split"].update(train_indices=train.tolist(), validation_indices=validation.tolist())
            results["dataset"] = {**(dataset_info or {"source": "provided_arrays"}),
                                  "samples": len(labels), "features": operator.n_features_in_}
        elif not (np.array_equal(reference_train, train) and np.array_equal(reference_validation, validation)):
            raise RuntimeError(f"{name}: training/validation split differs from the first configuration")
        train_X = X.iloc[train] if isinstance(X, pd.DataFrame) else np.asarray(X)[train]
        # Both metric calculations and all preprocessing stay in the operator.
        train_metrics = operator.evaluate(train_X, labels[train])
        validation_metrics = operator.evaluate()
        booster = operator.model_.get_booster()
        trees = booster.trees_to_dataframe()
        best_iteration = booster.attr("best_iteration")
        workers = execution["n_workers"] if operator.framework == "dask" else 1
        results["runs"].append({
            "name": name, "framework": operator.framework,
            "effective_params": dict(operator.params), "distributed_config": execution,
            "native_booster_config": getattr(operator.model_, "training_booster_config_",
                                             json.loads(booster.save_config())),
            "workers": workers, "total_threads": workers * operator.params["n_jobs"],
            "worker_evidence": operator.worker_evidence_, "split_verified": True,
            "train_metrics": train_metrics, "validation_metrics": validation_metrics,
            "timings": dict(operator.timings_), "total_train_seconds": elapsed,
            "complexity": {"features": len(operator.output_feature_names_),
                           "rounds": booster.num_boosted_rounds(), "trees": int(trees["Tree"].nunique()),
                           "nodes": len(trees), "leaves": int((trees["Feature"] == "Leaf").sum())},
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
        })

    (output / "comparison.json").write_text(json.dumps(results, indent=2, default=_json_value) + "\n")
    frame = pd.json_normalize(results["runs"])
    for column in frame:
        frame[column] = frame[column].map(lambda v: json.dumps(v, default=_json_value)
                                        if isinstance(v, (dict, list, tuple, np.ndarray)) else v)
    frame.to_csv(output / "comparison.csv", index=False)
    report = format_model_comparison(results)
    (output / "terminal_report.txt").write_text(report + "\n")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        positions = np.arange(len(results["runs"]))
        for ax, (key, label) in zip(axes, [("auc_roc", "ROC-AUC"), ("accuracy", "Accuracy"), ("f1", "F1")]):
            for offset, source, legend in [(-.18, "train_metrics", "Train"), (.18, "validation_metrics", "Validation")]:
                ax.bar(positions + offset, [run[source][key] for run in results["runs"]], .36, label=legend)
            ax.set_xticks(positions, [run["name"] for run in results["runs"]], rotation=25, ha="right")
            ax.set(title=label, ylim=(0, 1), ylabel="Score")
        axes[0].legend()
        fig.tight_layout(); fig.savefig(output / "train_validation.png", dpi=160); plt.close(fig)
    print(report)
    print(f"\nResults: {output / 'comparison.csv'} | {output / 'comparison.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    add_experiment_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("results/model_comparison"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        models, raw_experiment, raw_reduction, dataset = load_model_configuration(args.config, include_dataset=True)
        experiment = normalize_experiment(raw_experiment, workflow="models", overrides=experiment_overrides(args))
        # Preserve the old --seed behavior while allowing an independent model seed.
        if args.seed is not None and args.model_seed is None:
            experiment["model_seed"] = args.seed
        reduction = resolve_reduction(raw_reduction, method=args.feature_reduction,
                                      threshold=args.threshold, n_components=args.n_components)
        if args.seed is not None or args.model_seed is not None:
            if any(not isinstance(spec, dict) or not isinstance(spec.get("model", {}), dict) for spec in models):
                raise ValueError("Each models entry must be an object with a model dictionary")
            models = [{**spec, "model": {**spec.get("model", {}), "random_state": experiment["model_seed"]}}
                      for spec in models]
        X, y = load_dataset(experiment, dataset)
        compare_models(X, y, {"models": models, "experiment": experiment},
                       reduction=reduction, output_dir=args.output, plot=args.plot,
                       dataset_info=dataset_metadata(X, y, experiment, dataset))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
