"""Reproducible 200k-sample CPU experiment. Run as a script (spawn-safe)."""
import os

# Bound BLAS in client and spawned worker processes, independently of XGBoost OpenMP.
for variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import platform
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from distributed_xgb import DistributedXGBClassifier
from experiment_config import (add_experiment_arguments, dataset_metadata, experiment_overrides,
                               load_dataset, normalize_dataset, normalize_experiment, resolve_reduction)


def parse_configuration(argv=None):
    """Defaults < JSON < explicit CLI. Thread counts are derived, never tuned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON with model, distributed, experiment, dataset, feature_reduction sections")
    add_experiment_arguments(parser)
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--n-workers", "--workers", dest="n_workers", type=int)
    parser.add_argument("--total-threads", type=int)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--scheduler-address")
    args = parser.parse_args(argv)
    try:
        supplied = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(supplied, dict) or set(supplied) - {"model", "distributed", "experiment", "dataset", "feature_reduction"}:
            raise ValueError("config accepts model, distributed, experiment, dataset, feature_reduction sections")
        for name, value in supplied.items():
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a JSON object")
        dataset = normalize_dataset(supplied.get("dataset"))
        experiment = normalize_experiment(supplied.get("experiment"), overrides=experiment_overrides(args))
        distributed = {"n_workers": 2, "total_threads": 4, "memory_limit": "auto"}
        for name, defaults, allowed in [
            ("distributed", distributed, set(distributed) | {"threads_per_worker", "scheduler_address"}),
        ]:
            extra = set(supplied.get(name, {})) - allowed
            if extra:
                raise ValueError(f"Unknown {name} keys: {sorted(extra)}")
            defaults.update(supplied.get(name, {}))
        for key in ("n_workers", "total_threads", "scheduler_address"):
            if getattr(args, key) is not None:
                distributed[key] = getattr(args, key)
        for key, value in {"samples": experiment["samples"], "n_workers": distributed["n_workers"],
                           "total_threads": distributed["total_threads"]}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if experiment["samples"] < 20:
            raise ValueError("samples must be >=20")
        seed = experiment["model_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        size = experiment["test_size"]
        if isinstance(size, bool) or not isinstance(size, float) or not 0 < size < 1:
            raise ValueError("test_size must be a float in (0,1)")
        total, workers = distributed["total_threads"], distributed["n_workers"]
        if total % workers:
            raise ValueError("total-threads must be a positive multiple of n-workers")
        threads = distributed.get("threads_per_worker", total // workers)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads != total // workers:
            raise ValueError("threads_per_worker must equal total_threads / n_workers; remove it to derive automatically")
        distributed["threads_per_worker"] = threads
        if "n_jobs" in supplied.get("model", {}):
            raise ValueError("Set distributed.total_threads, not model.n_jobs; n_jobs is derived per training mode")
        model = {k: v for k, v in DistributedXGBClassifier.DEFAULTS.items() if k != "n_jobs"}
        model.update(random_state=seed)
        model.update(supplied.get("model", {}))
        if args.seed is not None:
            model["random_state"] = args.seed
        if args.model_seed is not None:
            model["random_state"] = args.model_seed
        if args.estimators is not None:
            model["n_estimators"] = args.estimators
        experiment["model_seed"] = model["random_state"]
        # Fail before creating data/cluster, using the operator's existing validation.
        reduction = resolve_reduction(supplied.get("feature_reduction"),
                                      default={"method": "variance_threshold", "threshold": 0.0},
                                      method=args.feature_reduction, threshold=args.threshold,
                                      n_components=args.n_components)
        DistributedXGBClassifier("dask", params=model, n_jobs=threads, reduction=reduction)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args, {"model": model, "distributed": distributed, "experiment": experiment,
                  "feature_reduction": reduction, "dataset": dataset}


@contextmanager
def training_diagnostics(path):
    """Capture native/worker console chatter without changing XGBoost verbosity.

    Local spawned workers inherit these descriptors. Warnings remain in the log;
    exceptions propagate after the terminal descriptors have been restored.
    """
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with path.open("a") as log:
        try:
            for fd in (1, 2):
                os.dup2(log.fileno(), fd)
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            for fd, previous in zip((1, 2), saved):
                os.dup2(previous, fd)
                os.close(previous)


def format_comparison(results):
    """Format only values in the saved result object; never recompute metrics."""
    s, d = (results["runs"][mode] for mode in ("single", "dask"))
    dataset, comparison = results["dataset"], results["comparison"]
    params = results["shared_params"]
    workers = results["workers"]
    verified = len({e["rank"] for e in d["worker_evidence"] if e["world_size"] == workers})
    lines = ["Distributed XGBoost comparison", "=" * 72,
             f"Dataset: {dataset['samples']:,} samples | {dataset['features']} input features | "
             f"{s['transformed_features']} after reduction",
             f"Split: {dataset['train_samples']:,} train / {dataset['validation_samples']:,} validation "
             f"(test_size={dataset['test_size']:.2f})",
             f"Seeds: data={dataset.get('seed', 'CSV')} | model/split={params['random_state']}",
             "", "XGBoost parameters (shared):"]
    lines.extend(textwrap.wrap(", ".join(f"{k}={v}" for k, v in params.items()), width=105,
                               initial_indent="  ", subsequent_indent="  ", break_long_words=False))
    lines.extend([f"CPU single: 1 process x {s['n_jobs']} threads = {results['total_training_threads']} threads",
                  f"CPU Dask:   {workers} workers x {d['n_jobs']} threads = {results['total_training_threads']} threads",
                  f"Dask n_jobs = threads_per_worker = {d['distributed_config']['threads_per_worker']}",
                  f"Workers verified: {verified}/{workers}", "",
                  f"{'Metric':<34}{'Single':>16}{'Dask':>16}", "-" * 66])
    for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("auc_roc", "ROC-AUC")]:
        lines.append(f"{label:<34}{s['metrics'][key]:>16.6f}{d['metrics'][key]:>16.6f}")
    lines.extend(["", f"{'Time (seconds)':<34}{'Single':>16}{'Dask':>16}", "-" * 66])
    for key, label in [("preprocessing_seconds", "Preprocessing"), ("cluster_startup_seconds", "Cluster startup"),
                       ("data_distribution_seconds", "Data distribution"), ("fit_seconds", "Fit")]:
        lines.append(f"{label:<34}{s['timings'][key]:>16.3f}{d['timings'][key]:>16.3f}")
    lines.extend([f"{'Total train (includes cleanup)':<34}{s['train_total_seconds']:>16.3f}{d['train_total_seconds']:>16.3f}",
                  "", f"Prediction-label agreement: {comparison['label_agreement']:.4%}",
                  f"Absolute probability difference: mean={comparison['probability_abs_difference_mean']:.8f} | "
                  f"P95={comparison['probability_abs_difference_p95']:.8f} | "
                  f"maximum={comparison['probability_abs_difference_max']:.8f}",
                  f"Training speedup (single fit time / distributed fit time): "
                  f"{comparison['fit_speedup_single_over_dask']:.3f}x",
                  "Model reload max probability difference: "
                  f"single={s['reload_max_probability_difference']:.8f} | "
                  f"Dask={d['reload_max_probability_difference']:.8f}"])
    return "\n".join(lines)


def write_report(results, output):
    s, d = results["runs"]["single"], results["runs"]["dask"]
    c = results["comparison"]
    lines = ["# 实际验证报告", "", "## 环境与数据", "",
             f"- 平台：{results['environment']['platform']}，Python {results['environment']['python']}。",
             f"- {results['dataset']['source']} 样本数 {results['dataset']['samples']:,}；"
             f"训练/验证 = {results['dataset']['train_samples']:,}/{results['dataset']['validation_samples']:,}。"
             f"原始 {results['dataset']['features']} 维，经配置的预处理后为 {s['transformed_features']} 维。",
             f"- 总 XGBoost 线程预算 {results['total_training_threads']}：单机 n_jobs={s['n_jobs']}，"
             f"Dask {results['workers']} Worker × n_jobs={d['n_jobs']}。",
             "- 模型参数和版本详情见 comparison.json；双方数据和验证索引相同。", "",
             "## 分类指标", "", "| 指标 | 单机 | Dask |", "| --- | ---: | ---: |"]
    for k in s["metrics"]:
        lines.append(f"| {k} | {s['metrics'][k]:.8f} | {d['metrics'][k]:.8f} |")
    lines.extend(["", "## 耗时（秒）", "", "| 阶段 | 单机 | Dask |", "| --- | ---: | ---: |"])
    for k in s["timings"]:
        lines.append(f"| {k} | {s['timings'][k]:.6f} | {d['timings'][k]:.6f} |")
    lines.extend([f"| train 总耗时（含回收） | {s['train_total_seconds']:.6f} | {d['train_total_seconds']:.6f} |",
                  "", f"单机 fit / Dask fit = {c['fit_speedup_single_over_dask']:.3f}。"
                  f"本次{'Dask' if c['fit_speedup_single_over_dask'] > 1 else '单机'} fit 更快。"
                  "本报告记录最后一次顺序执行，未做多轮统计，不作跨设备或大数据性能结论。", "", "## 一致性与验收边界", "",
                  f"- 标签一致率：{c['label_agreement']:.4%}。",
                  f"- 概率绝对差：均值 {c['probability_abs_difference_mean']:.8f}，"
                  f"P95 {c['probability_abs_difference_p95']:.8f}，最大值 {c['probability_abs_difference_max']:.8f}。",
                  "- 以上是实际预测差异；本项目不保证分布式与单机逐点一致。"
                  "保留真实分布式训练并透明记录结果，没有强制共享切点、复用单机模型或修改预测。",
                  "- 两份保存包均在训练集群关闭后重新加载，概率最大差为 0。", "",
                  "## 真实多 Worker 训练证据", "", "| rank | world_size | PID | 地址 |",
                  "| ---: | ---: | ---: | --- |"])
    for e in sorted(d["worker_evidence"], key=lambda e: e["rank"]):
        lines.append(f"| {e['rank']} | {e['world_size']} | {e['pid']} | {e['address']} |")
    lines.extend(["", "证据来自 Worker 内 XGBoost collective 回调，而非仅统计集群进程数量。", "",
                  "## 图与复查", "", "![单机/分布式对比](comparison.png)", "",
                  "混淆矩阵、ROC 和特征重要性见 single/、dask/；验证索引、标签和概率见 validation_predictions.npz。",
                  "自动化测试日志和 JUnit 结果见 tests.log、tests.xml。", "",
                  "## 限制", "", "输入、切分、降维在驱动内存完成；只实测同机多进程 CPU 二分类。"
                  "没有大规模集群容错或网络压力测试。模型包使用 joblib，应仅加载可信来源并匹配依赖版本。", ""])
    (output / "report.md").write_text("\n".join(lines))


def main():
    args, configuration = parse_configuration()
    params, execution, experiment = (configuration[key] for key in ("model", "distributed", "experiment"))
    workers, total_threads = execution["n_workers"], execution["total_threads"]
    start = time.perf_counter()
    try:
        X, y = load_dataset(experiment, configuration["dataset"], float32=True)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Dataset error: {exc}") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    generation_seconds = time.perf_counter() - start
    results = {"dataset": {**dataset_metadata(X, y, experiment, configuration["dataset"]), "generation_seconds": generation_seconds},
               "environment": {"platform": platform.platform(), "machine": platform.machine(),
                               "python": platform.python_version(), "logical_cpus": os.cpu_count(),
                               "packages": {name: importlib.metadata.version(name) for name in
                                            ["xgboost", "dask", "distributed", "numpy", "pandas", "scikit-learn"]}},
               "shared_params": params, "total_training_threads": total_threads,
               "configuration": configuration, "workers": workers, "runs": {}}
    models, probabilities = {}, {}
    for mode in ("single", "dask"):
        threads = total_threads if mode == "single" else execution["threads_per_worker"]
        model = DistributedXGBClassifier(mode, params=params, n_jobs=threads,
                                         reduction=configuration["feature_reduction"])
        config = None if mode == "single" else {k: v for k, v in execution.items() if k != "total_threads"}
        start = time.perf_counter()
        with training_diagnostics(args.output / "training.log"), threadpool_limits(limits=1, user_api="blas"):
            model.train(X, y, test_size=experiment["test_size"], distributed_config=config)
        elapsed = time.perf_counter() - start
        metrics = model.evaluate(plot_dir=args.output / mode)
        validation_X = X.iloc[model.validation_indices_] if hasattr(X, "iloc") else X[model.validation_indices_]
        probability = model.predict_proba(validation_X)[:, 1]
        path = model.save_model(args.output / mode)
        # Training cluster has already closed. Verify artifact inference now.
        loaded = DistributedXGBClassifier.load_model(path)
        reloaded_probability = loaded.predict_proba(validation_X)[:, 1]
        reload_diff = float(np.max(np.abs(probability - reloaded_probability)))
        if reload_diff != 0:
            raise AssertionError(f"Reload changed probabilities by {reload_diff}")
        results["runs"][mode] = dict(metrics=metrics, timings=model.timings_,
                                           train_total_seconds=elapsed, n_jobs=threads,
                                           effective_params=dict(model.params), distributed_config=config,
                                           native_booster_config=getattr(model.model_, "training_booster_config_",
                                                                       json.loads(model.model_.get_booster().save_config())),
                                           transformed_features=len(model.output_feature_names_),
                                           worker_evidence=model.worker_evidence_,
                                           reload_max_probability_difference=reload_diff,
                                           reload_verified=reload_diff == 0)
        models[mode], probabilities[mode] = model, probability
    results["dataset"].update(train_samples=len(models["single"].train_indices_),
                              validation_samples=len(models["single"].validation_indices_))
    if not np.array_equal(models["single"].validation_indices_, models["dask"].validation_indices_):
        raise AssertionError("Validation splits differ")
    delta = np.abs(probabilities["single"] - probabilities["dask"])
    results["comparison"] = {
        "identical_split": True,
        "label_agreement": float(np.mean((probabilities["single"] >= .5) == (probabilities["dask"] >= .5))),
        "probability_abs_difference_mean": float(delta.mean()),
        "probability_abs_difference_max": float(delta.max()),
        "probability_abs_difference_p95": float(np.quantile(delta, .95)),
        "fit_speedup_single_over_dask": results["runs"]["single"]["timings"]["fit_seconds"] /
                                        results["runs"]["dask"]["timings"]["fit_seconds"],
        "metric_dask_minus_single": {k: results["runs"]["dask"]["metrics"][k] -
                                      results["runs"]["single"]["metrics"][k] for k in metrics}}
    np.savez_compressed(args.output / "validation_predictions.npz", indices=models["single"].validation_indices_,
                        y=y[models["single"].validation_indices_], single=probabilities["single"],
                        dask=probabilities["dask"])
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["single", "dask"]
    axes[0].bar(labels, [results["runs"][m]["timings"]["fit_seconds"] for m in labels])
    axes[0].set(title="Model fitting (equal total threads)", ylabel="Seconds")
    keys = list(metrics)
    x = np.arange(len(keys))
    for offset, m in zip([-.18, .18], labels):
        axes[1].bar(x + offset, [results["runs"][m]["metrics"][k] for k in keys], .36, label=m)
    lower = min(.8, max(0.0, min(results["runs"][m]["metrics"][k] for m in labels for k in keys) - .05))
    axes[1].set_xticks(x, keys, rotation=30); axes[1].set_ylim(lower, 1); axes[1].legend()
    axes[1].set_title("Validation metrics")
    axes[2].hist(delta, bins=50)
    axes[2].set(xlabel="Absolute probability difference", ylabel="Samples",
                title=f"Label agreement: {results['comparison']['label_agreement']:.3%}")
    fig.tight_layout(); fig.savefig(args.output / "comparison.png", dpi=180); plt.close(fig)
    write_report(results, args.output)
    terminal_report = format_comparison(results)
    (args.output / "terminal_report.txt").write_text(terminal_report + "\n")
    print(terminal_report, flush=True)
    print(f"\nDetailed results: {args.output / 'comparison.json'}")
    print(f"Training diagnostics: {args.output / 'training.log'}")


if __name__ == "__main__":
    main()
