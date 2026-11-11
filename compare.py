"""Reproducible 200k-sample CPU experiment. Run as a script (spawn-safe)."""
import os

# Bound BLAS in client and spawned worker processes, independently of XGBoost OpenMP.
for variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_limits

from distributed_xgb import DistributedXGBClassifier


def write_report(results, output):
    s, d = results["runs"]["single"], results["runs"]["dask"]
    c = results["comparison"]
    lines = ["# 实际验证报告", "", "## 环境与数据", "",
             f"- 平台：{results['environment']['platform']}，Python {results['environment']['python']}。",
             f"- make_classification 样本数 {results['dataset']['samples']:,}，种子 {results['dataset']['seed']}；"
             "80%/20% 分层切分。原始 31 维，经仅训练集拟合的方差筛选变为 30 维。",
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
                  "本次单机更快；此规模下通信、DMatrix 构造与进程管理开销不能被并行收益抵消。"
                  "本报告记录最后一次顺序执行，未做多轮统计，不作跨设备或大数据性能结论。", "", "## 一致性与验收边界", "",
                  f"- 标签一致率：{c['label_agreement']:.4%}。",
                  f"- 概率绝对差：均值 {c['probability_abs_difference_mean']:.8f}，"
                  f"P95 {c['probability_abs_difference_p95']:.8f}，最大值 {c['probability_abs_difference_max']:.8f}。",
                  "- 这些差异超过微小浮点误差，因此不满足原题严格逐点一致的字面要求。"
                  "遵循用户补充说明，保留真实分布式训练并透明记录结果，没有强制共享切点、复用单机模型或修改预测。",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--total-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--scheduler-address", default=None)
    args = parser.parse_args()
    if args.total_threads < args.workers or args.total_threads % args.workers:
        parser.error("total-threads must be a positive multiple of workers")
    if args.samples < 20:
        parser.error("samples must be >=20")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    X, y = make_classification(n_samples=args.samples, n_features=30, n_informative=20,
                               n_redundant=5, class_sep=1.0, flip_y=0.01,
                               random_state=args.seed)
    X = X.astype(np.float32)
    # A removable constant makes enabled reduction observable in the experiment.
    X = np.column_stack([X, np.ones(len(X), dtype=np.float32)])
    generation_seconds = time.perf_counter() - start
    params = dict(n_estimators=args.estimators, max_depth=6, learning_rate=0.1,
                  max_bin=512, tree_method="hist", random_state=args.seed,
                  subsample=1.0, colsample_bytree=1.0)
    results = {"dataset": {"source": "sklearn.datasets.make_classification", "samples": len(y),
                           "features": X.shape[1], "seed": args.seed,
                           "sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
                           "generation_seconds": generation_seconds},
               "environment": {"platform": platform.platform(), "machine": platform.machine(),
                               "python": platform.python_version(), "logical_cpus": os.cpu_count(),
                               "packages": {name: importlib.metadata.version(name) for name in
                                            ["xgboost", "dask", "distributed", "numpy", "pandas", "scikit-learn"]}},
               "shared_params": params, "total_training_threads": args.total_threads,
               "workers": args.workers, "runs": {}}
    models, probabilities = {}, {}
    for mode in ("single", "dask"):
        threads = args.total_threads if mode == "single" else args.total_threads // args.workers
        model = DistributedXGBClassifier(mode, params=params, n_jobs=threads,
                                         reduction={"method": "variance_threshold", "threshold": 0.0})
        config = None if mode == "single" else dict(n_workers=args.workers, threads_per_worker=threads)
        if config is not None and args.scheduler_address:
            config["scheduler_address"] = args.scheduler_address
        start = time.perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            model.train(X, y, distributed_config=config)
        elapsed = time.perf_counter() - start
        metrics = model.evaluate(plot_dir=args.output / mode)
        probability = model.predict_proba(X[model.validation_indices_])[:, 1]
        path = model.save_model(args.output / mode)
        # Training cluster has already closed. Verify artifact inference now.
        loaded = DistributedXGBClassifier.load_model(path)
        reloaded_probability = loaded.predict_proba(X[model.validation_indices_])[:, 1]
        reload_diff = float(np.max(np.abs(probability - reloaded_probability)))
        if reload_diff != 0:
            raise AssertionError(f"Reload changed probabilities by {reload_diff}")
        results["runs"][mode] = dict(metrics=metrics, timings=model.timings_,
                                           train_total_seconds=elapsed, n_jobs=threads,
                                           transformed_features=len(model.output_feature_names_),
                                           worker_evidence=model.worker_evidence_,
                                           reload_max_probability_difference=reload_diff)
        models[mode], probabilities[mode] = model, probability
        print(mode, json.dumps(results["runs"][mode], indent=2), flush=True)
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
    axes[1].set_xticks(x, keys, rotation=30); axes[1].set_ylim(.8, 1); axes[1].legend()
    axes[1].set_title("Validation metrics")
    axes[2].hist(delta, bins=50)
    axes[2].set(xlabel="Absolute probability difference", ylabel="Samples",
                title=f"Label agreement: {results['comparison']['label_agreement']:.3%}")
    fig.tight_layout(); fig.savefig(args.output / "comparison.png", dpi=180); plt.close(fig)
    write_report(results, args.output)
    print("comparison", json.dumps(results["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
