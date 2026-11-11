# XGBoost 分类基础算子（题目一）

统一接口封装单机 `XGBClassifier` 与真正进行 collective 训练的 `DaskXGBClassifier`。
原始题库保持不变；设计见 [design.md](design.md)，实际 AI 协作记录见 [ai_collaboration.md](ai_collaboration.md)。

## 安装与运行

在项目根目录执行。验证环境为 macOS Apple Silicon CPU、Python 3.10.4，无 CUDA。
已有 `.venv` 已复用并安装依赖；新环境建议 Python 3.10–3.12。

```bash
python3 -m venv .venv                 # 仅新环境需要
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python compare.py --samples 200000 --workers 2 --total-threads 4
```

完整依赖版本另见 `requirements.lock.txt`，同类环境可用 `pip install -r requirements.lock.txt`。
macOS 若 XGBoost 报缺少 `libomp.dylib`，使用 `brew install libomp` 安装 OpenMP 运行库。
启动多进程的代码必须放在 `if __name__ == '__main__':` 内；请用脚本运行，避免 notebook/spawn 问题。

## 使用

```python
from sklearn.datasets import make_classification
from distributed_xgb import DistributedXGBClassifier

def main():
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    op = DistributedXGBClassifier(
        framework="dask",                 # 或 single
        params={"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100,
                "max_leaves": 64},         # 额外有效 XGBoost 参数
        n_jobs=2,
    )
    op.feature_reduction(method="variance_threshold", threshold=0.0)
    op.train(X, y, test_size=0.2,
             distributed_config={"n_workers": 2, "threads_per_worker": 2})
    print(op.evaluate(plot_dir="my_results"))
    op.save_model("my_results")
    loaded = DistributedXGBClassifier.load_model("my_results")
    print(loaded.predict(X[:5]))           # 不需要活动集群；输入原始特征
    print(loaded.evaluate(X, y))           # 加载后需显式传评估数据

if __name__ == "__main__":
    main()
```

`single` 模式的 `train` 不传 `distributed_config`。`set_params(**params)` 验证后更新配置，返回自身，且使旧模型失效。
构造函数接受 `params` 字典和关键字参数，重复时关键字参数优先。

### 参数与输入约定

| 参数 | 默认值 | 约束或说明 |
| --- | --- | --- |
| learning_rate | 0.1 | (0,1] |
| max_depth / n_estimators | 6 / 100 | 正整数 |
| subsample / colsample_bytree | 1 / 1 | (0,1] |
| min_child_weight | 1 | 非负有限数 |
| gamma / reg_alpha / reg_lambda | 0 / 0 / 1 | 非负有限数 |
| objective | binary:logistic | 仅二分类概率目标 |
| tree_method / max_bin | hist / 512 | hist、approx；单机额外支持 exact；max_bin ≥2 |
| random_state | 42 | [0,2³²) 内整数，同时控制分层切分与模型种子 |
| n_jobs | 2 | 单机总线程数；分布式每 Worker 线程数，必须为正整数 |
| device | cpu | 支持 Apple Silicon，无 GPU 路径 |
| eval_metric | logloss | 可通过字典设置其他有效指标 |

额外参数交给当前安装版本的 XGBoost 原生引擎做小数据探测校验，未知或未使用参数报错。
支持树 booster `gbtree` 和 `dart`。为避免覆盖校验值，拒绝 `eta/nthread/seed/lambda/alpha` 别名，使用表中规范名称。
不支持多分类、GPU、原生类别特征、自定义 objective 或自定义 callable 指标；类别特征需先编码。
X 为二维 NumPy 数组或数值 pandas DataFrame；y 为两个类别的一维标签，支持字符串。
阳性类别为 `LabelEncoder.classes_[1]`，阈值 0.5。DataFrame 预测检查列名与顺序；NumPy 使用位置语义。
未启用降维时 NaN 由 XGBoost 处理，inf 始终拒绝；启用降维需先处理所有缺失值。

### 降维与防泄漏

默认关闭；`feature_reduction(method=None)` 关闭。该方法**仅配置**，返回算子，不在切分前拟合，
即使传入兼容旧接口的 `X, y` 也不使用它们拟合。配置变化会使旧模型失效。

- `method="variance_threshold", threshold=0.0`：删除方差 ≤ threshold 的特征，threshold 是绝对方差而非保留比例。
- `method="pca", n_components=0.95`：累计解释方差至少 95%；整数表示保留维数，浮点数必须严格在 (0,1) 内。
- PCA 使用完整 SVD，不自动标准化；量纲差异大时应自行设计仅在训练集拟合的标准化流程。

`train` 先固定种子分层切分原始数据，仅用训练部分拟合降维器，再转换验证集。
`predict/predict_proba/evaluate` 接受原始特征，自动应用同一个降维器，避免重复变换。
PCA 的重要性图展示主成分重要性，不解释为原始特征的重要性。

### Dask 资源与独立启动

默认创建本地进程集群，成功或异常都会在 finally 关闭 Client 和自建 Cluster。
`distributed_config` 支持 `n_workers`、`threads_per_worker`、`memory_limit`、`scheduler_address`。
本地模式默认 2 Worker，线程默认等于 n_jobs，内存限制默认 auto。

```bash
# 终端一：持续运行；Ctrl-C 关闭
.venv/bin/python start_cluster.py --workers 2 --threads-per-worker 2 --port 8786
# 终端二：仅连接已有集群，结束后只关闭自己的 Client
.venv/bin/python compare.py --scheduler-address tcp://127.0.0.1:8786
```

外部集群须与配置的 Worker 数量和线程数一致，数据在各 Worker 间按行划分。
可配置多机 scheduler，但本项目实测范围是同机多进程 CPU。无故障自动重试；Worker 故障会传播训练异常。
每个 Worker 内部的 `after_training` 回调记录真实 collective rank、world_size、PID 和地址，并校验每个 rank 都出现。

## 复现实验与产物

数据源是 [scikit-learn make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)：
200,000 样本，30 个生成特征（20 informative、5 redundant），再加一个常数特征验证方差筛选。
`class_sep=1`、`flip_y=.01`、种子 42，80%/20% 分层切分，方差筛选后 30 维。
两种模式使用相同数据、索引、降维配置和模型超参数。CPU 总预算都为 4 个 XGBoost 线程：单机 4；分布式 2 Worker × 2。
因此每个估计器的 n_jobs 分别为 4 和 2，这是匹配总资源的必要区别；BLAS 单线程。

`results/` 中包含真实执行生成的文件：

- `comparison.json`：指标、分段耗时、平台与版本、数据哈希、差异统计、各训练 rank 的证据。
- `comparison.png`：两种模式的时间/指标/预测差异对比图，作为题目要求的对比截图。
- `single/`、`dask/`：各自的 `model.joblib`、`metrics.json`、混淆矩阵、ROC、特征重要性 PNG。
- `validation_predictions.npz`：相同验证集索引、标签和双方概率，方便复查。
- `experiment.log`、`tests.log`、`tests.xml`：实际运行日志与测试报告。
- `report.md`：实际结果与验收偏差说明。
- `startup_smoke.log`：独立启动脚本与外部集群训练验证；算子结束后外部集群仍存活，脚本 Ctrl-C 正常退出。
- `reload_fresh_process.log`：全新 Python 进程、无 Dask Client 下，重新生成数据并逐点核对双方各 40,000 个预测概率。

提交包 `XGBoost分类基础算子_未署名_2026-11-10.zip` 包含源码、文档、测试和所有结果；提交前按本人姓名重命名。

拟合计时包含估计器构造 DMatrix/quantile sketch/collective 训练；集群启动、显式数据分发分别记录。
总 train 耗时还含预处理、模型转换和集群关闭。报告记录最后一次顺序执行，没有做多轮统计性能基准。
不做阈值后处理或共享模型来强制两种模型一致：分布式直方图切点可能不同，报告实际标签和概率差异。
题库“仅微小浮点误差”的严格要求可能因此无法满足，详见实际报告；按照用户补充要求透明呈现。

joblib 会执行反序列化代码，仅加载可信产物。保存包保留模型、标签编码器、列信息和降维器，
不保存验证集缓存或训练索引；跨大版本部署建议重新验证，原生 UBJ 模型也可由 `op.model_.save_model(...)` 导出。

官方参考：[XGBoost Dask 教程](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html)、
[参数说明](https://xgboost.readthedocs.io/en/stable/parameter.html)。
