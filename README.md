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
.venv/bin/python compare.py --config config.example.json --n-workers 2 --total-threads 4
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

### JSON 配置与覆盖顺序

修改 [config.example.json](config.example.json) 中的 `model` 来调整超参数，无自动调参。
配置按职责分五部分：

- `model`：XGBoost 参数，例如 `learning_rate`、`max_depth`、`n_estimators`、`max_bin`；
  可直接添加当前安装版本支持的参数，如 `max_delta_step`、`scale_pos_weight`、`max_leaves`、
  `grow_policy`、`colsample_bynode`。算子不使用固定参数名白名单，仍通过原生引擎校验额外参数。
- `distributed`：执行资源，包括 `n_workers`、`total_threads`、`memory_limit`、可选 `scheduler_address`。
  `threads_per_worker` 通常省略，自动计算为 `total_threads / n_workers`；如显式填写，必须与计算值一致。
- `experiment`：样本数、特征组成、类别分布、噪声、数据种子和验证比例，详见下方数据参数表。
- `dataset`：选择 `synthetic`（默认）或外部 `csv`，详见下方示例。
- `feature_reduction`：是否启用、`variance_threshold` 或 `pca` 及方法参数。两种脚本均支持此配置。

```bash
# 使用默认配置（无需 JSON）
.venv/bin/python compare.py --n-workers 2 --total-threads 4
# JSON 配置 + CLI 覆盖；--workers 是 --n-workers 的兼容别名
.venv/bin/python compare.py --config config.example.json --n-workers 4 --total-threads 4 --estimators 100
# 在独立目录验证，保留已有数值结果、模型和图表
.venv/bin/python compare.py --config config.example.json --samples 2000 --estimators 20 --output results/my_new_comparison
```

覆盖顺序为**内置默认值 → JSON → 显式 CLI**；未指定的 CLI 参数不会覆盖 JSON。
`--estimators` 覆盖 `model.n_estimators`；`--seed` 同时覆盖数据种子和 `model.random_state`。
未指定种子 CLI 时，`compare.py` 的模型/切分种子依次取 `model.random_state`、`experiment.model_seed`、
`experiment.seed`。数据种子只取 experiment.seed（别名 random_state）。
`--model-seed` 显式指定模型/切分种子，并优先于同时提供的 `--seed`，因此可使用数据种子 17、模型/切分种子 23。
旧 `--seed` 行为不变：单独使用时同时覆盖数据与模型/切分种子。
config.example.json 显式写有 model.random_state=42；若希望 experiment.model_seed 生效，请删除该项或同步修改，
也可用优先级最高的 `--model-seed`。这是模型种子覆盖规则，不会修改用于生成数据的 seed。
`--output` 指定输出目录，默认仍为 `results`，在相同目录运行会更新其中的同名结果。

比较脚本不接受 `model.n_jobs`，因为它按资源预算分别设置：单机 `n_jobs=total_threads`；
Dask `n_jobs=threads_per_worker=total_threads/n_workers`，总训练线程数相同。
`total_threads` 必须是正整数且能被 Worker 数整除；请将其设置在实际分配的 CPU 预算内。
BLAS 在驱动和自建 Worker 中保持单线程，避免与 XGBoost 线程叠加。
直接使用算子时仍可设置 `n_jobs`，Dask 下必须与 `distributed_config.threads_per_worker` 一致。
外部集群的实际 Worker 数和每 Worker 线程数也必须匹配，无法由客户端调整已运行的 Worker。
模型参数放在 `distributed`、执行参数放在 `model`、未知键、资源冲突都会明确报错，不会静默覆盖。

下面是 compare.py 可直接使用的独立 JSON 示例（保存为 `small.compare.json`）：

```json
{
  "model": {
    "max_depth": 3, "n_estimators": 10,
    "colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.6
  },
  "distributed": {"n_workers": 2, "total_threads": 2},
  "experiment": {
    "n_samples": 600, "n_features": 14, "n_informative": 7,
    "n_redundant": 3, "n_repeated": 1, "n_classes": 2,
    "class_sep": 1.5, "flip_y": 0.02, "weights": [0.65, 0.35],
    "seed": 17, "model_seed": 23, "test_size": 0.25, "append_constant": false
  },
  "feature_reduction": {"enabled": true, "method": "pca", "n_components": 5}
}
```

```bash
.venv/bin/python compare.py --config small.compare.json --output results/small_compare
```

### 终端报告与完整结果

终端按统一精度输出数据规模与切分、全部共享配置参数、双方 CPU 配置、五项指标表、五项耗时、
标签一致率、概率差（均值/P95/最大值）、模型加载校验，以及 `Workers verified: 4/4`。
指标用 6 位小数，概率差用 8 位，耗时统一为秒并保留 3 位；标签一致率保留 4 位百分数。
`Training speedup = single fit time / distributed fit time`，大于 1 表示分布式 fit 更快，
不包含集群启动时间。`Total train` 包含预处理、集群启动、分发、拟合、模型转换与资源关闭，
不包含数据生成、评估绘图或写模型文件。

详细信息仍在 `comparison.json`：包括每种模式的 `effective_params`（完整传入配置）、
`native_booster_config`（原生模型配置及其内部默认值）、`distributed_config`、全部指标/耗时、
Worker 地址/PID/rank/world_size、预测差异和 `reload_verified`。
Dask 的原生训练配置在转为本地预测模型前保留，避免加载模型后重置的内部默认值被误记为训练采样参数；
模型列表脚本也保存这份配置。这只是诊断元数据，不改变树或预测。
终端报告直接读取同一个结果对象，保存在 `terminal_report.txt`，不会修改 JSON 数值精度。
原生训练日志及警告写入 `training.log`，不在比较终端逐条展示 Worker 地址和 rank；训练异常仍向调用者传播。
连接外部集群时，其 Worker 日志仍由启动集群的终端或集群日志设施管理。

本次验证见 `results/configuration_check/`（40 项测试、20 万样本、4 Worker × 1 线程），
其中 `validation.json` 记录输出与 JSON 的逐项核对、历史文件完整性、原有训练/评估方法未变的检查。
历史 `results/` 下的原始结果、图表和模型未覆盖。数据、索引和单机预测逐点相同，
Dask 重跑概率有变化；`control_original/` 使用原版比较脚本重跑，同样出现变化。
因此原有分布式执行不保证跨启动逐点复现，不能将本次配置/显示更新视为消除了这一限制。
当前分块按排序后的 Worker 地址分配，而 collective rank 按 Worker 名称分配，
新集群的地址次序可改变各 rank 接收到的数据。为保留训练逻辑，本次没有修改这项分配策略。

### 参数与输入约定

以下均放在 `model` 中；默认值为算子显式默认或当前 XGBoost 原生默认。额外有效参数仍通过原接口传入。

| 参数 | 默认值 | 含义与实际作用 | 有效范围/限制 |
| --- | --- | --- | --- |
| learning_rate | 0.1 | 缩小每轮新树的贡献；较小通常需要更多树，拟合更缓慢 | 算子约束 (0,1] |
| max_depth | 6 | 每棵树最大深度；更深能表达复杂交互，也增加过拟合、计算和内存风险 | 算子要求正整数；不改变输入维数 |
| n_estimators | 100 | 最大 boosting 轮数；更多轮增加拟合能力和时间，early stopping 可提前结束 | 正整数 |
| subsample | 1.0 | 每轮使用的训练行比例；小于 1 引入行随机采样，可能抑制过拟合 | (0,1] |
| colsample_bytree | 1.0 | 每棵树抽取候选特征，整棵树从该子集中选分裂变量 | (0,1] |
| colsample_bylevel | 1.0 | 每个新深度层从当前树的特征子集中再次采样 | (0,1] |
| colsample_bynode | 1.0 | 每次评估分裂节点时，从当前层的特征子集中采样 | (0,1]；exact 方法不支持节点采样 |
| min_child_weight | 1.0 | 子节点所需最小 Hessian 权重和；增大可阻止证据不足的小叶子 | 非负有限数；不是最少样本数 |
| gamma | 0.0 | 新分裂需要达到的最小损失下降；更大使分裂更保守 | 非负有限数 |
| reg_alpha | 0.0 | 叶子权重的 L1 正则强度，鼓励更稀疏的叶子权重 | 非负有限数 |
| reg_lambda | 1.0 | 叶子权重的 L2 正则强度，抑制极端权重和过拟合 | 非负有限数 |
| objective | binary:logistic | 优化二分类 logistic 损失，输出阳性类别概率 | 本算子仅支持此目标 |
| tree_method | hist | 控制候选切点构造方式；hist 用直方图近似，approx 用近似分位数，exact 枚举精确切点 | hist/approx；单机额外支持 exact |
| max_bin | 512 | hist/approx 每列最大分箱数；更大提供更细的切点，也增加时间/内存 | 整数 ≥2；exact 不使用分箱 |
| random_state | 42 | 控制模型随机采样，同时被现有算子用作分层切分种子 | [0,2³²) 整数；同一比较必须共享 |
| n_jobs | 2 | 每个估计器的 XGBoost 线程数；单机为总线程，Dask 为每 Worker 线程 | 正整数；compare.py 根据总预算派生，不接受 model.n_jobs |
| device | cpu | 选择 CPU 计算，不需要 CUDA | 本项目仅 cpu |
| eval_metric | logloss | 拟合期间验证集监控指标；存在 early stopping 时决定何时停止，多个指标时使用最后一个 | 原生有效指标名/列表；不支持 callable |
| booster | gbtree | 使用普通提升树；dart 会在训练中随机丢弃部分树贡献，改变正则化方式 | gbtree/dart；不支持线性 booster |
| max_leaves | 0 | 限制每棵树叶子数，控制模型容量 | 非负整数，0 表示不限制；适用于 hist/approx |
| grow_policy | depthwise | 按深度扩展树；lossguide 优先扩展损失下降最大的节点 | depthwise/lossguide；适用于 hist/approx |
| max_delta_step | 0 | 限制每轮叶子权重更新幅度；极不平衡 logistic 问题可用正值使更新更保守 | 非负数；0 不限制 |
| scale_pos_weight | 1 | 提高阳性类别样本的损失权重，改变不平衡类别的优化取舍 | 非负数；不是生成数据的 weights |
| early_stopping_rounds | null | 验证指标连续若干轮未改善时停止；验证集同时用于选择最佳迭代 | null 或正整数；默认不启用 |
| monotone_constraints | 无 | 限制指定特征增大时预测的单调方向，施加领域先验 | 位置 tuple/list 或字符串，元素 -1/0/1；不支持名称字典 |
| importance_type | null（树模型默认 gain） | 选择返回的特征重要性统计，如平均增益、分裂次数或覆盖量 | null/weight/gain/cover/total_gain/total_cover |
| verbosity | 1 | 控制原生 XGBoost 诊断详细程度，不改变模型超参数搜索 | 整数 0–3；脚本将训练日志写入 training.log |

三种 `colsample_*` 是可叠加的随机候选特征采样，不改变 X 的列数，也不会永久删除特征。
例如 64 列依次按 0.5/0.5/0.5 采样时，每次分裂可用候选约为 8 列。
各阶段含义及累积行为见 [XGBoost 参数文档](https://xgboost.readthedocs.io/en/stable/parameter.html)。

### 合成数据参数（experiment）

数据生成使用 [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
下面“对比/模型列表”分别指 compare.py / compare_models.py；保留两者原有默认数据，不统一改写历史实验。

| 参数 | 默认值（对比 / 模型列表） | 含义与实际作用 | 有效范围/限制 |
| --- | --- | --- | --- |
| samples / n_samples | 200000 / 2000 | 生成样本行数；增加样本通常增加拟合时间并改善统计稳定性 | 整数 ≥20；两种别名同时写时须相同 |
| n_features | 30 / 20 | 生成器输出总特征数，含信息、冗余、重复及剩余纯噪声列 | 正整数，至少容纳下列三部分之和 |
| n_informative | 20 / 12 | 真正用于构造类别分离簇的特征数；信息维数决定问题结构 | 正整数；须满足簇数约束 |
| n_redundant | 5 / 4 | 信息特征的随机线性组合，增加相关性，不增加独立生成信号 | 非负整数 |
| n_repeated | 0 | 从信息/冗余列复制的重复列，用于模拟重复测量 | 非负整数 |
| n_classes | 2 | 生成目标类别数 | 固定为 2；不支持多分类 |
| n_clusters_per_class | 2 | 每个类别的高斯簇数量；更多簇可能使边界更复杂 | 正整数，2 × 此值 ≤ 2**n_informative |
| class_sep | 1.0 | 缩放生成类别簇中心间的距离；更大通常更容易区分 | 有限数 >0 |
| flip_y | 0.01 | 随机重新赋标签的样本比例，用于引入标签噪声；重赋值也可能与原标签相同 | [0,1] |
| weights | null | 两个类别的近似生成比例，制造类别不平衡；flip_y 会进一步影响实际比例 | null 为平衡；长度 1 自动补余数，或长度 2 且和为 1；元素 [0,1] |
| seed / random_state | 42 | 只控制数据生成随机序列；同一配置和种子生成同一数据 | [0,2³²) 整数；两别名同时写时须相同 |
| model_seed | 跟随 seed | 指定共享模型与切分种子；可与数据种子不同而不重新生成数据 | [0,2³²) 整数；compare.py 的 model.random_state 优先 |
| test_size | 0.2 | 分层留作验证集的比例，改变训练样本数和验证估计精度 | 浮点数 (0,1)，两侧至少 2 行且可进行分层切分 |
| append_constant | true / false | 生成后额外追加全 1 列，保留旧 compare.py 的零方差筛选演示 | 布尔值；true 时原始输入列数为 n_features+1 |

`n_informative + n_redundant + n_repeated <= n_features`；剩余列是纯噪声。
不自动降低信息列数来凑合较小的 n_features，非法组合会报清楚的错误。
极端 weights、小样本或噪声可能导致实际类别不足，生成后仍检查每类至少 2 行，分层切分失败也明确传播。
每次比较只生成一次数据，所有模型共享该数组和相同的分层索引。

CLI 支持上述参数的连字符形式，如 `--n-features 12 --n-informative 6 --n-redundant 2 --n-repeated 1`、
`--class-sep 1.5 --flip-y 0.02 --weights 0.6 0.4 --test-size 0.25`。
`--samples`/`--n-samples`、`--seed`/`--random-state` 为别名；`--append-constant`/`--no-append-constant` 显式控制附加列。
未提供的 CLI 项均不会覆盖 JSON。

### 合成数据与外部 CSV（dataset）

两个脚本均接受顶层 `dataset`。省略该节或使用 `{"source": "synthetic"}` 时，保留原有合成数据、
参数、精度及附加常数列行为，两份示例 JSON 默认均为 synthetic。运行小规模合成数据：

```bash
.venv/bin/python compare.py --config config.example.json --samples 200 --estimators 3 --output results/synthetic_new
.venv/bin/python compare_models.py --config models.example.json --samples 200 --output results/synthetic_models_new
```

外部 CSV 的 compare.py 配置示例：

```json
{
  "dataset": {
    "source": "csv", "path": "data/train.csv", "target_column": "label",
    "feature_columns": ["age", "score"]
  },
  "experiment": {"model_seed": 23, "test_size": 0.25},
  "model": {"n_estimators": 3, "max_depth": 2},
  "distributed": {"n_workers": 2, "total_threads": 2},
  "feature_reduction": {"enabled": false}
}
```

compare_models.py 使用相同的 dataset 配置；例如：

```json
{
  "dataset": {"source": "csv", "path": "data/train.csv", "target_column": "label"},
  "experiment": {"model_seed": 23, "test_size": 0.25},
  "feature_reduction": {"enabled": false},
  "models": [
    {"name": "shallow", "framework": "single", "model": {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}},
    {"name": "deeper", "framework": "single", "model": {"n_estimators": 3, "max_depth": 3, "n_jobs": 1}}
  ]
}
```

CSV 约定与限制：

- 使用 pandas 默认 CSV 解析（逗号分隔、首行为列名）；`path` 和 `target_column` 必填。
  相对路径以**运行命令的工作目录**为基准，不以 JSON 所在目录为基准。
- `feature_columns` 可省略，此时使用除目标外的全部列；显式指定时须非空、无重复、列存在，
  且不能包含目标。顺序即输入顺序。ID、索引或其他不应训练的列需自行排除。
- 特征只支持数值列，转换为算子使用的 float32；拒绝文本类别、无穷大及 float32 溢出。
  不提供自动编码、填补、缩放或其他预处理管线。缺失数值仅在关闭降维时交给 XGBoost；
  开启 PCA/方差筛选要求全部特征有限。compare.py 默认启用方差筛选，缺失数据需先处理或显式关闭。
- 标签须无缺失、数值标签须有限，恰好两个类别且每类至少两行，支持字符串标签；
  阳性类别仍为算子 LabelEncoder 排序后的第二类。样本数和 test_size 须允许分层切分及双方二分类评估。
- CSV 每次运行只加载一次，所有配置共享 X/y、模型种子和分层切分规则。
  相同 CSV、列顺序、模型/切分种子和 test_size 可复现切分；分布式概率仍有前述跨启动限制。
  CSV 不使用 samples、n_features 等合成参数，也不追加常数列；这些参数仍按原规则校验。
- 数据和切分仍在驱动内存完成，不是流式或分布式 CSV 读取。结果记录路径、目标、所选列和数据哈希，
  评估、绘图、降维及模型持久化沿用原流程。
  Python `compare_models(X, y, ...)` 继续以传入数组为准，dataset 仅供 CLI 加载。

下面的命令从项目根目录运行，生成固定种子的 80 行 CSV，同时演示两种脚本。
输入和输出均位于新建临时目录，不覆盖历史结果：

```bash
csv_demo_dir=$(mktemp -d)
.venv/bin/python - "$csv_demo_dir" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

output = Path(sys.argv[1])
rng = np.random.default_rng(17)
y = np.tile([0, 1], 40)
pd.DataFrame({"age": rng.integers(18, 70, 80), "score": rng.normal(size=80) + y,
              "label": y}).to_csv(output / "train.csv", index=False)
shared = {"dataset": {"source": "csv", "path": str(output / "train.csv"), "target_column": "label"},
          "experiment": {"model_seed": 23, "test_size": 0.25},
          "feature_reduction": {"enabled": False}}
model = {"n_estimators": 3, "max_depth": 2}
(output / "compare.json").write_text(json.dumps({**shared, "model": model,
    "distributed": {"n_workers": 2, "total_threads": 2}}))
(output / "models.json").write_text(json.dumps({**shared, "models": [
    {"name": "shallow", "framework": "single", "model": {**model, "n_jobs": 1}},
    {"name": "deeper", "framework": "single", "model": {**model, "max_depth": 3, "n_jobs": 1}}]}))
PY
.venv/bin/python compare.py --config "$csv_demo_dir/compare.json" --output "$csv_demo_dir/comparison"
.venv/bin/python compare_models.py --config "$csv_demo_dir/models.json" --output "$csv_demo_dir/models" --plot
```

### 特征数量、采样、降维和树深度的区别

`n_features` 决定生成数据的列数；`colsample_*` 只影响建树时每次从哪些列寻找切点，未来预测仍必须输入原始全部列。
VarianceThreshold 永久筛掉满足阈值条件的列，PCA 则把列映射成主成分；两者都由保存的预处理器应用于未来数据。
`max_depth` 只限制树沿一条决策路径能分裂多少层，既不是特征数量，也不是降维开关。

数据 seed 决定 X/y；模型种子决定随机采样。现有算子额外把模型种子用于 train/validation 切分，
因此本次没有引入独立于模型的切分随机种子。比较中必须共享同一个模型/切分种子，才能保证逐索引一致。

额外参数交给当前安装版本的 XGBoost 原生引擎做小数据探测校验，未知或未使用参数报错。
支持树 booster `gbtree` 和 `dart`。为避免覆盖校验值，拒绝 `eta/nthread/seed/lambda/alpha` 别名，使用表中规范名称。
不支持多分类、GPU、原生类别特征、自定义 objective 或自定义 callable 指标；类别特征需先编码。
“全部参数可配置”限于现有 CPU、二分类、数值特征、树模型接口范围；保留表中校验约束，
例如 learning_rate 限于 (0,1]、max_depth 必须为正整数，不能据此声称支持 XGBoost 的所有运行模式。
`monotone_constraints` 支持按特征位置配置的 tuple/list/string；校验探针会匹配约束维度。
不支持依赖原生特征名的约束字典，因为训练输入会转成 NumPy 数组。
X 为二维 NumPy 数组或数值 pandas DataFrame；y 为两个类别的一维标签，支持字符串。
阳性类别为 `LabelEncoder.classes_[1]`，阈值 0.5。DataFrame 预测检查列名与顺序；NumPy 使用位置语义。
未启用降维时 NaN 由 XGBoost 处理，inf 始终拒绝；启用降维需先处理所有缺失值。

### 降维与防泄漏

直接使用算子、模型列表脚本默认关闭；compare.py 为兼容旧版默认启用零方差筛选。
两份 JSON 都可使用同样的共享 `feature_reduction` 对象：

```json
{"enabled": true, "method": "pca", "n_components": 0.95}
```

| 参数 | 默认值 | 含义与实际作用 | 有效范围/限制 |
| --- | --- | --- | --- |
| enabled | compare.py 为 true；compare_models.py 为 false | 是否在训练切分后拟合并应用预处理；false 时保留所有原始列 | 布尔值 |
| method | 启用时默认 variance_threshold | 选择删低方差列或用主成分替代原始列；不控制 XGBoost 随机采样 | variance_threshold / pca |
| threshold | 0.0 | 方差 ≤ 阈值的列被删除；零阈值只删常数，更大值也可能删除有用的低尺度变量 | 非负有限数；仅方差筛选使用，是绝对方差而非保留比例 |
| n_components | 0.95 | PCA 整数指定保留维数，浮点指定累计解释方差目标；更少维数压缩更多信息 | 正整数 ≤ min(训练行数, 输入列数)，或浮点 (0,1)；整数 3/1 有效，浮点 3.0/1.0 无效 |

启用方法时只使用对应参数，另一方法的字段可以留在 JSON 里以便切换；disabled 时不拟合任何降维器。
CLI `--feature-reduction none` 关闭，`--feature-reduction variance_threshold --threshold 0.01` 启用方差筛选，
`--feature-reduction pca --n-components 4` 或 `--n-components 0.95` 启用 PCA。
只修改 threshold/n_components 不会自动打开已关闭的降维；请同时显式选择方法，否则报错。
模型列表文件的降维配置对整个 models 数组共享；要比较不同降维配置，可用不同文件显式分次运行，当前不提供逐模型降维覆盖。

算子接口 `feature_reduction(method=None)` 关闭。该方法**仅配置**，返回算子，不在切分前拟合，
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

| 参数 | 默认值 | 含义与实际作用 | 有效范围/限制 |
| --- | --- | --- | --- |
| n_workers | 2 | 并行训练的独立 Dask Worker 进程数，每个处理部分行，并参与 collective 聚合 | 正整数；训练/验证数据均至少每 Worker 一行 |
| threads_per_worker | compare.py 为 total_threads/n_workers；模型列表默认 n_jobs | 每个 Worker 可使用的线程数；匹配估计器 n_jobs，避免每个进程默认占满所有核 | 正整数；显式值须与 n_jobs 一致 |
| total_threads | compare.py 为 4 | 单机和分布式双方的 XGBoost 总线程预算；单机全用，分布式按 Worker 均分 | compare.py 专用，正整数且整除 n_workers；模型列表直接设置 n_jobs/threads_per_worker |
| memory_limit | auto | 每个自建 Worker 的内存预算；影响溢写/暂停/重启策略，不是树的超参数 | Dask 支持的字节整数、内存字符串、auto 或 0（不限） |
| scheduler_address | 未设置 | 不设置则创建并回收本地集群；设置则连接已有集群，仅关闭自己的 Client | 可连接的 scheduler 地址；外部实际线程/Worker 数须一致 |
| port | 8786 | 独立 start_cluster.py 启动脚本的 scheduler 监听端口 | CLI 专用；合法端口，0 为动态分配 |

`n_workers × threads_per_worker` 是 Dask 总训练线程数；`n_jobs` 是每个估计器的线程数，不能再当作全局预算乘一次。
例如总预算 4，可比较 single n_jobs=4 与 Dask n_workers=2、threads_per_worker=n_jobs=2。

```bash
# 终端一：持续运行；Ctrl-C 关闭
.venv/bin/python start_cluster.py --n-workers 2 --threads-per-worker 2 --port 8786
# 终端二：仅连接已有集群，结束后只关闭自己的 Client
.venv/bin/python compare.py --scheduler-address tcp://127.0.0.1:8786
```

外部集群须与配置的 Worker 数量和线程数一致，数据在各 Worker 间按行划分。
可配置多机 scheduler，但本项目实测范围是同机多进程 CPU。无故障自动重试；Worker 故障会传播训练异常。
每个 Worker 内部的 `after_training` 回调记录真实 collective rank、world_size、PID 和地址，并校验每个 rank 都出现。

## 比较用户定义的模型配置

新增独立的 [compare_models.py](compare_models.py)；原有 `compare.py` 单机/分布式工作流保持不变。
只运行用户列出的配置，按输入顺序展示结果，不生成参数网格、不搜索、不自动选最优模型。

```bash
# 小样本演示，输出到独立目录，不覆盖历史结果
.venv/bin/python compare_models.py --config models.example.json --samples 2000 --plot --output results/my_new_models
```

[models.example.json](models.example.json) 现在推荐使用共享 experiment、feature_reduction 和 models 数组的对象。
原先直接以命名模型数组作为顶层 JSON 的格式继续支持，省略共享字段时采用原有默认值。
下面是可直接保存为独立文件执行的完整配置（非默认数据维数与三种特征采样）：

```json
{
  "experiment": {
    "n_samples": 600, "n_features": 12, "n_informative": 6,
    "n_redundant": 2, "n_repeated": 1, "n_classes": 2,
    "class_sep": 1.5, "flip_y": 0.02, "weights": [0.6, 0.4],
    "seed": 17, "model_seed": 23, "test_size": 0.25, "append_constant": false
  },
  "feature_reduction": {"enabled": true, "method": "pca", "n_components": 4},
  "models": [
    {"name": "single", "framework": "single",
     "model": {"max_depth": 3, "n_estimators": 10, "n_jobs": 2,
               "colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.6}},
    {"name": "dask", "framework": "dask",
     "model": {"max_depth": 3, "n_estimators": 10, "n_jobs": 1,
               "colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.6},
     "distributed": {"n_workers": 2, "threads_per_worker": 1}}
  ]
}
```

上例使用相同模型超参数、相同总线程预算比较执行模式；可添加不同 max_depth/learning_rate 的命名项来比较模型参数。
项目内 models.example.json 保留用户已有的三组模型设置，分别为单机、2 Worker、4 Worker，共享相同数据配置。
未写出的模型参数采用现有算子默认值（如 hist、max_bin=512、learning_rate=0.1），不重复实现训练或指标计算。

命名项中 `name` 是非空唯一的结果标识，不改变训练；`framework` 必须为 single/dask，决定训练后端。
`model` 和 `distributed` 分别使用上方模型与执行参数表。`--config` 只选择配置文件，不执行自动搜索。
`--output` 是结果目录（compare.py 默认 results，compare_models.py 默认 results/model_comparison）；
`--plot` 默认 false，仅控制模型列表脚本的训练/验证图，不改变训练或指标。

可直接使用已有 NumPy/pandas 数据，函数返回完整结果字典：

```python
from compare_models import compare_models

# 在脚本的 if __name__ == "__main__": 中调用，X/y 为原始特征和二分类标签。
results = compare_models(
    X, y, "models.example.json",  # 也可以直接传入上面的字典列表
    random_state=42, test_size=0.2,
    reduction={"method": "variance_threshold", "threshold": 0.0},
    output_dir="results/my_models", plot=True,
)
```

所有配置共享 `random_state`、`test_size` 和 `reduction`（未配置时默认不降维）。实际切分和降维拟合始终由
`DistributedXGBClassifier.train()` 完成，函数逐项检查训练/验证索引完全相同，再调用算子的 evaluate
分别评估训练集和验证集。降维器仅在各自相同的训练集上拟合，评估接收原始特征。
由于现有算子用同一个 random_state 控制模型与切分，配置中的 model.random_state 必须与共享种子一致；
冲突会明确报错，不静默改写。共享种子默认取 experiment.model_seed，省略则跟随数据 seed。
CLI 显式 `--seed`（或 `--model-seed`）会覆盖各 model.random_state，以遵循 CLI 优先规则；不提供 CLI 种子时不会覆盖 JSON。
`--test-size` 控制共享验证比例，未提供时遵循 JSON。
Python 调用可传入配置列表、共享配置对象或文件路径；显式 random_state/test_size/reduction 参数优先于文件。
Python 接收的 X/y 是权威输入，不会按 experiment 字段重新生成或替换这些数组；数据生成字段仅由 CLI 使用。

Dask 的 `distributed` 支持原有 `n_workers`、`threads_per_worker`、`memory_limit` 和 `scheduler_address`。
省略 Worker 数时为 2；省略线程数时使用 model.n_jobs（再省略则用算子默认 2）。
若只写 threads_per_worker，则自动设置 model.n_jobs；显式填写两者时必须一致。
single 模式不接受非空 distributed。终端展示各配置的总线程数，比较速度时应自行匹配实际 CPU 预算。
所有配置先做参数检查；Python 列表中的参数也须可保存为 JSON，支持 NumPy 标量/数组，不接受回调对象。

输出文件包括：

- `terminal_report.txt`：训练/验证 ROC-AUC、Accuracy、F1、fit 秒数对照表，以及 Worker/线程、转换后特征数、
  boosting 轮数、实际树数、节点数和叶子数。分数 6 位小数、fit 时间 3 位，表格按输入顺序排列。
- `comparison.json`：完整有效参数、资源配置、Worker 训练证据、共享切分索引、全部训练/验证指标、
  分阶段耗时和模型复杂度；若启用 early stopping，额外记录 best_iteration。
- `comparison.csv`：全部逐配置结果展开为列，复杂列表字段以 JSON 字符串保存，便于 pandas 读取。
- `train_validation.png`：可选的训练/验证 ROC-AUC、Accuracy、F1 对比图，由 `plot=True` 或 `--plot` 开启。
- `training.log`：底层训练诊断；Dask 集群仍由现有算子管理和关闭。

训练集指标用于观察拟合程度；原有验证集也供 early stopping 使用，因此不是独立最终测试集。
复杂度统计覆盖模型实际保存的全部树；有 early stopping 时，预测可能只使用 best_iteration 对应的树。
fit 时间排除预处理和集群启动，总 train 时间包含这些阶段及资源关闭，不包括评估和写报告。
沿用现有 Dask 跨启动数值不完全稳定的限制，不强制不同模式的预测相同。

## 本次配置扩展的验证

`results/experiment_config_check/tests.log` 为完整自动测试结果。
`compare_demo.json`/`models_demo.json` 是实际执行的非默认 600 样本配置，分别测试 14 维输入 → PCA 5 维，
以及 12 维生成特征 + 常数列 → 方差筛选 12 维；三种 colsample 分别为 0.8/0.75/0.6，数据种子 17、模型/切分种子 23。
所有新产物放在该目录下，历史结果、旧模型和初版 ZIP 均不覆盖。

```bash
.venv/bin/python compare.py --config results/experiment_config_check/compare_demo.json --output results/another_compare_demo
.venv/bin/python compare_models.py --config results/experiment_config_check/models_demo.json --output results/another_models_demo --plot
```

旧版默认生成数据的逐元素一致性由测试覆盖；分布式跨启动概率的既有限制仍保留。

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
该 ZIP 是初版快照，本次配置与终端格式更新以当前源码为准，未改写历史提交包。

拟合计时包含估计器构造 DMatrix/quantile sketch/collective 训练；集群启动、显式数据分发分别记录。
总 train 耗时还含预处理、模型转换和集群关闭。报告记录最后一次顺序执行，没有做多轮统计性能基准。
不做阈值后处理或共享模型来强制两种模型一致：分布式直方图切点可能不同，报告实际标签和概率差异。
题库“仅微小浮点误差”的严格要求可能因此无法满足，详见实际报告；按照用户补充要求透明呈现。

joblib 会执行反序列化代码，仅加载可信产物。保存包保留模型、标签编码器、列信息和降维器，
不保存验证集缓存或训练索引；跨大版本部署建议重新验证，原生 UBJ 模型也可由 `op.model_.save_model(...)` 导出。

官方参考：[XGBoost Dask 教程](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html)、
[参数说明](https://xgboost.readthedocs.io/en/stable/parameter.html)。

## 开发过程与实验总结

### 分布式训练的理解与验证。 
项目初期，我对分布式运算并不熟悉，因此首先从较小规模的实验入手，理解单机多线程与 Dask 多 Worker 训练的区别，并逐步验证不同 Worker 是否真正参与模型训练。在此过程中，我重点研究了 XGBoost 的 hist 和 approx method、Worker 间的数据划分与统计量聚合，以及单机和分布式训练结果为何可能存在数值差异。

### 合成数据与模型参数实验。 
在完成基本训练流程后，我使用 scikit-learn 的 make_classification 生成不同规模和特征结构的二分类数据，比较了 max_depth、n_estimators、learning_rate 和正则化参数对模型表现及训练成本的影响。其中，增加树深度能够增强模型的拟合能力，但节点数量和训练耗时也会显著增加；使用较浅的树并增加 Boosting 轮数，则可以在保持较高验证性能的同时降低模型复杂度。我还测试了 PCA 和 VarianceThreshold，观察不同特征组成与降维设置对输入维度和预测结果的影响。

### 外部数据集与可复用性测试。 
由于合成数据上的验证 AUC 经常接近 0.99，我进一步考虑了人工生成数据与实际分类任务之间的差异，并将测试重点转向外部数据集。首先使用 scikit-learn 自带的 Breast Cancer Wisconsin 数据集，验证算子能够处理非合成的二分类数据；随后考虑到该数据集仅有 569 条记录，我又选择了规模更大的 Forest Covertypes 数据集，将原始七分类标签转换为“Lodgepole Pine 与其他覆盖类型”的二分类任务，使用全部 581,012 条记录进行训练与比较。