# 实际 AI 协作记录

本项目由 Codex 在本次会话中根据用户附件实现、运行和修正。没有虚构人工复审、额外用户轮次或多代理协作。
以下三条关键提示词是**同一份用户请求中的原文片段**，不是三次独立对话；题库要求“至少3条关键提示词”，
此处按三项实际指令记录。完整提示词来源为用户附加的 `Pasted text.txt`，原题库为权威规格。

## 1. 初次实现

### Task: Implement a Distributed XGBoost Classification Operator

Implement a complete, reproducible Python project for **题目一：XGBoost分类基础算子（支持分布式）** in `算法工程师笔试题库v1.md`, located in the current project directory.

Read the examination file first and treat it as the **authoritative requirements specification**.

The project currently contains only the existing `.venv` environment and the examination `.md` file. The environment was created for previous experiments whose code has since been deleted. Check whether dependencies need to be updated, and reuse the environment if possible.

#### 1. Overall Design

Implement a reusable `DistributedXGBClassifier` class using Python, XGBoost, Dask Distributed, scikit-learn, NumPy, pandas, matplotlib, seaborn, and joblib.

Use `tree_method="hist"` and `max_bin=512` as defaults.

Support two training modes:

- **Single-machine:** `XGBClassifier`.
- **Distributed:** `DaskXGBClassifier` with configurable Worker count.

Both modes must share the same user-facing interface.

Keep the implementation modular, readable, and lightweight. Avoid unnecessary engineering complexity.

#### 2. Operator Interface

Follow the class interface suggested in the examination file:

```python
class DistributedXGBClassifier:
    def __init__(self, framework="dask", **params):
        ...

    def set_params(self, **params):
        ...

    def feature_reduction(self, X, y=None, method="variance_threshold", **kwargs):
        ...

    def train(self, X, y, test_size=0.2, distributed_config=None):
        ...

    def save_model(self, path):
        ...

    def evaluate(self, X_test=None, y_test=None, plot_dir=None):
        ...
```

You may refine the interface where necessary, but maintain consistency and simplicity.

**All relevant XGBoost hyperparameters must be externally configurable**, including:

- `learning_rate`, `max_depth`, `n_estimators`
- `subsample`, `colsample_bytree`, `min_child_weight`
- `gamma`, `reg_alpha`, `reg_lambda`
- `objective`, `tree_method`, `max_bin`
- `random_state`, `n_jobs`

Support additional valid XGBoost parameters through a configuration dictionary without unnecessarily restricting the interface.

Implement parameter validation with clear errors for invalid values, unknown parameters, and unsupported configurations. Document sensible defaults.

**Binary classification is sufficient.** Multiclass support is not required.

#### 3. Additional Implementation Requirements

Follow all remaining requirements in the examination file, with these clarifications:

- Ensure identical train/validation splits across single-machine and distributed modes, using a fixed configurable random seed.
- Perform feature reduction only after splitting to prevent data leakage. Fit the reducer on training data and apply the same transformation to validation and future prediction data.
- Support optional VarianceThreshold and PCA, with appropriate parameter semantics.
- Use genuine multi-Worker Dask distributed training, not Dask preprocessing followed by single-machine training.
- Support local Dask clusters on Apple Silicon CPU without CUDA.
- Properly manage and close Dask resources.
- Save the trained model together with preprocessing information, and ensure predictions work after reloading without an active cluster.
- Generate all required evaluation metrics and visualizations specified in the examination file.

#### 4. Validation

Include a reproducible experiment using at least **200,000 synthetic samples**, with identical data and model hyperparameters for single-machine and distributed training.

Compare training time, classification metrics, predicted-label agreement, and prediction-probability differences.

Use comparable CPU resources when evaluating performance. Report cluster startup separately from model fitting.

**Do not assume the two trained models must be numerically identical.** Distributed histogram construction may produce different split thresholds. Report actual differences transparently rather than forcing equivalence.

Provide lightweight automated tests for the main operator functionality, parameter validation, preprocessing, and model persistence.

#### 5. Deliverables

Create the complete project, including:

- Source code and distributed startup script.
- `requirements.txt`.
- `README.md` with installation and execution instructions.
- `design.md` (approximately one page, following the examination requirements).
- `ai_collaboration.md` documenting actual AI assistance without fabricating interactions or human reviews.
- Test code and reproducible comparison results.
- Required evaluation plots and saved model artifacts.

Use a compact project structure. Do not modify or delete the original examination file.

#### 6. Execution

Implement the project, install any missing dependencies, and run the tests and demonstration experiments.

Verify that multiple Dask Worker processes actually participate in training and that saved models can be reloaded successfully.

Fix implementation errors encountered during testing.

Finally, summarize the created files, executed tests, actual experimental results, and any remaining limitations.

Prioritize **correctness, reproducibility, and compliance with the examination requirements** over unnecessary features.


## 2. 防泄漏与持久化

**实际提示词**：“Perform feature reduction only after splitting to prevent data leakage. Fit the reducer on training data and apply the same transformation to validation and future prediction data.”

**意图**：防止把验证数据统计信息用于预处理；保证保存后仍可预测。

**AI 执行摘要**：将 feature_reduction 定义为配置操作，在 train 切分之后才 fit；
保存标签编码器、列元数据和降维器。Dask 模型提取为普通本地估计器，关闭集群后验证 reload。

**验证方式**：测试 reducer 的方差/均值与训练索引对应样本的统计值一致，验证 PCA 两种参数语义、
原始输入预测、列顺序错误，以及方差筛选/PCA/Dask 保存加载后的概率完全不变。

## 3. 真实分布式与对比

**实际提示词**：“Do not assume the two trained models must be numerically identical. Distributed histogram construction may produce different split thresholds. Report actual differences transparently rather than forcing equivalence.”

**意图**：验证真正的多 Worker 训练，避免为了满足原题一致性而篡改结果。

**AI 执行摘要**：在 XGBoost collective 训练结束回调中采集 rank/world_size/PID；
使用相同 20 万样本和切分，按总线程预算匹配资源，分别记录集群启动、数据分发、fit 耗时。

**验证方式**：自动测试和完整实验校验 rank 集合、不同进程和 world_size；
保存双方验证概率与索引，用真实标签一致率和概率差统计评估一致性，结果见 results/report.md。

## 实现中的实际调试

初稿参数探测仅将少量 wrapper 参数送给原生引擎，自检时发现这会漏掉 max_leaves 等值校验，
在首次测试前改用安装版本的 get_xgb_params() 提取原生参数。
首次自动测试发现两处错误：未训练 predict 先访问标签编码器，抛出了不清晰的 AttributeError；
以及禁用全局默认 Client 后，distributed.wait 仍需当前 Client 上下文。
分别添加显式拟合检查和局部 client.as_current() 上下文，保留首次日志 tests_initial.log 并重跑验证。
后续补充 early_stopping_rounds 和 importance_type 的错误校验，最终 23 项测试通过。
另用 start_cluster.py 启动外部进程集群，验证训练后集群仍存活、模型加载预测不变，以及 Ctrl-C 退出码为 0。
在补充参数校验和重置 Worker 参与证据之后，重新执行 20 万样本实验；两次标签一致率和概率差统计相同，
报告保留最终执行的实际耗时。另在全新 Python 进程中加载双方模型，核对保存的全部 40,000 个验证概率。
最终自动测试和实验日志保存在 results/，实际修正以这些日志和源代码为准。
所有结论来自本次工具运行和官方文档参考，未编造用户反馈或人工优化记录。
