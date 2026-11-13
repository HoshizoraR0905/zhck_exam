# Experiment configuration validation

All 93 tests passed. Both scripts ran with 600 samples and non-default feature settings.
Data seed: 17. Model/split seed: 23. Split: 450 training / 150 validation.
Feature sampling: bytree=0.8, bylevel=0.75, bynode=0.6. Native training configs verified for both backends.

| Script | Generated / input features | Reduction | Output features | Verified workers |
| --- | --- | --- | --- | --- |
| compare.py | 14 / 14 | PCA, n_components=5 | 5 | 2/2 |
| compare_models.py | 12 / 13 (constant added) | VarianceThreshold, threshold=0 | 12 | 2/2 |

PCA mean equals the mean of the training rows only. Saved model probabilities match after reload.
All 67 pre-existing result files, the examination, and original ZIP are byte-for-byte unchanged.
Training and evaluation logic is unchanged; Dask now retains diagnostic configuration before local model conversion.
Legacy dataset defaults and top-level model arrays remain supported and tested.

Detailed metrics/timings, reports, CSV, plots, and raw inputs are under compare_demo/ and models_demo/.
Dask cross-start numerical variability and the coupling of model and split seeds remain existing limitations.
