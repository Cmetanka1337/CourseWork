# Feature Iteration 01: Calendar + Inflow/Outflow + Regularity

## What was added
- Calendar: `week_of_year`, `month`, `quarter`, `week_of_month`, `is_month_start_week`, `is_month_end_week`.
- Flow dynamics: `delta_inflow`, `delta_outflow`, `inflow_outflow_ratio`, `inflow_share`, explicit lag columns.
- Regularity (anti-leakage): 8-week rolling mean/std + frequency with `shift(1)`, plus `weeks_since_inflow/outflow`.
- Fill policy: lag/rolling fields -> `0.0`; final feature files validated to be NaN-free.

## Metrics before vs after
| target | mode | model | F1 before | F1 after | delta | bal_acc before | bal_acc after | delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| net | full | logistic_regression | 0.3658 | 0.4206 | 0.0548 | 0.5123 | 0.5707 | 0.0584 |
| net | full | random_forest | 0.4820 | 0.4928 | 0.0108 | 0.4972 | 0.5077 | 0.0105 |
| net | full | sgd_classifier | 0.4196 | 0.4557 | 0.0360 | 0.4064 | 0.6095 | 0.2031 |
| net | quick | logistic_regression | 0.4341 | 0.4376 | 0.0036 | 0.4369 | 0.4412 | 0.0043 |
| net | quick | random_forest | 0.4917 | 0.5436 | 0.0518 | 0.4886 | 0.5433 | 0.0546 |
| net | quick | sgd_classifier | 0.4159 | 0.4566 | 0.0407 | 0.4150 | 0.4364 | 0.0214 |
| spend | full | logistic_regression | 0.3635 | 0.4609 | 0.0974 | 0.4020 | 0.5248 | 0.1229 |
| spend | full | random_forest | 0.4595 | 0.4750 | 0.0156 | 0.5355 | 0.5557 | 0.0202 |
| spend | full | sgd_classifier | 0.3869 | 0.4565 | 0.0696 | 0.4233 | 0.5168 | 0.0936 |
| spend | quick | logistic_regression | 0.3274 | 0.2614 | -0.0659 | 0.3390 | 0.3010 | -0.0380 |
| spend | quick | random_forest | 0.4145 | 0.3699 | -0.0445 | 0.4212 | 0.4038 | -0.0174 |
| spend | quick | sgd_classifier | 0.2720 | 0.2637 | -0.0083 | 0.3278 | 0.3034 | -0.0244 |

## Spend target acceptance check (full run, RF)
- Relative gain vs persistence (before/after): **1.3140 / 1.3924**
- Full RF F1_macro delta: **0.0156**
- Full RF balanced_accuracy delta: **0.0202**
- Majority baseline F1 (after): **0.1446**

## Per-class changes (RF, spend target, full)
| class | precision before | precision after | recall before | recall after | f1 before | f1 after | support after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4489 | 0.4807 | 0.2392 | 0.2322 | 0.3121 | 0.3132 | 6760 |
| 1 | 0.3233 | 0.3475 | 0.8738 | 0.9046 | 0.4719 | 0.5021 | 4422 |
| 2 | 0.7392 | 0.7551 | 0.3377 | 0.3557 | 0.4636 | 0.4836 | 12455 |
| 3 | 0.5149 | 0.5110 | 0.6914 | 0.7302 | 0.5902 | 0.6013 | 6979 |

## Fold stability (RF, spend target)
- Head/tail F1 mean before: **0.5992 / 0.5693** (relative drop **0.0499**).
- Head/tail F1 mean after: **0.6235 / 0.5936** (relative drop **0.0480**).
- Fold artifacts: `step3_model_training_berka/outputs/full_spend_fold_metrics.csv`, `step3_model_training_berka/outputs/full_spend_fold_per_class_metrics.csv`, `step3_model_training_berka/outputs/full_spend_fold_confusion_matrices.json`.

## Optional RF tuning (`--tune-rf`)
- Quick tuning used: `RandomizedSearchCV(TimeSeriesSplit(n_splits=3))` with `n_iter=6`.
- Best params: `{'n_estimators': 160, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None}`.
- Best CV F1_macro: **0.5509 +- 0.0669**.
- Quick tuned RF test F1_macro: **0.3666** vs untuned **0.3699**.

## Decision
- **Keep**: required uplift is met (>= +0.02 on full spend F1_macro or balanced_accuracy) and gain vs persistence remains strong.
