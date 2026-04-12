# Financial Intensity Range Validation

## Stage 1 - Target discretization
- threshold_low: 47.7400
- threshold_high: 311.4400
- quantile_mode: default ([0.25, 0.75])
- class_distribution_pct: {'0': 30.047840624224058, '1': 17.488160310025837, '2': 34.975980480526445, '3': 17.488018585223656}
- mean_amount_per_class: {'0': 0.0, '1': 16.58655681804691, '2': 148.61424942042774, '3': 710.8345259745715}

## Stage 2 - Temporal signal
- global_spearman: 0.5204
- per_category_spearman: {'groceries': 0.4538749561574069, 'fuel': 0.38981573992673796, 'shopping': 0.4200646725783469}

Transition matrix P(bucket_t+7 | bucket_t):

| bucket | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0.0000 | 0.5561 | 0.1790 | 0.2267 | 0.0382 |
| 1.0000 | 0.3055 | 0.3831 | 0.2390 | 0.0724 |
| 2.0000 | 0.1925 | 0.1186 | 0.5302 | 0.1587 |
| 3.0000 | 0.0650 | 0.0723 | 0.3126 | 0.5501 |

## Stage 3 - User entropy
- entropy_quantiles: {'p10': 0.43274512368878465, 'p50': 1.0158491883593959, 'p90': 1.1950483455811467}
- entropy_share: {'low': 0.18921668362156663, 'medium': 0.8097660223804679, 'high': 0.001017293997965412}

## Stage 4 - Baseline model test
- model_f1_weighted: 0.8810
- baseline_f1_weighted: 0.1873
- model_accuracy: 0.8810
- baseline_accuracy: 0.3564
- model_balanced_accuracy: 0.8788
- baseline_balanced_accuracy: 0.2500
- train_rows_used: 250000
- test_rows_used: 150000
- rf_config: n_estimators=120, max_depth=14, n_jobs=1

Confusion matrix:

| index | pred_0 | pred_1 | pred_2 | pred_3 |
| --- | --- | --- | --- | --- |
| true_0 | 39331 | 2048 | 2091 | 199 |
| true_1 | 1937 | 22319 | 1664 | 231 |
| true_2 | 2211 | 1721 | 46881 | 2652 |
| true_3 | 167 | 220 | 2706 | 23622 |

## Verdict
- class_separation_ratio_c2_c1: 8.9599
- predictive_lift_ratio: 4.7033
- best_category_spearman: 0.4539
- criteria_passed: 3/3
- FINAL VERDICT: GO