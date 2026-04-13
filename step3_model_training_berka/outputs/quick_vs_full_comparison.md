# Quick vs Full Comparison (Classification)

## Main table
| target | model | balanced_accuracy_full | balanced_accuracy_quick | f1_macro_full | f1_macro_quick | relative_gain_vs_persistence_full | relative_gain_vs_persistence_quick | f1_delta_full_minus_quick |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| net | logistic_regression | 0.5123 | 0.4369 | 0.3658 | 0.4341 | 0.7172 | 0.3512 | -0.0682 |
| net | random_forest | 0.4972 | 0.4886 | 0.4820 | 0.4917 | 1.2625 | 0.5308 | -0.0098 |
| net | sgd_classifier | 0.4064 | 0.4150 | 0.4196 | 0.4159 | 0.9699 | 0.2946 | 0.0038 |
| spend | logistic_regression | 0.4020 | 0.3390 | 0.3635 | 0.3274 | 0.8308 | 0.9208 | 0.0362 |
| spend | random_forest | 0.5355 | 0.4212 | 0.4595 | 0.4145 | 1.3140 | 1.4318 | 0.0450 |
| spend | sgd_classifier | 0.4233 | 0.3278 | 0.3869 | 0.2720 | 0.9486 | 0.5957 | 0.1149 |

## Acceptance focus
- Relative gain formula: `(F1_model - F1_persistence) / F1_persistence`.
- Recommended threshold: at least `+0.50` (50%) and stretch target `+0.70` (70%).
