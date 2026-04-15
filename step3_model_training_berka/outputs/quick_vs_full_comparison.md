# Quick vs Full Comparison (Classification)

## Main table
| target | model | balanced_accuracy_full | balanced_accuracy_quick | f1_macro_full | f1_macro_quick | relative_gain_vs_persistence_full | relative_gain_vs_persistence_quick | f1_delta_full_minus_quick |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| net | logistic_regression | 0.5707 | 0.4412 | 0.4206 | 0.4376 | 0.9745 | 0.3624 | -0.0170 |
| net | random_forest | 0.5077 | 0.5433 | 0.4928 | 0.5436 | 1.3134 | 0.6921 | -0.0508 |
| net | sgd_classifier | 0.6095 | 0.4364 | 0.4557 | 0.4566 | 1.1391 | 0.4213 | -0.0009 |
| spend | logistic_regression | 0.5248 | 0.3010 | 0.4609 | 0.2614 | 1.3213 | 0.5339 | 0.1995 |
| spend | random_forest | 0.5557 | 0.4038 | 0.4750 | 0.3699 | 1.3924 | 1.1706 | 0.1051 |
| spend | sgd_classifier | 0.5168 | 0.3034 | 0.4565 | 0.2637 | 1.2989 | 0.5470 | 0.1928 |

## Acceptance focus
- Relative gain formula: `(F1_model - F1_persistence) / F1_persistence`.
- Recommended threshold: at least `+0.50` (50%) and stretch target `+0.70` (70%).
