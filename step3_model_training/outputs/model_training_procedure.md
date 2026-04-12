# Model Training Procedure (Step 3)

1. Load `train_features_tier3.csv` and `test_features_tier3.csv`.
2. Validate schema (21 columns), missing values, and target labels.
3. Sort rows by `week_t` and `user_id` for temporal consistency.
4. Train RF with `RandomizedSearchCV` and `TimeSeriesSplit(n_splits=5)`.
5. Train LR and SGD with scaling and `GridSearchCV`.
6. Fit final scaler on full train features and evaluate on holdout test set.
7. Export models, metrics, confusion matrices, and feature attribution artifacts.
8. Generate iOS parity documentation and scaler JSON export.
