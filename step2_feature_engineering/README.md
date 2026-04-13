# Step 2: Feature Engineering

This step builds leakage-safe Tier 1 + Tier 2 features from Step 1 outputs.

## Inputs
- `step1_validation/outputs/train_dataset.csv`
- `step1_validation/outputs/test_dataset.csv`
- `step1_validation/outputs/train_lag_features.csv`
- `step1_validation/outputs/test_lag_features.csv`
- `step1_validation/outputs/metadata.json`

## Outputs
- `step2_feature_engineering/outputs/train_features_engineered.csv`
- `step2_feature_engineering/outputs/test_features_engineered.csv`
- `step2_feature_engineering/outputs/train_features_tier3.csv`
- `step2_feature_engineering/outputs/test_features_tier3.csv`
- `step2_feature_engineering/outputs/feature_engineering_report.json`
- `step2_feature_engineering/outputs/tier3_feature_engineering_report.json`
- `step2_feature_engineering/outputs/feature_definitions.md`
- `step2_feature_engineering/outputs/feature_definitions_tier3.md`
- `step2_feature_engineering/outputs/step2_user_statistics.csv`

## Run
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/run_feature_engineering.py
```

## Bucketing Strategy

### User-Specific Quantile Mode (default)

Per-user Q25 and Q75 thresholds are computed from the **training split only** and then applied to both train and test.
- Improves class balance by adapting to each user's spending profile.
- Avoids leakage because test thresholds are derived from train statistics.
- Uses global-train fallback when a user has fewer than 4 train rows.

### Global Quantile Mode (legacy)

Single Q25 and Q75 thresholds are computed across all train users.
- Kept for reproducibility of the previous pipeline.
- Can cause severe class skew when user spending is highly heterogeneous.

### Command-line usage

```zsh
# Recommended (default)
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/run_feature_engineering.py

# Legacy mode
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/run_feature_engineering.py --bucket-mode global_quantile
```

### Expected distributions

- `user_quantile`: train/test should be reasonably balanced (majority class under 80%).
- `global_quantile`: can become degenerate (majority class near 99%), which breaks Step 3 model training.

## Quick check
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/test_feature_engineering.py
```
