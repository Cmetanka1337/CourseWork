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

## Quick check
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/test_feature_engineering.py
```
