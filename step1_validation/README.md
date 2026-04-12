# Step 1: Data Validation & Temporal Split

This folder contains an isolated pipeline for strict dataset validation, user quality filtering, temporal split, leakage-safe bucketization, and naive baseline creation.

## Script
- `run_validation.py`

## Outputs
Saved to `step1_validation/outputs/`:
- `train_dataset.csv`
- `test_dataset.csv`
- `train_lag_features.csv`
- `test_lag_features.csv`
- `validation_report.json`
- `metadata.json`

## Run
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step1_validation/run_validation.py
```

Optional explicit column mapping:
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step1_validation/run_validation.py \
  --csv-path /absolute/path/to/credit_card_transactions.csv \
  --user-col cc_num \
  --date-col trans_date_trans_time \
  --amount-col amt \
  --category-col category
```

