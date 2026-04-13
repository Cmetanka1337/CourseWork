# Berka Evaluation Pipeline

## 1) Data placement

Place Berka CSV files into:

```text
data/
  berka/
    raw/
      trans.csv
      account.csv
      disp.csv
      client.csv
      order.csv
      loan.csv
      card.csv
      district.csv
```

`trans.csv` is required.

## 2) Run steps

```bash
python3 step0_berka_ingestion/run_berka_ingestion.py \
  --input-dir data/berka/raw \
  --output-dir data/berka/processed

python3 step1_berka_weekly_builder/run_build_weekly.py \
  --input-csv data/berka/processed/transactions_normalized.csv \
  --output-dir step1_berka_weekly_builder/outputs

python3 step3_model_training_berka/train_classification.py \
  --input-dir step1_berka_weekly_builder/outputs/classification \
  --output-dir step3_model_training_berka/outputs

python3 step3_model_training_berka/train_classification.py \
  --input-dir step1_berka_weekly_builder/outputs/classification \
  --output-dir step3_model_training_berka/outputs \
  --target bucket_net_t_plus_1

python3 step3_regression_training/train_regression.py \
  --input-dir step1_berka_weekly_builder/outputs/regression \
  --output-dir step3_regression_training/outputs

python3 step3_model_training_berka/generate_quick_vs_full_comparison.py \
  --input-dir step3_model_training_berka/outputs

python3 reports/berka_feasibility/generate_feasibility_report.py \
  --weekly-dir step1_berka_weekly_builder/outputs \
  --classification-report step3_model_training_berka/outputs/classification_report_full.json \
  --regression-report step3_regression_training/outputs/regression_report_full.json \
  --ingestion-report data/berka/processed/ingestion_report.json \
  --output-dir reports/berka_feasibility

python3 reports/berka_feasibility/generate_dataset_analysis.py \
  --weekly-dir step1_berka_weekly_builder/outputs \
  --classification-output-dir step3_model_training_berka/outputs \
  --regression-output-dir step3_regression_training/outputs
```

## 3) Quick smoke mode

```bash
python3 step0_berka_ingestion/run_berka_ingestion.py --max-rows 10000
python3 step1_berka_weekly_builder/run_build_weekly.py --max-users 200 --max-weeks 40
python3 step3_model_training_berka/train_classification.py --quick
python3 step3_model_training_berka/train_classification.py --quick --target bucket_net_t_plus_1
python3 step3_regression_training/train_regression.py --quick
python3 step3_model_training_berka/generate_quick_vs_full_comparison.py
```

## 4) Required full-run artifacts

- `step3_model_training_berka/outputs/classification_report_full.json`
- `step3_model_training_berka/outputs/classification_report_quick.json`
- `step3_model_training_berka/outputs/fold_metrics.csv`
- `step3_model_training_berka/outputs/quick_vs_full_comparison.md`
- `step3_model_training_berka/outputs/*stability_f1_by_fold.png`
- `step3_regression_training/outputs/regression_report_full.json`
- `step3_regression_training/outputs/regression_report_quick.json`
- `step3_regression_training/outputs/*fold_metrics.csv`
- `reports/berka_feasibility/berka_dataset_analysis.md`

## 5) Leakage controls

- Features for week `t+1` are built from data available at week `t` or earlier only.
- Rolling/lag features use explicit shifts, so no future values leak into predictors.
- Classification and regression use holdout last weeks + `TimeSeriesSplit` for time-aware CV.
- Pipelines fit scalers/encoders on train folds only.
- Baselines are mandatory: persistence + majority for classification, persistence + rolling mean for regression.

## 6) Why this validation setup

- Time-series requires walk-forward validation; random CV is optimistic and can leak temporal structure.
- Train-only preprocessing avoids peeking into future distributions.
- In time-series tasks, simple baselines are often strong, so model gains must be measured relative to them.

