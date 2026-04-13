# Dataset Validation Pipeline (Stage 0-9)

This project runs a full validation pipeline for transaction data before any ML modeling.
It is aligned with the strict plan: sanity -> category collapse -> time series -> sparsity -> distribution -> autocorrelation -> baselines -> target validation -> user variance -> category-level signal.

## What it produces

- `outputs/report.md` - short human-readable conclusions
- `outputs/report.json` - machine-readable metrics
- `outputs/tables/*.csv` - stage tables
- `outputs/figures/*.png` - distribution plots

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run with local CSV

```bash
python3 main.py \
  --csv-path /absolute/path/to/your.csv \
  --user-col user_id \
  --date-col timestamp \
  --amount-col amount \
  --category-col merchant_name \
  --collapse-mode manual \
  --output-dir outputs
```

## Run with Kaggle dataset download

```bash
python3 main.py \
  --dataset priyamchoksi/credit-card-transactions-dataset \
  --dataset-file credit_card_transactions.csv \
  --collapse-mode manual \
  --output-dir outputs
```

If `--dataset-file` is not provided, the script picks the largest CSV from the downloaded dataset folder.

## Category collapse modes

- `manual` - keyword mapping into coarse categories (`groceries`, `fuel`, `utilities`, ...)
- `top` - keep top-N labels and map the rest to `other`
- `tfidf` - cluster text labels with TF-IDF + KMeans

## Key decision logic

- `CASE 1: no_signal` - high sparsity and/or weak lag-7 autocorrelation
- `CASE 2: weak_signal` - partial signal, conservative approach needed
- `CASE 3: good_signal` - usable signal for ML after baseline checks

## Quick smoke test

```bash
python3 smoke_test.py
```

## Berka feasibility evaluation

The repository now includes a parallel Berka pipeline for honest time-based feasibility checks:

- `step0_berka_ingestion/run_berka_ingestion.py`
- `step1_berka_weekly_builder/run_build_weekly.py`
- `step3_model_training_berka/train_classification.py`
- `step3_regression_training/train_regression.py`
- `reports/berka_feasibility/generate_feasibility_report.py`

Quick run:

```bash
python3 step0_berka_ingestion/run_berka_ingestion.py
python3 step1_berka_weekly_builder/run_build_weekly.py
python3 step3_model_training_berka/train_classification.py --quick
python3 step3_model_training_berka/train_classification.py --quick --target bucket_net_t_plus_1
python3 step3_regression_training/train_regression.py --quick
python3 step3_model_training_berka/generate_quick_vs_full_comparison.py
python3 reports/berka_feasibility/generate_feasibility_report.py
python3 reports/berka_feasibility/generate_dataset_analysis.py
```

See full instructions in `docs/berka_evaluation.md`.

Key Berka artifacts:

- `step3_model_training_berka/outputs/classification_report_full.json`
- `step3_model_training_berka/outputs/quick_vs_full_comparison.md`
- `step3_regression_training/outputs/regression_report_full.json`
- `reports/berka_feasibility/berka_dataset_analysis.md`

## Quick smoke tests

```bash
python3 smoke_test.py
python3 smoke_test_berka.py
```
