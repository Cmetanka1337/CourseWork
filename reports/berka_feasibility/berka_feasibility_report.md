# Berka Feasibility Report

Generated: 2026-04-13T12:11:15.918317+00:00

## Signal checks
- Accounts: **4500**
- Weeks/account p25-p50-p75: **64.0 / 98.0 / 164.0**
- corr(outflow_t, outflow_t-1): **0.0953**
- corr(inflow_t-1, outflow_t+1): **0.2148**

## Bucket classification feasibility
- Verdict: **GO**
- Best model F1_macro gain vs persistence: **0.2609**
- Baseline persistence F1_macro: **0.1986**
- RF/LR/SGD test F1_macro: **0.4595 / 0.3635 / 0.3869**

## Regression feasibility (single model, multi-category)
- Verdict: **GO**
- Relative MAE improvement vs persistence: **0.1024**
- Baseline MAE (persistence): **3933.4632**
- Ridge/ElasticNet/SGD MAE: **3530.5118 / 3530.6072 / 3538.6338**

## Methodological checks
- Time-based validation: **TimeSeriesSplit / holdout last weeks**
- Leakage policy: **lags only from past, fit preprocessors on train only**
- Baselines included: **persistence + majority (classification), persistence + rolling mean (regression)**
