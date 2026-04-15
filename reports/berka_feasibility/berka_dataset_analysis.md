# Berka Dataset Analysis (Signal Suitability)

## 1) Coverage and weekly history
- Accounts: **4500**
- Weeks/account quantiles p10/p25/p50/p75/p90: **48.9 / 72.0 / 106.0 / 171.0 / 204.0**
- Short accounts share (<8w / <12w): **0.0000 / 0.0007**

## 2) Signal diagnostics for lag-based models (RF/LR)
- corr(outflow_t, outflow_t-1): **0.0944**
- corr(net_t, net_t-1): **-0.3303**
- corr(inflow_t-1, outflow_t+1): **0.2127**
- Outflow CV quantiles (p25/p50/p75): **1.0758 / 1.2058 / 1.3743**

Interpretation for RF suitability:
- Lag/rolling predictors can work when short-memory correlations are non-trivial.
- RF typically captures nonlinear interactions in engineered lag features, but does not model very long sequential dependencies natively.
- If temporal drift is strong, fold-wise metrics may degrade in late folds even with good average scores.

## 3) Regression sparsity and category structure
- Total categories: **12**
- Tail share categories with support < 50: **0.0000**
- Top categories by support:
- `outflow:vyber`: 218205
- `inflow:urok`: 172053
- `outflow:sluzby`: 151436
- `inflow:vklad`: 139666
- `outflow:sipo`: 114768
- `outflow:prevod na ucet`: 54617
- `inflow:prevod z uctu`: 34022
- `inflow:duchod`: 29598
- `outflow:pojistne`: 17990
- `outflow:uver`: 12898

## 4) Full-run model evidence snapshot
- Classification (spend target): baseline=0.1986, RF=0.4750, LR=0.4609, relative_gain=1.3924
- Classification (net target): baseline=0.2130, RF=0.4928, LR=0.4206, relative_gain=1.3134
- Regression full: baseline MAE=3933.46, Ridge=3530.51, SGD=3538.63, relative_improvement=0.1024

## 5) Methodological references
- Walk-forward / time-series CV rationale: avoid leakage and optimistic estimates in autocorrelated series (MDPI Sensors review context: https://www.mdpi.com/1424-8220/21/7/2430).
- Leakage from improper temporal features and preprocessing fit on full data: https://www.mhtechin.com/support/improper-temporal-feature-extraction-creating-future-leaks-the-core-challenge-in-time-series-machine-learning/
- Why persistence/naive baselines are mandatory in time series benchmarking: https://datascience.stackexchange.com/questions/130838/why-linear-regression-doing-well-in-time-series-data
