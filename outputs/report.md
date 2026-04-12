# Dataset Validation Report (Stage 0-10)

## Stage 0 - Sanity
- Rows: 1296675
- Users: 983
- Date range: 2019-01-01 .. 2020-06-21
- Avg tx/user: 1319.10
- Share users with >=200 tx: 0.924

## Stage 1 - Category collapse
- Raw unique categories: 14
- Collapsed unique categories: 7
- Collapse mode: manual

## Stage 3 - Sparsity
- Median active days %: 18.22
- Share <20%: 0.535
- Share 20-50%: 0.338
- Share >=50%: 0.127

## Stage 4 - Distribution
- Zero share: 0.760
- p95: 134.810
- p99: 349.270
- Skewness: 46.512

## Stage 5 - Autocorrelation
- Median corr lag1: 0.013
- Median corr lag7: 0.004
- Median corr lag30: -0.007
- Share lag7 < 0.1: 0.917

## Decision
- CASE 1: no_signal
- Forecasting is likely infeasible: weak autocorrelation or extreme sparsity.

## Stage 10 - Event-level analysis
- Pairs analyzed: 6881
- Median event rate: 0.182
- Median corr lag1: 0.020
- Median corr lag7: 0.026
- Median P(event_t=1 | event_t-1=0): 0.178
- Median P(event_t=1 | event_t-1=1): 0.222
- Always-0 accuracy: 0.760
- Frequency accuracy: 0.800
- Frequency balanced accuracy: 0.500