# Berka Feature Passport: Spend Bucket Classification

Model scope: `bucket_spend_t_plus_1` classification in `step3_model_training_berka/train_classification.py`.

## Launch threshold

- Minimum history before production inference: **8 completed weekly points per `user_id`**.
- Rationale: 8-week rolling and regularity features (`*_rolling_*_8w`, `*_frequency_8w`, `weeks_since_*`) become meaningfully stable only after at least one full 8-week window.
- Current pipeline remains trainable with shorter history because missing/early windows are zero-filled; this threshold is a deployment guardrail.

## Feature order (exact model input order)

The model expects features in this exact order:

1. `bucket_spend_t`
2. `bucket_net_t`
3. `weekly_inflow_t`
4. `weekly_outflow_t`
5. `weekly_net_t`
6. `txn_count_t`
7. `category_diversity_t`
8. `weekly_inflow_t_minus_1`
9. `weekly_outflow_t_minus_1`
10. `weekly_net_t_minus_1`
11. `weekly_inflow_t_minus_2`
12. `weekly_outflow_t_minus_2`
13. `outflow_inflow_ratio_t`
14. `week_of_year`
15. `month`
16. `quarter`
17. `week_of_month`
18. `is_month_start_week`
19. `is_month_end_week`
20. `delta_inflow`
21. `delta_outflow`
22. `inflow_outflow_ratio`
23. `inflow_share`
24. `inflow_rolling_mean_8w`
25. `inflow_rolling_std_8w`
26. `outflow_rolling_mean_8w`
27. `outflow_rolling_std_8w`
28. `inflow_frequency_8w`
29. `outflow_frequency_8w`
30. `weeks_since_inflow`
31. `weeks_since_outflow`

## Feature specification

Notation per user-week `t`:
- `inflow_t = weekly_inflow_t`
- `outflow_t = weekly_outflow_t`
- `net_t = inflow_t - outflow_t`
- `eps = 1e-6`
- train-only bucket quantiles: `q25_spend`, `q75_spend`, `q25_net`, `q75_net`

| # | Feature | Formula / definition | Missing / clipping policy |
| --- | --- | --- | --- |
| 1 | `bucket_spend_t` | `0 if outflow_t == 0 else 1 if outflow_t <= q25_spend else 2 if outflow_t <= q75_spend else 3` | Integer bucket, no fill expected |
| 2 | `bucket_net_t` | `0 if net_t == 0 else 1 if net_t <= q25_net else 2 if net_t <= q75_net else 3` | Integer bucket, no fill expected |
| 3 | `weekly_inflow_t` | Weekly sum of inflow amounts for user-week | From aggregation; no fill expected |
| 4 | `weekly_outflow_t` | Weekly sum of outflow amounts for user-week | From aggregation; no fill expected |
| 5 | `weekly_net_t` | `weekly_inflow_t - weekly_outflow_t` | Deterministic; no fill expected |
| 6 | `txn_count_t` | Count of transactions in user-week | From aggregation; no fill expected |
| 7 | `category_diversity_t` | Number of unique categories in user-week | From aggregation; no fill expected |
| 8 | `weekly_inflow_t_minus_1` | `shift(weekly_inflow_t, 1)` per user | `fillna(0.0)` |
| 9 | `weekly_outflow_t_minus_1` | `shift(weekly_outflow_t, 1)` per user | `fillna(0.0)` |
| 10 | `weekly_net_t_minus_1` | `shift(weekly_net_t, 1)` per user | `fillna(0.0)` |
| 11 | `weekly_inflow_t_minus_2` | `shift(weekly_inflow_t, 2)` per user | `fillna(0.0)` |
| 12 | `weekly_outflow_t_minus_2` | `shift(weekly_outflow_t, 2)` per user | `fillna(0.0)` |
| 13 | `outflow_inflow_ratio_t` | `outflow_t / inflow_t` | If `inflow_t == 0` -> NaN -> `0.0`; inf -> NaN -> `0.0`; no extra clipping |
| 14 | `week_of_year` | ISO week number from `week_start` Monday | Integer calendar value |
| 15 | `month` | Month from `week_start` | Integer calendar value |
| 16 | `quarter` | Quarter from `week_start` | Integer calendar value |
| 17 | `week_of_month` | `1 + floor((day(week_start)-1)/7)` | Integer calendar value |
| 18 | `is_month_start_week` | `1 if day(week_start) <= 7 else 0` | Binary calendar flag |
| 19 | `is_month_end_week` | `1 if week_end is within last 7 days of month else 0` (`week_end = week_start + 6d`) | Binary calendar flag |
| 20 | `delta_inflow` | `weekly_inflow_t - weekly_inflow_t_minus_1` | NaN/inf -> `0.0` |
| 21 | `delta_outflow` | `weekly_outflow_t - weekly_outflow_t_minus_1` | NaN/inf -> `0.0` |
| 22 | `inflow_outflow_ratio` | `weekly_inflow_t / (weekly_outflow_t + eps)` | Clipped to `[0.0, 10.0]`; then NaN/inf -> `0.0` |
| 23 | `inflow_share` | `weekly_inflow_t / (weekly_inflow_t + weekly_outflow_t + eps)` | Clipped to `[0.0, 1.0]`; then NaN/inf -> `0.0` |
| 24 | `inflow_rolling_mean_8w` | `rolling_mean_8(shift(weekly_inflow_t,1))` per user | NaN/inf -> `0.0` |
| 25 | `inflow_rolling_std_8w` | `rolling_std_8(shift(weekly_inflow_t,1))` per user | NaN/inf -> `0.0` |
| 26 | `outflow_rolling_mean_8w` | `rolling_mean_8(shift(weekly_outflow_t,1))` per user | NaN/inf -> `0.0` |
| 27 | `outflow_rolling_std_8w` | `rolling_std_8(shift(weekly_outflow_t,1))` per user | NaN/inf -> `0.0` |
| 28 | `inflow_frequency_8w` | `rolling_mean_8( I(shift(weekly_inflow_t,1) > 0) )` | NaN/inf -> `0.0` |
| 29 | `outflow_frequency_8w` | `rolling_mean_8( I(shift(weekly_outflow_t,1) > 0) )` | NaN/inf -> `0.0` |
| 30 | `weeks_since_inflow` | Weeks since last positive inflow based on `shift(weekly_inflow_t,1)` | Capped at `52`; initial values set to `52`; then NaN/inf -> `0.0` |
| 31 | `weeks_since_outflow` | Weeks since last positive outflow based on `shift(weekly_outflow_t,1)` | Capped at `52`; initial values set to `52`; then NaN/inf -> `0.0` |

## Global fill and validation rules

- Columns with lag/rolling/regularity/ratio dynamics are normalized with: `replace([inf, -inf], NaN).fillna(0.0)`.
- Final safety check in builder rejects output if any NaN remains in classification lag features.
- All rolling and regularity features are leakage-safe (`shift(1)` before rolling computations).

