# On-device Calibrator Simulation (Berka spend bucket)

Generated: 2026-04-22T20:25:38.791304+00:00

## Setup
- Base model: `/Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training_berka/outputs/full_spend_tuned_rf_model.pkl`
- Target: `bucket_spend_t_plus_1`
- Walk-forward: per-user chronological simulation, no future labels in updates.
- Warm-up: first `8` weeks (collect only).
- Update cadence: every `2` weeks.
- Update data: cumulative per-user labeled buffer capped at `20` examples.

## Overall metrics on test timeline
| variant | f1_macro | balanced_accuracy |
| --- | ---: | ---: |
| RF base | 0.4878 | 0.5664 |
| RF + calibrator | 0.5230 | 0.5579 |
| RF + blended | 0.5005 | 0.5756 |

## Per-user delta (calibrated - base, macro F1)
- Improved users: **1730**
- Worsened users: **1446**
- Unchanged users: **1300**
- Delta quantiles p10/p25/p50/p75/p90: **-0.1685 / -0.0417 / 0.0000 / 0.0595 / 0.1354**
- Histogram: `/Users/vsevolodburtik/CourseWork/pythonProject/reports/on_device_calibrator/delta_f1_hist.png`

## Small-data stress test (users with 12-20 total weeks)
- Users: **14**
- Base F1 / balanced accuracy: **0.3631 / 0.4487**
- Calibrated F1 / balanced accuracy: **0.1818 / 0.3077**

## Time-to-benefit
- Users reaching benefit: **1902 / 4476**
- Median updates until benefit: **49.0**
- Median weeks until benefit: **3.0**
- Note: `weeks_until_benefit` counts the elapsed weeks between the first eligible update and the update where benefit is first observed, using the configured update cadence.

## Swift port spec
- Formula: `p_adj = softmax(W * p_rf + b)` with `W` shape `4x4`, `b` shape `4`.
- Initialization: `W = I`, `b = 0` (identity behavior before updates).
- SGD update:
  - `logits = W * p_rf + b`
  - `p_adj = softmax(logits)`
  - `grad_logits = p_adj - one_hot(y)`
  - `grad_W = grad_logits outer p_rf + l2 * W`
  - `grad_b = grad_logits`
  - clip global grad norm to `clip`
  - `W -= lr * grad_W`, `b -= lr * grad_b`
- Recommended defaults:
  - `lr = 0.05`
  - `l2 = 0.001`
  - `clip = 5.0`
  - update cadence: every `2` weeks
  - warm-up: `8` weeks
  - history cap: `20` examples
- On-device state to store:
  - calibrator params `W`, `b`
  - per-user ring buffer of `(p_rf, y_true, week_idx)` up to `K=20`
  - counters: weeks since last update, updates count
