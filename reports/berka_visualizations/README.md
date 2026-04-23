# Berka Visualizations Pack

This pack visualizes bucket prediction quality for:
- RF base (release candidate)
- RF + blended calibrator (alpha after warmup)
- Persistence baseline (`bucket_t -> bucket_t+1`)

## What to open first
1. `confusion_rf.png`, `confusion_blended.png`, `confusion_persistence.png` - error structure and one-bucket collapse check.
2. `user_<id>_trace.png` - week-by-week behavior for specific users.
3. `transition_accuracy.png` - quality on transition vs stable weeks.
4. `confidence_curve.png` - confidence (pmax) vs accuracy relationship.
5. `per_user_f1_distribution.png` + `delta_f1_vs_history.png` - gain/loss distribution across users.

## Run
```bash
python3 reports/berka_visualizations/generate_visualizations.py --model-path models/release_candidate/full_spend_tuned_rf_model.pkl
```

## Main outputs
- `visualization_report.md`
- `predictions_sample.csv`
- PNG charts in the same folder
