# On-device Calibrator

## Components

- `calibrator.py` - multiclass softmax calibrator over RF probabilities.
- `run_personalization_simulation.py` - per-user walk-forward simulator and evidence report generator.

## Quick run

```bash
python3 on_device_calibrator/run_personalization_simulation.py \
  --model-path step3_model_training_berka/outputs/full_spend_tuned_rf_model.pkl
```

Outputs:

- `reports/on_device_calibrator/calibrator_simulation_report.json`
- `reports/on_device_calibrator/calibrator_simulation_report.md`
- `reports/on_device_calibrator/delta_f1_hist.png`

