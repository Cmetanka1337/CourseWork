# Step 3: Model Training & Comparison

This step trains and compares three classifiers for next-week spend bucket prediction:
- Random Forest (global baseline)
- Logistic Regression (personalization candidate)
- SGDClassifier (updatable alternative)

## Inputs
- `step2_feature_engineering/outputs/train_features_tier3.csv`
- `step2_feature_engineering/outputs/test_features_tier3.csv`

## Outputs
- `step3_model_training/outputs/*.pkl`
- `step3_model_training/outputs/*.json`
- `step3_model_training/outputs/*.csv`
- `step3_model_training/outputs/*.md`
- `step3_model_training/outputs/*.png`

## Run (full)
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training/run_model_training.py
```

## Run (quick smoke)
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training/run_model_training.py --quick
```

## Validate outputs
```zsh
python3 -u /Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training/test_model_training.py
```

