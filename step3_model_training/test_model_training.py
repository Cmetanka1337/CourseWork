import json
from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("/Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training/outputs")

    required_files = [
        output_dir / "rf_model_best.pkl",
        output_dir / "lr_model_best.pkl",
        output_dir / "sgd_model_best.pkl",
        output_dir / "scaler.pkl",
        output_dir / "model_training_report.json",
        output_dir / "comparison_metrics.json",
        output_dir / "fold_variance_analysis.json",
        output_dir / "feature_importance_analysis.json",
        output_dir / "feature_importance_rf.csv",
        output_dir / "coefficients_lr.csv",
        output_dir / "coefficients_sgd.csv",
        output_dir / "cv_results_rf.csv",
        output_dir / "cv_results_lr.csv",
        output_dir / "confusion_matrix_rf_test.json",
        output_dir / "confusion_matrix_lr_test.json",
        output_dir / "confusion_matrices_per_fold.json",
        output_dir / "model_training_procedure.md",
        output_dir / "ios_feature_parity_guide.md",
        output_dir / "scaler_export_guide.json",
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise RuntimeError(f"Missing required output file: {file_path}")

    rf_imp = pd.read_csv(output_dir / "feature_importance_rf.csv")
    lr_coef = pd.read_csv(output_dir / "coefficients_lr.csv")
    sgd_coef = pd.read_csv(output_dir / "coefficients_sgd.csv")

    if list(rf_imp.columns) != ["feature", "importance"]:
        raise RuntimeError("feature_importance_rf.csv schema mismatch")
    if not {"class_label", "feature", "coefficient", "abs_coefficient"}.issubset(lr_coef.columns):
        raise RuntimeError("coefficients_lr.csv schema mismatch")
    if not {"class_label", "feature", "coefficient", "abs_coefficient"}.issubset(sgd_coef.columns):
        raise RuntimeError("coefficients_sgd.csv schema mismatch")

    if len(rf_imp) != 15:
        raise RuntimeError("RF importances must contain exactly 15 rows")

    report = json.loads((output_dir / "model_training_report.json").read_text(encoding="utf-8"))
    root = report["model_training_report"]
    for model_key in ["random_forest", "logistic_regression", "sgd_classifier"]:
        metrics = root[model_key]
        for metric_name in ["f1_macro", "accuracy", "precision_macro", "recall_macro"]:
            val = metrics[metric_name]
            if not (0.0 <= float(val) <= 1.0):
                raise RuntimeError(f"{model_key}.{metric_name} out of [0,1]")

    fold_report = json.loads((output_dir / "fold_variance_analysis.json").read_text(encoding="utf-8"))
    folds = fold_report["fold_variance_analysis"]["folds"]
    if len(folds) != 5:
        raise RuntimeError("Expected exactly 5 folds in variance analysis")

    cm_rf = json.loads((output_dir / "confusion_matrix_rf_test.json").read_text(encoding="utf-8"))
    cm_lr = json.loads((output_dir / "confusion_matrix_lr_test.json").read_text(encoding="utf-8"))
    for cm in [cm_rf, cm_lr]:
        matrix = cm["matrix"]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise RuntimeError("Confusion matrix must be 4x4")

    scaler_export = json.loads((output_dir / "scaler_export_guide.json").read_text(encoding="utf-8"))
    if len(scaler_export["feature_names"]) != 15:
        raise RuntimeError("Scaler export must contain 15 feature names")
    if len(scaler_export["mean"]) != 15 or len(scaler_export["scale"]) != 15:
        raise RuntimeError("Scaler export vectors must contain 15 values")

    print("Step 3 model training validation passed")


if __name__ == "__main__":
    main()

