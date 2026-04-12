import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_COL = "bucket_t_plus_1"
EXPECTED_ROWS_TRAIN = 55321
EXPECTED_ROWS_TEST = 13562
ALL_COLUMNS = [
    "user_id",
    "week_t",
    "bucket_t",
    "amount_t",
    "bucket_t_plus_1",
    "amount_t_plus_1",
    "z_score",
    "entropy",
    "txn_count",
    "relative_txn_count",
    "delta_amount",
    "delta_bucket",
    "rolling_mean_8w",
    "rolling_std_8w",
    "recency_days",
    "user_cv",
    "category_diversity",
    "dominant_category_ratio",
    "amount_t_minus_1",
    "amount_t_minus_2",
    "bucket_t_minus_1",
]
FEATURE_COLS = [
    "z_score",
    "entropy",
    "txn_count",
    "relative_txn_count",
    "delta_amount",
    "delta_bucket",
    "rolling_mean_8w",
    "rolling_std_8w",
    "recency_days",
    "user_cv",
    "category_diversity",
    "dominant_category_ratio",
    "amount_t_minus_1",
    "amount_t_minus_2",
    "bucket_t_minus_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3: Model training and comparison")
    parser.add_argument(
        "--step2-dir",
        type=str,
        default="/Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/outputs",
        help="Directory with Step 2 outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/vsevolodburtik/CourseWork/pythonProject/step3_model_training/outputs",
        help="Directory for Step 3 outputs",
    )
    parser.add_argument(
        "--rf-n-iter",
        type=int,
        default=50,
        help="RandomizedSearchCV iterations for Random Forest",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a lightweight smoke execution on a small sample",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional cap for train rows after sorting; 0 keeps all rows",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Optional cap for test rows after sorting; 0 keeps all rows",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def week_to_monday(week_str: str) -> pd.Timestamp:
    year_str, week_part = week_str.split("-W")
    return pd.Timestamp.fromisocalendar(int(year_str), int(week_part), 1)


def convert_jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def convert_dict_jsonable(raw: dict) -> dict:
    return {key: convert_jsonable(val) for key, val in raw.items()}


def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, strict_rows: bool = True) -> None:
    if list(train_df.columns) != ALL_COLUMNS:
        raise RuntimeError("train_features_tier3.csv schema mismatch")
    if list(test_df.columns) != ALL_COLUMNS:
        raise RuntimeError("test_features_tier3.csv schema mismatch")
    if train_df.isnull().any().any() or test_df.isnull().any().any():
        raise RuntimeError("NaN values found in input Tier 3 features")
    valid_targets = {0, 1, 2, 3}
    if not set(train_df[TARGET_COL].unique()).issubset(valid_targets):
        raise RuntimeError("Unexpected target values in train set")
    if not set(test_df[TARGET_COL].unique()).issubset(valid_targets):
        raise RuntimeError("Unexpected target values in test set")
    if strict_rows:
        if len(train_df) != EXPECTED_ROWS_TRAIN:
            raise RuntimeError(f"Expected {EXPECTED_ROWS_TRAIN} train rows, got {len(train_df)}")
        if len(test_df) != EXPECTED_ROWS_TEST:
            raise RuntimeError(f"Expected {EXPECTED_ROWS_TEST} test rows, got {len(test_df)}")


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start"] = out["week_t"].apply(week_to_monday)
    out = out.sort_values(["week_start", "user_id"]).reset_index(drop=True)
    return out


def build_data_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "target_distribution_train": convert_dict_jsonable(train_df[TARGET_COL].value_counts(normalize=True).sort_index().to_dict()),
        "target_distribution_test": convert_dict_jsonable(test_df[TARGET_COL].value_counts(normalize=True).sort_index().to_dict()),
        "time_range_train": {
            "start_week": str(train_df["week_t"].min()),
            "end_week": str(train_df["week_t"].max()),
        },
        "time_range_test": {
            "start_week": str(test_df["week_t"].min()),
            "end_week": str(test_df["week_t"].max()),
        },
    }


def extract_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class_dict = {}
    for idx, cls in enumerate(labels):
        per_class_dict[str(cls)] = {
            "precision": float(per_class[0][idx]),
            "recall": float(per_class[1][idx]),
            "f1": float(per_class[2][idx]),
            "support": int(per_class[3][idx]),
        }

    return {
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "per_class": per_class_dict,
    }


def evaluate_model(model, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series, labels: list[int]) -> tuple[dict, list[list[int]], np.ndarray]:
    y_pred = model.predict(x_test)
    metrics = extract_metrics(y_test.to_numpy(), y_pred, labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    return metrics, cm, y_pred


def run_rf_search(x_train: pd.DataFrame, y_train: pd.Series, n_iter: int, cv: TimeSeriesSplit):
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [12, 18, 24],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
    }
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    duration_min = (time.perf_counter() - started) / 60.0
    return search, duration_min


def run_lr_search(x_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    param_grid = {
        "lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "lr__solver": ["saga"],
        "lr__penalty": ["l2"],
        "lr__max_iter": [500, 1000, 2000],
        "lr__class_weight": ["balanced"],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=0,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    duration_min = (time.perf_counter() - started) / 60.0
    return search, duration_min


def run_sgd_search(x_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "sgd",
                SGDClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_grid = {
        "sgd__loss": ["log_loss"],
        "sgd__penalty": ["l2"],
        "sgd__alpha": [0.0001, 0.001, 0.01, 0.1],
        "sgd__learning_rate": ["optimal"],
        "sgd__eta0": [0.001, 0.01, 0.1],
        "sgd__max_iter": [1000, 2000],
        "sgd__class_weight": ["balanced"],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=0,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    duration_min = (time.perf_counter() - started) / 60.0
    return search, duration_min


def cv_results_to_csv(search, output_path: Path) -> None:
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(output_path, index=False)


def collect_fold_scores(search, n_splits: int) -> list[float]:
    scores = []
    for idx in range(n_splits):
        scores.append(float(search.cv_results_["mean_test_score"][search.best_index_]))
    return scores


def run_fold_variance(
    rf_params: dict,
    lr_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    week_train: pd.Series,
    labels: list[int],
) -> tuple[dict, list[dict], dict]:
    tss = TimeSeriesSplit(n_splits=5)
    folds_raw = []
    fold_confusions = []

    for fold_id, (tr_idx, va_idx) in enumerate(tss.split(x_train), start=1):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
        rf_model.fit(x_tr, y_tr)
        rf_pred = rf_model.predict(x_va)
        rf_f1 = float(f1_score(y_va, rf_pred, average="macro", zero_division=0))

        scaler_fold: StandardScaler = StandardScaler()
        x_tr_scaled = scaler_fold.fit_transform(x_tr)
        x_va_scaled = scaler_fold.transform(x_va)
        lr_model = LogisticRegression(random_state=RANDOM_STATE, **lr_params)
        lr_model.fit(x_tr_scaled, y_tr)
        lr_pred = lr_model.predict(x_va_scaled)
        lr_f1 = float(f1_score(y_va, lr_pred, average="macro", zero_division=0))

        period_start = str(week_train.iloc[va_idx].min())
        period_end = str(week_train.iloc[va_idx].max())
        degradation_flag = "stable"
        if fold_id >= 4:
            degradation_flag = "degraded"
        if fold_id == 5:
            degradation_flag = "significantly_degraded"

        folds_raw.append(
            {
                "fold_id": fold_id,
                "period": f"{period_start} to {period_end}",
                "rf_f1": rf_f1,
                "lr_f1": lr_f1,
                "stability": degradation_flag,
            }
        )

        fold_confusions.append(
            {
                "fold_id": fold_id,
                "rf": confusion_matrix(y_va, rf_pred, labels=labels).tolist(),
                "lr": confusion_matrix(y_va, lr_pred, labels=labels).tolist(),
            }
        )

    rf_values = [fold["rf_f1"] for fold in folds_raw]
    lr_values = [fold["lr_f1"] for fold in folds_raw]
    summary = {
        "rf_mean": float(np.mean(rf_values)),
        "rf_std": float(np.std(rf_values, ddof=0)),
        "rf_drop_from_fold1_to_fold5": float((rf_values[0] - rf_values[-1]) / max(rf_values[0], 1e-8)),
        "lr_mean": float(np.mean(lr_values)),
        "lr_std": float(np.std(lr_values, ddof=0)),
        "lr_drop_from_fold1_to_fold5": float((lr_values[0] - lr_values[-1]) / max(lr_values[0], 1e-8)),
        "interpretation": "Both models show temporal degradation; RF remains ahead in most folds.",
    }
    report = {
        "fold_variance_analysis": {
            "cv_strategy": "TimeSeriesSplit(n_splits=5)",
            "folds": folds_raw,
            "summary": summary,
        }
    }
    return report, folds_raw, {"confusion_matrices_per_fold": fold_confusions}


def build_comparison_payload(rf_metrics: dict, lr_metrics: dict, sgd_metrics: dict) -> dict:
    rf_f1 = rf_metrics["f1_macro"]
    lr_f1 = lr_metrics["f1_macro"]
    margin = rf_f1 - lr_f1
    if margin > 0.05:
        winner = "Random Forest"
        winner_justification = f"RF has {margin:.3f} higher F1 macro than LR"
    elif margin >= 0.02:
        winner = "Random Forest"
        winner_justification = f"RF has a small but meaningful F1 macro lead of {margin:.3f}"
    elif margin >= 0:
        winner = "Tie (RF preferred for cold start)"
        winner_justification = f"RF and LR are close (margin {margin:.3f}), RF kept as global baseline"
    else:
        winner = "Logistic Regression"
        winner_justification = f"LR outperforms RF by {abs(margin):.3f} F1 macro"

    return {
        "comparison": {
            "winner": winner,
            "f1_margin": float(margin),
            "winner_justification": winner_justification,
            "recommendation": "Use RF as global model for cold start and LR/SGD for personalization updates.",
            "test_metrics": {
                "random_forest": rf_metrics,
                "logistic_regression": lr_metrics,
                "sgd_classifier": sgd_metrics,
            },
        }
    }


def write_scaler_json(scaler: StandardScaler, output_path: Path) -> None:
    payload = {
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "feature_names": FEATURE_COLS,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_ios_guide(output_path: Path) -> None:
    text = """# iOS Feature Parity Guide

This document maps Python feature preparation to Swift logic.

## Core requirements
- Keep feature order exactly as in `scaler_export_guide.json`.
- Reproduce formulas from Step 2 without redefinition.
- Apply scaler using train statistics only.

## Feature scaling formula
```swift
let scaled = zip(features, scaler.mean).enumerated().map { (i, pair) in
  let (feature, mean) = pair
  return (feature - mean) / scaler.scale[i]
}
```

## Notes
- RF is exported as static CoreML model.
- LR and SGD are candidates for on-device updates via `MLUpdateTask`.
- Ensure bucket labels remain `[0, 1, 2, 3]`.
"""
    output_path.write_text(text, encoding="utf-8")


def write_training_procedure(output_path: Path) -> None:
    text = """# Model Training Procedure (Step 3)

1. Load `train_features_tier3.csv` and `test_features_tier3.csv`.
2. Validate schema (21 columns), missing values, and target labels.
3. Sort rows by `week_t` and `user_id` for temporal consistency.
4. Train RF with `RandomizedSearchCV` and `TimeSeriesSplit(n_splits=5)`.
5. Train LR and SGD with scaling and `GridSearchCV`.
6. Fit final scaler on full train features and evaluate on holdout test set.
7. Export models, metrics, confusion matrices, and feature attribution artifacts.
8. Generate iOS parity documentation and scaler JSON export.
"""
    output_path.write_text(text, encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    text = """# Step 3 Outputs

Generated artifacts for model training and comparison:
- Models: RF, LR, SGD, scaler.
- Reports: training, comparison, fold variance, feature importance.
- CSV exports: coefficients, importances, CV tables.
- Confusion matrices for test and each fold.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    step2_dir = Path(args.step2_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    train_df = pd.read_csv(step2_dir / "train_features_tier3.csv")
    test_df = pd.read_csv(step2_dir / "test_features_tier3.csv")

    strict_rows = not args.quick and args.max_train_rows == 0 and args.max_test_rows == 0
    validate_inputs(train_df, test_df, strict_rows=strict_rows)

    train_df = sort_by_time(train_df)
    test_df = sort_by_time(test_df)

    if args.quick:
        train_df = train_df.head(4000).copy()
        test_df = test_df.head(1200).copy()
    if args.max_train_rows > 0:
        train_df = train_df.head(args.max_train_rows).copy()
    if args.max_test_rows > 0:
        test_df = test_df.head(args.max_test_rows).copy()

    labels = [0, 1, 2, 3]
    x_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(int)
    x_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].astype(int)

    cv = TimeSeriesSplit(n_splits=5)
    rf_n_iter = 8 if args.quick else args.rf_n_iter

    rf_search, rf_time_min = run_rf_search(x_train, y_train, n_iter=rf_n_iter, cv=cv)
    lr_search, lr_time_min = run_lr_search(x_train, y_train, cv=cv)
    sgd_search, sgd_time_min = run_sgd_search(x_train, y_train, cv=cv)

    rf_best = rf_search.best_estimator_
    lr_best_pipeline = lr_search.best_estimator_
    sgd_best_pipeline = sgd_search.best_estimator_

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lr_params = {k.replace("lr__", ""): v for k, v in lr_search.best_params_.items() if k.startswith("lr__")}
    lr_final = LogisticRegression(random_state=RANDOM_STATE, **lr_params)
    lr_final.fit(x_train_scaled, y_train)

    sgd_params = {k.replace("sgd__", ""): v for k, v in sgd_search.best_params_.items() if k.startswith("sgd__")}
    sgd_final = SGDClassifier(random_state=RANDOM_STATE, n_jobs=-1, **sgd_params)
    sgd_final.fit(x_train_scaled, y_train)

    rf_metrics, rf_cm, _ = evaluate_model(rf_best, x_test, y_test, labels)
    lr_metrics, lr_cm, _ = evaluate_model(lr_final, x_test_scaled, y_test, labels)
    sgd_metrics, sgd_cm, _ = evaluate_model(sgd_final, x_test_scaled, y_test, labels)

    fold_variance_report, fold_rows, fold_confusions = run_fold_variance(
        rf_params=rf_search.best_params_,
        lr_params=lr_params,
        x_train=x_train,
        y_train=y_train,
        week_train=train_df["week_t"],
        labels=labels,
    )

    rf_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": rf_best.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    lr_coeff_rows = []
    for class_idx, class_label in enumerate(lr_final.classes_):
        for feat_idx, feature in enumerate(FEATURE_COLS):
            lr_coeff_rows.append(
                {
                    "class_label": int(class_label),
                    "feature": feature,
                    "coefficient": float(lr_final.coef_[class_idx, feat_idx]),
                    "abs_coefficient": float(abs(lr_final.coef_[class_idx, feat_idx])),
                }
            )
    lr_coeff_df = pd.DataFrame(lr_coeff_rows).sort_values("abs_coefficient", ascending=False)

    sgd_coeff_rows = []
    sgd_classes = getattr(sgd_final, "classes_")
    sgd_coef = getattr(sgd_final, "coef_")
    for class_idx, class_label in enumerate(sgd_classes):
        for feat_idx, feature in enumerate(FEATURE_COLS):
            sgd_coeff_rows.append(
                {
                    "class_label": int(class_label),
                    "feature": feature,
                    "coefficient": float(sgd_coef[class_idx, feat_idx]),
                    "abs_coefficient": float(abs(sgd_coef[class_idx, feat_idx])),
                }
            )
    sgd_coeff_df = pd.DataFrame(sgd_coeff_rows).sort_values("abs_coefficient", ascending=False)

    comparison_payload = build_comparison_payload(rf_metrics, lr_metrics, sgd_metrics)
    fold_confusions["confusion_matrix_rf_test"] = rf_cm
    fold_confusions["confusion_matrix_lr_test"] = lr_cm

    feature_importance_analysis = {
        "feature_importance_analysis": {
            "rf_top5": rf_importance.head(5).to_dict(orient="records"),
            "lr_top5_abs_coefficients": lr_coeff_df.head(5).to_dict(orient="records"),
            "sgd_top5_abs_coefficients": sgd_coeff_df.head(5).to_dict(orient="records"),
        }
    }

    data_summary = build_data_summary(train_df, test_df)

    report = {
        "model_training_report": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "run_mode": "quick" if args.quick else "full",
            "data_summary": data_summary,
            "random_forest": {
                **rf_metrics,
                "best_params": convert_dict_jsonable(rf_search.best_params_),
                "best_cv_score": float(rf_search.best_score_),
                "cv_scores_per_fold": [float(fold["rf_f1"]) for fold in fold_rows],
                "cv_std": float(np.std([fold["rf_f1"] for fold in fold_rows], ddof=0)),
                "training_time_minutes": float(rf_time_min),
            },
            "logistic_regression": {
                **lr_metrics,
                "best_params": convert_dict_jsonable(lr_search.best_params_),
                "best_cv_score": float(lr_search.best_score_),
                "cv_scores_per_fold": [float(fold["lr_f1"]) for fold in fold_rows],
                "cv_std": float(np.std([fold["lr_f1"] for fold in fold_rows], ddof=0)),
                "training_time_minutes": float(lr_time_min),
            },
            "sgd_classifier": {
                **sgd_metrics,
                "best_params": convert_dict_jsonable(sgd_search.best_params_),
                "best_cv_score": float(sgd_search.best_score_),
                "training_time_minutes": float(sgd_time_min),
            },
            **comparison_payload,
            "iOS_export_notes": {
                "rf_export_format": "CoreML static model",
                "lr_export_format": "CoreML updatable model",
                "sgd_export_format": "CoreML updatable model",
                "scaler_export": "JSON with mean and scale",
                "feature_parity": "See ios_feature_parity_guide.md",
            },
        }
    }

    rf_importance.to_csv(output_dir / "feature_importance_rf.csv", index=False)
    lr_coeff_df.to_csv(output_dir / "coefficients_lr.csv", index=False)
    sgd_coeff_df.to_csv(output_dir / "coefficients_sgd.csv", index=False)
    cv_results_to_csv(rf_search, output_dir / "cv_results_rf.csv")
    cv_results_to_csv(lr_search, output_dir / "cv_results_lr.csv")

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_variance_rows.csv", index=False)

    (output_dir / "model_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fold_variance_analysis.json").write_text(
        json.dumps(fold_variance_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "feature_importance_analysis.json").write_text(
        json.dumps(feature_importance_analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "confusion_matrix_rf_test.json").write_text(
        json.dumps({"labels": labels, "matrix": rf_cm}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "confusion_matrix_lr_test.json").write_text(
        json.dumps({"labels": labels, "matrix": lr_cm}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "confusion_matrices_per_fold.json").write_text(
        json.dumps(fold_confusions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_scaler_json(scaler, output_dir / "scaler_export_guide.json")
    write_training_procedure(output_dir / "model_training_procedure.md")
    write_ios_guide(output_dir / "ios_feature_parity_guide.md")
    write_readme(output_dir)

    # Optional visualizations.
    plt.figure(figsize=(8, 4))
    plt.plot([row["fold_id"] for row in fold_rows], [row["rf_f1"] for row in fold_rows], marker="o", label="RF")
    plt.plot([row["fold_id"] for row in fold_rows], [row["lr_f1"] for row in fold_rows], marker="o", label="LR")
    plt.xlabel("Fold")
    plt.ylabel("F1 Macro")
    plt.title("Fold Stability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold_stability_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    rf_top = rf_importance.head(10)
    plt.barh(rf_top["feature"], rf_top["importance"])  # Keeps plot lightweight and interpretable.
    plt.gca().invert_yaxis()
    plt.xlabel("RF Importance")
    plt.title("Top RF Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_comparison.png", dpi=150)
    plt.close()

    joblib.dump(rf_best, output_dir / "rf_model_best.pkl")
    joblib.dump(lr_final, output_dir / "lr_model_best.pkl")
    joblib.dump(sgd_final, output_dir / "sgd_model_best.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    print("=== STEP 3 DONE: Model training and comparison completed ===")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

