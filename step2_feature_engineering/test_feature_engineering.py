import json
from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("/Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/outputs")

    required_files = [
        output_dir / "train_features_engineered.csv",
        output_dir / "test_features_engineered.csv",
        output_dir / "train_features_tier3.csv",
        output_dir / "test_features_tier3.csv",
        output_dir / "feature_engineering_report.json",
        output_dir / "tier3_feature_engineering_report.json",
        output_dir / "feature_definitions.md",
        output_dir / "feature_definitions_tier3.md",
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise RuntimeError(f"Missing required output file: {file_path}")

    train_df = pd.read_csv(output_dir / "train_features_engineered.csv")
    test_df = pd.read_csv(output_dir / "test_features_engineered.csv")
    train_t3 = pd.read_csv(output_dir / "train_features_tier3.csv")
    test_t3 = pd.read_csv(output_dir / "test_features_tier3.csv")

    expected_cols = [
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
    ]

    if list(train_df.columns) != expected_cols:
        raise RuntimeError("train_features_engineered.csv schema mismatch")
    if list(test_df.columns) != expected_cols:
        raise RuntimeError("test_features_engineered.csv schema mismatch")

    if train_df.isnull().any().any():
        raise RuntimeError("NaN detected in train features")
    if test_df.isnull().any().any():
        raise RuntimeError("NaN detected in test features")

    tier3_extra = [
        "category_diversity",
        "dominant_category_ratio",
        "amount_t_minus_1",
        "amount_t_minus_2",
        "bucket_t_minus_1",
    ]
    expected_t3_cols = expected_cols + tier3_extra
    if list(train_t3.columns) != expected_t3_cols:
        raise RuntimeError("train_features_tier3.csv schema mismatch")
    if list(test_t3.columns) != expected_t3_cols:
        raise RuntimeError("test_features_tier3.csv schema mismatch")
    if train_t3.isnull().any().any() or test_t3.isnull().any().any():
        raise RuntimeError("NaN detected in Tier 3 features")
    if len(train_t3.columns) != 21 or len(test_t3.columns) != 21:
        raise RuntimeError("Tier 3 outputs must contain 21 columns")

    # Target class balance checks: distribution must be non-degenerate for Step 3 training.
    train_dist = train_t3["bucket_t_plus_1"].value_counts(normalize=True)
    test_dist = test_t3["bucket_t_plus_1"].value_counts(normalize=True)
    train_majority = float(train_dist.max())
    test_majority = float(test_dist.max())
    if train_majority >= 0.80:
        raise RuntimeError(f"Train majority class {train_majority:.1%} exceeds 80% threshold")
    if test_majority >= 0.80:
        raise RuntimeError(f"Test majority class {test_majority:.1%} exceeds 80% threshold")
    if int((train_dist > 0.01).sum()) < 3:
        raise RuntimeError("Train must have at least 3 classes with >1% support")
    if int((test_dist > 0.005).sum()) < 3:
        raise RuntimeError("Test must have at least 3 classes with >0.5% support")

    report = json.loads((output_dir / "feature_engineering_report.json").read_text(encoding="utf-8"))
    root = report["feature_engineering_report"]
    if "quality_checks" in root:
        verdict = root["quality_checks"]["nan_values"]
    else:
        verdict = root["tier3_feature_engineering_report"]["quality_checks"]["nan_values"]
    if verdict["train"] != 0 or verdict["test"] != 0:
        raise RuntimeError("Report says NaNs exist in engineered features")

    bucket_cfg = root.get("bucket_configuration")
    if not bucket_cfg:
        raise RuntimeError("Missing bucket_configuration in Step 2 report")
    if bucket_cfg.get("mode") not in {"user_quantile", "global_quantile"}:
        raise RuntimeError("Invalid bucket mode in report")
    if bucket_cfg.get("mode") == "user_quantile":
        uq_cfg = bucket_cfg.get("user_quantile_config", {})
        summary = uq_cfg.get("user_thresholds_summary")
        if not summary:
            raise RuntimeError("Missing user_thresholds_summary for user_quantile mode")
        if summary.get("fallback_count") is None:
            raise RuntimeError("Missing fallback_count in user_thresholds_summary")

    tier3_report = json.loads((output_dir / "tier3_feature_engineering_report.json").read_text(encoding="utf-8"))
    rows_match = tier3_report["tier3_feature_engineering_report"]["quality_checks"]["row_counts_match"]
    if not rows_match:
        raise RuntimeError("Tier 3 report indicates row count mismatch")

    print("Step 2 feature engineering validation passed")


if __name__ == "__main__":
    main()

