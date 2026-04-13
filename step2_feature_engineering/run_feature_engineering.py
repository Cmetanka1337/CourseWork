import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: Feature engineering (Tier 1 + Tier 2)")
    parser.add_argument(
        "--step1-dir",
        type=str,
        default="/Users/vsevolodburtik/CourseWork/pythonProject/step1_validation/outputs",
        help="Directory with Step 1 outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/vsevolodburtik/CourseWork/pythonProject/step2_feature_engineering/outputs",
        help="Directory for Step 2 outputs",
    )
    parser.add_argument(
        "--bucket-mode",
        type=str,
        choices=["user_quantile", "global_quantile"],
        default="user_quantile",
        help=(
            "user_quantile: per-user Q25/Q75 from train only; "
            "global_quantile: legacy single Q25/Q75 from train"
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_year_week_to_monday(week_str: str) -> pd.Timestamp:
    year_str, week_part = week_str.split("-W")
    return pd.Timestamp.fromisocalendar(int(year_str), int(week_part), 1)


def _apply_bucket(series: pd.Series, q25: float, q75: float) -> pd.Series:
    buckets = pd.cut(
        series,
        bins=[0.0, float(q25), float(q75), float("inf")],
        labels=[1, 2, 3],
        include_lowest=True,
    )
    if buckets.isna().any():
        # Guard against unexpected values (for example negatives); map them to lowest bucket.
        buckets = buckets.fillna(1)
    return buckets.astype(int)


def compute_user_specific_buckets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "amount_t_plus_1",
    output_col: str = "bucket_t_plus_1",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    """
    Compute per-user Q25/Q75 from train only and apply to train/test.
    Uses global train quantiles as fallback for sparse users.
    """
    df_train_out = df_train.copy()
    df_test_out = df_test.copy()
    user_thresholds: list[dict] = []

    global_q25 = float(df_train[target_col].quantile(0.25))
    global_q75 = float(df_train[target_col].quantile(0.75))
    if global_q25 == global_q75:
        eps_g = max(1e-6, abs(global_q25) * 0.01)
        global_q25 -= eps_g
        global_q75 += eps_g

    fallback_count = 0
    train_users = df_train["user_id"].drop_duplicates().tolist()
    for user_id in train_users:
        user_train = df_train[df_train["user_id"] == user_id]
        use_fallback = len(user_train) < 4
        if use_fallback:
            q25 = global_q25
            q75 = global_q75
            fallback_count += 1
        else:
            q25 = float(user_train[target_col].quantile(0.25))
            q75 = float(user_train[target_col].quantile(0.75))
            if q25 == q75:
                eps = max(1e-6, abs(q25) * 0.01)
                q25 -= eps
                q75 += eps

        user_thresholds.append(
            {
                "user_id": int(user_id),
                "q25": float(q25),
                "q75": float(q75),
                "fallback": bool(use_fallback),
            }
        )

        user_train_mask = df_train["user_id"] == user_id
        df_train_out.loc[user_train_mask, output_col] = _apply_bucket(df_train.loc[user_train_mask, target_col], q25, q75)

        user_test_rows = df_test[df_test["user_id"] == user_id]
        if not user_test_rows.empty:
            user_test_mask = df_test["user_id"] == user_id
            df_test_out.loc[user_test_mask, output_col] = _apply_bucket(df_test.loc[user_test_mask, target_col], q25, q75)

    # Safety: users only in test still receive train-derived global thresholds.
    unmatched_test_mask = df_test_out[output_col].isna()
    if unmatched_test_mask.any():
        df_test_out.loc[unmatched_test_mask, output_col] = _apply_bucket(
            df_test.loc[unmatched_test_mask, target_col],
            global_q25,
            global_q75,
        )

    df_train_out[output_col] = df_train_out[output_col].astype(int)
    df_test_out[output_col] = df_test_out[output_col].astype(int)
    return df_train_out, df_test_out, user_thresholds, fallback_count


def validate_target_balance(y_train: pd.Series, y_test: pd.Series, bucket_mode: str) -> tuple[pd.Series, pd.Series]:
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    train_majority = float(train_dist.max())
    test_majority = float(test_dist.max())
    if train_majority > 0.90 or test_majority > 0.90:
        raise RuntimeError(
            f"Target imbalance detected ({bucket_mode}): "
            f"train majority={train_majority:.1%}, test majority={test_majority:.1%}. "
            "This will break downstream modeling. Review bucketing logic."
        )
    return train_dist, test_dist


def compute_entropy(bucket_list: list[int]) -> float:
    if not bucket_list:
        return 0.0
    counts = pd.Series(bucket_list).value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts + 1e-10)))


def add_entropy_feature(lag_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict] = []
    for user_id, grp in lag_df.groupby("user_id"):
        user_group = grp.sort_values("week_start").reset_index(drop=True)
        for idx, row in user_group.iterrows():
            if idx < 1:
                entropy_val = 0.0
            else:
                hist_size = min(4, idx)
                history_buckets = user_group.iloc[idx - hist_size : idx]["bucket_t"].astype(int).tolist()
                entropy_val = compute_entropy(history_buckets)
            results.append({"user_id": user_id, "week_t": row["week_t"], "entropy": entropy_val})
    return pd.DataFrame(results)


def add_recency_days_feature(features: pd.DataFrame, source_dataset: pd.DataFrame) -> pd.Series:
    work = features[["user_id", "week_t", "week_start"]].copy()
    source = source_dataset[["user_id", "transaction_date"]].copy()
    source["transaction_date"] = pd.to_datetime(source["transaction_date"], errors="coerce")

    out = pd.Series(index=work.index, dtype=float)
    for user_id, grp in work.groupby("user_id"):
        user_weeks = grp.sort_values("week_start")
        tx_dates = source[source["user_id"] == user_id]["transaction_date"].dropna().sort_values().to_numpy()
        week_vals = user_weeks["week_start"].to_numpy(dtype="datetime64[ns]")

        if len(tx_dates) == 0:
            out.loc[user_weeks.index] = 180.0
            continue

        # Last transaction strictly before week start (no future leakage).
        pos = np.searchsorted(tx_dates, week_vals, side="left") - 1
        recency = []
        for i, p in enumerate(pos):
            if p < 0:
                recency.append(180.0)
            else:
                delta = (week_vals[i] - tx_dates[p]) / np.timedelta64(1, "D")
                recency.append(float(delta))
        out.loc[user_weeks.index] = recency

    return out.clip(lower=0, upper=180)


def build_features(
    lag_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    txn_count_weekly: pd.DataFrame,
    user_avg_txn_per_week: pd.DataFrame,
    source_dataset: pd.DataFrame,
) -> pd.DataFrame:
    features = lag_df.merge(user_stats, on="user_id", how="left")
    features = features.merge(txn_count_weekly, on=["user_id", "week_t"], how="left")
    features = features.merge(user_avg_txn_per_week, on="user_id", how="left")

    if features[["user_mean_amount", "user_std_amount", "user_avg_txn_count", "txn_count"]].isna().any().any():
        missing = features[["user_mean_amount", "user_std_amount", "user_avg_txn_count", "txn_count"]].isna().sum()
        raise RuntimeError(f"Missing critical joined stats in features: {missing.to_dict()}")

    features = features.sort_values(["user_id", "week_start"]).reset_index(drop=True)

    # Tier 1
    denom = features["user_std_amount"].clip(lower=1e-6)
    features["z_score"] = ((features["amount_t"] - features["user_mean_amount"]) / denom).clip(-10, 10)

    entropy_df = add_entropy_feature(features[["user_id", "week_t", "week_start", "bucket_t"]])
    features = features.merge(entropy_df, on=["user_id", "week_t"], how="left")

    features["relative_txn_count"] = (
        features["txn_count"] / features["user_avg_txn_count"].clip(lower=1e-6)
    ).clip(0, 10)

    # Tier 2
    features["delta_amount"] = features.groupby("user_id", sort=False)["amount_t"].diff().fillna(0.0)
    features["delta_amount"] = features["delta_amount"].clip(-1000, 1000)

    features["delta_bucket"] = (
        features.groupby("user_id", sort=False)["bucket_t"].diff().fillna(0).astype(int)
    )

    shifted_amount = features.groupby("user_id", sort=False)["amount_t"].shift(1)
    features["rolling_mean_8w"] = shifted_amount.groupby(features["user_id"]).transform(
        lambda s: s.rolling(window=8, min_periods=1).mean()
    )
    features["rolling_std_8w"] = shifted_amount.groupby(features["user_id"]).transform(
        lambda s: s.rolling(window=8, min_periods=1).std()
    )
    features["rolling_mean_8w"] = features["rolling_mean_8w"].fillna(0.0)
    features["rolling_std_8w"] = features["rolling_std_8w"].fillna(0.0).clip(lower=1e-6)

    features["recency_days"] = add_recency_days_feature(features, source_dataset)

    features["user_cv"] = (
        features["user_std_amount"] / features["user_mean_amount"].clip(lower=1e-6)
    ).clip(0, 10)

    return features


def validate_no_nan(df: pd.DataFrame, feature_cols: list[str], split_name: str) -> None:
    nan_sum = df[feature_cols].isna().sum()
    bad = nan_sum[nan_sum > 0]
    if not bad.empty:
        raise RuntimeError(f"NaN values in {split_name} features: {bad.to_dict()}")


def year_week_from_dates(date_series: pd.Series) -> pd.Series:
    iso = date_series.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def compute_category_diversity(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    df = raw_dataset.copy()
    df["week_t"] = year_week_from_dates(pd.to_datetime(df["transaction_date"], errors="coerce"))
    return (
        df.groupby(["user_id", "week_t"], as_index=False)["category"]
        .nunique()
        .rename(columns={"category": "category_diversity"})
    )


def compute_dominant_category_ratio(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    df = raw_dataset.copy()
    df["week_t"] = year_week_from_dates(pd.to_datetime(df["transaction_date"], errors="coerce"))
    grp = df.groupby(["user_id", "week_t", "category"], as_index=False).size().rename(columns={"size": "cnt"})
    agg = grp.groupby(["user_id", "week_t"], as_index=False).agg(dominant_count=("cnt", "max"), total_txn=("cnt", "sum"))
    agg["dominant_category_ratio"] = agg["dominant_count"] / agg["total_txn"].clip(lower=1)
    return agg[["user_id", "week_t", "dominant_category_ratio"]]


def compute_amount_bucket_lags(lag_features_df: pd.DataFrame) -> pd.DataFrame:
    work = lag_features_df.sort_values(["user_id", "week_start"]).copy()
    work["amount_t_minus_1"] = work.groupby("user_id", sort=False)["amount_t"].shift(1)
    work["amount_t_minus_2"] = work.groupby("user_id", sort=False)["amount_t"].shift(2)
    work["bucket_t_minus_1"] = work.groupby("user_id", sort=False)["bucket_t"].shift(1)
    return work[["user_id", "week_t", "amount_t_minus_1", "amount_t_minus_2", "bucket_t_minus_1"]]


def main() -> None:
    args = parse_args()
    step1_dir = Path(args.step1_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    train_dataset = pd.read_csv(step1_dir / "train_dataset.csv")
    test_dataset = pd.read_csv(step1_dir / "test_dataset.csv")
    train_lag = pd.read_csv(step1_dir / "train_lag_features.csv")
    test_lag = pd.read_csv(step1_dir / "test_lag_features.csv")
    metadata = json.loads((step1_dir / "metadata.json").read_text(encoding="utf-8"))

    for df in [train_dataset, test_dataset]:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    for lag_df in [train_lag, test_lag]:
        lag_df["week_start"] = lag_df["week_t"].apply(parse_year_week_to_monday)

    # Recompute bucket labels in Step 2 to avoid degenerate global bucketing from Step 1.
    if args.bucket_mode == "user_quantile":
        train_lag, test_lag, user_thresholds_t1, fallback_count_t1 = compute_user_specific_buckets(
            train_lag,
            test_lag,
            target_col="amount_t_plus_1",
            output_col="bucket_t_plus_1",
        )
        train_lag, test_lag, user_thresholds_t0, fallback_count_t0 = compute_user_specific_buckets(
            train_lag,
            test_lag,
            target_col="amount_t",
            output_col="bucket_t",
        )
        print(
            f"User-specific bucketing enabled: "
            f"fallback users (t+1)={fallback_count_t1}, fallback users (t)={fallback_count_t0}"
        )
    else:
        q25_global_next = float(train_lag["amount_t_plus_1"].quantile(0.25))
        q75_global_next = float(train_lag["amount_t_plus_1"].quantile(0.75))
        q25_global_curr = float(train_lag["amount_t"].quantile(0.25))
        q75_global_curr = float(train_lag["amount_t"].quantile(0.75))
        train_lag["bucket_t_plus_1"] = _apply_bucket(train_lag["amount_t_plus_1"], q25_global_next, q75_global_next)
        test_lag["bucket_t_plus_1"] = _apply_bucket(test_lag["amount_t_plus_1"], q25_global_next, q75_global_next)
        train_lag["bucket_t"] = _apply_bucket(train_lag["amount_t"], q25_global_curr, q75_global_curr)
        test_lag["bucket_t"] = _apply_bucket(test_lag["amount_t"], q25_global_curr, q75_global_curr)
        user_thresholds_t1 = []
        fallback_count_t1 = 0

    # Phase 1: user statistics on train only.
    train_weekly_amount = train_dataset.copy()
    iso_weekly = train_weekly_amount["transaction_date"].dt.isocalendar()
    train_weekly_amount["week_t"] = (
        iso_weekly["year"].astype(str) + "-W" + iso_weekly["week"].astype(str).str.zfill(2)
    )
    train_weekly_amount = (
        train_weekly_amount.groupby(["user_id", "week_t"], as_index=False)
        .agg(weekly_amount=("amount", "sum"), weekly_txn_count=("amount", "size"), category_count=("category", "nunique"))
    )

    user_stats = (
        train_weekly_amount.groupby("user_id", as_index=False)
        .agg(
            user_mean_amount=("weekly_amount", "mean"),
            user_std_amount=("weekly_amount", "std"),
            user_median_amount=("weekly_amount", "median"),
            user_min_amount=("weekly_amount", "min"),
            user_max_amount=("weekly_amount", "max"),
            user_category_count=("category_count", "mean"),
            user_total_txn_count=("weekly_txn_count", "sum"),
        )
        .copy()
    )
    user_stats["user_category_count"] = user_stats["user_category_count"].round().astype(int)
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0.0).replace(0.0, 1e-6)
    user_stats.to_csv(output_dir / "step2_user_statistics.csv", index=False)

    if user_stats.empty:
        raise RuntimeError("No user statistics computed")
    if user_stats["user_id"].nunique() != len(user_stats):
        raise RuntimeError("Duplicate user IDs in user statistics")
    if not (user_stats["user_std_amount"] > 0).all():
        raise RuntimeError("Zero std after guard in user statistics")

    # Weekly transaction counts by split.
    def build_weekly_txn_count(df: pd.DataFrame) -> pd.DataFrame:
        temp = df.copy()
        iso = temp["transaction_date"].dt.isocalendar()
        temp["week_t"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        return temp.groupby(["user_id", "week_t"], as_index=False).size().rename(columns={"size": "txn_count"})

    train_txn_count_weekly = build_weekly_txn_count(train_dataset)
    test_txn_count_weekly = build_weekly_txn_count(test_dataset)

    user_avg_txn_per_week = (
        train_txn_count_weekly.groupby("user_id", as_index=False)["txn_count"].mean().rename(
            columns={"txn_count": "user_avg_txn_count"}
        )
    )

    # Phase 2 + 3.
    train_features = build_features(
        train_lag,
        user_stats,
        train_txn_count_weekly,
        user_avg_txn_per_week,
        train_dataset,
    )
    test_features = build_features(
        test_lag,
        user_stats,
        test_txn_count_weekly,
        user_avg_txn_per_week,
        test_dataset,
    )

    # Leakage check: user means must come from train statistics.
    train_user_mean_computed = train_weekly_amount.groupby("user_id", as_index=False)["weekly_amount"].mean()
    train_user_mean_computed = train_user_mean_computed.rename(columns={"weekly_amount": "computed_mean"})
    sample_merge = train_features[["user_id", "user_mean_amount"]].drop_duplicates().merge(
        train_user_mean_computed, on="user_id", how="left"
    )
    if not np.allclose(sample_merge["user_mean_amount"], sample_merge["computed_mean"], rtol=0.01, atol=1e-8):
        raise RuntimeError("User statistics mismatch, leakage check failed")

    feature_cols = [
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

    validate_no_nan(train_features, feature_cols, "train")
    validate_no_nan(test_features, feature_cols, "test")

    final_cols = [
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

    train_final = train_features[final_cols].copy()
    test_final = test_features[final_cols].copy()

    train_dist, test_dist = validate_target_balance(
        train_final["bucket_t_plus_1"],
        test_final["bucket_t_plus_1"],
        args.bucket_mode,
    )

    stats_summary = {}
    for col in feature_cols:
        stats_summary[col] = {
            "train_mean": float(train_final[col].mean()),
            "train_std": float(train_final[col].std()),
            "train_min": float(train_final[col].min()),
            "train_max": float(train_final[col].max()),
            "test_mean": float(test_final[col].mean()),
            "test_std": float(test_final[col].std()),
            "train_non_null": int(train_final[col].notna().sum()),
            "test_non_null": int(test_final[col].notna().sum()),
        }

    distribution_check = {}
    for col in feature_cols:
        ks_stat, p_value = ks_2samp(train_final[col].dropna(), test_final[col].dropna())
        distribution_check[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "distributions_similar": bool(p_value > 0.05),
        }

    train_out = output_dir / "train_features_engineered.csv"
    test_out = output_dir / "test_features_engineered.csv"
    train_final.to_csv(train_out, index=False)
    test_final.to_csv(test_out, index=False)

    # --- Tier 3 extension ---
    train_cat_div = compute_category_diversity(train_dataset)
    test_cat_div = compute_category_diversity(test_dataset)
    train_dom_cat = compute_dominant_category_ratio(train_dataset)
    test_dom_cat = compute_dominant_category_ratio(test_dataset)

    train_t3 = train_final.merge(train_cat_div, on=["user_id", "week_t"], how="left")
    test_t3 = test_final.merge(test_cat_div, on=["user_id", "week_t"], how="left")
    train_t3 = train_t3.merge(train_dom_cat, on=["user_id", "week_t"], how="left")
    test_t3 = test_t3.merge(test_dom_cat, on=["user_id", "week_t"], how="left")

    train_lags_t3 = compute_amount_bucket_lags(train_lag)
    test_lags_t3 = compute_amount_bucket_lags(test_lag)
    train_t3 = train_t3.merge(train_lags_t3, on=["user_id", "week_t"], how="left")
    test_t3 = test_t3.merge(test_lags_t3, on=["user_id", "week_t"], how="left")

    user_mean_map = user_stats.set_index("user_id")["user_mean_amount"]
    for col in ["amount_t_minus_1", "amount_t_minus_2"]:
        train_t3[col] = train_t3[col].fillna(train_t3["user_id"].map(user_mean_map)).fillna(0.0)
        test_t3[col] = test_t3[col].fillna(test_t3["user_id"].map(user_mean_map)).fillna(0.0)

    train_t3["bucket_t_minus_1"] = train_t3["bucket_t_minus_1"].fillna(3).astype(int)
    test_t3["bucket_t_minus_1"] = test_t3["bucket_t_minus_1"].fillna(3).astype(int)

    tier3_cols = [
        "category_diversity",
        "dominant_category_ratio",
        "amount_t_minus_1",
        "amount_t_minus_2",
        "bucket_t_minus_1",
    ]

    # Tier 3 validations
    validate_no_nan(train_t3, tier3_cols, "train_tier3")
    validate_no_nan(test_t3, tier3_cols, "test_tier3")
    if not (train_t3["category_diversity"] > 0).all() or not (test_t3["category_diversity"] > 0).all():
        raise RuntimeError("category_diversity must be > 0")
    if not train_t3["dominant_category_ratio"].between(0, 1).all() or not test_t3["dominant_category_ratio"].between(0, 1).all():
        raise RuntimeError("dominant_category_ratio out of [0,1]")
    if not train_t3["bucket_t_minus_1"].isin([0, 1, 2, 3]).all() or not test_t3["bucket_t_minus_1"].isin([0, 1, 2, 3]).all():
        raise RuntimeError("bucket_t_minus_1 has invalid values")

    # Keep requested deterministic column order.
    final_t3_cols = final_cols + tier3_cols
    train_t3_final = train_t3[final_t3_cols].copy()
    test_t3_final = test_t3[final_t3_cols].copy()

    if len(train_t3_final.columns) != 21 or len(test_t3_final.columns) != 21:
        raise RuntimeError("Tier3 schema mismatch: expected 21 columns")

    consistency_check = {}
    for col in ["category_diversity", "dominant_category_ratio", "amount_t_minus_1", "amount_t_minus_2"]:
        ks_stat, p_value = ks_2samp(train_t3_final[col].dropna(), test_t3_final[col].dropna())
        consistency_check[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "consistent": bool(p_value > 0.05),
        }

    tier3_stats = {}
    for col in ["category_diversity", "dominant_category_ratio", "amount_t_minus_1", "amount_t_minus_2", "bucket_t_minus_1"]:
        tier3_stats[col] = {
            "train_mean": float(train_t3_final[col].mean()),
            "train_std": float(train_t3_final[col].std()),
            "train_min": float(train_t3_final[col].min()),
            "train_max": float(train_t3_final[col].max()),
            "test_mean": float(test_t3_final[col].mean()),
        }

    train_t3_out = output_dir / "train_features_tier3.csv"
    test_t3_out = output_dir / "test_features_tier3.csv"
    train_t3_final.to_csv(train_t3_out, index=False)
    test_t3_final.to_csv(test_t3_out, index=False)

    report = {
        "feature_engineering_report": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "bucket_configuration": {
                "mode": args.bucket_mode,
                "user_quantile_config": {
                    "q_percentiles": [0.25, 0.75],
                    "fallback_threshold_min_weeks": 4,
                    "user_thresholds_summary": {
                        "q25_min": float(min([t["q25"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "q25_median": float(np.median([t["q25"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "q25_max": float(max([t["q25"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "q75_min": float(min([t["q75"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "q75_median": float(np.median([t["q75"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "q75_max": float(max([t["q75"] for t in user_thresholds_t1])) if user_thresholds_t1 else None,
                        "fallback_count": int(fallback_count_t1),
                    }
                    if args.bucket_mode == "user_quantile"
                    else None,
                },
                "target_distribution_train_percentages": {str(k): float(v) for k, v in train_dist.to_dict().items()},
                "target_distribution_test_percentages": {str(k): float(v) for k, v in test_dist.to_dict().items()},
            },
            "phase_1_user_statistics": {
                "total_users": int(len(user_stats)),
                "user_mean_amount_stats": {
                    "mean": float(user_stats["user_mean_amount"].mean()),
                    "std": float(user_stats["user_mean_amount"].std()),
                    "min": float(user_stats["user_mean_amount"].min()),
                    "max": float(user_stats["user_mean_amount"].max()),
                },
                "user_std_amount_stats": {
                    "mean": float(user_stats["user_std_amount"].mean()),
                    "std": float(user_stats["user_std_amount"].std()),
                },
            },
            "tier_1_features": {
                "z_score": {
                    "description": "Normalized deviation from user mean",
                    "formula": "(amount_t - user_mean) / user_std",
                    "range": "[-10, 10]",
                    "train_stats": stats_summary.get("z_score", {}),
                },
                "entropy": {
                    "description": "Shannon entropy of bucket distribution",
                    "formula": "-sum(p_i * log2(p_i))",
                    "range": "[0, 2]",
                    "train_stats": stats_summary.get("entropy", {}),
                },
                "txn_count": {
                    "description": "Number of individual transactions in week",
                    "range": "[1, inf)",
                    "train_stats": stats_summary.get("txn_count", {}),
                },
                "relative_txn_count": {
                    "description": "Transaction count relative to user average",
                    "formula": "txn_count / user_avg_txn_count",
                    "range": "[0, 10]",
                    "train_stats": stats_summary.get("relative_txn_count", {}),
                },
            },
            "tier_2_features": {
                "delta_amount": {
                    "description": "Change in amount from previous week",
                    "formula": "amount_t - amount_t-1",
                    "range": "[-1000, 1000]",
                    "train_stats": stats_summary.get("delta_amount", {}),
                },
                "delta_bucket": {
                    "description": "Change in bucket classification",
                    "formula": "bucket_t - bucket_t-1",
                    "range": "[-3, 3]",
                    "train_stats": stats_summary.get("delta_bucket", {}),
                },
                "rolling_mean_8w": {
                    "description": "Mean amount over last 8 prior weeks",
                    "range": "[0, inf)",
                    "train_stats": stats_summary.get("rolling_mean_8w", {}),
                },
                "rolling_std_8w": {
                    "description": "Std amount over last 8 prior weeks",
                    "range": "[0, inf)",
                    "train_stats": stats_summary.get("rolling_std_8w", {}),
                },
                "recency_days": {
                    "description": "Days since last transaction before current week",
                    "range": "[0, 180]",
                    "train_stats": stats_summary.get("recency_days", {}),
                },
                "user_cv": {
                    "description": "Coefficient of variation (std/mean)",
                    "formula": "user_std / user_mean",
                    "range": "[0, 10]",
                    "train_stats": stats_summary.get("user_cv", {}),
                },
            },
            "tier3_feature_engineering_report": {
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "Tier 3 Addition",
                "input_files": {
                    "train_features": str(train_out),
                    "test_features": str(test_out),
                    "train_dataset": str(step1_dir / "train_dataset.csv"),
                    "test_dataset": str(step1_dir / "test_dataset.csv"),
                },
                "tier3_features_added": {
                    "category_diversity": {
                        "description": "Number of unique categories per user per week",
                        "train_stats": tier3_stats.get("category_diversity", {}),
                    },
                    "dominant_category_ratio": {
                        "description": "Share of most common category (0-1)",
                        "train_stats": tier3_stats.get("dominant_category_ratio", {}),
                    },
                    "amount_t_minus_1": {
                        "description": "Amount spent in previous week",
                        "train_stats": tier3_stats.get("amount_t_minus_1", {}),
                    },
                    "amount_t_minus_2": {
                        "description": "Amount spent 2 weeks ago",
                        "train_stats": tier3_stats.get("amount_t_minus_2", {}),
                    },
                    "bucket_t_minus_1": {
                        "description": "Bucket classification in previous week",
                        "range": "[0, 1, 2, 3]",
                        "train_stats": tier3_stats.get("bucket_t_minus_1", {}),
                    },
                },
                "total_features": 21,
                "features_breakdown": {
                    "metadata": 6,
                    "tier_1": 4,
                    "tier_2": 6,
                    "tier_3": 5,
                },
                "quality_checks": {
                    "data_leakage": "No leakage detected",
                    "nan_values": {
                        "train": int(train_final.isnull().sum().sum()),
                        "test": int(test_final.isnull().sum().sum()),
                    },
                    "distribution_similarity": distribution_check,
                },
                "output_files": {
                    "train_features": str(train_out),
                    "test_features": str(test_out),
                    "train_rows": int(len(train_final)),
                    "test_rows": int(len(test_final)),
                    "total_features": 10,
                    "train_features_tier3": str(train_t3_out),
                    "test_features_tier3": str(test_t3_out),
                    "total_features_tier3": 15,
                },
                "metadata_used": {
                    "Q25_threshold": float(metadata.get("Q25_threshold", np.nan)),
                    "Q75_threshold": float(metadata.get("Q75_threshold", np.nan)),
                },
            },
        }
    }

    report_path = output_dir / "feature_engineering_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    tier3_report = {
        "tier3_feature_engineering_report": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "Tier 3 Addition",
            "input_files": {
                "train_features": str(train_out),
                "test_features": str(test_out),
                "train_dataset": str(step1_dir / "train_dataset.csv"),
                "test_dataset": str(step1_dir / "test_dataset.csv"),
            },
            "tier3_features_added": {
                "category_diversity": {
                    "description": "Number of unique categories per user per week",
                    "train_stats": tier3_stats.get("category_diversity", {}),
                },
                "dominant_category_ratio": {
                    "description": "Share of most common category (0-1)",
                    "train_stats": tier3_stats.get("dominant_category_ratio", {}),
                },
                "amount_t_minus_1": {
                    "description": "Amount spent in previous week",
                    "train_stats": tier3_stats.get("amount_t_minus_1", {}),
                },
                "amount_t_minus_2": {
                    "description": "Amount spent 2 weeks ago",
                    "train_stats": tier3_stats.get("amount_t_minus_2", {}),
                },
                "bucket_t_minus_1": {
                    "description": "Bucket classification in previous week",
                    "range": "[0, 1, 2, 3]",
                    "train_stats": tier3_stats.get("bucket_t_minus_1", {}),
                },
            },
            "total_features": 21,
            "features_breakdown": {
                "metadata": 6,
                "tier_1": 4,
                "tier_2": 6,
                "tier_3": 5,
            },
            "quality_checks": {
                "no_nan_values": True,
                "row_counts_match": bool(len(train_t3_final) == len(train_lag) and len(test_t3_final) == len(test_lag)),
                "value_ranges_valid": True,
                "distribution_consistency": consistency_check,
            },
            "output_files": {
                "train_features": str(train_t3_out),
                "test_features": str(test_t3_out),
                "train_rows": int(len(train_t3_final)),
                "test_rows": int(len(test_t3_final)),
            },
        }
    }

    tier3_report_path = output_dir / "tier3_feature_engineering_report.json"
    tier3_report_path.write_text(json.dumps(tier3_report, indent=2, ensure_ascii=False), encoding="utf-8")

    feature_docs = """# Feature Definitions (Tier 1 + Tier 2)

## Tier 1: Critical Features

### z_score
- **Formula:** (amount_t - user_mean_amount) / user_std_amount
- **Range:** [-10, 10]
- **Interpretation:** How many standard deviations current week is from user average
- **Why it matters:** Normalizes spending across users with different spending habits

### entropy
- **Formula:** -sum(p_i * log2(p_i)) where p_i = P(bucket_i) in last 4 weeks
- **Range:** [0, 2]
- **Interpretation:** 0 = predictable (always same bucket), 2 = unpredictable (uniform)
- **Why it matters:** Indicates user reliability; high entropy means model should be cautious

### txn_count
- **Definition:** Number of individual transactions in week_t
- **Range:** [1, inf)
- **Why it matters:** Distinguishes between few large purchases vs. many small purchases

### relative_txn_count
- **Formula:** txn_count / user_avg_txn_count
- **Range:** [0, 10] (clipped)
- **Interpretation:** 1 = average activity, >1 = high activity, <1 = low activity
- **Why it matters:** Normalized activity indicator

## Tier 2: Dynamic + RFM Features

### delta_amount
- **Formula:** amount_t - amount_t-1
- **Range:** [-1000, 1000] (clipped)
- **Interpretation:** Positive = increasing spending, negative = decreasing
- **Why it matters:** Captures momentum in spending changes

### delta_bucket
- **Formula:** bucket_t - bucket_t-1
- **Range:** [-3, 3]
- **Interpretation:** -3 = big drop, 0 = stable, +3 = big increase
- **Why it matters:** Discrete indicator of spending transitions

### rolling_mean_8w
- **Formula:** mean(amount from prior 8 weeks)
- **Why it matters:** Medium-term baseline for user spending

### rolling_std_8w
- **Formula:** std(amount from prior 8 weeks)
- **Why it matters:** Medium-term volatility; indicates spending consistency

### recency_days
- **Formula:** days since last transaction before current week (clipped to [0, 180])
- **Why it matters:** RFM metric; recent activity correlates with future activity

### user_cv
- **Formula:** user_std_amount / user_mean_amount
- **Range:** [0, 10] (clipped)
- **Interpretation:** Relative volatility; high CV = unpredictable user
- **Why it matters:** Indicates user spending consistency independent of amount
"""

    feature_docs_path = output_dir / "feature_definitions.md"
    feature_docs_path.write_text(feature_docs, encoding="utf-8")

    feature_docs_tier3 = """# Feature Definitions (Tier 1 + Tier 2 + Tier 3)

## Summary
Total features: 21 (6 metadata + 4 Tier 1 + 6 Tier 2 + 5 Tier 3)

## Tier 3: Category Diversity & Historical Patterns

### category_diversity
- **Definition:** Number of unique transaction categories in week_t
- **Range:** [1, 15]
- **Interpretation:** Higher diversity = more varied spending patterns

### dominant_category_ratio
- **Formula:** max(category_count) / total_transactions
- **Range:** [0, 1]
- **Interpretation:** 0.5 = spread; 0.9 = one category dominates

### amount_t_minus_1
- **Definition:** Total amount spent in week t-1
- **How built:** Shifted by 1 week; filled with train user_mean_amount for first week

### amount_t_minus_2
- **Definition:** Total amount spent in week t-2
- **How built:** Shifted by 2 weeks; filled with train user_mean_amount for early weeks

### bucket_t_minus_1
- **Definition:** Bucket classification in week t-1
- **Range:** [0, 1, 2, 3]
- **How built:** Shifted by 1 week; filled with mode=3 for first week

## Quality
- No temporal leakage (lags only from past)
- Train-only statistics used for fallback fills in test
- No NaN values in final datasets
"""
    feature_docs_tier3_path = output_dir / "feature_definitions_tier3.md"
    feature_docs_tier3_path.write_text(feature_docs_tier3, encoding="utf-8")

    print("=== STEP 2 DONE: Feature engineering completed ===")
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")
    print(f"Saved: {report_path}")
    print(f"Saved: {feature_docs_path}")
    print(f"Saved: {train_t3_out}")
    print(f"Saved: {test_t3_out}")
    print(f"Saved: {tier3_report_path}")
    print(f"Saved: {feature_docs_tier3_path}")


if __name__ == "__main__":
    main()

