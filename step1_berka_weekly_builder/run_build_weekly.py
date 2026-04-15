import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 Berka: Build weekly datasets")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/berka/processed/transactions_normalized.csv",
        help="Normalized transactions CSV from step0",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="step1_berka_weekly_builder/outputs",
        help="Output directory",
    )
    parser.add_argument("--test-weeks", type=int, default=12, help="Number of latest weeks reserved for holdout test")
    parser.add_argument("--max-users", type=int, default=0, help="Optional user cap for quick runs")
    parser.add_argument("--max-weeks", type=int, default=0, help="Optional recent week cap for quick runs")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def week_start_monday(date_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(date_series, errors="coerce")
    return ts - pd.to_timedelta(ts.dt.weekday, unit="D")


def week_label(ws: pd.Series) -> pd.Series:
    iso = ws.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def assign_bucket(values: pd.Series, q25: float, q75: float) -> pd.Series:
    out = np.select([values.eq(0), values.le(q25), values.le(q75)], [0, 1, 2], default=3)
    return pd.Series(out, index=values.index).astype(int)


def compute_weeks_since_positive(shifted_series: pd.Series, cap: int = 52) -> pd.Series:
    """Weeks since the last positive event using only past data (series should already be shifted)."""
    values = shifted_series.fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=float)
    last_positive = None
    for idx, val in enumerate(values):
        if last_positive is None:
            out[idx] = float(cap)
        else:
            gap = idx - last_positive
            out[idx] = float(cap if gap > cap else gap)
        if val > 0:
            last_positive = idx
    return pd.Series(out, index=shifted_series.index)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ws = pd.to_datetime(df["week_start"], errors="coerce")
    iso = ws.dt.isocalendar()
    week_end = ws + pd.to_timedelta(6, unit="D")
    month_end = week_end + pd.offsets.MonthEnd(0)

    df["week_of_year"] = iso["week"].astype(int)
    df["month"] = ws.dt.month.astype(int)
    df["quarter"] = ws.dt.quarter.astype(int)
    df["week_of_month"] = (1 + ((ws.dt.day - 1) // 7)).astype(int)
    df["is_month_start_week"] = (ws.dt.day <= 7).astype(int)
    df["is_month_end_week"] = ((month_end - week_end).dt.days <= 6).astype(int)
    return df


def build_classification(weekly_user: pd.DataFrame, split_week: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_base = weekly_user[weekly_user["week_start"] <= split_week].copy()
    q25_spend = float(train_base["weekly_outflow_t"].quantile(0.25))
    q75_spend = float(train_base["weekly_outflow_t"].quantile(0.75))
    q25_net = float(train_base["weekly_net_t"].quantile(0.25))
    q75_net = float(train_base["weekly_net_t"].quantile(0.75))

    all_df = weekly_user.sort_values(["user_id", "week_start"]).copy()
    all_df["bucket_spend_t"] = assign_bucket(all_df["weekly_outflow_t"], q25_spend, q75_spend)
    all_df["bucket_net_t"] = assign_bucket(all_df["weekly_net_t"], q25_net, q75_net)

    all_df = add_calendar_features(all_df)
    by_user = all_df.groupby("user_id", sort=False)
    all_df["target_spend_t_plus_1"] = by_user["weekly_outflow_t"].shift(-1)
    all_df["target_net_t_plus_1"] = by_user["weekly_net_t"].shift(-1)
    all_df["bucket_spend_t_plus_1"] = by_user["bucket_spend_t"].shift(-1)
    all_df["bucket_net_t_plus_1"] = by_user["bucket_net_t"].shift(-1)

    all_df["weekly_inflow_t_minus_1"] = by_user["weekly_inflow_t"].shift(1).fillna(0.0)
    all_df["weekly_outflow_t_minus_1"] = by_user["weekly_outflow_t"].shift(1).fillna(0.0)
    all_df["weekly_net_t_minus_1"] = by_user["weekly_net_t"].shift(1).fillna(0.0)
    all_df["weekly_inflow_t_minus_2"] = by_user["weekly_inflow_t"].shift(2).fillna(0.0)
    all_df["weekly_outflow_t_minus_2"] = by_user["weekly_outflow_t"].shift(2).fillna(0.0)

    all_df["delta_inflow"] = all_df["weekly_inflow_t"] - all_df["weekly_inflow_t_minus_1"]
    all_df["delta_outflow"] = all_df["weekly_outflow_t"] - all_df["weekly_outflow_t_minus_1"]

    eps = 1e-6
    all_df["inflow_outflow_ratio"] = (all_df["weekly_inflow_t"] / (all_df["weekly_outflow_t"] + eps)).clip(0.0, 10.0)
    all_df["inflow_share"] = (
        all_df["weekly_inflow_t"] / (all_df["weekly_inflow_t"] + all_df["weekly_outflow_t"] + eps)
    ).clip(0.0, 1.0)

    shifted_inflow = by_user["weekly_inflow_t"].shift(1)
    shifted_outflow = by_user["weekly_outflow_t"].shift(1)
    all_df["inflow_rolling_mean_8w"] = shifted_inflow.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    all_df["inflow_rolling_std_8w"] = shifted_inflow.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).std()
    )
    all_df["outflow_rolling_mean_8w"] = shifted_outflow.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    all_df["outflow_rolling_std_8w"] = shifted_outflow.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).std()
    )

    shifted_inflow_positive = (shifted_inflow > 0).astype(float)
    shifted_outflow_positive = (shifted_outflow > 0).astype(float)
    all_df["inflow_frequency_8w"] = shifted_inflow_positive.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    all_df["outflow_frequency_8w"] = shifted_outflow_positive.groupby(all_df["user_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )

    all_df["weeks_since_inflow"] = by_user["weekly_inflow_t"].transform(
        lambda s: compute_weeks_since_positive(s.shift(1), cap=52)
    )
    all_df["weeks_since_outflow"] = by_user["weekly_outflow_t"].transform(
        lambda s: compute_weeks_since_positive(s.shift(1), cap=52)
    )

    all_df["outflow_inflow_ratio_t"] = all_df["weekly_outflow_t"] / all_df["weekly_inflow_t"].replace(0, np.nan)
    all_df["outflow_inflow_ratio_t"] = all_df["outflow_inflow_ratio_t"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fill_zero_cols = [
        "weekly_inflow_t_minus_1",
        "weekly_outflow_t_minus_1",
        "weekly_net_t_minus_1",
        "weekly_inflow_t_minus_2",
        "weekly_outflow_t_minus_2",
        "delta_inflow",
        "delta_outflow",
        "inflow_outflow_ratio",
        "inflow_share",
        "inflow_rolling_mean_8w",
        "inflow_rolling_std_8w",
        "outflow_rolling_mean_8w",
        "outflow_rolling_std_8w",
        "inflow_frequency_8w",
        "outflow_frequency_8w",
        "weeks_since_inflow",
        "weeks_since_outflow",
        "outflow_inflow_ratio_t",
    ]
    all_df[fill_zero_cols] = all_df[fill_zero_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lag_df = all_df.dropna(subset=["bucket_spend_t_plus_1", "target_spend_t_plus_1"]).copy()
    lag_df["bucket_spend_t_plus_1"] = lag_df["bucket_spend_t_plus_1"].astype(int)
    lag_df["bucket_net_t_plus_1"] = lag_df["bucket_net_t_plus_1"].astype(int)

    train_lag = lag_df[lag_df["week_start"] <= split_week].copy()
    test_lag = lag_df[lag_df["week_start"] > split_week].copy()

    if train_lag.isna().any().any() or test_lag.isna().any().any():
        raise RuntimeError("NaN values detected in classification lag features after fill policy")

    meta = {
        "classification_target_default": "bucket_spend_t_plus_1",
        "alternative_target": "bucket_net_t_plus_1",
        "feature_fill_policy": "Lag/rolling/regularity NaN filled with 0.0",
        "calendar_definition": {
            "week_start": "monday",
            "week_of_month": "1 + floor((day(week_start)-1)/7)",
            "is_month_start_week": "week_start.day <= 7",
            "is_month_end_week": "week_end in last 7 days of month",
        },
        "q25_spend_train": q25_spend,
        "q75_spend_train": q75_spend,
        "q25_net_train": q25_net,
        "q75_net_train": q75_net,
    }
    return train_lag, test_lag, meta


def build_regression(tx: pd.DataFrame, weekly_user: pd.DataFrame, split_week: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cat_weekly = (
        tx.groupby(["user_id", "week_start", "week_t", "category", "flow_direction"], as_index=False)
        .agg(amount_cat_t=("amount_abs", "sum"), txn_count_cat_t=("amount_abs", "size"))
        .sort_values(["user_id", "category", "week_start"])
    )

    cat_group = cat_weekly.groupby(["user_id", "category"], sort=False)
    cat_weekly["amount_cat_t_minus_1"] = cat_group["amount_cat_t"].shift(1)
    cat_weekly["amount_cat_t_minus_2"] = cat_group["amount_cat_t"].shift(2)
    cat_weekly["amount_cat_t_plus_1"] = cat_group["amount_cat_t"].shift(-1)

    shifted = cat_group["amount_cat_t"].shift(1)
    cat_weekly["rolling_mean_4"] = shifted.groupby([cat_weekly["user_id"], cat_weekly["category"]]).transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    cat_weekly["rolling_std_4"] = shifted.groupby([cat_weekly["user_id"], cat_weekly["category"]]).transform(
        lambda s: s.rolling(4, min_periods=1).std()
    )

    user_week = weekly_user[["user_id", "week_start", "weekly_inflow_t", "weekly_outflow_t", "weekly_net_t"]].copy()
    user_week = user_week.sort_values(["user_id", "week_start"])
    user_grp = user_week.groupby("user_id", sort=False)
    user_week["weekly_inflow_t_minus_1"] = user_grp["weekly_inflow_t"].shift(1).fillna(0.0)
    user_week["weekly_outflow_t_minus_1"] = user_grp["weekly_outflow_t"].shift(1).fillna(0.0)
    user_week["weekly_net_t_minus_1"] = user_grp["weekly_net_t"].shift(1).fillna(0.0)

    reg = cat_weekly.merge(
        user_week[
            [
                "user_id",
                "week_start",
                "weekly_inflow_t_minus_1",
                "weekly_outflow_t_minus_1",
                "weekly_net_t_minus_1",
            ]
        ],
        on=["user_id", "week_start"],
        how="left",
    )
    reg = reg.dropna(subset=["amount_cat_t_plus_1"]).copy()

    fill_cols = ["amount_cat_t_minus_1", "amount_cat_t_minus_2", "rolling_mean_4", "rolling_std_4"]
    reg[fill_cols] = reg[fill_cols].fillna(0.0)
    reg["rolling_std_4"] = reg["rolling_std_4"].fillna(0.0)

    train_reg = reg[reg["week_start"] <= split_week].copy()
    test_reg = reg[reg["week_start"] > split_week].copy()

    meta = {
        "regression_format": "observed-only",
        "zero_rate_target_train": float((train_reg["amount_cat_t_plus_1"] == 0).mean()),
        "zero_rate_target_test": float((test_reg["amount_cat_t_plus_1"] == 0).mean()),
    }
    return train_reg, test_reg, meta


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    cls_out = out_dir / "classification"
    reg_out = out_dir / "regression"
    ensure_dir(cls_out)
    ensure_dir(reg_out)

    tx = pd.read_csv(input_csv)
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx = tx.dropna(subset=["transaction_date", "user_id", "amount"])
    tx["user_id"] = tx["user_id"].astype(int)

    if args.max_users > 0:
        keep_users = tx["user_id"].value_counts().head(args.max_users).index
        tx = tx[tx["user_id"].isin(keep_users)].copy()

    tx["week_start"] = week_start_monday(tx["transaction_date"])
    tx["week_t"] = week_label(tx["week_start"])

    if args.max_weeks > 0:
        recent = sorted(tx["week_start"].dropna().unique())[-args.max_weeks :]
        tx = tx[tx["week_start"].isin(recent)].copy()

    tx["inflow"] = np.where(tx["flow_direction"].eq("inflow"), tx["amount_abs"], 0.0)
    tx["outflow"] = np.where(tx["flow_direction"].eq("outflow"), tx["amount_abs"], 0.0)

    weekly_user = (
        tx.groupby(["user_id", "week_start", "week_t"], as_index=False)
        .agg(
            weekly_inflow_t=("inflow", "sum"),
            weekly_outflow_t=("outflow", "sum"),
            txn_count_t=("amount", "size"),
            category_diversity_t=("category", "nunique"),
        )
        .sort_values(["user_id", "week_start"])
    )
    weekly_user["weekly_net_t"] = weekly_user["weekly_inflow_t"] - weekly_user["weekly_outflow_t"]

    unique_weeks = sorted(weekly_user["week_start"].dropna().unique())
    if len(unique_weeks) <= args.test_weeks + 2:
        raise RuntimeError("Not enough weekly history for requested split")
    split_week = pd.Timestamp(unique_weeks[-args.test_weeks - 1])

    train_cls, test_cls, cls_meta = build_classification(weekly_user, split_week)
    train_reg, test_reg, reg_meta = build_regression(tx, weekly_user, split_week)

    train_weekly = weekly_user[weekly_user["week_start"] <= split_week].copy()
    test_weekly = weekly_user[weekly_user["week_start"] > split_week].copy()

    train_weekly.to_csv(cls_out / "train_dataset.csv", index=False)
    test_weekly.to_csv(cls_out / "test_dataset.csv", index=False)
    train_cls.to_csv(cls_out / "train_lag_features.csv", index=False)
    test_cls.to_csv(cls_out / "test_lag_features.csv", index=False)

    train_reg.to_csv(reg_out / "train_regression.csv", index=False)
    test_reg.to_csv(reg_out / "test_regression.csv", index=False)

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "split_logic": "holdout_last_weeks",
        "split_week": str(split_week.date()),
        "total_users": int(tx["user_id"].nunique()),
        "total_weeks": int(len(unique_weeks)),
        "classification": {
            **cls_meta,
            "train_rows": int(len(train_cls)),
            "test_rows": int(len(test_cls)),
            "train_weeks": int(train_weekly["week_t"].nunique()),
            "test_weeks": int(test_weekly["week_t"].nunique()),
        },
        "regression": {
            **reg_meta,
            "train_rows": int(len(train_reg)),
            "test_rows": int(len(test_reg)),
            "train_categories": int(train_reg["category"].nunique()),
            "test_categories": int(test_reg["category"].nunique()),
        },
    }

    (cls_out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (reg_out / "metadata_regression.json").write_text(
        json.dumps(metadata["regression"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== BERKA STEP 1 DONE ===")
    print(f"Split week: {split_week.date()}")
    print(f"Classification rows: train={len(train_cls)}, test={len(test_cls)}")
    print(f"Regression rows: train={len(train_reg)}, test={len(test_reg)}")


if __name__ == "__main__":
    main()

