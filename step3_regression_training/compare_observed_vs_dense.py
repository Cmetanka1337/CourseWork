import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


TARGET = "amount_cat_t_plus_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare observed-only vs dense-format regression framing")
    parser.add_argument("--input-dir", type=str, default="step1_berka_weekly_builder/outputs/regression")
    parser.add_argument("--output-dir", type=str, default="step3_regression_training/outputs")
    return parser.parse_args()


def make_dense(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    key_cols = ["user_id", "week_start"]
    user_weeks = df[key_cols].drop_duplicates().copy()
    user_weeks["_k"] = 1
    cat_df = pd.DataFrame({"category": categories, "_k": 1})
    grid = user_weeks.merge(cat_df, on="_k", how="inner").drop(columns=["_k"])

    merged = grid.merge(
        df,
        on=["user_id", "week_start", "category"],
        how="left",
        suffixes=("", "_src"),
    )

    merged["flow_direction"] = merged["flow_direction"].fillna(merged["category"].astype(str).str.split(":").str[0])
    fill_zero_cols = [
        "amount_cat_t",
        "txn_count_cat_t",
        "amount_cat_t_minus_1",
        "amount_cat_t_minus_2",
        "rolling_mean_4",
        "rolling_std_4",
        "weekly_inflow_t_minus_1",
        "weekly_outflow_t_minus_1",
        "weekly_net_t_minus_1",
        TARGET,
    ]
    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    return merged


def baseline_eval(df: pd.DataFrame) -> dict:
    y = df[TARGET].to_numpy(dtype=float)
    pers = df["amount_cat_t"].to_numpy(dtype=float)
    roll = df["rolling_mean_4"].to_numpy(dtype=float)
    return {
        "rows": int(len(df)),
        "target_zero_rate": float(np.mean((y == 0).astype(float))),
        "persistence_mae": float(mean_absolute_error(y, pers)),
        "rolling4_mae": float(mean_absolute_error(y, roll)),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_regression.csv")
    test_df = pd.read_csv(input_dir / "test_regression.csv")

    categories = sorted(train_df["category"].dropna().astype(str).unique().tolist())
    dense_test = make_dense(test_df, categories)

    observed_stats = baseline_eval(test_df)
    dense_stats = baseline_eval(dense_test)

    payload = {
        "categories": len(categories),
        "observed_only_test": observed_stats,
        "dense_test": dense_stats,
        "delta_dense_minus_observed": {
            "persistence_mae": dense_stats["persistence_mae"] - observed_stats["persistence_mae"],
            "rolling4_mae": dense_stats["rolling4_mae"] - observed_stats["rolling4_mae"],
            "target_zero_rate": dense_stats["target_zero_rate"] - observed_stats["target_zero_rate"],
        },
    }

    (output_dir / "observed_vs_dense_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = f"""# Observed-only vs Dense (Regression Framing)

- Categories used in dense grid: **{payload['categories']}**

## Observed-only test
- Rows: **{observed_stats['rows']}**
- Zero-rate target: **{observed_stats['target_zero_rate']:.4f}**
- Persistence MAE: **{observed_stats['persistence_mae']:.4f}**
- Rolling(4) MAE: **{observed_stats['rolling4_mae']:.4f}**

## Dense test
- Rows: **{dense_stats['rows']}**
- Zero-rate target: **{dense_stats['target_zero_rate']:.4f}**
- Persistence MAE: **{dense_stats['persistence_mae']:.4f}**
- Rolling(4) MAE: **{dense_stats['rolling4_mae']:.4f}**

## Delta (dense - observed)
- Delta zero-rate: **{payload['delta_dense_minus_observed']['target_zero_rate']:.4f}**
- Delta persistence MAE: **{payload['delta_dense_minus_observed']['persistence_mae']:.4f}**
- Delta rolling(4) MAE: **{payload['delta_dense_minus_observed']['rolling4_mae']:.4f}**
"""
    (output_dir / "observed_vs_dense_report.md").write_text(md, encoding="utf-8")
    print("Observed vs dense report written")


if __name__ == "__main__":
    main()

