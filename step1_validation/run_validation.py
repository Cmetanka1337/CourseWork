import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Data validation and temporal split")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to CSV file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="priyamchoksi/credit-card-transactions-dataset",
        help="Kaggle dataset slug owner/dataset",
    )
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--user-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--amount-col", type=str, default=None)
    parser.add_argument("--category-col", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="step1_validation/outputs",
        help="Dedicated folder for this step outputs",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_column(columns: list[str], override: str | None, candidates: list[str], label: str) -> str:
    if override:
        if override not in columns:
            raise ValueError(f"Provided {label} column '{override}' is not in dataset")
        return override

    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        f"Could not detect {label} column automatically. Use --{label}-col. Columns: {columns}"
    )


def find_csv_in_dir(root: Path, preferred: str | None = None) -> Path:
    if preferred:
        candidate = root / preferred
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"dataset-file '{preferred}' was not found inside {root}")

    csv_files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset folder: {root}")
    return max(csv_files, key=lambda p: p.stat().st_size)


def resolve_input_csv(args: argparse.Namespace) -> Path:
    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return csv_path

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required when --csv-path is not provided. Install with: pip install kagglehub"
        ) from exc

    dataset_root = Path(kagglehub.dataset_download(args.dataset)).resolve()
    return find_csv_in_dir(dataset_root, args.dataset_file)


def compute_max_gap_days(dates: pd.Series) -> int:
    if dates.empty:
        return 0
    unique_sorted = dates.drop_duplicates().sort_values()
    if len(unique_sorted) <= 1:
        return 0
    diffs = unique_sorted.diff().dt.days.fillna(0)
    return int(diffs.max())


def assign_bucket(amount: pd.Series, q25: float, q75: float) -> pd.Series:
    return pd.Series(
        np.select(
            [amount.eq(0), amount.le(q25), amount.le(q75)],
            [0, 1, 2],
            default=3,
        ),
        index=amount.index,
        dtype=np.int8,
    )


def distribution_to_dict(s: pd.Series) -> dict[str, float]:
    out = {}
    for k in [0, 1, 2, 3]:
        out[str(k)] = float(s.get(k, 0.0))
    return out


def evaluate_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def create_weekly_dataset(df: pd.DataFrame, q25: float, q75: float) -> pd.DataFrame:
    work = df.copy()
    week_key = work["transaction_date"] - pd.to_timedelta(work["transaction_date"].dt.weekday, unit="D")
    work["week_key"] = week_key

    weekly = (
        work.groupby(["user_id", "week_key"], as_index=False)
        .agg(
            amount_sum=("amount", "sum"),
            week_start=("transaction_date", "min"),
            week_end=("transaction_date", "max"),
            txn_count=("amount", "size"),
        )
        .sort_values(["user_id", "week_key"])
        .reset_index(drop=True)
    )

    iso = weekly["week_key"].dt.isocalendar()
    weekly["year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    weekly["bucket"] = assign_bucket(weekly["amount_sum"], q25, q75)
    return weekly[["user_id", "year_week", "amount_sum", "week_start", "week_end", "txn_count", "bucket"]]


def create_lag_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    work = weekly_df.sort_values(["user_id", "week_start"]).copy()
    work["bucket_t_plus_1"] = work.groupby("user_id", sort=False)["bucket"].shift(-1)
    work["amount_t_plus_1"] = work.groupby("user_id", sort=False)["amount_sum"].shift(-1)

    out = work.dropna(subset=["bucket_t_plus_1", "amount_t_plus_1"]).copy()
    out["bucket_t_plus_1"] = out["bucket_t_plus_1"].astype(np.int8)
    out = out.rename(columns={"year_week": "week_t", "bucket": "bucket_t", "amount_sum": "amount_t"})

    return out[["user_id", "week_t", "bucket_t", "amount_t", "bucket_t_plus_1", "amount_t_plus_1"]]


def q_stats(series: pd.Series, labels: tuple[float, ...]) -> dict[str, float]:
    return {f"p{int(q*100)}": float(series.quantile(q)) for q in labels}


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    critical_issues: list[str] = []
    warnings: list[str] = []

    csv_path = resolve_input_csv(args)
    header = pd.read_csv(csv_path, nrows=0)

    user_col = detect_column(
        list(header.columns),
        args.user_col,
        ["user_id", "cc_num", "customer_id", "client_id", "account_id", "user"],
        "user",
    )
    date_col = detect_column(
        list(header.columns),
        args.date_col,
        ["transaction_date", "trans_date_trans_time", "date", "datetime", "timestamp"],
        "date",
    )
    amount_col = detect_column(
        list(header.columns),
        args.amount_col,
        ["amount", "amt", "transaction_amount", "value"],
        "amount",
    )

    category_col = None
    if args.category_col is not None:
        if args.category_col not in header.columns:
            raise ValueError(f"Provided category column '{args.category_col}' is not in dataset")
        category_col = args.category_col
    else:
        for candidate in ["category", "merchant", "merchant_name", "description", "mcc"]:
            if candidate in {c.lower() for c in header.columns}:
                # Recover original case-sensitive name.
                category_col = {c.lower(): c for c in header.columns}[candidate]
                break
        if category_col is None:
            warnings.append("Category column is missing. Continue without category analysis.")

    usecols = [user_col, date_col, amount_col] + ([category_col] if category_col else [])
    df = pd.read_csv(csv_path, usecols=usecols)

    normalized = {
        user_col: "user_id",
        date_col: "transaction_date",
        amount_col: "amount",
    }
    if category_col:
        normalized[category_col] = "category"
    df = df.rename(columns=normalized)

    for required in ["user_id", "transaction_date", "amount"]:
        if required not in df.columns:
            raise RuntimeError(f"Missing required column after normalization: {required}")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.tz_localize(None).dt.floor("D")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "category" not in df.columns:
        df["category"] = "unknown"
    else:
        df["category"] = df["category"].astype(str).str.strip().replace("", "unknown")

    nan_critical = {
        "user_id_nan": int(df["user_id"].isna().sum() + (df["user_id"] == "").sum()),
        "transaction_date_nan": int(df["transaction_date"].isna().sum()),
        "amount_nan": int(df["amount"].isna().sum()),
    }
    if sum(nan_critical.values()) > 0:
        warnings.append(f"Dropped rows with missing critical values: {nan_critical}")

    df = df.dropna(subset=["transaction_date", "amount"])
    df = df[df["user_id"] != ""]

    if df.empty:
        raise RuntimeError("Dataset became empty after cleaning critical fields")

    user_tx_counts = df.groupby("user_id").size()
    if int((user_tx_counts > 50000).sum()) > 0:
        warnings.append("Detected users with more than 50K transactions.")
    if float((df["amount"] > 100000).any()):
        warnings.append("Detected transaction amount > 100K.")

    total_users = int(df["user_id"].nunique())
    total_transactions = int(len(df))
    start_date = df["transaction_date"].min()
    end_date = df["transaction_date"].max()
    date_range_days = int((end_date - start_date).days)

    sanity_pass = True
    if total_users < 200:
        sanity_pass = False
        critical_issues.append("Fewer than 200 unique users")
    if total_transactions < 500000:
        sanity_pass = False
        critical_issues.append("Fewer than 500K transactions")
    if date_range_days < 365:
        sanity_pass = False
        critical_issues.append("Data spans less than 1 year")

    sanity_report = {
        "total_users": total_users,
        "total_transactions": total_transactions,
        "date_range_start": str(start_date.date()),
        "date_range_end": str(end_date.date()),
        "date_range_days": date_range_days,
        "avg_transactions_per_user": float(total_transactions / max(total_users, 1)),
        "avg_transactions_per_user_per_day": float(total_transactions / max(date_range_days * total_users, 1)),
        "status": "PASS" if sanity_pass else "FAIL",
    }

    # Per-user coverage analysis
    grouped = df.groupby("user_id", as_index=True)
    user_cov = grouped.agg(
        first_transaction_date=("transaction_date", "min"),
        last_transaction_date=("transaction_date", "max"),
        total_transactions=("transaction_date", "size"),
        categories_count=("category", "nunique"),
        avg_transaction_amount=("amount", "mean"),
        median_transaction_amount=("amount", "median"),
    )
    user_cov["activity_span_days"] = (
        user_cov["last_transaction_date"] - user_cov["first_transaction_date"]
    ).dt.days.astype(int)
    user_cov["avg_txn_per_week"] = (
        user_cov["total_transactions"] / user_cov["activity_span_days"].clip(lower=1)
    ) * 7.0

    max_gaps = grouped["transaction_date"].apply(compute_max_gap_days).rename("max_consecutive_gap_days")
    user_cov = user_cov.join(max_gaps).reset_index()

    coverage_summary = {
        "total_transactions_quantiles": q_stats(user_cov["total_transactions"], (0.10, 0.25, 0.50, 0.75, 0.90)),
        "activity_span_quantiles": q_stats(user_cov["activity_span_days"], (0.10, 0.50, 0.90)),
        "avg_txn_per_week_quantiles": q_stats(user_cov["avg_txn_per_week"], (0.10, 0.50, 0.90)),
        "users_exceeding_thresholds": {
            "transactions_ge_300": int((user_cov["total_transactions"] >= 300).sum()),
            "activity_span_ge_365": int((user_cov["activity_span_days"] >= 365).sum()),
            "txn_per_week_ge_2_5": int((user_cov["avg_txn_per_week"] >= 2.5).sum()),
            "max_gap_le_90": int((user_cov["max_consecutive_gap_days"] <= 90).sum()),
        },
    }

    # User filtering quality gate
    m_tx = user_cov["total_transactions"] >= 300
    m_span = user_cov["activity_span_days"] >= 365
    m_week = user_cov["avg_txn_per_week"] >= 2.5
    m_gap = user_cov["max_consecutive_gap_days"] <= 90
    valid_mask = m_tx & m_span & m_week & m_gap

    valid_users = set(user_cov.loc[valid_mask, "user_id"])
    excluded_any = user_cov.loc[~valid_mask, "user_id"]

    retention_rate = float(len(valid_users) / max(total_users, 1))
    user_filter_status = "PASS"
    if len(valid_users) < 100:
        user_filter_status = "FAIL"
        critical_issues.append("Valid users after filtering < 100")
    elif retention_rate < 0.50:
        user_filter_status = "WARN"
        warnings.append("Retention rate below 50% after user quality filtering")

    user_filtering = {
        "original_users": total_users,
        "excluded_users_count": int((~valid_mask).sum()),
        "excluded_by_reason": {
            "insufficient_transactions": int((~m_tx).sum()),
            "insufficient_activity_span": int((~m_span).sum()),
            "insufficient_weekly_rate": int((~m_week).sum()),
            "excessive_data_gap": int((~m_gap).sum()),
        },
        "valid_users_count": int(len(valid_users)),
        "retention_rate": float(retention_rate),
        "status": user_filter_status,
        "excluded_user_ids_sample": [str(x) for x in excluded_any.head(50).tolist()],
    }

    coverage_summary["interpretation"] = (
        f"{retention_rate * 100.0:.2f}% of users meet all minimum quality thresholds"
    )

    if len(valid_users) == 0:
        raise RuntimeError("No valid users remain after filtering")

    df_cleaned = df[df["user_id"].isin(valid_users)].copy()
    df_cleaned = df_cleaned.sort_values("transaction_date").reset_index(drop=True)

    if not df_cleaned["transaction_date"].is_monotonic_increasing:
        raise RuntimeError("Global sort by transaction_date failed")

    # Temporal split
    available_dates = np.sort(df_cleaned["transaction_date"].dropna().unique())
    split_idx = int(len(available_dates) * 0.80)
    split_idx = min(max(split_idx, 0), len(available_dates) - 1)
    split_date = pd.Timestamp(available_dates[split_idx])

    df_train = df_cleaned[df_cleaned["transaction_date"] <= split_date].copy()
    df_test = df_cleaned[df_cleaned["transaction_date"] > split_date].copy()

    if len(df_train) + len(df_test) != len(df_cleaned):
        raise RuntimeError("Temporal split lost data")
    if not df_test.empty and df_train["transaction_date"].max() > df_test["transaction_date"].min():
        raise RuntimeError("Temporal overlap detected between train and test")

    train_users = set(df_train["user_id"].unique())
    test_users = set(df_test["user_id"].unique())
    overlapping_users = len(train_users & test_users)
    users_only_train = len(train_users - test_users)
    users_only_test = len(test_users - train_users)

    valid_users_count = max(len(valid_users), 1)
    overlap_ok = overlapping_users >= 0.80 * valid_users_count
    only_test_ok = users_only_test < 0.20 * valid_users_count
    temporal_status = "PASS" if (overlap_ok and only_test_ok) else "FAIL"
    if temporal_status == "FAIL":
        critical_issues.append("Temporal split user overlap constraints failed")

    temporal_report = {
        "split_date": str(split_date.date()),
        "train_period": {
            "start_date": str(df_train["transaction_date"].min().date()),
            "end_date": str(df_train["transaction_date"].max().date()),
            "days_span": int((df_train["transaction_date"].max() - df_train["transaction_date"].min()).days),
            "transactions_count": int(len(df_train)),
            "unique_users": int(len(train_users)),
        },
        "test_period": {
            "start_date": str(df_test["transaction_date"].min().date()) if not df_test.empty else None,
            "end_date": str(df_test["transaction_date"].max().date()) if not df_test.empty else None,
            "days_span": int((df_test["transaction_date"].max() - df_test["transaction_date"].min()).days)
            if len(df_test) > 1
            else 0,
            "transactions_count": int(len(df_test)),
            "unique_users": int(len(test_users)),
        },
        "train_test_ratio": f"{len(df_train)}:{len(df_test)}",
        "overlapping_users": int(overlapping_users),
        "users_only_in_train": int(users_only_train),
        "users_only_in_test": int(users_only_test),
        "status": temporal_status,
    }

    # Bucket computation without leakage (thresholds on train only)
    q25_train = float(df_train["amount"].quantile(0.25))
    q75_train = float(df_train["amount"].quantile(0.75))

    df_train["bucket"] = assign_bucket(df_train["amount"], q25_train, q75_train)
    df_test["bucket"] = assign_bucket(df_test["amount"], q25_train, q75_train)

    train_dist = df_train["bucket"].value_counts(normalize=True).sort_index()
    test_dist = df_test["bucket"].value_counts(normalize=True).sort_index()

    train_counts = df_train["bucket"].value_counts().reindex([0, 1, 2, 3], fill_value=0).to_numpy()
    test_counts = df_test["bucket"].value_counts().reindex([0, 1, 2, 3], fill_value=0).to_numpy()

    contingency = np.vstack([train_counts, test_counts])
    active_cols = contingency.sum(axis=0) > 0
    if active_cols.sum() >= 2:
        chi2_stat, p_value, _, _ = chi2_contingency(contingency[:, active_cols])
    else:
        chi2_stat, p_value = 0.0, 1.0
        warnings.append("Chi-square skipped: fewer than two non-empty bucket columns")

    shifts = {
        "bucket_0": float(abs(train_dist.get(0, 0.0) - test_dist.get(0, 0.0)) * 100.0),
        "bucket_1": float(abs(train_dist.get(1, 0.0) - test_dist.get(1, 0.0)) * 100.0),
        "bucket_2": float(abs(train_dist.get(2, 0.0) - test_dist.get(2, 0.0)) * 100.0),
        "bucket_3": float(abs(train_dist.get(3, 0.0) - test_dist.get(3, 0.0)) * 100.0),
    }
    max_shift = float(max(shifts.values()))

    bucket_status = "PASS"
    if max_shift >= 10 and p_value > 0.05:
        bucket_status = "WARN"
        warnings.append("Bucket distribution shift >=10%, but chi2 is not significant")
    if p_value < 0.05:
        bucket_status = "FAIL"
        critical_issues.append("Significant bucket distribution shift (chi2 p < 0.05)")

    bucket_report = {
        "train_distribution": distribution_to_dict(train_dist),
        "test_distribution": distribution_to_dict(test_dist),
        "distribution_shift_pct": shifts,
        "max_shift_pct": max_shift,
        "chi2_statistic": float(chi2_stat),
        "chi2_p_value": float(p_value),
        "status": bucket_status,
    }

    # Baseline model (naive forecast) on weekly aggregation
    weekly_train = create_weekly_dataset(df_train, q25_train, q75_train)
    weekly_test = create_weekly_dataset(df_test, q25_train, q75_train)

    train_lag = create_lag_features(weekly_train)
    test_lag = create_lag_features(weekly_test)

    if len(weekly_test["year_week"].unique()) < 10:
        warnings.append("Test dataset has fewer than 10 weeks of data")

    baseline_train_true = train_lag["bucket_t_plus_1"].to_numpy(dtype=np.int8)
    baseline_train_pred = train_lag["bucket_t"].to_numpy(dtype=np.int8)
    baseline_test_true = test_lag["bucket_t_plus_1"].to_numpy(dtype=np.int8)
    baseline_test_pred = test_lag["bucket_t"].to_numpy(dtype=np.int8)

    baseline_report = {
        "description": "Naive forecast: bucket(t+1) = bucket(t)",
        "train_metrics": evaluate_baseline(baseline_train_true, baseline_train_pred)
        if len(train_lag) > 0
        else None,
        "test_metrics": evaluate_baseline(baseline_test_true, baseline_test_pred) if len(test_lag) > 0 else None,
        "interpretation": "This is the minimum bar. ML models must beat test F1_macro to be meaningful.",
    }

    baseline_f1_macro = (
        baseline_report["test_metrics"]["f1_macro"] if baseline_report["test_metrics"] is not None else 0.0
    )

    if baseline_f1_macro <= 0.25:
        critical_issues.append("Baseline F1_macro <= 0.25 on test")

    # Export datasets
    train_file = output_dir / "train_dataset.csv"
    test_file = output_dir / "test_dataset.csv"
    train_lag_file = output_dir / "train_lag_features.csv"
    test_lag_file = output_dir / "test_lag_features.csv"

    df_train[["user_id", "transaction_date", "amount", "category", "bucket"]].sort_values(
        "transaction_date"
    ).to_csv(train_file, index=False)
    df_test[["user_id", "transaction_date", "amount", "category", "bucket"]].sort_values(
        "transaction_date"
    ).to_csv(test_file, index=False)
    train_lag.to_csv(train_lag_file, index=False)
    test_lag.to_csv(test_lag_file, index=False)

    # Final decision
    decision_checks = {
        "sanity_check": sanity_report["status"] == "PASS",
        "user_filtering": retention_rate >= 0.50,
        "temporal_split": overlap_ok,
        "bucket_stability": max_shift < 15.0,
        "baseline_signal": baseline_f1_macro > 0.25,
    }

    failed_checks = [k for k, v in decision_checks.items() if not v]
    if len(failed_checks) == 0:
        overall_status = "GO"
        reason = "Dataset is suitable for model training."
    elif len(failed_checks) <= 2:
        overall_status = "BORDERLINE"
        reason = f"Proceed with caution. Issues: {failed_checks}"
    else:
        overall_status = "NO_GO"
        reason = f"Dataset has too many issues: {failed_checks}"

    if user_filter_status == "FAIL" or sanity_report["status"] == "FAIL":
        overall_status = "NO_GO"

    metadata = {
        "Q25_threshold": q25_train,
        "Q75_threshold": q75_train,
        "split_date": str(split_date.date()),
        "valid_users_count": int(len(valid_users)),
        "train_users": int(len(train_users)),
        "test_users": int(len(test_users)),
        "baseline_f1_macro": float(baseline_f1_macro),
        "dataset_verdict": overall_status,
        "bucket_definitions": {
            "0": "amount == 0",
            "1": f"0 < amount <= {q25_train:.6f}",
            "2": f"{q25_train:.6f} < amount <= {q75_train:.6f}",
            "3": f"amount > {q75_train:.6f}",
        },
        "train_weeks": int(weekly_train["year_week"].nunique()),
        "test_weeks": int(weekly_test["year_week"].nunique()),
    }

    report = {
        "validation_report": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": str(csv_path.name),
            "section_1_sanity_check": sanity_report,
            "section_1_2_user_coverage": coverage_summary,
            "section_2_user_filtering": user_filtering,
            "section_3_temporal_split": temporal_report,
            "section_4_bucket_stability": bucket_report,
            "section_5_baseline_model": baseline_report,
            "final_verdict": {
                "overall_status": overall_status,
                "reason_for_verdict": reason,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "recommendations": [
                    "Use train-only thresholds for all future bucketization.",
                    "Use temporal CV only after this strict split.",
                    "Target model should beat baseline F1_macro in test.",
                ],
                "decision_logic": {
                    "sanity_check_passed": decision_checks["sanity_check"],
                    "user_filtering_passed": decision_checks["user_filtering"],
                    "temporal_split_valid": decision_checks["temporal_split"],
                    "bucket_stability_passed": decision_checks["bucket_stability"],
                    "baseline_f1_macro": baseline_f1_macro,
                    "suitable_for_modeling": overall_status in {"GO", "BORDERLINE"},
                },
            },
            "data_export": {
                "train_file_path": str(train_file),
                "test_file_path": str(test_file),
                "lag_features_path": {
                    "train": str(train_lag_file),
                    "test": str(test_lag_file),
                },
                "metadata": {
                    "Q25_threshold": q25_train,
                    "Q75_threshold": q75_train,
                    "bucket_definitions": metadata["bucket_definitions"],
                    "train_users": int(len(train_users)),
                    "test_users": int(len(test_users)),
                    "train_weeks": int(weekly_train["year_week"].nunique()),
                    "test_weeks": int(weekly_test["year_week"].nunique()),
                },
            },
        }
    }

    validation_report_path = output_dir / "validation_report.json"
    metadata_path = output_dir / "metadata.json"

    validation_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== STEP 1 DONE: Data validation and temporal split ===")
    print(f"Verdict: {overall_status}")
    print(f"Q25: {q25_train:.6f}, Q75: {q75_train:.6f}")
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")
    print(f"Baseline test F1_macro: {baseline_f1_macro:.4f}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()

