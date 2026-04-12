import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from main import (
    AMOUNT_COL_CANDIDATES,
    CATEGORY_COL_CANDIDATES,
    DATE_COL_CANDIDATES,
    USER_COL_CANDIDATES,
    collapse_categories,
    detect_column,
    ensure_dir,
    preprocess,
    resolve_input_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial Intensity Range validation (Stage 1-4)")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to local CSV file")
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
    parser.add_argument("--collapse-mode", choices=["manual", "top", "tfidf"], default="manual")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="outputs/intensity")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for input rows")
    parser.add_argument("--max-users", type=int, default=None, help="Optional cap for users after preprocessing")
    parser.add_argument("--max-train-rows", type=int, default=250000, help="Cap train rows for RF to limit RAM")
    parser.add_argument("--max-test-rows", type=int, default=150000, help="Cap test rows for RF to limit RAM")
    parser.add_argument("--rf-n-estimators", type=int, default=120)
    parser.add_argument("--rf-max-depth", type=int, default=14)
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    return parser.parse_args()


def future_7day_sum(series: pd.Series) -> pd.Series:
    return series.shift(-1).rolling(7, min_periods=7).sum().shift(-6)


def assign_buckets(y: pd.Series, q_low: float, q_high: float) -> pd.Series:
    out = pd.Series(np.nan, index=y.index)
    out[y == 0] = 0
    mask_pos = y > 0
    out[mask_pos & (y <= q_low)] = 1
    out[mask_pos & (y > q_low) & (y <= q_high)] = 2
    out[mask_pos & (y > q_high)] = 3
    return out


def tune_quantile_thresholds(y: pd.Series) -> tuple[float, float, dict]:
    pos = y[y > 0]
    if pos.empty:
        return 0.0, 0.0, {"mode": "degenerate", "quantiles": [0.0, 0.0]}

    q25 = float(pos.quantile(0.25))
    q75 = float(pos.quantile(0.75))
    base_buckets = assign_buckets(y, q25, q75)
    base_dist = base_buckets.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)

    # Trigger tuning only when one of positive classes is too small.
    if min(base_dist[1], base_dist[2], base_dist[3]) >= 0.10:
        return q25, q75, {"mode": "default", "quantiles": [0.25, 0.75]}

    best = None
    for ql in np.arange(0.20, 0.50, 0.05):
        for qh in np.arange(0.55, 0.90, 0.05):
            if ql >= qh:
                continue
            low = float(pos.quantile(float(ql)))
            high = float(pos.quantile(float(qh)))
            buckets = assign_buckets(y, low, high)
            dist = buckets.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)
            min_pos_share = min(dist[1], dist[2], dist[3])
            score = (min_pos_share, -abs(dist[1] - dist[2]) - abs(dist[2] - dist[3]))
            if best is None or score > best[0]:
                best = (score, low, high, ql, qh)

    if best is None:
        return q25, q75, {"mode": "fallback_default", "quantiles": [0.25, 0.75]}

    _, low, high, ql, qh = best
    return low, high, {"mode": "tuned", "quantiles": [float(ql), float(qh)]}


def stage1_discretization(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = ts.copy()
    df = df.sort_values(["user_id", "category", "event_date"]).reset_index(drop=True)
    df["target_7d_sum"] = (
        df.groupby(["user_id", "category"], sort=False)["amount"].transform(future_7day_sum)
    )

    valid_y = df["target_7d_sum"].dropna()
    q_low, q_high, tune_info = tune_quantile_thresholds(valid_y)
    df["bucket"] = assign_buckets(df["target_7d_sum"], q_low, q_high)

    dist = (
        df["bucket"].dropna().astype(int).value_counts(normalize=True).sort_index().reindex([0, 1, 2, 3], fill_value=0.0)
    )
    means = (
        df.groupby("bucket", dropna=True)["target_7d_sum"].mean().reindex([0, 1, 2, 3], fill_value=np.nan)
    )

    summary = {
        "threshold_low": q_low,
        "threshold_high": q_high,
        "tuning": tune_info,
        "class_distribution_pct": {str(i): float(100.0 * dist.loc[i]) for i in [0, 1, 2, 3]},
        "mean_amount_per_class": {str(i): float(means.loc[i]) if pd.notna(means.loc[i]) else np.nan for i in [0, 1, 2, 3]},
    }
    return df, summary


def stage2_temporal_signal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.sort_values(["user_id", "category", "event_date"]).copy()
    work["bucket_t_plus_7"] = work.groupby(["user_id", "category"], sort=False)["bucket"].shift(-7)
    pairs = work.dropna(subset=["bucket", "bucket_t_plus_7"]).copy()
    pairs["bucket"] = pairs["bucket"].astype(int)
    pairs["bucket_t_plus_7"] = pairs["bucket_t_plus_7"].astype(int)

    transition = pd.crosstab(pairs["bucket"], pairs["bucket_t_plus_7"], normalize="index")
    transition = transition.reindex(index=[0, 1, 2, 3], columns=[0, 1, 2, 3], fill_value=0.0)

    if len(pairs) >= 3:
        global_spearman = float(spearmanr(pairs["bucket"], pairs["bucket_t_plus_7"]).correlation)
    else:
        global_spearman = np.nan

    per_cat = {}
    for cat in ["groceries", "fuel", "shopping"]:
        sub = pairs[pairs["category"] == cat]
        if len(sub) >= 3:
            per_cat[cat] = float(spearmanr(sub["bucket"], sub["bucket_t_plus_7"]).correlation)
        else:
            per_cat[cat] = np.nan

    summary = {
        "global_spearman": global_spearman,
        "per_category_spearman": per_cat,
    }
    return transition, summary


def shannon_entropy_from_series(vals: pd.Series) -> float:
    p = vals.value_counts(normalize=True)
    if p.empty:
        return np.nan
    return float(-(p * np.log2(p)).sum())


def stage3_user_entropy(df: pd.DataFrame, figures_dir: Path) -> tuple[pd.DataFrame, dict]:
    user_daily = df.groupby(["user_id", "event_date"], as_index=False)["amount"].sum()
    user_daily = user_daily.sort_values(["user_id", "event_date"]).reset_index(drop=True)
    user_daily["target_7d_sum"] = user_daily.groupby("user_id", sort=False)["amount"].transform(future_7day_sum)

    y = user_daily["target_7d_sum"].dropna()
    q_low, q_high, _ = tune_quantile_thresholds(y)
    user_daily["bucket"] = assign_buckets(user_daily["target_7d_sum"], q_low, q_high)

    entropy_table = (
        user_daily.dropna(subset=["bucket"])
        .groupby("user_id")["bucket"]
        .apply(shannon_entropy_from_series)
        .reset_index(name="entropy")
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(entropy_table["entropy"].dropna(), bins=40)
    plt.title("User bucket entropy histogram")
    plt.xlabel("Shannon entropy (bits)")
    plt.tight_layout()
    plt.savefig(figures_dir / "stage3_user_entropy_hist.png", dpi=140)
    plt.close()

    q10 = float(entropy_table["entropy"].quantile(0.10))
    q50 = float(entropy_table["entropy"].quantile(0.50))
    q90 = float(entropy_table["entropy"].quantile(0.90))

    def level(x: float) -> str:
        if x < 0.8:
            return "low"
        if x < 1.4:
            return "medium"
        return "high"

    entropy_table["entropy_level"] = entropy_table["entropy"].map(level)
    shares = entropy_table["entropy_level"].value_counts(normalize=True).to_dict()

    summary = {
        "entropy_quantiles": {"p10": q10, "p50": q50, "p90": q90},
        "entropy_share": {
            "low": float(shares.get("low", 0.0)),
            "medium": float(shares.get("medium", 0.0)),
            "high": float(shares.get("high", 0.0)),
        },
    }
    return entropy_table, summary


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.sort_values(["user_id", "category", "event_date"]).copy()

    # Required lag features.
    frame["prev_bucket_1"] = frame.groupby(["user_id", "category"], sort=False)["bucket"].shift(1)
    frame["prev_bucket_2"] = frame.groupby(["user_id", "category"], sort=False)["bucket"].shift(2)

    frame["rolling_sum_7d"] = (
        frame.groupby(["user_id", "category"], sort=False)["amount"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    )
    frame["rolling_sum_30d"] = (
        frame.groupby(["user_id", "category"], sort=False)["amount"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
    )

    frame["is_weekend"] = frame["event_date"].dt.dayofweek.isin([5, 6]).astype(int)
    frame["is_month_end"] = frame["event_date"].dt.is_month_end.astype(int)

    user_daily = frame.groupby(["user_id", "event_date"], as_index=False)["amount"].sum()
    user_daily = user_daily.sort_values(["user_id", "event_date"])
    user_daily["cumulative_spend_30d"] = (
        user_daily.groupby("user_id", sort=False)["amount"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
    )

    frame = frame.merge(
        user_daily[["user_id", "event_date", "cumulative_spend_30d"]],
        on=["user_id", "event_date"],
        how="left",
    )

    model_cols = [
        "prev_bucket_1",
        "prev_bucket_2",
        "rolling_sum_7d",
        "rolling_sum_30d",
        "is_weekend",
        "is_month_end",
        "cumulative_spend_30d",
    ]

    out = frame.dropna(subset=["bucket", *model_cols]).copy()
    out["bucket"] = out["bucket"].astype(np.int8)
    out["prev_bucket_1"] = out["prev_bucket_1"].astype(np.int8)
    out["prev_bucket_2"] = out["prev_bucket_2"].astype(np.int8)
    out["is_weekend"] = out["is_weekend"].astype(np.int8)
    out["is_month_end"] = out["is_month_end"].astype(np.int8)
    for col in ["rolling_sum_7d", "rolling_sum_30d", "cumulative_spend_30d"]:
        out[col] = out[col].astype(np.float32)
    return out


def _cap_rows(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def stage4_baseline_model(
    df: pd.DataFrame,
    max_train_rows: int,
    max_test_rows: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    model_df = build_model_frame(df)

    feature_cols = [
        "prev_bucket_1",
        "prev_bucket_2",
        "rolling_sum_7d",
        "rolling_sum_30d",
        "is_weekend",
        "is_month_end",
        "cumulative_spend_30d",
    ]

    split_date = model_df["event_date"].quantile(0.80)
    train = model_df[model_df["event_date"] <= split_date]
    test = model_df[model_df["event_date"] > split_date]

    if train.empty or test.empty:
        raise RuntimeError("Train/test split is empty. Cannot run Stage 4 model.")

    train = _cap_rows(train, max_train_rows)
    test = _cap_rows(test, max_test_rows)

    X_train = train[feature_cols]
    y_train = train["bucket"]
    X_test = test[feature_cols]
    y_test = test["bucket"]

    clf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=42,
        n_jobs=rf_n_jobs,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    most_freq = int(y_train.value_counts().idxmax())
    base_pred = np.full(len(y_test), most_freq, dtype=int)

    metrics = {
        "model_f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "model_accuracy": float(accuracy_score(y_test, pred)),
        "model_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "baseline_f1_weighted": float(f1_score(y_test, base_pred, average="weighted")),
        "baseline_accuracy": float(accuracy_score(y_test, base_pred)),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y_test, base_pred)),
        "train_rows_used": int(len(train)),
        "test_rows_used": int(len(test)),
        "rf_n_estimators": int(rf_n_estimators),
        "rf_max_depth": int(rf_max_depth),
        "rf_n_jobs": int(rf_n_jobs),
    }

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1", "true_2", "true_3"], columns=["pred_0", "pred_1", "pred_2", "pred_3"])

    return cm_df, metrics


def compute_verdict(stage1: dict, stage2: dict, stage4: dict) -> dict:
    mean_c1 = stage1["mean_amount_per_class"].get("1", np.nan)
    mean_c2 = stage1["mean_amount_per_class"].get("2", np.nan)
    class_sep_ratio = float(mean_c2 / mean_c1) if pd.notna(mean_c1) and mean_c1 > 0 else np.nan
    class_sep_ok = bool(pd.notna(class_sep_ratio) and class_sep_ratio >= 2.0)

    model_f1 = stage4["model_f1_weighted"]
    base_f1 = stage4["baseline_f1_weighted"]
    lift_ratio = float(model_f1 / base_f1) if base_f1 > 0 else np.nan
    lift_ok = bool(pd.notna(lift_ratio) and model_f1 >= base_f1 * 1.10)

    spears = stage2["per_category_spearman"]
    best_cat = max([v for v in spears.values() if pd.notna(v)], default=np.nan)
    temporal_ok = bool(pd.notna(best_cat) and best_cat >= 0.15)

    passed = int(class_sep_ok) + int(lift_ok) + int(temporal_ok)
    verdict = "GO" if passed >= 2 else "NO GO"

    return {
        "class_separation_ratio_c2_c1": class_sep_ratio,
        "class_separation_ok": class_sep_ok,
        "predictive_lift_ratio": lift_ratio,
        "predictive_lift_ok": lift_ok,
        "best_category_spearman": best_cat,
        "temporal_signal_ok": temporal_ok,
        "criteria_passed": passed,
        "verdict": verdict,
    }


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    work = df.copy()
    work = work.reset_index()
    headers = [str(c) for c in work.columns]

    def _fmt(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v)

    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in work.iterrows():
        rows.append("| " + " | ".join(_fmt(row[c]) for c in work.columns) + " |")
    return "\n".join(rows)


def write_markdown(path: Path, report: dict, transition: pd.DataFrame, cm: pd.DataFrame) -> None:
    s1 = report["stage1"]
    s2 = report["stage2"]
    s3 = report["stage3"]
    s4 = report["stage4"]
    vd = report["verdict"]

    lines = [
        "# Financial Intensity Range Validation",
        "",
        "## Stage 1 - Target discretization",
        f"- threshold_low: {s1['threshold_low']:.4f}",
        f"- threshold_high: {s1['threshold_high']:.4f}",
        f"- quantile_mode: {s1['tuning']['mode']} ({s1['tuning']['quantiles']})",
        f"- class_distribution_pct: {s1['class_distribution_pct']}",
        f"- mean_amount_per_class: {s1['mean_amount_per_class']}",
        "",
        "## Stage 2 - Temporal signal",
        f"- global_spearman: {s2['global_spearman']:.4f}",
        f"- per_category_spearman: {s2['per_category_spearman']}",
        "",
        "Transition matrix P(bucket_t+7 | bucket_t):",
        "",
        _df_to_markdown_table(transition),
        "",
        "## Stage 3 - User entropy",
        f"- entropy_quantiles: {s3['entropy_quantiles']}",
        f"- entropy_share: {s3['entropy_share']}",
        "",
        "## Stage 4 - Baseline model test",
        f"- model_f1_weighted: {s4['model_f1_weighted']:.4f}",
        f"- baseline_f1_weighted: {s4['baseline_f1_weighted']:.4f}",
        f"- model_accuracy: {s4['model_accuracy']:.4f}",
        f"- baseline_accuracy: {s4['baseline_accuracy']:.4f}",
        f"- model_balanced_accuracy: {s4['model_balanced_accuracy']:.4f}",
        f"- baseline_balanced_accuracy: {s4['baseline_balanced_accuracy']:.4f}",
        f"- train_rows_used: {s4['train_rows_used']}",
        f"- test_rows_used: {s4['test_rows_used']}",
        f"- rf_config: n_estimators={s4['rf_n_estimators']}, max_depth={s4['rf_max_depth']}, n_jobs={s4['rf_n_jobs']}",
        "",
        "Confusion matrix:",
        "",
        _df_to_markdown_table(cm),
        "",
        "## Verdict",
        f"- class_separation_ratio_c2_c1: {vd['class_separation_ratio_c2_c1']:.4f}",
        f"- predictive_lift_ratio: {vd['predictive_lift_ratio']:.4f}",
        f"- best_category_spearman: {vd['best_category_spearman']:.4f}",
        f"- criteria_passed: {vd['criteria_passed']}/3",
        f"- FINAL VERDICT: {vd['verdict']}",
    ]

    if vd["verdict"] == "NO GO":
        lines.extend(
            [
                "",
                "## NO GO reason",
                "- Lack of temporal dependency at bucket level (weak category Spearman).",
                "- Limited predictive lift over the most-frequent-class baseline.",
                "- High sparsity inherited from the original daily signal.",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def stage_2_daily_grid_memory_safe(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["user_id", "event_date", "category"], as_index=False)["amount"]
        .sum()
        .sort_values(["user_id", "event_date", "category"])
    )

    all_days = pd.date_range(grouped["event_date"].min(), grouped["event_date"].max(), freq="D")
    days_df = pd.DataFrame({"event_date": all_days})

    pairs = grouped[["user_id", "category"]].drop_duplicates().copy()
    pairs["_k"] = 1
    days_df["_k"] = 1

    full = pairs.merge(days_df, on="_k", how="inner").drop(columns=["_k"])
    dense = full.merge(grouped, on=["user_id", "category", "event_date"], how="left")
    dense["amount"] = dense["amount"].fillna(0.0).astype(np.float32)

    return dense.sort_values(["user_id", "category", "event_date"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    csv_path = resolve_input_csv(args)

    header = pd.read_csv(csv_path, nrows=0)
    user_col = detect_column(header, args.user_col, USER_COL_CANDIDATES, "user")
    date_col = detect_column(header, args.date_col, DATE_COL_CANDIDATES, "date")
    amount_col = detect_column(header, args.amount_col, AMOUNT_COL_CANDIDATES, "amount")
    category_col = detect_column(header, args.category_col, CATEGORY_COL_CANDIDATES, "category")

    raw = pd.read_csv(
        csv_path,
        usecols=[user_col, date_col, amount_col, category_col],
        nrows=args.max_rows,
    )

    df = preprocess(raw, user_col, date_col, amount_col, category_col)
    if args.max_users is not None and df["user_id"].nunique() > args.max_users:
        keep = df["user_id"].drop_duplicates().head(args.max_users)
        df = df[df["user_id"].isin(set(keep))].copy()

    df["category"] = collapse_categories(df["raw_category"], args.collapse_mode, args.top_n, args.n_clusters)
    df["category"] = df["category"].astype("category")
    ts = stage_2_daily_grid_memory_safe(df)

    stage1_df, stage1_summary = stage1_discretization(ts)
    transition, stage2_summary = stage2_temporal_signal(stage1_df)
    entropy_table, stage3_summary = stage3_user_entropy(stage1_df, figures_dir)
    cm_df, stage4_summary = stage4_baseline_model(
        stage1_df,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_n_jobs=args.rf_n_jobs,
    )

    verdict = compute_verdict(stage1_summary, stage2_summary, stage4_summary)

    transition.to_csv(tables_dir / "stage2_transition_matrix.csv")
    entropy_table.to_csv(tables_dir / "stage3_user_entropy.csv", index=False)
    cm_df.to_csv(tables_dir / "stage4_confusion_matrix.csv")

    report = {
        "input_csv": str(csv_path),
        "columns": {
            "user": user_col,
            "date": date_col,
            "amount": amount_col,
            "category": category_col,
        },
        "stage1": stage1_summary,
        "stage2": stage2_summary,
        "stage3": stage3_summary,
        "stage4": stage4_summary,
        "verdict": verdict,
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "result.md", report, transition, cm_df)

    print("=== DONE: Financial Intensity Range analysis completed ===")
    print(f"Report: {output_dir / 'result.md'}")
    print(f"JSON: {output_dir / 'report.json'}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()

