import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_SEED = 42

USER_COL_CANDIDATES = [
    "user_id",
    "customer_id",
    "client_id",
    "card_id",
    "cc_num",
    "account_id",
    "user",
    "customer",
]
DATE_COL_CANDIDATES = [
    "timestamp",
    "transaction_date",
    "trans_date",
    "trans_date_trans_time",
    "date",
    "datetime",
    "time",
]
AMOUNT_COL_CANDIDATES = ["amount", "amt", "transaction_amount", "value"]
CATEGORY_COL_CANDIDATES = [
    "category",
    "merchant",
    "merchant_name",
    "mcc",
    "description",
    "transaction_description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full dataset validation pipeline (Stage 0-9)")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to local CSV file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="priyamchoksi/credit-card-transactions-dataset",
        help="Kaggle dataset slug owner/dataset",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Specific file inside downloaded dataset directory",
    )
    parser.add_argument("--user-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--amount-col", type=str, default=None)
    parser.add_argument("--category-col", type=str, default=None)
    parser.add_argument(
        "--collapse-mode",
        choices=["manual", "top", "tfidf"],
        default="manual",
        help="Category collapse strategy for Stage 1",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top-N categories for top mode")
    parser.add_argument("--n-clusters", type=int, default=15, help="Cluster count for tfidf mode")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_column(df: pd.DataFrame, override: str | None, candidates: list[str], label: str) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Provided {label} column '{override}' is not in dataset")
        return override

    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        f"Could not detect {label} column automatically. Use --{label}-col. Columns: {list(df.columns)}"
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


def preprocess(df: pd.DataFrame, user_col: str, date_col: str, amount_col: str, category_col: str) -> pd.DataFrame:
    out = df[[user_col, date_col, amount_col, category_col]].copy()
    out.columns = ["user_id", "event_time", "amount", "raw_category"]

    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["event_time", "user_id", "amount", "raw_category"])
    out["event_date"] = out["event_time"].dt.tz_convert(None).dt.floor("D")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["amount"])

    out["user_id"] = out["user_id"].astype(str).str.strip()
    out["raw_category"] = out["raw_category"].astype(str).str.strip().str.lower()
    out = out[(out["user_id"] != "") & (out["raw_category"] != "")]
    return out


def manual_map_category(text: str) -> str:
    t = text.lower()
    mapping_rules = {
        "groceries": ["grocery", "supermarket", "food market", "mart"],
        "fuel": ["fuel", "gas", "petrol", "diesel", "station"],
        "shopping": ["shop", "store", "mall", "retail", "amazon", "ebay"],
        "restaurants": ["restaurant", "cafe", "coffee", "bar", "pizza", "burger", "dining"],
        "utilities": ["utility", "electric", "water", "internet", "phone", "telecom", "bill"],
        "transport": ["uber", "lyft", "taxi", "bus", "metro", "train", "flight"],
        "healthcare": ["pharmacy", "hospital", "clinic", "medical", "drug"],
        "entertainment": ["movie", "cinema", "netflix", "spotify", "game", "entertain"],
        "travel": ["hotel", "booking", "airbnb", "travel", "trip"],
    }
    for label, keys in mapping_rules.items():
        if any(k in t for k in keys):
            return label
    return "other"


def collapse_categories(series: pd.Series, mode: str, top_n: int, n_clusters: int) -> pd.Series:
    s = series.fillna("other").astype(str).str.lower().str.strip()

    if mode == "manual":
        return s.map(manual_map_category)

    if mode == "top":
        top_values = set(s.value_counts().head(top_n).index)
        return s.where(s.isin(top_values), "other")

    # TF-IDF + KMeans for merchant-like high-cardinality text labels
    if s.nunique() < n_clusters:
        n_clusters = max(2, s.nunique())

    if s.nunique() <= 2:
        return s

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform(s)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(matrix)
    return pd.Series([f"cluster_{x:02d}" for x in labels], index=s.index)


def stage_1_category_diagnostics(raw_categories: pd.Series, collapsed_categories: pd.Series) -> dict:
    raw = raw_categories.astype(str).str.lower().str.strip()
    collapsed = collapsed_categories.astype(str).str.lower().str.strip()
    return {
        "raw_unique_categories": int(raw.nunique()),
        "collapsed_unique_categories": int(collapsed.nunique()),
        "top10_raw": raw.value_counts().head(10).to_dict(),
        "top10_collapsed": collapsed.value_counts().head(10).to_dict(),
    }


def stage_0_sanity(df: pd.DataFrame) -> dict:
    tx_per_user = df.groupby("user_id").size()
    result = {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "min_date": str(df["event_date"].min().date()),
        "max_date": str(df["event_date"].max().date()),
        "avg_tx_per_user": float(tx_per_user.mean()),
        "median_tx_per_user": float(tx_per_user.median()),
        "users_with_200plus_tx": int((tx_per_user >= 200).sum()),
        "share_users_with_200plus_tx": float((tx_per_user >= 200).mean()),
    }
    return result


def stage_2_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["user_id", "event_date", "category"], as_index=False)["amount"]
        .sum()
        .sort_values(["user_id", "event_date", "category"])
    )

    users = grouped["user_id"].unique()
    categories = grouped["category"].unique()
    all_days = pd.date_range(grouped["event_date"].min(), grouped["event_date"].max(), freq="D")

    full_index = pd.MultiIndex.from_product(
        [users, all_days, categories], names=["user_id", "event_date", "category"]
    )
    dense = grouped.set_index(["user_id", "event_date", "category"]).reindex(full_index, fill_value=0.0)
    dense = dense.reset_index()
    return dense


def stage_3_sparsity(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = (
        ts.assign(active=(ts["amount"] > 0).astype(int))
        .groupby(["user_id", "category"], as_index=False)["active"]
        .mean()
    )
    stats["active_days_pct"] = 100.0 * stats["active"]
    stats = stats.drop(columns=["active"])

    summary = {
        "median_active_days_pct": float(stats["active_days_pct"].median()),
        "share_pairs_below_20pct": float((stats["active_days_pct"] < 20).mean()),
        "share_pairs_20_to_50pct": float(
            ((stats["active_days_pct"] >= 20) & (stats["active_days_pct"] < 50)).mean()
        ),
        "share_pairs_50plus_pct": float((stats["active_days_pct"] >= 50).mean()),
    }
    return stats, summary


def stage_4_distribution(ts: pd.DataFrame, figures_dir: Path) -> dict:
    amounts = ts["amount"].to_numpy(dtype=float)
    zero_share = float(np.mean(amounts == 0))

    plt.figure(figsize=(9, 5))
    sns.histplot(amounts, bins=80)
    plt.title("Histogram of daily spend")
    plt.xlabel("Daily spend")
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_hist_daily_spend.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(np.log1p(amounts), bins=80)
    plt.title("Histogram of log1p(daily spend)")
    plt.xlabel("log1p(daily spend)")
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_hist_log_daily_spend.png", dpi=140)
    plt.close()

    return {
        "zero_share": zero_share,
        "mean": float(np.mean(amounts)),
        "median": float(np.median(amounts)),
        "p95": float(np.percentile(amounts, 95)),
        "p99": float(np.percentile(amounts, 99)),
        "skewness": float(np.squeeze(skew(amounts, bias=False))) if len(amounts) > 2 else np.nan,
    }


def _safe_autocorr(values: pd.Series, lag: int) -> float:
    if len(values) <= lag + 1:
        return np.nan
    if np.isclose(values.var(), 0.0) or np.isclose(values.std(), 0.0):
        return np.nan
    shifted = values.shift(lag)
    aligned = pd.DataFrame({"x": values, "y": shifted}).dropna()
    if aligned.empty:
        return np.nan
    if np.isclose(aligned["x"].std(), 0.0) or np.isclose(aligned["y"].std(), 0.0):
        return np.nan
    return float(aligned["x"].corr(aligned["y"]))


def stage_5_autocorrelation(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (user_id, category), g in ts.groupby(["user_id", "category"]):
        v = g.sort_values("event_date")["amount"]
        rows.append(
            {
                "user_id": user_id,
                "category": category,
                "corr_lag1": _safe_autocorr(v, 1),
                "corr_lag7": _safe_autocorr(v, 7),
                "corr_lag30": _safe_autocorr(v, 30),
            }
        )

    ac = pd.DataFrame(rows)
    corr_ref = ac["corr_lag7"].dropna()
    summary = {
        "median_corr_lag1": float(ac["corr_lag1"].median(skipna=True)),
        "median_corr_lag7": float(ac["corr_lag7"].median(skipna=True)),
        "median_corr_lag30": float(ac["corr_lag30"].median(skipna=True)),
        "share_lag7_below_0_1": float((corr_ref < 0.1).mean()) if not corr_ref.empty else np.nan,
        "share_lag7_0_1_to_0_3": float(((corr_ref >= 0.1) & (corr_ref < 0.3)).mean())
        if not corr_ref.empty
        else np.nan,
        "share_lag7_0_3_plus": float((corr_ref >= 0.3).mean()) if not corr_ref.empty else np.nan,
    }
    return ac, summary


def _future_7day_target(s: pd.Series) -> pd.Series:
    return s.shift(-1).rolling(7, min_periods=7).sum().shift(-6)


def stage_6_baselines(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (user_id, category), g in ts.groupby(["user_id", "category"]):
        s = g.sort_values("event_date")["amount"].reset_index(drop=True)
        y = _future_7day_target(s)
        pred_mean = pd.Series(np.full(len(s), s.mean() * 7.0))
        pred_roll7 = s.rolling(7, min_periods=1).mean() * 7.0
        pred_last = s * 7.0

        frame = pd.DataFrame(
            {
                "y": y,
                "pred_mean": pred_mean,
                "pred_roll7": pred_roll7,
                "pred_last": pred_last,
            }
        ).dropna()

        if frame.empty:
            continue

        for model_col, model_name in [
            ("pred_mean", "mean_predictor"),
            ("pred_roll7", "rolling_mean_7"),
            ("pred_last", "last_value"),
        ]:
            err = frame[model_col] - frame["y"]
            rows.append(
                {
                    "user_id": user_id,
                    "category": category,
                    "baseline": model_name,
                    "n_points": int(len(frame)),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                }
            )

    return pd.DataFrame(rows)


def stage_7_target_validation(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (user_id, category), g in ts.groupby(["user_id", "category"]):
        s = g.sort_values("event_date")["amount"].reset_index(drop=True)
        y = _future_7day_target(s)
        roll = s.rolling(7, min_periods=1).mean() * 7.0
        tmp = pd.DataFrame({"target": y, "rolling_mean": roll}).dropna()
        if len(tmp) < 3:
            corr = np.nan
        elif np.isclose(tmp["target"].std(), 0.0) or np.isclose(tmp["rolling_mean"].std(), 0.0):
            corr = np.nan
        else:
            corr = tmp["target"].corr(tmp["rolling_mean"])
        rows.append({"user_id": user_id, "category": category, "corr_target_vs_roll7": corr})

    tv = pd.DataFrame(rows)
    valid = tv["corr_target_vs_roll7"].dropna()
    summary = {
        "median_corr": float(valid.median()) if not valid.empty else np.nan,
        "share_corr_0_9_plus": float((valid >= 0.9).mean()) if not valid.empty else np.nan,
    }
    return tv, summary


def stage_8_user_variance(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_user_daily = ts.groupby(["user_id", "event_date"], as_index=False)["amount"].sum()
    user_stats = per_user_daily.groupby("user_id")["amount"].agg(["mean", "std", "median"]).reset_index()
    user_stats.columns = ["user_id", "daily_mean", "daily_std", "daily_median"]

    profile = ts.groupby(["user_id", "category"], as_index=False)["amount"].mean()
    profile_wide = profile.pivot(index="user_id", columns="category", values="amount").fillna(0.0)
    between_user_var = float(profile_wide.var(axis=0).mean()) if not profile_wide.empty else np.nan

    summary = {
        "cv_daily_mean_spend": float(user_stats["daily_mean"].std() / (user_stats["daily_mean"].mean() + 1e-9)),
        "between_user_category_variance": between_user_var,
    }
    return user_stats, summary


def stage_9_category_signal(sparsity: pd.DataFrame, ac: pd.DataFrame) -> pd.DataFrame:
    focus = ["groceries", "fuel", "utilities"]
    sp = sparsity.groupby("category", as_index=False)["active_days_pct"].median()
    ac_m = ac.groupby("category", as_index=False)["corr_lag7"].median()
    out = sp.merge(ac_m, on="category", how="outer")
    out = out[out["category"].isin(focus)].copy()
    return out.sort_values("category")


def stage_10_event_level_analysis(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    baseline_rows = []

    for (user_id, category), g in ts.groupby(["user_id", "category"]):
        g = g.sort_values("event_date").reset_index(drop=True)
        y = (g["amount"] > 0).astype(int)
        event_rate = float(y.mean())

        lag1 = _safe_autocorr(y.astype(float), 1)
        lag7 = _safe_autocorr(y.astype(float), 7)

        prev = y.shift(1)
        trans = pd.DataFrame({"prev": prev, "curr": y}).dropna()
        if trans.empty:
            p_1_given_0 = np.nan
            p_1_given_1 = np.nan
            freq_accuracy = np.nan
            freq_balanced_accuracy = np.nan
        else:
            prev0 = trans[trans["prev"] == 0]
            prev1 = trans[trans["prev"] == 1]
            p_1_given_0 = float(prev0["curr"].mean()) if not prev0.empty else np.nan
            p_1_given_1 = float(prev1["curr"].mean()) if not prev1.empty else np.nan

            y_true = trans["curr"].astype(int)
            always0_pred = np.zeros(len(y_true), dtype=int)
            freq_pred = np.full(len(y_true), int(event_rate >= 0.5), dtype=int)

            y_arr = y_true.to_numpy()
            always0_accuracy = float(np.mean(always0_pred == y_arr))
            freq_accuracy = float(np.mean(freq_pred == y_arr))

            tpr = float(np.sum((freq_pred == 1) & (y_arr == 1)) / max(int(np.sum(y_arr == 1)), 1))
            tnr = float(np.sum((freq_pred == 0) & (y_arr == 0)) / max(int(np.sum(y_arr == 0)), 1))
            freq_balanced_accuracy = 0.5 * (tpr + tnr)

            baseline_rows.append(
                {
                    "user_id": user_id,
                    "category": category,
                    "n_points": int(len(y_true)),
                    "event_rate": event_rate,
                    "always_0_accuracy": always0_accuracy,
                    "frequency_accuracy": freq_accuracy,
                    "frequency_balanced_accuracy": freq_balanced_accuracy,
                }
            )

        rows.append(
            {
                "user_id": user_id,
                "category": category,
                "event_rate": event_rate,
                "corr_lag1": lag1,
                "corr_lag7": lag7,
                "p_event_given_prev0": p_1_given_0,
                "p_event_given_prev1": p_1_given_1,
            }
        )

    ev = pd.DataFrame(rows)
    base = pd.DataFrame(baseline_rows)

    valid_corr = ev["corr_lag1"].dropna()
    valid_lag7 = ev["corr_lag7"].dropna()
    summary = {
        "pairs": int(len(ev)),
        "median_event_rate": float(ev["event_rate"].median()),
        "median_corr_lag1": float(valid_corr.median()) if not valid_corr.empty else np.nan,
        "median_corr_lag7": float(valid_lag7.median()) if not valid_lag7.empty else np.nan,
        "median_p_event_given_prev0": float(ev["p_event_given_prev0"].median(skipna=True)),
        "median_p_event_given_prev1": float(ev["p_event_given_prev1"].median(skipna=True)),
    }

    if not base.empty:
        summary.update(
            {
                "always_0_accuracy": float(base["always_0_accuracy"].mean()),
                "frequency_accuracy": float(base["frequency_accuracy"].mean()),
                "frequency_balanced_accuracy": float(base["frequency_balanced_accuracy"].mean()),
            }
        )
    else:
        summary.update(
            {
                "always_0_accuracy": np.nan,
                "frequency_accuracy": np.nan,
                "frequency_balanced_accuracy": np.nan,
            }
        )

    return ev, summary


def decision_logic(sparsity_summary: dict, ac_summary: dict, dist_summary: dict) -> dict:
    sparse_bad = sparsity_summary.get("share_pairs_below_20pct", np.nan)
    lag7_bad = ac_summary.get("share_lag7_below_0_1", np.nan)
    zero_share = dist_summary.get("zero_share", np.nan)

    if (pd.notna(lag7_bad) and lag7_bad > 0.7) or (pd.notna(sparse_bad) and sparse_bad > 0.7):
        status = "CASE 1: no_signal"
        note = "Forecasting is likely infeasible: weak autocorrelation or extreme sparsity."
    elif pd.notna(lag7_bad) and lag7_bad > 0.4:
        status = "CASE 2: weak_signal"
        note = "Weak signal detected: use conservative hybrid approach and strong baselines."
    else:
        status = "CASE 3: good_signal"
        note = "Signal appears usable: ML can be justified after robust baseline checks."

    if pd.notna(zero_share) and zero_share >= 0.95:
        note += " Daily series is mostly zeros; expect many near-zero predictions."

    return {"decision": status, "note": note}


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Dataset Validation Report (Stage 0-10)",
        "",
        "## Stage 0 - Sanity",
        f"- Rows: {report['stage0']['rows']}",
        f"- Users: {report['stage0']['users']}",
        f"- Date range: {report['stage0']['min_date']} .. {report['stage0']['max_date']}",
        f"- Avg tx/user: {report['stage0']['avg_tx_per_user']:.2f}",
        f"- Share users with >=200 tx: {report['stage0']['share_users_with_200plus_tx']:.3f}",
        "",
        "## Stage 3 - Sparsity",
        f"- Median active days %: {report['stage3']['median_active_days_pct']:.2f}",
        f"- Share <20%: {report['stage3']['share_pairs_below_20pct']:.3f}",
        f"- Share 20-50%: {report['stage3']['share_pairs_20_to_50pct']:.3f}",
        f"- Share >=50%: {report['stage3']['share_pairs_50plus_pct']:.3f}",
        "",
        "## Stage 4 - Distribution",
        f"- Zero share: {report['stage4']['zero_share']:.3f}",
        f"- p95: {report['stage4']['p95']:.3f}",
        f"- p99: {report['stage4']['p99']:.3f}",
        f"- Skewness: {report['stage4']['skewness']:.3f}",
        "",
        "## Stage 5 - Autocorrelation",
        f"- Median corr lag1: {report['stage5']['median_corr_lag1']:.3f}",
        f"- Median corr lag7: {report['stage5']['median_corr_lag7']:.3f}",
        f"- Median corr lag30: {report['stage5']['median_corr_lag30']:.3f}",
        f"- Share lag7 < 0.1: {report['stage5']['share_lag7_below_0_1']:.3f}",
        "",
        "## Decision",
        f"- {report['decision']['decision']}",
        f"- {report['decision']['note']}",
    ]
    lines[8:8] = [
        "",
        "## Stage 1 - Category collapse",
        f"- Raw unique categories: {report['stage1']['raw_unique_categories']}",
        f"- Collapsed unique categories: {report['stage1']['collapsed_unique_categories']}",
        f"- Collapse mode: {report['collapse_mode']}",
    ]
    lines.extend(
        [
            "",
            "## Stage 10 - Event-level analysis",
            f"- Pairs analyzed: {report['stage10']['pairs']}",
            f"- Median event rate: {report['stage10']['median_event_rate']:.3f}",
            f"- Median corr lag1: {report['stage10']['median_corr_lag1']:.3f}",
            f"- Median corr lag7: {report['stage10']['median_corr_lag7']:.3f}",
            f"- Median P(event_t=1 | event_t-1=0): {report['stage10']['median_p_event_given_prev0']:.3f}",
            f"- Median P(event_t=1 | event_t-1=1): {report['stage10']['median_p_event_given_prev1']:.3f}",
            f"- Always-0 accuracy: {report['stage10']['always_0_accuracy']:.3f}",
            f"- Frequency accuracy: {report['stage10']['frequency_accuracy']:.3f}",
            f"- Frequency balanced accuracy: {report['stage10']['frequency_balanced_accuracy']:.3f}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    csv_path = resolve_input_csv(args)
    raw = pd.read_csv(csv_path)

    user_col = detect_column(raw, args.user_col, USER_COL_CANDIDATES, "user")
    date_col = detect_column(raw, args.date_col, DATE_COL_CANDIDATES, "date")
    amount_col = detect_column(raw, args.amount_col, AMOUNT_COL_CANDIDATES, "amount")
    category_col = detect_column(raw, args.category_col, CATEGORY_COL_CANDIDATES, "category")

    df = preprocess(raw, user_col, date_col, amount_col, category_col)
    stage0 = stage_0_sanity(df)

    df["category"] = collapse_categories(df["raw_category"], args.collapse_mode, args.top_n, args.n_clusters)
    stage1 = stage_1_category_diagnostics(df["raw_category"], df["category"])
    ts = stage_2_daily_grid(df)
    sparsity_table, stage3 = stage_3_sparsity(ts)
    stage4 = stage_4_distribution(ts, figures_dir)
    ac_table, stage5 = stage_5_autocorrelation(ts)
    baseline_table = stage_6_baselines(ts)
    target_corr_table, stage7 = stage_7_target_validation(ts)
    user_var_table, stage8 = stage_8_user_variance(ts)
    category_signal = stage_9_category_signal(sparsity_table, ac_table)
    event_table, stage10 = stage_10_event_level_analysis(ts)

    baseline_summary = (
        baseline_table.groupby("baseline", as_index=False)[["mae", "rmse"]].mean()
        if not baseline_table.empty
        else pd.DataFrame(columns=["baseline", "mae", "rmse"])
    )
    decision = decision_logic(stage3, stage5, stage4)

    sparsity_table.to_csv(tables_dir / "stage3_sparsity.csv", index=False)
    ac_table.to_csv(tables_dir / "stage5_autocorrelation.csv", index=False)
    baseline_table.to_csv(tables_dir / "stage6_baselines_per_series.csv", index=False)
    baseline_summary.to_csv(tables_dir / "stage6_baselines_summary.csv", index=False)
    target_corr_table.to_csv(tables_dir / "stage7_target_validation.csv", index=False)
    user_var_table.to_csv(tables_dir / "stage8_user_variance.csv", index=False)
    category_signal.to_csv(tables_dir / "stage9_category_signal.csv", index=False)
    event_table.to_csv(tables_dir / "stage10_event_level_analysis.csv", index=False)
    pd.DataFrame(stage1["top10_raw"].items(), columns=["raw_category", "count"]).to_csv(
        tables_dir / "stage1_top10_raw.csv", index=False
    )
    pd.DataFrame(stage1["top10_collapsed"].items(), columns=["category", "count"]).to_csv(
        tables_dir / "stage1_top10_collapsed.csv", index=False
    )

    report = {
        "input_csv": str(csv_path),
        "columns": {
            "user": user_col,
            "date": date_col,
            "amount": amount_col,
            "category": category_col,
        },
        "stage0": stage0,
        "stage1": stage1,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
        "stage7": stage7,
        "stage8": stage8,
        "stage10": stage10,
        "decision": decision,
        "collapse_mode": args.collapse_mode,
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(output_dir / "report.md", report)

    print("=== DONE: full Stage 0-10 analysis completed ===")
    print(f"Input CSV: {csv_path}")
    print(f"Report: {(output_dir / 'report.md')}")
    print(f"JSON: {(output_dir / 'report.json')}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
