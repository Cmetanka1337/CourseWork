import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deep Berka dataset suitability analysis")
    parser.add_argument("--weekly-dir", type=str, default="step1_berka_weekly_builder/outputs")
    parser.add_argument("--classification-output-dir", type=str, default="step3_model_training_berka/outputs")
    parser.add_argument("--regression-output-dir", type=str, default="step3_regression_training/outputs")
    parser.add_argument("--output-md", type=str, default="reports/berka_feasibility/berka_dataset_analysis.md")
    parser.add_argument("--output-json", type=str, default="reports/berka_feasibility/berka_dataset_analysis.json")
    return parser.parse_args()


def corr_safe(df: pd.DataFrame, a: str, b: str) -> float:
    part = df[[a, b]].dropna()
    if part.empty:
        return 0.0
    return float(part.corr().iloc[0, 1])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(args: argparse.Namespace) -> dict:
    weekly_dir = Path(args.weekly_dir).resolve()
    cls_dir = Path(args.classification_output_dir).resolve()
    reg_dir = Path(args.regression_output_dir).resolve()

    train_weekly = pd.read_csv(weekly_dir / "classification" / "train_dataset.csv")
    test_weekly = pd.read_csv(weekly_dir / "classification" / "test_dataset.csv")
    all_weekly = pd.concat([train_weekly, test_weekly], ignore_index=True)
    all_weekly = all_weekly.sort_values(["user_id", "week_start"]).reset_index(drop=True)

    by_user = all_weekly.groupby("user_id", sort=False)
    all_weekly["outflow_t_minus_1"] = by_user["weekly_outflow_t"].shift(1)
    all_weekly["net_t_minus_1"] = by_user["weekly_net_t"].shift(1)
    all_weekly["inflow_t_minus_1"] = by_user["weekly_inflow_t"].shift(1)
    all_weekly["outflow_t_plus_1"] = by_user["weekly_outflow_t"].shift(-1)

    weeks_per_user = all_weekly.groupby("user_id")["week_t"].nunique()
    weeks_per_user_values = weeks_per_user.to_numpy(dtype=float)
    short_8 = float((weeks_per_user < 8).mean())
    short_12 = float((weeks_per_user < 12).mean())

    volatility = all_weekly.groupby("user_id").agg(outflow_mean=("weekly_outflow_t", "mean"), outflow_std=("weekly_outflow_t", "std"))
    volatility["outflow_cv"] = volatility["outflow_std"] / volatility["outflow_mean"].replace(0, np.nan)

    trend = (
        all_weekly.groupby("week_t", as_index=False)
        .agg(outflow_sum=("weekly_outflow_t", "sum"), inflow_sum=("weekly_inflow_t", "sum"), net_sum=("weekly_net_t", "sum"))
        .sort_values("week_t")
    )

    reg_train = pd.read_csv(weekly_dir / "regression" / "train_regression.csv")
    reg_test = pd.read_csv(weekly_dir / "regression" / "test_regression.csv")
    reg_all = pd.concat([reg_train, reg_test], ignore_index=True)
    category_support = reg_all["category"].value_counts()

    cls_full_spend = load_json(cls_dir / "full_spend_classification_report.json") if (cls_dir / "full_spend_classification_report.json").exists() else {}
    cls_full_net = load_json(cls_dir / "full_net_classification_report.json") if (cls_dir / "full_net_classification_report.json").exists() else {}
    reg_full = load_json(reg_dir / "full_regression_report.json") if (reg_dir / "full_regression_report.json").exists() else {}

    payload = {
        "population": {
            "accounts": int(all_weekly["user_id"].nunique()),
            "weeks_per_account_quantiles": {
                "p10": float(np.quantile(weeks_per_user_values, 0.10)),
                "p25": float(np.quantile(weeks_per_user_values, 0.25)),
                "p50": float(np.quantile(weeks_per_user_values, 0.50)),
                "p75": float(np.quantile(weeks_per_user_values, 0.75)),
                "p90": float(np.quantile(weeks_per_user_values, 0.90)),
            },
            "short_accounts_share": {
                "lt_8_weeks": short_8,
                "lt_12_weeks": short_12,
            },
        },
        "signal": {
            "corr_outflow_t_vs_t_minus_1": corr_safe(all_weekly, "weekly_outflow_t", "outflow_t_minus_1"),
            "corr_net_t_vs_t_minus_1": corr_safe(all_weekly, "weekly_net_t", "net_t_minus_1"),
            "corr_inflow_t_minus_1_vs_outflow_t_plus_1": corr_safe(all_weekly, "inflow_t_minus_1", "outflow_t_plus_1"),
            "volatility_cv_quantiles": {
                "p25": float(volatility["outflow_cv"].dropna().quantile(0.25)),
                "p50": float(volatility["outflow_cv"].dropna().quantile(0.50)),
                "p75": float(volatility["outflow_cv"].dropna().quantile(0.75)),
            },
        },
        "temporal": {
            "weekly_points": int(len(trend)),
            "time_start": str(trend["week_t"].min()),
            "time_end": str(trend["week_t"].max()),
        },
        "regression_sparsity": {
            "categories_total": int(category_support.shape[0]),
            "top10_categories": [
                {"category": str(cat), "support": int(cnt)} for cat, cnt in category_support.head(10).items()
            ],
            "tail_share_below_50": float((category_support < 50).mean()),
        },
        "model_evidence": {
            "classification_full_spend": cls_full_spend,
            "classification_full_net": cls_full_net,
            "regression_full": reg_full,
        },
    }
    return payload


def to_markdown(payload: dict) -> str:
    pop = payload["population"]
    sig = payload["signal"]
    spar = payload["regression_sparsity"]

    cls_spend = payload["model_evidence"].get("classification_full_spend", {})
    cls_net = payload["model_evidence"].get("classification_full_net", {})
    reg = payload["model_evidence"].get("regression_full", {})

    def cls_line(rep: dict, title: str) -> str:
        if not rep:
            return f"- {title}: n/a"
        b = rep["test"]["baseline_persistence"]["f1_macro"]
        rf = rep["test"]["random_forest"]["f1_macro"]
        lr = rep["test"]["logistic_regression"]["f1_macro"]
        gain = (max(rf, lr) - b) / max(b, 1e-9)
        return f"- {title}: baseline={b:.4f}, RF={rf:.4f}, LR={lr:.4f}, relative_gain={gain:.4f}"

    reg_line = "- Regression full: n/a"
    if reg:
        b = reg["baselines_test"]["persistence_last_value"]["mae"]
        ridge = reg["models_test"]["ridge"]["mae"]
        sgd = reg["models_test"]["sgd_regressor"]["mae"]
        gain = (b - min(ridge, sgd)) / max(b, 1e-9)
        reg_line = f"- Regression full: baseline MAE={b:.2f}, Ridge={ridge:.2f}, SGD={sgd:.2f}, relative_improvement={gain:.4f}"

    return f"""# Berka Dataset Analysis (Signal Suitability)

## 1) Coverage and weekly history
- Accounts: **{pop['accounts']}**
- Weeks/account quantiles p10/p25/p50/p75/p90: **{pop['weeks_per_account_quantiles']['p10']:.1f} / {pop['weeks_per_account_quantiles']['p25']:.1f} / {pop['weeks_per_account_quantiles']['p50']:.1f} / {pop['weeks_per_account_quantiles']['p75']:.1f} / {pop['weeks_per_account_quantiles']['p90']:.1f}**
- Short accounts share (<8w / <12w): **{pop['short_accounts_share']['lt_8_weeks']:.4f} / {pop['short_accounts_share']['lt_12_weeks']:.4f}**

## 2) Signal diagnostics for lag-based models (RF/LR)
- corr(outflow_t, outflow_t-1): **{sig['corr_outflow_t_vs_t_minus_1']:.4f}**
- corr(net_t, net_t-1): **{sig['corr_net_t_vs_t_minus_1']:.4f}**
- corr(inflow_t-1, outflow_t+1): **{sig['corr_inflow_t_minus_1_vs_outflow_t_plus_1']:.4f}**
- Outflow CV quantiles (p25/p50/p75): **{sig['volatility_cv_quantiles']['p25']:.4f} / {sig['volatility_cv_quantiles']['p50']:.4f} / {sig['volatility_cv_quantiles']['p75']:.4f}**

Interpretation for RF suitability:
- Lag/rolling predictors can work when short-memory correlations are non-trivial.
- RF typically captures nonlinear interactions in engineered lag features, but does not model very long sequential dependencies natively.
- If temporal drift is strong, fold-wise metrics may degrade in late folds even with good average scores.

## 3) Regression sparsity and category structure
- Total categories: **{spar['categories_total']}**
- Tail share categories with support < 50: **{spar['tail_share_below_50']:.4f}**
- Top categories by support:
{chr(10).join([f"- `{row['category']}`: {row['support']}" for row in spar['top10_categories']])}

## 4) Full-run model evidence snapshot
{cls_line(cls_spend, 'Classification (spend target)')}
{cls_line(cls_net, 'Classification (net target)')}
{reg_line}

## 5) Methodological references
- Walk-forward / time-series CV rationale: avoid leakage and optimistic estimates in autocorrelated series (MDPI Sensors review context: https://www.mdpi.com/1424-8220/21/7/2430).
- Leakage from improper temporal features and preprocessing fit on full data: https://www.mhtechin.com/support/improper-temporal-feature-extraction-creating-future-leaks-the-core-challenge-in-time-series-machine-learning/
- Why persistence/naive baselines are mandatory in time series benchmarking: https://datascience.stackexchange.com/questions/130838/why-linear-regression-doing-well-in-time-series-data
"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)

    md_path = Path(args.output_md).resolve()
    json_path = Path(args.output_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")
    print("Dataset analysis written:", md_path)


if __name__ == "__main__":
    main()

