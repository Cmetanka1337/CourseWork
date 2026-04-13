import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Berka feasibility report")
    parser.add_argument("--weekly-dir", type=str, default="step1_berka_weekly_builder/outputs")
    parser.add_argument("--classification-report", type=str, default="step3_model_training_berka/outputs/classification_report.json")
    parser.add_argument("--regression-report", type=str, default="step3_regression_training/outputs/regression_report.json")
    parser.add_argument("--ingestion-report", type=str, default="data/berka/processed/ingestion_report.json")
    parser.add_argument("--output-dir", type=str, default="reports/berka_feasibility")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def verdict_cls(report: dict) -> tuple[str, float]:
    base = report["test"]["baseline_persistence"]["f1_macro"]
    best = max(
        report["test"]["random_forest"]["f1_macro"],
        report["test"]["logistic_regression"]["f1_macro"],
        report["test"]["sgd_classifier"]["f1_macro"],
    )
    gain = best - base
    if gain >= 0.03:
        return "GO", float(gain)
    if gain >= 0.01:
        return "BORDERLINE", float(gain)
    return "NO_GO", float(gain)


def verdict_reg(report: dict) -> tuple[str, float]:
    base_mae = report["baselines_test"]["persistence_last_value"]["mae"]
    model_mae = min(
        report["models_test"]["ridge"]["mae"],
        report["models_test"]["elasticnet"]["mae"],
        report["models_test"]["sgd_regressor"]["mae"],
    )
    improvement = (base_mae - model_mae) / max(base_mae, 1e-9)
    if improvement >= 0.05:
        return "GO", float(improvement)
    if improvement >= 0.01:
        return "BORDERLINE", float(improvement)
    return "NO_GO", float(improvement)


def build_signal_section(weekly_train_path: Path) -> dict:
    df = pd.read_csv(weekly_train_path)
    df = df.sort_values(["user_id", "week_start"]).copy()
    by_user = df.groupby("user_id", sort=False)
    df["outflow_t_minus_1"] = by_user["weekly_outflow_t"].shift(1)
    df["inflow_t_minus_1"] = by_user["weekly_inflow_t"].shift(1)
    df["outflow_t_plus_1"] = by_user["weekly_outflow_t"].shift(-1)

    c1 = float(df[["weekly_outflow_t", "outflow_t_minus_1"]].corr().iloc[0, 1])
    c2 = float(df[["inflow_t_minus_1", "outflow_t_plus_1"]].corr().iloc[0, 1])

    weeks_per_user = df.groupby("user_id")["week_t"].nunique().to_numpy(dtype=float)
    return {
        "accounts": int(df["user_id"].nunique()),
        "weeks_per_account_quantiles": {
            "p25": float(np.quantile(weeks_per_user, 0.25)),
            "p50": float(np.quantile(weeks_per_user, 0.50)),
            "p75": float(np.quantile(weeks_per_user, 0.75)),
        },
        "corr_outflow_t_vs_t_minus_1": c1,
        "corr_inflow_t_minus_1_vs_outflow_t_plus_1": c2,
    }


def to_markdown(payload: dict) -> str:
    s = payload["signal"]
    cls = payload["classification"]
    reg = payload["regression"]
    return f"""# Berka Feasibility Report

Generated: {payload['execution_timestamp']}

## Signal checks
- Accounts: **{s['accounts']}**
- Weeks/account p25-p50-p75: **{s['weeks_per_account_quantiles']['p25']:.1f} / {s['weeks_per_account_quantiles']['p50']:.1f} / {s['weeks_per_account_quantiles']['p75']:.1f}**
- corr(outflow_t, outflow_t-1): **{s['corr_outflow_t_vs_t_minus_1']:.4f}**
- corr(inflow_t-1, outflow_t+1): **{s['corr_inflow_t_minus_1_vs_outflow_t_plus_1']:.4f}**

## Bucket classification feasibility
- Verdict: **{cls['verdict']}**
- Best model F1_macro gain vs persistence: **{cls['gain_vs_persistence_f1']:.4f}**
- Baseline persistence F1_macro: **{cls['baseline_f1_macro']:.4f}**
- RF/LR/SGD test F1_macro: **{cls['rf_f1']:.4f} / {cls['lr_f1']:.4f} / {cls['sgd_f1']:.4f}**

## Regression feasibility (single model, multi-category)
- Verdict: **{reg['verdict']}**
- Relative MAE improvement vs persistence: **{reg['improvement_vs_persistence']:.4f}**
- Baseline MAE (persistence): **{reg['baseline_mae']:.4f}**
- Ridge/ElasticNet/SGD MAE: **{reg['ridge_mae']:.4f} / {reg['enet_mae']:.4f} / {reg['sgd_mae']:.4f}**

## Methodological checks
- Time-based validation: **TimeSeriesSplit / holdout last weeks**
- Leakage policy: **lags only from past, fit preprocessors on train only**
- Baselines included: **persistence + majority (classification), persistence + rolling mean (regression)**
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    weekly_dir = Path(args.weekly_dir).resolve()
    cls_report = json.loads(Path(args.classification_report).read_text(encoding="utf-8"))
    reg_report = json.loads(Path(args.regression_report).read_text(encoding="utf-8"))
    ingestion_report = json.loads(Path(args.ingestion_report).read_text(encoding="utf-8"))

    signal = build_signal_section(weekly_dir / "classification" / "train_dataset.csv")
    cls_verdict, cls_gain = verdict_cls(cls_report)
    reg_verdict, reg_impr = verdict_reg(reg_report)

    payload = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "ingestion_summary": {
            "rows_normalized": ingestion_report.get("rows_normalized"),
            "unique_users": ingestion_report.get("unique_users"),
            "date_range_start": ingestion_report.get("date_range_start"),
            "date_range_end": ingestion_report.get("date_range_end"),
        },
        "signal": signal,
        "classification": {
            "verdict": cls_verdict,
            "gain_vs_persistence_f1": cls_gain,
            "baseline_f1_macro": cls_report["test"]["baseline_persistence"]["f1_macro"],
            "rf_f1": cls_report["test"]["random_forest"]["f1_macro"],
            "lr_f1": cls_report["test"]["logistic_regression"]["f1_macro"],
            "sgd_f1": cls_report["test"]["sgd_classifier"]["f1_macro"],
        },
        "regression": {
            "verdict": reg_verdict,
            "improvement_vs_persistence": reg_impr,
            "baseline_mae": reg_report["baselines_test"]["persistence_last_value"]["mae"],
            "ridge_mae": reg_report["models_test"]["ridge"]["mae"],
            "enet_mae": reg_report["models_test"]["elasticnet"]["mae"],
            "sgd_mae": reg_report["models_test"]["sgd_regressor"]["mae"],
            "top_category_metrics_ridge": reg_report.get("top_category_metrics_ridge", []),
        },
    }

    (out_dir / "berka_feasibility_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "berka_feasibility_report.md").write_text(to_markdown(payload), encoding="utf-8")

    print("=== BERKA FEASIBILITY REPORT DONE ===")
    print(f"Classification verdict: {cls_verdict}")
    print(f"Regression verdict: {reg_verdict}")


if __name__ == "__main__":
    main()

