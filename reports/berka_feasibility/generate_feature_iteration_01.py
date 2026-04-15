import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ["random_forest", "logistic_regression", "sgd_classifier"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Berka feature iteration 01 comparison report")
    parser.add_argument("--output-dir", type=str, default="step3_model_training_berka/outputs")
    parser.add_argument(
        "--before-dir",
        type=str,
        default="step3_model_training_berka/outputs/feature_iteration_01_before",
    )
    parser.add_argument("--report-path", type=str, default="reports/berka_feasibility/feature_iteration_01.md")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_gain_vs_persistence(report: dict, model_name: str = "random_forest") -> float:
    baseline = float(report["test"]["baseline_persistence"]["f1_macro"])
    model = float(report["test"][model_name]["f1_macro"])
    return (model - baseline) / max(baseline, 1e-9)


def get_fold_drop(fold_csv: Path, model_name: str = "random_forest") -> tuple[float, float, float]:
    fold_df = pd.read_csv(fold_csv)
    model_df = fold_df[fold_df["model"] == model_name].sort_values("fold")
    head = float(model_df["f1_macro"].head(2).mean())
    tail = float(model_df["f1_macro"].tail(2).mean())
    drop = float((head - tail) / max(abs(head), 1e-9))
    return head, tail, drop


def build_comparison_rows(before_root: Path, after_root: Path) -> pd.DataFrame:
    rows = []
    targets = ["spend", "net"]
    modes = ["quick", "full"]
    for target in targets:
        for mode in modes:
            before_rep = load_json(before_root / f"{mode}_{target}_classification_report.json")
            after_rep = load_json(after_root / f"{mode}_{target}_classification_report.json")
            for model in MODELS:
                rows.append(
                    {
                        "target": target,
                        "mode": mode,
                        "model": model,
                        "f1_before": float(before_rep["test"][model]["f1_macro"]),
                        "f1_after": float(after_rep["test"][model]["f1_macro"]),
                        "bal_before": float(before_rep["test"][model]["balanced_accuracy"]),
                        "bal_after": float(after_rep["test"][model]["balanced_accuracy"]),
                    }
                )
    return pd.DataFrame(rows)


def tuning_section(after_root: Path, untuned_quick_spend: dict) -> list[str]:
    tuned_path = after_root / "quick_spend_tuned_classification_report.json"
    if not tuned_path.exists():
        return ["- Tuning run not found. Execute with `--tune-rf` to populate this section."]

    tuned = load_json(tuned_path)
    rf_tuning = tuned.get("rf_tuning", {})
    return [
        f"- Quick tuning used: `{rf_tuning.get('strategy', 'n/a')}` with `n_iter={rf_tuning.get('n_iter', 'n/a')}`.",
        f"- Best params: `{rf_tuning.get('best_params', {})}`.",
        (
            "- Best CV F1_macro: "
            f"**{float(rf_tuning.get('best_cv_mean_f1_macro', 0.0)):.4f} +- "
            f"{float(rf_tuning.get('best_cv_std_f1_macro', 0.0)):.4f}**."
        ),
        (
            "- Quick tuned RF test F1_macro: "
            f"**{float(tuned['test']['random_forest']['f1_macro']):.4f}** vs untuned "
            f"**{float(untuned_quick_spend['test']['random_forest']['f1_macro']):.4f}**."
        ),
    ]


def main() -> None:
    args = parse_args()
    after_root = Path(args.output_dir).resolve()
    before_root = Path(args.before_dir).resolve()
    report_path = Path(args.report_path).resolve()

    before_full_spend = load_json(before_root / "full_spend_classification_report.json")
    after_full_spend = load_json(after_root / "full_spend_classification_report.json")
    after_quick_spend = load_json(after_root / "quick_spend_classification_report.json")

    comparison = build_comparison_rows(before_root=before_root, after_root=after_root)

    head_before, tail_before, drop_before = get_fold_drop(before_root / "full_spend_fold_metrics.csv")
    head_after, tail_after, drop_after = get_fold_drop(after_root / "full_spend_fold_metrics.csv")

    rf_f1_delta = (
        float(after_full_spend["test"]["random_forest"]["f1_macro"])
        - float(before_full_spend["test"]["random_forest"]["f1_macro"])
    )
    rf_bal_delta = (
        float(after_full_spend["test"]["random_forest"]["balanced_accuracy"])
        - float(before_full_spend["test"]["random_forest"]["balanced_accuracy"])
    )

    keep_decision = (
        "- **Keep**: required uplift is met (>= +0.02 on full spend F1_macro or balanced_accuracy) "
        "and gain vs persistence remains strong."
        if (rf_f1_delta >= 0.02 or rf_bal_delta >= 0.02)
        else "- **Do not keep as-is**: uplift threshold is not met; run ablations/tuning before adoption."
    )

    lines = [
        "# Feature Iteration 01: Calendar + Inflow/Outflow + Regularity",
        "",
        "## What was added",
        "- Calendar: `week_of_year`, `month`, `quarter`, `week_of_month`, `is_month_start_week`, `is_month_end_week`.",
        "- Flow dynamics: `delta_inflow`, `delta_outflow`, `inflow_outflow_ratio`, `inflow_share`, explicit lag columns.",
        "- Regularity (anti-leakage): 8-week rolling mean/std + frequency with `shift(1)`, plus `weeks_since_inflow/outflow`.",
        "- Fill policy: lag/rolling fields -> `0.0`; final feature files validated to be NaN-free.",
        "",
        "## Metrics before vs after",
        "| target | mode | model | F1 before | F1 after | delta | bal_acc before | bal_acc after | delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in comparison.sort_values(["target", "mode", "model"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['mode']} | {row['model']} | "
            f"{row['f1_before']:.4f} | {row['f1_after']:.4f} | {row['f1_after'] - row['f1_before']:.4f} | "
            f"{row['bal_before']:.4f} | {row['bal_after']:.4f} | {row['bal_after'] - row['bal_before']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Spend target acceptance check (full run, RF)",
            (
                "- Relative gain vs persistence (before/after): "
                f"**{rel_gain_vs_persistence(before_full_spend):.4f} / "
                f"{rel_gain_vs_persistence(after_full_spend):.4f}**"
            ),
            f"- Full RF F1_macro delta: **{rf_f1_delta:.4f}**",
            f"- Full RF balanced_accuracy delta: **{rf_bal_delta:.4f}**",
            (
                "- Majority baseline F1 (after): "
                f"**{float(after_full_spend['test']['baseline_majority']['f1_macro']):.4f}**"
            ),
            "",
            "## Per-class changes (RF, spend target, full)",
            "| class | precision before | precision after | recall before | recall after | f1 before | f1 after | support after |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for cls in ["0", "1", "2", "3"]:
        before_cls = before_full_spend["test"]["random_forest"]["per_class"][cls]
        after_cls = after_full_spend["test"]["random_forest"]["per_class"][cls]
        lines.append(
            f"| {cls} | {before_cls['precision']:.4f} | {after_cls['precision']:.4f} | "
            f"{before_cls['recall']:.4f} | {after_cls['recall']:.4f} | "
            f"{before_cls['f1']:.4f} | {after_cls['f1']:.4f} | {after_cls['support']} |"
        )

    lines.extend(
        [
            "",
            "## Fold stability (RF, spend target)",
            (
                "- Head/tail F1 mean before: "
                f"**{head_before:.4f} / {tail_before:.4f}** (relative drop **{drop_before:.4f}**)."
            ),
            (
                "- Head/tail F1 mean after: "
                f"**{head_after:.4f} / {tail_after:.4f}** (relative drop **{drop_after:.4f}**)."
            ),
            "- Fold artifacts: `step3_model_training_berka/outputs/full_spend_fold_metrics.csv`, `step3_model_training_berka/outputs/full_spend_fold_per_class_metrics.csv`, `step3_model_training_berka/outputs/full_spend_fold_confusion_matrices.json`.",
            "",
            "## Optional RF tuning (`--tune-rf`)",
            *tuning_section(after_root=after_root, untuned_quick_spend=after_quick_spend),
            "",
            "## Decision",
            keep_decision,
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Feature iteration report written:", report_path)


if __name__ == "__main__":
    main()

