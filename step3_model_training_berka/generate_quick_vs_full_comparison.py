import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = ["spend", "net"]
MODELS = ["random_forest", "logistic_regression", "sgd_classifier"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quick-vs-full comparison for Berka classification")
    parser.add_argument("--input-dir", type=str, default="step3_model_training_berka/outputs")
    parser.add_argument("--output-md", type=str, default="step3_model_training_berka/outputs/quick_vs_full_comparison.md")
    parser.add_argument("--output-csv", type=str, default="step3_model_training_berka/outputs/quick_vs_full_comparison.csv")
    return parser.parse_args()


def report_path(root: Path, mode: str, target: str) -> Path:
    return root / f"{mode}_{target}_classification_report.json"


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).resolve()

    rows = []
    for target in TARGETS:
        for mode in ["quick", "full"]:
            path = report_path(root, mode, target)
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            baseline = float(report["test"]["baseline_persistence"]["f1_macro"])
            for model_name in MODELS:
                model_f1 = float(report["test"][model_name]["f1_macro"])
                gain_rel = (model_f1 - baseline) / max(baseline, 1e-9)
                rows.append(
                    {
                        "target": target,
                        "mode": mode,
                        "model": model_name,
                        "f1_macro": model_f1,
                        "balanced_accuracy": float(report["test"][model_name]["balanced_accuracy"]),
                        "baseline_f1_macro": baseline,
                        "relative_gain_vs_persistence": gain_rel,
                    }
                )

    if not rows:
        raise RuntimeError("No classification reports found for quick/full comparison")

    df = pd.DataFrame(rows)
    df = df.sort_values(["target", "model", "mode"]).reset_index(drop=True)
    df.to_csv(Path(args.output_csv).resolve(), index=False)

    pivot = df.pivot_table(index=["target", "model"], columns="mode", values=["f1_macro", "balanced_accuracy", "relative_gain_vs_persistence"])
    pivot.columns = [f"{metric}_{mode}" for metric, mode in pivot.columns]
    pivot = pivot.reset_index()
    if {"f1_macro_quick", "f1_macro_full"}.issubset(set(pivot.columns)):
        pivot["f1_delta_full_minus_quick"] = pivot["f1_macro_full"] - pivot["f1_macro_quick"]

    lines = ["# Quick vs Full Comparison (Classification)", "", "## Main table"]
    cols = list(pivot.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in pivot.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Acceptance focus")
    lines.append("- Relative gain formula: `(F1_model - F1_persistence) / F1_persistence`.")
    lines.append("- Recommended threshold: at least `+0.50` (50%) and stretch target `+0.70` (70%).")

    Path(args.output_md).resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Comparison written:", args.output_md)


if __name__ == "__main__":
    main()

