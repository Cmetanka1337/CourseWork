import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Berka RF release candidate for spend target")
    parser.add_argument("--outputs-dir", type=str, default="step3_model_training_berka/outputs")
    parser.add_argument("--release-dir", type=str, default="models/release_candidate")
    parser.add_argument(
        "--feature-passport",
        type=str,
        default="docs/berka_feature_passport_spend_bucket.md",
        help="Path to feature passport markdown",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stability_value(report: dict) -> float:
    try:
        return float(report["cv"]["stability"]["random_forest"].get("relative_drop", 0.0))
    except Exception:
        return float("inf")


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    release_dir = Path(args.release_dir).resolve()
    passport_path = Path(args.feature_passport).resolve()

    reports = sorted(outputs_dir.glob("*classification_report*.json"))
    candidates = []

    for report_path in reports:
        try:
            report = load_json(report_path)
        except Exception:
            continue

        if report.get("target") != "bucket_spend_t_plus_1":
            continue

        test = report.get("test", {})
        rf = test.get("random_forest", {})
        if "f1_macro" not in rf or "balanced_accuracy" not in rf:
            continue

        candidates.append(
            {
                "path": report_path,
                "report": report,
                "rf_f1_macro": float(rf["f1_macro"]),
                "rf_balanced_accuracy": float(rf["balanced_accuracy"]),
                "stability_relative_drop": stability_value(report),
            }
        )

    if not candidates:
        raise RuntimeError("No spend-target classification reports found in outputs directory")

    spend_specific = [c for c in candidates if "spend" in c["path"].name.lower()]
    if spend_specific:
        candidates = spend_specific

    # Sort by: highest f1, highest balanced_accuracy, then lowest stability drop
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["rf_f1_macro"],
            x["rf_balanced_accuracy"],
            -x["stability_relative_drop"],
        ),
        reverse=True,
    )
    best = candidates[0]

    release_dir.mkdir(parents=True, exist_ok=True)
    chosen_report_src = best["path"]
    chosen_report_dst = release_dir / chosen_report_src.name
    shutil.copy2(chosen_report_src, chosen_report_dst)

    copied_artifacts = [str(chosen_report_dst)]

    # Link/copy model file if naming convention matches <prefix>_classification_report.json
    report_stem = chosen_report_src.stem
    prefix = report_stem.removesuffix("_classification_report")
    model_src = outputs_dir / f"{prefix}_rf_model.pkl"
    if model_src.exists():
        model_dst = release_dir / model_src.name
        shutil.copy2(model_src, model_dst)
        copied_artifacts.append(str(model_dst))

    # Copy golden inference set if matching prefix is present.
    golden_json = outputs_dir / f"golden_inference_set_{prefix}.json"
    golden_csv = outputs_dir / f"golden_inference_set_{prefix}.csv"
    if golden_json.exists():
        dst = release_dir / golden_json.name
        shutil.copy2(golden_json, dst)
        copied_artifacts.append(str(dst))
    if golden_csv.exists():
        dst = release_dir / golden_csv.name
        shutil.copy2(golden_csv, dst)
        copied_artifacts.append(str(dst))

    if passport_path.exists():
        passport_dst = release_dir / passport_path.name
        shutil.copy2(passport_path, passport_dst)
        copied_artifacts.append(str(passport_dst))

    report = best["report"]
    manifest = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "primary": "max test.random_forest.f1_macro",
            "tie_breaker_1": "max test.random_forest.balanced_accuracy",
            "tie_breaker_2": "min cv.stability.random_forest.relative_drop",
        },
        "selected_report": str(chosen_report_src),
        "selected_report_copy": str(chosen_report_dst),
        "selected_prefix": prefix,
        "selected_model_path": str(model_src) if model_src.exists() else None,
        "target": report.get("target"),
        "run_mode": report.get("run_mode"),
        "metrics": {
            "rf_f1_macro": best["rf_f1_macro"],
            "rf_balanced_accuracy": best["rf_balanced_accuracy"],
            "stability_relative_drop": best["stability_relative_drop"],
            "relative_gain_vs_persistence": report.get("acceptance", {}).get("relative_gain_vs_persistence"),
        },
        "feature_count": report.get("cv", {}).get("feature_count"),
        "feature_passport_path": str(passport_path),
        "copied_artifacts": copied_artifacts,
        "all_ranked_candidates": [
            {
                "path": str(c["path"]),
                "rf_f1_macro": c["rf_f1_macro"],
                "rf_balanced_accuracy": c["rf_balanced_accuracy"],
                "stability_relative_drop": c["stability_relative_drop"],
            }
            for c in candidates
        ],
    }

    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Release candidate selected:", chosen_report_src.name)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

