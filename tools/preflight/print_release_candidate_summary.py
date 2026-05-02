#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def format_float(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


def parse_feature_order(passport_path: Path) -> list[str]:
    pattern = re.compile(r"^\s*\d+\.\s+`([^`]+)`")
    features: list[str] = []
    for line in passport_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            features.append(match.group(1))
    if not features:
        raise ValueError("Feature order section not found in feature passport.")
    return features


def main() -> int:
    root = repo_root()
    manifest_path = root / "models/release_candidate/release_manifest.json"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    required_paths = [
        ("selected_report", manifest.get("selected_report")),
        ("selected_report_copy", manifest.get("selected_report_copy")),
        ("selected_model_path", manifest.get("selected_model_path")),
        ("feature_passport_path", manifest.get("feature_passport_path")),
    ]
    copied_artifacts = manifest.get("copied_artifacts", [])
    for idx, item in enumerate(copied_artifacts):
        required_paths.append((f"copied_artifacts[{idx}]", item))

    missing = []
    for label, value in required_paths:
        if not value:
            missing.append(f"{label} (empty)")
            continue
        resolved = resolve_path(value, root)
        if not resolved.exists():
            missing.append(f"{label}: {resolved}")

    if missing:
        print("Missing required files:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1

    passport_path = resolve_path(manifest["feature_passport_path"], root)
    features = parse_feature_order(passport_path)

    metrics = manifest.get("metrics", {})
    print("Release candidate summary")
    print(f"Selected prefix: {manifest.get('selected_prefix', 'n/a')}")
    print(f"RF macro-F1: {format_float(metrics.get('rf_f1_macro'))}")
    print(f"RF balanced accuracy: {format_float(metrics.get('rf_balanced_accuracy'))}")
    print(
        "Gain vs persistence: "
        f"{format_float(metrics.get('relative_gain_vs_persistence'))}"
    )
    print(f"Feature count: {manifest.get('feature_count', 'n/a')}")
    print("Paths (repo-relative):")
    print(f"- selected_report: {manifest.get('selected_report')}")
    print(f"- selected_report_copy: {manifest.get('selected_report_copy')}")
    print(f"- selected_model_path: {manifest.get('selected_model_path')}")
    print(f"- feature_passport_path: {manifest.get('feature_passport_path')}")
    print("Required inputs (feature order):")
    print(f"- total_features: {len(features)}")
    for idx, name in enumerate(features, start=1):
        print(f"  {idx:02d}. {name}")

    if manifest.get("feature_count") and manifest["feature_count"] != len(features):
        print(
            "Warning: feature_count does not match feature passport.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

