#!/usr/bin/env python3
import json
import re
import shutil
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


def copy_file(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def main() -> int:
    root = repo_root()
    manifest_path = root / "models/release_candidate/release_manifest.json"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    bundle_dir = root / "artifacts/ios_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    passport_path = resolve_path(manifest["feature_passport_path"], root)
    if not passport_path.exists():
        print(f"Missing feature passport: {passport_path}", file=sys.stderr)
        return 1

    features = parse_feature_order(passport_path)

    copy_file(passport_path, bundle_dir)
    copy_file(manifest_path, bundle_dir)

    copied_artifacts = manifest.get("copied_artifacts", [])
    golden_json = None
    golden_csv = None
    for item in copied_artifacts:
        name = Path(item).name
        if name.startswith("golden_inference_set") and name.endswith(".json"):
            golden_json = resolve_path(item, root)
        if name.startswith("golden_inference_set") and name.endswith(".csv"):
            golden_csv = resolve_path(item, root)

    if not golden_json or not golden_json.exists():
        print("Missing golden inference set JSON.", file=sys.stderr)
        return 1
    if not golden_csv or not golden_csv.exists():
        print("Missing golden inference set CSV.", file=sys.stderr)
        return 1

    copy_file(golden_json, bundle_dir)
    copy_file(golden_csv, bundle_dir)

    feature_contract = {
        "feature_order": features,
        "feature_types": {name: "Double" for name in features},
        "label_mapping": {
            "0": "bucket_0",
            "1": "bucket_1",
            "2": "bucket_2",
            "3": "bucket_3",
        },
        "guardrails": {
            "warmup_weeks": 8,
            "alpha_after_warmup": 0.2,
            "confidence_threshold": None,
        },
    }

    contract_path = bundle_dir / "feature_contract.json"
    contract_path.write_text(
        json.dumps(feature_contract, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"iOS bundle ready: {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

