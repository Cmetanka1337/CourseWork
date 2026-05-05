#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any

import joblib

try:
    import coremltools as ct
except ImportError:  # pragma: no cover
    ct = None


CLASSES = [0, 1, 2, 3]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_votes(votes: dict) -> tuple[dict[int, float], float]:
    normalized: dict[int, float] = {k: 0.0 for k in CLASSES}
    total = 0.0
    for key, value in votes.items():
        try:
            key_int = int(key)
        except Exception:
            continue
        if key_int in normalized:
            val = float(value)
            normalized[key_int] = val
            total += val
    if total > 0:
        normalized = {k: v / total for k, v in normalized.items()}
    return normalized, total


def detect_probability_output(model) -> str | None:
    spec = model.get_spec()
    for feature in spec.description.output:
        if feature.type.HasField("dictionaryType"):
            return feature.name
    return None


def format_swift_double(value: float) -> str:
    return f"{value:.6f}"


def dict_to_swift_map(data: dict[Any, float]) -> str:
    items = []
    for key, value in data.items():
        items.append(f"{int(key)}: {format_swift_double(float(value))}")
    return "[" + ", ".join(items) + "]"


def feature_dict_to_swift(features: dict[str, float]) -> str:
    lines = []
    for key, value in features.items():
        lines.append(f"\"{key}\": {format_swift_double(value)}")
    return "[\n        " + ",\n        ".join(lines) + "\n    ]"


def main() -> int:
    if ct is None:
        print("coremltools is required to generate iOS sample cases.", file=sys.stderr)
        return 1

    root = repo_root()
    feature_contract = load_json(root / "artifacts/ios_bundle/feature_contract.json")
    feature_order = feature_contract["feature_order"]
    golden = load_json(root / "artifacts/ios_bundle/golden_inference_set_full_spend_tuned.json")
    records = golden.get("records", [])[:3]

    coreml_path = root / "artifacts/coreml/BerkaSpendBucketRF.mlpackage"
    if not coreml_path.exists():
        print("Missing CoreML model: artifacts/coreml/BerkaSpendBucketRF.mlpackage", file=sys.stderr)
        return 1

    model = ct.models.MLModel(str(coreml_path))
    prob_output_name = detect_probability_output(model)
    if prob_output_name is None:
        print("CoreML model has no probability output.", file=sys.stderr)
        return 1

    sklearn_model = joblib.load(root / "models/release_candidate/full_spend_tuned_rf_model.pkl")

    cases = []
    for record in records:
        features = {name: float(record[name]) for name in feature_order}
        coreml_out = model.predict(features)
        votes = coreml_out.get(prob_output_name, {})
        votes = votes if isinstance(votes, dict) else {}
        normalized, sum_votes = normalize_votes(votes)
        probs = [normalized[c] for c in CLASSES]
        pred_class = int(max(range(len(probs)), key=lambda idx: probs[idx]))

        cases.append(
            {
                "features": features,
                "expectedVotes": votes,
                "expectedSumVotes": sum_votes,
                "expectedProbs": probs,
                "expectedSumProbs": sum(probs),
                "expectedPredClass": pred_class,
            }
        )

    output_dir = root / "reports/ios_handoff"
    output_dir.mkdir(parents=True, exist_ok=True)

    swift_lines = [
        "import Foundation",
        "",
        "struct IOSSampleCase {",
        "    let features: [String: Double]",
        "    let expectedVotes: [Int: Double]",
        "    let expectedSumVotes: Double",
        "    let expectedProbs: [Double]",
        "    let expectedSumProbs: Double",
        "    let expectedPredClass: Int",
        "}",
        "",
        "let iosSampleCases: [IOSSampleCase] = [",
    ]

    for case in cases:
        votes_map = normalize_votes(case["expectedVotes"])[0]
        swift_lines.extend(
            [
                "    IOSSampleCase(",
                f"        features: {feature_dict_to_swift(case['features'])},",
                f"        expectedVotes: {dict_to_swift_map(case['expectedVotes'])},",
                f"        expectedSumVotes: {format_swift_double(case['expectedSumVotes'])},",
                f"        expectedProbs: [{', '.join(format_swift_double(p) for p in case['expectedProbs'])}],",
                f"        expectedSumProbs: {format_swift_double(case['expectedSumProbs'])},",
                f"        expectedPredClass: {case['expectedPredClass']}",
                "    ),",
            ]
        )

    swift_lines.append("]")

    (output_dir / "IOSSampleCases.swift").write_text(
        "\n".join(swift_lines) + "\n", encoding="utf-8"
    )

    notes_lines = [
        "# iOS Handoff Notes",
        "",
        "- CoreML classProbability outputs vote counts (sum ≈ n_estimators = 420).",
        "- Normalize votes: p_i = votes_i / sum(votes).",
        "- Map dictionary keys to classes [0,1,2,3] before ordering p0..p3.",
    ]
    (output_dir / "ios_handoff_notes.md").write_text(
        "\n".join(notes_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote iOS handoff samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

