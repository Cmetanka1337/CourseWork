#!/usr/bin/env python3
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_feature_dict(features: dict[str, float]) -> str:
    lines = []
    for key, value in features.items():
        lines.append(f"\"{key}\": {value}")
    return ",\n            ".join(lines)


def main() -> int:
    root = repo_root()
    golden = load_json(root / "artifacts/ios_bundle/golden_inference_set_full_spend_tuned.json")
    feature_contract = load_json(root / "artifacts/ios_bundle/feature_contract.json")
    feature_order = feature_contract["feature_order"]

    examples = []
    for record in golden.get("records", [])[:3]:
        features = {name: float(record[name]) for name in feature_order}
        examples.append(
            {
                "expected": int(record["bucket_spend_t_plus_1"]),
                "features": features,
            }
        )

    swift_lines = [
        "import CoreML",
        "import Foundation",
        "",
        "struct GoldenExample {",
        "    let expectedClass: Int",
        "    let features: [String: Double]",
        "}",
        "",
        "// Option A (preferred if generated interface exists):",
        "// let model = try BerkaSpendBucketRF(configuration: MLModelConfiguration())",
        "// let output = try model.prediction(bucket_spend_t: ..., weeks_since_outflow: ...)",
        "",
        "func loadModel() throws -> MLModel {",
        "    if let url = Bundle.main.url(forResource: \"BerkaSpendBucketRF\", withExtension: \"mlpackage\") {",
        "        return try MLModel(contentsOf: url)",
        "    }",
        "    let fallback = URL(fileURLWithPath: \"artifacts/coreml/BerkaSpendBucketRF.mlpackage\")",
        "    return try MLModel(contentsOf: fallback)",
        "}",
        "",
        "func extractClassLabel(_ output: MLFeatureProvider) -> Int? {",
        "    for name in output.featureNames {",
        "        guard let value = output.featureValue(for: name) else { continue }",
        "        if value.type == .int64 { return Int(value.int64Value) }",
        "        if value.type == .string, let parsed = Int(value.stringValue) { return parsed }",
        "    }",
        "    return nil",
        "}",
        "",
        "func extractProbabilities(_ output: MLFeatureProvider) -> [String: Double]? {",
        "    for name in output.featureNames {",
        "        guard let value = output.featureValue(for: name) else { continue }",
        "        if value.type == .dictionary {",
        "            let dict = value.dictionaryValue",
        "            var result: [String: Double] = [:]",
        "            for (key, val) in dict {",
        "                result[String(describing: key)] = val.doubleValue",
        "            }",
        "            return result",
        "        }",
        "    }",
        "    return nil",
        "}",
        "",
        "let examples: [GoldenExample] = [",
    ]

    for example in examples:
        features_block = format_feature_dict(example["features"])
        swift_lines.extend(
            [
                "    GoldenExample(",
                f"        expectedClass: {example['expected']},",
                "        features: [",
                f"            {features_block}",
                "        ]",
                "    ),",
            ]
        )

    swift_lines.extend(
        [
            "]",
            "",
            "do {",
            "    let model = try loadModel()",
            "    for (idx, example) in examples.enumerated() {",
            "        let provider = try MLDictionaryFeatureProvider(dictionary: example.features)",
            "        let start = CFAbsoluteTimeGetCurrent()",
            "        let prediction = try model.prediction(from: provider)",
            "        let durationMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0",
            "        let predictedClass = extractClassLabel(prediction)",
            "        let probs = extractProbabilities(prediction)",
            "        print(\"Row \\(idx): expected=\\(example.expectedClass) predicted=\\(predictedClass ?? -1) latency_ms=\\(durationMs)\")",
            "        if let probs = probs {",
            "            print(\"probs=\\(probs)\")",
            "        }",
            "    }",
            "} catch {",
            "    print(\"Smoke test failed: \\(error)\")",
            "}",
        ]
    )

    output_dir = root / "reports/preflight_coreml"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "XcodeSmokeTest.swift"
    output_path.write_text("\n".join(swift_lines) + "\n", encoding="utf-8")
    print(f"Swift smoke test written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

