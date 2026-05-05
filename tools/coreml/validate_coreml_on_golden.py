#!/usr/bin/env python3
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import coremltools as ct
except ImportError:  # pragma: no cover
    ct = None


@dataclass
class ValidationResult:
    success: bool
    match_rate: float
    matches: int
    total: int
    prob_sum_min: Optional[float]
    prob_sum_max: Optional[float]
    prob_sum_mean: Optional[float]
    invalid_prob_rows: int
    example: Optional[dict[str, Any]]
    error: Optional[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def detect_output_names(spec) -> tuple[Optional[str], Optional[str]]:
    class_output = None
    prob_output = None
    for output in spec.description.output:
        output_type = output.type
        if output_type.HasField("dictionaryType"):
            prob_output = output.name
        if output_type.HasField("int64Type") or output_type.HasField("stringType"):
            class_output = output.name
    return class_output, prob_output


def extract_class(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def normalize_probabilities(prob_dict: dict) -> dict[str, float]:
    normalized = {}
    for key, value in prob_dict.items():
        try:
            normalized[str(int(key))] = float(value)
        except Exception:
            normalized[str(key)] = float(value)
    return normalized


def validate_coreml(coreml_path: Path) -> ValidationResult:
    if ct is None:
        return ValidationResult(
            success=False,
            match_rate=0.0,
            matches=0,
            total=0,
            prob_sum_min=None,
            prob_sum_max=None,
            prob_sum_mean=None,
            invalid_prob_rows=0,
            example=None,
            error="coremltools is not installed. Install coremltools to validate CoreML models.",
        )

    root = repo_root()
    model = ct.models.MLModel(str(coreml_path))
    spec = model.get_spec()
    class_output, prob_output = detect_output_names(spec)
    if prob_output is None:
        return ValidationResult(
            success=False,
            match_rate=0.0,
            matches=0,
            total=0,
            prob_sum_min=None,
            prob_sum_max=None,
            prob_sum_mean=None,
            invalid_prob_rows=0,
            example=None,
            error="CoreML model does not expose probabilities output.",
        )

    feature_contract = load_json(root / "artifacts/ios_bundle/feature_contract.json")
    golden_path = root / "artifacts/ios_bundle/golden_inference_set_full_spend_tuned.json"
    golden = load_json(golden_path)

    feature_order = feature_contract["feature_order"]
    records = golden.get("records", [])[:10]
    if not records:
        return ValidationResult(
            success=False,
            match_rate=0.0,
            matches=0,
            total=0,
            prob_sum_min=None,
            prob_sum_max=None,
            prob_sum_mean=None,
            invalid_prob_rows=0,
            example=None,
            error="Golden inference set has no records.",
        )

    matches = 0
    prob_sums = []
    invalid_prob_rows = 0
    example = None

    for record in records:
        inputs = {name: float(record[name]) for name in feature_order}
        output = model.predict(inputs)

        prob_dict = None
        if prob_output:
            prob_value = output.get(prob_output)
            if isinstance(prob_value, dict):
                prob_dict = prob_value

        predicted_class = extract_class(output.get(class_output)) if class_output else None
        if predicted_class is None and prob_dict is not None:
            prob_dict_norm = normalize_probabilities(prob_dict)
            predicted_class = int(
                max(prob_dict_norm.items(), key=lambda item: item[1])[0]
            )

        expected_class = int(record["bucket_spend_t_plus_1"])
        if predicted_class is not None and predicted_class == expected_class:
            matches += 1

        if prob_dict is None:
            invalid_prob_rows += 1
        else:
            prob_dict_norm = normalize_probabilities(prob_dict)
            prob_sum = sum(prob_dict_norm.values())
            prob_sums.append(prob_sum)
            if math.isnan(prob_sum) or abs(prob_sum - 1.0) > 0.05:
                invalid_prob_rows += 1

        if example is None:
            example = {
                "expected_class": expected_class,
                "predicted_class": predicted_class,
                "probabilities": normalize_probabilities(prob_dict) if prob_dict else None,
                "test_row_index": record.get("test_row_index"),
                "user_id": record.get("user_id"),
            }

    total = len(records)
    match_rate = matches / total if total else 0.0
    prob_sum_min = min(prob_sums) if prob_sums else None
    prob_sum_max = max(prob_sums) if prob_sums else None
    prob_sum_mean = sum(prob_sums) / len(prob_sums) if prob_sums else None

    if invalid_prob_rows > 0:
        return ValidationResult(
            success=False,
            match_rate=match_rate,
            matches=matches,
            total=total,
            prob_sum_min=prob_sum_min,
            prob_sum_max=prob_sum_max,
            prob_sum_mean=prob_sum_mean,
            invalid_prob_rows=invalid_prob_rows,
            example=example,
            error="Probabilities failed normalization check; sum is not ~1.0.",
        )

    return ValidationResult(
        success=True,
        match_rate=match_rate,
        matches=matches,
        total=total,
        prob_sum_min=prob_sum_min,
        prob_sum_max=prob_sum_max,
        prob_sum_mean=prob_sum_mean,
        invalid_prob_rows=invalid_prob_rows,
        example=example,
        error=None,
    )


def main() -> int:
    root = repo_root()
    coreml_dir = root / "artifacts/coreml"
    if not coreml_dir.exists():
        print("Missing artifacts/coreml directory.", file=sys.stderr)
        return 1

    candidates = list(coreml_dir.glob("*.mlpackage")) + list(coreml_dir.glob("*.mlmodel"))
    if not candidates:
        print("No CoreML model found in artifacts/coreml.", file=sys.stderr)
        return 1

    result = validate_coreml(candidates[0])
    if not result.success:
        print(f"Validation failed: {result.error}", file=sys.stderr)
        return 1

    print(
        f"Golden match rate: {result.match_rate:.3f} "
        f"({result.matches}/{result.total})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

