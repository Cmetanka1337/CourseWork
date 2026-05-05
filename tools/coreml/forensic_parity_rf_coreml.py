#!/usr/bin/env python3
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

try:
    import coremltools as ct
except ImportError:  # pragma: no cover
    ct = None


CLASSES = [0, 1, 2, 3]


@dataclass
class SpecSummary:
    input_names: list[str]
    input_types: dict[str, str]
    output_names: list[str]
    class_output_name: Optional[str]
    probability_output_name: Optional[str]
    class_labels: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def type_to_str(value) -> str:
    if value.HasField("doubleType"):
        return "double"
    if value.HasField("int64Type"):
        return "int64"
    if value.HasField("stringType"):
        return "string"
    if value.HasField("multiArrayType"):
        return "multiArray"
    if value.HasField("dictionaryType"):
        return "dictionary"
    return "unknown"


def summarize_spec(model) -> SpecSummary:
    spec = model.get_spec()
    input_names = []
    input_types = {}
    for feature in spec.description.input:
        input_names.append(feature.name)
        input_types[feature.name] = type_to_str(feature.type)

    output_names = [feature.name for feature in spec.description.output]
    class_output = None
    prob_output = None
    for feature in spec.description.output:
        if feature.type.HasField("dictionaryType"):
            prob_output = feature.name
        if feature.type.HasField("int64Type") or feature.type.HasField("stringType"):
            class_output = feature.name

    class_labels = []
    class_labels_field = getattr(spec.description, "classLabels", None)
    if class_labels_field is not None:
        kind = class_labels_field.WhichOneof("Type")
        if kind == "stringClassLabels":
            class_labels = list(class_labels_field.stringClassLabels)
        elif kind == "int64ClassLabels":
            class_labels = [str(value) for value in class_labels_field.int64ClassLabels]

    return SpecSummary(
        input_names=input_names,
        input_types=input_types,
        output_names=output_names,
        class_output_name=class_output,
        probability_output_name=prob_output,
        class_labels=class_labels,
    )


def normalize_probabilities(prob_dict: dict) -> dict[int, float]:
    normalized: dict[int, float] = {k: 0.0 for k in CLASSES}
    for key, value in prob_dict.items():
        try:
            key_int = int(key)
        except Exception:
            continue
        if key_int in normalized:
            normalized[key_int] = float(value)
    return normalized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_order(vector: np.ndarray) -> list[int]:
    return list(np.argsort(vector)[::-1])


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if ct is None:
        print("coremltools is required for this script.", file=sys.stderr)
        return 1

    root = repo_root()
    output_dir = root / "reports/forensics_coreml_rf"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_contract = load_json(root / "artifacts/ios_bundle/feature_contract.json")
    feature_order = feature_contract["feature_order"]
    golden = load_json(root / "artifacts/ios_bundle/golden_inference_set_full_spend_tuned.json")
    records = golden.get("records", [])[:10]

    sklearn_model = joblib.load(root / "models/release_candidate/full_spend_tuned_rf_model.pkl")

    coreml_path = root / "artifacts/coreml/BerkaSpendBucketRF.mlpackage"
    model = ct.models.MLModel(str(coreml_path))
    spec_summary = summarize_spec(model)

    required_set = set(feature_order)
    input_set = set(spec_summary.input_names)
    missing_inputs = sorted(required_set - input_set)
    extra_inputs = sorted(input_set - required_set)

    report = {
        "decision": None,
        "root_cause": None,
        "notes": [],
        "spec": asdict(spec_summary),
        "feature_check": {
            "required_count": len(feature_order),
            "input_count": len(spec_summary.input_names),
            "missing_inputs": missing_inputs,
            "extra_inputs": extra_inputs,
        },
        "sklearn_match_rate": None,
        "coreml_match_rate": None,
        "coreml_argmax_norm_match_rate": None,
        "cosine_similarity_mean": None,
        "rank_order_match_count": None,
        "prob_sum_min": None,
        "prob_sum_max": None,
        "prob_sum_mean": None,
        "decision_rule": None,
    }

    if missing_inputs or extra_inputs or len(spec_summary.input_names) != len(feature_order):
        report["root_cause"] = "feature_mismatch"
        report["decision"] = "ABANDON_RF"
        report["decision_rule"] = "Input feature mismatch detected; export mapping likely broken."
        write_markdown(
            output_dir / "forensic_report.md",
            [
                "# CoreML RF Forensic Report",
                "",
                "Decision: **ABANDON RF**",
                "",
                "## Input mismatch",
                f"- required_count: {len(feature_order)}",
                f"- input_count: {len(spec_summary.input_names)}",
                f"- missing_inputs: {missing_inputs}",
                f"- extra_inputs: {extra_inputs}",
            ],
        )
        (output_dir / "forensic_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return 1

    X = np.array([[float(record[name]) for name in feature_order] for record in records])
    sklearn_pred = sklearn_model.predict(X)
    sklearn_proba = sklearn_model.predict_proba(X)
    expected = np.array([int(record["bucket_spend_t_plus_1"]) for record in records])
    sklearn_match_rate = float(np.mean(sklearn_pred == expected))
    if sklearn_match_rate < 0.7:
        report["notes"].append("sklearn_vs_golden_low_match")

    rows = []
    prob_sums = []
    cosine_scores = []
    rank_matches = 0
    argmax_norm_matches = 0
    coreml_match_rate = 0

    first_mismatch = None

    for idx, record in enumerate(records):
        inputs = {name: float(record[name]) for name in feature_order}
        output = model.predict(inputs)

        prob_output = spec_summary.probability_output_name
        class_output = spec_summary.class_output_name

        raw_probs = output.get(prob_output) if prob_output else None
        raw_probs = raw_probs if isinstance(raw_probs, dict) else {}
        coreml_raw = normalize_probabilities(raw_probs)
        sum_raw = sum(coreml_raw.values())
        prob_sums.append(sum_raw)

        if sum_raw > 0.0:
            coreml_norm = {k: v / sum_raw for k, v in coreml_raw.items()}
        else:
            coreml_norm = {k: 0.0 for k in CLASSES}

        argmax_raw = max(coreml_raw, key=coreml_raw.get)
        argmax_norm = max(coreml_norm, key=coreml_norm.get)

        class_label = output.get(class_output) if class_output else None
        try:
            class_label_int = int(class_label)
        except Exception:
            class_label_int = None

        sklearn_vec = sklearn_proba[idx]
        coreml_vec = np.array([coreml_norm[c] for c in CLASSES])
        cosine = cosine_similarity(sklearn_vec, coreml_vec)
        cosine_scores.append(cosine)

        rank_match = rank_order(sklearn_vec) == rank_order(coreml_vec)
        if rank_match:
            rank_matches += 1
        if argmax_norm == int(sklearn_pred[idx]):
            argmax_norm_matches += 1
        if class_label_int == int(expected[idx]):
            coreml_match_rate += 1

        sum_norm = sum(coreml_norm.values())

        notes = []
        if sum_raw > 2.0:
            notes.append("raw_probs_large_sum")
        if math.isclose(sum_raw, getattr(sklearn_model, "n_estimators", 0), rel_tol=1e-3):
            notes.append("sum_raw≈n_estimators")

        row = {
            "row_id": record.get("test_row_index", idx),
            "expected_class": int(expected[idx]),
            "sklearn_pred": int(sklearn_pred[idx]),
            "coreml_classLabel": class_label_int,
            "argmax_coreml_raw": argmax_raw,
            "argmax_coreml_norm": argmax_norm,
            "sum_coreml_raw": sum_raw,
            "sum_coreml_norm": sum_norm,
            "cosine_similarity": cosine,
            "rank_order_match": rank_match,
            "notes": ";".join(notes),
        }
        rows.append(row)

        mismatch = (
            class_label_int != int(sklearn_pred[idx])
            or argmax_norm != int(sklearn_pred[idx])
            or cosine < 0.95
        )
        if mismatch and first_mismatch is None:
            first_mismatch = {
                "features": inputs,
                "expected_class": int(expected[idx]),
                "sklearn_pred": int(sklearn_pred[idx]),
                "sklearn_proba": sklearn_vec.tolist(),
                "coreml_classLabel": class_label_int,
                "coreml_raw": coreml_raw,
                "coreml_norm": coreml_norm,
                "sum_raw": sum_raw,
                "argmax_norm": argmax_norm,
                "cosine_similarity": cosine,
            }

    total = len(records)
    coreml_match_rate = coreml_match_rate / total if total else 0.0
    argmax_norm_rate = argmax_norm_matches / total if total else 0.0
    cosine_mean = float(np.mean(cosine_scores)) if cosine_scores else 0.0

    report.update(
        {
            "sklearn_match_rate": sklearn_match_rate,
            "coreml_match_rate": coreml_match_rate,
            "coreml_argmax_norm_match_rate": argmax_norm_rate,
            "cosine_similarity_mean": cosine_mean,
            "rank_order_match_count": rank_matches,
            "prob_sum_min": min(prob_sums) if prob_sums else None,
            "prob_sum_max": max(prob_sums) if prob_sums else None,
            "prob_sum_mean": float(np.mean(prob_sums)) if prob_sums else None,
        }
    )

    input_ok = not missing_inputs and not extra_inputs and len(spec_summary.input_names) == len(feature_order)
    sum_ok = (
        math.isclose(report["prob_sum_mean"], 1.0, rel_tol=1e-2)
        or math.isclose(report["prob_sum_mean"], getattr(sklearn_model, "n_estimators", 0), rel_tol=1e-2)
    )
    keep = (
        input_ok
        and sum_ok
        and argmax_norm_rate >= 0.9
        and (cosine_mean >= 0.95 or rank_matches >= 9)
    )

    if keep:
        report["decision"] = "KEEP_RF"
        report["root_cause"] = "normalized_votes"
        report["decision_rule"] = "CoreML probabilities appear to be vote counts but normalize cleanly; parity is high."
    else:
        report["decision"] = "ABANDON_RF"
        if not sum_ok:
            report["root_cause"] = "raw_votes_not_normalized"
        elif argmax_norm_rate <= 0.6:
            report["root_cause"] = "low_parity"
        else:
            report["root_cause"] = "borderline_parity"
        report["decision_rule"] = "Parity thresholds not met."

    csv_path = output_dir / "forensic_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if first_mismatch:
        write_markdown(
            output_dir / "debug_first_mismatch.md",
            [
                "# CoreML RF First Mismatch",
                "",
                f"- expected_class: {first_mismatch['expected_class']}",
                f"- sklearn_pred: {first_mismatch['sklearn_pred']}",
                f"- coreml_classLabel: {first_mismatch['coreml_classLabel']}",
                f"- sum_raw: {first_mismatch['sum_raw']}",
                f"- argmax_norm: {first_mismatch['argmax_norm']}",
                f"- cosine_similarity: {first_mismatch['cosine_similarity']}",
                "",
                "## Features",
                json.dumps(first_mismatch["features"], indent=2, ensure_ascii=True),
                "",
                "## Sklearn probabilities",
                json.dumps(first_mismatch["sklearn_proba"], indent=2, ensure_ascii=True),
                "",
                "## CoreML raw probabilities",
                json.dumps(first_mismatch["coreml_raw"], indent=2, ensure_ascii=True),
                "",
                "## CoreML normalized probabilities",
                json.dumps(first_mismatch["coreml_norm"], indent=2, ensure_ascii=True),
            ],
        )
    else:
        write_markdown(
            output_dir / "debug_first_mismatch.md",
            [
                "# CoreML RF First Mismatch",
                "",
                "No mismatches found between CoreML and sklearn on the 10 golden rows.",
            ],
        )

    report_lines = [
        "# CoreML RF Forensic Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Inputs",
        f"- required_features: {len(feature_order)}",
        f"- coreml_inputs: {len(spec_summary.input_names)}",
        f"- missing_inputs: {missing_inputs}",
        f"- extra_inputs: {extra_inputs}",
        "",
        "## Output summary",
        f"- class_output_name: {spec_summary.class_output_name}",
        f"- probability_output_name: {spec_summary.probability_output_name}",
        f"- prob_sum_mean: {report['prob_sum_mean']}",
        f"- prob_sum_min: {report['prob_sum_min']}",
        f"- prob_sum_max: {report['prob_sum_max']}",
        "",
        "## Parity metrics",
        f"- sklearn_match_rate: {sklearn_match_rate}",
        f"- coreml_match_rate: {coreml_match_rate}",
        f"- argmax_norm_match_rate: {argmax_norm_rate}",
        f"- cosine_similarity_mean: {cosine_mean}",
        f"- rank_order_match_count: {rank_matches}/10",
        f"- notes: {', '.join(report['notes']) if report['notes'] else 'none'}",
        "",
        "## Decision rule",
        f"- root_cause: {report['root_cause']}",
        f"- rationale: {report['decision_rule']}",
    ]
    write_markdown(output_dir / "forensic_report.md", report_lines)
    (output_dir / "forensic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return 0 if report["decision"] == "KEEP_RF" else 1


if __name__ == "__main__":
    raise SystemExit(main())

