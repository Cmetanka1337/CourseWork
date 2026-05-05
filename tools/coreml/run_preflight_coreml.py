#!/usr/bin/env python3
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coreml.export_release_candidate_to_coreml import export_release_candidate
from tools.coreml.generate_xcode_smoke_test import main as generate_swift
from tools.coreml.validate_coreml_on_golden import validate_coreml


def repo_root() -> Path:
    return ROOT


def rel_path(path: str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return path


def write_report(report: dict) -> None:
    root = repo_root()
    output_dir = root / "reports/preflight_coreml"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "coreml_export_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# CoreML Preflight Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Export",
        f"- requested_model: {report['export']['requested_model']}",
        f"- exported_model: {report['export'].get('exported_model')}",
        f"- fallback_used: {report['export'].get('fallback_used')}",
        f"- fallback_error: {report['export'].get('fallback_error')}",
        f"- coreml_path: {report['export'].get('coreml_path')}",
        f"- coreml_format: {report['export'].get('coreml_format')}",
        f"- probabilities_available: {report['export'].get('probabilities_available')}",
        f"- class_output_name: {report['export'].get('class_output_name')}",
        f"- probability_output_name: {report['export'].get('probability_output_name')}",
    ]

    if report["export"].get("error"):
        md_lines.extend(["", "Export error:", f"- {report['export']['error']}"])

    if report.get("rf_validation_error"):
        md_lines.extend(["", "RF validation error:", f"- {report['rf_validation_error']}"])

    md_lines.extend(["", "## Golden validation"])
    validation = report.get("validation")
    if validation:
        md_lines.extend(
            [
                f"- match_rate: {validation.get('match_rate')}",
                f"- matches: {validation.get('matches')}/{validation.get('total')}",
                f"- prob_sum_min: {validation.get('prob_sum_min')}",
                f"- prob_sum_max: {validation.get('prob_sum_max')}",
                f"- prob_sum_mean: {validation.get('prob_sum_mean')}",
                f"- invalid_prob_rows: {validation.get('invalid_prob_rows')}",
            ]
        )
        example = validation.get("example")
        if example:
            md_lines.extend(
                [
                    "",
                    "Example output:",
                    f"- expected_class: {example.get('expected_class')}",
                    f"- predicted_class: {example.get('predicted_class')}",
                    f"- probabilities: {example.get('probabilities')}",
                ]
            )
    else:
        md_lines.append("- validation not run")

    md_lines.extend(
        [
            "",
            "## Xcode next step",
            "- Drag the .mlpackage into Xcode and run reports/preflight_coreml/XcodeSmokeTest.swift",
        ]
    )

    (output_dir / "coreml_export_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    root = repo_root()
    report = {
        "status": "failure",
        "export": {
            "requested_model": "rf",
            "fallback_used": False,
            "fallback_error": None,
        },
        "validation": None,
        "rf_validation_error": None,
    }

    export_result = export_release_candidate("rf")
    if not export_result.success:
        report["export"].update(
            {
                "exported_model": None,
                "coreml_path": None,
                "coreml_format": None,
                "probabilities_available": False,
                "class_output_name": export_result.class_output_name,
                "probability_output_name": export_result.probability_output_name,
                "error": export_result.error,
            }
        )
        fallback = export_release_candidate("lr")
        if fallback.success:
            report["export"]["fallback_used"] = True
            export_result = fallback
        else:
            report["export"]["fallback_error"] = fallback.error
    else:
        report["export"]["requested_model"] = "rf"

    if export_result.success:
        report["export"].update(
            {
                "exported_model": export_result.model_type,
                "coreml_path": rel_path(export_result.coreml_path, root),
                "coreml_format": export_result.coreml_format,
                "probabilities_available": True,
                "class_output_name": export_result.class_output_name,
                "probability_output_name": export_result.probability_output_name,
                "error": None,
            }
        )
        validation = validate_coreml(Path(export_result.coreml_path))
        report["validation"] = asdict(validation)
        if validation.success:
            report["status"] = "success"
        else:
            report["rf_validation_error"] = validation.error
            if export_result.model_type == "BerkaSpendBucketRF":
                fallback = export_release_candidate("lr")
                if fallback.success:
                    report["export"]["fallback_used"] = True
                    report["export"]["fallback_error"] = None
                    report["export"].update(
                        {
                            "exported_model": fallback.model_type,
                            "coreml_path": rel_path(fallback.coreml_path, root),
                            "coreml_format": fallback.coreml_format,
                            "probabilities_available": True,
                            "class_output_name": fallback.class_output_name,
                            "probability_output_name": fallback.probability_output_name,
                            "error": None,
                        }
                    )
                    validation = validate_coreml(Path(fallback.coreml_path))
                    report["validation"] = asdict(validation)
                    if validation.success:
                        report["status"] = "success"
                else:
                    report["export"]["fallback_error"] = fallback.error

    generate_swift()
    write_report(report)

    if report["status"] != "success":
        print("CoreML preflight failed. See reports/preflight_coreml/coreml_export_report.md")
        return 1

    print("CoreML preflight succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

