#!/usr/bin/env python3
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib


LABELS = [0, 1, 2, 3]


@dataclass
class ExportResult:
    success: bool
    model_type: str
    coreml_path: Optional[str]
    coreml_format: Optional[str]
    class_output_name: Optional[str]
    probability_output_name: Optional[str]
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


def feature_inputs(feature_order: list[str]) -> list[tuple[str, Any]]:
    ct, _ = get_coremltools()
    if ct is None:
        return []
    return [(name, ct.models.datatypes.Double()) for name in feature_order]


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


def save_model(mlmodel, output_dir: Path, base_name: str) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / f"{base_name}.mlpackage"
    try:
        mlmodel.save(str(mlpackage_path))
        return str(mlpackage_path), "mlpackage"
    except Exception:
        mlmodel_path = output_dir / f"{base_name}.mlmodel"
        mlmodel.save(str(mlmodel_path))
        return str(mlmodel_path), "mlmodel"


def find_lr_model_path(root: Path, selected_prefix: str) -> Optional[Path]:
    candidates = [
        root / f"models/release_candidate/{selected_prefix}_lr_model.pkl",
        root / f"step3_model_training_berka/outputs/{selected_prefix}_lr_model.pkl",
        root / "step3_model_training_berka/outputs/lr_model.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - handled in runtime
        return None, f"coremltools import failed: {exc}"
    return ct, None


def sklearn_version_supported() -> tuple[bool, str | None]:
    try:
        import sklearn
    except ImportError as exc:
        return False, f"scikit-learn import failed: {exc}"
    version = sklearn.__version__
    try:
        from packaging.version import Version
    except ImportError:
        return False, "packaging is required to check scikit-learn version."
    if Version(version) > Version("1.5.1"):
        return False, (
            "scikit-learn version is not supported by coremltools. "
            f"Detected {version}; install scikit-learn<=1.5.1."
        )
    return True, None


def convert_sklearn_model(ct, model, inputs, model_name: str):
    conversion_attempts = [
        {
            "input_features": inputs,
            "class_labels": LABELS,
            "model_name": model_name,
        },
        {
            "input_features": inputs,
            "class_labels": LABELS,
        },
        {
            "input_features": inputs,
        },
    ]
    last_error = None
    for kwargs in conversion_attempts:
        try:
            return ct.converters.sklearn.convert(model, **kwargs)
        except TypeError as exc:
            last_error = exc
        except Exception:
            raise
    raise TypeError(last_error) if last_error else TypeError("Conversion failed.")


def export_model(model_path: Path, feature_order: list[str], model_name: str) -> ExportResult:
    ct, ct_error = get_coremltools()
    if ct is None:
        return ExportResult(
            success=False,
            model_type=model_name,
            coreml_path=None,
            coreml_format=None,
            class_output_name=None,
            probability_output_name=None,
            error=ct_error or "coremltools is not installed.",
        )

    sklearn_ok, sklearn_error = sklearn_version_supported()
    if not sklearn_ok:
        return ExportResult(
            success=False,
            model_type=model_name,
            coreml_path=None,
            coreml_format=None,
            class_output_name=None,
            probability_output_name=None,
            error=sklearn_error,
        )

    if not model_path.exists():
        return ExportResult(
            success=False,
            model_type=model_name,
            coreml_path=None,
            coreml_format=None,
            class_output_name=None,
            probability_output_name=None,
            error=f"Missing model artifact: {model_path}",
        )

    model = joblib.load(model_path)
    inputs = feature_inputs(feature_order)
    try:
        mlmodel = convert_sklearn_model(ct, model, inputs, model_name)
    except Exception as exc:
        return ExportResult(
            success=False,
            model_type=model_name,
            coreml_path=None,
            coreml_format=None,
            class_output_name=None,
            probability_output_name=None,
            error=f"CoreML conversion failed: {exc}",
        )

    try:
        mlmodel.short_description = f"{model_name} (exported via coremltools)"
    except Exception:
        pass

    spec = mlmodel.get_spec()
    class_output, prob_output = detect_output_names(spec)
    if prob_output is None:
        return ExportResult(
            success=False,
            model_type=model_name,
            coreml_path=None,
            coreml_format=None,
            class_output_name=class_output,
            probability_output_name=None,
            error="CoreML export did not expose a probability output.",
        )

    output_dir = repo_root() / "artifacts/coreml"
    coreml_path, coreml_format = save_model(mlmodel, output_dir, model_name)
    return ExportResult(
        success=True,
        model_type=model_name,
        coreml_path=coreml_path,
        coreml_format=coreml_format,
        class_output_name=class_output,
        probability_output_name=prob_output,
        error=None,
    )


def export_release_candidate(model_type: str = "rf") -> ExportResult:
    root = repo_root()
    manifest = load_json(root / "models/release_candidate/release_manifest.json")
    feature_contract = load_json(root / "artifacts/ios_bundle/feature_contract.json")
    feature_order = feature_contract["feature_order"]

    if model_type == "rf":
        model_path = resolve_path(manifest["selected_model_path"], root)
        return export_model(model_path, feature_order, "BerkaSpendBucketRF")

    selected_prefix = manifest.get("selected_prefix", "full_spend_tuned")
    lr_path = find_lr_model_path(root, selected_prefix)
    if lr_path is None:
        return ExportResult(
            success=False,
            model_type="BerkaSpendBucketLR",
            coreml_path=None,
            coreml_format=None,
            class_output_name=None,
            probability_output_name=None,
            error=(
                "No LogisticRegression artifact found. Expected one of: "
                f"{selected_prefix}_lr_model.pkl in release_candidate or step3 outputs."
            ),
        )
    return export_model(lr_path, feature_order, "BerkaSpendBucketLR")


def main() -> int:
    result = export_release_candidate("rf")
    if result.success:
        print(f"CoreML export complete: {result.coreml_path}")
        return 0

    print(f"Export failed: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

