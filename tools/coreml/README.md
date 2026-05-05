# CoreML Preflight

Exports the current release candidate to CoreML, validates outputs on the golden inference set, and generates a Swift smoke test.

## Usage

```bash
python3 tools/coreml/run_preflight_coreml.py
```

Individual steps:

```bash
python3 tools/coreml/export_release_candidate_to_coreml.py
python3 tools/coreml/validate_coreml_on_golden.py
python3 tools/coreml/generate_xcode_smoke_test.py
```

## Outputs
- `artifacts/coreml/BerkaSpendBucketRF.mlpackage` (or `.mlmodel` fallback)
- `reports/preflight_coreml/coreml_export_report.json`
- `reports/preflight_coreml/coreml_export_report.md`
- `reports/preflight_coreml/XcodeSmokeTest.swift`

