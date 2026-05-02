# Preflight Tools

These scripts prepare the release candidate artifacts for iOS handoff and generate a Swift calibrator reference.

## Usage

```bash
python3 tools/preflight/print_release_candidate_summary.py
python3 tools/preflight/export_bundle_for_ios.py
python3 tools/preflight/generate_swift_calibrator_reference.py
```

Or run all steps:

```bash
python3 tools/preflight/run_all_preflight.py
```

## Outputs
- `artifacts/ios_bundle/feature_contract.json`
- `artifacts/ios_bundle/SoftmaxCalibratorReference.swift`
- `artifacts/ios_bundle/golden_inference_set_*.json`
- `artifacts/ios_bundle/golden_inference_set_*.csv`
- `artifacts/ios_bundle/release_manifest.json`

