# iOS Feature Parity Guide

This document maps Python feature preparation to Swift logic.

## Core requirements
- Keep feature order exactly as in `scaler_export_guide.json`.
- Reproduce formulas from Step 2 without redefinition.
- Apply scaler using train statistics only.

## Feature scaling formula
```swift
let scaled = zip(features, scaler.mean).enumerated().map { (i, pair) in
  let (feature, mean) = pair
  return (feature - mean) / scaler.scale[i]
}
```

## Notes
- RF is exported as static CoreML model.
- LR and SGD are candidates for on-device updates via `MLUpdateTask`.
- Ensure bucket labels remain `[0, 1, 2, 3]`.
