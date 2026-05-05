# CoreML Preflight Report

Status: **failure**

## Export
- requested_model: rf
- exported_model: BerkaSpendBucketRF
- fallback_used: False
- fallback_error: CoreML conversion failed: 'LogisticRegression' object has no attribute 'multi_class'
- coreml_path: artifacts/coreml/BerkaSpendBucketRF.mlpackage
- coreml_format: mlpackage
- probabilities_available: True
- class_output_name: classLabel
- probability_output_name: classProbability

RF validation error:
- Probabilities failed normalization check; sum is not ~1.0.

## Golden validation
- match_rate: 0.5
- matches: 5/10
- prob_sum_min: 419.9999999999997
- prob_sum_max: 420.0000000000002
- prob_sum_mean: 420.0
- invalid_prob_rows: 10

Example output:
- expected_class: 0
- predicted_class: 1
- probabilities: {'0': 20.088006086192358, '3': 89.46854609964953, '2': 152.62424668743844, '1': 157.8192011267194}

## IMPORTANT: classProbability are votes, must normalize
- CoreML `classProbability` sums to `n_estimators` (≈420) for RF.
- Normalize before using confidence: `p_i = votes_i / sum(votes)`.
- Map dict keys to classes `[0,1,2,3]` before ordering p0..p3.

## Xcode next step
- Drag the .mlpackage into Xcode and run reports/preflight_coreml/XcodeSmokeTest.swift
