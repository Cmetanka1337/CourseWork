# CoreML RF Forensic Report

Decision: **KEEP_RF**

## Inputs
- required_features: 31
- coreml_inputs: 31
- missing_inputs: []
- extra_inputs: []

## Output summary
- class_output_name: classLabel
- probability_output_name: classProbability
- prob_sum_mean: 420.0
- prob_sum_min: 419.9999999999997
- prob_sum_max: 420.0000000000002

## Parity metrics
- sklearn_match_rate: 0.5
- coreml_match_rate: 0.5
- argmax_norm_match_rate: 1.0
- cosine_similarity_mean: 1.0
- rank_order_match_count: 10/10
- notes: sklearn_vs_golden_low_match

## Decision rule
- root_cause: normalized_votes
- rationale: CoreML probabilities appear to be vote counts but normalize cleanly; parity is high.
