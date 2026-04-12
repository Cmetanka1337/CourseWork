# Feature Definitions (Tier 1 + Tier 2 + Tier 3)

## Summary
Total features: 21 (6 metadata + 4 Tier 1 + 6 Tier 2 + 5 Tier 3)

## Tier 3: Category Diversity & Historical Patterns

### category_diversity
- **Definition:** Number of unique transaction categories in week_t
- **Range:** [1, 15]
- **Interpretation:** Higher diversity = more varied spending patterns

### dominant_category_ratio
- **Formula:** max(category_count) / total_transactions
- **Range:** [0, 1]
- **Interpretation:** 0.5 = spread; 0.9 = one category dominates

### amount_t_minus_1
- **Definition:** Total amount spent in week t-1
- **How built:** Shifted by 1 week; filled with train user_mean_amount for first week

### amount_t_minus_2
- **Definition:** Total amount spent in week t-2
- **How built:** Shifted by 2 weeks; filled with train user_mean_amount for early weeks

### bucket_t_minus_1
- **Definition:** Bucket classification in week t-1
- **Range:** [0, 1, 2, 3]
- **How built:** Shifted by 1 week; filled with mode=3 for first week

## Quality
- No temporal leakage (lags only from past)
- Train-only statistics used for fallback fills in test
- No NaN values in final datasets
