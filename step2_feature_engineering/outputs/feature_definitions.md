# Feature Definitions (Tier 1 + Tier 2)

## Tier 1: Critical Features

### z_score
- **Formula:** (amount_t - user_mean_amount) / user_std_amount
- **Range:** [-10, 10]
- **Interpretation:** How many standard deviations current week is from user average
- **Why it matters:** Normalizes spending across users with different spending habits

### entropy
- **Formula:** -sum(p_i * log2(p_i)) where p_i = P(bucket_i) in last 4 weeks
- **Range:** [0, 2]
- **Interpretation:** 0 = predictable (always same bucket), 2 = unpredictable (uniform)
- **Why it matters:** Indicates user reliability; high entropy means model should be cautious

### txn_count
- **Definition:** Number of individual transactions in week_t
- **Range:** [1, inf)
- **Why it matters:** Distinguishes between few large purchases vs. many small purchases

### relative_txn_count
- **Formula:** txn_count / user_avg_txn_count
- **Range:** [0, 10] (clipped)
- **Interpretation:** 1 = average activity, >1 = high activity, <1 = low activity
- **Why it matters:** Normalized activity indicator

## Tier 2: Dynamic + RFM Features

### delta_amount
- **Formula:** amount_t - amount_t-1
- **Range:** [-1000, 1000] (clipped)
- **Interpretation:** Positive = increasing spending, negative = decreasing
- **Why it matters:** Captures momentum in spending changes

### delta_bucket
- **Formula:** bucket_t - bucket_t-1
- **Range:** [-3, 3]
- **Interpretation:** -3 = big drop, 0 = stable, +3 = big increase
- **Why it matters:** Discrete indicator of spending transitions

### rolling_mean_8w
- **Formula:** mean(amount from prior 8 weeks)
- **Why it matters:** Medium-term baseline for user spending

### rolling_std_8w
- **Formula:** std(amount from prior 8 weeks)
- **Why it matters:** Medium-term volatility; indicates spending consistency

### recency_days
- **Formula:** days since last transaction before current week (clipped to [0, 180])
- **Why it matters:** RFM metric; recent activity correlates with future activity

### user_cv
- **Formula:** user_std_amount / user_mean_amount
- **Range:** [0, 10] (clipped)
- **Interpretation:** Relative volatility; high CV = unpredictable user
- **Why it matters:** Indicates user spending consistency independent of amount
