---
name: ma-distance-search
description: Use when user asks to find optimal MA period and entry distance, search for best {MA period, distance%} parameters, backtest a distance-from-MA strategy, or optimize entry points relative to a moving average. Triggers on "距離均線", "distance from MA", "MA距離策略", finding the best {X, Y} for MA-based entry, or any two-parameter MA search combining period with entry threshold.
---

# MA Distance Strategy — Two-Parameter Search

Find the optimal {X (MA period), Y (distance %)} combination for a "buy near MA, sell below MA" strategy by exhaustive grid search with two-phase refinement.

## Strategy Definition

| Signal | Condition |
|--------|-----------|
| **Buy** | Not holding AND `(close - MA(X)) / MA(X) × 100 ≤ Y` |
| **Sell** | Holding AND `close < MA(X)` (跌破均線) |

**Y parameter interpretation:**
- Y = 5%: buy when price ≤ 5% above MA (loose, frequent)
- Y = 0%: buy when price ≤ MA
- Y = -5%: buy when price ≥ 5% below MA (strict, dip-buying only)

**Trade rules:**
- Close-to-close pricing, no same-day buy+sell
- Unclosed positions not counted
- Win = sell price > buy price

## Two-Phase Grid Search

The coarse search covers the full space cheaply, then the fine search zooms in for precision. This avoids wasting time on the 24,000+ full-resolution grid.

### Phase 1 — Coarse

| Parameter | Range | Step | Count |
|-----------|-------|------|-------|
| MA period (X) | 5 ~ 250 | 5 | 50 |
| Distance (Y) | -5.0% ~ 5.0% | 0.5% | 21 |
| **Total** | | | **1,050** |

### Phase 2 — Fine (around Phase 1 winner)

| Parameter | Range | Step |
|-----------|-------|------|
| MA period | best ± 15 | 1 |
| Distance | best ± 1.5% | 0.1% |

### Filtering

- **Minimum trades ≥ 10** (so win rate is statistically meaningful)
- Adjust threshold by data length: 5yr → ≥5, 10yr → ≥10
- Rank by win rate descending, tiebreak by avg return descending

## Stock Split Adjustment

yfinance often doesn't record Taiwan stock splits. Handle manually:

1. **Detect**: find dates with daily return < -50%
2. **Confirm split ratio with user** (e.g., 1 張 → 22 張)
3. **Adjust**: post-split Close/Open/High/Low × split_ratio, Volume ÷ split_ratio
4. **Verify**: check that prices before and after the split date are continuous

This is critical — without it, the MA breaks at the split boundary and produces false signals across that date.

## SMA Performance Trick

Use cumulative sum for O(n) SMA, not rolling window:

```python
cs = np.cumsum(close)
ma[period-1] = cs[period-1] / period
ma[period:] = (cs[period:] - cs[:n-period]) / period
```

Each backtest then iterates once through the price array with a state machine (holding / not holding). Total runtime for 1,050 combos × ~2,800 days ≈ seconds.

## Output Checklist

Produce all of these in order:

1. **Phase 1 table**: top 20 by win rate, columns: MA, Y, win rate, trades, avg return, cumulative return
2. **Phase 2 table**: top 20 (same columns)
3. **Best combo summary**: one-liner with all metrics
4. **Trade details**: every trade of the best combo — buy date, sell date, buy/sell price, return%, win/loss mark
5. **Next buy point** (from latest data):
   - Current MA(X) value
   - Current price distance from MA (%)
   - Buy trigger price = MA × (1 + Y/100)
   - Remaining gap (%) to trigger
   - If stock had a split: show both adjusted and actual price

## Additional Query: Threshold Search

When user asks "win rate > N% with fewest trades" or similar:

- Run **full fine-grain search**: MA 5~250 step 1, Y -5.0%~5.0% step 0.1% (24,846 combos)
- Filter: win rate > N% AND trades ≥ 5
- Sort by trade count ascending (fewest first)
- Report how many combos meet the threshold (this number itself is insightful — if only 5 out of 24,846 qualify, the threshold is very strict)

## Implementation Notes

- Download data with `yf.download(ticker, period='max')`
- Handle yfinance MultiIndex columns: `df.columns = df.columns.get_level_values(0)`
- Taiwan stocks use `.TW` suffix (e.g., `00631L.TW`, `2330.TW`)
- Write as standalone script (no dependency on project's backtest_engine)
- Print progress every ~20% for long searches so user knows it's working

## Common Pitfalls

- **Not handling splits** → MA breaks at boundary, entire search result is wrong
- **Min trades too low** → 100% win rate from 2 trades is noise, not signal
- **Skipping Phase 2** → missing the actual optimum between coarse grid points
- **Negative Y edge case** → buying below MA means next-day sell is likely if no bounce; this isn't a bug, it's how the strategy works — the search will naturally rank these lower if they don't perform
- **Overfitting concern** → with 24,846 parameter combos, some will look good by chance; having ≥10 trades and checking if top results cluster (same MA region) increases confidence
