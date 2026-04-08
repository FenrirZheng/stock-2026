---
name: ma-streak-search
description: Use when user asks to find an MA period that maximizes consecutive days above the moving average, analyze how long a stock stays above its MA, search for the most stable MA period, or evaluate MA streak/duration patterns. Triggers on "連續站上天數", "站上均線幾天", "均線上方持續", "streak above MA", "consecutive days above MA", "MA stability", "最穩定的均線", finding the MA period where price stays above longest, or any analysis focused on duration above a moving average rather than returns or win rate.
---

# MA Streak Search — Consecutive Days Above Moving Average

Find the optimal MA period that maximizes the average number of consecutive days the price stays above the moving average. This is a structural analysis — it measures how "stable" a given MA period is, not trading returns.

## When to Use

- User asks "which MA period keeps the stock above longest?"
- User wants to know how long the stock typically stays above MA(X)
- User asks about MA stability or streak duration
- User defines success as "staying above MA for N+ days" (e.g., 13 days)
- Script path pattern: `TW/{stock_code}/ma_streak_search.py`

## Strategy Definition

This is NOT a trading strategy with buy/sell signals. It's a statistical analysis:

| Concept | Definition |
|---------|-----------|
| **Above MA** | Close > MA(x) (no envelope, pure comparison) |
| **Streak** | Consecutive trading days where Close > MA(x) |
| **Success** | A streak lasting ≥ N days (default N=13) |
| **Failure** | A streak lasting < N days (a "false breakout") |
| **Target metric** | Average streak duration `d` — maximize this |

Key difference from other MA skills: no ±1% envelope, no entry/exit price analysis — purely about duration.

## Execution

```bash
# Default: 13-day success threshold, show top 20
python3 -m TW.{stock_code}.ma_streak_search

# Custom success threshold and top-N
python3 -m TW.{stock_code}.ma_streak_search --min-streak 20 --top-n 30
```

**Must run from project root** (not from TW/ subdirectory).

## Setting Up for a New Stock

If `ma_streak_search.py` doesn't exist for a stock yet, copy from `TW/TWII/ma_streak_search.py` and change these constants:

```python
STOCK_NAME = "加權指數"    # Display name
STOCK_CODE = "TWII"       # Directory name
```

Data must already exist as `TW/{stock_code}/data.csv`. No yfinance download in this script — use existing cached data.

## Algorithm

For each MA period x (5 ~ 250):

1. Compute SMA(x) using numpy cumsum (O(n), fast)
2. Mark each day: `above = (close > ma).astype(int)`
3. Detect streak boundaries via `np.diff(above, prepend=0, append=0)`
   - `+1` transitions = streak start (price crosses above MA)
   - `-1` transitions = streak end (price falls below MA)
4. Calculate streak lengths = end positions - start positions
5. Classify: streak ≥ N days → success, else → failure
6. Aggregate: total segments, success count, success rate, avg/median/min/max duration

Rank all periods by **average streak duration `d`** descending.

## Output Checklist

Produce all of these in order:

1. **Results table**: top N periods ranked by average streak duration
   - Columns: rank, MA period, segment count, successes, failures, success rate, avg days, median days, min, max
2. **Best period streak details**: every streak of the winning MA period
   - Columns: #, start date, end date, days, start price, end price, result (✓/✗)
3. **Current status**: latest close vs best MA — if above, show how many consecutive days so far
4. **Chart**: `TW/{stock_code}/results/streak_chart.png`
   - Line: average streak days vs MA period (green)
   - Bar: success rate per period (blue, right axis)
   - Horizontal reference line at y = N (the success threshold)
   - Star marker on the best period

## Result Interpretation

**What the results tell you:**

- Longer MA periods naturally produce longer streaks (less sensitive to noise) but also fewer total segments
- A high average with low success rate means "either a very long streak or a 1-day fake breakout" — the distribution is bimodal
- A high success rate with moderate average means the MA period is consistently stable — most crossings lead to sustained periods above
- Check both the **average** and the **median** — if they differ significantly, the distribution is skewed by a few very long streaks

**Typical findings for TWII:**
- Top periods cluster around MA 185~225 (long-term moving averages)
- These show avg ~70 days but success rate only ~50% — half the crossings are brief fake breakouts that resolve within a few days
- Successful streaks tend to be very long (100~500 days), failures tend to be very short (1~3 days)

## Common Mistakes

- **Confusing this with a trading strategy** — this measures structural stability, not profitability. Use `single_ma_search.py` for return-based analysis
- **Ignoring median** — average can be misleading when a few 500-day streaks dominate. Median gives the "typical" experience
- **Too low success threshold** — with N=5, almost every MA period looks good. N=13 (roughly 2.5 weeks) is a reasonable default for identifying genuinely sustained trends
- **Running from wrong directory** — must be project root for module path to resolve
