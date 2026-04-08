---
name: single-ma-winrate-search
description: Use when user asks to find optimal moving average period by win rate, compare MA periods, exhaustive MA search, or identify best single MA for a stock using single_ma_search.py
---

# Single MA Win Rate Search

## Overview

Exhaustively search all MA periods (3~240) for a stock using single MA + 1% envelope strategy, evaluate by **win rate** (not cumulative return), and find the optimal period.

## When to Use

- User asks "which MA period is best for stock X?"
- User wants to find optimal single moving average by win rate
- User asks to exhaustively test MA periods
- Script path pattern: `TW/{stock_code}/single_ma_search.py`

## Strategy Rules

| Signal | Condition |
|--------|-----------|
| Buy | Close > MA(period) x 1.01 |
| Sell | Close < MA(period) x 0.99 |

- Full in/out per trade, no position sizing
- No fees or slippage

## Execution

```bash
# Default: min 5 trades filter
python3 -m TW.{stock_code}.single_ma_search

# Custom minimum trades filter
python3 -m TW.{stock_code}.single_ma_search --min-trades 10
```

**Must run from project root** (not from TW/ subdirectory).

## Setting Up for a New Stock

If `single_ma_search.py` doesn't exist for a stock yet, copy from an existing one (e.g., `TW/TWII/single_ma_search.py`) and change these constants:

```python
SYMBOL = "2303.TW"    # yfinance symbol
STOCK_NAME = "聯電"    # Display name
STOCK_CODE = "2303"    # Directory name
```

The `load_data()` function downloads via yfinance and caches to `data.csv`. Delete `data.csv` to re-download with different date range.

## Result Interpretation

**Selection criteria (priority order):**
1. Highest win rate
2. Trade count — more trades = more statistically reliable (use `--min-trades` to filter)
3. Cumulative return as tiebreaker

**Rule of thumb for `--min-trades`:**
- 5 year data: `--min-trades 5`
- 10 year data: `--min-trades 10`
- Fewer trades = higher variance, less trustworthy

## Output

1. **Terminal**: Top 10 periods ranked by win rate, with trade count and cumulative return
2. **Chart**: `TW/{stock_code}/results/winrate_chart.png` — bar chart (win rate) + line (trade count)
3. **Current signal**: Buy/sell/hold based on best period's MA position

## Engine

The shared backtest function is `run_single_ma_backtest()` in `TW/backtest_engine.py`. It returns a list of trade dicts with `buy_date`, `sell_date`, `buy_price`, `sell_price`, `return`, `win` fields.

## Common Mistakes

- Not using `--min-trades` — periods with 2-3 trades can show misleadingly high win rates
- Forgetting to delete `data.csv` when changing date range — cached data won't update
- Running from wrong directory — must be project root for module imports to work
- Comparing across different time periods — always use same data range for fair comparison
