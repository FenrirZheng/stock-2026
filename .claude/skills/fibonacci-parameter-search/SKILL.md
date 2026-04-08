---
name: fibonacci-parameter-search
description: Use when user asks to find optimal Fibonacci retracement level, compare fib parameters, or identify entry points using fibonacci_backtest.py
---

# Fibonacci Parameter Search

## Overview

Systematically search all standard Fibonacci retracement levels for a stock's backtest script, compare results, and identify current entry points.

## When to Use

- User asks "which fib level is best for stock X?"
- User wants to compare Fibonacci retracement parameters
- User asks for current entry point using Fibonacci strategy
- Script path pattern: `TW/{stock_code}/fibonacci_backtest.py`

## Standard Fibonacci Levels

Always test all 5 levels — never skip any:

| Level | Value | Character |
|-------|-------|-----------|
| 23.6% | `0.236` | Shallow pullback |
| 38.2% | `0.382` | Moderate pullback |
| 50.0% | `0.500` | Half retracement |
| 61.8% | `0.618` | Golden ratio (script default) |
| 78.6% | `0.786` | Deep pullback |

## Execution Pattern

**MUST run all 5 in parallel** (no dependencies between them):

```bash
python3 TW/{code}/fibonacci_backtest.py --ma {MA} --fib 0.236
python3 TW/{code}/fibonacci_backtest.py --ma {MA} --fib 0.382
python3 TW/{code}/fibonacci_backtest.py --ma {MA} --fib 0.5
python3 TW/{code}/fibonacci_backtest.py --ma {MA} --fib 0.618
python3 TW/{code}/fibonacci_backtest.py --ma {MA} --fib 0.786
```

If `--ma` is not specified, script searches MA 10~200 (slower).

## Result Comparison

Collect from each run and present as table:

| Fib Level | Best swing | Max Return | Trade Count |
|-----------|-----------|------------|-------------|

**Selection criteria (priority order):**
1. Highest cumulative return
2. Trade count — more trades = more statistically reliable
3. Sweet spot analysis: very shallow (0.236) triggers often but small profit; very deep (0.786) rarely triggers

## Entry Point Analysis

From the best parameter's signal output, check 3 conditions:

1. **Uptrend**: Close > MA → required
2. **Swing order**: Swing Low before Swing High → ascending wave pullback structure
3. **Price at Fib level**: Low <= Fib price AND Close >= Fib price

All 3 must be met for entry. Report which conditions are met/unmet and the target entry price.

## Common Mistakes

- Running fib levels sequentially instead of in parallel — wastes time
- Only testing 0.618 (the default) — always test all 5
- Ignoring trade count when comparing — a high return from 1-2 trades is unreliable
- Not checking all 3 entry conditions — all must be satisfied simultaneously
