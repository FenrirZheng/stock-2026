# MA Period Bayesian Optimization Search — Design Spec

## Overview

A Python package (`ma_search/`) that uses Bayesian Optimization to find the optimal simple moving average (SMA) period `n` for a MA crossover strategy. Compares BO efficiency against brute-force search with train/test validation.

**Target:** 00635U.TW (元大S&P黃金)
**Run:** `python -m ma_search.main`

---

## Strategy

**MA Crossover (position-based, long/cash):**

```
position[t] = 1.0  if close[t] > SMA(n)[t]
              0.0  otherwise (including when SMA is NaN)

strategy_return[t] = position[t-1] × stock_return[t]
stock_return[t]    = (close[t] - close[t-1]) / close[t-1]
```

- `position[t-1]` shift avoids look-ahead bias
- Cash days contribute return = 0 to Sharpe calculation
- Fully vectorized (no row-by-row iteration)

---

## Objective Function

**Annualized Sharpe Ratio (risk-free rate = 0):**

```
Sharpe = mean(daily_returns) / std(daily_returns) × √252
```

- Includes cash-day zeros (penalizes low exposure)
- Returns 0.0 when std == 0

---

## Search Configuration

| Parameter | Value |
|---|---|
| Search space | n ∈ [2, 59] (58 integer values) |
| Optuna sampler | TPESampler(seed=42, n_startup_trials=5) |
| Total BO trials | 20 |
| Brute-force | All 58 values |
| Baseline | n = 20 |

---

## Data

| Item | Value |
|---|---|
| Ticker | 00635U.TW |
| Source | yfinance |
| Period | 2016-04-01 to 2026-04-01 (~10 years) |
| Train split | First 70% of rows |
| Test split | Last 30% of rows |

---

## Module Design

### `config.py`

Constants only:

```python
TICKER = "00635U.TW"
START_DATE = "2016-04-01"
END_DATE = "2026-04-01"
TRAIN_RATIO = 0.7
MA_MIN = 2
MA_MAX = 59
N_STARTUP_TRIALS = 5
N_TOTAL_TRIALS = 20
BASELINE_MA = 20
TRADING_DAYS_PER_YEAR = 252
```

### `data_fetcher.py`

Functions:
- `fetch_stock_data(ticker, start, end) -> pd.DataFrame` — download via yfinance, handle MultiIndex columns, strip timezone, validate columns
- `add_sma(df, period) -> pd.DataFrame` — add `SMA` column via `rolling(period).mean()`
- `split_train_test(df, ratio) -> tuple[pd.DataFrame, pd.DataFrame]` — integer index split

### `strategy.py`

Dataclass:

```python
@dataclass
class StrategyResult:
    daily_returns: pd.Series
    positions: pd.Series
    stock_returns: pd.Series
```

Functions:
- `compute_positions(close, sma) -> pd.Series` — 1.0 if close > sma, 0.0 otherwise (NaN sma → 0.0)
- `compute_strategy_returns(close, positions) -> StrategyResult` — vectorized returns with position shift

### `backtest.py`

Dataclass:

```python
@dataclass
class BacktestResult:
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    n_days_held: int
    n_days_total: int
    exposure_ratio: float
```

Functions:
- `compute_sharpe(daily_returns, trading_days=252) -> float`
- `run_backtest(df, ma_period) -> BacktestResult` — full pipeline: add SMA → positions → returns → Sharpe

### `optimizer.py`

Dataclass:

```python
@dataclass
class SearchResult:
    best_n: int
    best_train_sharpe: float
    best_test_sharpe: float
    all_results: dict[int, float]           # n → train Sharpe (brute-force only)
    trial_history: list[tuple[int, float]]  # (eval_count, best_sharpe_so_far)
```

Functions:
- `create_objective(train_df) -> Callable` — closure for Optuna
- `run_bayesian_search(train_df, test_df, ...) -> SearchResult`
- `run_brute_force_search(train_df, test_df) -> SearchResult`

### `reporter.py`

Function:
- `print_report(bo_result, bf_result, baseline_train, baseline_test, config_info)` — formatted console output

Output sections:
1. Data Summary (periods, split sizes)
2. Bayesian Optimization result (best n, train/test Sharpe)
3. Brute-Force result (best n, train/test Sharpe)
4. Baseline MA(20) result
5. Convergence comparison table (Sharpe at eval 5, 10, 15, 20)
6. Overfitting check (train→test gap)

### `main.py`

```python
def main():
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    train_df, test_df = split_train_test(df, TRAIN_RATIO)
    baseline_train = run_backtest(train_df, BASELINE_MA)
    baseline_test = run_backtest(test_df, BASELINE_MA)
    bo_result = run_bayesian_search(train_df, test_df)
    bf_result = run_brute_force_search(train_df, test_df)
    print_report(bo_result, bf_result, baseline_train, baseline_test, ...)
```

---

## Tests

### `test_strategy.py`

- Position above/below/equal SMA
- Position when SMA is NaN → 0
- Return shift correctness (no look-ahead)
- Cash days return 0

### `test_backtest.py`

- Sharpe with constant returns → 0.0
- Sharpe with known mean/std → verify formula
- Always-holding scenario: Sharpe matches buy-and-hold
- Always-cash scenario: Sharpe = 0
- Full pipeline integration with synthetic data

---

## Verification

1. `pytest ma_search/tests/ -v` — all tests pass
2. `python -m ma_search.main` — runs full pipeline, prints report
3. Sanity check: BO best_n should be close to brute-force best_n
4. Out-of-sample Sharpe should be lower than in-sample (expected)
