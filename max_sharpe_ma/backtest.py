from dataclasses import dataclass

import pandas as pd

from .data_fetcher import add_sma
from .strategy import compute_positions, compute_strategy_returns


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    n_days_held: int
    n_days_total: int
    exposure_ratio: float


def compute_sharpe(daily_returns: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化 Sharpe Ratio = mean(r) / std(r) × √trading_days。std=0 時回傳 0。"""
    if len(daily_returns) == 0:
        return 0.0
    std_r = daily_returns.std(ddof=1)
    if std_r == 0 or pd.isna(std_r):
        return 0.0
    return float(daily_returns.mean() / std_r * (trading_days ** 0.5))


def run_backtest(df: pd.DataFrame, ma_period: int) -> BacktestResult:
    """完整回測：加 SMA → 計算持倉 → 計算報酬 → 算 Sharpe。"""
    df_sma = add_sma(df, ma_period)
    positions = compute_positions(df_sma["Close"], df_sma["SMA"])
    result = compute_strategy_returns(df_sma["Close"], positions)

    daily_rets = result.daily_returns
    sharpe = compute_sharpe(daily_rets)
    n_days = len(daily_rets)
    total_return = float((1 + daily_rets).prod() - 1) if n_days > 0 else 0.0
    ann_return = float(daily_rets.mean() * TRADING_DAYS_PER_YEAR) if n_days > 0 else 0.0
    ann_vol = float(daily_rets.std(ddof=1) * (TRADING_DAYS_PER_YEAR ** 0.5)) if n_days > 0 else 0.0

    shifted_pos = positions.shift(1).dropna()
    n_held = int((shifted_pos > 0).sum())

    return BacktestResult(
        sharpe_ratio=sharpe,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        total_return=total_return,
        n_days_held=n_held,
        n_days_total=n_days,
        exposure_ratio=n_held / n_days if n_days > 0 else 0.0,
    )
